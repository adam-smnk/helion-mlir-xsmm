"""Golden IR signatures for performance-critical kernel shapes.

A signature is the op tree of every function with attributes and result types, minus SSA
names, locations and generated helper symbol names. It is taken after the same
``inline,canonicalize,cse`` cleanup the executor runs, i.e. on the IR lighthouse receives.
Any change shows up as a diff here; intentional changes are re-recorded with
``pytest --update-golden``.
"""

from __future__ import annotations

import difflib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from mlir import ir
from mlir.passmanager import PassManager
import pytest
import torch

from tests.harness import blocked_matmul_cases

from helion_mlir_backend import generate_mlir

if TYPE_CHECKING:
    from collections.abc import Callable

_GOLDEN_DIR = Path(__file__).parent / "golden"
_cpu_matmul = importlib.import_module("helion_mlir_utils.matmul")


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[16, 16])
)
def golden_elementwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.relu(x[tm, tn] * 2.0 + y[tm, tn])
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32, 32])
)
def golden_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc.to(out.dtype)
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def golden_einsum_two_reductions(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _, _ = x.size()
    _, _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = acc + torch.einsum("mkl,kln->mn", x[tm, :, :], y[:, :, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[1, 1, 16])
)
def golden_nested_grid(x: torch.Tensor) -> torch.Tensor:
    a, b, k = x.size()
    out = torch.empty_like(x)
    for i in hl.grid(a):
        for j in hl.grid(b):
            for tk in hl.tile(k):
                out[i, j, tk] = x[i, j, tk] * 2.0
    return out


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16)


def _blocked(name: str) -> tuple[Callable[..., object], Callable[[], list[object]]]:
    return (
        blocked_matmul_cases()[name][0],
        lambda: blocked_matmul_cases()[name][1],
    )


_CASES: dict[str, tuple[Callable[..., object], Callable[[], list[object]]]] = {
    "elementwise": (
        golden_elementwise,
        lambda: [torch.randn(32, 32), torch.randn(32, 32)],
    ),
    "addmm_f32": (golden_addmm, lambda: [torch.randn(64, 64), torch.randn(64, 64)]),
    "addmm_bf16": (golden_addmm, lambda: [_bf16(64, 64), _bf16(64, 64)]),
    "einsum_two_reductions": (
        golden_einsum_two_reductions,
        lambda: [torch.randn(64, 4, 2), torch.randn(4, 2, 64)],
    ),
    "nested_grid": (golden_nested_grid, lambda: [torch.randn(2, 3, 32)]),
    "blocked_matmul": _blocked("blocked_matmul"),
    "blocked_matmul_bias": _blocked("blocked_matmul_bias"),
    "blocked_matmul_affine": _blocked("blocked_matmul_affine"),
    "pack_a": (
        _cpu_matmul._pack_a_kernel,
        lambda: [_bf16(64, 64), hl.constexpr(64), hl.constexpr(64)],
    ),
    "pack_b": (
        _cpu_matmul._pack_b_kernel,
        lambda: [_bf16(64, 64), hl.constexpr(64), hl.constexpr(64)],
    ),
    "pack_b_vnni": (
        _cpu_matmul._pack_b_vnni_kernel,
        lambda: [_bf16(64, 64), hl.constexpr(64), hl.constexpr(64)],
    ),
}


def ir_signature(module: object) -> str:
    """Render ``module`` as a stable op tree (see module docstring)."""
    with module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(inline,canonicalize,cse)").run(
            module.operation
        )
    symbols: dict[str, str] = {}
    lines: list[str] = []

    def symbol(name: str) -> str:
        return symbols.setdefault(name, f"@sym{len(symbols)}")

    def visit(op: object, depth: int) -> None:
        operation = op.operation
        attributes = []
        for name in operation.attributes:
            value = str(operation.attributes[name])
            if name in ("sym_name", "callee"):
                value = symbol(value.strip('"@'))
            attributes.append(f"{name}={value}")
        line = "  " * depth + operation.name
        if attributes:
            line += " {" + ", ".join(sorted(attributes)) + "}"
        if operation.results:
            line += " -> " + ", ".join(str(result.type) for result in operation.results)
        lines.append(line)
        for region in operation.regions:
            for block in region.blocks:
                if block.arguments:
                    types = ", ".join(str(arg.type) for arg in block.arguments)
                    lines.append("  " * (depth + 1) + f"^({types})")
                for child in block.operations:
                    visit(child, depth + 1)

    for op in module.body.operations:
        visit(op, 0)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_golden_ir(case: str, request: pytest.FixtureRequest) -> None:
    kernel, make_args = _CASES[case]
    signature = ir_signature(generate_mlir(kernel, make_args()))
    golden = _GOLDEN_DIR / f"{case}.txt"
    if request.config.getoption("--update-golden"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(signature)
        return
    assert golden.exists(), f"missing {golden}; run pytest --update-golden"
    expected = golden.read_text()
    if signature != expected:
        diff = difflib.unified_diff(
            expected.splitlines(),
            signature.splitlines(),
            "golden",
            "actual",
            lineterm="",
        )
        pytest.fail(
            "IR signature changed:\n" + "\n".join(list(diff)[:80]), pytrace=False
        )
