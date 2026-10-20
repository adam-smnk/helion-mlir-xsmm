"""Loop-carried elementwise updates write into their iter arg (``mlir/in_place.py``)."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.execution import inline_module


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16))
def row_sum_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16, 16))
def online_softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            values = x[tm, tn]
            mi_next = torch.maximum(mi, torch.amax(values, dim=1))
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tn in hl.tile(n):
            out[tm, tn] = torch.exp(x[tm, tn] - mi[:, None]) / di[:, None]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16))
def scaled_running_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = torch.maximum(acc * 0.5, x[tm, tn].amax(dim=-1))
        out[tm] = acc
    return out


def _scaled_running_max_reference(x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(x.size(0))
    for start in range(0, x.size(1), 16):
        acc = torch.maximum(acc * 0.5, x[:, start : start + 16].amax(dim=-1))
    return acc


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16, 16))
def decayed_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc * 0.5
            acc = acc + torch.einsum("mk,kn->mn", x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16, 16))
def conditionally_decayed_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            if a.sum(-1).sum() > 0:
                acc = acc * 0.5
            acc = acc + torch.einsum("mk,kn->mn", a, y[tk, tn])
        out[tm, tn] = acc
    return out


def _decayed_matmul_reference(
    x: torch.Tensor, y: torch.Tensor, conditional: bool = False
) -> torch.Tensor:
    acc = torch.zeros(x.size(0), y.size(1))
    for start in range(0, x.size(1), 16):
        rows = []
        for row in range(0, x.size(0), 8):
            a = x[row : row + 8, start : start + 16]
            decay = 0.5 if not conditional or a.sum() > 0 else 1.0
            rows.append(acc[row : row + 8] * decay + a @ y[start : start + 16])
        acc = torch.cat(rows)
    return acc


def _iter_arg_destinations(kernel: helion.Kernel, *args: torch.Tensor) -> list[str]:
    """The parent op of every ``linalg.generic`` writing into an ``scf.for`` iter
    arg, after the in-place updates."""
    module = inline_module(generate_mlir(kernel, list(args)))
    found = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "linalg.generic":
            for output in op.opview.outputs:
                if (
                    isinstance(output, ir.BlockArgument)
                    and output.owner.owner.name == "scf.for"
                ):
                    found.append(op.parent.name)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def _writes_into_iter_arg(kernel: helion.Kernel, x: torch.Tensor) -> list[bool]:
    """For each loop-carried value updated by a ``linalg.generic``, whether the
    update's destination is the loop's iter arg."""
    module = inline_module(generate_mlir(kernel, [x]))
    found = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            body = op.regions[0].blocks[0]
            yielded = list(body.operations)[-1].operands
            for arg, value in zip(list(body.arguments)[1:], yielded, strict=True):
                producer = (
                    value.owner.operation if isinstance(value, ir.OpResult) else None
                )
                if producer is not None and producer.name == "linalg.generic":
                    found.append(producer.opview.outputs[value.result_number] == arg)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def test_accumulation_writes_into_its_iter_arg() -> None:
    assert _writes_into_iter_arg(row_sum_loop, torch.randn(16, 64)) == [True]


def test_accumulated_reduction_starts_from_its_iter_arg() -> None:
    # ``acc + x.sum(-1)``: the sum reduces into ``acc``, no separate add is left.
    module = inline_module(generate_mlir(row_sum_loop, [torch.randn(16, 64)]))
    loops = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            loops.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    (loop,) = loops
    body = loop.regions[0].blocks[0]
    (value,) = list(body.operations)[-1].operands
    producer = value.owner.operation
    assert "reduction" in str(producer.attributes["iterator_types"])
    assert producer.opview.outputs[0] == list(body.arguments)[1]
    generics = [op for op in body.operations if op.name == "linalg.generic"]
    assert len(generics) == 1


def test_update_reading_the_old_value_later_is_still_in_place() -> None:
    # ``mi`` is read after ``mi_next`` is computed: bufferization copies it.
    # ``di``'s update is ``add(mul(di, ...), ...)``: the add does not read ``di``.
    found = _writes_into_iter_arg(online_softmax, torch.randn(16, 64))
    assert sorted(found) == [False, True]


def test_update_not_reading_the_iter_arg_keeps_its_destination() -> None:
    assert _writes_into_iter_arg(scaled_running_max, torch.randn(16, 64)) == [False]


def test_update_accumulated_into_by_a_contraction_is_in_place() -> None:
    # ``acc * 0.5`` is the contraction's destination, not the yielded value.
    args = (torch.randn(16, 64), torch.randn(64, 32))
    assert _iter_arg_destinations(decayed_matmul, *args) == ["scf.for"]


def test_conditional_update_is_in_place() -> None:
    args = (torch.randn(16, 64), torch.randn(64, 32))
    assert _iter_arg_destinations(conditionally_decayed_matmul, *args) == ["scf.if"]


@pytest.mark.parametrize(
    ("kernel", "reference"),
    [
        (row_sum_loop, lambda x: x.sum(-1)),
        (online_softmax, lambda x: x.softmax(-1)),
        (scaled_running_max, _scaled_running_max_reference),
    ],
    ids=["row_sum_loop", "online_softmax", "scaled_running_max"],
)
def test_in_place_updates_execute(kernel, reference) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        [torch.randn(16, 60)],
        paths=("direct", "generated"),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("kernel", "conditional"),
    [(decayed_matmul, False), (conditionally_decayed_matmul, True)],
    ids=["decayed_matmul", "conditionally_decayed_matmul"],
)
def test_accumulated_in_place_updates_execute(
    kernel: helion.Kernel, conditional: bool
) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        lambda x, y: _decayed_matmul_reference(x, y, conditional),
        [torch.randn(16, 64), torch.randn(64, 32)],
        atol=1e-4,
        rtol=1e-4,
    )
