"""``xevm_bf16_extensions``: in XeVM GPU modules, the bf16-to-f32 ``fpext``s
that an f32-to-bf16 ``fptrunc`` is computed from become integer shifts (an IGC
rounding workaround); everything else is left alone. IR only: no XPU or SPIR-V
target needed. The device regressions are ``tests/test_xpu_runtime.py``'s
``test_bf16_arithmetic_rounds_to_nearest_even``."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import PROBLEMS

from helion_mlir_backend._compiler.mlir_transforms.xevm_bf16_extensions import (
    bf16_extensions_as_shifts,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# --- Kernels of the device regressions (tests/test_xpu_runtime.py) ------------
# One rounding of f32 math on bf16 values: bf16 arithmetic (computed in f32
# between two conversions) and its explicit f32 form; IGC shrank both to bf16
# arithmetic that truncates.


def _bf16_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + y[tile]
    return out


def _bf16_sub(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] - y[tile]
    return out


def _bf16_mul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * y[tile]
    return out


def _bf16_div(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] / y[tile]
    return out


def _f32_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile].to(torch.float32) + y[tile].to(torch.float32)).to(
            out.dtype
        )
    return out


def _f32_sub(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile].to(torch.float32) - y[tile].to(torch.float32)).to(
            out.dtype
        )
    return out


def _f32_mul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile].to(torch.float32) * y[tile].to(torch.float32)).to(
            out.dtype
        )
    return out


def _f32_div(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile].to(torch.float32) / y[tile].to(torch.float32)).to(
            out.dtype
        )
    return out


def _scalar_mul(a: torch.Tensor, s: hl.constexpr, out: torch.Tensor) -> torch.Tensor:
    """KernelBench level 1 ``5_Matrix_scalar_multiplication`` (``A * s``) on a
    bf16 ``A``. A scalar exact in bf16 (e.g. 3.140625, 3.14 rounded) lets IGC
    shrink the product too."""
    for tile in hl.tile(a.shape):
        out[tile] = a[tile] * s
    return out


BF16_OPS: dict[str, tuple[Callable[..., torch.Tensor], ...]] = {
    "add": (_bf16_add, _f32_add, operator.add),
    "sub": (_bf16_sub, _f32_sub, operator.sub),
    "mul": (_bf16_mul, _f32_mul, operator.mul),
}
"""Per op: the bf16 kernel, the f32 kernel and the op (torch's reference).
Division was right before the workaround (not shrunk) and stays so (one test of
its own: f32 division of special values differs from torch's)."""


def bf16_kernel(
    fn: Callable[..., torch.Tensor], block_sizes: list[int]
) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes),
    )(fn)


bf16_add = bf16_kernel(_bf16_add, [32, 256])
scalar_mul = bf16_kernel(_scalar_mul, [32, 256])


# --- The rewrite ---------------------------------------------------------------


def _module(body: str, args: str, target: str = '[#xevm.target<chip = "bmg">]') -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_kernelcc @k({args}, %p: !llvm.ptr<1>) {{
      {body}
      llvm.return
    }}
  }}
  func.func @host(%h: bf16, %q: !llvm.ptr) {{
    %e = llvm.fpext %h : bf16 to f32
    %s = llvm.fadd %e, %e : f32
    %t = llvm.fptrunc %s : f32 to bf16
    llvm.store %t, %q : bf16, !llvm.ptr
    return
  }}
}}"""


def _rewrite(text: str, *, apply: bool = True) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = bf16_extensions_as_shifts(module) if apply else 0
        assert module.operation.verify()
        return count, str(module)


def _kernel(text: str) -> str:
    return text[text.index("gpu.module") : text.index("func.func @host")]


_ADD = """%ea = llvm.fpext %a fastmath<contract> : vector<1xbf16> to vector<1xf32>
      %eb = llvm.fpext %b : vector<1xbf16> to vector<1xf32>
      %s = llvm.fadd %ea, %eb : vector<1xf32>
      %t = llvm.fptrunc %s fastmath<contract> : vector<1xf32> to vector<1xbf16>
      llvm.store %t, %p : vector<1xbf16>, !llvm.ptr<1>"""


def test_extensions_of_a_rounded_sum_become_shifts() -> None:
    """``bf16(f32(a) + f32(b))``: both extensions are integer shifts, the f32
    sum and its conversion to bf16 stay."""
    count, text = _rewrite(_module(_ADD, "%a: vector<1xbf16>, %b: vector<1xbf16>"))
    assert count == 2
    kernel = _kernel(text)
    assert "llvm.fpext" not in kernel
    assert kernel.count("llvm.bitcast %arg") == 2  # bf16 bits as i16
    assert kernel.count(": vector<1xbf16> to vector<1xi16>") == 2
    assert kernel.count("llvm.zext") == 2
    assert kernel.count(": vector<1xi16> to vector<1xi32>") == 2
    assert kernel.count("llvm.mlir.constant(dense<16> : vector<1xi32>)") == 2
    assert kernel.count("llvm.shl") == 2
    assert kernel.count(": vector<1xi32> to vector<1xf32>") == 2
    assert "llvm.fadd" in kernel
    assert "llvm.fptrunc" in kernel and "vector<1xf32> to vector<1xbf16>" in kernel
    # Host code keeps its extension.
    assert "llvm.fpext %arg0 : bf16 to f32" in text[text.index("func.func @host") :]


def test_scalar_extensions_through_several_ops() -> None:
    """Scalars, and an extension reached through ops with float results (a
    product, a select, a call): every extension the rounded value is computed
    from, each once, also when two truncations share it."""
    body = """%ea = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      %c0 = llvm.mlir.constant(0.0 : f32) : f32
      %m = llvm.fmul %ea, %eb : f32
      %gt = llvm.fcmp "ogt" %m, %c0 : f32
      %sel = llvm.select %gt, %m, %c0 : i1, f32
      %ex = llvm.call spir_funccc @_Z15__spirv_ocl_expf(%sel) : (f32) -> f32
      %t = llvm.fptrunc %ex : f32 to bf16
      %t2 = llvm.fptrunc %m : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>
      llvm.store %t2, %p : bf16, !llvm.ptr<1>"""
    source = _module(body, "%a: bf16, %b: bf16").replace(
        "    llvm.func spir_kernelcc",
        "    llvm.func spir_funccc @_Z15__spirv_ocl_expf(f32) -> f32\n"
        "    llvm.func spir_kernelcc",
        1,
    )
    count, text = _rewrite(source)
    assert count == 2
    kernel = _kernel(text)
    assert "llvm.fpext" not in kernel
    assert kernel.count(": bf16 to i16") == 2
    assert kernel.count(": i16 to i32") == 2
    assert kernel.count("llvm.mlir.constant(16 : i32) : i32") == 2
    assert kernel.count(": i32 to f32") == 2
    assert kernel.count("llvm.fptrunc") == 2


_DIAMOND = """%ea = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      %keep = llvm.fpext %a : bf16 to f32
      llvm.store %keep, %q : f32, !llvm.ptr<1>
      llvm.cond_br %c, ^left, ^right
    ^left:
      llvm.br ^join(%ea, %eb : f32, f32)
    ^right:
      llvm.br ^join(%eb, %ea : f32, f32)
    ^join(%v: f32, %u: f32):
      %s = llvm.fmul %v, %u : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>"""

_COND_BR_OPERANDS = """%ea = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      %keep = llvm.fpext %b : bf16 to f32
      llvm.cond_br %c, ^join(%ea : f32), ^other(%keep : f32)
    ^other(%w: f32):
      llvm.store %w, %q : f32, !llvm.ptr<1>
      llvm.br ^join(%eb : f32)
    ^join(%v: f32):
      %s = llvm.fmul %v, %v : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>"""

_BR_POSITIONS = """%keep = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      llvm.br ^join(%keep, %eb : f32, f32)
    ^join(%v: f32, %u: f32):
      llvm.store %v, %q : f32, !llvm.ptr<1>
      %s = llvm.fmul %u, %u : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>"""

_LOOP = """%ea = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      %zero = llvm.mlir.constant(0 : i32) : i32
      %step = llvm.mlir.constant(1 : i32) : i32
      %n = llvm.mlir.constant(4 : i32) : i32
      llvm.br ^head(%zero, %ea : i32, f32)
    ^head(%i: i32, %acc: f32):
      %more = llvm.icmp "slt" %i, %n : i32
      llvm.cond_br %more, ^body, ^exit
    ^body:
      %next = llvm.fadd %acc, %eb : f32
      %i1 = llvm.add %i, %step : i32
      llvm.br ^head(%i1, %next : i32, f32)
    ^exit:
      %t = llvm.fptrunc %acc : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>"""

_SWITCH = """%ea = llvm.fpext %a : bf16 to f32
      %eb = llvm.fpext %b : bf16 to f32
      %sel = llvm.zext %c : i1 to i32
      llvm.switch %sel : i32, ^join(%ea : f32) [
        1: ^join(%eb : f32)
      ]
    ^join(%v: f32):
      %s = llvm.fmul %v, %v : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>"""

_BRANCH_ARGS = "%a: bf16, %b: bf16, %c: i1, %q: !llvm.ptr<1>"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_DIAMOND, 2),
        (_COND_BR_OPERANDS, 2),
        (_BR_POSITIONS, 1),
        (_LOOP, 2),
        (_SWITCH, 2),
    ],
    ids=["diamond", "cond_br_operands", "br_positions", "loop_carried", "switch"],
)
def test_extensions_through_block_arguments(body: str, expected: int) -> None:
    """A truncated value computed from block arguments (phis): the extensions
    the predecessors pass are rewritten (IGC folds a phi of ``fpext``s into an
    ``fpext`` of a phi and shrinks the product; device repro ``phi_mul``).
    ``llvm.br`` passes its operands by position (only the argument that
    reaches the truncation counts), ``llvm.cond_br`` each successor its
    segment (the extension passed to the other block, stored as f32, is not a
    candidate), a loop passes its init and latch values; another terminator
    (``switch``) is taken conservatively. Extensions feeding only an f32 store
    (``%keep``) stay."""
    count, text = _rewrite(_module(body, _BRANCH_ARGS))
    assert count == expected
    kernel = _kernel(text)
    kept = 1 if body in (_DIAMOND, _COND_BR_OPERANDS, _BR_POSITIONS) else 0
    assert kernel.count("llvm.fpext") == kept
    assert kernel.count("llvm.shl") == expected
    assert kernel.count("llvm.fptrunc") == 1


def test_function_arguments_end_the_walk() -> None:
    """A truncated value computed from an f32 function argument (an entry
    block argument, no predecessors) has no extension to rewrite."""
    body = """%e = llvm.fpext %a : bf16 to f32
      %s = llvm.fmul %x, %x : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>
      llvm.store %e, %p : f32, !llvm.ptr<1>"""
    source = _module(body, "%a: bf16, %x: f32")
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


@pytest.mark.parametrize(
    ("body", "args"),
    [
        # fptrunc(fpext x): exact, nothing to shrink.
        (
            """%e = llvm.fpext %a : vector<1xbf16> to vector<1xf32>
      %t = llvm.fptrunc %e : vector<1xf32> to vector<1xbf16>
      llvm.store %t, %p : vector<1xbf16>, !llvm.ptr<1>""",
            "%a: vector<1xbf16>",
        ),
        # An f32 result: no truncation.
        (
            """%e = llvm.fpext %a : bf16 to f32
      %s = llvm.fadd %e, %e : f32
      llvm.store %s, %p : f32, !llvm.ptr<1>""",
            "%a: bf16",
        ),
        # f16 extensions and an f16 result.
        (
            """%e = llvm.fpext %a : f16 to f32
      %s = llvm.fadd %e, %e : f32
      %t = llvm.fptrunc %s : f32 to f16
      llvm.store %t, %p : f16, !llvm.ptr<1>""",
            "%a: f16",
        ),
        # bf16-derived, rounded to f16: not a bf16 truncation.
        (
            """%e = llvm.fpext %a : bf16 to f32
      %s = llvm.fadd %e, %e : f32
      %t = llvm.fptrunc %s : f32 to f16
      llvm.store %t, %p : f16, !llvm.ptr<1>""",
            "%a: bf16",
        ),
        # A rounded f32 input (e.g. a GEMM accumulator): no extension.
        (
            """%s = llvm.fadd %a, %a : f32
      %t = llvm.fptrunc %s : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # Through an integer op: the f32 value is not computed by float ops.
        (
            """%e = llvm.fpext %a : bf16 to f32
      %i = llvm.bitcast %e : f32 to i32
      %f = llvm.bitcast %i : i32 to f32
      %t = llvm.fptrunc %f : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>""",
            "%a: bf16",
        ),
    ],
    ids=["exact_roundtrip", "f32_result", "f16", "f16_result", "f32_input", "integer"],
)
def test_other_conversions_are_untouched(body: str, args: str) -> None:
    source = _module(body, args)
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


def test_non_xevm_gpu_modules_are_untouched() -> None:
    source = _module(_ADD, "%a: vector<1xbf16>, %b: vector<1xbf16>", target="")
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


@pytest.mark.isolated
def test_bf16_add_kernel_extensions_are_rewritten() -> None:
    """``bf16_add`` (the device regression's kernel) at ``xevm``: every bf16
    extension feeds a rounded sum, and is rewritten; a GEMM (1_Square, an f32
    accumulator rounded to bf16) has none."""
    text = lower_xpu(bf16_add, [xpu_tensor(256, 512, dtype=torch.bfloat16)] * 3, "xevm")
    kernel = text[text.index("gpu.module") :]
    extensions = kernel.count("llvm.fpext")
    assert extensions > 0 and "llvm.fptrunc" in kernel
    count, rewritten = _rewrite(text)
    assert count == extensions
    assert "llvm.fpext" not in rewritten[rewritten.index("gpu.module") :]
    assert rewritten.count("llvm.shl") >= count

    problem = next(p for p in PROBLEMS if p.name == "1_Square_matrix_multiplication_")
    gemm = lower_xpu(
        problem.kernel, [xpu_tensor(512, 512, dtype=torch.bfloat16)] * 2, "xevm"
    )
    assert "llvm.fptrunc" in gemm
    count, rewritten = _rewrite(gemm)
    assert count == 0
