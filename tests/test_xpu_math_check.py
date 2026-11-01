"""``xpu_checks.check_math``: math functions no device test verified are rejected
before lowering in the forms IGC miscompiled, verified ones pass. IR only (the
device matrix is ``test_xpu_runtime.py::test_math_functions``)."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests import xpu_math_kernels as mk
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.targets.xpu_checks import check_math


def _check(body: str, args: str = "%a: f32, %h: bf16, %g: f16") -> None:
    text = f"""
func.func @k({args}) {{
  {body}
  return
}}"""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        check_math(module.body.operations[0].operation)


@pytest.mark.parametrize(
    "body",
    [
        # f32 results.
        "%t = math.tanh %a : f32",
        "%t = math.cbrt %a : f32",
        # More math after the function, then the rounding.
        """%t = math.cbrt %a : f32
  %m = arith.mulf %t, %a : f32
  %r = arith.truncf %m : f32 to bf16""",
        # Verified functions rounded straight to 16 bits.
        """%e = math.tanh %a : f32
  %r = arith.truncf %e : f32 to bf16""",
        """%e = math.log %a : f32
  %r = arith.truncf %e : f32 to f16""",
        """%e = math.absf %a : f32
  %r = arith.truncf %e : f32 to bf16""",
        """%e = math.floor %a : f32
  %r = arith.truncf %e : f32 to bf16""",
        # 16-bit math, extended to f32 and rounded back.
        """%t = math.sin %h : bf16
  %x = arith.extf %t : bf16 to f32""",
        "%t = math.cos %g : f16",
        # Verified functions of bf16 values, also after arithmetic on them.
        """%x = arith.extf %h : bf16 to f32
  %e = math.absf %x : f32""",
        """%x = arith.extf %h : bf16 to f32
  %n = arith.negf %x : f32
  %e = math.absf %n : f32""",
        """%x = arith.extf %h : bf16 to f32
  %c = arith.constant 1.0 : f32
  %m = arith.mulf %x, %c : f32
  %e = math.atan %m : f32""",
        """%x = arith.extf %h : bf16 to f32
  %e = math.cosh %x : f32""",
        # Unverified functions of f16 values.
        """%x = arith.extf %g : f16 to f32
  %e = math.cbrt %x : f32""",
        # 16-bit math used by more math.
        """%t = math.cbrt %g : f16
  %e = math.exp %g : f16
  %m = arith.mulf %t, %e : f16""",
    ],
    ids=[
        "tanh-f32",
        "cbrt-f32",
        "more-math",
        "tanh-to-bf16",
        "log-to-f16",
        "abs-to-bf16",
        "floor-to-bf16",
        "sin-bf16-math",
        "cos-f16-math",
        "abs-of-bf16",
        "abs-of-neg-bf16",
        "atan-of-scaled-bf16",
        "cosh-of-bf16",
        "cbrt-of-f16",
        "f16-math-computed",
    ],
)
def test_verified_forms_pass(body: str) -> None:
    _check(body)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        # Functions no device test ran, in the forms IGC miscompiled.
        (
            """%t = math.cbrt %a : f32
  %r = arith.truncf %t : f32 to bf16""",
            r"math\.cbrt rounded to bf16",
        ),
        (
            """%t = math.asinh %a : f32
  %r = arith.truncf %t : f32 to f16""",
            r"math\.asinh rounded to f16",
        ),
        # 16-bit math is extended to f32 and rounded back.
        (
            """%t = math.erfc %h : bf16
  %x = arith.extf %t : bf16 to f32""",
            r"math\.erfc rounded to bf16",
        ),
        ("%t = math.acosh %g : f16", r"math\.acosh rounded to f16"),
        (
            """%x = arith.extf %h : bf16 to f32
  %e = math.cbrt %x : f32""",
            r"math\.cbrt of a bf16 value",
        ),
        # After arithmetic on the bf16 value.
        (
            """%x = arith.extf %h : bf16 to f32
  %n = arith.negf %x : f32
  %e = math.acosh %n : f32""",
            r"math\.acosh of a bf16 value",
        ),
    ],
    ids=[
        "cbrt-bf16",
        "asinh-f16",
        "erfc-bf16-math",
        "acosh-f16-math",
        "cbrt-of-bf16",
        "acosh-of-neg-bf16",
    ],
)
def test_unverified_forms_are_rejected(body: str, match: str) -> None:
    with pytest.raises(UnsupportedOperationError, match=match):
        _check(body)


def _asinh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.asinh(x[tile].to(torch.float32)).to(out.dtype)
    return out


@pytest.mark.isolated
def test_kernels_are_checked_before_lowering() -> None:
    """Through the XPU lowering: an unverified function rounded to bf16 is
    rejected; 22_Tanh's and 30_Softsign's forms lower."""
    args = [xpu_tensor(256, 512, dtype=torch.bfloat16)] * 2
    kernel = helion.kernel(
        backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[128, 256])
    )(_asinh)
    with pytest.raises(UnsupportedOperationError, match=r"math\.asinh rounded to bf16"):
        lower_xpu(kernel, args, "xegpu-wg")
    assert "xegpu.load_nd" in lower_xpu(mk.k_tanh, args, "xegpu-wg")
    assert "xegpu.load_nd" in lower_xpu(mk.k_softsign, args, "xegpu-wg")


@pytest.mark.isolated
def test_gemm_epilogue_and_row_results_lower() -> None:
    """The reviewer's GEMM and reduction forms (gk.py): an epilogue's tanh
    rounded to bf16, tanh before a row maximum, tanh of a row maximum stored as
    a bf16 column (rejected before ``EnableVectorizer=0``)."""
    a = xpu_tensor(512, 256, dtype=torch.bfloat16)
    b = xpu_tensor(256, 512, dtype=torch.bfloat16)
    assert "xegpu.dpas" in lower_xpu(mk.mm_tanh, [a, b], "xegpu-wg")
    x = [xpu_tensor(128, 256, dtype=torch.bfloat16)]
    assert "vector.multi_reduction" in lower_xpu(mk.row_tanh_max, x, "xegpu-wg")
    assert "vector.multi_reduction" in lower_xpu(mk.row_max_tanh, x, "xegpu-wg")
