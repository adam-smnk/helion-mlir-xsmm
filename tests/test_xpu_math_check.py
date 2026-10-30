"""``xpu_checks.check_math``: math functions in forms the device computes wrong
are rejected before lowering, verified forms pass. IR only (the device matrix is
``test_xpu_runtime.py::test_math_functions``)."""

from __future__ import annotations

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
        # More math after the function, then the rounding.
        """%t = math.tanh %a : f32
  %m = arith.mulf %t, %a : f32
  %r = arith.truncf %m : f32 to bf16""",
        # Verified functions rounded straight to 16 bits.
        """%e = math.exp %a : f32
  %r = arith.truncf %e : f32 to bf16""",
        """%e = math.erf %a : f32
  %r = arith.truncf %e : f32 to f16""",
        """%e = math.absf %a : f32
  %r = arith.truncf %e : f32 to f16""",
        # Verified functions of bf16 values, also after arithmetic on them.
        """%x = arith.extf %h : bf16 to f32
  %n = arith.negf %x : f32
  %e = math.exp %n : f32""",
        """%x = arith.extf %h : bf16 to f32
  %c = arith.constant 0.4 : f32
  %m = arith.mulf %x, %c : f32
  %e = math.log1p %m : f32""",
        # f16 values.
        """%x = arith.extf %g : f16 to f32
  %e = math.cosh %x : f32""",
        # Verified rounded straight to 16 bits.
        """%e = math.floor %a : f32
  %r = arith.truncf %e : f32 to bf16""",
        """%e = math.asin %a : f32
  %r = arith.truncf %e : f32 to f16""",
        # abs of an f16 value.
        """%x = arith.extf %g : f16 to f32
  %e = math.absf %x : f32""",
        # 16-bit math used by more math (gelu_tanh_cast's bf16 tanh * exp).
        """%t = math.tanh %g : f16
  %e = math.exp %g : f16
  %m = arith.mulf %t, %e : f16""",
    ],
    ids=[
        "f32",
        "more-math",
        "exp-bf16",
        "erf-f16",
        "abs-f16",
        "exp-of-neg-bf16",
        "log1p-of-scaled-bf16",
        "cosh-of-f16",
        "floor-bf16",
        "asin-f16",
        "abs-f16-in",
        "f16-math-computed",
    ],
)
def test_verified_forms_pass(body: str) -> None:
    _check(body)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            """%t = math.tanh %a : f32
  %r = arith.truncf %t : f32 to bf16""",
            r"math\.tanh rounded to bf16",
        ),
        (
            """%t = math.log %a : f32
  %r = arith.truncf %t : f32 to f16""",
            r"math\.log rounded to f16",
        ),
        # 16-bit math is extended to f32 and rounded back.
        (
            """%t = math.sin %h : bf16
  %x = arith.extf %t : bf16 to f32""",
            r"math\.sin rounded to bf16",
        ),
        ("%t = math.cos %g : f16", r"math\.cos rounded to f16"),
        (
            """%e = math.absf %a : f32
  %r = arith.truncf %e : f32 to bf16""",
            r"math\.absf rounded to bf16",
        ),
        (
            """%x = arith.extf %h : bf16 to f32
  %e = math.absf %x : f32""",
            r"math\.absf of a bf16 value",
        ),
        (
            """%x = arith.extf %h : bf16 to f32
  %e = math.atan %x : f32""",
            r"math\.atan of a bf16 value",
        ),
        # The reviewer's bypasses of a single-user rule: cosh, and abs or atan
        # after arithmetic on the bf16 value (all wrong on the device).
        (
            """%x = arith.extf %h : bf16 to f32
  %e = math.cosh %x : f32""",
            r"math\.cosh of a bf16 value",
        ),
        (
            """%x = arith.extf %h : bf16 to f32
  %n = arith.negf %x : f32
  %e = math.absf %n : f32""",
            r"math\.absf of a bf16 value",
        ),
        (
            """%x = arith.extf %h : bf16 to f32
  %c = arith.constant 1.0 : f32
  %m = arith.mulf %x, %c : f32
  %e = math.atan %m : f32""",
            r"math\.atan of a bf16 value",
        ),
        # abs of a bf16 value other ops use too (softsign): right in the
        # cases measured, but not separable from the wrong forms.
        (
            """%x = arith.extf %h : bf16 to f32
  %e = math.absf %x : f32
  %s = arith.addf %e, %x : f32""",
            r"math\.absf of a bf16 value",
        ),
        # Unverified functions rounded straight to 16 bits.
        (
            """%t = math.tan %a : f32
  %r = arith.truncf %t : f32 to bf16""",
            r"math\.tan rounded to bf16",
        ),
        (
            """%t = math.sinh %a : f32
  %r = arith.truncf %t : f32 to f16""",
            r"math\.sinh rounded to f16",
        ),
    ],
    ids=[
        "tanh-bf16",
        "log-f16",
        "sin-bf16-math",
        "cos-f16-math",
        "abs-to-bf16",
        "abs-of-bf16",
        "atan-of-bf16",
        "cosh-of-bf16",
        "abs-of-neg-bf16",
        "atan-of-scaled-bf16",
        "abs-shared-bf16",
        "tan-bf16",
        "sinh-f16",
    ],
)
def test_wrong_forms_are_rejected(body: str, match: str) -> None:
    with pytest.raises(UnsupportedOperationError, match=match):
        _check(body)


@pytest.mark.isolated
def test_kernels_are_checked_before_lowering() -> None:
    """Through the XPU lowering: 22_Tanh's form is rejected, tanh followed by
    more math lowers."""
    args = [xpu_tensor(256, 512, dtype=torch.bfloat16)] * 2
    with pytest.raises(UnsupportedOperationError, match=r"math\.tanh rounded to bf16"):
        lower_xpu(mk.k_tanh, args, "xegpu-wg")
    assert "xegpu.load_nd" in lower_xpu(mk.k_tanh_x2, args, "xegpu-wg")


@pytest.mark.isolated
def test_gemm_epilogue_and_row_results_are_checked() -> None:
    """The reviewer's GEMM and reduction forms (gk.py): an epilogue's tanh
    rounded to bf16 is rejected; tanh before a row maximum lowers; tanh of a
    row maximum stored as a bf16 column is rejected (knowingly conservative:
    right on the device)."""
    a = xpu_tensor(512, 256, dtype=torch.bfloat16)
    b = xpu_tensor(256, 512, dtype=torch.bfloat16)
    with pytest.raises(UnsupportedOperationError, match=r"math\.tanh rounded to bf16"):
        lower_xpu(mk.mm_tanh, [a, b], "xegpu-wg")
    x = [xpu_tensor(128, 256, dtype=torch.bfloat16)]
    assert "vector.multi_reduction" in lower_xpu(mk.row_tanh_max, x, "xegpu-wg")
    with pytest.raises(UnsupportedOperationError, match=r"math\.tanh rounded to bf16"):
        lower_xpu(mk.row_max_tanh, x, "xegpu-wg")
