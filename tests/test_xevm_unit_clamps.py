"""``xevm_unit_clamps``: a compare-and-select with 1 of XeVM GPU modules compares
with the number next to 1; everything else is left alone. IR
only: no XPU or SPIR-V target needed. The binaries and the device are in
``tests/test_xpu_igc_workarounds.py``."""

from __future__ import annotations

import math
import re
import struct

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir_transforms import xevm_unit_clamps
from helion_mlir_backend._compiler.mlir_transforms.xevm_unit_clamps import (
    unit_clamps_without_sat,
)


def _module(body: str, target: str = '[#xevm.target<chip = "bmg">]') -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_kernelcc @k(%x: f32, %b: bf16, %h: f16, %v: vector<16xf32>) {{
      %zero = llvm.mlir.constant(0.0 : f32) : f32
      %mzero = llvm.mlir.constant(-0.0 : f32) : f32
      %one = llvm.mlir.constant(1.0 : f32) : f32
      %two = llvm.mlir.constant(2.0 : f32) : f32
      %vzero = llvm.mlir.constant(dense<0.0> : vector<16xf32>) : vector<16xf32>
      %vone = llvm.mlir.constant(dense<1.0> : vector<16xf32>) : vector<16xf32>
      %bzero = llvm.mlir.constant(0.0 : bf16) : bf16
      %bone = llvm.mlir.constant(1.0 : bf16) : bf16
      %hzero = llvm.mlir.constant(0.0 : f16) : f16
      %hone = llvm.mlir.constant(1.0 : f16) : f16
      {body}
      llvm.return
    }}
  }}
}}"""


def _rewrite(text: str) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = unit_clamps_without_sat(module)
        assert module.operation.verify()
        return count, str(module)


def _parsed(text: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(text))


_MAX_THEN_MIN = """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "{p}" %y, %one : f32
      %z = llvm.select %c1, {selected} : i1, f32"""


@pytest.mark.parametrize(
    ("predicate", "selected", "replacement", "bits"),
    [
        ("ogt", "%one, %y", "oge", "above"),
        ("ole", "%y, %one", "olt", "above"),
        ("oge", "%one, %y", "ogt", "below"),
        ("olt", "%y, %one", "ole", "below"),
        ("ugt", "%one, %y", "uge", "above"),
        ("ule", "%y, %one", "ult", "above"),
        ("uge", "%one, %y", "ugt", "below"),
        ("ult", "%y, %one", "ule", "below"),
    ],
)
def test_compare_with_one_moves_to_the_next_number(
    predicate: str, selected: str, replacement: str, bits: str
) -> None:
    count, text = _rewrite(
        _module(_MAX_THEN_MIN.format(p=predicate, selected=selected))
    )
    assert count == 1
    value = "1.00000012" if bits == "above" else "0.99999994"
    assert f'llvm.fcmp "{replacement}" ' in text
    # Only the compare with 0 (an olt) stays.
    assert text.count(f'llvm.fcmp "{predicate}" ') == (predicate == "olt")
    assert f"llvm.mlir.constant({value}" in text
    assert text.count("llvm.select") == 2


def test_min_then_max_changes_the_inner_compare() -> None:
    body = """
      %c0 = llvm.fcmp "ogt" %x, %one : f32
      %y = llvm.select %c0, %one, %x : i1, f32
      %c1 = llvm.fcmp "olt" %y, %zero : f32
      %z = llvm.select %c1, %zero, %y : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "oge" %arg0' in text
    assert 'llvm.fcmp "olt" ' in text


def test_constant_on_the_left_of_the_compare() -> None:
    """``1 < y`` is ``y > 1``: ``y >= 1 + ulp``."""
    body = """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "olt" %one, %y : f32
      %z = llvm.select %c1, %one, %y : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "oge" ' in text
    assert "llvm.mlir.constant(1.00000012" in text


def test_splat_vectors() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %v, %vzero : vector<16xf32>
      %y = llvm.select %c0, %vzero, %v : vector<16xi1>, vector<16xf32>
      %c1 = llvm.fcmp "ogt" %y, %vone : vector<16xf32>
      %z = llvm.select %c1, %vone, %y : vector<16xi1>, vector<16xf32>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert "llvm.mlir.constant(dense<1.00000012> : vector<16xf32>)" in text
    assert 'llvm.fcmp "oge" ' in text


def test_bf16_selects_with_compares_on_the_extensions() -> None:
    """What a bf16 ``clamp_max(clamp_min(x, 0.0), 1.0)`` folds to; IGC's ``(sat)``
    of it does nothing. The compare stays in f32: its next number is f32's."""
    body = """
      %e0 = llvm.fpext %b : bf16 to f32
      %c0 = llvm.fcmp "olt" %e0, %zero : f32
      %y = llvm.select %c0, %bzero, %b : i1, bf16
      %e1 = llvm.fpext %y : bf16 to f32
      %c1 = llvm.fcmp "ogt" %e1, %one : f32
      %z = llvm.select %c1, %bone, %y : i1, bf16"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert "llvm.mlir.constant(1.00000012 : f32)" in text
    assert 'llvm.fcmp "ogt" ' not in text


def test_a_conversion_between_the_selects() -> None:
    """A rounding between the two: IGC folds it, so it does not stop the match."""
    body = """
      %e0 = llvm.fpext %b : bf16 to f32
      %c0 = llvm.fcmp "olt" %e0, %zero : f32
      %s = llvm.select %c0, %zero, %e0 : i1, f32
      %y = llvm.fptrunc %s : f32 to bf16
      %e1 = llvm.fpext %y : bf16 to f32
      %c1 = llvm.fcmp "ogt" %e1, %one : f32
      %z = llvm.select %c1, %one, %e1 : i1, f32"""
    assert _rewrite(_module(body))[0] == 1


def test_f16_compares_in_f16() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %h, %hzero : f16
      %y = llvm.select %c0, %hzero, %h : i1, f16
      %c1 = llvm.fcmp "ogt" %y, %hone : f16
      %z = llvm.select %c1, %hone, %y : i1, f16"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert re.search(r"llvm\.mlir\.constant\(1\.00098\S* : f16\)", text)


def test_negative_zero_bound() -> None:
    """IGC's zero test takes -0 too."""
    body = """
      %c0 = llvm.fcmp "olt" %x, %mzero : f32
      %y = llvm.select %c0, %mzero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32"""
    assert _rewrite(_module(body))[0] == 1


def test_a_shared_compare_stays_for_its_other_user() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32
      %w = llvm.select %c1, %x, %y : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "ogt" ' in text
    assert 'llvm.fcmp "oge" ' in text


def test_clamp_of_a_clamp_changes_each_compare_with_one_once() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y0 = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y0, %one : f32
      %y1 = llvm.select %c1, %one, %y0 : i1, f32
      %c2 = llvm.fcmp "olt" %y1, %zero : f32
      %y2 = llvm.select %c2, %zero, %y1 : i1, f32
      %c3 = llvm.fcmp "ogt" %y2, %one : f32
      %y3 = llvm.select %c3, %one, %y2 : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 2
    assert 'llvm.fcmp "ogt" ' not in text


def test_one_select_with_one() -> None:
    """Regression: no select with 0 beside it (``F.hardtanh(F.relu(x))``: IGC's
    optimizer forms the pair by folding what lies between)."""
    body = """
      %c1 = llvm.fcmp "ogt" %x, %one : f32
      %z = llvm.select %c1, %one, %x : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "oge" ' in text and 'llvm.fcmp "ogt" ' not in text


def test_two_selects_with_one() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %x, %one : f32
      %y = llvm.select %c0, %one, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 2
    assert 'llvm.fcmp "ole" ' in text and 'llvm.fcmp "oge" ' in text


def test_select_between_the_clamps() -> None:
    """A select with -1 between the two: IGC folds it."""
    body = """
      %minus = llvm.mlir.constant(-1.0 : f32) : f32
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y0 = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "olt" %y0, %minus : f32
      %y1 = llvm.select %c1, %minus, %y0 : i1, f32
      %c2 = llvm.fcmp "ogt" %y1, %one : f32
      %z = llvm.select %c2, %one, %y1 : i1, f32"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "oge" ' in text


def test_bound_through_conversions() -> None:
    """The constant 1 behind conversions (bf16: the no-op ``mov (sat) :uw``)."""
    body = """
      %e0 = llvm.fpext %b : bf16 to f32
      %c0 = llvm.fcmp "olt" %e0, %zero : f32
      %y = llvm.select %c0, %bzero, %b : i1, bf16
      %e1 = llvm.fpext %y : bf16 to f32
      %wide = llvm.fpext %bone : bf16 to f32
      %c1 = llvm.fcmp "ogt" %e1, %wide : f32
      %narrow = llvm.fptrunc %one : f32 to bf16
      %z = llvm.select %c1, %narrow, %y : i1, bf16"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert 'llvm.fcmp "oge" ' in text


def test_zero_bound_as_llvm_zero() -> None:
    body = """
      %z0 = llvm.mlir.zero : f32
      %c0 = llvm.fcmp "olt" %x, %z0 : f32
      %y = llvm.select %c0, %z0, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32"""
    assert _rewrite(_module(body))[0] == 1


def test_bitcast_between_the_selects() -> None:
    body = """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %i = llvm.bitcast %y : f32 to i32
      %w = llvm.bitcast %i : i32 to f32
      %c1 = llvm.fcmp "ogt" %w, %one : f32
      %z = llvm.select %c1, %one, %w : i1, f32"""
    assert _rewrite(_module(body))[0] == 1


_LEFT_ALONE = {
    "other_upper_bound": """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %two : f32
      %z = llvm.select %c1, %two, %y : i1, f32""",
    "only_with_zero": """
      %c1 = llvm.fcmp "olt" %x, %zero : f32
      %z = llvm.select %c1, %zero, %x : i1, f32""",
    "other_bounds": """
      %c0 = llvm.fcmp "olt" %x, %two : f32
      %y = llvm.select %c0, %two, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %zero : f32
      %z = llvm.select %c1, %zero, %y : i1, f32""",
    "equality": """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "oeq" %y, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32""",
    "compare_of_another_value": """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %x, %one : f32
      %z = llvm.select %c1, %one, %y : i1, f32""",
    "selects_another_constant": """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %one : f32
      %z = llvm.select %c1, %two, %y : i1, f32""",
    "non_constant_bound": """
      %c0 = llvm.fcmp "olt" %x, %zero : f32
      %y = llvm.select %c0, %zero, %x : i1, f32
      %c1 = llvm.fcmp "ogt" %y, %x : f32
      %z = llvm.select %c1, %x, %y : i1, f32""",
    "minnum_maxnum": """
      %y = llvm.intr.maxnum(%x, %zero) : (f32, f32) -> f32
      %z = llvm.intr.minnum(%y, %one) : (f32, f32) -> f32""",
}


@pytest.mark.parametrize("body", list(_LEFT_ALONE.values()), ids=list(_LEFT_ALONE))
def test_others_are_left_alone(body: str) -> None:
    text = _module(body)
    assert _rewrite(text) == (0, _parsed(text))


def test_modules_without_an_xevm_target_are_left_alone() -> None:
    text = _module(_MAX_THEN_MIN.format(p="ogt", selected="%one, %y"), target="")
    assert _rewrite(text) == (0, _parsed(text))


@pytest.mark.parametrize(
    ("kind", "decode"),
    [
        (ir.F32Type, lambda bits: struct.unpack(">f", bytes.fromhex(bits))[0]),
        (ir.F16Type, lambda bits: struct.unpack(">e", bytes.fromhex(bits))[0]),
        (ir.F64Type, lambda bits: struct.unpack(">d", bytes.fromhex(bits))[0]),
        (
            ir.BF16Type,
            lambda bits: struct.unpack(">f", bytes.fromhex(bits + "0000"))[0],
        ),
    ],
    ids=["f32", "f16", "f64", "bf16"],
)
def test_next_numbers_are_the_neighbours_of_one(kind: type, decode: object) -> None:
    above, below = (decode(text[2:]) for text in xevm_unit_clamps._NEXT_TO_ONE[kind])
    ulp = above - 1.0
    assert ulp > 0.0 and 1.0 - below == ulp / 2
    assert math.frexp(ulp)[1] < 0  # no number of the type lies between


def _holds(predicate: int, lhs: float, rhs: float) -> bool:
    return {
        xevm_unit_clamps._OGT: lhs > rhs,
        xevm_unit_clamps._OGE: lhs >= rhs,
        xevm_unit_clamps._OLT: lhs < rhs,
        xevm_unit_clamps._OLE: lhs <= rhs,
        xevm_unit_clamps._UGT: not lhs <= rhs,
        xevm_unit_clamps._UGE: not lhs < rhs,
        xevm_unit_clamps._ULT: not lhs >= rhs,
        xevm_unit_clamps._ULE: not lhs > rhs,
    }[predicate]


@pytest.mark.parametrize("predicate", list(xevm_unit_clamps._AGAINST_ONE))
def test_each_replacement_is_the_same_test(predicate: int) -> None:
    """For every f32 ``y``: both sides of 1, the bounds, NaN and the infinities."""
    above = struct.unpack(">f", bytes.fromhex("3F800001"))[0]
    below = struct.unpack(">f", bytes.fromhex("3F7FFFFF"))[0]
    replacement, is_above = xevm_unit_clamps._AGAINST_ONE[predicate]
    bound = above if is_above else below
    for y in (
        math.nan,
        math.inf,
        -math.inf,
        0.0,
        -0.0,
        -1.0,
        below,
        1.0,
        above,
        2.0,
        1e-45,
        3e38,
    ):
        assert _holds(predicate, y, 1.0) == _holds(replacement, y, bound)
