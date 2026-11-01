"""``fold_slice_unit_dims``: a collapse that only drops unit dims of a slice's
result folded into a rank-reducing slice (e.g. one K chunk of a split K)."""

from __future__ import annotations

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest

from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import fold_slice_unit_dims


def _apply(source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            fold_slice_unit_dims(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


def _function(slice_type: str, slice_args: str, collapse: str) -> str:
    return f"""
func.func @f(%t: tensor<256x32x1024xbf16>, %s: index, %k: index) -> {collapse.split(" into ")[1]} {{
  %c0 = arith.constant 0 : index
  %e = tensor.extract_slice %t{slice_args} : tensor<256x32x1024xbf16> to {slice_type}
  %r = tensor.collapse_shape %e {collapse}
  return %r : {collapse.split(" into ")[1]}
}}
"""


def test_fold_slice_unit_dims_inner_unit_dim() -> None:
    source = _function(
        "tensor<256x1x32xbf16>",
        "[0, %s, %k] [256, 1, 32] [1, 1, 1]",
        "[[0, 1], [2]] : tensor<256x1x32xbf16> into tensor<256x32xbf16>",
    )
    text = _apply(source)
    assert "tensor.collapse_shape" not in text
    assert (
        "tensor.extract_slice %arg0[0, %arg1, %arg2] [256, 1, 32] [1, 1, 1] : "
        "tensor<256x32x1024xbf16> to tensor<256x32xbf16>"
    ) in text


@pytest.mark.parametrize(
    ("slice_type", "slice_args", "collapse"),
    [
        # Merges two non-unit dims: not a rank reduction.
        (
            "tensor<2x1x32xbf16>",
            "[0, %s, %k] [2, 1, 32] [1, 1, 1]",
            "[[0, 1, 2]] : tensor<2x1x32xbf16> into tensor<64xbf16>",
        ),
        # A strided slice.
        (
            "tensor<256x1x32xbf16>",
            "[0, %s, %k] [256, 1, 32] [1, 1, 2]",
            "[[0, 1], [2]] : tensor<256x1x32xbf16> into tensor<256x32xbf16>",
        ),
    ],
    ids=["merges_dims", "strided"],
)
def test_fold_slice_unit_dims_leaves_others(
    slice_type: str, slice_args: str, collapse: str
) -> None:
    source = _function(slice_type, slice_args, collapse)
    assert _apply(source) == _parsed(source)


def test_fold_slice_unit_dims_leaves_collapse_of_non_slice() -> None:
    source = """
func.func @f(%t: tensor<256x1x32xbf16>) -> tensor<256x32xbf16> {
  %r = tensor.collapse_shape %t [[0, 1], [2]] : tensor<256x1x32xbf16> into tensor<256x32xbf16>
  return %r : tensor<256x32xbf16>
}
"""
    assert _apply(source) == _parsed(source)


@pytest.mark.parametrize(
    ("source", "reduced"),
    [
        (
            "%t: tensor<256x32x1024xbf16>",
            "%e = tensor.extract_slice %t[64, 5, 96] [64, 1, 32] [1, 1, 1] : "
            "tensor<256x32x1024xbf16> to tensor<64x1x32xbf16>\n"
            "  %r = tensor.collapse_shape %e [[0, 1], [2]] : tensor<64x1x32xbf16> "
            "into tensor<64x32xbf16>",
        ),
        (
            "%t: tensor<4x256x8x1024xbf16>, %a: index, %s: index",
            "%e = tensor.extract_slice %t[%a, 0, %s, 0] [1, 256, 1, 32] [1, 1, 1, 1] "
            ": tensor<4x256x8x1024xbf16> to tensor<1x256x1x32xbf16>\n"
            "  %r = tensor.collapse_shape %e [[0, 1], [2, 3]] : "
            "tensor<1x256x1x32xbf16> into tensor<256x32xbf16>",
        ),
        (
            "%t: tensor<4x1x256x32xbf16>, %a: index",
            "%e = tensor.extract_slice %t[%a, 0, 0, 0] [1, 1, 256, 32] [1, 1, 1, 1] "
            ": tensor<4x1x256x32xbf16> to tensor<1x256x32xbf16>\n"
            "  %r = tensor.collapse_shape %e [[0, 1], [2]] : tensor<1x256x32xbf16> "
            "into tensor<256x32xbf16>",
        ),
    ],
    ids=["static_offsets", "several_unit_dims", "already_rank_reduced"],
)
def test_fold_slice_unit_dims_more_slices(source: str, reduced: str) -> None:
    """Static non-zero and dynamic offsets, several unit dims, and a slice that
    already drops some unit dims all fold to one rank-reducing slice of the
    collapse's type."""
    result = reduced.rsplit(" into ", 1)[1]
    text = _apply(
        f"func.func @f({source}) -> {result} {{\n  {reduced}\n"
        f"  return %r : {result}\n}}"
    )
    assert "tensor.collapse_shape" not in text
    assert f"to {result}" in text
