"""``rank_reduce_transfers``: transfers of a lower-rank vector from a buffer
moved onto the matrix view its leading indices select."""

from __future__ import annotations

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest

from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import rank_reduce_transfers


def _apply(source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            rank_reduce_transfers(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


def _read(buffer: str) -> str:
    """A 2-D tile read from the 3-D ``buffer`` at batch ``%b``."""
    return f"""
func.func @f(%buf: {buffer}, %b: index) -> vector<16x32xf32> {{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.transfer_read %buf[%b, %c0, %c0], %pad {{in_bounds = [true, true]}}
      : {buffer}, vector<16x32xf32>
  return %r : vector<16x32xf32>
}}
"""


def test_rank_reduce_transfers_leading_dim() -> None:
    text = _apply(_read("memref<8x64x32xf32>"))
    assert "memref.collapse_shape" in text
    assert "vector.transfer_read %collapse_shape[%c0, %c0]" in text


@pytest.mark.parametrize(
    "buffer",
    # No strided layout: get_strides_and_offset raises RuntimeError.
    ["memref<8x64x32xf32, affine_map<(d0, d1, d2) -> (d2, d1, d0)>>"],
    ids=["affine_layout"],
)
def test_rank_reduce_transfers_leaves_non_strided(buffer: str) -> None:
    source = _read(buffer)
    assert _apply(source) == _parsed(source)


def test_rank_reduce_transfers_leading_zero_index() -> None:
    """Regression (6_large_K's ``_sum_splits``): at a constant zero leading
    index of a contiguous buffer the view has the identity layout, and so must
    the collapse (``strided<[32, 1]>`` failed to verify)."""
    source = """
func.func @f(%buf: memref<8x64x32xf32>) -> vector<16x32xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.transfer_read %buf[%c0, %c0, %c0], %pad {in_bounds = [true, true]}
      : memref<8x64x32xf32>, vector<16x32xf32>
  return %r : vector<16x32xf32>
}
"""
    text = _apply(source)
    assert "into memref<64x32xf32>" in text


def test_rank_reduce_transfers_inner_dim() -> None:
    """A unit dim inside the tile (one K chunk ``[%m, %s, %k]`` of an
    ``[M, S, K/S]`` view): the view keeps the outer dim's stride (rows K
    apart) and collapses the dropped dim into the next one's group."""
    source = """
func.func @f(%buf: memref<256x32x1024xbf16>, %s: index, %k: index)
    -> vector<256x32xbf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : bf16
  %r = vector.transfer_read %buf[%c0, %s, %k], %pad
      {in_bounds = [true, true],
       permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>}
      : memref<256x32x1024xbf16>, vector<256x32xbf16>
  return %r : vector<256x32xbf16>
}
"""
    text = _apply(source)
    assert "[[0], [1, 2]]" in text
    assert "into memref<256x1024xbf16, strided<[32768, 1], offset: ?>>" in text
    assert "vector.transfer_read %collapse_shape[%c0, %arg2]" in text


def test_rank_reduce_transfers_leaves_transposing_maps() -> None:
    """A map that drops a dim and transposes the others is left alone."""
    source = """
func.func @f(%buf: memref<8x64x32xf32>, %b: index) -> vector<32x64xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.transfer_read %buf[%b, %c0, %c0], %pad
      {in_bounds = [true, true],
       permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>}
      : memref<8x64x32xf32>, vector<32x64xf32>
  return %r : vector<32x64xf32>
}
"""
    assert _apply(source) == _parsed(source)


@pytest.mark.parametrize(
    ("buffer", "indices", "vector", "map_attr", "collapsed"),
    [
        # A kept dim of size 1 shares its group with the dropped dim: the
        # collapsed dim takes the dropped dim's stride (as the verifier infers).
        (
            "memref<16x4x1xf32, strided<[8, 2, 1]>>",
            "%c0, %b, %c0",
            "vector<16x1xf32>",
            ", permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>",
            "memref<16x1xf32, strided<[8, 2], offset: ?>>",
        ),
        # The same with a dropped leading dim.
        (
            "memref<4x1x16xf32, strided<[64, 32, 1]>>",
            "%b, %c0, %c0",
            "vector<1x16xf32>",
            "",
            "memref<1x16xf32, strided<[64, 1], offset: ?>>",
        ),
    ],
    ids=["inner", "leading"],
)
def test_rank_reduce_transfers_unit_kept_dim(
    buffer: str, indices: str, vector: str, map_attr: str, collapsed: str
) -> None:
    """Regression (review): ``strides[kept]`` gave a collapse type the verifier
    rejects when a kept dim of size 1 sits in a group with a dropped dim of a
    non-contiguous buffer."""
    source = f"""
func.func @f(%buf: {buffer}, %b: index) -> {vector} {{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.transfer_read %buf[{indices}], %pad
      {{in_bounds = [true, true]{map_attr}}} : {buffer}, {vector}
  return %r : {vector}
}}
"""
    assert f"into {collapsed}" in _apply(source)


def test_rank_reduce_transfers_leaves_dynamic_shapes() -> None:
    source = _read("memref<?x?x32xf32>")
    assert _apply(source) == _parsed(source)
