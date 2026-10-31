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
