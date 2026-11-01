"""``materialize_tile_copies``: a grid loop's store of a tile it loaded
unchanged becomes a ``linalg.copy`` into the stored slice (XPU: a kernel with
no compute op had nothing to lower)."""

from __future__ import annotations

import re

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest

from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import materialize_tile_copies


def _apply(source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            materialize_tile_copies(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


def _grid_loop(body: str, store_sizes: str, out: str = "tensor<64x256xbf16>") -> str:
    """A grid loop over 2 row tiles storing ``%t`` (``tensor<32x256xbf16>``),
    computed by ``body``."""
    return f"""
#map = affine_map<(d0) -> (d0 * 32)>
func.func @f(%a: tensor<64x256xbf16>, %o: {out}) -> {out} {{
  %r = scf.forall (%i) in (2) shared_outs(%s = %o) -> ({out}) {{
    %off = affine.apply #map(%i)
    {body}
    scf.forall.in_parallel {{
      tensor.parallel_insert_slice %t into %s{store_sizes} : tensor<32x256xbf16> into {out}
    }}
  }}
  return %r : {out}
}}
"""


_LOAD = (
    "%t = tensor.extract_slice %a[%off, 0] [32, 256] [1, 1] : "
    "tensor<64x256xbf16> to tensor<32x256xbf16>"
)


def test_tile_copy_becomes_linalg_copy() -> None:
    text = _apply(_grid_loop(_LOAD, "[%off, 0] [32, 256] [1, 1]"))
    # The copy writes the stored slice of the shared output, which is stored.
    assert (
        "%extracted_slice_0 = tensor.extract_slice %arg3[%1, 0] [32, 256] [1, 1] "
        ": tensor<64x256xbf16> to tensor<32x256xbf16>"
    ) in text
    assert re.search(
        r"%2 = linalg\.copy [^\n]*ins\(%extracted_slice : tensor<32x256xbf16>\) "
        r"outs\(%extracted_slice_0 : tensor<32x256xbf16>\)",
        text,
    )
    assert "tensor.parallel_insert_slice %2 into %arg3[%1, 0]" in text


def test_one_dim_tile_copy() -> None:
    source = """
#map = affine_map<(d0) -> (d0 * 256)>
func.func @f(%a: tensor<4096xf32>, %o: tensor<4096xf32>) -> tensor<4096xf32> {
  %r = scf.forall (%i) in (16) shared_outs(%s = %o) -> (tensor<4096xf32>) {
    %off = affine.apply #map(%i)
    %t = tensor.extract_slice %a[%off] [256] [1] : tensor<4096xf32> to tensor<256xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %t into %s[%off] [256] [1] : tensor<256xf32> into tensor<4096xf32>
    }
  }
  return %r : tensor<4096xf32>
}
"""
    text = _apply(source)
    assert re.search(
        r"linalg\.copy [^\n]*ins\(%extracted_slice : tensor<256xf32>\)", text
    )


_COMPUTED = (
    "%t0 = tensor.extract_slice %a[%off, 0] [32, 256] [1, 1] : "
    "tensor<64x256xbf16> to tensor<32x256xbf16>\n"
    "    %e = tensor.empty() : tensor<32x256xbf16>\n"
    "    %t = linalg.copy ins(%t0 : tensor<32x256xbf16>) outs(%e : "
    "tensor<32x256xbf16>) -> tensor<32x256xbf16>"
)


@pytest.mark.parametrize(
    ("body", "store_sizes", "out"),
    [
        # The stored tile is computed (a linalg op stores it): nothing to do.
        (_COMPUTED, "[%off, 0] [32, 256] [1, 1]", "tensor<64x256xbf16>"),
        # A rank-changing store (a unit dim added): not the exact pattern.
        (_LOAD, "[0, %off, 0] [1, 32, 256] [1, 1, 1]", "tensor<1x64x256xbf16>"),
    ],
    ids=["computed_tile", "rank_changing_store"],
)
def test_other_stores_are_left_alone(body: str, store_sizes: str, out: str) -> None:
    source = _grid_loop(body, store_sizes, out)
    assert _apply(source) == _parsed(source)


def test_runtime_shaped_tile_is_left_alone() -> None:
    source = """
func.func @f(%a: tensor<64x256xbf16>, %o: tensor<64x256xbf16>, %n: index) -> tensor<64x256xbf16> {
  %r = scf.forall (%i) in (2) shared_outs(%s = %o) -> (tensor<64x256xbf16>) {
    %t = tensor.extract_slice %a[%i, 0] [%n, 256] [1, 1] : tensor<64x256xbf16> to tensor<?x256xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %t into %s[%i, 0] [%n, 256] [1, 1] : tensor<?x256xbf16> into tensor<64x256xbf16>
    }
  }
  return %r : tensor<64x256xbf16>
}
"""
    assert _apply(source) == _parsed(source)


def test_sequential_insert_is_left_alone() -> None:
    """Only grid-loop stores (``parallel_insert_slice``) are tile stores."""
    source = """
func.func @f(%a: tensor<64x256xbf16>, %o: tensor<64x256xbf16>) -> tensor<64x256xbf16> {
  %t = tensor.extract_slice %a[0, 0] [32, 256] [1, 1] : tensor<64x256xbf16> to tensor<32x256xbf16>
  %r = tensor.insert_slice %t into %o[32, 0] [32, 256] [1, 1] : tensor<32x256xbf16> into tensor<64x256xbf16>
  return %r : tensor<64x256xbf16>
}
"""
    assert _apply(source) == _parsed(source)
