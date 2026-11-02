"""``fold_region_insert`` on handwritten IR: the exact pattern folds and the module
verifies, anything else is left untouched."""

from __future__ import annotations

from mlir import ir
import pytest

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir_transforms.fold_region_inserts import (
    fold_region_insert,
)

TEMPLATE = """
func.func @f(%x: tensor<1x64xf32>, %c: tensor<8x512x64xf32>, %pos: index, %y: tensor<64xf32>) -> tensor<8x512x64xf32> {{
  %r = scf.forall (%i) in (8) shared_outs(%o = %c) -> tensor<8x512x64xf32> {{
    {body}
    scf.forall.in_parallel {{
      {par}
    }}
  }}
  return %r : tensor<8x512x64xf32>
}}
"""
CASES = {
    "positive, dynamic inner offset": (
        True,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "positive, static inner offset": (
        True,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, 511, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "positive, both dynamic in one dim": (
        True,
        """
    %l = tensor.extract_slice %o[0, %i, 0] [8, 1, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<8x1x64xf32>
    %z = tensor.empty() : tensor<8x1x32xf32>
    %s = tensor.insert_slice %z into %l[0, 0, %pos] [8, 1, 32] [1, 1, 1] : tensor<8x1x32xf32> into tensor<8x1x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[0, %i, 0] [8, 1, 64] [1, 1, 1] : tensor<8x1x64xf32> into tensor<8x512x64xf32>",
    ),
    "positive, region read elsewhere": (
        True,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %read = tensor.extract_slice %l[0, 3, 0] [1, 1, 64] [1, 1, 1] : tensor<1x512x64xf32> to tensor<1x1x64xf32>
    %s = tensor.insert_slice %read into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "other offsets of the outer": (
        False,
        """
    %l = tensor.extract_slice %o[0, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "load from another tensor": (
        False,
        """
    %l = tensor.extract_slice %c[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "stored value has another use": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
    %u = tensor.extract_slice %s[0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x512x64xf32> to tensor<1x1x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "two inserts chained": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s0 = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %s0[0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "inner stride 2": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 2, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "load stride 2": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 256, 64] [1, 2, 1] : tensor<8x512x64xf32> to tensor<1x256x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x256x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 256, 64] [1, 2, 1] : tensor<1x256x64xf32> into tensor<8x512x64xf32>",
    ),
    "rank-reduced load": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<512x64xf32>
    %s = tensor.insert_slice %x into %l[%pos, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<512x64xf32> into tensor<8x512x64xf32>",
    ),
    "dynamic size of the inner": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %d = tensor.cast %x : tensor<1x64xf32> to tensor<1x?xf32>
    %n = arith.constant 64 : index
    %s = tensor.insert_slice %d into %l[0, %pos, 0] [1, 1, %n] [1, 1, 1] : tensor<1x?xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "dynamic size of the region": (
        False,
        """
    %n = arith.constant 512 : index
    %l = tensor.extract_slice %o[%i, 0, 0] [1, %n, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x?x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x?x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, %n, 64] [1, 1, 1] : tensor<1x?x64xf32> into tensor<8x512x64xf32>",
    ),
    "stored value is not an insert": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.empty() : tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
    "two parallel stores of one load": (
        None,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
    %s2 = tensor.insert_slice %x into %l[0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        """tensor.parallel_insert_slice %s into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>
      tensor.parallel_insert_slice %s2 into %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>""",
    ),
    "load after the insert? outer writes another region of the same size": (
        False,
        """
    %l = tensor.extract_slice %o[%i, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<8x512x64xf32> to tensor<1x512x64xf32>
    %s = tensor.insert_slice %x into %l[0, %pos, 0] [1, 1, 64] [1, 1, 1] : tensor<1x64xf32> into tensor<1x512x64xf32>
  """,
        "tensor.parallel_insert_slice %s into %o[0, 0, 0] [1, 512, 64] [1, 1, 1] : tensor<1x512x64xf32> into tensor<8x512x64xf32>",
    ),
}


def _inserts(module: ir.Module) -> list[ir.Operation]:
    found = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "tensor.parallel_insert_slice":
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


@pytest.mark.parametrize("name", sorted(CASES))
def test_fold(name: str) -> None:
    expected, body, stores = CASES[name]
    with ir.Context():
        module = ir.Module.parse(TEMPLATE.format(body=body, par=stores))
        before = str(module)
        folded = [fold_region_insert(op.opview) for op in _inserts(module)]
        assert module.operation.verify()
        if expected is None:  # two stores of one load: each stored slice is its own
            return
        assert folded == [expected]
        if not expected:
            assert str(module) == before
