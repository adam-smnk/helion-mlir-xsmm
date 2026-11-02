"""``_drop_row_masks`` on hand-written IR: a row condition goes only where all it
decides is a mask, a select of a constant when false."""

from __future__ import annotations

from mlir import ir
import pytest

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir_transforms import partial_tiles

HEAD = """module { func.func @f(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>, %n: index) -> tensor<4x4xf32> {
 %r = scf.forall (%i) in (1) shared_outs(%o = %x) -> tensor<4x4xf32> {
 %g = linalg.generic {indexing_maps=[affine_map<(a,b)->(a,b)>,affine_map<(a,b)->(a,b)>,affine_map<(a,b)->(a,b)>], iterator_types=["parallel","parallel"]} ins(%x, %y: tensor<4x4xf32>, tensor<4x4xf32>) outs(%o: tensor<4x4xf32>) {
 ^bb0(%v: f32, %old: f32, %u: f32):
  %i0 = linalg.index 0 : index
  %i1 = linalg.index 1 : index
  %c = arith.cmpi ult, %i0, %n : index
  %d = arith.cmpi ult, %i1, %n : index
  %z = arith.constant 0.0 : f32
"""
TAIL = """
 } -> tensor<4x4xf32>
 scf.forall.in_parallel { tensor.parallel_insert_slice %g into %o[0,0][4,4][1,1] : tensor<4x4xf32> into tensor<4x4xf32> } }
 return %r : tensor<4x4xf32> } }"""

GOES = {
    "and to a mask",
    "as select's TRUE operand-side: select(c, old, const)",
    "mask only: select(c, v, const)",
}
CASES = {
    "mask only: select(c, v, const)": "%s=arith.select %c, %v, %z : f32\n linalg.yield %s : f32",
    "keep only: select(c, v, old)": "%s=arith.select %c, %v, %old : f32\n linalg.yield %s : f32",
    "both: select(c,v,const), select(c,v,old)": "%s=arith.select %c, %v, %z : f32\n %t=arith.select %c, %s, %old : f32\n linalg.yield %t : f32",
    "and to a mask": "%a=arith.andi %c, %d : i1\n %s=arith.select %a, %v, %z : f32\n linalg.yield %s : f32",
    "and to a keep": "%a=arith.andi %c, %d : i1\n %s=arith.select %a, %v, %old : f32\n linalg.yield %s : f32",
    "and to both": "%a=arith.andi %c, %d : i1\n %s=arith.select %a, %v, %z : f32\n %t=arith.select %a, %s, %old : f32\n linalg.yield %t : f32",
    "as select's TRUE operand-side: select(c, old, const)": "%s=arith.select %c, %old, %z : f32\n linalg.yield %s : f32",
    "derived from old: select(c, v, old+1)": "%one=arith.constant 1.0 : f32\n %p=arith.addf %old, %one : f32\n %s=arith.select %c, %v, %p : f32\n linalg.yield %s : f32",
    "derived from old: select(c, v, neg old)": "%p=arith.negf %old : f32\n %s=arith.select %c, %v, %p : f32\n linalg.yield %s : f32",
    "from another block arg-free op: select(c, v, v*2)": "%two=arith.constant 2.0 : f32\n %p=arith.mulf %v, %two : f32\n %s=arith.select %c, %v, %p : f32\n linalg.yield %s : f32",
    "nested: select(c, v, select(d, old, const))": "%q=arith.select %d, %old, %z : f32\n %s=arith.select %c, %v, %q : f32\n linalg.yield %s : f32",
    "nested: select(c, v, select(d, const, old))": "%q=arith.select %d, %z, %old : f32\n %s=arith.select %c, %v, %q : f32\n linalg.yield %s : f32",
    "condition as a value: uitofp": "%f=arith.uitofp %c : i1 to f32\n %s=arith.mulf %v, %f : f32\n linalg.yield %s : f32",
    "not: xori": "%t=arith.constant true\n %nc=arith.xori %c, %t : i1\n %s=arith.select %nc, %v, %z : f32\n linalg.yield %s : f32",
    "or": "%a=arith.ori %c, %d : i1\n %s=arith.select %a, %v, %z : f32\n linalg.yield %s : f32",
    "through a loop arg": "%lo=arith.constant 0 : index\n %hi=arith.constant 2 : index\n %st=arith.constant 1 : index\n %t=arith.constant true\n %e=scf.for %q=%lo to %hi step %st iter_args(%w=%c) -> (i1) { %m=arith.andi %w, %t : i1\n scf.yield %m : i1 }\n %s=arith.select %e, %v, %old : f32\n linalg.yield %s : f32",
    "through a loop arg to a mask": "%lo=arith.constant 0 : index\n %hi=arith.constant 2 : index\n %st=arith.constant 1 : index\n %t=arith.constant true\n %e=scf.for %q=%lo to %hi step %st iter_args(%w=%c) -> (i1) { scf.yield %w : i1 }\n %s=arith.select %e, %v, %z : f32\n linalg.yield %s : f32",
    "through scf.if result": "%e=scf.if %d -> (i1) { scf.yield %c : i1 } else { %f=arith.constant false\n scf.yield %f : i1 }\n %s=arith.select %e, %v, %old : f32\n linalg.yield %s : f32",
    "select of i1: select(d, c, false) to a keep": "%f=arith.constant false\n %e=arith.select %d, %c, %f : i1\n %s=arith.select %e, %v, %old : f32\n linalg.yield %s : f32",
}


@pytest.mark.parametrize("name", list(CASES))
def test_row_condition(name: str) -> None:
    with ir.Context():
        module = ir.Module.parse(HEAD + " " + CASES[name] + TAIL)
        with ir.Location.unknown():
            partial_tiles._drop_row_masks(module)
        text = str(module)
    # %d, the condition on columns, stays either way.
    assert (text.count("arith.cmpi") == 1) == (name in GOES), text
