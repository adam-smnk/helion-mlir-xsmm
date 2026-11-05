"""``row_maxima``: which ``maxnumf`` reductions may stay. On IR text, with the ops
as vectorization leaves them; kernels are in ``tests/test_xpu_ir.py``."""

from __future__ import annotations

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir_transforms.row_maxima import nan_cannot_be_lost
from helion_mlir_backend._compiler.mlir_transforms.row_maxima import (
    restore_nan_propagating_maxima,
)

_M = "vector<8x4xf32>"
_R = "vector<8xf32>"


def _module(step: str, after: str = "") -> str:
    """A grid loop of one K/V loop: ``%s`` scores, ``%max`` the reduction, ``%shift``
    its broadcast, and ``step``; ``after`` follows the loop."""
    return f"""
func.func @f(%out: tensor<8x4xf32>, %s: {_M}, %init: {_R}) -> tensor<8x4xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant dense<0.0> : {_M}
  %zeros = arith.constant dense<0.0> : {_R}
  %r = scf.forall (%i) in (2) shared_outs(%o = %out) -> (tensor<8x4xf32>) {{
    %last:2 = scf.for %j = %c0 to %c4 step %c1
        iter_args(%running = %init, %acc = %zero) -> ({_R}, {_M}) {{
      %max = vector.multi_reduction <maxnumf>, %s, %init [1] : {_M} to {_R}
      %wider = arith.cmpf ogt, %running, %max : {_R}
      %new = arith.select %wider, %running, %max : vector<8xi1>, {_R}
      %b = vector.broadcast %new : {_R} to vector<4x8xf32>
      %shift = vector.transpose %b, [1, 0] : vector<4x8xf32> to {_M}
      {step}
    }}
    {after}
    %w = vector.transfer_write %last#1, %o[%c0, %c0] {{in_bounds = [true, true]}}
        : {_M}, tensor<8x4xf32>
    scf.forall.in_parallel {{
      tensor.parallel_insert_slice %w into %o[0, 0] [8, 4] [1, 1]
          : tensor<8x4xf32> into tensor<8x4xf32>
    }}
  }} {{helion.kernel = 0 : i64}}
  return %r : tensor<8x4xf32>
}}
"""


_SOFTMAX = f"""
      %d = arith.subf %s, %shift : {_M}
      %p = math.exp %d : {_M}
      %sum = vector.multi_reduction <add>, %p, %zeros [1] : {_M} to {_R}
      %bs = vector.broadcast %sum : {_R} to vector<4x8xf32>
      %ts = vector.transpose %bs, [1, 0] : vector<4x8xf32> to {_M}
      %next = arith.addf %acc, %ts : {_M}
      scf.yield %new, %next : {_R}, {_M}
"""


def _reduction(text: str) -> tuple[ir.Module, ir.Operation]:
    module = ir.Module.parse(text)
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "vector.multi_reduction" and "maxnumf" in str(
            op.attributes["kind"]
        ):
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return module, found[0]


@pytest.fixture(autouse=True)
def _context() -> object:
    with ir.Context() as context:
        context.allow_unregistered_dialects = False
        context.load_all_available_dialects()
        yield


def test_softmax_shift_may_ignore_nan() -> None:
    _, reduction = _reduction(_module(_SOFTMAX))
    assert nan_cannot_be_lost(reduction)


@pytest.mark.parametrize(
    ("name", "step"),
    [
        (
            "another-source",
            _SOFTMAX.replace("arith.subf %s, %shift", "arith.subf %zero, %shift"),
        ),
        (
            # A NaN stays in its element: a NaN maximum makes the whole row NaN.
            "no-reduction",
            f"""
      %d = arith.subf %s, %shift : {_M}
      %p = math.exp %d : {_M}
      %next = arith.addf %acc, %p : {_M}
      scf.yield %new, %next : {_R}, {_M}
""",
        ),
        (
            # A compare can lose the NaN of the exp.
            "exp-compared",
            f"""
      %d = arith.subf %s, %shift : {_M}
      %p = math.exp %d : {_M}
      %c = arith.cmpf ogt, %p, %zero : {_M}
      %q = arith.select %c, %p, %zero : vector<8x4xi1>, {_M}
      %sum = vector.multi_reduction <add>, %q, %zeros [1] : {_M} to {_R}
      %bs = vector.broadcast %sum : {_R} to vector<4x8xf32>
      %ts = vector.transpose %bs, [1, 0] : vector<4x8xf32> to {_M}
      %next = arith.addf %acc, %ts : {_M}
      scf.yield %new, %next : {_R}, {_M}
""",
        ),
        (
            "shift-compared",
            f"""
      %d = arith.subf %s, %shift : {_M}
      %c = arith.cmpf ogt, %d, %zero : {_M}
      %p = math.exp %d : {_M}
      %sum = vector.multi_reduction <add>, %p, %zeros [1] : {_M} to {_R}
      %bs = vector.broadcast %sum : {_R} to vector<4x8xf32>
      %ts = vector.transpose %bs, [1, 0] : vector<4x8xf32> to {_M}
      %next = arith.addf %acc, %ts : {_M}
      scf.yield %new, %next : {_R}, {_M}
""",
        ),
        (
            "maximum-multiplied",
            f"""
      %d = arith.mulf %s, %shift : {_M}
      %next = arith.addf %acc, %d : {_M}
      scf.yield %new, %next : {_R}, {_M}
""",
        ),
    ],
    ids=lambda value: value if "\n" not in str(value) else "",
)
def test_other_uses_keep_propagating_nan(name: str, step: str) -> None:
    _, reduction = _reduction(_module(step))
    assert not nan_cannot_be_lost(reduction), name


def test_stored_loop_result_keeps_propagating_nan() -> None:
    after = f"""
    %bl = vector.broadcast %last#0 : {_R} to vector<4x8xf32>
    %tl = vector.transpose %bl, [1, 0] : vector<4x8xf32> to {_M}
    %w2 = vector.transfer_write %tl, %o[%c0, %c0] {{in_bounds = [true, true]}}
        : {_M}, tensor<8x4xf32>
"""
    _, reduction = _reduction(_module(_SOFTMAX, after))
    assert not nan_cannot_be_lost(reduction)


@pytest.mark.parametrize(
    ("kernels", "kept", "kind"),
    [({0}, 1, "maxnumf"), ({1}, 0, "maximumf"), (set(), 0, "maximumf")],
)
def test_only_named_kernels_keep_them(kernels: set[int], kept: int, kind: str) -> None:
    module, _ = _reduction(_module(_SOFTMAX))
    assert restore_nan_propagating_maxima(module, kernels, "helion.kernel") == kept
    assert f"<{kind}>" in str(module)


def _two_kv_loops(after: str = "") -> str:
    """Two K/V loops of one online softmax: the second starts from the first's running
    maximum and output, and compares its own maximum with the running one."""

    def loop(name: str, source: str, inits: tuple[str, str]) -> str:
        return f"""
    %{name}:2 = scf.for %j{name} = %c0 to %c4 step %c1
        iter_args(%running{name} = {inits[0]}, %acc{name} = {inits[1]}) -> ({_R}, {_M}) {{
      %max{name} = vector.multi_reduction <maxnumf>, {source}, %init [1] : {_M} to {_R}
      %wider{name} = arith.cmpf ogt, %running{name}, %max{name} : {_R}
      %new{name} = arith.select %wider{name}, %running{name}, %max{name}
          : vector<8xi1>, {_R}
      %b{name} = vector.broadcast %new{name} : {_R} to vector<4x8xf32>
      %shift{name} = vector.transpose %b{name}, [1, 0] : vector<4x8xf32> to {_M}
      %d{name} = arith.subf {source}, %shift{name} : {_M}
      %p{name} = math.exp %d{name} : {_M}
      %sum{name} = vector.multi_reduction <add>, %p{name}, %zeros [1] : {_M} to {_R}
      %bs{name} = vector.broadcast %sum{name} : {_R} to vector<4x8xf32>
      %ts{name} = vector.transpose %bs{name}, [1, 0] : vector<4x8xf32> to {_M}
      %next{name} = arith.addf %acc{name}, %ts{name} : {_M}
      scf.yield %new{name}, %next{name} : {_R}, {_M}
    }}"""

    return f"""
func.func @f(%out: tensor<8x4xf32>, %s: {_M}, %t: {_M}, %init: {_R}) -> tensor<8x4xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant dense<0.0> : {_M}
  %zeros = arith.constant dense<0.0> : {_R}
  %r = scf.forall (%i) in (2) shared_outs(%o = %out) -> (tensor<8x4xf32>) {{
    {loop("a", "%s", ("%init", "%zero"))}
    {loop("c", "%t", ("%a#0", "%a#1"))}
    {after}
    %w = vector.transfer_write %c#1, %o[%c0, %c0] {{in_bounds = [true, true]}}
        : {_M}, tensor<8x4xf32>
    scf.forall.in_parallel {{
      tensor.parallel_insert_slice %w into %o[0, 0] [8, 4] [1, 1]
          : tensor<8x4xf32> into tensor<8x4xf32>
    }}
  }} {{helion.kernel = 0 : i64}}
  return %r : tensor<8x4xf32>
}}
"""


def _max_reductions_of(text: str) -> list[bool]:
    with ir.Context() as context:
        context.allow_unregistered_dialects = False
        module = ir.Module.parse(text)
        found: list[bool] = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "vector.multi_reduction" and "add" not in str(
                op.attributes["kind"]
            ):
                found.append(nan_cannot_be_lost(op))
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        return found


def test_maxima_of_later_kv_loops_are_one_chain() -> None:
    """Both loops' maxima may ignore NaN; a NaN in either loop's scores reaches its
    ``exp``, and the output through the carried accumulator."""
    assert _max_reductions_of(_two_kv_loops()) == [True, True]


def test_stored_maximum_of_two_loops_keeps_propagating_nan() -> None:
    """Negative: the running maximum's other use, after both loops."""
    after = f"""
    %use = vector.transfer_write %c#0, %o[%c0, %c0] {{in_bounds = [true]}}
        : {_R}, tensor<8x4xf32>"""
    assert _max_reductions_of(_two_kv_loops(after)) == [False, False]
