"""``compare_lanes``: in XeVM GPU modules, a vector ``llvm.fcmp`` or ``llvm.icmp`` becomes one
scalar compare per lane and a vector of their results.

The result is the same, by definition of a vector compare. It works around a miscompile in IGC
(no LLVM or IGC patch, by policy). IGC scalarizes early, but its InstCombine runs after that
and still folds vector compares of unscalarized operands:

- ``(x > c) | (x < -c)``, its chains of selects and its negations, to a vector ``llvm.fabs``;
- ``select(x < 0, 0 - x, x)`` on integers to a vector ``llvm.abs``.

Codegen then writes one row of the vector, ``mov (M1, 16) dst(0,0) (abs)src(0,0)``, and the
other rows are read unwritten: silent wrong elements, in lanes other than the first. On the B50:
``((t > 1) | (t < -1)).to(f32)`` 9830 of 32768, ``where(t < 0, 0 - t, t)`` on int32 30471 of
32768. With scalar compares the fold makes a scalar ``fabs``, one per lane, which is right.

Not covered: ``and(bitcast x, 0x7FFFFFFF)``, which InstCombine also makes a vector ``fabs``. No
kernel reaches it: ``torch.abs`` is :mod:`.xevm_fabs`'s ``umin``.

Bug and removal: a named workaround, reproduced in helion-mlir-dev's
``agent-wiki/upstream-repro/igc-vector-fabs/``. This stage goes when, on the IGC builds the XPU
target supports, that directory's ``idiom`` and ``iabs`` kernels compile to 16 ``(abs)`` movs
with ``visa_uninit.py`` clean, and ``tests/test_xpu_scalar_compares.py`` is right on the device
without the stage. ``tests/igc_dumps.py``'s check finds this class in the dumps of the kernels
``tests/test_xpu_igc_workarounds.py`` compiles, and only there: nothing checks a user's kernel.

Matched exactly: those two ops on 1D vectors in a ``gpu.module`` with an ``#xevm.target``.
Scalars and host code stay.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules
from .xevm_bf16_extensions import llvm_op

_COMPARES = frozenset({"llvm.fcmp", "llvm.icmp"})


def _per_lane(op: ir.Operation) -> None:
    lhs, rhs = op.operands[0], op.operands[1]
    lanes = ir.VectorType(lhs.type).shape[0]
    element = ir.VectorType(lhs.type).element_type
    result_type = op.results[0].type
    i1 = ir.IntegerType.get_signless(1)
    i32 = ir.IntegerType.get_signless(32)
    attributes = {name: op.attributes[name] for name in op.attributes}
    with ir.InsertionPoint(op), op.location:
        vector = ir.Operation.create("llvm.mlir.poison", results=[result_type]).results[
            0
        ]
        for lane in range(lanes):
            position = ir.Operation.create(
                "llvm.mlir.constant",
                results=[i32],
                attributes={"value": ir.IntegerAttr.get(i32, lane)},
            ).results[0]
            compared = ir.Operation.create(
                op.name,
                results=[i1],
                operands=[
                    llvm_op("llvm.extractelement", element, lhs, position),
                    llvm_op("llvm.extractelement", element, rhs, position),
                ],
                attributes=attributes,
            ).results[0]
            vector = llvm_op(
                "llvm.insertelement", result_type, vector, compared, position
            )
    op.results[0].replace_all_uses_with(vector)
    op.erase()


def compare_lanes(module: ir.Module) -> int:
    """Rewrite ``module``'s vector compares in XeVM modules in place; how many."""
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _COMPARES and isinstance(op.operands[0].type, ir.VectorType):
            found.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit, ir.WalkOrder.PRE_ORDER)
    for op in found:
        _per_lane(op)
    return len(found)
