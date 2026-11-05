"""``select_bool_casts``: in GPU modules, ``arith.uitofp`` and ``arith.sitofp`` of a vector
of ``i1`` become an ``arith.select`` of two float constants.

The result is the same, by definition of the casts: ``true`` is 1.0 unsigned and -1.0
signed, ``false`` is 0.0. It works around an abort in LLVM's SPIR-V backend (no LLVM patch,
by policy): ``selectIToF`` selects the bool to an integer vector first, and for 8 or 16 lanes
llc stops with ``VReg's low-level type and register class have different sizes``. 4 lanes,
a scalar, ``zext`` and a float ``select`` all compile. It takes no more than a ``uitofp``
in the module, with no MLIR or GPU involved.

Bug and removal: reported in helion-mlir-dev's
``agent-wiki/helion-mlir-backend/upstream-issues-xpu.md``, with the reproducer
``agent-wiki/upstream-repro/spirv-bool-to-float/``. This stage can go when ``llc`` compiles
the reproducer on the LLVM the XPU target uses.

Matched exactly: those casts with a vector of ``i1`` as source. Other casts, scalars and host
code are left alone.
"""

from __future__ import annotations

from mlir import ir

_ONE = {"arith.uitofp": 1.0, "arith.sitofp": -1.0}


def _select_cast(op: ir.Operation) -> None:
    result_type = op.results[0].type
    element = ir.VectorType(result_type).element_type
    with ir.InsertionPoint(op):
        true, false = (
            ir.Operation.create(
                "arith.constant",
                results=[result_type],
                attributes={
                    "value": ir.DenseElementsAttr.get_splat(
                        result_type, ir.FloatAttr.get(element, value)
                    )
                },
            ).results[0]
            for value in (_ONE[op.name], 0.0)
        )
        select = ir.Operation.create(
            "arith.select",
            results=[result_type],
            operands=[op.operands[0], true, false],
        )
    op.results[0].replace_all_uses_with(select.results[0])
    op.erase()


def select_bool_casts(module: ir.Module) -> int:
    """Rewrite the casts in the ``gpu.module`` ops of ``module``; how many."""
    casts: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _ONE and isinstance(op.operands[0].type, ir.VectorType):
            element = ir.VectorType(op.operands[0].type).element_type
            if isinstance(element, ir.IntegerType) and element.width == 1:
                casts.append(op)
        return ir.WalkResult.ADVANCE

    for top in module.body.operations:
        if top.operation.name == "gpu.module":
            top.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    for op in casts:
        _select_cast(op)
    return len(casts)
