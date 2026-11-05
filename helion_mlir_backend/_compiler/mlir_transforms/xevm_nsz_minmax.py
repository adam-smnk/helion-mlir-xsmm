"""``nsz_minmax``: ``llvm.intr.maximum``/``minimum`` of XeVM GPU modules get the ``nsz`` flag
that :mod:`nsz_reductions` set on the reductions they come from. The cost and what the flag
gives up are in that module.

Works around XeGPU's lowering dropping a reduction's fastmath flags (no patch, by policy):
``WgToSgMultiDimReductionOp`` and the lane distribution create their reductions without
``getFastmath()``, where ``vector``'s own lowering keeps it. Reproduced in helion-mlir-dev's
``agent-wiki/upstream-repro/xegpu-reduction-fastmath/``.

Removal: when the flag reaches ``llvm.intr.maximum`` without this stage, which
``tests/test_xevm_nsz_minmax.py``'s ``test_xegpu_still_drops_the_reductions_flag`` reports by
failing. Delete this stage and that test.

Only ops without fast-math flags and only ``gpu.module``s with an ``#xevm.target`` are changed:
host code is kept.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules

_OPS = ("llvm.intr.maximum", "llvm.intr.minimum")
_NONE = "#llvm.fastmath<none>"


def nsz_minmax(module: ir.Module) -> int:
    """Flag the minima and maxima of ``module``'s XeVM GPU modules; how many."""
    nsz = ir.Attribute.parse("#llvm.fastmath<nsz>", module.context)
    count = 0

    def visit(op: ir.Operation) -> ir.WalkResult:
        nonlocal count
        if op.name in _OPS and str(op.attributes["fastmathFlags"]) == _NONE:
            op.attributes["fastmathFlags"] = nsz
            count += 1
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    return count
