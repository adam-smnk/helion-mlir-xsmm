"""``nsz_reductions``: the maximum and minimum reductions of a module's kernels get the
``nsz`` flag: a cheap row maximum that still propagates NaN.

The LLVM SPIR-V backend expands ``llvm.maximum`` to seven instructions (OpenCL ``fmax``; two
``copysign``, a compare and a select to order -0 below +0; an ``OpUnordered`` and a select
for NaN). With ``nsz`` the signed-zero part goes and three stay, which made 97's flash
attention much faster on the Arc B50; it now also ignores NaN in its maxima
(``xegpu_nan_ignoring_maxima``).

The flag gives up only the order of -0 and +0, and only these reductions have it (``amax``,
``amin``, ``max``, ``min``, softmax's row maximum). ``torch.maximum`` and ``torch.minimum``
are compare-and-select pairs (``nan_minmax``) and keep their result. A reduction over rows of
mixed zeros can differ in the sign of its zero result from torch on the CPU and from eager
torch on the XPU: 1 row of 256 in the review's comparison of ``amax``, ``amin``, ``max`` and
``min`` with torch (the row of alternating -0 and +0);
``tests/test_xpu_runtime.py``'s ``test_reductions_of_mixed_zeros_give_a_zero`` holds what is
kept. NaN still propagates.

Set on the reduction, where the kernel still says what it means: a choice of the math, not
a workaround. XeGPU's lowering drops the flag for now, so :mod:`xevm_nsz_minmax` puts it back
after LLVM lowering. Only reductions without flags, and only those in a ``gpu.module``, are
changed.
"""

from __future__ import annotations

from mlir import ir

_KINDS = ("#vector.kind<maximumf>", "#vector.kind<minimumf>")
_NONE = "#arith.fastmath<none>"


def nsz_reductions(module: ir.Module) -> int:
    """Flag the maximum and minimum reductions of ``module``'s kernels; how many."""
    nsz = ir.Attribute.parse("#arith.fastmath<nsz>", module.context)
    count = 0

    def visit(op: ir.Operation) -> ir.WalkResult:
        nonlocal count
        if (
            op.name == "vector.multi_reduction"
            and str(op.attributes["kind"]) in _KINDS
            and (
                "fastmath" not in op.attributes
                or str(op.attributes["fastmath"]) == _NONE
            )
        ):
            op.attributes["fastmath"] = nsz
            count += 1
        return ir.WalkResult.ADVANCE

    for top in module.body.operations:
        if top.operation.name == "gpu.module":
            top.operation.walk(visit)
    return count
