"""``nsz_minmax``: ``llvm.intr.maximum``/``minimum`` of XeVM GPU modules get
the ``nsz`` flag: a cheap row maximum that still propagates NaN.

The LLVM SPIR-V backend expands ``llvm.maximum`` to seven instructions (OpenCL
``fmax``; two ``copysign``, a compare and a select to order -0 below +0; an
``OpUnordered`` and a select for NaN). With ``nsz`` the signed-zero part goes
and three stay. On the Arc B50, 97's flash attention (mlir-gpu-bench, bench
device time) takes 4.4-4.5 ms instead of 5.98 ms (3.82 ms with NaN-ignoring
``fmax``); end to end, about 4.8 ms against 6.4 ms and 3.9 ms.

The flag gives up only the order of -0 and +0, and only ``llvm.intr.maximum`` and
``minimum`` have it: the maxima and minima of reductions (``amax``, ``amin``,
``max``, ``min``, softmax's row maximum). ``torch.maximum`` and ``torch.minimum``
are compare-and-select pairs (``nan_minmax``) and keep their result. A reduction
over rows of mixed zeros can differ in the sign of its zero result from torch on
the CPU and from eager torch on the XPU: 1 row of 256 in the review's comparison
of ``amax``, ``amin``, ``max`` and ``min`` with torch (the row of alternating -0
and +0); ``tests/test_xpu_runtime.py``'s ``test_reductions_of_mixed_zeros_give_a_zero``
holds what is kept. NaN still propagates. Only ops without
fast-math flags and only ``gpu.module``s with an ``#xevm.target`` are changed:
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
