"""``fabs_as_integer_min``: f32 ``llvm.intr.fabs`` in XeVM GPU modules becomes
``bitcast(umin(bits, bits ^ 0x80000000))``, which clears the sign bit in
integer ops.

Works around an IGC miscompile (no LLVM or IGC patch, by policy). IGC's
InstCombine narrows ``fptrunc(fabs x)`` and ``fabs(fpext x)`` to
``llvm.fabs.bf16``, and its codegen emits that as ``mov (abs):uw``: an ``(abs)``
source modifier on an unsigned type does nothing, so the operand comes out
unchanged. Vector forms (``llvm.fabs.v4f16``) were wrong as well (Arc B50, IGC
2.41). Reproducer (helion-mlir-dev): ``agent-wiki/upstream-repro/igc-bf16-rounding/``.

The obvious ``and bits, 0x7FFFFFFF`` does not survive: InstCombine turns a
one-use ``and`` of a bitcast back into ``fabs`` and narrows it, unless the bits
come straight from a load. The minimum is the same bits: of ``bits`` and
``bits ^ signbit``, the smaller unsigned value is the one with the sign bit
clear, for every input (zeros, infinities, NaNs and subnormals). In IGC's
optimized IR it stays a ``umin`` on computed values, and becomes that ``and`` on
loaded bits (38_L1Norm_, ``abs`` of an f32 tile), where no ``fabs`` comes back; no
``llvm.fabs`` was left in the kernels checked (``abs``, ``abs(-x)``, ``abs(x) * 2``,
softsign, 38).

f64 too, as IGC emits ``llvm.fabs.v16f64`` no better (see :mod:`.xevm_lane_compares`). A module
narrowing f64 also compiles with ``IGCVectorizer`` off: see ``_narrows_f64`` in ``targets.xpu``.

Matched exactly: ``llvm.intr.fabs`` of an f32 or f64 scalar or vector in a
``gpu.module`` with an ``#xevm.target``. Other types and host code are left
alone. Runs after ``bf16_extensions_as_shifts``.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules
from .xevm_bf16_extensions import constant
from .xevm_bf16_extensions import element
from .xevm_bf16_extensions import like
from .xevm_bf16_extensions import llvm_op

_FABS = "llvm.intr.fabs"
_SIGN_BITS = {32: -(2**31), 64: -(2**63)}


def _rewrite(fabs: ir.Operation) -> None:
    value = fabs.operands[0]
    width = element(value.type).width
    integer = like(value.type, ir.IntegerType.get_signless(width))
    with ir.InsertionPoint(fabs), fabs.location:
        bits = llvm_op("llvm.bitcast", integer, value)
        flipped = llvm_op(
            "llvm.xor", integer, bits, constant(integer, _SIGN_BITS[width])
        )
        cleared = llvm_op("llvm.intr.umin", integer, bits, flipped)
        result = llvm_op("llvm.bitcast", fabs.results[0].type, cleared)
    fabs.results[0].replace_all_uses_with(result)
    fabs.erase()


def fabs_as_integer_min(module: ir.Module) -> int:
    """Rewrites ``module``'s matching XeVM ``fabs`` ops in place; returns how
    many."""
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == _FABS and isinstance(
            element(op.results[0].type), ir.F32Type | ir.F64Type
        ):
            found.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    for op in found:
        _rewrite(op)
    return len(found)
