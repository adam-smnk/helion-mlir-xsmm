"""``convert_lanes_where_computed``: in XeVM GPU modules, a narrowing vector
conversion (f32 to bf16 or f16, f64 to f32, bf16 or f16) of a vector built lane by
lane becomes one scalar conversion per lane, right after the lane's value, and a
vector of the converted lanes.

The result is the same, by definition of ``fptrunc``. It works around a
miscompile in ``IGCVectorizer`` (no LLVM or IGC patch, by policy), which emits a
vectorized cast before the vector it casts (use before def), so the kernel
converts whatever those registers hold. ``handleCastInstruction`` puts the cast
before the first scalar cast, and when the operands are not vectorized,
``createVector`` builds their vector after that first cast if the operands are
in different basic blocks (``getInsertPointForVector``, ``!SameBB && OneUse``).
An OpenCL builtin with branches (``sin``, ``log``, ...) inlines to a result in a
block of its own for every lane. With the conversions next to their lanes, they
are in different blocks, and the vectorizer's check that a slice shares one
block rejects them.

Bug and removal: an LLVM-level workaround for the bug reported as "``fptrunc`` before
its operand" in helion-mlir-dev's
``agent-wiki/upstream-repro/igc-vector-minmax/BUG-REPORT.md``. The reproducer is
``agent-wiki/upstream-repro/igc-vector-minmax/fptrunc/``, with
``IGC_VectorizerAllowFPTRUNC=0`` as its control (IGC 2.41, B50). This stage can go when
the reproducer's ``_optimized.ll`` has no use before its def and its output is right
without the control, on the IGC builds the XPU target supports, and
``tests/test_xpu_igc_workarounds.py`` passes without the stage.

A conversion of an f64 vector needs the same treatment: ``fptrunc <8 x double>`` to
f32 has the same wrong order. Its converted lanes then go through a second round
when they are narrowed further, so the rewrite runs until no conversion matches.

Lanes that are all calls of ``native_exp`` (:data:`_KEEPS_VECTOR_CONVERSION`) stay
as they are: IGC inlines them to one instruction per lane (no branches), and the
vectorizer's insert points are right in one block. Converted next to its lane,
such a slice spreads past the vectorizer's scheduling window and stays scalar,
which costs attention (``native_exp`` to bf16) what ``VectorizerAllowFPTRUNC=0``
does. This holds for that builtin, the one measured; other ``native_`` builtins
convert lane by lane.

Matched exactly: ``llvm.fptrunc`` of an f32 or f64 vector to bf16, f16 or f32 in a
``gpu.module`` with an ``#xevm.target``, every lane of whose operand comes from a
chain of ``llvm.insertelement`` at constant positions, through
``llvm.shufflevector`` (2D tiles convert slices of a wider vector). Any other
source (a load, arithmetic, an unwritten lane, a computed position), lanes that
are all ``native_exp`` calls, other types, scalar conversions and host code stay.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules
from .xevm_bf16_extensions import element
from .xevm_bf16_extensions import llvm_op

_FPTRUNC = "llvm.fptrunc"
_CALL = "llvm.call"
_INSERT = "llvm.insertelement"
_SHUFFLE = "llvm.shufflevector"
_CONSTANTS = ("llvm.mlir.constant", "arith.constant")
_KEEPS_VECTOR_CONVERSION = "_Z22__spirv_ocl_native_expf"
"""The f32 ``native_exp`` builtin: see the module's docstring."""


def _is_narrowing_conversion(op: ir.Operation) -> bool:
    if op.name != _FPTRUNC or not isinstance(op.operands[0].type, ir.VectorType):
        return False
    vector = ir.VectorType(op.operands[0].type)
    return (
        vector.rank == 1
        and not vector.scalable
        and isinstance(vector.element_type, (ir.F32Type, ir.F64Type))
        and isinstance(
            element(op.results[0].type), (ir.BF16Type, ir.F16Type, ir.F32Type)
        )
    )


def _defining_op(value: ir.Value) -> ir.Operation | None:
    if isinstance(value.owner, ir.Block):
        return None
    return value.owner.operation


def _constant(value: ir.Value) -> int | None:
    defining = _defining_op(value)
    if defining is None or defining.name not in _CONSTANTS:
        return None
    return ir.IntegerAttr(defining.attributes["value"]).value


def _lane(vector: ir.Value, index: int) -> ir.Value | None:
    """The value of lane ``index`` of ``vector``; None if it is not written by a
    chain of inserts, or a computed position may write it."""
    current = _defining_op(vector)
    while current is not None:
        if current.name == _INSERT:
            position = _constant(current.operands[2])
            if position is None:
                return None
            if position == index:
                return current.operands[1]
            vector = current.operands[0]
        elif current.name == _SHUFFLE:
            mask = current.attributes["mask"]
            index = ir.DenseI32ArrayAttr(mask)[index]
            first = current.operands[0]
            width = ir.VectorType(first.type).shape[0]
            if index < 0:
                return None
            vector = first if index < width else current.operands[1]
            index = index if index < width else index - width
        else:
            return None
        current = _defining_op(vector)
    return None


def _lanes(vector: ir.Value) -> list[ir.Value] | None:
    lanes = []
    for index in range(ir.VectorType(vector.type).shape[0]):
        lane = _lane(vector, index)
        if lane is None:
            return None
        lanes.append(lane)
    return lanes


def _erase_if_dead(value: ir.Value) -> None:
    """Erases the inserts and shuffles that only built ``value``."""
    seen: set[ir.Operation] = set()
    postorder: list[ir.Operation] = []

    def visit(operand: ir.Value) -> None:
        op = _defining_op(operand)
        if op is None or op.name not in (_INSERT, _SHUFFLE) or op in seen:
            return
        seen.add(op)
        for inner in op.operands:
            if isinstance(inner.type, ir.VectorType):
                visit(inner)
        postorder.append(op)

    visit(value)
    for op in reversed(postorder):
        if not list(op.results[0].uses):
            op.erase()


def _keeps_vector_conversion(value: ir.Value) -> bool:
    """``value`` is the result of a call of :data:`_KEEPS_VECTOR_CONVERSION`."""
    call = _defining_op(value)
    return (
        call is not None
        and call.name == _CALL
        and "callee" in call.attributes
        and ir.FlatSymbolRefAttr(call.attributes["callee"]).value
        == _KEEPS_VECTOR_CONVERSION
    )


def _convert_once(
    value: ir.Value,
    conversion: ir.Operation,
    converted: dict[tuple[ir.Value, str], ir.Value],
) -> ir.Value:
    """``value`` converted, right after its definition (first in its block if it
    is an argument); a lane in several vectors converts once."""
    target = element(conversion.results[0].type)
    key = (value, str(target))
    if key not in converted:
        op = ir.Operation.create(
            _FPTRUNC,
            results=[target],
            operands=[value],
            attributes=dict(conversion.attributes),
        )
        defining = _defining_op(value)
        if defining is not None:
            op.move_after(defining)
        else:
            op.move_before(value.owner.operations[0])
        converted[key] = op.results[0]
    return converted[key]


def _rewrite(
    conversion: ir.Operation,
    lanes: list[ir.Value],
    converted: dict[tuple[ir.Value, str], ir.Value],
) -> None:
    source = conversion.operands[0]
    result_type = conversion.results[0].type
    i64 = ir.IntegerType.get_signless(64)
    with ir.InsertionPoint(conversion), conversion.location:
        vector = ir.Operation.create("llvm.mlir.poison", results=[result_type]).results[
            0
        ]
        for index, value in enumerate(lanes):
            scalar = _convert_once(value, conversion, converted)
            position = ir.Operation.create(
                "llvm.mlir.constant",
                results=[i64],
                attributes={"value": ir.IntegerAttr.get(i64, index)},
            ).results[0]
            vector = llvm_op(_INSERT, result_type, vector, scalar, position)
    conversion.results[0].replace_all_uses_with(vector)
    conversion.erase()
    _erase_if_dead(source)


def _convert_round(module: ir.Module) -> int:
    found: list[tuple[ir.Operation, list[ir.Value]]] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if _is_narrowing_conversion(op):
            lanes = _lanes(op.operands[0])
            if lanes is not None and not all(map(_keeps_vector_conversion, lanes)):
                found.append((op, lanes))
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    converted: dict[tuple[ir.Value, str], ir.Value] = {}
    for conversion, lanes in found:
        _rewrite(conversion, lanes, converted)
    return len(found)


def convert_lanes_where_computed(module: ir.Module) -> int:
    """Rewrites ``module``'s matching XeVM conversions in place, until none matches
    (f64 to f32 to bf16 takes two rounds; each one removes a vector conversion and
    adds none); returns how many."""
    total = 0
    while count := _convert_round(module):
        total += count
    return total
