"""``bf16_extensions_as_shifts``: in XeVM GPU modules, the ``llvm.fpext``s
from bf16 to f32 that an f32-to-bf16 ``llvm.fptrunc`` is computed from become
the same extension in integer ops: ``bitcast(zext(bitcast x : i16) << 16)``.

Works around wrong rounding by IGC (no LLVM or IGC patch, by policy). The
LLVM and SPIR-V we emit are right: e.g. ``bf16(f32(a) + f32(b))`` is an
``OpFAdd`` of f32 values and an ``OpFConvert`` to bf16 without a rounding-mode
decoration (round to nearest even). IGC's optimizer shrinks
``fptrunc(fadd(fpext a, fpext b))`` to ``fadd bfloat`` (InstCombine: exact for
add, sub and mul if the narrow op rounds to nearest even) and emits it as a
mixed-mode ``add (16) r:bf ...`` on Xe2 (Arc B50), which truncates: 27 % of bf16
adds and 47 % of bf16 muls are one ulp off round-to-nearest-even, none off
truncation (bf16 arithmetic of Helion kernels too: the pipeline computes it in
f32 between the same two conversions). A plain f32-to-bf16 conversion
(``mov :bf``) rounds right, so does the f32 arithmetic once its operands are
not ``fpext``s: the shrinking needs them (or constants exact in bf16).
Reproducer (helion-mlir-dev): ``agent-wiki/upstream-repro/igc-bf16-rounding/``.

Rejected alternatives, tried on the device: an explicit round-to-nearest-even
in integer ops instead of the ``fptrunc`` is right on its own, but its NaN test
(compare and select) is miscompiled by IGC next to other predicated code
(ELU, Softplus: wrong lanes); an ``FPRoundingMode RTE`` decoration makes IGC
call a ``__spirv_FConvert_RDF16b_rte`` it does not define;
``SPV_INTEL_bfloat16_conversion`` is not enabled by the XeVM target.

The rewrite is exact (a bf16 value's bits are the upper half of its f32 value's
bits, also for infinities, NaNs and subnormals). It is not free in code size:
IGC emits a zero-extending ``mov`` and a shift per conversion (the bf16 add
kernel at ``[1, 8192]`` tiles: 401 instructions against 145). Measured on the
B50 (review of the change): level 1 elementwise kernels are within noise
(memory bound), level 2 40_Matmul_Scaling_ResidualAdd at 4096x4096x256
(epilogue-heavy) about 2.3 % slower. A variant without the ``zext`` (two bf16
values bitcast as one i32, then ``shl 16`` and ``and 0xFFFF0000``) is a
possible later improvement.

Matched exactly: ``llvm.fpext`` of a bf16 scalar or vector to f32 in a
``gpu.module`` with an ``#xevm.target``, from which an ``llvm.fptrunc`` of f32
to bf16 is computed through ops with float results and no regions and through
block arguments (phis: IGC folds a phi of ``fpext``s into an ``fpext`` of a phi,
then shrinks), in any number of steps. That covers what IGC was seen to shrink
(the op feeding the ``fptrunc``, also behind a phi) without being a proof of
coverage for all of IGC's passes; extensions that reach no bf16 truncation are
not rewritten, so the IR behind ``xpu_checks.check_math``'s device-verified
tables stays as it was.
``fptrunc(fpext x)`` (exact, ``x``) does not count; other extensions (e.g. to
f32 outputs, or of f16) and host code are left alone.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules

_FPTRUNC = "llvm.fptrunc"
_FPEXT = "llvm.fpext"


def _element(type_: ir.Type) -> ir.Type:
    if isinstance(type_, ir.VectorType):
        return ir.VectorType(type_).element_type
    return type_


def _like(type_: ir.Type, element: ir.Type) -> ir.Type:
    """``type_`` (a scalar or vector type) with element type ``element``."""
    if isinstance(type_, ir.VectorType):
        return ir.VectorType.get(ir.VectorType(type_).shape, element)
    return element


def _is_float(type_: ir.Type) -> bool:
    return isinstance(
        _element(type_), (ir.F16Type, ir.BF16Type, ir.F32Type, ir.F64Type)
    )


def is_bf16_extension(op: ir.Operation) -> bool:
    """``op`` is an ``llvm.fpext`` of bf16 (scalar or vector) to f32."""
    return (
        op.name == _FPEXT
        and isinstance(_element(op.operands[0].type), ir.BF16Type)
        and isinstance(_element(op.results[0].type), ir.F32Type)
    )


def is_bf16_truncation(op: ir.Operation) -> bool:
    """``op`` is an ``llvm.fptrunc`` of f32 (scalar or vector) to bf16."""
    return (
        op.name == _FPTRUNC
        and isinstance(_element(op.operands[0].type), ir.F32Type)
        and isinstance(_element(op.results[0].type), ir.BF16Type)
    )


def _successor_operands(terminator: ir.Operation, index: int) -> list[ir.Value] | None:
    """The operands ``terminator`` passes to its successor ``index``, or
    ``None`` if its operand layout is not known."""
    operands = list(terminator.operands)
    if terminator.name == "llvm.br":
        return operands
    if terminator.name == "llvm.cond_br":
        segments = list(
            ir.DenseI32ArrayAttr(terminator.attributes["operandSegmentSizes"])
        )
        _condition, on_true, _on_false = segments
        return operands[1 : 1 + on_true] if index == 0 else operands[1 + on_true :]
    return None


def incoming_values(argument: ir.Value) -> list[ir.Value]:
    """The values a block argument takes from its predecessors (a phi's
    incoming values): the matching successor operand of each ``llvm.br`` and
    ``llvm.cond_br``; of another terminator, every operand of the argument's
    type (conservative). None for an entry block's (function) arguments."""
    block = argument.owner
    position = ir.BlockArgument(argument).arg_number
    values: list[ir.Value] = []
    for predecessor in block.predecessors:
        terminator = list(predecessor.operations)[-1]
        for index, successor in enumerate(terminator.successors):
            if successor != block:
                continue
            passed = _successor_operands(terminator, index)
            if passed is None:
                values.extend(o for o in terminator.operands if o.type == argument.type)
            else:
                values.append(passed[position])
    return values


def extensions_of(truncation: ir.Operation, seen: set[ir.Value]) -> list[ir.Operation]:
    """The bf16 extensions ``truncation`` (an f32-to-bf16 ``fptrunc``) is
    computed from, through ops with float results and no regions and through
    block arguments (to the values their predecessors pass); none for
    ``fptrunc(fpext x)``. Values in ``seen`` were walked already (by an earlier
    truncation) and are skipped. Iterative: unrolled reductions make long
    chains."""
    source = truncation.operands[0]
    owner = source.owner
    if not isinstance(owner, ir.Block) and is_bf16_extension(owner):
        return []
    found: list[ir.Operation] = []
    stack = [source]
    while stack:
        value = stack.pop()
        if value in seen:
            continue
        seen.add(value)
        if not _is_float(value.type):
            continue
        owner = value.owner
        if isinstance(owner, ir.Block):
            stack.extend(incoming_values(value))
            continue
        if owner.regions:
            continue
        if is_bf16_extension(owner):
            found.append(owner)
            continue
        stack.extend(owner.operands)
    return found


def _constant(type_: ir.Type, value: int) -> ir.Value:
    element = _element(type_)
    if isinstance(type_, ir.VectorType):
        attr = ir.DenseElementsAttr.get_splat(type_, ir.IntegerAttr.get(element, value))
    else:
        attr = ir.IntegerAttr.get(element, value)
    return ir.Operation.create(
        "llvm.mlir.constant", results=[type_], attributes={"value": attr}
    ).results[0]


def _op(name: str, result: ir.Type, *operands: ir.Value) -> ir.Value:
    return ir.Operation.create(name, results=[result], operands=list(operands)).results[
        0
    ]


def _rewrite(extension: ir.Operation) -> None:
    source = extension.operands[0]
    i16 = _like(source.type, ir.IntegerType.get_signless(16))
    i32 = _like(source.type, ir.IntegerType.get_signless(32))
    with ir.InsertionPoint(extension), extension.location:
        bits = _op("llvm.zext", i32, _op("llvm.bitcast", i16, source))
        shifted = _op("llvm.shl", i32, bits, _constant(i32, 16))
        result = _op("llvm.bitcast", extension.results[0].type, shifted)
    extension.results[0].replace_all_uses_with(result)
    extension.erase()


def bf16_extensions_as_shifts(module: ir.Module) -> int:
    """Rewrites ``module``'s matching XeVM extensions in place; returns how
    many."""
    truncations: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if is_bf16_truncation(op):
            truncations.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    seen: set[ir.Value] = set()
    extensions: dict[ir.Value, ir.Operation] = {}  # by result: one rewrite each
    for truncation in truncations:
        for extension in extensions_of(truncation, seen):
            extensions.setdefault(extension.results[0], extension)
    for extension in extensions.values():
        _rewrite(extension)
    return len(extensions)
