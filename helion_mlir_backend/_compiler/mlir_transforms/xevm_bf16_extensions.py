"""``bf16_extensions_as_shifts``: in XeVM GPU modules, the ``llvm.fpext``s
from bf16 to f32 become the same extension in integer ops:
``bitcast(zext(bitcast x : i16) << 16)``.

Works around IGC narrowing f32 math to bf16 (no LLVM or IGC patch, by policy),
which rounds wrong. The LLVM and SPIR-V we emit are right: e.g. ``bf16(f32(a) + f32(b))`` is an
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
B50 for the first, narrower form of this rewrite (only extensions a bf16
truncation is computed from): level 1 elementwise within noise, level 2
40_Matmul_Scaling_ResidualAdd at 4096x4096x256 (epilogue-heavy) about 2.3 % slower;
a later review of the current form saw about 1 % on 40. A variant without the
``zext`` (two bf16 values bitcast as one i32, then ``shl 16`` and ``and
0xFFFF0000``) is a possible later improvement.

The same narrowing breaks functions on extended values: OpenCL's ``atan`` takes
an ``fabs`` of its argument, and after inlining ``fabs(fpext x)`` becomes
``fpext(fabs x)`` on bf16, a no-op (:mod:`xevm_fabs`; seen in IGC's dump). ``atan``
of a bf16 value was wrong below -1 and ``cosh`` wrong, also with f32 results;
both are right on the shifted extensions.

Matched exactly: ``llvm.fpext`` of a bf16 scalar or vector to f32 in a
``gpu.module`` with an ``#xevm.target``. An extension only rounded straight back
to bf16 (``fptrunc(fpext x)``, exact: ``x``) stays, so do other extensions
(e.g. of f16) and host code.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules

_FPTRUNC = "llvm.fptrunc"
_FPEXT = "llvm.fpext"


def element(type_: ir.Type) -> ir.Type:
    if isinstance(type_, ir.VectorType):
        return ir.VectorType(type_).element_type
    return type_


def like(type_: ir.Type, element_type: ir.Type) -> ir.Type:
    """``type_`` (a scalar or vector type) with element type ``element_type``."""
    if isinstance(type_, ir.VectorType):
        return ir.VectorType.get(ir.VectorType(type_).shape, element_type)
    return element_type


def is_bf16_extension(op: ir.Operation) -> bool:
    """``op`` is an ``llvm.fpext`` of bf16 (scalar or vector) to f32."""
    return (
        op.name == _FPEXT
        and isinstance(element(op.operands[0].type), ir.BF16Type)
        and isinstance(element(op.results[0].type), ir.F32Type)
    )


def is_bf16_truncation(op: ir.Operation) -> bool:
    """``op`` is an ``llvm.fptrunc`` of f32 (scalar or vector) to bf16."""
    return (
        op.name == _FPTRUNC
        and isinstance(element(op.operands[0].type), ir.F32Type)
        and isinstance(element(op.results[0].type), ir.BF16Type)
    )


def constant(type_: ir.Type, value: int) -> ir.Value:
    scalar = ir.IntegerAttr.get(element(type_), value)
    attr = (
        ir.DenseElementsAttr.get_splat(type_, scalar)
        if isinstance(type_, ir.VectorType)
        else scalar
    )
    return ir.Operation.create(
        "llvm.mlir.constant", results=[type_], attributes={"value": attr}
    ).results[0]


def llvm_op(name: str, result: ir.Type, *operands: ir.Value) -> ir.Value:
    return ir.Operation.create(name, results=[result], operands=list(operands)).results[
        0
    ]


def _rewrite(extension: ir.Operation) -> None:
    source = extension.operands[0]
    i16 = like(source.type, ir.IntegerType.get_signless(16))
    i32 = like(source.type, ir.IntegerType.get_signless(32))
    with ir.InsertionPoint(extension), extension.location:
        bits = llvm_op("llvm.zext", i32, llvm_op("llvm.bitcast", i16, source))
        shifted = llvm_op("llvm.shl", i32, bits, constant(i32, 16))
        result = llvm_op("llvm.bitcast", extension.results[0].type, shifted)
    extension.results[0].replace_all_uses_with(result)
    extension.erase()


def _only_rounded_back(extension: ir.Operation) -> bool:
    uses = [use.owner for use in extension.results[0].uses]
    return bool(uses) and all(is_bf16_truncation(user) for user in uses)


def bf16_extensions_as_shifts(module: ir.Module) -> int:
    """Rewrites ``module``'s matching XeVM extensions in place; returns how
    many."""
    extensions: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if is_bf16_extension(op) and not _only_rounded_back(op):
            extensions.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    for extension in extensions:
        _rewrite(extension)
    return len(extensions)
