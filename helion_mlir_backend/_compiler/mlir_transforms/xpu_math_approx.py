"""``approximate_math``: f32 ``math.tanh`` and ``math.log1p`` in XeVM GPU
modules become branch-free ``arith``/``math`` ops, in place of OpenCL builtin
calls.

XPU math without a native form goes to upstream's scalar OpenCL patterns: one
``__spirv_ocl_<fn>f`` call per element, each inlined by IGC with its branches
(a vector ``__spirv_ocl_log1p`` is no better: IGC expands it per lane too,
1029 instructions for 8 elements). At 64 elements per work item that is
about 14000 instructions of predicated SIMD16 code, and the compute limits the
kernel below the DRAM ceiling: 29_Softplus 154 GB/s on the Arc B50, torch 191.

Upstream has polynomial patterns for both, but only as one all-or-nothing
pattern set: it would also expand ``exp`` (a native hardware function here)
and every other function, which pushed a Mish GEMM past the compile timeout.

- ``tanh``: the rational approximation of upstream's ``TanhApproximation`` (and
  Eigen's), with ``x`` itself below 0.0004 and the argument clamped by selects,
  so NaN stays NaN. 5 f32 ulps at most over all 2^32 patterns, the same as
  OpenCL's ``tanh``, as torch-xpu.
- ``log1p``: W. Kahan's ``x * log(u) / (u - 1)`` with ``u = 1 + x``, ``x`` if
  ``u`` is 1 or infinite, as upstream's ``Log1pApproximation``, on upstream's
  ``LogApproximation`` (Cephes' polynomial of the mantissa). Users of f32
  ``log1p`` get up to 5 ulps over all 2^32 patterns, where torch-xpu has 2: the
  price of the speed. In its place, measured with 128 GRFs on Softplus (194 GB/s
  with this one): OpenCL's ``log`` has 3 ulps in a 1/7 sample but 90 GB/s, the
  native ``log`` 194 GB/s but 76 ulps near 1. With the 256 GRFs of
  ``xpu_transforms.IGC_OPTIONS`` Softplus runs at 186 GB/s (the cause of the
  difference is not established).

For bf16 inputs both give the same bits as an fp64 reference rounded to bf16,
for all 65536 patterns (``tests/test_xpu_runtime.py``).

Long chains of these keep many tile-sized temporaries live:
:mod:`xevm_tile_order`.

Matched exactly: the two ops on f32 scalars or vectors in a ``gpu.module`` with
an ``#xevm.target``. bf16 math has been extended to f32 before (the XPU target's
``xevm-math`` stage); f16 math stays f16 until upstream's pipeline extends it
later, so it keeps OpenCL's builtins. Other functions, types and host code are
left alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir import ir
import torch

from .xevm_alloca_arrays import xevm_modules

if TYPE_CHECKING:
    from collections.abc import Callable

_TANH_CLAMP = 7.99881172180175781
_TANH_TINY = 0.0004
_TANH_NUMERATOR = (
    -2.76076847742355e-16,
    2.00018790482477e-13,
    -8.60467152213735e-11,
    5.12229709037114e-08,
    1.48572235717979e-05,
    6.37261928875436e-04,
    4.89352455891786e-03,
)
"""The coefficients of ``x^13`` to ``x^1``, in powers of ``x^2``."""
_TANH_DENOMINATOR = (
    1.19825839466702e-06,
    1.18534705686654e-04,
    2.26843463243900e-03,
    4.89352518554385e-03,
)
"""The coefficients of ``x^6`` to ``x^0``, in powers of ``x^2``."""
_LOG_POLYNOMIAL = (
    (7.0376836292e-2, -1.1514610310e-1, 1.1676998740e-1),
    (-1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1),
    (2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1),
)
"""Cephes' ``logf``, three Horner parts of ``x^8`` to ``x^0``, as upstream."""
_LN2 = 0.693147180559945309417232121458176568
_SQRT_HALF = 0.707106781186547524
_F32_EXPONENT_OF_HALF = 126
_F32_MANTISSA = 0x007FFFFF
_F32_BITS_OF_HALF = 0x3F000000
_OEQ, _OGT, _OLT = 1, 2, 4
"""``arith.cmpf`` predicates."""


class _Builder:
    """Ops of the type of ``value`` (f32, scalar or vector) at the insertion
    point."""

    def __init__(self, value: ir.Value) -> None:
        self.type = value.type
        i1 = ir.IntegerType.get_signless(1)
        self.mask = (
            ir.VectorType.get(ir.VectorType(self.type).shape, i1)
            if isinstance(self.type, ir.VectorType)
            else i1
        )
        i32 = ir.IntegerType.get_signless(32)
        self.integers = (
            ir.VectorType.get(ir.VectorType(self.type).shape, i32)
            if isinstance(self.type, ir.VectorType)
            else i32
        )

    def integer(self, value: int) -> ir.Value:
        attr: ir.Attribute = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
        if isinstance(self.integers, ir.VectorType):
            attr = ir.DenseElementsAttr.get_splat(self.integers, attr)
        return ir.Operation.create(
            "arith.constant", results=[self.integers], attributes={"value": attr}
        ).results[0]

    def integer_op(self, name: str, a: ir.Value, b: ir.Value) -> ir.Value:
        return ir.Operation.create(
            name, results=[self.integers], operands=[a, b]
        ).results[0]

    def convert(self, name: str, value: ir.Value, type_: ir.Type) -> ir.Value:
        return ir.Operation.create(name, results=[type_], operands=[value]).results[0]

    def constant(self, value: float) -> ir.Value:
        attr: ir.Attribute = ir.FloatAttr.get(ir.F32Type.get(), value)
        if isinstance(self.type, ir.VectorType):
            attr = ir.DenseElementsAttr.get_splat(self.type, attr)
        return ir.Operation.create(
            "arith.constant", results=[self.type], attributes={"value": attr}
        ).results[0]

    def op(self, name: str, *operands: ir.Value) -> ir.Value:
        return ir.Operation.create(
            name, results=[self.type], operands=list(operands)
        ).results[0]

    def compare(self, predicate: int, a: ir.Value, b: ir.Value) -> ir.Value:
        i64 = ir.IntegerType.get_signless(64)
        return ir.Operation.create(
            "arith.cmpf",
            results=[self.mask],
            operands=[a, b],
            attributes={"predicate": ir.IntegerAttr.get(i64, predicate)},
        ).results[0]

    def select(self, mask: ir.Value, a: ir.Value, b: ir.Value) -> ir.Value:
        return self.op("arith.select", mask, a, b)

    def horner(self, x: ir.Value, coefficients: tuple[float, ...]) -> ir.Value:
        result = self.constant(coefficients[0])
        for coefficient in coefficients[1:]:
            result = self.op("math.fma", x, result, self.constant(coefficient))
        return result


def _tanh(value: ir.Value) -> ir.Value:
    b = _Builder(value)
    high, low = b.constant(_TANH_CLAMP), b.constant(-_TANH_CLAMP)
    x = b.select(
        b.compare(_OGT, value, high),
        high,
        b.select(b.compare(_OLT, value, low), low, value),
    )
    square = b.op("arith.mulf", x, x)
    numerator = b.op("arith.mulf", x, b.horner(square, _TANH_NUMERATOR))
    denominator = b.horner(square, _TANH_DENOMINATOR)
    ratio = b.op("arith.divf", numerator, denominator)
    tiny = b.compare(_OLT, b.op("math.absf", value), b.constant(_TANH_TINY))
    return b.select(tiny, value, ratio)


def _log(b: _Builder, value: ir.Value) -> ir.Value:
    """``log(value)``; finite ``value``s, infinities are for the caller."""
    i32 = b.integers
    bits = b.convert("arith.bitcast", value, i32)
    exponent = b.integer_op(
        "arith.subi",
        b.integer_op("arith.shrui", bits, b.integer(23)),
        b.integer(_F32_EXPONENT_OF_HALF),
    )
    mantissa = b.integer_op(
        "arith.ori",
        b.integer_op("arith.andi", bits, b.integer(_F32_MANTISSA)),
        b.integer(_F32_BITS_OF_HALF),
    )
    m = b.convert("arith.bitcast", mantissa, b.type)
    e = b.convert("arith.sitofp", exponent, b.type)
    below = b.compare(_OLT, m, b.constant(_SQRT_HALF))
    zero = b.constant(0.0)
    x = b.op(
        "arith.addf", b.op("arith.subf", m, b.constant(1.0)), b.select(below, m, zero)
    )
    e = b.op("arith.subf", e, b.select(below, b.constant(1.0), zero))
    square = b.op("arith.mulf", x, x)
    cube = b.op("arith.mulf", square, x)
    y0, y1, y2 = (b.horner(x, part) for part in _LOG_POLYNOMIAL)
    y = b.op("math.fma", y0, cube, y1)
    y = b.op("math.fma", y, cube, y2)
    y = b.op("arith.mulf", y, cube)
    y = b.op("math.fma", b.constant(-0.5), square, y)
    log = b.op("math.fma", e, b.constant(_LN2), b.op("arith.addf", x, y))
    log = b.select(b.compare(_OEQ, value, zero), b.constant(float("-inf")), log)
    return b.select(b.compare(_OLT, value, zero), b.constant(float("nan")), log)


def _log1p(value: ir.Value) -> ir.Value:
    b = _Builder(value)
    one = b.constant(1.0)
    u = b.op("arith.addf", value, one)
    scale = b.op("arith.divf", _log(b, u), b.op("arith.subf", u, one))
    exact = ir.Operation.create(
        "arith.ori",
        results=[b.mask],
        operands=[
            b.compare(_OEQ, u, one),
            b.compare(_OEQ, u, b.constant(float("inf"))),
        ],
    ).results[0]
    return b.select(exact, value, b.op("arith.mulf", value, scale))


_APPROXIMATIONS: dict[str, Callable[[ir.Value], ir.Value]] = {
    "math.tanh": _tanh,
    "math.log1p": _log1p,
}


def _is_f32(type_: ir.Type) -> bool:
    if isinstance(type_, ir.VectorType):
        type_ = ir.VectorType(type_).element_type
    return isinstance(type_, ir.F32Type)


def _splat_value(constant: ir.Operation) -> float:
    """The value of a bf16 or f16 splat constant, read from its printed form (the
    bindings have no ``float`` of these types); the printer round-trips, in hex
    if decimal does not."""
    attr = ir.DenseElementsAttr(constant.attributes["value"])
    bf16 = isinstance(ir.ShapedType(attr.type).element_type, ir.BF16Type)
    dtype = torch.bfloat16 if bf16 else torch.float16
    text = str(attr.get_splat_value()).split(":")[0].strip()
    if text.startswith("0x"):
        bits = torch.tensor([int(text, 16)], dtype=torch.int32).to(torch.int16)
        return bits.view(dtype).float().item()
    return torch.tensor(float(text), dtype=dtype).float().item()


def fold_extended_splats(module: ir.Module) -> int:
    """``arith.extf`` of a bf16 or f16 splat ``arith.constant`` to f32 in XeVM GPU
    modules as an f32 splat (exact). ``math-extend-to-supported-types`` leaves one for
    the constant operand of bf16 ``powf`` (``torch.pow(x, 2.5)``, ``torch.exp2``),
    and XeGPU's subgroup distribution then leaves that operand at tile size: an
    invalid op. Returns how many."""
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if (
            op.name == "arith.extf"
            and isinstance(op.results[0].type, ir.VectorType)
            and _is_f32(op.results[0].type)
        ):
            source = op.operands[0].owner
            if (
                not isinstance(source, ir.Block)
                and source.name == "arith.constant"
                and isinstance(source.attributes["value"], ir.DenseElementsAttr)
                and ir.DenseElementsAttr(source.attributes["value"]).is_splat
                and isinstance(
                    ir.VectorType(op.operands[0].type).element_type,
                    (ir.BF16Type, ir.F16Type),
                )
            ):
                found.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    for op in found:
        value = _splat_value(op.operands[0].owner)
        with ir.InsertionPoint(op), op.location:
            result = _Builder(op.results[0]).constant(value)
        op.results[0].replace_all_uses_with(result)
        op.erase()
    return len(found)


def approximate_math(module: ir.Module) -> int:
    """Rewrites ``module``'s matching ops in place; returns how many."""
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _APPROXIMATIONS and _is_f32(op.results[0].type):
            found.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    for op in found:
        with ir.InsertionPoint(op), op.location:
            result = _APPROXIMATIONS[op.name](op.operands[0])
        op.results[0].replace_all_uses_with(result)
        op.erase()
    return len(found)
