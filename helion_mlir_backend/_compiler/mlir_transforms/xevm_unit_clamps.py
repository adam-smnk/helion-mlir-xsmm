"""``unit_clamps_without_sat``: in XeVM GPU modules, a compare-and-select with 1
compares with the number next to 1, so IGC does not form a saturation from the
clamps to ``[0, 1]`` we emit. Works around two IGC miscompiles (no IGC patch, by
policy).

IGC's ``isSat`` takes ``select(fcmp a, b), a, b`` for a minimum or maximum,
whatever the predicate, and a minimum with 1 of a maximum with 0 (in either
order) for a ``(sat)`` modifier. The IR's answer differs (IGC 2.41, B50; reproducer
in helion-mlir-dev: ``agent-wiki/upstream-repro/xpu-nan/README.md``):

- ``(sat)`` gives 0 for NaN and +0 for -0. The IR gives its input for both:
  ``x < 0 ? 0 : x`` compares false. Torch's clamps do too (see ``nan_minmax``).
- On a ``bfloat`` select with its compare on the ``fpext`` (what a bf16
  ``clamp_max(clamp_min(x, 0.0), 1.0)`` folds to), ``(sat)`` lands on a bf16
  ``mov (sat) :uw :uw``: nothing happens, -3.375 stays -3.375.

So the compare with 1 changes to the same test against the neighbouring number:
``y > 1`` is ``y >= 1 + ulp``, ``y < 1`` is ``y <= 1 - ulp/2``. It is the same
test for every ``y``, NaN and infinities too (both numbers are normal, so
flushing subnormals does not matter). The compare no longer has the select's
operands, so IGC finds no minimum with 1. Kernels then take two compares and two
selects, as a clamp to any other range does.

Every compare-and-select with 1 changes, not only those next to a select with 0:
IGC's optimizer forms such a pair later, by folding what lies between
(``F.hardtanh(F.relu(x))``). The rewrite cannot change a result, so a wide match
costs only the odd compare (32_HardTanh and 81_Gemm: another constant, the same
number of instructions).

Matched: an ``llvm.select`` of an ``llvm.fcmp`` with an ordering predicate of the
select's operands, one operand a float constant of 1 (scalar or splat, also
behind ``fpext``, ``fptrunc``). The other operand may differ between the two by
those conversions, which IGC folds. Only ``gpu.module``s with an ``#xevm.target``
are changed.

Still reaching ``(sat)``, with no frontend path found to them (the review's
attempts, e.g. ``torch.fmin(torch.clamp_min(x, 0), ones)``, came out right):

- bounds IGC folds from non-constants: ``sitofp``, ``bitcast``, an ``fmul`` of
  constants;
- constants that differ per lane (not splats);
- ``llvm.minnum`` mixed with a compare-and-select maximum.

Plain ``llvm.minnum``/``llvm.maxnum`` pairs also become ``(sat)``, which is legal:
they may map NaN to the other operand and order zeros either way.
"""

from __future__ import annotations

from mlir import ir

from .xevm_alloca_arrays import xevm_modules

_OGT, _OGE, _OLT, _OLE = 2, 3, 4, 5
_UGT, _UGE, _ULT, _ULE = 9, 10, 11, 12
_MIRRORED = {
    _OGT: _OLT,
    _OGE: _OLE,
    _OLT: _OGT,
    _OLE: _OGE,
    _UGT: _ULT,
    _UGE: _ULE,
    _ULT: _UGT,
    _ULE: _UGE,
}
_AGAINST_ONE = {
    # y P 1 as y P' c: the predicate P' and whether c is above 1.
    _OGT: (_OGE, True),
    _OLE: (_OLT, True),
    _OGE: (_OGT, False),
    _OLT: (_OLE, False),
    _UGT: (_UGE, True),
    _ULE: (_ULT, True),
    _UGE: (_UGT, False),
    _ULT: (_ULE, False),
}
_NEXT_TO_ONE = {
    # Above and below 1, in the type's hexadecimal form.
    ir.F32Type: ("0x3F800001", "0x3F7FFFFF"),
    ir.F16Type: ("0x3C01", "0x3BFF"),
    ir.BF16Type: ("0x3F81", "0x3F7F"),
    ir.F64Type: ("0x3FF0000000000001", "0x3FEFFFFFFFFFFFFF"),
}


def _unconverted(value: ir.Value) -> ir.Value:
    """``value`` without the ``fpext``s and ``fptrunc``s it comes from."""
    while isinstance(value, ir.OpResult) and value.owner.name in (
        "llvm.fpext",
        "llvm.fptrunc",
    ):
        value = value.owner.operands[0]
    return value


def _constant_value(value: ir.Value) -> float | None:
    """The number of a float ``llvm.mlir.constant`` (scalar or splat) or
    ``llvm.mlir.zero``, behind any conversions."""
    value = _unconverted(value)
    if isinstance(value, ir.OpResult) and value.owner.name == "llvm.mlir.zero":
        return 0.0
    if not isinstance(value, ir.OpResult) or value.owner.name != "llvm.mlir.constant":
        return None
    attribute = value.owner.attributes["value"]
    if isinstance(attribute, ir.FloatAttr):
        return attribute.value
    if isinstance(attribute, ir.DenseFPElementsAttr) and attribute.is_splat:
        return ir.FloatAttr(attribute.get_splat_value()).value
    return None


def _compare_with_one(select: ir.Operation) -> ir.Operation | None:
    """If ``select`` is a compare-and-select of a value and a constant 1 (see
    the module): the compare."""
    condition, *selected = select.operands
    if not isinstance(condition, ir.OpResult) or condition.owner.name != "llvm.fcmp":
        return None
    compare = condition.owner
    if ir.IntegerAttr(compare.attributes["predicate"]).value not in _MIRRORED:
        return None
    selected_bounds = [_constant_value(value) for value in selected]
    compared = list(compare.operands)
    compared_bounds = [_constant_value(value) for value in compared]
    if (
        sum(bound is not None for bound in selected_bounds) != 1
        or sum(bound is not None for bound in compared_bounds) != 1
        or 1.0 not in selected_bounds
        or 1.0 not in compared_bounds
    ):
        return None
    other = selected[selected_bounds.index(None)]
    if _unconverted(compared[compared_bounds.index(None)]) != _unconverted(other):
        return None
    return compare


def _next_to_one(type_: ir.Type, above: bool) -> ir.Attribute | None:
    element = (
        ir.VectorType(type_).element_type if isinstance(type_, ir.VectorType) else type_
    )
    bits = next(
        (forms for kind, forms in _NEXT_TO_ONE.items() if isinstance(element, kind)),
        None,
    )
    if bits is None:
        return None
    text = bits[0 if above else 1]
    if isinstance(type_, ir.VectorType):
        return ir.Attribute.parse(f"dense<{text}> : {type_}")
    return ir.Attribute.parse(f"{text} : {type_}")


def _compare_next_to_one(select: ir.Operation, compare: ir.Operation) -> bool:
    """``select`` compares with the number next to 1 (see the module); whether
    it could (the compare's type has that number)."""
    predicate = ir.IntegerAttr(compare.attributes["predicate"]).value
    lhs, rhs = compare.operands
    if _constant_value(lhs) is not None:
        lhs, predicate = rhs, _MIRRORED[predicate]
    replacement, above = _AGAINST_ONE[predicate]
    attribute = _next_to_one(lhs.type, above)
    if attribute is None:
        return False
    with ir.InsertionPoint(compare), compare.location:
        constant = ir.Operation.create(
            "llvm.mlir.constant", results=[lhs.type], attributes={"value": attribute}
        ).results[0]
        flipped = ir.Operation.create(
            "llvm.fcmp",
            results=[compare.results[0].type],
            operands=[lhs, constant],
            attributes={
                "predicate": ir.IntegerAttr.get(
                    compare.attributes["predicate"].type, replacement
                ),
                "fastmathFlags": compare.attributes["fastmathFlags"],
            },
        ).results[0]
    select.operands[0] = flipped
    if not list(compare.results[0].uses):
        compare.erase()
    return True


def unit_clamps_without_sat(module: ir.Module) -> int:
    """Rewrites the matching compares of ``module``'s XeVM modules in place;
    returns how many."""
    selects: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "llvm.select":
            selects.append(op)
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit)
    count = 0
    for select in selects:
        compare = _compare_with_one(select)
        if compare is not None:
            count += _compare_next_to_one(select, compare)
    return count
