"""``restore_nan_propagating_maxima``: ``maxnumf`` vector reductions back to
``maximumf``, except an attention kernel's that may stay.

Lighthouse's ``vectorize_schedule`` (``enable_fastmath_optimizations``) turns every
``maximumf`` multi-reduction into a ``maxnumf`` one, which ignores NaN operands (its
FIXME). Torch's maxima propagate NaN (``amax``, ``max``, the row maximum of a
softmax), and Helion kernels produce no ``maxnumf`` reduction of their own (torch
has no NaN-ignoring maximum reduction), so every ``maxnumf`` reduction here is
lighthouse's rewrite.

In a kernel with ``xegpu_nan_ignoring_maxima``, a reduction stays ``maxnumf`` if a
NaN it ignores cannot change the output, which :func:`nan_cannot_be_lost` checks on
its def-use graph: an online softmax's maximum ``m`` only shifts ``exp(s - m)``, so a
NaN score makes its ``exp`` NaN, and so the row, whatever ``m`` was. Any other use of
the maximum (stored, compared with anything but another maximum, ...) keeps
``maximumf``, as does a NaN that an ``exp`` could lose or that would stay in one
element.

**A ``maxnumf`` reduction depends on IGC's options.** With ``llvm.maxnum`` row maxima
IGC's vectorizer forms a vector compare and select around its joint ``WaveAll``,
which IGC takes for a min/max that writes one element: the rest read stale registers
(``VectorizerAllowCMP`` in ``xpu_transforms.IGC_OPTIONS``). ``llvm.maximum`` expands
to scalar compares that do not take that path. Either of ``igc_vectorizer``'s
choices avoids it, but a build that loses the options gets wrong rows with these
reductions and right ones without; nothing refuses that at compile time. Which
reductions may stay does not depend on the choice: this checks the kernel's def-use
graph before IGC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir import ir

if TYPE_CHECKING:
    from collections.abc import Collection

_REDUCTION = "vector.multi_reduction"
_MAXNUM = "#vector.kind<maxnumf>"
_MAXIMUM = "#vector.kind<maximumf>"
_SHAPE_OPS = frozenset({"vector.broadcast", "vector.transpose", "vector.shape_cast"})
_MAX_OPS = frozenset({"arith.maximumf", "arith.maxnumf"})
_NAN_KEEPING = frozenset(
    {
        *_SHAPE_OPS,
        "arith.addf",
        "arith.divf",
        "arith.extf",
        "arith.mulf",
        "arith.negf",
        "arith.subf",
        "arith.truncf",
    }
)
"""Ops whose result is NaN where an operand is: elementwise."""
_NAN_KEEPING_KINDS = frozenset(
    {"add", "mul", "maximumf", "minimumf", "maxnumf"},
)
"""Reductions over a row that a NaN makes NaN. ``maxnumf`` does not, but then that
reduction is the one checked, for its own NaNs."""
_MIXING = frozenset({_REDUCTION, "vector.contract"})
"""Ops that spread a NaN over the row."""


def _kind(op: ir.Operation) -> str:
    return str(op.attributes["kind"])


def _carried(op: ir.Operation, value: ir.Value) -> list[ir.Value] | None:
    """The values a ``scf.yield`` ``op`` of a ``scf.for`` passes ``value`` to: the
    iteration argument and the loop result. ``None`` for another parent."""
    loop = op.parent
    if loop is None or loop.name != "scf.for":
        return None
    arguments = list(loop.regions[0].blocks[0].arguments)
    carried: list[ir.Value] = []
    for index, operand in enumerate(op.operands):
        if operand == value:
            carried += [arguments[index + 1], loop.results[index]]
    return carried


def _nan_reaches_the_whole_row(exp: ir.Operation, source_shift: bool) -> bool:
    """The forward slice of ``exp``'s result keeps NaN up to stores, of which there
    is one at least. With ``source_shift``, what is stored has also been through a
    reduction or a contraction, or a NaN would stay in its element, where a NaN
    maximum makes the whole row NaN."""
    values: set[ir.Value] = set()
    ops: list[ir.Operation] = []
    stores: list[ir.Operation] = []
    work: list[ir.Value] = [exp.results[0]]
    while work:
        value = work.pop()
        if value in values:
            continue
        values.add(value)
        for use in value.uses:
            op = use.owner
            name = op.name
            if name == "vector.transfer_write":
                if op.operands[0] != value:
                    return False
                stores.append(op)
            elif name == "scf.yield":
                carried = _carried(op, value)
                if carried is None:
                    return False
                work += carried
            elif (
                name in _NAN_KEEPING
                or name == "vector.contract"
                or (name == _REDUCTION and _kind_name(op) in _NAN_KEEPING_KINDS)
            ):
                ops.append(op)
                work += list(op.results)
            else:
                return False
    if not stores:
        return False
    if not source_shift:
        return True
    mixed: set[ir.Value] = set()
    changed = True
    while changed:
        changed = False
        for op in ops:
            if op.results[0] in mixed:
                continue
            if op.name in _MIXING or any(operand in mixed for operand in op.operands):
                mixed.add(op.results[0])
                changed = True
        for value in list(mixed):
            for use in value.uses:
                if use.owner.name == "scf.yield":
                    for carried in _carried(use.owner, value) or ():
                        if carried in values and carried not in mixed:
                            mixed.add(carried)
                            changed = True
    return all(store.operands[0] in mixed for store in stores)


def _kind_name(op: ir.Operation) -> str:
    return _kind(op).removeprefix("#vector.kind<").removesuffix(">")


def nan_cannot_be_lost(reduction: ir.Operation) -> bool:
    """A NaN that the ``maxnumf`` ``reduction`` ignores still reaches the output as
    with a propagating maximum. Its result reaches only:

    - other maxima of the same chain: a running maximum, as compare and select,
      ``maximumf`` or ``maxnumf`` of chain values, carried over a ``scf.for``,
      broadcast or transposed;
    - ``subf`` with a chain value on the right, whose only use is ``math.exp``.

    At least one ``subf`` has the reduction's own source on the left: the NaN is in
    an ``exp``. Those ``exp`` keep it up to the stores, over the whole row
    (:func:`_nan_reaches_the_whole_row`)."""
    source = reduction.operands[0]
    chain: set[ir.Value] = set()
    compares: list[ir.Operation] = []
    selects: list[ir.Operation] = []
    maxima: list[ir.Operation] = []
    shifts: list[ir.Operation] = []
    work: list[ir.Value] = [reduction.results[0]]
    while work:
        value = work.pop()
        if value in chain:
            continue
        chain.add(value)
        for use in value.uses:
            op = use.owner
            name = op.name
            if name in _SHAPE_OPS:
                work.append(op.results[0])
            elif name in _MAX_OPS:
                maxima.append(op)
                work.append(op.results[0])
            elif name == "arith.select":
                selects.append(op)
                work.append(op.results[0])
            elif name == "arith.cmpf":
                compares.append(op)
            elif name == "arith.subf":
                shifts.append(op)
            elif name == "scf.yield":
                carried = _carried(op, value)
                if carried is None:
                    return False
                work += carried
            else:
                return False
    conditions = {op.results[0] for op in compares}
    if not all(set(op.operands) <= chain for op in compares + maxima):
        return False
    if not all(
        all(use.owner.name == "arith.select" for use in op.results[0].uses)
        for op in compares
    ):
        return False
    for op in selects:
        condition, *chosen = op.operands
        if condition not in conditions or not set(chosen) <= chain:
            return False
    found_source = False
    for shift in shifts:
        left, right = shift.operands
        users = [use.owner for use in shift.results[0].uses]
        if right not in chain or len(users) != 1 or users[0].name != "math.exp":
            return False
        is_source = left == source
        if not is_source and left not in chain:
            return False
        found_source = found_source or is_source
        if not _nan_reaches_the_whole_row(users[0], is_source):
            return False
    return found_source


def _reductions(root: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == _REDUCTION and _kind(op) == _MAXNUM:
            found.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def restore_nan_propagating_maxima(
    module: ir.Module, ignoring_kernels: Collection[int], kernel_attr: str
) -> int:
    """``module``'s ``maxnumf`` reductions as ``maximumf``, but in the grid loops
    tagged ``kernel_attr`` with an index in ``ignoring_kernels`` those that pass
    :func:`nan_cannot_be_lost`; returns how many stay."""
    kept: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.forall" and kernel_attr in op.attributes:
            index = ir.IntegerAttr(op.attributes[kernel_attr]).value
            if index in ignoring_kernels:
                kept.extend(
                    reduction
                    for reduction in _reductions(op)
                    if nan_cannot_be_lost(reduction)
                )
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    maximum = ir.Attribute.parse(_MAXIMUM, module.context)
    for reduction in _reductions(module.operation):
        if reduction not in kept:
            reduction.attributes["kind"] = maximum
    return len(kept)
