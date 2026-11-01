"""``check_lane_reductions``: refuses kernels with a cross-lane reduction that
stops before all 16 lanes, read from the lowered IR at the end of the ``xevm``
stage.

XeGPU lowers a cross-lane reduction to ``sub_group_shuffle_xor`` with masks 1, 2,
4, 8 in turn, each on the combination of the previous step, up to the lanes the
reduced dim spreads over. A 2D block load holds one column per lane, so an honest
reduction of a loaded tile runs the whole chain (16 lanes: Xe2 ``bmg``, the only
target of this pipeline). A chain that stops early comes from a source layout of
pairs or fours of lanes per row (``lane_layout = [8, 2]``), which XeGPU derives
from a consumer with ``lane_layout = [8]`` and imposes on the load without moving
data: wrong rows (:mod:`keep_row_dims` has the details and rewrites the stores
known to cause it).

The rule is on def-use chains: a shuffle of mask 1, 2 or 4 must reach a shuffle
of twice its mask, as that shuffle's value, through ops of its own block and
through no value that another shuffle takes. Counting masks is not enough (CSE
merges the first shuffles of two reductions of one value, and the spare counts
hide a short chain), nor is any path through the function (a short chain's value
can flow into a later reduction, which has a mask 2 of its own). Upstream warns about the layout on the load, but that warning does
not reach a Helion lowering; the lowered IR is what actually runs, whichever
layout the propagation resolved in this run.

Caught: reductions over 2, 4 or 8 lanes, the wrong row reductions of the layout
above. Also refused though right: a full reduction of a vector of 2, 4 or 8
elements to a scalar (nothing in the KernelBench XPU problems or the tests does
that). Not caught: a wrong layout that lowers without such a shuffle, e.g. a load
with ``lane_layout = [8, 2]`` that nothing reduces across lanes, a lane-local
reduction, or a store at the wrong row; shuffles with masks that are not constants.
"""

from __future__ import annotations

from mlir import ir

from ..mlir.support.errors import UnsupportedOperationError

_SHUFFLE_XOR = "_Z21sub_group_shuffle_xor"
_LAST_MASK = 8
"""A subgroup has 16 lanes on the target, so a full chain ends at mask 8."""


def _is_shuffle(op: ir.Operation) -> bool:
    if op.name != "llvm.call":
        return False
    callee = op.attributes.get("callee")
    return callee is not None and callee.value.startswith(_SHUFFLE_XOR)


def _mask(shuffle: ir.Operation) -> int | None:
    """The constant mask of a ``sub_group_shuffle_xor`` call, if it is one."""
    if len(shuffle.operands) != 2:
        return None
    owner = shuffle.operands[1].owner
    if isinstance(owner, ir.Block) or owner.name != "llvm.mlir.constant":
        return None
    value = owner.attributes["value"]
    return value.value if isinstance(value, ir.IntegerAttr) else None


def _continues(shuffle: ir.Operation, mask: int) -> bool:
    """A shuffle of mask ``2 * mask`` takes a value computed from ``shuffle``,
    in its block. A value another shuffle takes is the start of another
    reduction, and is not followed."""
    block = shuffle.operation.block
    seen: set[ir.Value] = set()
    stack = list(shuffle.results)
    while stack:
        value = stack.pop()
        if value in seen:
            continue
        seen.add(value)
        users = [use.owner for use in value.uses if use.owner.operation.block == block]
        shuffles = [user for user in users if _is_shuffle(user)]
        if any(
            user.operands[0] == value and _mask(user) == 2 * mask for user in shuffles
        ):
            return True
        if not any(user.operands[0] == value for user in shuffles):
            for user in users:
                stack.extend(user.results)
    return False


def partial_chains(func: ir.Operation) -> list[int]:
    """The masks at which ``func``'s (an ``llvm.func``) shuffle chains stop
    before the last one, sorted."""
    ends: set[int] = set()

    def visit(op: ir.Operation) -> ir.WalkResult:
        if _is_shuffle(op):
            mask = _mask(op)
            if mask in (1, 2, 4) and not _continues(op, mask):
                ends.add(mask)
        return ir.WalkResult.ADVANCE

    func.walk(visit)
    return sorted(ends)


def check_lane_reductions(module: ir.Module) -> None:
    """Raise :class:`UnsupportedOperationError` if an ``llvm.func`` in ``module``'s
    GPU modules has a cross-lane reduction over part of a subgroup."""
    found: list[tuple[str, list[int]]] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.module":
            for inner in op.regions[0].blocks[0].operations:
                if inner.name == "llvm.func" and (masks := partial_chains(inner)):
                    found.append((inner.attributes["sym_name"].value, masks))
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    if found:
        name, masks = found[0]
        raise UnsupportedOperationError(
            "row reduction over part of a subgroup",
            reason=(
                f"kernel '{name}': a cross-lane reduction ends at shuffle mask "
                f"{masks[0]} instead of {_LAST_MASK}, so XeGPU laid out its source "
                "with several rows per subgroup and fewer than 16 lanes per row, "
                "which 2D block loads do not give (wrong rows)"
            ),
            alternatives=[
                "store row results with keepdim=True ([rows, 1])",
                "target='cpu' (CPU tensors)",
            ],
        )
