"""``schedule_amx_loads``: AMX operand tile loads interleaved with the
dot-products, oneDNN's order."""

import operator

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import opview

_AMX_DOTS = ("x86.amx.tile_mulf", "x86.amx.tile_muli")
_AMX_TILE_REGISTERS = 8
# Ops an AMX operand load may be moved across: none writes memory.
_AMX_SCHEDULABLE = (
    *_AMX_DOTS,
    "x86.amx.tile_load",
    "affine.apply",
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def _is_tile(value: ir.Value) -> bool:
    return str(value.type).startswith("!x86.amx.tile")


def _defining_op(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    return None if isinstance(owner, ir.Block) else opview(owner).operation


def _tile_registers(order: list[ir.Operation]) -> int:
    """The most AMX tile registers live at once running ``order`` (one block's
    ops): a dot-product accumulates into its accumulator's register."""
    last_use: dict[ir.Value, int] = {}
    live: set[ir.Value] = set()
    for i, op in enumerate(order):
        for operand in op.operands:
            if _is_tile(operand):
                last_use[operand] = i
                if (defining := _defining_op(operand)) is None or defining not in order:
                    live.add(operand)
    most = len(live)
    for i, op in enumerate(order):
        results = [r for r in op.results if _is_tile(r) and r in last_use]
        dying = {v for v in op.operands if _is_tile(v) and last_use.get(v) == i}
        reused = 1 if op.name in _AMX_DOTS and op.operands[2] in dying else 0
        most = max(most, len(live) + len(results) - reused)
        live -= dying
        live.update(results)
    return most


def _schedule_amx_loads(block: ir.Block, distance: int) -> None:
    """Move each AMX operand tile load of ``block`` before the dot-product
    ``distance`` dot-products ahead of its first use (oneDNN's order for 1:
    the next operand loads while the current dot-product runs)."""
    order = [op.operation for op in block.operations]
    dots = [op for op in order if op.name in _AMX_DOTS]
    if len(dots) < 2:
        return
    loads = []
    for op in order:
        if op.name != "x86.amx.tile_load":
            continue
        uses = list(op.results[0].uses)
        if not uses or any(
            opview(use.owner).operation not in dots or use.operand_number > 1
            for use in uses
        ):
            continue
        first_use = min(dots.index(opview(use.owner).operation) for use in uses)
        loads.append((max(first_use - distance, 0), op))
    if not loads:
        return
    start = min(order.index(dots[0]), *(order.index(load) for _, load in loads))
    crossed = order[start : order.index(dots[-1]) + 1]
    if any(
        op.name not in _AMX_SCHEDULABLE and not op.name.startswith("arith.")
        for op in crossed
    ):
        return
    loads.sort(key=operator.itemgetter(0))
    moved = {load for _, load in loads}
    new_order = [op for op in order if op not in moved]
    for target, load in loads:
        new_order.insert(new_order.index(dots[target]), load)
    position = {op: i for i, op in enumerate(new_order)}
    for _, load in loads:
        for operand in load.operands:
            defining = _defining_op(operand)
            if defining in position and position[defining] > position[load]:
                return
    if _tile_registers(new_order) > _AMX_TILE_REGISTERS:
        return
    for target, load in loads:
        load.move_before(dots[target])


@transform_op(modifies_payload=True)
class ScheduleAmxLoadsOp(HelionTransformDialect.Operation, name="schedule_amx_loads"):
    """Interleave the AMX operand tile loads of every block with its
    dot-products (see ``_schedule_amx_loads``). Blocks writing memory between
    their dot-products, or needing more tile registers so, stay as they are."""

    target: ext.Operand[transform.AnyOpType]
    distance: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)
    )

    @staticmethod
    def run(
        op: "ScheduleAmxLoadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        blocks: list[ir.Block] = []

        def collect(visited: ir.Operation) -> ir.WalkResult:
            if visited.name in _AMX_DOTS and visited.block not in blocks:
                blocks.append(visited.block)
            return ir.WalkResult.ADVANCE

        for target in state.get_payload_ops(op.target):
            target.operation.walk(collect)
        for block in blocks:
            _schedule_amx_loads(block, op.distance.value)
        return DiagnosedSilenceableFailure.Success


def schedule_amx_loads(target: ir.Value, distance: int = 1) -> ScheduleAmxLoadsOp:
    """snake_case wrapper to create a ScheduleAmxLoadsOp."""
    return ScheduleAmxLoadsOp(
        target=target,
        distance=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), distance),
    )
