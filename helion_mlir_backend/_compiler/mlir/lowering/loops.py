"""Loops: one ``scf.forall`` (or a sequential ``scf.for`` nest) per root graph,
one ``scf.for`` per block id of a nested ``_for_loop``.

Written host tensors are threaded as SSA values (``tensor_state``): a forall
iteration works on its owned region of each written tensor and inserts it back
in the terminator; an ``scf.for`` carries the tensors its body writes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import arith as arith_d
from mlir.dialects import scf as scf_d
import mlir.ir as ir
import torch

from ..analysis.geometry import LoopBounds
from ..analysis.geometry import loop_block_ids
from ..analysis.tensor_effects import accessed_tensor
from ..support import NodeLoweringError
from . import emit
from .control_flow import carried_positions
from .control_flow import lower_subgraph
from .control_flow import with_carried
from .registry import lowers
from .slice_plan import tile_window
from .tensor_state import OwnedDim
from .tensor_state import owned_dims
from .tile_index_ops import scalar_tile_value

if TYPE_CHECKING:
    from ..build_context import BuildContext

log = logging.getLogger(__name__)


def build_phase_body(
    ctx: BuildContext, root_positions: list[int], outputs: tuple[str, ...]
) -> list[ir.Value]:
    """Lower the root graphs at ``root_positions`` in order; return ``outputs``' values.

    Every written host tensor (``outputs``) starts as its function argument.
    """
    device_ir = ctx.host_function.device_ir
    for name in outputs:
        ctx.tensors.bind(name, ctx.param_to_value[name])
    for position in root_positions:
        _lower_root(
            ctx, device_ir.root_ids[position], list(device_ir.grid_block_ids[position])
        )
    return [ctx.tensors.value(name) for name in outputs]


def _lower_root(ctx: BuildContext, graph_id: int, grid_ids: list[int]) -> None:
    """Parallel ``scf.forall`` if each iteration writes disjoint regions, else sequential."""
    geometry = ctx.geometry
    bounds = [
        LoopBounds(ctx.sizes.value(bound.begin), ctx.sizes.value(bound.end), bound.step)
        for bound in (geometry.root_bounds[block_id] for block_id in grid_ids)
    ]
    trips = [_trip_count(bound) for bound in bounds]

    names = list(ctx.effects.writes(graph_id))
    accesses = ctx.effects.accesses(graph_id)
    owned = {
        name: owned_dims(
            ctx,
            [node for node in accesses if accessed_tensor(node) == name],
            ir.RankedTensorType(ctx.tensors.value(name).type).rank,
            grid_ids,
        )
        for name in names
    }
    unowned = [
        (name, block_id)
        for name in names
        for block_id in grid_ids
        if block_id not in {dim.block_id for dim in owned[name].values()}
    ]
    graph = ctx.host_function.device_ir.graphs[graph_id].graph
    if not unowned:
        _emit_forall(ctx, graph, grid_ids, bounds, trips, names, owned)
        return
    log.debug(
        "root graph %d runs sequentially: %s is not partitioned by block_id %d",
        graph_id,
        *unowned[0],
    )
    _emit_sequential(ctx, graph, grid_ids, bounds, trips, names)


def _trip_count(bound: LoopBounds) -> int | ir.Value:
    """``ceildiv(end - begin, step)``, static when both bounds are."""
    if isinstance(bound.begin, int) and isinstance(bound.end, int):
        return -(-(bound.end - bound.begin) // bound.step)
    operands: list[ir.Value] = []
    exprs = []
    for value in (bound.end, bound.begin):
        if isinstance(value, ir.Value):
            exprs.append(ir.AffineSymbolExpr.get(len(operands)))
            operands.append(value)
        else:
            exprs.append(ir.AffineConstantExpr.get(value))
    end, begin = exprs
    expr = ir.AffineExpr.get_ceil_div(
        end - begin, ir.AffineConstantExpr.get(bound.step)
    )
    return affine_d.AffineApplyOp(
        ir.AffineMap.get(0, len(operands), [expr]), operands
    ).result


def _emit_forall(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int | ir.Value],
    names: list[str],
    owned: dict[str, dict[int, OwnedDim]],
) -> None:
    rank = len(grid_ids)
    forall = scf_d.ForallOp(
        [0] * rank,
        trips,
        [1] * rank,
        shared_outs=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(forall.body):
        for block_id, bound, trip_iv in zip(
            grid_ids, bounds, forall.induction_variables, strict=True
        ):
            _bind_grid_iv(ctx, block_id, bound, trip_iv)
        shared = dict(zip(names, forall.inner_iter_args, strict=True))
        regions = {
            name: _owned_region(ctx, name, shared[name], owned[name]) for name in names
        }
        region_values = {
            name: emit.extract_slice(shared[name], *regions[name]) for name in names
        }
        partial = [
            block_id for block_id in grid_ids if block_id in ctx.block_id_to_valid
        ]
        values = _lower_tile(ctx, graph, region_values, owned, partial)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            for name, value in zip(names, values, strict=True):
                emit.parallel_insert_slice(value, shared[name], *regions[name])
    for name, result in zip(names, forall.results, strict=True):
        ctx.tensors.bind(name, result)


def _lower_tile(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    region_values: dict[str, ir.Value],
    owned: dict[str, dict[int, OwnedDim]],
    partial: list[int],
) -> list[ir.Value]:
    """Lower one forall iteration on ``region_values``, the regions it owns of
    the tensors it writes; their final values.

    A tile that may be partial along ``partial`` grid dims is versioned on being
    full, one dim at a time: a full tile is lowered with static sizes (no padded
    loads or masked stores), only a partial one with Helion's tile mask. Every
    version works on the same region values, so they stay in place.
    """
    if not partial:
        for name, value in region_values.items():
            ctx.tensors.bind(name, value, owned[name])
        ctx.lower_graph(graph)
        return [ctx.tensors.value(name) for name in region_values]
    block_id, rest = partial[0], partial[1:]
    valid = ctx.block_id_to_valid[block_id]
    tile = ctx.geometry.tile_extent(block_id)
    is_full = arith_d.cmpi(arith_d.CmpIPredicate.eq, valid, ctx.index_const(tile))
    if_op = scf_d.IfOp(
        is_full, [value.type for value in region_values.values()], has_else=True
    )
    with ir.InsertionPoint(if_op.then_block):
        ctx.bind_loop(
            block_id,
            ctx.block_id_to_iv[block_id],
            ctx.block_id_to_bounds[block_id],
            span=tile,
        )
        scf_d.YieldOp(_lower_tile(ctx, graph, region_values, owned, rest))
    ctx.block_id_to_valid[block_id] = valid
    with ir.InsertionPoint(if_op.else_block):
        scf_d.YieldOp(_lower_tile(ctx, graph, region_values, owned, rest))
    return list(if_op.results)


def _emit_sequential(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int | ir.Value],
    names: list[str],
    level: int = 0,
) -> None:
    """One ``scf.for`` per grid dim carrying the full written tensors."""
    if level == len(grid_ids):
        ctx.lower_graph(graph)
        return
    for_op = scf_d.ForOp(
        ctx.index_const(0),
        ctx.as_index(trips[level]),
        ctx.index_const(1),
        iter_args=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(for_op.body):
        _bind_grid_iv(ctx, grid_ids[level], bounds[level], for_op.induction_variable)
        for name, arg in zip(names, for_op.inner_iter_args, strict=True):
            ctx.tensors.rebind(name, arg)
        _emit_sequential(ctx, graph, grid_ids, bounds, trips, names, level + 1)
        scf_d.YieldOp([ctx.tensors.value(name) for name in names])
    for name, result in zip(names, for_op.results, strict=True):
        ctx.tensors.rebind(name, result)


def _bind_grid_iv(
    ctx: BuildContext, block_id: int, bound: LoopBounds, trip_iv: ir.Value
) -> None:
    ctx.block_id_to_trip_iv[block_id] = trip_iv
    ctx.bind_loop(
        block_id,
        _tile_offset(trip_iv, bound.begin, bound.step),
        (bound.begin, bound.end),
    )


def _tile_offset(trip_iv: ir.Value, begin: int | ir.Value, step: int) -> ir.Value:
    """Absolute tile offset ``begin + trip_iv * step`` for a normalized loop IV."""
    if isinstance(begin, int) and begin == 0 and step == 1:
        return trip_iv
    scaled = ir.AffineExpr.get_mul(
        ir.AffineDimExpr.get(0), ir.AffineConstantExpr.get(step)
    )
    if isinstance(begin, ir.Value):
        expr = ir.AffineExpr.get_add(scaled, ir.AffineSymbolExpr.get(0))
        return affine_d.AffineApplyOp(
            ir.AffineMap.get(1, 1, [expr]), [trip_iv, begin]
        ).result
    expr = ir.AffineExpr.get_add(scaled, ir.AffineConstantExpr.get(begin))
    return affine_d.AffineApplyOp(ir.AffineMap.get(1, 0, [expr]), [trip_iv]).result


def _owned_region(
    ctx: BuildContext, name: str, tensor: ir.Value, owned: dict[int, OwnedDim]
) -> tuple[list[ir.Value], list[emit.Size]]:
    """Offsets and sizes of one iteration's region of host tensor ``name``: the
    real part of its tile along owned dims, everything elsewhere."""
    offsets: list[ir.Value] = []
    sizes: list[emit.Size] = []
    for dim in range(ir.RankedTensorType(tensor.type).rank):
        extent = ctx.sizes.extent(tensor, dim, name)
        owner = owned.get(dim)
        if owner is None:
            offsets.append(ctx.index_const(0))
            sizes.append(extent)
        elif owner.point:
            offsets.append(scalar_tile_value(ctx, owner.block_id, owner.kind))
            sizes.append(1)
        else:
            offset, size, _ = tile_window(ctx, owner.block_id, 0, extent)
            offsets.append(offset)
            sizes.append(size)
    return offsets, sizes


@lowers(tracing_ops._for_loop, tracing_ops._for_loop_step)
def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """Lower a (possibly multi-dimensional) nested ``_for_loop``/``_for_loop_step``
    to one ``scf.for`` per block id.

    ``_for_loop(graph_id, begin, end, args[, step])``: the body graph's
    placeholders are ``args``. Each ``scf.for`` carries the variables the body
    assigns, then the host tensors it writes. Block ids come from the loop's
    ``ForLoopGraphInfo``.
    """
    graph_id, args = node.args[0], node.args[3]
    body = ctx.host_function.device_ir.graphs[graph_id].graph
    block_ids = loop_block_ids(ctx.host_function, node)
    bounds = ctx.geometry.loop_bounds(node, block_ids)
    inputs = [ctx.get_value(arg) for arg in args]
    positions = carried_positions(node, args, body)
    names = list(ctx.effects.writes(graph_id))

    def emit_level(level: int, carried: list[ir.Value]) -> list[ir.Value]:
        if level == len(block_ids):
            return lower_subgraph(ctx, body, with_carried(inputs, positions, carried))
        begin = _resolve_loop_bound(ctx, node, bounds[level].begin)
        end = _resolve_loop_bound(ctx, node, bounds[level].end)
        for_op = scf_d.ForOp(
            ctx.as_index(begin),
            ctx.as_index(end),
            ctx.index_const(bounds[level].step),
            iter_args=[*carried, *(ctx.tensors.value(name) for name in names)],
        )
        count = len(carried)
        block_args = list(for_op.inner_iter_args)
        with (
            ir.InsertionPoint(for_op.body),
            ctx.enter_for_loop(
                block_ids[level],
                for_op.induction_variable,
                (begin, end),
                _enclosing_span(ctx, block_ids[level]),
            ),
        ):
            for name, arg in zip(names, block_args[count:], strict=True):
                ctx.tensors.rebind(name, arg)
            yields = emit_level(level + 1, block_args[:count])
            scf_d.YieldOp([*yields, *(ctx.tensors.value(name) for name in names)])
        results = list(for_op.results)
        for name, result in zip(names, results[count:], strict=True):
            ctx.tensors.rebind(name, result)
        return results[:count]

    return emit.Results(emit_level(0, [inputs[position] for position in positions]))


def _enclosing_span(ctx: BuildContext, block_id: int) -> int | None:
    """The static width of the enclosing tile a loop over one tile's range spans,
    if that tile is full (its loop has no partial tile)."""
    enclosing = ctx.geometry.enclosing_tiles.get(block_id)
    if enclosing is None or enclosing not in ctx.block_id_to_iv:
        return None
    if enclosing in ctx.block_id_to_valid:
        return None
    return ctx.geometry.tile_extent(enclosing)


def _resolve_loop_bound(
    ctx: BuildContext,
    node: torch.fx.Node,
    source: object,
) -> int | ir.Value:
    """Resolve a ``_for_loop`` begin/end (an int or an FX node) to a static int
    or an index value.

    A runtime scalar carrying a size resolves through ``ctx.sizes``, so a loop
    over a tensor's extent shares that extent's value.
    """
    if isinstance(source, int):
        return source
    value = None
    if isinstance(source, torch.fx.Node):
        meta_val = source.meta.get("val")
        if (
            source.target is tracing_ops._get_symnode
            and source.args[0] in ctx.scalars
            and isinstance(meta_val, torch.SymInt)
        ):
            return ctx.sizes.value(meta_val)
        value = ctx.get_value(source)
    if value is None:
        raise NodeLoweringError(
            node,
            reason=f"Unsupported loop bound: {source!r}",
            recovery_hint="Ensure loop bounds are integer constants or scalar tensor values",
        )
    return ctx.cast_to_index(value)
