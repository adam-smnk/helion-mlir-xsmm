"""``match_row_result_stores``: the stores of a reduction kernel's row results.

Lighthouse's reduction class anchors the layout of a reduction kernel on its
first ``xegpu.store_nd``: the 2D tile it computes from the reduced rows (e.g.
softmax). A kernel that stores the reduced rows themselves (a row sum, mean or
maximum; ``keepdim`` or not) stores them with ``xegpu.store`` (scatter): a row
vector is no 2D block, and a ``[rows, 1]`` column is narrower than a 2D block
surface. Those stores have no anchor and the layout propagation cannot place
them. Recognized by property: a store of a value with the reduction result's
shape (the anchor ``vector.multi_reduction``'s result, ``[rows]``) or that
shape with the reduced dims kept as unit dims (``[rows, 1]``).

The schedule anchors them with the reduction's layout (lighthouse's
``infer_xegpu_reduction_params`` + ``compute_sg_layout``): ``[rows]`` stores as
its slice over the reduced dims, ``[rows, 1]`` stores with the reduced dims'
data set to 1.

``infer_xegpu_reduction_tiles``: the tile loads and stores of a reduction
kernel and their per-instruction tiles. Lighthouse anchors no ``inst_data``, so
the reduction's loads take the layout propagation's and lower to small block
reads (one row each). The tiles a reduction kernel loads and stores in full
are those of the reduced source's shape (the anchor's source, ``[rows, cols]``
per workgroup): ``xegpu.load_nd``/``prefetch_nd``/``store_nd`` of that tile
shape. The schedule anchors them with the reduction's ``sg_layout``/``sg_data``
and, as ``inst_data``, the 2D block load tile on loads and the block store tile
on stores, by the elementwise kernels' rules
(``xegpu_elementwise.block_load_tile``/``block_store_tile``) for the subgroup
tile ``sg_data`` (inferred: the largest; ``[8, 16]`` for f32 and lighthouse's
8-row subgroups).
"""

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .xegpu_elementwise import BLOCK_COLS
from .xegpu_elementwise import ELEMENT_BITS
from .xegpu_elementwise import block_load_tile
from .xegpu_elementwise import block_store_tile

_STORES = frozenset({"xegpu.store", "xegpu.store_nd"})


def _stored_shape(op: ir.Operation) -> tuple[int, ...] | None:
    value_type = op.operands[0].type
    if not isinstance(value_type, ir.VectorType):
        return None
    return tuple(ir.VectorType(value_type).shape)


def row_result_stores(
    target: ir.Operation, anchor: ir.Operation
) -> tuple[list[ir.Operation], list[ir.Operation]]:
    """The stores in ``target`` of ``anchor``'s (a ``vector.multi_reduction``)
    result shape and of that shape with unit reduced dims."""
    source = tuple(ir.VectorType(anchor.operands[0].type).shape)
    reduced = set(ir.DenseI64ArrayAttr(anchor.attributes["reduction_dims"]))
    result = tuple(size for dim, size in enumerate(source) if dim not in reduced)
    kept = tuple(1 if dim in reduced else size for dim, size in enumerate(source))
    sliced: list[ir.Operation] = []
    keepdim: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _STORES:
            shape = _stored_shape(op)
            if shape == result:
                sliced.append(op)
            elif shape == kept and kept != source:
                keepdim.append(op)
        return ir.WalkResult.ADVANCE

    target.walk(visit, ir.WalkOrder.PRE_ORDER)
    return sliced, keepdim


@transform_op(modifies_payload=False)
class MatchRowResultStoresOp(
    HelionTransformDialect.Operation, name="match_row_result_stores"
):
    """The row-result stores in ``target`` of the reduction ``anchor`` (one
    ``vector.multi_reduction``): ``sliced``, stores of its result shape;
    ``keepdim``, of that shape with unit reduced dims (see the module
    docstring). With ``require`` set, a silenceable failure when there are
    none."""

    target: ext.Operand[transform.AnyOpType]
    anchor: ext.Operand[transform.AnyOpType]
    sliced: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    keepdim: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    require: ir.IntegerAttr = ext.attribute(default_factory=lambda: _i64(0))

    @staticmethod
    def run(
        op: "MatchRowResultStoresOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        anchors = state.get_payload_ops(op.anchor)
        if len(targets) != 1 or len(anchors) != 1:
            return DiagnosedSilenceableFailure.SilenceableFailure
        anchor = anchors[0].operation
        if anchor.name != "vector.multi_reduction":
            return DiagnosedSilenceableFailure.SilenceableFailure
        sliced, keepdim = row_result_stores(targets[0].operation, anchor)
        if ir.IntegerAttr(op.require).value and not sliced and not keepdim:
            return DiagnosedSilenceableFailure.SilenceableFailure
        results.set_ops(op.sliced, sliced)
        results.set_ops(op.keepdim, keepdim)
        return DiagnosedSilenceableFailure.Success


def match_row_result_stores(
    target: ir.Value, anchor: ir.Value, *, require: bool = False
) -> MatchRowResultStoresOp:
    """snake_case wrapper to create a MatchRowResultStoresOp."""
    return MatchRowResultStoresOp(
        target=target, anchor=anchor, require=_i64(int(require))
    )


# --- The reduced source's tile loads and stores -----------------------------------

_TILE_LOADS = frozenset({"xegpu.load_nd", "xegpu.prefetch_nd"})


def _tdesc_type(op: ir.Operation) -> ir.ShapedType:
    tdesc = next(
        value.type
        for value in op.operands
        if str(value.type).startswith("!xegpu.tensor_desc")
    )
    return ir.ShapedType(tdesc)


def source_tiles(
    target: ir.Operation, anchor: ir.Operation
) -> tuple[list[ir.Operation], list[ir.Operation], list[int]]:
    """The 2D block loads (and prefetches) and stores in ``target`` of tiles of
    ``anchor``'s (a ``vector.multi_reduction``) source shape, and the element
    sizes loaded."""
    source = tuple(ir.VectorType(anchor.operands[0].type).shape)
    loads: list[ir.Operation] = []
    stores: list[ir.Operation] = []
    bits: set[int] = set()

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _TILE_LOADS or op.name == "xegpu.store_nd":
            tile = _tdesc_type(op)
            if tuple(tile.shape) == source:
                if op.name == "xegpu.store_nd":
                    stores.append(op)
                else:
                    loads.append(op)
                    bits.add(tile.element_type.width)
        return ir.WalkResult.ADVANCE

    target.walk(visit, ir.WalkOrder.PRE_ORDER)
    return loads, stores, sorted(bits)


def reduction_tiles(
    sg_data: Sequence[int],
    load_bits: Sequence[int],
    load_tile: Sequence[int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The load and store tiles (``inst_data``) of a 2D reduction's source tiles
    for its subgroup tile ``sg_data`` (``[rows, cols]``), loading
    ``load_bits``-bit elements; a forced ``load_tile`` is checked, a missing one
    inferred (see the module docstring). Raises ``ValueError`` naming the
    violated constraint."""
    if len(sg_data) != 2:
        raise ValueError(
            f"{len(sg_data)}D subgroup tile {list(sg_data)}: load tiles apply to "
            "2D row reductions"
        )
    unsupported = sorted(set(load_bits) - ELEMENT_BITS)
    if unsupported:
        raise ValueError(
            f"{unsupported[0]}-bit elements: 2D block loads of reduction tiles take "
            "16- or 32-bit elements"
        )
    if sg_data[1] % BLOCK_COLS:
        raise ValueError(
            f"the subgroup tile {list(sg_data)} has {sg_data[1]} columns, not a "
            f"multiple of {BLOCK_COLS} (a block load or store row)"
        )
    return block_load_tile(sg_data, load_bits, load_tile), block_store_tile(sg_data)


def _params(values: Sequence[int]) -> list[ir.IntegerAttr]:
    return [_i64(value) for value in values]


def _param_values(state: transform.TransformState, handle: ir.Value) -> list[int]:
    values = []
    for attr in state.get_params(handle):
        if isinstance(attr, ir.ArrayAttr):
            values.extend(ir.IntegerAttr(a).value for a in attr)
        else:
            values.append(ir.IntegerAttr(attr).value)
    return values


@transform_op(modifies_payload=False)
class InferXeGPUReductionTilesOp(
    HelionTransformDialect.Operation, name="infer_xegpu_reduction_tiles"
):
    """The tile loads and stores in ``target`` of the reduction ``anchor`` (one
    ``vector.multi_reduction``) and their ``inst_data`` (see the module
    docstring).

    Operands: ``sg_rows``/``sg_cols``, the reduction's ``sg_data`` (one i64
    each, as ``compute_sg_layout`` returns it); the optional
    ``force_load_tile`` param (one i64 per dim) forces the load tile.

    Results: ``loads`` (``load_nd`` and ``prefetch_nd``) and ``stores``
    (``store_nd``) of the source tile shape; ``load_tile`` and ``store_tile``,
    params holding one i64 per dim. When no block tile fits an inferred layout
    (e.g. columns not a multiple of 16, 8-bit elements), ``loads`` and
    ``stores`` are empty (nothing to anchor: the propagated layouts stay); a
    forced load tile that does not fit is a silenceable failure.
    """

    target: ext.Operand[transform.AnyOpType]
    anchor: ext.Operand[transform.AnyOpType]
    sg_rows: ext.Operand[transform.AnyParamType]
    sg_cols: ext.Operand[transform.AnyParamType]
    force_load_tile: ext.Operand[transform.AnyParamType] | None = None
    loads: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    stores: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    load_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    store_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @staticmethod
    def run(
        op: "InferXeGPUReductionTilesOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        anchors = state.get_payload_ops(op.anchor)
        if len(targets) != 1 or len(anchors) != 1:
            return DiagnosedSilenceableFailure.SilenceableFailure
        anchor = anchors[0].operation
        if anchor.name != "vector.multi_reduction":
            return DiagnosedSilenceableFailure.SilenceableFailure
        sg_data = [
            *_param_values(state, op.sg_rows),
            *_param_values(state, op.sg_cols),
        ]
        forced = (
            _param_values(state, op.force_load_tile)
            if op.force_load_tile is not None
            else None
        )
        loads, stores, bits = source_tiles(targets[0].operation, anchor)
        try:
            load_tile, store_tile = reduction_tiles(sg_data, bits, forced)
        except ValueError as exc:
            if forced is not None:
                op.location.emit_error(f"infer_xegpu_reduction_tiles: {exc}")
                return DiagnosedSilenceableFailure.SilenceableFailure
            # No block tile fits: leave the tiles to the layout propagation.
            loads, stores = [], []
            load_tile = store_tile = (1, BLOCK_COLS)
        results.set_ops(op.loads, loads)
        results.set_ops(op.stores, stores)
        results.set_params(op.load_tile, _params(load_tile))
        results.set_params(op.store_tile, _params(store_tile))
        return DiagnosedSilenceableFailure.Success


def infer_xegpu_reduction_tiles(
    target: ir.Value,
    anchor: ir.Value,
    sg_data: Sequence[ir.Value],
    load_tile: Sequence[int] | None = None,
) -> InferXeGPUReductionTilesOp:
    """snake_case wrapper to create an InferXeGPUReductionTilesOp (``sg_data``:
    the two params ``compute_sg_layout`` returns)."""
    force = None
    if load_tile is not None:
        force = transform.ParamConstantOp(
            transform.AnyParamType.get(), ir.ArrayAttr.get(_params(load_tile))
        ).param
    sg_rows, sg_cols = sg_data
    return InferXeGPUReductionTilesOp(
        target=target,
        anchor=anchor,
        sg_rows=sg_rows,
        sg_cols=sg_cols,
        force_load_tile=force,
    )


def _i64(value: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
