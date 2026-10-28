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
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

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


def _i64(value: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
