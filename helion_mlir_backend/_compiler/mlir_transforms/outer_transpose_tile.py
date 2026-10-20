"""``outer_transpose_tile``: the tile size of a transpose's outer loop
(``transpose_tiles``), as a param."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import is_transpose
from .utils import transpose_tiles


@transform_op(modifies_payload=False)
class OuterTransposeTileOp(
    HelionTransformDialect.Operation, name="outer_transpose_tile"
):
    """The ``transpose_tiles`` size of one transpose's outer loop, as a param."""

    target: ext.Operand[transform.AnyOpType]
    size: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @staticmethod
    def run(
        op: "OuterTransposeTileOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        if len(targets) != 1 or not is_transpose(targets[0].opview):
            return DiagnosedSilenceableFailure.SilenceableFailure
        size = transpose_tiles(targets[0].opview).get(0, 1)
        i64 = ir.IntegerType.get_signless(64)
        results.set_params(op.size, [ir.IntegerAttr.get(i64, size)])
        return DiagnosedSilenceableFailure.Success


def outer_transpose_tile(target: ir.Value) -> ir.Value:
    """snake_case wrapper to create an OuterTransposeTileOp."""
    return OuterTransposeTileOp(target=target).size
