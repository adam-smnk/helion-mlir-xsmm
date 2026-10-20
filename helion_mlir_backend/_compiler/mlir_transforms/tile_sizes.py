"""``tile_sizes``: the tile sizes making a linalg op's loop ranges bounded and
its vectors small enough to vectorize (``linalg_tile_sizes``), as a param."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import linalg_tile_sizes


@transform_op(modifies_payload=False)
class TileSizesOp(HelionTransformDialect.Operation, name="tile_sizes"):
    """The ``linalg_tile_sizes`` of one linalg op, as one param."""

    target: ext.Operand[transform.AnyOpType]
    sizes: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @staticmethod
    def run(
        op: "TileSizesOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        sizes = linalg_tile_sizes(targets[0].opview) if len(targets) == 1 else None
        if sizes is None:
            return DiagnosedSilenceableFailure.SilenceableFailure
        i64 = ir.IntegerType.get_signless(64)
        results.set_params(op.sizes, [ir.IntegerAttr.get(i64, s) for s in sizes])
        return DiagnosedSilenceableFailure.Success


def tile_sizes(target: ir.Value) -> ir.Value:
    """snake_case wrapper to create a TileSizesOp."""
    return TileSizesOp(target=target).sizes
