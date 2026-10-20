"""``loop_bounds``: the constant upper bounds of a linalg op's loop ranges, as
params (vector sizes of a masked vectorization)."""

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import linalg_loop_bounds


@transform_op(modifies_payload=False)
class LoopBoundsOp(HelionTransformDialect.Operation, name="loop_bounds"):
    """The constant upper bounds of one linalg op's loop ranges, one param each."""

    target: ext.Operand[transform.AnyOpType]
    bounds: Sequence[ext.Result[transform.AnyParamType]]

    @staticmethod
    def run(
        op: "LoopBoundsOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        bounds = linalg_loop_bounds(targets[0].opview) if len(targets) == 1 else None
        if bounds is None or None in bounds or len(bounds) != len(op.bounds):
            return DiagnosedSilenceableFailure.SilenceableFailure
        i64 = ir.IntegerType.get_signless(64)
        for handle, bound in zip(op.bounds, bounds, strict=True):
            results.set_params(handle, [ir.IntegerAttr.get(i64, bound)])
        return DiagnosedSilenceableFailure.Success


def loop_bounds(target: ir.Value, loops: int) -> list[ir.Value]:
    """snake_case wrapper to create a LoopBoundsOp of an op with ``loops`` loops."""
    bounds = [transform.AnyParamType.get()] * loops
    return list(LoopBoundsOp(target=target, bounds=bounds).bounds)
