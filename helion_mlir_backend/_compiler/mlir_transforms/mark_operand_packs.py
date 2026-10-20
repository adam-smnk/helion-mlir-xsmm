"""``mark_operand_packs``: operand packs (``is_operand_pack``) marked with
``OPERAND_PACK_ATTR_NAME``, for schedules to match them."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import is_operand_pack
from .utils import payload_ops

OPERAND_PACK_ATTR_NAME = "helion.operand_pack"


@transform_op(modifies_payload=True)
class MarkOperandPacksOp(HelionTransformDialect.Operation, name="mark_operand_packs"):
    """Mark every ``is_operand_pack`` op in the target with
    ``OPERAND_PACK_ATTR_NAME``."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MarkOperandPacksOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for linalg_op in payload_ops(
            state, op.target, (linalg.TransposeOp, linalg.GenericOp)
        ):
            if is_operand_pack(linalg_op):
                linalg_op.operation.attributes[OPERAND_PACK_ATTR_NAME] = (
                    ir.UnitAttr.get()
                )
        return DiagnosedSilenceableFailure.Success


def mark_operand_packs(target: ir.Value) -> MarkOperandPacksOp:
    """snake_case wrapper to create a MarkOperandPacksOp."""
    return MarkOperandPacksOp(target=target)
