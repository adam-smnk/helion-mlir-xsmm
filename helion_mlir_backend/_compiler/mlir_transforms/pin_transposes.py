"""``pin_transposes``: lighthouse's register tiling unrolls a transpose with a
narrow inner dim into per-element pieces, read from a temporary of its padded
source tile. A small transpose that only moves a loaded tile is pinned untiled,
so it becomes one vector transpose; an operand pack stays in memory for AMX."""

import math

from lighthouse.dialects.transform.transform_ext.utils.tile_size_analysis import (
    TILE_SIZES_ATTR_NAME,
)
from mlir import ir
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import MAX_VECTOR_ELEMENTS
from .utils import is_linalg
from .utils import is_operand_pack
from .utils import is_transpose
from .utils import payload_ops


def _is_small_transpose(op: ir.OpView) -> bool:
    """A static transpose of at most ``MAX_VECTOR_ELEMENTS`` elements with no
    linalg producer or user, which tiling would fuse it with."""
    if not is_transpose(op):
        return False
    if math.prod(ir.ShapedType(op.operands[1].type).shape) > MAX_VECTOR_ELEMENTS:
        return False
    if is_linalg(op.operands[0].owner):
        return False
    return not any(is_linalg(use.owner) for use in op.results[0].uses)


@transform_op(modifies_payload=True)
class PinTransposesOp(HelionTransformDialect.Operation, name="pin_transposes"):
    """Annotate every ``_is_small_transpose`` or ``is_operand_pack`` op in the
    target with zero tile sizes: lighthouse's tiling keeps annotated sizes, so it
    stays one op, a fusion boundary."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "PinTransposesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for linalg_op in payload_ops(
            state, op.target, (linalg.TransposeOp, linalg.GenericOp)
        ):
            if _is_small_transpose(linalg_op) or is_operand_pack(linalg_op):
                maps = linalg.get_indexing_maps(linalg_op)
                n_dims = ir.AffineMapAttr(maps[0]).value.n_dims
                linalg_op.operation.attributes[TILE_SIZES_ATTR_NAME] = (
                    ir.DenseI64ArrayAttr.get([0] * n_dims)
                )
        return DiagnosedSilenceableFailure.Success


def pin_transposes(target: ir.Value) -> PinTransposesOp:
    """snake_case wrapper to create a PinTransposesOp."""
    return PinTransposesOp(target=target)
