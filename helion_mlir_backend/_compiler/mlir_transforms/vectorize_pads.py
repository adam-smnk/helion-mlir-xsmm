"""``vectorize_pads``: bufferized, a pad is a temporary zeroed and then copied
into, and vectorize_all's copy forwarding (upstream
``LinalgCopyVTRForwardingPattern``) reads past the copy with poison padding. As
a vector read of the pad's source with the pad value past its end, written into
an empty tensor, there is no such copy."""

import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import MAX_VECTOR_ELEMENTS
from .utils import dim_bound
from .utils import pad_amount
from .utils import pad_value
from .utils import payload_ops


def _vectorize(pad: tensor.PadOp, rewriter: transform.TransformRewriter) -> None:
    result_type = ir.RankedTensorType(pad.result.type)
    if list(pad.low) or any(pad.static_low):
        return
    rank = result_type.rank
    # A runtime extent is read up to its bound, masked past the tensor's end.
    shape = [dim_bound(pad.result, dim) for dim in range(rank)]
    if None in shape or (
        not result_type.has_static_shape and math.prod(shape) > MAX_VECTOR_ELEMENTS
    ):
        return
    with ir.InsertionPoint(pad), pad.location:
        padding = pad_value(pad)
        if padding is None:
            return
        source_shape = ir.RankedTensorType(pad.source.type).shape
        identity = ir.AffineMap.get_minor_identity(rank, rank)
        index = ir.IndexType.get()
        origin = [arith.ConstantOp(index, 0).result] * rank
        read = vector.TransferReadOp(
            ir.VectorType.get(shape, result_type.element_type),
            pad.source,
            origin,
            identity,
            padding,
            [real == padded for real, padded in zip(source_shape, shape, strict=True)],
        ).result
        sizes: list[int | ir.Value] = list(result_type.shape)
        for dim in range(rank):
            if result_type.is_dynamic_dim(dim):
                size = tensor.DimOp(pad.source, arith.ConstantOp(index, dim)).result
                high = pad_amount(pad, True, dim)
                if high != 0:
                    if isinstance(high, int):
                        high = arith.ConstantOp(index, high).result
                    size = arith.AddIOp(size, high).result
                sizes[dim] = size
        empty = tensor.EmptyOp(sizes, result_type.element_type).result
        written = vector.TransferWriteOp(
            result_type,
            read,
            empty,
            origin,
            identity,
            [not result_type.is_dynamic_dim(dim) for dim in range(rank)],
        ).result
    rewriter.replace_op(pad, [written])


@transform_op(modifies_payload=True)
class VectorizePadsOp(HelionTransformDialect.Operation, name="vectorize_pads"):
    """Vectorize every ``tensor.pad`` in the target with no low padding, a
    constant padding value and evident bounds of its runtime extents."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "VectorizePadsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for pad in payload_ops(state, op.target, tensor.PadOp):
            _vectorize(pad, rewriter)
        return DiagnosedSilenceableFailure.Success


def vectorize_pads(target: ir.Value) -> VectorizePadsOp:
    """snake_case wrapper to create a VectorizePadsOp."""
    return VectorizePadsOp(target=target)
