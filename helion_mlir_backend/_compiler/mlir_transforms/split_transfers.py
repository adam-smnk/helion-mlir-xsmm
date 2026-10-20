"""``split_transfers``: tile extents are runtime values, so every tile's transfer
may be out of bounds and lowers to masked accesses. Each is split on an
in-bounds check, so only edge tiles take the masked path. Upstream's
``vector.split_transfer_full_partial`` stages n-D vectors through a stack buffer
with ``vector.type_cast``, which overflows it when the inner vector dim is not a
power of two in bytes, and loops forever on rank-reducing transfers."""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops


def _split(
    transfer: vector.TransferReadOp | vector.TransferWriteOp,
    rewriter: transform.TransformRewriter,
) -> None:
    base_type = transfer.base.type
    if not isinstance(base_type, ir.MemRefType) or transfer.mask is not None:
        return
    is_read = isinstance(transfer, vector.TransferReadOp)
    vector_type = ir.VectorType(
        transfer.result.type if is_read else transfer.valueToStore.type
    )
    rank, vector_rank = base_type.rank, vector_type.rank
    in_bounds = [ir.BoolAttr(flag).value for flag in transfer.in_bounds]
    minor_identity = ir.AffineMap.get_minor_identity(rank, vector_rank)
    if all(in_bounds) or transfer.permutation_map.value != minor_identity:
        return
    index = ir.IndexType.get()
    indices = list(transfer.indices)
    with ir.InsertionPoint(transfer), transfer.location:
        fits = None
        for dim, (size, flag) in enumerate(
            zip(vector_type.shape, in_bounds, strict=True), rank - vector_rank
        ):
            if flag:
                continue
            end = arith.AddIOp(indices[dim], arith.ConstantOp(index, size)).result
            extent = memref.DimOp(transfer.base, arith.ConstantOp(index, dim)).result
            dim_fits = arith.CmpIOp(arith.CmpIPredicate.sle, end, extent).result
            fits = dim_fits if fits is None else arith.AndIOp(fits, dim_fits).result
        branch = scf.IfOp(fits, [vector_type] if is_read else [], has_else=True)
        for block, flags in (
            (branch.then_block, [True] * vector_rank),
            (branch.else_block, in_bounds),
        ):
            with ir.InsertionPoint(block):
                if is_read:
                    read = vector.TransferReadOp(
                        vector_type,
                        transfer.base,
                        indices,
                        minor_identity,
                        transfer.padding,
                        flags,
                    ).result
                    scf.YieldOp([read])
                else:
                    vector.TransferWriteOp(
                        None,
                        transfer.valueToStore,
                        transfer.base,
                        indices,
                        minor_identity,
                        flags,
                    )
                    scf.YieldOp([])
    if is_read:
        rewriter.replace_op(transfer, list(branch.results))
    else:
        rewriter.erase_op(transfer)


@transform_op(modifies_payload=True)
class SplitTransfersOp(HelionTransformDialect.Operation, name="split_transfers"):
    """Guard every possibly out-of-bounds, minor-identity, unmasked memref transfer
    in the target with an in-bounds check: an in-bounds transfer if it passes, the
    original otherwise."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SplitTransfersOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        transfer_types = (vector.TransferReadOp, vector.TransferWriteOp)
        for transfer in payload_ops(state, op.target, transfer_types):
            _split(transfer, rewriter)
        return DiagnosedSilenceableFailure.Success


def split_transfers(target: ir.Value) -> SplitTransfersOp:
    """snake_case wrapper to create a SplitTransfersOp."""
    return SplitTransfersOp(target=target)
