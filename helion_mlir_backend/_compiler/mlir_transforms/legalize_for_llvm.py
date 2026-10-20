"""``legalize_for_llvm``: the vector ops upstream's LLVM lowering rejects or gets
wrong become per-element scalar loads and stores: 0-d transfers (lowered only on
memrefs of unit inner stride) and i1 transfers (LLVM packs i1 vectors into bits,
while a memref holds one byte per i1). Contractions with operands narrower than
the accumulator (folded extensions, for x86 dot-product and AMX patterns) that
no x86 pattern took get their operands extended again."""

import itertools

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


def _scalarize(
    transfer: vector.TransferReadOp | vector.TransferWriteOp,
    rewriter: transform.TransformRewriter,
) -> None:
    base_type = transfer.base.type
    if not isinstance(base_type, ir.MemRefType):
        return
    is_read = isinstance(transfer, vector.TransferReadOp)
    vector_type = ir.VectorType(
        transfer.result.type if is_read else transfer.valueToStore.type
    )
    is_bool = vector_type.element_type == ir.IntegerType.get_signless(1)
    minor_identity = ir.AffineMap.get_minor_identity(base_type.rank, vector_type.rank)
    if (vector_type.rank != 0 and not is_bool) or (
        transfer.permutation_map.value != minor_identity
    ):
        return
    in_bounds = [ir.BoolAttr(flag).value for flag in transfer.in_bounds]
    index = ir.IndexType.get()
    leading = base_type.rank - vector_type.rank
    with ir.InsertionPoint(transfer), transfer.location:
        extents = {
            dim: memref.DimOp(
                transfer.base, arith.ConstantOp(index, leading + dim)
            ).result
            for dim, flag in enumerate(in_bounds)
            if not flag
        }
        result = (
            vector.BroadcastOp(vector_type, transfer.padding).result
            if is_read
            else None
        )
        for position in itertools.product(*map(range, vector_type.shape)):
            indices = list(transfer.indices)
            checks = []
            for dim, offset in enumerate(position):
                if offset:
                    step = arith.ConstantOp(index, offset).result
                    indices[leading + dim] = arith.AddIOp(
                        indices[leading + dim], step
                    ).result
                if dim in extents:
                    checks.append(
                        arith.CmpIOp(
                            arith.CmpIPredicate.slt,
                            indices[leading + dim],
                            extents[dim],
                        ).result
                    )
            if transfer.mask is not None:
                checks.append(vector.extract(transfer.mask, [], list(position)))
            guard = None
            for check in checks:
                guard = check if guard is None else arith.AndIOp(guard, check).result
            if is_read:
                element = vector_type.element_type
                if guard is None:
                    scalar = memref.LoadOp(transfer.base, indices).result
                else:
                    branch = scf.IfOp(guard, [element], has_else=True)
                    with ir.InsertionPoint(branch.then_block):
                        scf.YieldOp([memref.LoadOp(transfer.base, indices).result])
                    with ir.InsertionPoint(branch.else_block):
                        scf.YieldOp([transfer.padding])
                    scalar = branch.results[0]
                result = vector.insert(scalar, result, [], list(position))
            else:
                scalar = vector.extract(transfer.valueToStore, [], list(position))
                if guard is None:
                    memref.StoreOp(scalar, transfer.base, indices)
                else:
                    branch = scf.IfOp(guard, [], has_else=False)
                    with ir.InsertionPoint(branch.then_block):
                        memref.StoreOp(scalar, transfer.base, indices)
                        scf.YieldOp([])
    if is_read:
        rewriter.replace_op(transfer, [result])
    else:
        rewriter.erase_op(transfer)


def _extend_contraction(
    contract: vector.ContractionOp, rewriter: transform.TransformRewriter
) -> None:
    acc_type = contract.acc.type
    element = (
        ir.VectorType(acc_type).element_type
        if isinstance(acc_type, ir.VectorType)
        else acc_type
    )
    operand_types = [
        ir.VectorType(value.type) for value in (contract.lhs, contract.rhs)
    ]
    if not isinstance(element, ir.FloatType) or all(
        operand.element_type == element for operand in operand_types
    ):
        return
    if not all(
        isinstance(operand.element_type, ir.FloatType) for operand in operand_types
    ):
        return
    with ir.InsertionPoint(contract), contract.location:
        operands = [
            value
            if operand.element_type == element
            else arith.ExtFOp(ir.VectorType.get(operand.shape, element), value).result
            for value, operand in zip(
                (contract.lhs, contract.rhs), operand_types, strict=True
            )
        ]
        result = vector.ContractionOp(
            contract.result.type,
            *operands,
            contract.acc,
            contract.indexing_maps,
            contract.iterator_types,
            kind=contract.kind,
            fastmath=contract.fastmath,
        ).result
    rewriter.replace_op(contract, [result])


@transform_op(modifies_payload=True)
class LegalizeForLLVMOp(HelionTransformDialect.Operation, name="legalize_for_llvm"):
    """Rewrite every 0-d or i1 minor-identity memref transfer in the target as
    per-element scalar loads or stores, and every floating-point contraction with
    operands narrower than its accumulator as one of extended operands."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "LegalizeForLLVMOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        transfer_types = (vector.TransferReadOp, vector.TransferWriteOp)
        for transfer in payload_ops(state, op.target, transfer_types):
            _scalarize(transfer, rewriter)
        for contract in payload_ops(state, op.target, vector.ContractionOp):
            _extend_contraction(contract, rewriter)
        return DiagnosedSilenceableFailure.Success


def legalize_for_llvm(target: ir.Value) -> LegalizeForLLVMOp:
    """snake_case wrapper to create a LegalizeForLLVMOp."""
    return LegalizeForLLVMOp(target=target)
