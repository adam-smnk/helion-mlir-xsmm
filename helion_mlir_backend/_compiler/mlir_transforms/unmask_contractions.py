"""``unmask_contractions``: masked contractions (of a masked vectorization) as
unmasked ones of zeroed operands: x86 contraction patterns would rewrite a
masked one inside its mask."""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops


def _mask_sizes(mask: ir.Value) -> list[ir.Value | int] | None:
    """The per-dim sizes of a ``vector.create_mask``/``constant_mask`` value."""
    if not isinstance(mask, ir.OpResult):
        return None
    op = mask.owner.opview
    if isinstance(op, vector.CreateMaskOp):
        return list(op.operands)
    if isinstance(op, vector.ConstantMaskOp):
        return list(op.mask_dim_sizes)
    return None


def _unmask_contraction(
    masked: vector.MaskOp, rewriter: transform.TransformRewriter
) -> None:
    body = list(masked.maskRegion.blocks[0].operations)
    contract = body[0].opview if len(body) == 2 else None
    sizes = _mask_sizes(masked.mask)
    if (
        not isinstance(contract, vector.ContractionOp)
        or masked.passthru is not None
        or sizes is None
        or str(contract.kind) != "#vector.kind<add>"
        or not all(
            isinstance(ir.VectorType(value.type).element_type, ir.FloatType)
            for value in (contract.lhs, contract.rhs)
        )
    ):
        return
    maps = [ir.AffineMapAttr(affine_map).value for affine_map in contract.indexing_maps]
    if len(sizes) != maps[0].n_dims:
        return
    index = ir.IndexType.get()
    with ir.InsertionPoint(masked), masked.location:
        sizes = [
            size if isinstance(size, ir.Value) else arith.ConstantOp(index, size).result
            for size in sizes
        ]
        operands = []
        for operand, affine_map in zip(
            (contract.lhs, contract.rhs), maps, strict=False
        ):
            dims = [expr.position for expr in affine_map.results]
            vector_type = ir.VectorType(operand.type)
            mask = vector.CreateMaskOp(
                ir.VectorType.get(vector_type.shape, ir.IntegerType.get_signless(1)),
                [sizes[dim] for dim in dims],
            ).result
            zero = arith.ConstantOp(
                vector_type,
                ir.DenseElementsAttr.get_splat(
                    vector_type, ir.FloatAttr.get(vector_type.element_type, 0.0)
                ),
            ).result
            operands.append(arith.SelectOp(mask, operand, zero).result)
        result = vector.ContractionOp(
            contract.result.type,
            *operands,
            contract.acc,
            contract.indexing_maps,
            contract.iterator_types,
            kind=contract.kind,
            fastmath=contract.fastmath,
        ).result
    rewriter.replace_op(masked, [result])


@transform_op(modifies_payload=True)
class UnmaskContractionsOp(
    HelionTransformDialect.Operation, name="unmask_contractions"
):
    """Rewrite every masked floating-point add-contraction in the target as an
    unmasked one of operands zeroed where masked off."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "UnmaskContractionsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for masked in payload_ops(state, op.target, vector.MaskOp):
            _unmask_contraction(masked, rewriter)
        return DiagnosedSilenceableFailure.Success


def unmask_contractions(target: ir.Value) -> UnmaskContractionsOp:
    """snake_case wrapper to create an UnmaskContractionsOp."""
    return UnmaskContractionsOp(target=target)
