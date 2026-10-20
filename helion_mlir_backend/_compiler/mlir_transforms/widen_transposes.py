"""``widen_transposes``: vector transposes decomposed into 2-D transposes of
elements of at least 32 bits, which LLVM lowers to shuffles."""

import math

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects import ub
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import bits
from .utils import payload_ops

# Side of the 2-D transposes of 32-bit elements upstream lowers to shuffles.
_TRANSPOSE_BLOCK = 16


def _transposed(value: ir.Value, permutation: list[int]) -> ir.Value:
    """``value`` transposed by ``permutation``, as 2-D transposes of elements of
    at least 32 bits where possible, which LLVM lowers to shuffles: dims that
    stay adjacent are merged; leading dims kept in place are unrolled; inner
    dims of at most 64 bits kept in place are one wider integer each; a 2-D
    transpose of narrower elements is an interleave of row pairs followed by a
    transpose of the pairs; wide 2-D transposes are split into 16x16 blocks."""
    vector_type = ir.VectorType(value.type)
    shape, rank = list(vector_type.shape), vector_type.rank
    element_type = vector_type.element_type
    element_bits = bits(element_type)
    if permutation == list(range(rank)):
        return value
    result_shape = [shape[dim] for dim in permutation]
    # Runs of source dims kept adjacent and in order, in result order.
    groups: list[list[int]] = []
    for dim in permutation:
        if groups and groups[-1][-1] + 1 == dim:
            groups[-1].append(dim)
        else:
            groups.append([dim])
    if len(groups) < rank:
        order = sorted(range(len(groups)), key=lambda group: groups[group][0])
        merged = [math.prod(shape[dim] for dim in groups[group]) for group in order]
        value = vector.ShapeCastOp(ir.VectorType.get(merged, element_type), value)
        value = _transposed(value.result, [order.index(g) for g in range(len(groups))])
        return vector.ShapeCastOp(
            ir.VectorType.get(result_shape, element_type), value
        ).result
    if rank > 2 and permutation[0] == 0 and shape[0] > 1:
        result: ir.Value = ub.PoisonOp(
            ir.VectorType.get(result_shape, element_type)
        ).result
        for index in range(shape[0]):
            part = vector.ExtractOp(value, [], [index]).result
            part = _transposed(part, [dim - 1 for dim in permutation[1:]])
            result = vector.InsertOp(part, result, [], [index]).result
        return result
    kept, packed = 0, 1
    while (
        kept < rank - 2
        and permutation[rank - 1 - kept] == rank - 1 - kept
        and packed * shape[rank - 1 - kept] * element_bits <= 64
    ):
        packed *= shape[rank - 1 - kept]
        kept += 1
    if packed > 1 and not (packed * element_bits) & (packed * element_bits - 1):
        lead = shape[: rank - kept]
        result_lead = result_shape[: rank - kept]
        wide = ir.IntegerType.get_signless(packed * element_bits)
        value = vector.ShapeCastOp(
            ir.VectorType.get([*lead, packed], element_type), value
        ).result
        value = vector.BitCastOp(ir.VectorType.get([*lead, 1], wide), value).result
        value = vector.ShapeCastOp(ir.VectorType.get(lead, wide), value).result
        value = _transposed(value, permutation[: rank - kept])
        value = vector.ShapeCastOp(
            ir.VectorType.get([*result_lead, 1], wide), value
        ).result
        value = vector.BitCastOp(
            ir.VectorType.get([*result_lead, packed], element_type), value
        ).result
        return vector.ShapeCastOp(
            ir.VectorType.get(result_shape, element_type), value
        ).result
    if rank == 2 and element_bits < 32 and shape[0] > 2 and shape[0] % 2 == 0:
        rows, cols = shape
        # One two-row shuffle per pair, read as wider elements: rows read apart
        # are never concatenated element by element.
        interleave = [row * cols + col for col in range(cols) for row in (0, 1)]
        wide = ir.IntegerType.get_signless(2 * element_bits)
        pairs: ir.Value = ub.PoisonOp(ir.VectorType.get([rows // 2, cols], wide)).result
        for index in range(rows // 2):
            top = vector.ExtractOp(value, [], [2 * index]).result
            bottom = vector.ExtractOp(value, [], [2 * index + 1]).result
            pair = vector.shuffle(top, bottom, interleave)
            pair = vector.BitCastOp(ir.VectorType.get([cols], wide), pair).result
            pairs = vector.InsertOp(pair, pairs, [], [index]).result
        return vector.BitCastOp(
            ir.VectorType.get(result_shape, element_type),
            _transposed(pairs, [1, 0]),
        ).result
    block = _TRANSPOSE_BLOCK
    if (
        rank == 2
        and element_bits >= 32
        and shape != [block, block]
        and not shape[0] % block
        and not shape[1] % block
    ):
        # Upstream lowers 16x16 blocks to shuffles, larger ones to one huge
        # shuffle of the flattened vector.
        result = ub.PoisonOp(ir.VectorType.get(result_shape, element_type)).result
        block_type = ir.VectorType.get([block, block], element_type)
        for row in range(0, shape[0], block):
            for col in range(0, shape[1], block):
                part = vector.ExtractStridedSliceOp(
                    block_type, value, [row, col], [block, block], [1, 1]
                ).result
                part = _transposed(part, [1, 0])
                result = vector.InsertStridedSliceOp(
                    part, result, [col, row], [1, 1]
                ).result
        return result
    return vector.TransposeOp(
        ir.VectorType.get(result_shape, element_type), value, permutation
    ).result


def _widen(
    transpose: vector.TransposeOp, rewriter: transform.TransformRewriter
) -> None:
    """``transpose`` decomposed by ``_transposed``, if that changes it."""
    permutation = list(transpose.permutation)
    with ir.InsertionPoint(transpose), transpose.location:
        value = _transposed(transpose.vector, permutation)
    if (
        isinstance(value, ir.OpResult)
        and isinstance(value.owner.opview, vector.TransposeOp)
        and value.owner.opview.vector == transpose.vector
    ):
        value.owner.erase()
        return
    rewriter.replace_op(transpose, [value])


@transform_op(modifies_payload=True)
class WidenTransposesOp(HelionTransformDialect.Operation, name="widen_transposes"):
    """Decompose every ``vector.transpose`` in the target into 2-D transposes of
    elements of at least 32 bits where possible (see ``_transposed``): e.g. 16x16
    VNNI pairs of bf16 as a 16x16 transpose of i32, which lowers to shuffles."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "WidenTransposesOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for transpose in payload_ops(state, op.target, vector.TransposeOp):
            _widen(transpose, rewriter)
        return DiagnosedSilenceableFailure.Success


def widen_transposes(target: ir.Value) -> WidenTransposesOp:
    """snake_case wrapper to create a WidenTransposesOp."""
    return WidenTransposesOp(target=target)
