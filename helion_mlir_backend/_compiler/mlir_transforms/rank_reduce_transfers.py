"""``rank_reduce_transfers``: vector transfers of a tile from a higher-rank
buffer moved onto a view of the buffer with the tile's rank.

A kernel whose leading tile dims are unit (e.g. block size 1 on a batch dim)
reads and writes its rank-reduced tiles straight from the full buffers after
unit-dim folding and bufferization: a ``vector<256x32xbf16>`` transfer from a
``memref<8x1024x1024xbf16>`` at ``[%b, %m, %k]``, whose permutation map drops
the leading dim. Lighthouse's XeGPU analyses read a tile's global matrix off
the buffer it is transferred from, with the vector's dims, so they misread such
a buffer (its batch size as rows). The transfer is rewritten onto the matrix the
leading indices select, ``memref.collapse_shape`` of a ``memref.subview`` at
``[%b, 0, 0]``: a ``memref<1024x1024xbf16, strided<...>>`` the tile is read
from at ``[%m, %k]``.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

_TRANSFERS = {"vector.transfer_read": 0, "vector.transfer_write": 1}
"""The transfer ops, by the position of their source operand."""
_SEGMENTS = "operandSegmentSizes"


def _matrix_view(source: ir.Value, leading: list[ir.Value]) -> ir.Value | None:
    """The view of the ``memref`` ``source`` at the ``leading`` indices of its
    leading dims, with its other (trailing) dims; ``None`` if it has no
    strided layout."""
    source_type = ir.MemRefType(source.type)
    try:
        source_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        # No strided layout (e.g. an affine map).
        return None
    rank, count = source_type.rank, len(leading)
    shape = source_type.shape
    zero = [0] * (rank - count)
    view = memref.subview(
        source,
        offsets=[*leading, *zero],
        sizes=[1] * count + shape[count:],
        strides=[1] * rank,
    )
    view_type = ir.MemRefType(view.type)
    strides, offset = view_type.get_strides_and_offset()
    collapsed = ir.MemRefType.get(
        shape[count:],
        source_type.element_type,
        ir.StridedLayoutAttr.get(offset, strides[count:]),
        source_type.memory_space,
    )
    groups = [list(range(count + 1)), *[[dim] for dim in range(count + 1, rank)]]
    return memref.CollapseShapeOp(collapsed, view, groups).result


def _rank_reduce(op: ir.Operation) -> bool:
    """Move the transfer ``op`` onto its tile's matrix view if it reads or
    writes a lower-rank vector from a buffer, its permutation map dropping only
    leading dims (unmasked); whether it changed."""
    position = _TRANSFERS[op.name]
    operands = list(op.operands)
    source = operands[position]
    if not isinstance(source.type, ir.MemRefType):
        return False
    segments = list(ir.DenseI32ArrayAttr(op.attributes[_SEGMENTS]))
    if segments[-1]:
        return False
    rank = ir.MemRefType(source.type).rank
    vector = op.results[0] if position == 0 else operands[0]
    vector_rank = ir.VectorType(vector.type).rank
    permutation = ir.AffineMapAttr(op.attributes["permutation_map"]).value
    count = rank - vector_rank
    if count <= 0 or permutation != ir.AffineMap.get_minor_identity(rank, vector_rank):
        return False
    indices = operands[position + 1 : position + 1 + rank]
    with ir.InsertionPoint(op), op.location:
        view = _matrix_view(source, indices[:count])
        if view is None:
            return False
        attributes = {name: op.attributes[name] for name in op.attributes}
        attributes["permutation_map"] = ir.AffineMapAttr.get(
            ir.AffineMap.get_minor_identity(vector_rank, vector_rank)
        )
        segments[position + 1] = vector_rank
        attributes[_SEGMENTS] = ir.DenseI32ArrayAttr.get(segments)
        new_operands = [
            *operands[:position],
            view,
            *indices[count:],
            *operands[position + 1 + rank :],
        ]
        new = ir.Operation.create(
            op.name,
            results=[result.type for result in op.results],
            operands=new_operands,
            attributes=attributes,
        )
    for old, value in zip(op.results, new.results, strict=True):
        old.replace_all_uses_with(value)
    op.erase()
    return True


def _transfers(target: ir.Operation) -> list[ir.Operation]:
    """The vector transfer ops nested in ``target``."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name in _TRANSFERS:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    target.walk(visit)
    return found


@transform_op(modifies_payload=True)
class RankReduceTransfersOp(
    HelionTransformDialect.Operation, name="rank_reduce_transfers"
):
    """Rewrite the unmasked ``vector.transfer_read``/``transfer_write`` ops on
    buffers nested in each target op whose permutation map drops only leading
    dims (a lower-rank vector, e.g. a 2D tile of a 3D buffer) to transfer from
    the ``memref.collapse_shape`` of a ``memref.subview`` at the leading
    indices: the vector's rank, an identity map."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "RankReduceTransfersOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            for transfer in _transfers(target.operation):
                _rank_reduce(transfer)
        return DiagnosedSilenceableFailure.Success


def rank_reduce_transfers(target: ir.Value) -> RankReduceTransfersOp:
    """snake_case wrapper to create a RankReduceTransfersOp."""
    return RankReduceTransfersOp(target=target)
