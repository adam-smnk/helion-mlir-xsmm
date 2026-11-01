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
from at ``[%m, %k]``. A unit dim inside the tile (e.g. one K chunk of a
``[M, S, K/S]`` view of a split K, read at ``[%m, %s, %k]``) is dropped the same
way, collapsed into the next dim's group: a ``memref<M x K/S, strided<[K, 1],
...>>`` view.
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


def _matrix_view(source: ir.Value, dropped: dict[int, ir.Value]) -> ir.Value | None:
    """The view of the ``memref`` ``source`` at the indices ``dropped`` of the
    dims they key (sizes 1 there, collapsed away), with its other dims whole;
    ``None`` if it has no strided layout or a dynamic shape."""
    source_type = ir.MemRefType(source.type)
    try:
        source_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        # No strided layout (e.g. an affine map).
        return None
    if not source_type.has_static_shape:
        # The subview's type would need dynamic sizes spelled out.
        return None
    rank, shape = source_type.rank, source_type.shape
    kept = [dim for dim in range(rank) if dim not in dropped]
    view = memref.subview(
        source,
        offsets=[dropped.get(dim, 0) for dim in range(rank)],
        sizes=[1 if dim in dropped else shape[dim] for dim in range(rank)],
        strides=[1] * rank,
    )
    view_type = ir.MemRefType(view.type)
    strides, offset = view_type.get_strides_and_offset()
    view_shape = view_type.shape
    # Each dropped (unit) dim joins the next kept dim's group (the last kept
    # dim's, for trailing ones).
    groups: list[list[int]] = []
    pending: list[int] = []
    for dim in range(rank):
        pending.append(dim)
        if dim not in dropped:
            groups.append(pending)
            pending = []
    groups[-1].extend(pending)
    # A view at constant zero leading indices of a contiguous buffer has the
    # identity layout (the subview's type folds to it), and collapse_shape's
    # verifier then expects the identity layout too, not the equal
    # strided<[...]> one. Otherwise the strides are those the verifier infers
    # (:func:`_collapsed_stride`): a kept dim of size 1 in a group with a
    # dropped dim takes the dropped dim's stride.
    layout = (
        None
        if _identity_layout(view_type)
        else ir.StridedLayoutAttr.get(
            offset,
            [_collapsed_stride(group, view_shape, strides) for group in groups],
        )
    )
    collapsed = ir.MemRefType.get(
        [shape[dim] for dim in kept],
        source_type.element_type,
        layout,
        source_type.memory_space,
    )
    return memref.CollapseShapeOp(collapsed, view, groups).result


def _collapsed_stride(group: list[int], shape: list[int], strides: list[int]) -> int:
    """The stride of the dim ``group`` collapses into, as upstream's
    ``computeCollapsedLayoutMap`` infers it: that of the group's last dim whose
    size is not 1 (its first dim if all are); dynamic if that dim's size is
    dynamic in a group of several dims (it may be 1 at run time)."""
    ref = list(group)
    while shape[ref[-1]] == 1 and len(ref) > 1:
        ref.pop()
    if shape[ref[-1]] == ir.ShapedType.get_dynamic_size() and len(ref) > 1:
        return ir.ShapedType.get_dynamic_stride_or_offset()
    return strides[ref[-1]]


def _identity_layout(memref_type: ir.MemRefType) -> bool:
    """Whether ``memref_type`` has the identity layout."""
    layout = memref_type.layout
    return isinstance(layout, ir.AffineMapAttr) and layout.value == (
        ir.AffineMap.get_identity(memref_type.rank)
    )


def _kept_dims(
    permutation: ir.AffineMap, rank: int, vector_rank: int
) -> list[int] | None:
    """The buffer dims ``permutation`` (a transfer's map) keeps, if it is a
    projection keeping ``vector_rank`` dims in order (drops the others), else
    ``None``."""
    if (
        permutation.n_dims != rank
        or len(permutation.results) != vector_rank
        or not permutation.is_projected_permutation
    ):
        return None
    kept = [ir.AffineDimExpr(expr).position for expr in permutation.results]
    if kept != sorted(kept):
        return None
    return kept


def _rank_reduce(op: ir.Operation) -> bool:
    """Move the transfer ``op`` onto its tile's matrix view if it reads or
    writes a lower-rank vector from a buffer, its permutation map dropping
    dims and keeping the others in order (unmasked); whether it changed."""
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
    if rank - vector_rank <= 0:
        return False
    kept = _kept_dims(permutation, rank, vector_rank)
    if kept is None:
        return False
    indices = operands[position + 1 : position + 1 + rank]
    dropped = {dim: indices[dim] for dim in range(rank) if dim not in kept}
    with ir.InsertionPoint(op), op.location:
        view = _matrix_view(source, dropped)
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
            *[indices[dim] for dim in kept],
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
    buffers nested in each target op whose permutation map drops dims and keeps
    the others in order (a lower-rank vector, e.g. a 2D tile of a 3D buffer) to
    transfer from the ``memref.collapse_shape`` of a ``memref.subview`` at the
    dropped dims' indices: the vector's rank, an identity map. Transposing maps
    are left alone."""

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
