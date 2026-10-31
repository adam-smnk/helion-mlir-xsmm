"""``drop_accumulator_read_unit_dims``: a contraction accumulator read whose
permutation map skips unit dims of its buffer, moved onto a rank-reducing
subview without those dims, as a minor-identity read.

Upstream's x86 AMX rewrite of a K loop (``x86.vector_contract_to_amx_dot_product``)
only loads accumulator tiles in place from minor-identity reads. Any other
accumulator read makes it start from zero tiles and add the accumulator from
memory afterwards, with the read's memref taken as the register tile: its two
inner sizes as the tile's rows and columns, the tile at the memref's origin
(its two inner indices replaced). A second K loop accumulating onto the first
one's result reads it straight from the ``M x 1 x N`` buffer (a panel dim of
size 1): ``vector<16x16xf32>`` reads at ``[%c16, %c0, %n]``, map
``(d0, d1, d2) -> (d0, d2)``, once ``simplify_vector_ops`` folds the
rank-reducing slices of its tile into them. The rewrite then stores the result
tiles to a ``1 x N`` stack buffer and adds rows of the buffer at
``[16, %row, 0]``: it writes past a ``1 x N`` stack buffer (stack corruption,
a segfault) or fails an ``operator[]`` assertion.

Moved onto the subview, such a read is minor identity: ``move_offsets_to_subview``
and ``fit_subviews_to_reads`` (``amx_move_offsets``, which runs this first) then
give the register tile its own subview at its origin, as for any other
accumulator tile, and the rewrite loads it in place. The step also runs on
hosts without AMX, where it is harmless: the read is the same read of the same
memory, through a view.

Matched: a ``vector.transfer_read`` of a strided ``memref``, without mask (no
mask operand, not in ``vector.mask``), whose permutation map is no minor
identity but becomes one once the dims of static size 1 that it does not use
are dropped, and whose result is only used as ``vector.contract``
accumulators (directly, or as the init of ``scf.for`` iter args only used
so), and whose indices of the dropped dims are not constants other than 0
(an invalid static subview: the read would be out of bounds). The subview
takes the read's indices of the dropped dims as offsets and
the whole extent of the other dims; the read keeps its other indices,
padding and ``in_bounds`` (the dims it bounds-checks keep their sizes).

Left alone: reads used otherwise (contraction operands, which AMX tile loads
read in place as they are, or anything else), masked reads, maps that permute
dims or skip a dim that is not of static size 1 (unless leading, as a minor
identity allows), broadcast dims, reads of tensors or of memrefs without a
strided layout (e.g. an affine-map layout), and writes.
"""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import int_constant
from .utils import is_accumulator
from .utils import payload_ops


def _dropped_dims(read: vector.TransferReadOp) -> list[int] | None:
    """The unit dims of ``read``'s buffer to drop for a minor-identity map;
    ``None`` if ``read`` is not matched."""
    if not isinstance(read.base.type, ir.MemRefType) or read.mask is not None:
        return None
    if read.operation.parent.name == "vector.mask":
        return None
    base_type = ir.MemRefType(read.base.type)
    try:
        base_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        # No strided layout (e.g. an affine map): no subview to take.
        return None
    permutation = read.permutation_map.value
    rank = base_type.rank
    results = len(permutation.results)
    if permutation == ir.AffineMap.get_minor_identity(rank, results):
        return None
    used = []
    for expr in permutation.results:
        if not isinstance(expr, ir.AffineDimExpr):
            return None
        used.append(expr.position)
    if not used or used != sorted(set(used)):
        return None
    dropped = [
        dim
        for dim in range(used[0] + 1, rank)
        if dim not in used and base_type.shape[dim] == 1
    ]
    kept = [dim for dim in range(rank) if dim not in dropped]
    # Without the dropped dims the map must be a minor identity: the dims it
    # uses are the trailing kept ones.
    if kept[len(kept) - results :] != used:
        return None
    indices = list(read.indices)
    if any(int_constant(indices[dim]) not in (None, 0) for dim in dropped):
        return None
    if not is_accumulator(read.result):
        return None
    return dropped


def _rank_reduced_view(
    base: ir.Value, dropped: list[int], indices: list[ir.Value]
) -> ir.Value:
    """The subview of ``base`` at ``indices`` in the ``dropped`` (unit) dims,
    whole in the others, without the dropped dims."""
    base_type = ir.MemRefType(base.type)
    strides, offset = base_type.get_strides_and_offset()
    dynamic = ir.ShapedType.get_dynamic_size()
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    offsets: list[ir.Value | int] = []
    sizes: list[ir.Value | int] = []
    # As memref.subview infers it: a dim adds nothing at index 0 or with a
    # stride of 0, else a dynamic index or stride makes the offset dynamic.
    view_offset = offset
    for dim, size in enumerate(base_type.shape):
        if dim in dropped:
            constant = int_constant(indices[dim])
            offsets.append(indices[dim] if constant is None else constant)
            sizes.append(1)
            if constant == 0 or strides[dim] == 0:
                continue
            if dynamic_offset in (view_offset, strides[dim]) or constant is None:
                view_offset = dynamic_offset
            else:
                view_offset += constant * strides[dim]
            continue
        offsets.append(0)
        sizes.append(
            memref.dim(base, arith.constant(index, dim)) if size == dynamic else size
        )
    kept = [dim for dim in range(base_type.rank) if dim not in dropped]
    view_type = ir.MemRefType.get(
        [base_type.shape[dim] for dim in kept],
        base_type.element_type,
        ir.StridedLayoutAttr.get(view_offset, [strides[dim] for dim in kept]),
        base_type.memory_space,
    )
    return memref.subview(
        base, offsets, sizes, [1] * base_type.rank, result_type=view_type
    )


def _drop(read: vector.TransferReadOp, dropped: list[int]) -> None:
    indices = list(read.indices)
    with ir.InsertionPoint(read), read.location:
        view = _rank_reduced_view(read.base, dropped, indices)
        kept = [index for dim, index in enumerate(indices) if dim not in dropped]
        vector_rank = len(read.permutation_map.value.results)
        moved = vector.transfer_read(
            read.vector.type,
            view,
            kept,
            ir.AffineMapAttr.get(
                ir.AffineMap.get_minor_identity(len(kept), vector_rank)
            ),
            read.padding,
            read.in_bounds,
        )
    read.result.replace_all_uses_with(moved)
    read.operation.erase()


@transform_op(modifies_payload=True)
class DropAccumulatorReadUnitDimsOp(
    HelionTransformDialect.Operation, name="drop_accumulator_read_unit_dims"
):
    """Move every matched contraction accumulator read in the target onto a
    rank-reducing subview without the unit dims its map skips (see the module
    docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "DropAccumulatorReadUnitDimsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for read in payload_ops(state, op.target, vector.TransferReadOp):
            dropped = _dropped_dims(read)
            if dropped is not None:
                _drop(read, dropped)
        return DiagnosedSilenceableFailure.Success


def drop_accumulator_read_unit_dims(
    target: ir.Value,
) -> DropAccumulatorReadUnitDimsOp:
    """snake_case wrapper to create a DropAccumulatorReadUnitDimsOp."""
    return DropAccumulatorReadUnitDimsOp(target=target)
