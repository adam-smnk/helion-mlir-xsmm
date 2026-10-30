"""``fit_subviews_to_reads``: a subview only read by in-bounds vector reads of
contraction accumulators at constant indices, shrunk to the region those reads
cover.

``move_offsets_to_subview`` re-bases reads on subviews that keep their base's
sizes (they extend past its end). Upstream's x86 AMX rewrite of a K loop
(``x86.vector_contract_to_amx_dot_product``, flat operands, accumulators read
from memory) takes the accumulator read's memref as the register tile: it
stores ``M x N`` result tiles for the memref's two inner sizes, one per
contraction. A ``32x32`` register tile of a ``64x32`` accumulator buffer read
through a ``64x32`` subview gives 8 tile stores for 4 contractions, and the
rewrite indexes past its results (``SmallVector::operator[]`` assertion, or
memory corruption without assertions); it also reads the accumulator rows
from the memref's origin. Fitted, the subview is the ``32x32`` tile.

Matched: a ``memref.subview`` with unit strides, not rank-reducing, whose
every use is the base of a ``vector.transfer_read`` that is in bounds, unmasked
(no mask operand, not in ``vector.mask``) and minor identity, with constant
indices, and only used as ``vector.contract`` accumulators (directly, or as the
init of ``scf.for`` iter args only used so). Its reads are grouped by the
loops they are inits of, or the blocks of the contractions they accumulate
for (one register tile each). Each group gets a subview of its own covering exactly ``[min, max)`` of its reads
per dim (a read covers its indices plus its extent, 1 in dims the vector does
not cover): the minimum is added to the offsets and subtracted from the read
indices, so the tile starts at the subview's origin, as the rewrite assumes.
The subviews keep the strides (an explicit strided layout). Only groups that
are the accumulator tile the rewrite expects are fitted: 16x16 reads (unit dims
aside) tiling a region of 32x32 inner extents without overlap, lighthouse's AMX
register tile (2x2 contractions of 16x16).

Left alone: subviews with other uses (other ops, nested subviews), reads of
anything but accumulators (e.g. contraction operands, whose subviews AMX tile
loads read in place as they are), reads with runtime indices, permuted, masked
or bounds-checked reads, other strides, rank-reducing subviews, a single group
that already fills its subview, subviews some read would exceed (it would be
out of bounds), and subviews with a group of another shape. None of those
reaches the rewrite silently from lighthouse: it tiles AMX contractions 32x32
in registers, unrolled to 16x16 (``strategy_register_parallel`` /
``_unroll``), so its groups are 2x2, or 1x2 / 2x1 for a dim of 16, never
taller or wider. The rewrite pairs contractions along N and leaves 2x1 groups
alone ("Coudn't find the pair"); a 1x2 group (16x32) unfitted keeps its
buffer's sizes, so in a taller buffer the rewrite would store more result
tiles than it has results and assert (the eudsl wheel has assertions), as all
tiles did before this fit: loud, not silent."""

import math

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
from .utils import opview
from .utils import payload_ops


def _constant_index(value: ir.Value) -> int | None:
    if isinstance(value.owner, ir.Block):
        return None
    op = opview(value.owner)
    if not isinstance(op, arith.ConstantOp) or not isinstance(op.value, ir.IntegerAttr):
        return None
    return ir.IntegerAttr(op.value).value


def _is_accumulator(value: ir.Value) -> bool:
    """Whether every use of ``value`` is a ``vector.contract`` accumulator, or
    an ``scf.for`` init of an iter arg whose uses all are."""
    uses = list(value.uses)
    if not uses:
        return False
    for use in uses:
        owner = opview(use.owner)
        if isinstance(owner, vector.ContractionOp) and use.operand_number == 2:
            continue
        if not isinstance(owner, scf.ForOp) or use.operand_number < 3:
            return False
        iter_arg = owner.inner_iter_args[use.operand_number - 3]
        if not _is_accumulator(iter_arg):
            return False
    return True


def _accumulated_by(value: ir.Value) -> frozenset:
    """Where ``value`` accumulates: the loops it is an init of, and the blocks
    of the contractions it is the accumulator of (one register tile each)."""
    keys = set()
    for use in value.uses:
        owner = opview(use.owner)
        if isinstance(owner, vector.ContractionOp):
            keys.add(owner.operation.block)
        else:
            keys.add(owner)
    return frozenset(keys)


def _read_region(use: ir.OpOperand, rank: int) -> tuple[list[int], list[int]] | None:
    """The start and end, per dim, of the region the read ``use`` covers;
    ``None`` if it is no read as matched."""
    read = opview(use.owner)
    if not isinstance(read, vector.TransferReadOp) or use.operand_number != 0:
        return None
    if read.mask is not None or read.operation.parent.name == "vector.mask":
        return None
    if not all(ir.BoolAttr(flag).value for flag in read.in_bounds):
        return None
    shape = ir.VectorType(read.vector.type).shape
    permutation = read.permutation_map.value
    if permutation != ir.AffineMap.get_minor_identity(rank, len(shape)):
        return None
    if not _is_accumulator(read.result):
        return None
    indices = [_constant_index(index) for index in read.indices]
    if len(indices) != rank or any(index is None for index in indices):
        return None
    extents = [1] * (rank - len(shape)) + list(shape)
    return indices, [i + e for i, e in zip(indices, extents, strict=True)]


# The rows and columns of an AMX accumulator tile (f32).
_TILE = 16

# A group of reads to fit one subview to: their uses, the region's start and end.
_Group = tuple[list[ir.OpOperand], list[int], list[int]]


def _is_register_tile(
    reads: list[ir.OpOperand], low: list[int], high: list[int]
) -> bool:
    """Whether ``reads`` tile ``[low, high)`` as the accumulator tile the AMX
    rewrite expects: 16x16 tiles (unit dims aside), not overlapping, of a
    region whose two inner extents are 32x32: lighthouse's 2x2 register tile.
    The rewrite reads the C rows 32 wide (two halves, at columns 0 and 16)
    and pairs the contractions of such a tile; a taller group (e.g. 64x32, 8
    contractions) is no tile it handles."""
    for use in reads:
        shape = ir.VectorType(opview(use.owner).vector.type).shape
        if [size for size in shape if size != 1] != [_TILE, _TILE]:
            return False
    extents = [h - lo for lo, h in zip(low, high, strict=True)]
    return extents[-2:] == [2 * _TILE, 2 * _TILE] and math.prod(
        extents
    ) == _TILE * _TILE * len(reads)


def _fitted_groups(subview: memref.SubViewOp) -> list[_Group] | None:
    """The reads of ``subview`` grouped by the register tile they accumulate
    (``_accumulated_by``), with the region each group covers, if ``subview`` is
    matched, every group is such a tile and some covers less than it."""
    source_type = ir.MemRefType(subview.source.type)
    result_type = ir.MemRefType(subview.result.type)
    rank = result_type.rank
    if rank != source_type.rank or list(subview.static_strides) != [1] * rank:
        return None
    uses = list(subview.result.uses)
    if not uses:
        return None
    groups: dict[frozenset, _Group] = {}
    for use in uses:
        region = _read_region(use, rank)
        if region is None:
            return None
        start, end = region
        key = _accumulated_by(opview(use.owner).result)
        if key not in groups:
            groups[key] = ([], start, end)
        reads, low, high = groups[key]
        reads.append(use)
        groups[key] = (
            reads,
            list(map(min, low, start)),
            list(map(max, high, end)),
        )
    shape = result_type.shape
    dynamic = ir.ShapedType.get_dynamic_size()
    for reads, low, high in groups.values():
        if any(h > d for h, d in zip(high, shape, strict=True) if d != dynamic):
            return None
        if not _is_register_tile(reads, low, high):
            return None
    if len(groups) == 1:
        ((_, low, high),) = groups.values()
        if low == [0] * rank and high == list(shape):
            return None
    return list(groups.values())


def _shifted_offsets(
    subview: memref.SubViewOp, low: list[int]
) -> tuple[list[ir.Value], list[int]]:
    """The subview's dynamic and static offsets, each moved by ``low``."""
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    values = list(subview.offsets)
    dynamic: list[ir.Value] = []
    static: list[int] = []
    for offset, shift in zip(subview.static_offsets, low, strict=True):
        if offset != dynamic_offset:
            static.append(offset + shift)
            continue
        value = values.pop(0)
        if shift:
            value = arith.addi(value, arith.constant(index, shift))
        dynamic.append(value)
        static.append(dynamic_offset)
    return dynamic, static


def _fit(subview: memref.SubViewOp, groups: list[_Group]) -> None:
    result_type = ir.MemRefType(subview.result.type)
    strides, offset = result_type.get_strides_and_offset()
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    for reads, low, high in groups:
        fitted_offset = offset
        if offset != dynamic_offset and dynamic_offset not in strides:
            fitted_offset = offset + sum(
                lo * st for lo, st in zip(low, strides, strict=True)
            )
        elif any(low):
            fitted_offset = dynamic_offset
        fitted_type = ir.MemRefType.get(
            [h - lo for lo, h in zip(low, high, strict=True)],
            result_type.element_type,
            ir.StridedLayoutAttr.get(fitted_offset, strides),
            result_type.memory_space,
        )
        with ir.InsertionPoint(subview), subview.location:
            offsets, static_offsets = _shifted_offsets(subview, low)
            fitted = memref.SubViewOp(
                fitted_type,
                subview.source,
                offsets,
                [],
                list(subview.strides),
                static_offsets,
                [h - lo for lo, h in zip(low, high, strict=True)],
                subview.static_strides,
            )
        for use in reads:
            read = use.owner
            read.operands[0] = fitted.result
            if not any(low):
                continue
            with ir.InsertionPoint(read), read.location:
                for dim, shift in enumerate(low):
                    if shift:
                        old = _constant_index(read.operands[1 + dim])
                        read.operands[1 + dim] = arith.constant(index, old - shift)
    subview.operation.erase()


@transform_op(modifies_payload=True)
class FitSubviewsToReadsOp(
    HelionTransformDialect.Operation, name="fit_subviews_to_reads"
):
    """Shrink every matched ``memref.subview`` in the target to the region its
    reads cover (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FitSubviewsToReadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for subview in payload_ops(state, op.target, memref.SubViewOp):
            groups = _fitted_groups(subview)
            if groups is not None:
                _fit(subview, groups)
        return DiagnosedSilenceableFailure.Success


def fit_subviews_to_reads(target: ir.Value) -> FitSubviewsToReadsOp:
    """snake_case wrapper to create a FitSubviewsToReadsOp."""
    return FitSubviewsToReadsOp(target=target)
