"""``fold_region_inserts``: a grid loop's store of one slice of a region it owns as
a store of that slice alone.

A Helion store at a fixed index (``cache[tb, th, pos, td] = x[tb, th, td]``) lowers
to the whole region the iteration owns (``[1, 1, S, 64]``: tiled dims, the others
whole) loaded, the slice inserted and the region stored back. Lowered as it is, that
reads and writes all of ``S`` rows per workgroup, where other workgroups' regions do
not overlap, and the tile has more dims than a kernel class takes. The iteration
changes only the slice, so storing the slice is the same.

Exact pattern: a ``tensor.parallel_insert_slice`` of a value ``tensor.insert_slice``
makes into a ``tensor.extract_slice`` of the same destination with the same static
sizes and the same offsets and strides, all strides 1, that insert having no other
use. Anything else (several inserts, other users, other offsets) is left alone.
"""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops

_DYNAMIC = ir.ShapedType.get_dynamic_size()


def _offsets(op: object) -> list[object]:
    """``op``'s offsets, each an int or a value."""
    dynamic = iter(op.offsets)
    return [next(dynamic) if o == _DYNAMIC else o for o in op.static_offsets]


def _sum(a: object, b: object, index_type: ir.Type) -> object:
    """``a + b`` of ints and values; a value is built at the insertion point."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    values = [
        arith.constant(index_type, v).result if isinstance(v, int) else v
        for v in (a, b)
    ]
    return arith.addi(*values).result


def fold_region_insert(insert: tensor.ParallelInsertSliceOp) -> bool:
    """Store only the slice ``insert`` stores if it matches (see the module);
    whether it did."""
    value = insert.source
    if not isinstance(value, ir.OpResult):
        return False
    inner = value.owner.opview
    if not isinstance(inner, tensor.InsertSliceOp) or len(list(value.uses)) != 1:
        return False
    region = inner.dest
    if not isinstance(region, ir.OpResult):
        return False
    load = region.owner.opview
    if not isinstance(load, tensor.ExtractSliceOp) or load.source != insert.dest:
        return False
    outer_offsets = _offsets(insert)
    if (
        _offsets(load) != outer_offsets
        or list(load.static_sizes) != list(insert.static_sizes)
        or _DYNAMIC in insert.static_sizes
        or _DYNAMIC in inner.static_sizes
        or any(s != 1 for s in (*insert.static_strides, *inner.static_strides))
        or any(s != 1 for s in load.static_strides)
        or ir.RankedTensorType(region.type).rank != len(outer_offsets)
    ):
        return False
    in_parallel = insert.operation.parent
    if not isinstance(in_parallel.parent.opview, scf.ForallOp):
        return False
    index_type = ir.IndexType.get()
    with ir.InsertionPoint(in_parallel), insert.location:
        offsets = [
            _sum(a, b, index_type)
            for a, b in zip(outer_offsets, _offsets(inner), strict=True)
        ]
    dynamic = [o for o in offsets if not isinstance(o, int)]
    static = [_DYNAMIC if not isinstance(o, int) else o for o in offsets]
    with ir.InsertionPoint(insert), insert.location:
        tensor.ParallelInsertSliceOp(
            inner.source,
            insert.dest,
            dynamic,
            [],
            [],
            static_offsets=static,
            static_sizes=list(inner.static_sizes),
            static_strides=[1] * len(offsets),
        )
    insert.operation.erase()
    if not list(value.uses):
        inner.operation.erase()
    return True


@transform_op(modifies_payload=True)
class FoldRegionInsertsOp(HelionTransformDialect.Operation, name="fold_region_inserts"):
    """:func:`fold_region_insert` on every ``tensor.parallel_insert_slice`` nested in
    each target op."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FoldRegionInsertsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for insert in payload_ops(state, op.target, tensor.ParallelInsertSliceOp):
            fold_region_insert(insert)
        return DiagnosedSilenceableFailure.Success


def fold_region_inserts(target: ir.Value) -> FoldRegionInsertsOp:
    return FoldRegionInsertsOp(target=target)
