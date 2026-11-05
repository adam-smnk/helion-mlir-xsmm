"""``fold_slice_unit_dims``: a ``tensor.collapse_shape`` that only drops unit
dims of a ``tensor.extract_slice``'s result folded into a rank-reducing slice.

Helion indexes a size-1 tile of an inner dim (e.g. ``a[tm, ts, tk]`` of a
``[M, S, K/S]`` view, the chunk ``ts`` of a split K) as a slice keeping the
unit dim, ``[bm, 1, bk]``, then collapses it to ``[bm, bk]``. Upstream's
reassociative reshape folding only folds such a collapse into slices at zero
offsets (unpadding), so the tile stays 3D and the XPU kernel classes reject
it. The slice reads the same elements as a rank-reducing slice with the
collapse's type, which replaces the collapse; the original slice is left to
dead-code elimination.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op


def _drops_unit_dims(
    source: ir.RankedTensorType, result: ir.RankedTensorType, groups: ir.ArrayAttr
) -> bool:
    """Whether the collapse of ``source`` into ``result`` by ``groups`` only drops
    static unit dims (each group has at most one dim that is not 1)."""
    if not source.has_static_shape or not result.has_static_shape:
        return False
    shape = source.shape
    for group in groups:
        dims = [ir.IntegerAttr(dim).value for dim in group]
        if sum(shape[dim] != 1 for dim in dims) > 1:
            return False
    return True


def fold_collapse_of_slice(collapse: ir.Operation) -> bool:
    """Replace ``collapse`` (a ``tensor.collapse_shape``) of a unit-stride
    ``tensor.extract_slice`` by a rank-reducing slice of the collapse's type if
    the collapse only drops unit dims; whether it did. Anything else is left
    alone."""
    if collapse.name != "tensor.collapse_shape":
        return False
    source = collapse.operands[0]
    if not isinstance(source, ir.OpResult) or source.owner.name != (
        "tensor.extract_slice"
    ):
        return False
    slice_op = source.owner
    strides = list(ir.DenseI64ArrayAttr(slice_op.attributes["static_strides"]))
    if any(stride != 1 for stride in strides):
        return False
    source_type = ir.RankedTensorType(source.type)
    result_type = ir.RankedTensorType(collapse.results[0].type)
    groups = ir.ArrayAttr(collapse.attributes["reassociation"])
    if not _drops_unit_dims(source_type, result_type, groups):
        return False
    with ir.InsertionPoint(collapse), collapse.location:
        reduced = ir.Operation.create(
            "tensor.extract_slice",
            results=[result_type],
            operands=list(slice_op.operands),
            attributes={
                name: slice_op.attributes[name] for name in slice_op.attributes
            },
        )
    collapse.results[0].replace_all_uses_with(reduced.results[0])
    collapse.erase()
    return True


def _collapses(target: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name == "tensor.collapse_shape":
            found.append(nested)
        return ir.WalkResult.ADVANCE

    target.walk(visit)
    return found


@transform_op(modifies_payload=True)
class FoldSliceUnitDimsOp(
    HelionTransformDialect.Operation, name="fold_slice_unit_dims"
):
    """:func:`fold_collapse_of_slice` on every ``tensor.collapse_shape`` nested
    in each target op."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FoldSliceUnitDimsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            for collapse in _collapses(target.operation):
                fold_collapse_of_slice(collapse)
        return DiagnosedSilenceableFailure.Success


def fold_slice_unit_dims(target: ir.Value) -> FoldSliceUnitDimsOp:
    """snake_case wrapper to create a FoldSliceUnitDimsOp."""
    return FoldSliceUnitDimsOp(target=target)
