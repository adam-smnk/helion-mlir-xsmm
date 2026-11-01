"""``materialize_tile_copies``: a grid loop's store of a tile it loaded
unchanged as a ``linalg.copy`` into the stored slice.

A Helion tile copy (``out[t] = a[t]``, or one that folds to it: ``a[t] * 1.0``,
``a[t].to(a.dtype)``) is a ``tensor.extract_slice`` stored by the grid loop's
``tensor.parallel_insert_slice``, with no compute op between them. The XPU
pipeline lowers a kernel through its linalg ops (vectorization to transfer
reads and writes, then XeGPU loads and stores), so such a kernel had nothing to
vectorize and bufferized to a ``memref.copy`` the XeGPU conversion cannot
lower. As a ``linalg.copy`` (which, unlike an identity ``linalg.generic``,
canonicalization keeps), the tile copy lowers like an elementwise kernel.

Exact pattern: the source of the ``parallel_insert_slice`` is a static-shape
``tensor.extract_slice`` with the inserted slice's type; anything else (e.g. a
rank-changing insert, a runtime-shaped tile) is left alone.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops


def _is_tile_load(value: ir.Value) -> bool:
    return (
        isinstance(value, ir.OpResult)
        and isinstance(value.owner.opview, tensor.ExtractSliceOp)
        and ir.RankedTensorType(value.type).has_static_shape
    )


def materialize_tile_copy(insert: tensor.ParallelInsertSliceOp) -> bool:
    """Store the tile ``insert`` stores through a ``linalg.copy`` into the
    stored slice if its source is a tile load (see the module); whether it
    did."""
    source = insert.source
    if not _is_tile_load(source):
        return False
    in_parallel = insert.operation.parent
    forall = in_parallel.parent
    if not isinstance(forall.opview, scf.ForallOp):
        return False
    source_type = ir.RankedTensorType(source.type)
    sizes = list(insert.static_sizes)
    if sizes != list(source_type.shape):
        return False
    with ir.InsertionPoint(in_parallel), insert.location:
        destination = tensor.ExtractSliceOp(
            source_type,
            insert.dest,
            insert.offsets,
            insert.sizes,
            insert.strides,
            static_offsets=insert.static_offsets,
            static_sizes=insert.static_sizes,
            static_strides=insert.static_strides,
        ).result
        copied = linalg.copy(source, outs=[destination])
    insert.operation.operands[0] = copied
    return True


@transform_op(modifies_payload=True)
class MaterializeTileCopiesOp(
    HelionTransformDialect.Operation, name="materialize_tile_copies"
):
    """:func:`materialize_tile_copy` on every ``tensor.parallel_insert_slice``
    nested in each target op."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MaterializeTileCopiesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for insert in payload_ops(state, op.target, tensor.ParallelInsertSliceOp):
            materialize_tile_copy(insert)
        return DiagnosedSilenceableFailure.Success


def materialize_tile_copies(target: ir.Value) -> MaterializeTileCopiesOp:
    """snake_case wrapper to create a MaterializeTileCopiesOp."""
    return MaterializeTileCopiesOp(target=target)
