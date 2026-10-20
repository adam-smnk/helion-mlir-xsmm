"""``materialize_copies``: large slice moves as ``linalg.copy`` ops, runtime-shaped
ones copied row by row."""

import math

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import MAX_VECTOR_ELEMENTS
from .utils import copy_rows
from .utils import dim_sizes
from .utils import payload_ops


def _is_empty(value: ir.Value) -> bool:
    return isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, tensor.EmptyOp
    )


def _moved_slice(value: ir.Value) -> ir.Value | None:
    """The slice of another tensor ``value`` is, looking through inserts that fill
    a whole empty tensor (unit dims added), else ``None``."""
    if not isinstance(value, ir.OpResult):
        return None
    op = value.owner.opview
    if isinstance(op, tensor.ExtractSliceOp):
        return value
    if (
        isinstance(op, tensor.InsertSliceOp)
        and _is_empty(op.dest)
        and list(op.static_sizes) == list(ir.RankedTensorType(op.dest.type).shape)
    ):
        return _moved_slice(op.source)
    return None


def _destination_slice(
    insert: tensor.InsertSliceOp, source_type: ir.RankedTensorType
) -> ir.Value:
    return tensor.ExtractSliceOp(
        source_type,
        insert.dest,
        insert.offsets,
        insert.sizes,
        insert.strides,
        static_offsets=insert.static_offsets,
        static_sizes=insert.static_sizes,
        static_strides=insert.static_strides,
    ).result


def _anchor(insert: tensor.InsertSliceOp) -> ir.OpView:
    """Where ops feeding ``insert`` go: before it, or before the ``scf.forall``
    whose ``in_parallel`` region holds it."""
    if isinstance(insert, tensor.ParallelInsertSliceOp):
        return insert.operation.parent
    return insert


def _materialize_copy(insert: tensor.InsertSliceOp) -> None:
    if _is_empty(insert.dest):
        return
    source = _moved_slice(insert.source)
    if source is None:
        return
    source_type = ir.RankedTensorType(source.type)
    sizes = list(insert.static_sizes)
    if not source_type.has_static_shape:
        _materialize_row_copy(insert, source)
        return
    if math.prod(source_type.shape) <= MAX_VECTOR_ELEMENTS or [
        size for size in sizes if size != 1
    ] != [dim for dim in source_type.shape if dim != 1]:
        return
    with ir.InsertionPoint(_anchor(insert)), insert.location:
        destination = _destination_slice(insert, source_type)
        copied = linalg.copy(source, outs=[destination])
    insert.operation.operands[0] = copied


def _materialize_row_copy(insert: tensor.InsertSliceOp, source: ir.Value) -> None:
    """Copy a runtime-shaped ``source`` into the destination slice of
    ``insert`` row by row (see ``copy_rows``): a bufferized ``memref.copy``
    of a strided slice is an element-wise runtime library call."""
    source_type = ir.RankedTensorType(source.type)
    if insert.source != source or source_type.rank != len(insert.static_sizes):
        return
    with ir.InsertionPoint(_anchor(insert)), insert.location:
        destination = _destination_slice(insert, source_type)
        copied = copy_rows(source, destination, dim_sizes(source))
    insert.operation.operands[0] = copied


@transform_op(modifies_payload=True)
class MaterializeCopiesOp(HelionTransformDialect.Operation, name="materialize_copies"):
    """Insert a ``linalg.copy`` into the destination slice of every insert in the
    target of a static slice of more than ``MAX_VECTOR_ELEMENTS`` elements, and
    a row-by-row copy for a runtime-shaped slice."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MaterializeCopiesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        insert_types = (tensor.InsertSliceOp, tensor.ParallelInsertSliceOp)
        for insert in payload_ops(state, op.target, insert_types):
            _materialize_copy(insert)
        return DiagnosedSilenceableFailure.Success


def materialize_copies(target: ir.Value) -> MaterializeCopiesOp:
    """snake_case wrapper to create a MaterializeCopiesOp."""
    return MaterializeCopiesOp(target=target)
