"""``align_operand_rows``: contraction inputs read in place from rows not
starting at cache-line multiples copied into buffers of aligned rows, and every
contraction input hoisted out of the loops it does not vary in."""

from collections.abc import Iterator

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import copy_rows
from .utils import expanded_slice
from .utils import is_linalg
from .utils import misaligned_rows
from .utils import operand_chain
from .utils import payload_ops

# Ops without memory effects on tensors, with those nested in them.
_HOISTABLE = (
    "tensor.",
    "arith.",
    "affine.",
    "linalg.",
    "scf.for",
    "scf.if",
    "scf.yield",
)


def _within(op: ir.Operation, ancestor: ir.Operation) -> bool:
    ancestor = ancestor.operation
    while op is not None:
        if op.operation == ancestor:
            return True
        op = op.parent
    return False


def _defined_within(value: ir.Value, ancestor: ir.Operation) -> bool:
    owner = value.owner if isinstance(value, ir.OpResult) else value.owner.owner
    return _within(owner, ancestor)


def _nested_ops(op: ir.Operation) -> Iterator[ir.Operation]:
    yield op
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield from _nested_ops(inner.operation)


def _invariant_ops(
    op: ir.Operation, loop: ir.Operation, ops: list[ir.Operation]
) -> bool:
    """Add ``op``, of ``loop``'s body, and the ops of its body it reads to
    ``ops``, producers first, if none has memory effects or reads the loop's
    induction variable or iteration arguments."""
    if any(op == found for found in ops):
        return True
    for inner in _nested_ops(op):
        if not inner.name.startswith(_HOISTABLE) or any(
            isinstance(value.type, ir.MemRefType) for value in inner.operands
        ):
            return False
        for value in inner.operands:
            if not _defined_within(value, loop) or _defined_within(value, op):
                continue
            if (
                not isinstance(value, ir.OpResult)
                or value.owner.operation.parent.operation != loop.operation
                or not _invariant_ops(value.owner.operation, loop, ops)
            ):
                return False
    ops.append(op)
    return True


def _hoist(value: ir.Value) -> None:
    """Move the op defining ``value``, with the ops it reads, out of the
    ``scf.for`` loops it does not vary in (e.g. an operand tile's copy out of
    the loop over the tiles of the other operand)."""
    while isinstance(value, ir.OpResult):
        loop = value.owner.operation.parent
        ops: list[ir.Operation] = []
        if (
            loop is None
            or loop.name != "scf.for"
            or not _invariant_ops(value.owner.operation, loop, ops)
        ):
            return
        for op in ops:
            op.move_before(loop)


def _align_rows(contraction: ir.OpView) -> None:
    """Copy each input of ``contraction`` read in place, through reshapes and
    casts to a static shape, as a slice of misaligned rows into a buffer of
    its own, row by row."""
    for index in range(len(contraction.operands) - len(contraction.results)):
        chain, value = operand_chain(contraction, index)
        if not isinstance(value.type, ir.RankedTensorType):
            continue
        value_type = ir.RankedTensorType(value.type)
        if not value_type.has_static_shape:
            continue
        found = expanded_slice(value, value_type.shape)
        if found is None or not misaligned_rows(*found[:2]):
            continue
        slice_op, shape, user = found
        reader, operand = chain[0] if user is None else (user, 0)
        with ir.InsertionPoint(reader), slice_op.location:
            element_type = value_type.element_type
            sliced = slice_op.result
            static_type = ir.RankedTensorType.get(shape, element_type)
            if sliced.type != static_type:
                sliced = tensor.CastOp(static_type, sliced).result
            empty = tensor.EmptyOp(shape, element_type).result
            copied = copy_rows(sliced, empty, shape)
            if copied.type != slice_op.result.type:
                copied = tensor.CastOp(slice_op.result.type, copied).result
        reader.operation.operands[operand] = copied


@transform_op(modifies_payload=True)
class AlignOperandRowsOp(HelionTransformDialect.Operation, name="align_operand_rows"):
    """Give every contraction input in the target read in place from rows that
    do not start at cache-line multiples (e.g. a bf16 A of odd K) a copy with
    aligned rows (see ``_align_rows``): AMX tile loads of misaligned rows
    touch two cache lines per row. Then hoist every contraction input out of
    the loops it does not vary in (see ``_hoist``), so each copy is reused."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "AlignOperandRowsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for found in payload_ops(state, op.target, ir.OpView):
            if is_linalg(found) and linalg.isa_contraction_op(found):
                _align_rows(found)
                for index in range(len(found.operands) - len(found.results)):
                    _hoist(operand_chain(found, index)[1])
        return DiagnosedSilenceableFailure.Success


def align_operand_rows(target: ir.Value) -> AlignOperandRowsOp:
    """snake_case wrapper to create an AlignOperandRowsOp."""
    return AlignOperandRowsOp(target=target)
