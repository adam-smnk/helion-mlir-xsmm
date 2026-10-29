"""``carry_loop_inits``: loop-carried tensors updated in place, so that the
vectorizer's loop hoisting carries them as vectors.

An elementwise update of a loop-carried tensor (e.g. the running sum of a
flash-attention kernel, ``total = total * alpha + p.sum(-1)``) is a
``linalg.generic`` whose init (destination) is a fresh ``tensor.empty``: its
payload never reads the init. Yielded as the next iteration's value, the loop
then carries a tensor that is not updated in place. Vectorized, the update
becomes a ``vector.transfer_write`` into the empty tensor, not into the iter
arg, so the hoisting of loop-carried transfers (``loop_hoisting``) cannot carry
it as a vector, and bufferization fails ("Yield operand ... is not equivalent
to the corresponding iter bbArg").

This makes the iter arg the destination of such a generic. The init is not
read, so the result is the same; the update is in place when nothing else uses
the iter arg after it (else bufferization copies).
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op


def _nested(op: ir.Operation, name: str) -> list[ir.Operation]:
    """The ``name`` ops nested in ``op`` (``op`` included)."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name == name:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    op.walk(visit)
    return found


def _unread_init(generic: ir.Operation) -> int | None:
    """The operand index of the single init of ``generic`` if its payload does
    not read it."""
    if len(generic.results) != 1:
        return None
    body = generic.regions[0].blocks[0]
    init = len(generic.operands) - 1
    if len(body.arguments) != len(generic.operands):
        return None
    if list(body.arguments[init].uses):
        return None
    return init


def _carry(loop: ir.Operation) -> None:
    body = loop.regions[0].blocks[0]
    terminator = body.operations[len(body.operations) - 1]
    # Block arguments: the induction variable, then the iter args.
    iter_args = list(body.arguments)[1:]
    for yielded, iter_arg in zip(terminator.operands, iter_args, strict=True):
        if not isinstance(yielded, ir.OpResult):
            continue
        generic = yielded.owner
        if generic.name != "linalg.generic" or generic.parent != loop:
            continue
        init = _unread_init(generic)
        if init is None or generic.operands[init].type != iter_arg.type:
            continue
        if generic.operands[init] == iter_arg:
            continue
        generic.operands[init] = iter_arg


@transform_op(modifies_payload=True)
class CarryLoopInitsOp(HelionTransformDialect.Operation, name="carry_loop_inits"):
    """In each ``scf.for`` nested in the target ops, make each iter arg the
    destination (init) of the ``linalg.generic`` in the loop body whose single
    result the loop yields for it, when the generic does not read its init
    (e.g. an elementwise update whose init is a ``tensor.empty``) and the
    types match. Other loops and ops are left unchanged.
    """

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "CarryLoopInitsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            for loop in _nested(target.operation, "scf.for"):
                _carry(loop)
        return DiagnosedSilenceableFailure.Success


def carry_loop_inits(target: ir.Value) -> CarryLoopInitsOp:
    """snake_case wrapper to create a CarryLoopInitsOp."""
    return CarryLoopInitsOp(target=target)
