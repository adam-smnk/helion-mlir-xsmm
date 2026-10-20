"""``hoist_allocas``: static stack buffers hoisted out of loops, whose lowering
would grow the stack every iteration."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops


@transform_op(modifies_payload=True)
class HoistAllocasOp(HelionTransformDialect.Operation, name="hoist_allocas"):
    """Move every static ``memref.alloca`` in the target out of the loops of its
    ``alloca_scope``, ``omp.parallel`` region or function: in a loop, its
    lowering grows the stack every iteration until the scope ends. It goes to
    the entry of the block holding the outermost such loop, so buffers of
    exclusive ``scf.if`` branches above the loops do not add up."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "HoistAllocasOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        scopes = ("memref.alloca_scope", "omp.parallel", "func.func")
        loops = ("scf.for", "scf.while", "scf.parallel", "scf.forall")
        for alloca in payload_ops(state, op.target, memref.AllocaOp):
            if list(alloca.operands):
                continue
            outermost = None
            parent = alloca.operation.parent
            while parent.name not in scopes:
                if parent.name in loops:
                    outermost = parent
                parent = parent.parent
            if outermost is None:
                entry = parent.regions[0].blocks[0]
            else:
                entry = outermost.block
            alloca.operation.move_before(entry.operations[0])
        return DiagnosedSilenceableFailure.Success


def hoist_allocas(target: ir.Value) -> HoistAllocasOp:
    """snake_case wrapper to create a HoistAllocasOp."""
    return HoistAllocasOp(target=target)
