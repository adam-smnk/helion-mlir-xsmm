"""``split_bf16_loads``: on x86 with AVX, LLVM cannot generate code for a load of
a bf16 vector of 8k+1 elements (k > 0) and aborts ("Do not know how to soft
promote this operator's operand"). Widening the load, it loads the last element
as a scalar bf16 and puts it back in a vector (``scalar_to_vector``), which it
cannot do for a bf16 scalar, promoted for lack of bf16 arithmetic; other tails
are loaded as integers. Such loads come from rows of ragged bf16 tiles (pads,
copies, edge rows), whatever lowered them, so they are split where they all
meet: in the LLVM dialect, as a load of the first 8k elements and one of the
last, put together by a shuffle. Values of these types, stores and masked loads
are fine; an integer load bitcast to such a vector is not (LLVM folds the two
into a bf16 load) and none is emitted."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import llvm
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops

# Elements x86 widens a bf16 vector load by at least: a tail of one is scalar.
_WIDENING = 8


def _split(load: llvm.LoadOp, rewriter: transform.TransformRewriter) -> None:
    result_type = load.result.type
    if not isinstance(result_type, ir.VectorType):
        return
    vector_type = ir.VectorType(result_type)
    width = vector_type.shape[0] if vector_type.rank == 1 else 0
    if (
        not isinstance(vector_type.element_type, ir.BF16Type)
        or vector_type.scalable
        or width <= _WIDENING
        or width % _WIDENING != 1
        or load.volatile_
        or (load.ordering is not None and ir.IntegerAttr(load.ordering).value)
    ):
        return
    bf16 = vector_type.element_type
    head_type = ir.VectorType.get([width - 1], bf16)
    last_type = ir.VectorType.get([1], bf16)
    with ir.InsertionPoint(load), load.location:
        head = llvm.LoadOp(head_type, load.addr, alignment=load.alignment).result
        address = llvm.GEPOp(load.addr.type, load.addr, [], [width - 1], bf16, 0).result
        last = llvm.LoadOp(last_type, address, alignment=2).result
        # The last element as a vector of the head's type, then appended.
        last = llvm.ShuffleVectorOp(
            head_type, last, last, [0] + [-1] * (width - 2)
        ).result
        whole = llvm.ShuffleVectorOp(vector_type, head, last, list(range(width)))
    rewriter.replace_op(load, [whole.result])


@transform_op(modifies_payload=True)
class SplitBF16LoadsOp(HelionTransformDialect.Operation, name="split_bf16_loads"):
    """Split every ``llvm.load`` in the target of a 1-D, non-volatile,
    non-atomic bf16 vector of 8k+1 elements (k > 0) into a load of the first
    8k elements and one of the last, put together by a shuffle."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SplitBF16LoadsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for load in payload_ops(state, op.target, llvm.LoadOp):
            _split(load, rewriter)
        return DiagnosedSilenceableFailure.Success


def split_bf16_loads(target: ir.Value) -> SplitBF16LoadsOp:
    """snake_case wrapper to create a SplitBF16LoadsOp."""
    return SplitBF16LoadsOp(target=target)
