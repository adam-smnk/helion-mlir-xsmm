"""``widen_misread_amx_operands``: on an AMX target, a bf16 x bf16 -> f32
``vector.contract`` whose LHS or RHS upstream's AMX rewrite would misread is
moved off AMX: both operands are widened to f32 (``arith.extf``).

Upstream's x86 AMX rewrite (``x86.vector_contract_to_amx_dot_product``) takes
an operand's ``vector.transfer_read`` apart: it loads the tile (or the rows it
packs to VNNI) from the read's memref at the read's indices, row by row along
the memref's two inner dims. It ignores the read's permutation map, its mask
(operand or enclosing ``vector.mask``) and its ``in_bounds`` flags (the
padding of an out-of-bounds read), and ``x86.amx.tile_load`` requires a unit
innermost stride. Any other operand read is loaded as if it were the plain
read: silently wrong values (a transposing or unit-dim-skipping map, a mask,
padding), or a failure to compile (``tile_load requires unit innermost
stride``). The other layout fixes (``normalize_amx_contractions``,
``drop_contraction_read_unit_dims``) make the reads they match plain reads;
this is the safety net for every read they leave.

Matched: a ``vector.contract`` with bf16 LHS and RHS and an f32 accumulator,
on a target with AMX, whose LHS or RHS is not a plain read: the result of a
``vector.transfer_read`` of a memref with a strided layout and a static unit
innermost stride, without mask (no mask operand, not in ``vector.mask``),
with every dim in bounds and a minor-identity permutation map. Both operands
are then widened to f32: the rewrite takes only bf16 operands, so the
contraction lowers on the f32 path (exact products of bf16 values, f32
sums), as on a host without AMX.

Left alone: contractions with only plain operand reads (the rewrite reads
those right), of other element types, and every contraction on a target
without AMX (the rewrite matches no register tile there).
"""

from lighthouse.execution.target import TargetInfo
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import opview
from .utils import payload_ops


def _element_type(value: ir.Value) -> ir.Type | None:
    if not isinstance(value.type, ir.VectorType):
        return None
    return ir.VectorType(value.type).element_type


def _is_plain_read(value: ir.Value) -> bool:
    """Whether ``value`` is a read the AMX rewrite loads right (see the module
    docstring)."""
    if not isinstance(value, ir.OpResult):
        return False
    read = opview(value.owner)
    if not isinstance(read, vector.TransferReadOp):
        return False
    if read.mask is not None or read.operation.parent.name == "vector.mask":
        return False
    if not isinstance(read.base.type, ir.MemRefType):
        return False
    if not all(ir.BoolAttr(flag).value for flag in read.in_bounds):
        return False
    base_type = ir.MemRefType(read.base.type)
    try:
        strides, _ = base_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        return False
    if not strides or strides[-1] != 1:
        return False
    permutation = read.permutation_map.value
    return permutation == ir.AffineMap.get_minor_identity(
        base_type.rank, len(permutation.results)
    )


def _is_misread(op: vector.ContractionOp) -> bool:
    lhs, rhs, acc = op.lhs, op.rhs, op.acc
    bf16 = ir.BF16Type
    if not (
        isinstance(_element_type(lhs), bf16)
        and isinstance(_element_type(rhs), bf16)
        and isinstance(_element_type(acc), ir.F32Type)
    ):
        return False
    return not (_is_plain_read(lhs) and _is_plain_read(rhs))


def _widen(op: vector.ContractionOp) -> None:
    f32 = ir.F32Type.get()
    with ir.InsertionPoint(op), op.location:
        for index in (0, 1):
            value = op.operation.operands[index]
            shape = ir.VectorType(value.type).shape
            op.operation.operands[index] = arith.extf(
                ir.VectorType.get(shape, f32), value
            )


@transform_op(modifies_payload=True)
class WidenMisreadAmxOperandsOp(
    HelionTransformDialect.Operation, name="widen_misread_amx_operands"
):
    """Widen to f32 the operands of every bf16 contraction in the target that
    upstream's AMX rewrite would misread (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "WidenMisreadAmxOperandsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        if not TargetInfo.host().is_supported("amx"):
            return DiagnosedSilenceableFailure.Success
        for contract in payload_ops(state, op.target, vector.ContractionOp):
            if _is_misread(contract):
                _widen(contract)
        return DiagnosedSilenceableFailure.Success


def widen_misread_amx_operands(target: ir.Value) -> WidenMisreadAmxOperandsOp:
    """snake_case wrapper to create a WidenMisreadAmxOperandsOp."""
    return WidenMisreadAmxOperandsOp(target=target)
