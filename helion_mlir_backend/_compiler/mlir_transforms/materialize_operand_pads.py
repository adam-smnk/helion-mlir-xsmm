"""``materialize_operand_pads``: zero-padded and constant contraction operands
in memory, filled and copied row by row: kernels like AMX load operand tiles
from memory, not from vectors of masked reads."""

import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import copy_rows
from .utils import dim_sizes
from .utils import is_contraction_input
from .utils import is_linalg
from .utils import is_transpose
from .utils import opview
from .utils import pad_value
from .utils import payload_ops


def _feeds_contraction(
    value: ir.Value, packs: bool = True, branches: bool = False
) -> bool:
    """Whether every use of ``value`` reads it, possibly reshaped or, with
    ``packs``, transposed (e.g. packed), with ``branches`` yielded by an
    ``scf.if``, as a contraction input."""
    uses = list(value.uses)
    for use in uses:
        owner = opview(use.owner)
        if isinstance(owner, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)) or (
            packs
            and is_linalg(owner)
            and is_transpose(owner)
            and use.operand_number == 0
        ):
            if not _feeds_contraction(owner.results[0], packs, branches):
                return False
        elif (
            branches
            and isinstance(owner, scf.YieldOp)
            and isinstance(branch := opview(owner.operation.parent), scf.IfOp)
        ):
            result = branch.results[use.operand_number]
            if not _feeds_contraction(result, packs, branches):
                return False
        elif not is_contraction_input(use):
            return False
    return bool(uses)


def _materialize_pad(pad: tensor.PadOp, packed_only: bool) -> None:
    """A static zero-high-padded contraction operand as a filled tensor its
    source is copied into row by row: kernels like AMX load operand tiles from
    memory, not from vectors of masked reads. With ``packed_only``, only one read
    through an operand pack: register tiling fuses one read directly; without,
    also one yielded by a guard (e.g. of a register tile's empty slice)."""
    result_type = ir.RankedTensorType(pad.result.type)
    if (
        list(pad.low)
        or any(pad.static_low)
        or not result_type.has_static_shape
        or not _feeds_contraction(pad.result, branches=not packed_only)
        or (packed_only and _feeds_contraction(pad.result, packs=False))
    ):
        return
    with ir.InsertionPoint(pad), pad.location:
        padding = pad_value(pad)
        if padding is None:
            return
        shape = list(result_type.shape)
        sizes = dim_sizes(pad.source)
        # Rows are the trailing dims past the last padded one, collapsed: a
        # reshape may have left full dims inner to it (e.g. VNNI pairs).
        last = len(shape) - 1
        while last > 0 and sizes[last] == shape[last]:
            last -= 1
        inner = list(range(last + 1, len(shape)))
        groups = [[dim] for dim in range(last + 1)] + [inner]
        source = pad.source
        if len(inner) > 1:
            source_shape = ir.RankedTensorType(source.type).shape
            source = tensor.CollapseShapeOp(
                ir.RankedTensorType.get(
                    [*source_shape[: last + 1], math.prod(shape[last + 1 :])],
                    result_type.element_type,
                ),
                source,
                groups,
            ).result
        rows_shape = (
            [*shape[: last + 1], math.prod(shape[last + 1 :])]
            if len(inner) > 1
            else shape
        )
        empty = tensor.EmptyOp(rows_shape, result_type.element_type).result
        copied = copy_rows(source, empty, dim_sizes(source), padding)
        if len(inner) > 1:
            copied = tensor.ExpandShapeOp(
                result_type, copied, groups, [], static_output_shape=shape
            ).result
    guard = _empty_guard(pad)
    pad.result.replace_all_uses_with(copied)
    pad.operation.erase()
    if guard is not None:
        _inline_else(guard)


def _constant(value: ir.Value) -> ir.Attribute | None:
    if isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, arith.ConstantOp
    ):
        return value.owner.opview.value
    return None


def _yielded_constant(op: tensor.PadOp | tensor.GenerateOp) -> ir.Attribute | None:
    return _constant(list(op.regions[0].blocks[0].operations)[-1].operands[0])


def _empty_guard(pad: tensor.PadOp) -> scf.IfOp | None:
    """The ``scf.if`` yielding ``pad`` unless a size of the slice it pads is
    zero, then a tensor of its padding (tiling's guard of a pad's empty
    slices), else ``None``. Copied row by row (see ``copy_rows``), the pad
    yields the same then: every row is filled."""
    parent = opview(pad.operation.parent)
    uses = list(pad.result.uses)
    if (
        not isinstance(parent, scf.IfOp)
        or len(parent.results) != 1
        or len(uses) != 1
        or not isinstance(opview(uses[0].owner), scf.YieldOp)
        or len(parent.regions[1].blocks) != 1
        or pad.operation.block != parent.regions[1].blocks[0]
    ):
        return None
    generated = list(parent.regions[0].blocks[0].operations)[-1].operands[0]
    padding = _yielded_constant(pad)
    if (
        padding is None
        or not isinstance(generated, ir.OpResult)
        or not isinstance(generated.owner.opview, tensor.GenerateOp)
        or _yielded_constant(generated.owner.opview) != padding
    ):
        return None
    condition = parent.condition
    compare = condition.owner.opview if isinstance(condition, ir.OpResult) else None
    if (
        not isinstance(compare, arith.CmpIOp)
        or ir.IntegerAttr(compare.predicate).value != arith.CmpIPredicate.eq
        or _constant(compare.rhs) != ir.IntegerAttr.get(ir.IndexType.get(), 0)
    ):
        return None
    source = pad.source
    while isinstance(source, ir.OpResult) and isinstance(
        source.owner.opview, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)
    ):
        source = source.owner.opview.src
    if not isinstance(source, ir.OpResult) or not isinstance(
        source.owner.opview, tensor.ExtractSliceOp
    ):
        return None
    if not any(size == compare.lhs for size in source.owner.opview.sizes):
        return None
    return parent


def _inline_else(branch: scf.IfOp) -> None:
    """Replace ``branch`` by its else branch."""
    ops = list(branch.regions[1].blocks[0].operations)
    for op in ops[:-1]:
        op.move_before(branch)
    branch.results[0].replace_all_uses_with(ops[-1].operands[0])
    branch.operation.erase()


def _materialize_generate(generate: tensor.GenerateOp) -> None:
    """A static constant contraction operand (e.g. tiling's all-padding tile)
    filled one row, its trailing dims collapsed, at a time: register tiling
    unrolls a whole fill into one op per vector."""
    result_type = ir.RankedTensorType(generate.result.type)
    rank = result_type.rank
    if (
        not result_type.has_static_shape
        or rank < 2
        or not _feeds_contraction(generate.result)
    ):
        return
    with ir.InsertionPoint(generate), generate.location:
        value = pad_value(generate)
        if value is None:
            return
        shape = list(result_type.shape)
        rows, width = shape[0], math.prod(shape[1:])
        element_type = result_type.element_type
        index_type = ir.IndexType.get()
        empty = tensor.EmptyOp([rows, width], element_type).result
        loop = scf.ForOp(
            arith.ConstantOp(index_type, 0).result,
            arith.ConstantOp(index_type, rows).result,
            arith.ConstantOp(index_type, 1).result,
            [empty],
        )
        with ir.InsertionPoint(loop.body):
            buffer = loop.inner_iter_args[0]
            row = (
                [loop.induction_variable],
                [],
                [],
                [ir.ShapedType.get_dynamic_size(), 0],
                [1, width],
                [1, 1],
            )
            row_type = ir.RankedTensorType.get([width], element_type)
            filled = linalg.fill(
                value, outs=[tensor.extract_slice(row_type, buffer, *row)]
            )
            scf.YieldOp([tensor.insert_slice(filled, buffer, *row)])
        filled = loop.results[0]
        if rank > 2:
            filled = tensor.ExpandShapeOp(
                result_type,
                filled,
                [[0], list(range(1, rank))],
                [],
                static_output_shape=shape,
            ).result
    generate.result.replace_all_uses_with(filled)
    generate.operation.erase()


@transform_op(modifies_payload=True)
class MaterializeOperandPadsOp(
    HelionTransformDialect.Operation, name="materialize_operand_pads"
):
    """Materialize every zero-high-padded contraction operand in the target
    (see ``_materialize_pad``) and, unless ``packed_only``, every constant one
    (see ``_materialize_generate``)."""

    target: ext.Operand[transform.AnyOpType]
    packed_only: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
    )

    @staticmethod
    def run(
        op: "MaterializeOperandPadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        packed_only = bool(ir.IntegerAttr(op.packed_only).value)
        for pad in payload_ops(state, op.target, tensor.PadOp):
            _materialize_pad(pad, packed_only)
        if not packed_only:
            for generate in payload_ops(state, op.target, tensor.GenerateOp):
                _materialize_generate(generate)
        return DiagnosedSilenceableFailure.Success


def materialize_operand_pads(
    target: ir.Value, packed_only: bool = False
) -> MaterializeOperandPadsOp:
    """snake_case wrapper to create a MaterializeOperandPadsOp."""
    return MaterializeOperandPadsOp(
        target=target,
        packed_only=ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(packed_only)
        ),
    )
