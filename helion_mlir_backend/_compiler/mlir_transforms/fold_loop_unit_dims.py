"""``fold_loop_unit_dims``: loop-carried tensors without the unit dims that
upstream's unit-extent-dim folding drops from the ops computing them.

``linalg``'s unit-dim folding (via reshapes) rewrites each op to its
rank-reduced form between a ``tensor.collapse_shape`` of its operands and a
``tensor.expand_shape`` of its results, but stops at ``scf.for`` iter args:
a tile carried through a loop keeps its unit dims. Two forms occur:

- an accumulator updated in the loop (e.g. the ``[1, m, n]`` tile of a batched
  matmul with batch tile 1, in its K loop): the loop yields the expanded result
  of a folded op;
- a slab of the output written chunk by chunk (e.g. the ``[1, 1, rows, cols]``
  rows of a softmax's last pass): the loop yields a ``tensor.insert_slice`` of
  each chunk into the iter arg.

This rewrites such loops to carry the rank-reduced tensor (the insertions made
rank-reducing too), with reshapes at the loop's boundaries that
canonicalization and the reassociative reshape folding fold away (into the
slices the inits come from and the results go to).
"""

from dataclasses import dataclass

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op


@dataclass(frozen=True)
class _Fold:
    """How one iter arg is carried without its unit dims."""

    reduced_type: ir.RankedTensorType
    reassociation: ir.ArrayAttr
    """Of the carried type's dims, into the reduced type's."""
    insert: ir.OpView | None = None
    """The ``tensor.insert_slice`` into the iter arg the loop yields (a slab),
    ``None`` for an expansion the loop yields (an accumulator)."""


def _unit_expansion(value: ir.Value) -> _Fold | None:
    """The fold of an iter arg ``value`` is yielded as, if it is a
    ``tensor.expand_shape`` that only adds static unit dims (each reassociation
    group has at most one non-unit dim)."""
    if not isinstance(value, ir.OpResult) or value.owner.name != "tensor.expand_shape":
        return None
    expand = value.owner.opview
    result_type = ir.RankedTensorType(expand.result.type)
    if not result_type.has_static_shape:
        return None
    shape = result_type.shape
    for group in ir.ArrayAttr(expand.reassociation):
        dims = [ir.IntegerAttr(dim).value for dim in group]
        if sum(shape[dim] != 1 for dim in dims) > 1:
            return None
    return _Fold(ir.RankedTensorType(expand.src.type), expand.reassociation)


def _unit_groups(shape: list[int]) -> list[list[int]] | None:
    """The reassociation dropping every unit dim of ``shape`` (each joins the
    next non-unit dim, trailing ones the last); ``None`` without unit dims or
    without non-unit dims."""
    groups: list[list[int]] = []
    pending: list[int] = []
    for dim, size in enumerate(shape):
        pending.append(dim)
        if size != 1:
            groups.append(pending)
            pending = []
    if not groups or len(groups) == len(shape):
        return None
    groups[-1].extend(pending)
    return groups


def _non_unit(sizes: list[int]) -> list[int]:
    return [size for size in sizes if size != 1]


def _slab_update(arg: ir.BlockArgument, value: ir.Value) -> _Fold | None:
    """The fold of the iter arg ``arg`` yielded as ``value``, if ``value`` is a
    static ``tensor.insert_slice`` into ``arg`` (its only use) of a source that
    fits it without the unit dims."""
    if not isinstance(value, ir.OpResult) or value.owner.name != "tensor.insert_slice":
        return None
    insert = value.owner.opview
    uses = list(arg.uses)
    if insert.dest != arg or len(uses) != 1 or len(list(value.uses)) != 1:
        return None
    arg_type = ir.RankedTensorType(arg.type)
    if not arg_type.has_static_shape or list(insert.sizes) or list(insert.strides):
        return None
    shape = arg_type.shape
    groups = _unit_groups(shape)
    if groups is None:
        return None
    sizes = list(insert.static_sizes)
    kept = [sizes[dim] for dim, size in enumerate(shape) if size != 1]
    source_shape = ir.RankedTensorType(insert.source.type).shape
    if len(source_shape) > len(kept) or _non_unit(source_shape) != _non_unit(kept):
        return None
    reduced = ir.RankedTensorType.get(_non_unit(shape), arg_type.element_type)
    i64 = ir.IntegerType.get_signless(64)
    reassociation = ir.ArrayAttr.get(
        [ir.ArrayAttr.get([ir.IntegerAttr.get(i64, d) for d in g]) for g in groups]
    )
    return _Fold(reduced, reassociation, insert)


def _reduced_insert(insert: ir.OpView, dest: ir.Value) -> ir.Value:
    """``insert`` (into a tensor with unit dims) into the rank-reduced ``dest``."""
    shape = ir.RankedTensorType(insert.dest.type).shape
    dynamic = ir.ShapedType.get_dynamic_size()
    offsets = iter(insert.offsets)
    static_offsets, dynamic_offsets = [], []
    for dim, offset in enumerate(insert.static_offsets):
        value = next(offsets) if offset == dynamic else None
        if shape[dim] == 1:
            continue
        static_offsets.append(offset)
        if value is not None:
            dynamic_offsets.append(value)
    kept = [dim for dim, size in enumerate(shape) if size != 1]
    sizes, strides = list(insert.static_sizes), list(insert.static_strides)
    return tensor.InsertSliceOp(
        insert.source,
        dest,
        dynamic_offsets,
        [],
        [],
        static_offsets,
        [sizes[dim] for dim in kept],
        [strides[dim] for dim in kept],
    ).result


def _loops(target: ir.Operation) -> list[ir.Operation]:
    """The ``scf.for`` ops nested in ``target``, inner loops first."""
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            found.append(op)
        return ir.WalkResult.ADVANCE

    target.walk(visit, ir.WalkOrder.POST_ORDER)
    return found


def _expand(value: ir.Value, result_type: ir.Type, fold: _Fold) -> ir.Value:
    """``value`` (of the reduced type) expanded back to ``result_type``."""
    shape = ir.RankedTensorType(result_type).shape
    return tensor.ExpandShapeOp(
        result_type, value, fold.reassociation, [], shape
    ).result


def _fold_loop(loop: ir.Operation) -> bool:
    """Carry the iter args of ``loop`` yielded as unit-dim expansions or slab
    updates without their unit dims; whether anything changed."""
    view = loop.opview
    body = view.regions[0].blocks[0]
    yield_op = body.operations[len(body.operations) - 1]
    args = list(body.arguments)[1:]
    folds = [
        _unit_expansion(value) or _slab_update(arg, value)
        for arg, value in zip(args, yield_op.operands, strict=True)
    ]
    if not any(folds):
        return False
    with ir.InsertionPoint(loop), loop.location:
        inits = [
            init
            if fold is None
            else tensor.CollapseShapeOp(
                fold.reduced_type, init, fold.reassociation
            ).result
            for init, fold in zip(view.initArgs, folds, strict=True)
        ]
        new_loop = scf.ForOp(view.lowerBound, view.upperBound, view.step, inits)
    for name in loop.attributes:
        if name not in new_loop.operation.attributes:
            new_loop.operation.attributes[name] = loop.attributes[name]
    new_body = new_loop.regions[0].blocks[0]
    for op in list(body.operations):
        new_body.append(op)
    body.arguments[0].replace_all_uses_with(new_body.arguments[0])
    for index, (old, fold) in enumerate(zip(args, folds, strict=True)):
        new = new_body.arguments[index + 1]
        if fold is None:
            old.replace_all_uses_with(new)
        elif fold.insert is not None:
            with ir.InsertionPoint(fold.insert), loop.location:
                yield_op.operands[index] = _reduced_insert(fold.insert, new)
            fold.insert.operation.erase()
        else:
            with ir.InsertionPoint.at_block_begin(new_body), loop.location:
                old.replace_all_uses_with(_expand(new, old.type, fold))
            yield_op.operands[index] = yield_op.operands[index].owner.operands[0]
    with ir.InsertionPoint.after(new_loop), loop.location:
        for old, new, fold in zip(loop.results, new_loop.results, folds, strict=True):
            old.replace_all_uses_with(
                new if fold is None else _expand(new, old.type, fold)
            )
    loop.erase()
    return True


@transform_op(modifies_payload=True)
class FoldLoopUnitDimsOp(HelionTransformDialect.Operation, name="fold_loop_unit_dims"):
    """In each target op, rewrite the ``scf.for`` loops (inner first) that
    carry a tensor with unit dims, yielded as a ``tensor.expand_shape`` adding
    only unit dims (left by unit-extent-dim folding) or as a static
    ``tensor.insert_slice`` into the iter arg, to carry it without them: the
    init is collapsed before the loop, the result expanded after it, and
    inside the iter arg expanded where it is used, or the insertion made into
    the reduced tensor. Canonicalization and reassociative reshape folding fold
    the resulting reshapes."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FoldLoopUnitDimsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            for loop in _loops(target.operation):
                _fold_loop(loop)
        return DiagnosedSilenceableFailure.Success


def fold_loop_unit_dims(target: ir.Value) -> FoldLoopUnitDimsOp:
    """snake_case wrapper to create a FoldLoopUnitDimsOp."""
    return FoldLoopUnitDimsOp(target=target)
