"""Loop-carried updates in place: ``acc = f(acc, x)`` writes into ``acc``.

After inlining, an elementwise update of a loop-carried value is a
``linalg.generic`` with a fresh ``tensor.empty`` destination, so bufferization
allocates a buffer every iteration. Such an update instead takes the loop's
iter arg as its destination and reads it there, as contractions do
(``linalg.matmul outs(acc)``). Since the body then reads its output, the
optimizing pipeline's elementwise fusion keeps the destination. Bufferization
still copies when the old value is read after the update.

An update ``acc = acc + sum(x)`` (also ``*``, ``max``, ``min``) whose partial
reduction is used only there becomes a reduction that starts from ``acc``
instead of the identity, so the partial result needs no buffer either.

An update accumulated into further (``acc = acc * alpha + p @ v``: a contraction
or reduction whose destination is the update) writes into ``acc`` likewise, also
when the update is conditional (in the branches of an ``scf.if``).
"""

from __future__ import annotations

import math

import mlir.ir as ir

_PARALLEL = "#linalg.iterator_type<parallel>"
_IDENTITIES = {
    "arith.addf": 0.0,
    "arith.mulf": 1.0,
    "arith.maximumf": -math.inf,
    "arith.maxnumf": -math.inf,
    "arith.minimumf": math.inf,
    "arith.minnumf": math.inf,
}


def update_carried_values_in_place(module: ir.Module) -> None:
    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            _update_loop(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)


def _update_loop(loop: ir.Operation) -> None:
    body = loop.regions[0].blocks[0]
    iter_args = list(body.arguments)[1:]
    terminator = list(body.operations)[-1]
    for index, iter_arg in enumerate(iter_args):
        producer = _loop_op(terminator.operands[index], loop)
        if producer is not None and producer.name == "linalg.generic":
            value = terminator.operands[index]
            _write_into(producer, value.result_number, iter_arg)
            _start_reduction_from(producer)
        # The update may be accumulated into further, e.g. by a contraction.
        value = terminator.operands[index]
        while (op := _loop_op(value, loop)) is not None and _accumulates(op):
            value = op.operands[len(op.opview.inputs) + value.result_number]
        producer = _loop_op(value, loop)
        if producer is not None and producer.name == "linalg.generic":
            _write_into(producer, value.result_number, iter_arg)
        elif producer is not None and producer.name == "scf.if":
            # A conditional update, e.g. a rescale: each branch writes in place.
            for region in producer.regions:
                branch = list(region.blocks[0].operations)[-1]
                update = _loop_op(branch.operands[value.result_number], producer)
                if update is not None and update.name == "linalg.generic":
                    result = branch.operands[value.result_number].result_number
                    _write_into(update, result, iter_arg)


def _loop_op(value: ir.Value, loop: ir.Operation) -> ir.Operation | None:
    """The op in ``loop``'s body defining ``value``, if any."""
    if not isinstance(value, ir.OpResult):
        return None
    op = value.owner.operation
    return op if op.parent == loop else None


def _accumulates(op: ir.Operation) -> bool:
    """Whether ``op`` combines its inputs into its destinations' values."""
    if op.name in ("linalg.contract", "linalg.matmul", "linalg.batch_matmul"):
        return True
    if op.name != "linalg.generic":
        return False
    iterators = ir.ArrayAttr(op.attributes["iterator_types"])
    return any(str(iterator) != _PARALLEL for iterator in iterators)


def _write_into(generic: ir.Operation, result: int, iter_arg: ir.Value) -> None:
    """Make ``generic``'s result ``result`` write into ``iter_arg`` if one of its
    inputs reads ``iter_arg`` at the output's indices and its destination is unused."""
    if generic.results[result].type != iter_arg.type:
        return
    iterators = ir.ArrayAttr(generic.attributes["iterator_types"])
    if any(str(iterator) != _PARALLEL for iterator in iterators):
        return
    num_inputs = len(generic.opview.inputs)
    maps = list(ir.ArrayAttr(generic.attributes["indexing_maps"]))
    out_map = maps[num_inputs + result]
    block = generic.regions[0].blocks[0]
    destination = block.arguments[num_inputs + result]
    if any(True for _ in destination.uses) or not (
        ir.AffineMapAttr(out_map).value.is_permutation
    ):
        return
    for index, operand in enumerate(list(generic.operands)[:num_inputs]):
        if operand == iter_arg and maps[index] == out_map:
            generic.operands[num_inputs + result] = iter_arg
            block.arguments[index].replace_all_uses_with(destination)
            return


def _start_reduction_from(update: ir.Operation) -> None:
    """Replace ``update`` = ``dest op partial`` by the reduction producing
    ``partial`` with ``dest`` as its init, if that reduction combines with the same
    ``op``, starts from its identity and has no other use."""
    if len(update.results) != 1:
        return
    num_inputs = len(update.opview.inputs)
    combine = _combine_op(update)
    if combine is None:
        return
    operands = list(combine.operands)
    destination = update.regions[0].blocks[0].arguments[num_inputs]
    (partial_arg,) = [operand for operand in operands if operand != destination]
    index = ir.BlockArgument(partial_arg).arg_number
    maps = list(ir.ArrayAttr(update.attributes["indexing_maps"]))
    partial = update.operands[index]
    if maps[index] != maps[num_inputs] or not isinstance(partial, ir.OpResult):
        return
    reduction = partial.owner.operation
    if (
        reduction.name != "linalg.generic"
        or reduction.parent != update.parent
        or len(reduction.results) != 1
        or len(list(partial.uses)) != 1
        or partial.type != update.results[0].type
    ):
        return
    reduction_combine = _combine_op(reduction)
    init = reduction.operands[len(reduction.opview.inputs)]
    if (
        reduction_combine is None
        or reduction_combine.name != combine.name
        or not _fills_with(init, _IDENTITIES[combine.name])
    ):
        return
    reduction.operands[len(reduction.opview.inputs)] = update.operands[num_inputs]
    update.results[0].replace_all_uses_with(partial)
    update.erase()


def _combine_op(generic: ir.Operation) -> ir.Operation | None:
    """The body's only op if it is a known combiner of an input and the output."""
    block = generic.regions[0].blocks[0]
    ops = list(block.operations)
    if len(ops) != 2 or ops[0].name not in _IDENTITIES:
        return None
    op = ops[0].operation
    num_inputs = len(generic.opview.inputs)
    arguments = list(block.arguments)
    output = arguments[num_inputs]
    inputs = arguments[:num_inputs]
    operands = list(op.operands)
    if len(operands) != 2 or output not in operands:
        return None
    if not any(operand in inputs for operand in operands):
        return None
    return op


def _fills_with(value: ir.Value, constant: float) -> bool:
    if (
        not isinstance(value, ir.OpResult)
        or value.owner.operation.name != "linalg.fill"
    ):
        return False
    scalar = value.owner.operation.operands[0]
    if not isinstance(scalar, ir.OpResult):
        return False
    source = scalar.owner.operation
    if source.name != "arith.constant":
        return False
    return ir.FloatAttr(source.attributes["value"]).value == constant
