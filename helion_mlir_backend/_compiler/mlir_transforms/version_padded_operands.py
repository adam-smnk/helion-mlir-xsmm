"""``version_padded_operands``: contractions branched on their runtime-padded
inputs padding nothing, so the tiles of a partial tile but the edge ones read
their source in place."""

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
from .utils import Chain
from .utils import expanded_slice
from .utils import is_linalg
from .utils import misaligned_rows
from .utils import operand_chain
from .utils import payload_ops

# Ops of these dialects have no memory effects.
_PURE_DIALECTS = ("tensor.", "arith.", "affine.")


def _runtime_padding(pad: tensor.PadOp) -> list[ir.Value] | None:
    """The runtime padding amounts of ``pad`` if all others are zero, else ``None``."""
    dynamic = ir.ShapedType.get_dynamic_size()
    if any(
        amount not in (0, dynamic) for amount in [*pad.static_low, *pad.static_high]
    ):
        return None
    return [*pad.low, *pad.high] or None


def _only_read_by(value: ir.Value, chain: Chain) -> ir.OpView | None:
    """The op defining ``value`` if it is only read by the first op of
    ``chain``, in its block."""
    if not isinstance(value, ir.OpResult) or len(list(value.uses)) != 1:
        return None
    op = value.owner.opview
    return op if op.operation.block == chain[0][0].operation.block else None


def _pads_misaligned_rows(pad: tensor.PadOp) -> bool:
    """Whether ``pad`` reads a slice of misaligned rows (see
    ``misaligned_rows``): read in place, they would need a copy anyway."""
    found = expanded_slice(pad.source, ir.RankedTensorType(pad.result.type).shape)
    return found is not None and misaligned_rows(*found[:2])


def _versionable_pad(value: ir.Value, chain: Chain) -> tensor.PadOp | None:
    """The pad of runtime padding and static shape defining ``value``, only
    read by ``chain``, else ``None``."""
    pad = _only_read_by(value, chain)
    if (
        not isinstance(pad, tensor.PadOp)
        or not ir.RankedTensorType(pad.result.type).has_static_shape
        or _runtime_padding(pad) is None
        or _pads_misaligned_rows(pad)
    ):
        return None
    return pad


def _sinkable_branch(value: ir.Value, chain: Chain) -> scf.IfOp | None:
    """The side-effect-free ``scf.if`` defining ``value``, only read by
    ``chain``, with a branch yielding a versionable pad (e.g. tiling's guard of
    a pad's empty slices), else ``None``."""
    branch = _only_read_by(value, chain)
    if (
        not isinstance(branch, scf.IfOp)
        or len(branch.results) != 1
        or len(branch.regions[1].blocks) != 1
    ):
        return None
    padded = False
    for region in branch.regions:
        ops = list(region.blocks[0].operations)
        if not all(op.name.startswith(_PURE_DIALECTS) for op in ops[:-1]):
            return None
        yielded = ops[-1].operands[0]
        if isinstance(yielded, ir.OpResult) and isinstance(
            yielded.owner.opview, tensor.PadOp
        ):
            if _pads_misaligned_rows(yielded.owner.opview):
                return None
            padded = True
    return branch if padded else None


def _clone_chain(chain: Chain, value: ir.Value) -> list[ir.OpView]:
    """Copies of ``chain`` at the insertion point reading ``value``."""
    copies = []
    for op, index in chain:
        copy = op.operation.clone()
        copy.operation.operands[index] = value
        value = copy.results[0]
        copies.append(copy)
    return copies


def _replace(chain: Chain, branch: scf.IfOp) -> None:
    consumer = chain[-1][0]
    for old, new in zip(consumer.results, branch.results, strict=True):
        old.replace_all_uses_with(new)
    for op, _ in reversed(chain):
        op.operation.erase()


def _sink_into_branch(chain: Chain, branch: scf.IfOp) -> list[ir.OpView]:
    """Move ``branch``, read by ``chain``, to its consumer and ``chain`` into
    each of its branches; the consumer's copies."""
    consumer = chain[-1][0]
    with ir.InsertionPoint(consumer), consumer.location:
        sunk = scf.IfOp(
            branch.condition, [r.type for r in consumer.results], has_else=True
        )
    copies = []
    for region, block in zip(
        branch.regions, (sunk.then_block, sunk.else_block), strict=True
    ):
        ops = list(region.blocks[0].operations)
        with ir.InsertionPoint(block), consumer.location:
            cloned = _clone_chain(chain, ops[-1].operands[0])
            for op in ops[:-1]:
                op.move_before(cloned[0])
            scf.YieldOp(list(cloned[-1].results))
        copies.append(cloned[-1])
    _replace(chain, sunk)
    branch.operation.erase()
    return copies


def _version_on_pad(chain: Chain, pad: tensor.PadOp) -> list[ir.OpView]:
    """Branch ``chain``'s consumer on ``pad``, read by ``chain``, padding
    nothing at runtime: then it reads the pad's source in place; the copies."""
    consumer = chain[-1][0]
    with ir.InsertionPoint(consumer), consumer.location:
        zero = arith.ConstantOp(ir.IndexType.get(), 0).result
        unpadded = None
        for amount in _runtime_padding(pad):
            is_zero = arith.CmpIOp(arith.CmpIPredicate.eq, amount, zero).result
            unpadded = (
                is_zero if unpadded is None else arith.AndIOp(unpadded, is_zero).result
            )
        branch = scf.IfOp(unpadded, [r.type for r in consumer.results], has_else=True)
    with ir.InsertionPoint(branch.then_block), consumer.location:
        # The source has the pad's shape when it pads nothing.
        source = tensor.CastOp(pad.result.type, pad.source).result
        in_place = _clone_chain(chain, source)
        scf.YieldOp(list(in_place[-1].results))
    with ir.InsertionPoint(branch.else_block), consumer.location:
        padded = _clone_chain(chain, pad.result)
        pad.operation.move_before(padded[0])
        scf.YieldOp(list(padded[-1].results))
    _replace(chain, branch)
    return [in_place[-1], padded[-1]]


def _version_operands(consumer: ir.OpView, done: frozenset[int] = frozenset()) -> None:
    """Version ``consumer`` on each runtime-padded input not in ``done`` (see
    ``_version_on_pad``), first sunk into the branches defining it, so each
    reads its pad directly."""
    for index in range(len(consumer.operands) - len(consumer.results)):
        if index in done:
            continue
        chain, value = operand_chain(consumer, index)
        branch = _sinkable_branch(value, chain)
        if branch is not None:
            for copy in _sink_into_branch(chain, branch):
                _version_operands(copy, done)
            return
        pad = _versionable_pad(value, chain)
        if pad is not None:
            for copy in _version_on_pad(chain, pad):
                _version_operands(copy, done | {index})
            return


@transform_op(modifies_payload=True)
class VersionPaddedOperandsOp(
    HelionTransformDialect.Operation, name="version_padded_operands"
):
    """Branch every contraction in the target on each of its runtime-padded
    inputs padding nothing: tiles of a partial tile but the edge ones read the
    source in place, and only the edge tiles are padded."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "VersionPaddedOperandsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        contractions = [
            found
            for found in payload_ops(state, op.target, ir.OpView)
            if is_linalg(found)
            and linalg.isa_contraction_op(found)
            and len(found.results) == 1
        ]
        for contraction in contractions:
            _version_operands(contraction)
        return DiagnosedSilenceableFailure.Success


def version_padded_operands(target: ir.Value) -> VersionPaddedOperandsOp:
    """snake_case wrapper to create a VersionPaddedOperandsOp."""
    return VersionPaddedOperandsOp(target=target)
