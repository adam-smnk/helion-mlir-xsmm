"""Forms of a kernel's tile vectors that upstream's distribution has no path for,
rewritten to ones it has, before :mod:`xegpu_layouts` lays a kernel out. Each keeps the
math, matches one exact shape of IR and leaves anything else untouched; what is left is
for the annotator to lay out or to refuse.

- :func:`hoist_transposes`: a transpose of computed tiles becomes the same math on
  transposed reads. ``vector.transpose`` of anything but a read or a broadcast asserts in
  upstream's lane distribution (``computeDistributedCoordsForMatrixOp``, through the SLM
  path of a layout conversion).
- :func:`split_reads`: a read used in ways that need different reads becomes one read for
  each. A DPAS's B takes a VNNI read, a transposed use a transposed read, and anything
  else a plain one; one tile can't come back from one read in two of these. Also where a
  read can't be right: transposed, or split, from a buffer the kernel stores to; transposed
  with 32-bit elements.
- :func:`expand_elementwise_contractions`: ``sum(a * b)`` over a dim, which vectorizes to a
  ``vector.contract`` with one indexing map for both operands, becomes the product and a
  reduction: no XeGPU op takes the contraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir import ir

if TYPE_CHECKING:
    from collections.abc import Callable

_ELEMENTWISE_PREFIXES = ("arith.", "math.")
_READ = "xegpu.load_nd"
_TRANSPOSE = "vector.transpose"
_BROADCAST = "vector.broadcast"


def _walk(root: ir.Operation) -> list[ir.Operation]:
    ops: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op != root:
            ops.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit, ir.WalkOrder.PRE_ORDER)
    return ops


def _defining(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    return owner.operation if isinstance(owner, ir.OpView) else None


def _vector_shape(value: ir.Value) -> tuple[int, ...] | None:
    if isinstance(value.type, ir.VectorType):
        return tuple(ir.VectorType(value.type).shape)
    return None


def _attributes(op: ir.Operation) -> dict[str, ir.Attribute]:
    return {name: op.attributes[name] for name in op.attributes}


def _sweep_dead(func: ir.Operation) -> None:
    """Arithmetic, transposes and reads nothing uses, users first. One walk, each op once:
    without a live-operation map in the bindings, a wrapper of an erased op dangles, so
    nothing here holds one across an erasure."""
    for op in reversed(_walk(func)):
        if not (
            op.name.startswith(_ELEMENTWISE_PREFIXES) or op.name in (_READ, _TRANSPOSE)
        ):
            continue
        if not any(True for result in op.results for _ in result.uses):
            op.erase()


_ALLOCATIONS = frozenset({"gpu.alloc", "memref.alloc", "memref.alloca"})
_WRITES = {
    "memref.copy": 1,
    "memref.store": 1,
    "vector.store": 1,
    "vector.transfer_write": 1,
    "vector.maskedstore": 0,
}
"""Writes other than XeGPU's, and which operand is the buffer. None reaches a kernel today."""
_READS_BUFFERS = frozenset(
    {
        "xegpu.create_nd_tdesc",
        "gpu.dealloc",
        "memref.dealloc",
        "memref.load",
        "memref.dim",
        "memref.cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.memory_space_cast",
        "memref.reinterpret_cast",
        "memref.reshape",
        "memref.subview",
        "memref.view",
        "memref.transpose",
        "memref.extract_aligned_pointer_as_index",
        "memref.extract_strided_metadata",
    }
)
_VIEWS = frozenset(
    {
        "memref.cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.memory_space_cast",
        "memref.reinterpret_cast",
        "memref.reshape",
        "memref.subview",
        "memref.transpose",
        "memref.view",
        "memref.extract_aligned_pointer_as_index",
        "arith.index_cast",
        "arith.addi",
    }
)
"""What a kernel's IR can reach one buffer through. Arguments that share memory are the
caller's: the driver clones an input overlapping a written tensor, or refuses
(``_reads_earlier_writes``), so a kernel's arguments don't alias within a phase."""


def buffer_root(value: ir.Value) -> ir.Value:
    """The buffer ``value`` is a view of, or a pointer into, in the IR of its kernel. An
    argument of the kernel, or the value tracing stops at: :func:`known_buffer` says which."""
    while True:
        op = _defining(value)
        if op is None or op.name not in _VIEWS:
            return value
        value = op.operands[0]


def known_buffer(root: ir.Value, func: ir.Operation) -> bool:
    """Whether ``root`` is an argument of ``func`` or its own allocation. Anything else, a
    block argument of a loop or an ``scf.if`` or select result, could be any buffer."""
    if isinstance(root, ir.BlockArgument):
        return root.owner.owner == func
    op = _defining(root)
    return op is not None and op.name in _ALLOCATIONS


def stored_buffers(func: ir.Operation) -> tuple[set[ir.Value], bool]:
    """The buffers (:func:`buffer_root`) the kernel ``func`` writes, and whether it may write
    one that isn't traced: a store through an unknown root or a descriptor made elsewhere, or
    an op on a buffer that this module doesn't know to only read it."""
    stored: set[ir.Value] = set()
    unknown = False

    def add(buffer: ir.Value) -> None:
        nonlocal unknown
        root = buffer_root(buffer)
        stored.add(root)
        unknown = unknown or not known_buffer(root, func)

    for op in _walk(func):
        if op.name == "xegpu.store_nd":
            desc = _defining(op.operands[1])
            if desc is None or desc.name != "xegpu.create_nd_tdesc":
                unknown = True
            else:
                add(desc.operands[0])
        elif op.name == "xegpu.store":
            add(op.operands[1])
        elif op.name in _WRITES:
            add(op.operands[_WRITES[op.name]])
        elif op.name not in _READS_BUFFERS and any(
            isinstance(v.type, ir.MemRefType) for v in op.operands
        ):
            unknown = True
    return stored, unknown


# --- Transposes of computed tiles ---------------------------------------------------


def _splat_constant(op: ir.Operation) -> bool:
    if op.name != "arith.constant":
        return False
    value = op.attributes["value"]
    return isinstance(value, ir.DenseElementsAttr) and value.is_splat


def _transposable(value: ir.Value, rank: int) -> bool:
    """Whether the math of ``value`` can run on transposed tiles: every vector in it is
    of ``rank``, and comes from elementwise ops on reads and splat constants."""
    shape = _vector_shape(value)
    if shape is None or len(shape) != rank:
        return False
    op = _defining(value)
    if op is None:
        return False
    if op.name == _READ or _splat_constant(op):
        return True
    if not op.name.startswith(_ELEMENTWISE_PREFIXES) or len(op.results) != 1:
        return False
    vectors = [v for v in op.operands if _vector_shape(v) is not None]
    return bool(vectors) and all(_transposable(v, rank) for v in vectors)


def _permuted_type(value: ir.Value, permutation: list[int]) -> ir.VectorType:
    vector = ir.VectorType(value.type)
    shape = list(vector.shape)
    return ir.VectorType.get([shape[dim] for dim in permutation], vector.element_type)


def _transposed(
    value: ir.Value,
    permutation: list[int],
    attribute: ir.Attribute,
    done: dict[ir.Value, ir.Value],
) -> ir.Value:
    """``value``'s math on transposed tiles, at the insertion point."""
    if value in done:
        return done[value]
    op = _defining(value)
    result_type = _permuted_type(value, permutation)
    if op.name == _READ:
        new = ir.Operation.create(
            _TRANSPOSE,
            results=[result_type],
            operands=[value],
            attributes={"permutation": attribute},
        )
    elif op.name == "arith.constant":
        new = ir.Operation.create(
            "arith.constant",
            results=[result_type],
            attributes={
                "value": ir.DenseElementsAttr.get_splat(
                    result_type,
                    ir.DenseElementsAttr(op.attributes["value"]).get_splat_value(),
                )
            },
        )
    else:
        operands = [
            _transposed(v, permutation, attribute, done)
            if _vector_shape(v) is not None
            else v
            for v in op.operands
        ]
        new = ir.Operation.create(
            op.name,
            results=[result_type],
            operands=operands,
            attributes=_attributes(op),
        )
    done[value] = new.results[0]
    return new.results[0]


def hoist_transposes(func: ir.Operation) -> bool:
    """``transpose(f(read, constant))`` becomes ``f(transpose(read), constant')``. A read's
    or a broadcast's own transpose is left as it is, as is anything else; the chain stays
    for its other users. Whether anything changed."""
    changed = False
    for op in _walk(func):
        if op.name != _TRANSPOSE:
            continue
        source = op.operands[0]
        definition = _defining(source)
        if definition is None or definition.name in (_READ, _BROADCAST):
            continue
        permutation = list(ir.DenseI64ArrayAttr(op.attributes["permutation"]))
        if not _transposable(source, len(permutation)):
            continue
        with ir.InsertionPoint(op):
            new = _transposed(source, permutation, op.attributes["permutation"], {})
        op.results[0].replace_all_uses_with(new)
        op.erase()
        changed = True
    if changed:
        _sweep_dead(func)
    return changed


# --- One read for each use ------------------------------------------------------------


_TRANSPOSED_READ_BITS = 16
"""f32: ``xevm.blockload2d`` refuses the block, with a message about ``tile_width``."""
_Key = tuple[bool, bool]
"""Of a use of a read: through a transpose, and as a DPAS's B."""


def _leaf_uses(
    value: ir.Value, transposed: bool, leaves: list[tuple[ir.Operation, int, _Key]]
) -> None:
    """The uses of a read past its transposes, each with what read it needs."""
    for use in value.uses:
        owner = use.owner
        if owner.name == _TRANSPOSE:
            _leaf_uses(owner.results[0], True, leaves)
        else:
            b = owner.name == "xegpu.dpas" and use.operand_number == 1
            leaves.append((owner.operation, use.operand_number, (transposed, b)))


def _clone_path(
    value: ir.Value, read: ir.Value, target: ir.Value, memo: dict[ir.Value, ir.Value]
) -> ir.Value:
    """``target``'s chain of transposes from ``value``, on ``read``."""
    if target == value:
        return read
    if target in memo:
        return memo[target]
    op = _defining(target)
    source = _clone_path(value, read, op.operands[0], memo)
    new = op.clone(ip=ir.InsertionPoint(op))
    new.operands[0] = source
    memo[target] = new.results[0]
    return new.results[0]


def split_reads(func: ir.Operation, *, refuse: Callable[[str], Exception]) -> bool:
    """A ``load_nd`` whose uses need different reads becomes one ``load_nd`` per need,
    each next to the original. Uses on neither a transpose nor B (a DPAS's A, other
    math) share one read, as before.

    ``refuse`` makes the error for what can't be right on a buffer the kernel stores to, in
    any view of it: a transposed read, or two reads. Those are distributed unlike the store,
    so a subgroup reads tiles another one is storing: wrong rows, some runs and not others,
    with no diagnostic (``a[t] = a[t].T``, 26 of 30 runs at 512x512, block 256). Whether
    anything changed."""
    stored, unknown = stored_buffers(func)
    changed = False
    for op in [o for o in _walk(func) if o.name == _READ]:
        leaves: list[tuple[ir.Operation, int, _Key]] = []
        value = op.results[0]
        _leaf_uses(value, False, leaves)
        keys = sorted({key for _, _, key in leaves})
        if (
            any(transposed for transposed, _ in keys)
            and ir.VectorType(value.type).element_type.width != _TRANSPOSED_READ_BITS
        ):
            raise refuse(
                "a transposed read of 32-bit elements: only 16-bit tiles (bf16, f16) have "
                "one that lowers; transpose a copy in a kernel of its own"
            )
        desc = _defining(op.operands[0])
        root = None
        if desc is not None and desc.name == "xegpu.create_nd_tdesc":
            root = buffer_root(desc.operands[0])
        untraced = root is None or not known_buffer(root, func) or unknown
        if untraced or root in stored:
            if any(transposed for transposed, _ in keys):
                raise refuse(
                    "a transposed read of a buffer the kernel also stores to, or may: it is "
                    "distributed unlike the store, so subgroups would read each other's "
                    "tiles; store to another buffer"
                )
            if len(keys) >= 2:
                raise refuse(
                    "a tile read in ways that need different reads, from a buffer the "
                    "kernel also stores to, or may; store to another buffer"
                )
        if len(keys) < 2:
            continue
        for key in keys[1:]:
            # Next to the read: nothing writes in between.
            new_read = op.clone(ip=ir.InsertionPoint(op))
            memo: dict[ir.Value, ir.Value] = {}
            for owner, number, leaf_key in leaves:
                if leaf_key != key:
                    continue
                operand = owner.operands[number]
                owner.operands[number] = _clone_path(
                    value, new_read.results[0], operand, memo
                )
        changed = True
    if changed:
        _sweep_dead(func)
    return changed


# --- sum(a * b) -----------------------------------------------------------------------


def expand_elementwise_contractions(func: ir.Operation) -> bool:
    """A float ``vector.contract`` of kind add, with one identity indexing map for both
    operands, becomes ``arith.mulf`` and ``vector.multi_reduction <add>`` over the dims the
    result drops. Any other contraction (a matmul's maps differ) is left as it is."""
    changed = False
    for op in _walk(func):
        if op.name != "vector.contract" or len(op.operands) != 3:
            continue
        maps = list(ir.ArrayAttr(op.attributes["indexing_maps"]))
        lhs, rhs, acc = op.operands
        if (
            str(op.attributes["kind"]) != "#vector.kind<add>"
            or maps[0] != maps[1]
            or lhs.type != rhs.type
            or _kept_dims(ir.AffineMapAttr(maps[0]).value)
            != list(range(ir.VectorType(lhs.type).rank))
        ):
            continue
        element = ir.VectorType(lhs.type).element_type
        wide = (
            ir.VectorType(acc.type).element_type
            if isinstance(acc.type, ir.VectorType)
            else acc.type
        )
        if not (isinstance(element, ir.FloatType) and isinstance(wide, ir.FloatType)):
            continue
        if element.width > wide.width:
            continue
        rank = ir.VectorType(lhs.type).rank
        kept = _kept_dims(ir.AffineMapAttr(maps[2]).value)
        if kept is None:
            continue
        reduced = [dim for dim in range(rank) if dim not in kept]
        with ir.InsertionPoint(op):
            if element != wide:
                # What the contraction means by its element types (bf16 operands
                # into an f32 accumulator).
                wide_type = ir.VectorType.get(ir.VectorType(lhs.type).shape, wide)
                lhs, rhs = (
                    ir.Operation.create(
                        "arith.extf", results=[wide_type], operands=[v]
                    ).results[0]
                    for v in (lhs, rhs)
                )
            product = ir.Operation.create(
                "arith.mulf", results=[lhs.type], operands=[lhs, rhs]
            )
            reduction = ir.Operation.create(
                "vector.multi_reduction",
                results=[op.results[0].type],
                operands=[product.results[0], acc],
                attributes={
                    "kind": op.attributes["kind"],
                    "reduction_dims": ir.DenseI64ArrayAttr.get(reduced),
                },
            )
        op.results[0].replace_all_uses_with(reduction.results[0])
        op.erase()
        changed = True
    return changed


def _kept_dims(result_map: ir.AffineMap) -> list[int] | None:
    """The dims an indexing map keeps, if they are plain dims in order."""
    kept: list[int] = []
    for expr in result_map.results:
        if not isinstance(expr, ir.AffineDimExpr):
            return None
        kept.append(expr.position)
    return kept if kept == sorted(kept) else None
