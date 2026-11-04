"""``gpu.printf`` for each target (:mod:`..mlir.lowering.print_ops` emits it). Nothing here
changes a module without a print.

- **CPU:** a libc ``printf`` and ``fflush`` per line. libc locks the stream per call, so
  lines stay whole under OpenMP, in no fixed order; the flush keeps a line ahead of a
  later crash and of Python's own output.
- **XPU:** upstream's ``gpu-to-llvm-spv`` lowers the op, and IGC and the driver deliver
  it. The guard makes it one line per workgroup, not one per lane (:func:`guard_prints`);
  it orders no threads (:func:`prints_of_stored_buffers`).
"""

from __future__ import annotations

import warnings

from itertools import starmap

from mlir import ir
from mlir.dialects import arith as arith_d
from mlir.dialects import llvm as llvm_d
from mlir.dialects import scf as scf_d

from ..mlir.support import UnsupportedOperationError


def warn_on_each_compile(message: str) -> None:
    """A ``UserWarning`` for this compile. ``warnings.warn`` with Python's default filter shows
    one text once per location, so a later compile from here would stay silent; a registry
    of its own for each call leaves that to the user's filters."""
    warnings.warn_explicit(message, UserWarning, __file__, 0)


def warn_print_is_experimental(entry: str) -> None:
    """Once per compile of a kernel with a print, whatever its number of prints."""
    warn_on_each_compile(
        f"print in kernel {entry} (hl.device_print) is experimental: see "
        "'Printing in a kernel' in the helion-mlir-backend README."
    )


def _ops_named(root: ir.Operation, names: frozenset[str]) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in names:
            found.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def _prints(module: ir.Module) -> list[ir.Operation]:
    return _ops_named(module.operation, frozenset({"gpu.printf"}))


def _c_string(text: str) -> str:
    """``text`` as an MLIR string attribute's body, NUL-terminated."""
    return "".join(f"\\{byte:02X}" for byte in text.encode()) + "\\00"


def lower_prints_to_libc(module: ir.Module) -> int:
    """Each ``gpu.printf`` as ``printf`` and ``fflush``; the number rewritten. Variadic
    floats are doubles."""
    prints = _prints(module)
    if not prints:
        return 0
    ptr = ir.Type.parse("!llvm.ptr")
    i32 = ir.IntegerType.get_signless(32)
    printf = ir.FlatSymbolRefAttr.get("printf")
    fflush = ir.FlatSymbolRefAttr.get("fflush")
    printf_type = ir.TypeAttr.get(ir.Type.parse("!llvm.func<i32 (ptr, ...)>"))
    start = ir.InsertionPoint.at_block_begin(module.body)
    start.insert(ir.Operation.parse("llvm.func @printf(!llvm.ptr, ...) -> i32"))
    start.insert(ir.Operation.parse("llvm.func @fflush(!llvm.ptr) -> i32"))
    strings: dict[str, str] = {}
    for op in prints:
        text = op.attributes["format"].value
        if text not in strings:
            strings[text] = f"__helion_print_{len(strings)}"
            start.insert(
                ir.Operation.parse(
                    f"llvm.mlir.global private constant @{strings[text]}"
                    f'("{_c_string(text)}") {{addr_space = 0 : i32}}'
                )
            )
        with ir.InsertionPoint(op):
            symbol = ir.FlatSymbolRefAttr.get(strings[text])
            operands = [llvm_d.AddressOfOp(ptr, symbol).result]
            for value in op.operands:
                if isinstance(value.type, ir.FloatType) and value.type.width < 64:
                    value = arith_d.ExtFOp(ir.F64Type.get(), value).result
                operands.append(value)
            llvm_d.CallOp(
                i32, operands, [], [], var_callee_type=printf_type, callee=printf
            )
            llvm_d.CallOp(i32, [llvm_d.ZeroOp(ptr).result], [], [], callee=fflush)
        op.erase()
    return len(prints)


def guard_prints(module: ir.Module) -> int:
    """Each ``gpu.printf`` in ``if (gpu.thread_id x == 0)``; the number guarded.

    Every lane of every hardware thread of a workgroup runs the kernel, and a workgroup is
    ``[threads * lanes, 1, 1]`` here, so this is its first lane: else 512 lines per print
    at 32 threads of SIMD16. The region also gets the op through upstream's
    ``convert-parallel-loops-to-gpu``, which gives up on a direct child with a write
    effect on no named value (agent-wiki/upstream-repro/print-parallel-to-launch)."""
    prints = _prints(module)
    index = ir.IndexType.get()
    for op in prints:
        with ir.InsertionPoint(op):
            thread = ir.Operation.create(
                "gpu.thread_id",
                results=[index],
                attributes={"dimension": ir.Attribute.parse("#gpu<dim<x>>")},
            ).result
            zero = arith_d.ConstantOp(index, ir.IntegerAttr.get(index, 0)).result
            first = arith_d.CmpIOp(arith_d.CmpIPredicate.eq, thread, zero).result
            guard = scf_d.IfOp(first, [], has_else=False)
        with ir.InsertionPoint(guard.then_block):
            terminator = scf_d.YieldOp([])
        op.move_before(terminator)
    return len(prints)


def refuse_prints_needing_allocation(module: ir.Module) -> None:
    """A kernel with a print and a heap buffer: a clear error, in place of upstream's later
    failure on ``mgpuMemAlloc``. Reading a tile that is only computed in the kernel, or a
    partial tile, which is padded, makes bufferization allocate it, and a kernel cannot
    allocate."""
    if not _prints(module):
        return
    for function in _ops_named(module.operation, frozenset({"gpu.func"})):
        if _ops_named(function, frozenset({"memref.alloc", "gpu.alloc"})):
            raise UnsupportedOperationError(
                "hl.device_print of a computed or partial tile on XPU",
                reason="reading the tile needs a buffer in the kernel, which cannot "
                "allocate",
                alternatives=[
                    "print a full loaded tile of an input, or the output tile after "
                    "storing it",
                    "target='cpu' (CPU tensors)",
                ],
            )


_MEMREF_VIEWS = frozenset(
    {
        "memref.cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.reinterpret_cast",
        "memref.subview",
    }
)
_MEMREF_READS = frozenset({"memref.load", "vector.load", "vector.transfer_read"})
_MEMREF_NEUTRAL = frozenset({"memref.dim", "gpu.launch_func", "gpu.return"})


def _is_memref(value: ir.Value) -> bool:
    return isinstance(value.type, ir.MemRefType)


def _owner_op(value: ir.Value) -> ir.Operation | None:
    """The op defining ``value``, ``None`` for a block argument."""
    owner = value.owner
    return None if isinstance(owner, ir.Block) else owner.operation


def _base(value: ir.Value) -> ir.Value:
    """``value`` through its views."""
    while (op := _owner_op(value)) is not None and op.name in _MEMREF_VIEWS:
        value = op.operands[0]
    return value


def _ops_in(function: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        found.append(op.operation)
        return ir.WalkResult.ADVANCE

    function.walk(visit)
    return found[1:] if found and found[0] == function.operation else found


def _writes(function: ir.Operation) -> list[tuple[ir.Operation, ir.Value]]:
    """Each op of ``function`` taking a memref it isn't known only to read, and that memref's
    base: an op this doesn't know counts as a write."""
    writes = []
    for op in _ops_in(function):
        if op.name in _MEMREF_READS | _MEMREF_VIEWS | _MEMREF_NEUTRAL:
            continue
        operands = list(op.operands)
        if op.name == "memref.copy":
            operands = operands[1:]
        writes.extend((op, _base(v)) for v in operands if _is_memref(v))
    return writes


def _reads_behind(printed: ir.Value, function: ir.Operation) -> list[ir.Operation]:
    """The memory reads ``printed`` depends on, through ops and their regions."""
    reads: list[ir.Operation] = []
    seen: set[ir.Operation] = set()
    pending = [printed]
    while pending:
        value = pending.pop()
        op = _owner_op(value)
        if op is None:
            # A kernel argument, or a loop's value: what the loop reads counts. A loop's
            # position holds nothing read from memory.
            block = value.owner
            parent = block.owner.operation
            is_position = parent.name == "scf.for" and value == block.arguments[0]
            if parent != function and not is_position:
                op = parent
        if op is None or op in seen:
            continue
        seen.add(op)
        if op.name in _MEMREF_READS:
            reads.append(op)
        nested = _ops_in(op) if op.regions else []
        reads.extend(inner for inner in nested if inner.name in _MEMREF_READS)
        for inner in [op, *nested]:
            pending.extend(v for v in inner.operands if not _is_memref(v))
    return reads


def _constant_index(value: ir.Value) -> int | None:
    op = _owner_op(value)
    if op is None or op.name != "arith.constant":
        return None
    attribute = op.attributes["value"]
    return attribute.value if isinstance(attribute, ir.IntegerAttr) else None


def _position_in_dim(index: ir.Value, begin: ir.Value, size: int) -> bool:
    """Whether ``index`` is provably in ``[begin, begin + size)``: ``begin`` plus, or just,
    the induction variable of a loop from 0 by 1 up to at most ``size``."""
    induction = index
    op = _owner_op(index)
    if op is not None and op.name == "arith.addi":
        lhs, rhs = op.operands
        if lhs == begin:
            induction = rhs
        elif rhs == begin:
            induction = lhs
        else:
            return False
    elif _constant_index(begin) != 0:
        return index == begin
    if _owner_op(induction) is not None:
        return False
    loop = induction.owner.owner.operation
    if loop.name != "scf.for" or loop.regions[0].blocks[0].arguments[0] != induction:
        return False
    lower, upper, step = loop.operands[:3]
    bound = _constant_index(upper)
    return (
        _constant_index(lower) == 0
        and _constant_index(step) == 1
        and bound is not None
        and bound <= size
    )


def _store_covers(store: ir.Operation, read: ir.Operation) -> bool:
    """Whether the whole-tile ``store`` writes every element ``read`` reads."""
    if store.name != "vector.transfer_write" or read.name != "memref.load":
        return False
    # A mask would be one more operand; a transposed store has another map.
    begins = list(store.operands[2:])
    indices = list(read.operands[1:])
    shape = ir.VectorType(store.operands[0].type).shape
    identity = ir.AffineMap.get_identity(len(begins))
    return (
        store.operands[1] == read.operands[0]
        and len(begins) == len(indices) == len(shape)
        and str(store.attributes["permutation_map"])
        == str(ir.AffineMapAttr.get(identity))
        and all(starmap(_position_in_dim, zip(indices, begins, shape, strict=True)))
    )


def _entry_op(op: ir.Operation, block: ir.Block) -> ir.Operation | None:
    """``op``'s ancestor (or itself) directly in ``block``."""
    while op is not None:
        if op.block == block:
            return op
        op = op.parent.operation if op.parent is not None else None
    return None


def _shared_bases(module: ir.Module, function: ir.Operation) -> dict[ir.Value, int]:
    """Kernel arguments that are views of one host buffer map to one number."""
    name = function.attributes["sym_name"].value
    arguments = list(function.regions[0].blocks[0].arguments)
    groups: dict[ir.Value, int] = {}
    launches = [
        op
        for op in _ops_named(module.operation, frozenset({"gpu.launch_func"}))
        if str(op.attributes["kernel"]).endswith(f"::@{name}")
    ]
    hosts: list[ir.Value] = []
    for launch in launches:
        for argument, host in zip(
            arguments, list(launch.operands)[-len(arguments) :], strict=True
        ):
            base = _base(host)
            if base not in hosts:
                hosts.append(base)
            groups[argument] = hosts.index(base)
    return groups


def prints_of_stored_buffers(module: ir.Module) -> list[str]:
    """A warning for each print that reads a buffer its kernel also stores, unless it provably
    follows that store: lane 0 of thread 0 reads the whole tile, while the other hardware
    threads store their shares with nothing ordering them, so it can print a mix of old and
    new values. The kernel's own results aren't affected.

    Proved: the kernel's only write to the buffer (views and shared host buffers included)
    is one whole-tile ``vector.transfer_write`` in the kernel's body block, outside any
    loop, the print's read comes after it in that block and each index is in the stored
    tile. A buffer the kernel only reads prints wherever, silently. No print, no warning."""
    found: list[str] = []
    for function in _ops_named(module.operation, frozenset({"gpu.func"})):
        inside = [op for op in _ops_in(function) if op.name == "gpu.printf"]
        if not inside:
            continue
        body = function.regions[0].blocks[0]
        groups = _shared_bases(module, function)
        writes = [(op, groups.get(base, base)) for op, base in _writes(function)]
        kernel = function.attributes["sym_name"].value
        for printed in inside:
            if _may_see_other_threads(printed, function, body, groups, writes):
                label = printed.attributes["format"].value.split("[%ld")[0]
                found.append(
                    f"{label.strip()!r} printed in {kernel} may show wrong values: "
                    "it reads a tile of a buffer the kernel also stores, not provably "
                    "after that tile's store, and thread 0 reads while the other "
                    "hardware threads store. Print the tile after its store, in the "
                    "tile loop's body, or print an input the kernel only reads."
                )
    return found


def _may_see_other_threads(
    printed: ir.Operation,
    function: ir.Operation,
    body: ir.Block,
    groups: dict[ir.Value, int],
    writes: list[tuple[ir.Operation, object]],
) -> bool:
    for operand in printed.operands:
        for read in _reads_behind(operand, function):
            base = _base(read.operands[0])
            buffer = groups.get(base, base)
            stores = [op for op, stored in writes if stored == buffer]
            if not stores and _owner_op(base) is None:
                continue
            if len(stores) == 1 and _follows(stores[0], read, body):
                continue
            return True
    return False


def _follows(store: ir.Operation, read: ir.Operation, body: ir.Block) -> bool:
    if store.block != body or not _store_covers(store, read):
        return False
    entry = _entry_op(read, body)
    return entry is not None and store.is_before_in_block(entry)
