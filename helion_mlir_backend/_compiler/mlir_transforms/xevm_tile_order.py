"""``order_tile_chunks``: in XeVM GPU modules, the side-effect-free ops of a basic
block are reordered: each is computed right before the side-effecting op (or
later use) that needs it, depth first.

XeGPU's unrolling leaves a kernel's tile math in vector-op order: one op on all
chunks of the tile, then the next op on all chunks. Every intermediate of a
chain is then live for the whole tile at once. That is harmless for short math
and fatal for long chains: 29_Matmul_Mish_Mish (``mish(mish(x))`` over 256x256
tiles, 16 chunks of 8 f32 per work item, so 128 GRFs per value in SIMD16) has a
few dozen temporaries from nested ``tanh`` and ``log1p`` polynomials
(:mod:`xpu_math_approx`) and spilled 92544 bytes on the Arc B50, 11 ms a call
against 4.9 with OpenCL's builtins, whose temporaries live one lane at a time.
Depth first the chunks do not overlap: the peak is the inputs not yet consumed
plus one chunk's temporaries. No spill, 3.9 ms. IGC keeps this order but does not
make it: its own scheduler leaves the vector-op order spilling, with or without
its vectorizer.

Not a bug but a cost, in XeGPU's unrolling: accepted as a workaround (user, 2026-11-02).
Reproduced in helion-mlir-dev's ``agent-wiki/upstream-repro/xegpu-tile-order/``. Removal: when
that reproducer doesn't spill with this stage off, on the IGC builds the XPU target supports,
because XeGPU's unrolling emits chunks depth first or IGC's scheduler stops spilling.

The reorder only preserves values; it can also cost time when nothing spills
(elementwise Mish at ``[128, 256]``: about +7 %). So it applies per GPU module, to
those that spill: the XPU target's binary stage compiles, reads ``spill_size`` from
the binaries (:func:`spill_bytes`), compiles the spilling modules reordered, and
keeps a reordered binary if it spills less. That is one more IGC compile for a
spilling kernel. Modules of very long tile math (:data:`ORDER_FIRST_ABOVE`) go the
other way round: ordered first, plain only if that still spills.

A static estimate of the peak does not predict the spill: it follows the widest
polynomial, not the chain length (estimates of 272 GRFs spilled, ``sin * cos``,
17024 bytes; 648 did not, Softplus; Mish and Mish(Mish) estimate the same). The sum
of all results (:func:`tile_math_grfs`) separates the long end only: ``sin * cos``
spills at 3022, and the compile-first path finds that.

Matched exactly: blocks of ``llvm.func``s in a ``gpu.module`` with an
``#xevm.target``. An op moves only if it is on the allowlist of ops without side
effects (:data:`_PURE`, or a ``__spirv_ocl_*`` call with a result and no pointer
operand); anything else, memory accesses and other calls included, keeps its
relative order. A block holding an op with regions stays as it is: a value may be
used inside them.
"""

from __future__ import annotations

import re

from mlir import ir

from .xevm_alloca_arrays import xevm_modules

_PURE = frozenset(
    {
        "llvm.mlir.constant",
        "llvm.mlir.undef",
        "llvm.mlir.poison",
        "llvm.fadd",
        "llvm.fsub",
        "llvm.fmul",
        "llvm.fdiv",
        "llvm.fneg",
        "llvm.fcmp",
        "llvm.icmp",
        "llvm.select",
        "llvm.bitcast",
        "llvm.zext",
        "llvm.sext",
        "llvm.trunc",
        "llvm.sitofp",
        "llvm.uitofp",
        "llvm.fptosi",
        "llvm.fptoui",
        "llvm.fpext",
        "llvm.fptrunc",
        "llvm.add",
        "llvm.sub",
        "llvm.mul",
        "llvm.and",
        "llvm.or",
        "llvm.xor",
        "llvm.shl",
        "llvm.lshr",
        "llvm.ashr",
        "llvm.extractelement",
        "llvm.insertelement",
        "llvm.shufflevector",
        "llvm.intr.fma",
        "llvm.intr.fmuladd",
        "llvm.intr.fabs",
        "llvm.intr.umin",
        "llvm.intr.umax",
        "llvm.intr.smin",
        "llvm.intr.smax",
        "llvm.intr.floor",
        "llvm.intr.ceil",
        "llvm.intr.sqrt",
        "llvm.intr.copysign",
        "llvm.intr.maxnum",
        "llvm.intr.minnum",
        "llvm.intr.maximum",
        "llvm.intr.minimum",
    }
)
ORDER_FIRST_ABOVE = 12000
"""From this :func:`tile_math_grfs` a GPU module is expected to spill, and is
compiled ordered first: a spilling compile is the slow one (mish x6: 59 s, ordered
7 s), so the compile-first path can hit the compile timeout. An empirical line:
over 176 GEMM epilogues and elementwise kernels at 7 tile sizes (Arc B50, IGC 2.41),
nothing from here up was free of spill, nothing from 10238 up either, and 9563
was. Below it spills are found by compiling. A wrong prediction only costs run
time: up to about 7 % measured on kernels that fit (see the module docstring)."""
_CONSTANTS = frozenset({"llvm.mlir.constant", "llvm.mlir.undef", "llvm.mlir.poison"})
_SIMD = 16
_GRF_BYTES = 64
_OPENCL_MATH = "__spirv_ocl_"
_SPILL_SIZE = re.compile(r"spill_size:\s*(\d+)")


def _is_pure(op: ir.Operation) -> bool:
    if op.name in _PURE:
        return True
    if op.name != "llvm.call" or len(op.results) != 1:
        return False
    callee = op.attributes.get("callee")
    return (
        callee is not None
        and _OPENCL_MATH in str(callee)
        and not any(_is_pointer(operand.type) for operand in op.operands)
    )


def _is_pointer(type_: ir.Type) -> bool:
    return str(type_).startswith("!llvm.ptr")


def spill_bytes(binaries: ir.Operation) -> int:
    """The spill of all XPU kernels of the compiled module ``binaries``: the sum
    of each ``gpu.binary``'s ``spill_size`` of ``.ze_info``, read from its printed
    object (the info is plain text in the binary)."""
    total = [0]

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.binary":
            total[0] += sum(int(n) for n in _SPILL_SIZE.findall(str(op)))
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    binaries.walk(visit, ir.WalkOrder.PRE_ORDER)
    return total[0]


def _reorder(block: ir.Block) -> None:
    ops = [op.operation for op in block.operations]
    producer = {result: op for op in ops if _is_pure(op) for result in op.results}
    done: set[ir.Operation] = set()
    order: list[ir.Operation] = []

    def emit(root: ir.Operation) -> None:
        stack = [(root, iter(list(root.operands)))]
        while stack:
            op, operands = stack[-1]
            for operand in operands:
                source = producer.get(operand)
                if source is not None and source not in done:
                    done.add(source)
                    stack.append((source, iter(list(source.operands))))
                    break
            else:
                stack.pop()
                order.append(op)

    *body, terminator = ops
    for op in body:
        if not _is_pure(op):
            emit(op)
    # Used in other blocks only (or not at all): before the terminator.
    for op in body:
        if _is_pure(op) and op not in done:
            done.add(op)
            emit(op)
    emit(terminator)
    for op in order[:-1]:
        op.move_before(terminator)


def tile_math_grfs(gpu_module: ir.Operation) -> int:
    """Over the blocks of ``gpu_module``'s functions, the most registers (SIMD16)
    all results of side-effect-free ops add up to, whatever their live ranges."""
    best = [0]

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "llvm.func":
            for region in op.regions:
                for block in region.blocks:
                    best[0] = max(best[0], _block_math_grfs(block))
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    gpu_module.walk(visit, ir.WalkOrder.PRE_ORDER)
    return best[0]


def _block_math_grfs(block: ir.Block) -> int:
    total = 0
    for wrapped in block.operations:
        op = wrapped.operation
        if op.name in _CONSTANTS or not _is_pure(op):
            continue
        for result in op.results:
            total += _value_bytes(result.type)
    return total * _SIMD // _GRF_BYTES


def _value_bytes(type_: ir.Type) -> int:
    """Bytes per lane of a scalar or vector value."""
    if not isinstance(type_, ir.VectorType):
        return 4
    vector = ir.VectorType(type_)
    element = vector.element_type
    if isinstance(element, ir.IntegerType):
        bits = ir.IntegerType(element).width
    elif isinstance(element, ir.F32Type):
        bits = 32
    elif isinstance(element, ir.F64Type):
        bits = 64
    else:
        bits = 16
    return vector.get_dim_size(0) * bits // 8


def order_tile_chunks(module: ir.Module | ir.Operation) -> int:
    """Reorders ``module``'s matching blocks in place; returns how many."""
    blocks: list[ir.Block] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "llvm.func":
            for region in op.regions:
                blocks.extend(
                    block
                    for block in region.blocks
                    if not any(o.operation.regions for o in block.operations)
                )
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    for gpu_module in xevm_modules(module):
        gpu_module.walk(visit, ir.WalkOrder.PRE_ORDER)
    for block in blocks:
        _reorder(block)
    return len(blocks)
