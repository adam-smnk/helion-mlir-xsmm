"""``alloca_arrays``: constant-count ``llvm.alloca``s of XeVM GPU modules as
allocas of one array, addressed through a ``getelementptr [0, 0]``.

Works around an LLVM SPIR-V backend miscompile (no LLVM patch, by policy).
XeVMToLLVM gives each ``intel_sub_group_2d_block_read_*`` call its destination
as ``llvm.alloca %c8 x i32``. Since llvm-project 14a8bae5a766 ("[SPIRV] Lower
constant-count entry-block allocas without VLA", #232579) the backend folds such
an alloca in the entry block to ``[8 x i32]``, then sizes the variable by the
pointee of the mangled builtin's ``Pj`` parameter: an ``OpVariable`` of one
``uint`` (4 bytes). The block read writes past it and the kernel reads zeros
(block reads in the entry block: elementwise and reduction tiles, a C tile
loaded before the K loop). An array alloca addressed by a GEP keeps its type
(``OpVariable`` of the array). Reproducer (helion-mlir-dev):
``agent-wiki/upstream-repro/spirv-alloca/spirv_alloca_count_mangled_call.ll``.

The rewrite keeps the semantics (the same number of elements, alignment and
address space, one allocation where the old one was), so it stays harmless
once the backend is fixed. Only allocas whose count is an ``llvm.mlir.constant``
greater than 1 are rewritten, only in ``gpu.module``s with an ``#xevm.target``;
dynamic counts, single elements, array element types, allocas with attributes
other than ``elem_type`` and ``alignment`` (e.g. ``inalloca``, which the new
alloca would drop) and host code are kept.
"""

from __future__ import annotations

from mlir import ir

_ALLOCA = "llvm.alloca"
_CONSTANT = "llvm.mlir.constant"


def _xevm_modules(module: ir.Module) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.module":
            targets = op.attributes.get("targets")
            if targets is not None and any(
                str(target).startswith("#xevm.target") for target in targets
            ):
                found.append(op)
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def constant_count(alloca: ir.Operation) -> int | None:
    """The element count of ``alloca`` if an ``llvm.mlir.constant`` gives it."""
    owner = alloca.operands[0].owner
    if isinstance(owner, ir.Block) or owner.name != _CONSTANT:
        return None
    value = owner.attributes["value"]
    if not isinstance(value, ir.IntegerAttr):
        return None
    return value.value


_KEPT_ATTRIBUTES = frozenset({"elem_type", "alignment"})
"""The attributes an alloca may have to be rewritten: the rewrite keeps them."""


def rewritable(alloca: ir.Operation) -> int | None:
    """The count of an alloca to rewrite (constant, more than one element of a
    non-array type, no attributes the rewrite would drop), else ``None``."""
    names = {getattr(attr, "name", attr) for attr in alloca.attributes}
    if names - _KEPT_ATTRIBUTES:
        return None
    count = constant_count(alloca)
    if count is None or count <= 1:
        return None
    element = ir.TypeAttr(alloca.attributes["elem_type"]).value
    if str(element).startswith("!llvm.array"):
        return None
    return count


def _rewrite(alloca: ir.Operation, count: int) -> None:
    element = ir.TypeAttr(alloca.attributes["elem_type"]).value
    array = ir.Type.parse(f"!llvm.array<{count} x {element}>")
    count_type = alloca.operands[0].type
    pointer = alloca.results[0].type
    with ir.InsertionPoint(alloca), alloca.location:
        one = ir.Operation.create(
            _CONSTANT,
            results=[count_type],
            attributes={"value": ir.IntegerAttr.get(count_type, 1)},
        )
        attributes = {"elem_type": ir.TypeAttr.get(array)}
        if "alignment" in alloca.attributes:
            attributes["alignment"] = alloca.attributes["alignment"]
        allocation = ir.Operation.create(
            _ALLOCA,
            results=[pointer],
            operands=[one.results[0]],
            attributes=attributes,
        )
        address = ir.Operation.create(
            "llvm.getelementptr",
            results=[pointer],
            operands=[allocation.results[0]],
            attributes={
                "elem_type": ir.TypeAttr.get(array),
                "rawConstantIndices": ir.DenseI32ArrayAttr.get([0, 0]),
            },
        )
    alloca.results[0].replace_all_uses_with(address.results[0])
    alloca.erase()


def alloca_arrays(module: ir.Module) -> int:
    """Rewrites ``module``'s XeVM constant-count allocas in place; returns how
    many."""
    allocas: list[tuple[ir.Operation, int]] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == _ALLOCA and (count := rewritable(op)) is not None:
            allocas.append((op, count))
        return ir.WalkResult.ADVANCE

    for gpu_module in _xevm_modules(module):
        gpu_module.walk(visit)
    for alloca, count in allocas:
        _rewrite(alloca, count)
    return len(allocas)
