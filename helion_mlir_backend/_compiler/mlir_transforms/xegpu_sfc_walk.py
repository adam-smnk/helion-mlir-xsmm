"""A kernel's workgroups in the order of a space-filling curve (``xegpu_sfc_walk``).

The GPU's walker takes workgroups in about linear order, the first grid dim fastest
(a ticket test, ``agent-wiki/helion-mlir-backend/xpu-gemm-perf-plan.md``), so nearby
workgroups in that order can read very different parts of a GEMM's operands.

The grid stays as it is. Each workgroup takes the tile that its linear id
``x + grid_x * y`` has on lighthouse's generalized Hilbert curve (``gilbert2d``, as
``sfc_remap_forall`` on the CPU), from a table in the kernel. That is a bijection of
the grid: only the order changes, and a grid loop is parallel, so its results don't
depend on the order. Lighthouse's own op isn't used here: its tables on the host are
refused by the XPU target's host checks.

Opt-in per kernel, as the gain depends on the tile and the card:
``xpu_params.SFC_WALK``.
"""

from __future__ import annotations

from array import array
import re

from lighthouse.dialects.transform.transform_ext.utils.sfc import gilbert2d
from mlir import ir

DIMS = ("x", "y")
MAX_WORKGROUPS = 65532
"""The most workgroups a table can hold. One more makes the SPIR-V backend abort the
process, unable to emit the constant without ``SPV_INTEL_long_composites``: 3 + n words
of at most 65535 fit one instruction (inferred from the message and the boundary,
258x254 against 923x71)."""


def curve_tables(grid_x: int, grid_y: int) -> tuple[list[int], list[int]]:
    """For each linear id ``x + grid_x * y`` of a grid, the ``x`` and ``y`` of the tile
    it takes. Along the longer dim first, as ``sfc_remap_forall``."""
    points = list(gilbert2d(grid_y, grid_x))
    return [x for _, x in points], [y for y, _ in points]


def _block_ids(func: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.block_id":
            found.append(op)
        return ir.WalkResult.ADVANCE

    func.walk(visit)
    return found


def _dim(op: ir.Operation) -> str:
    """The dim of a ``gpu.block_id``: the attribute prints ``#gpu.dim<x>``."""
    return re.search("<([xyz])>", str(op.attributes["dimension"]))[1]


def _table(gpu_module: ir.Operation, name: str, values: list[int]) -> None:
    i32 = ir.IntegerType.get_signless(32)
    type_ = ir.MemRefType.get([len(values)], i32)
    dense = ir.DenseElementsAttr.get(
        array("i", values),
        type=i32,
        shape=[len(values)],
    )
    with ir.InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0]):
        ir.Operation.create(
            "memref.global",
            attributes={
                "sym_name": ir.StringAttr.get(name),
                "sym_visibility": ir.StringAttr.get("private"),
                "type": ir.TypeAttr.get(type_),
                "initial_value": dense,
                "constant": ir.UnitAttr.get(),
            },
        )


def _constant(value: int) -> ir.Value:
    return ir.Operation.create(
        "arith.constant",
        results=[ir.IndexType.get()],
        attributes={"value": ir.IntegerAttr.get(ir.IndexType.get(), value)},
    ).results[0]


def _binary(name: str, left: ir.Value, right: ir.Value) -> ir.Value:
    return ir.Operation.create(
        name, results=[ir.IndexType.get()], operands=[left, right]
    ).results[0]


def _looked_up(name: str, size: int, index: ir.Value) -> ir.Value:
    i32 = ir.IntegerType.get_signless(32)
    table = ir.Operation.create(
        "memref.get_global",
        results=[ir.MemRefType.get([size], i32)],
        attributes={"name": ir.FlatSymbolRefAttr.get(name)},
    ).results[0]
    value = ir.Operation.create(
        "memref.load", results=[i32], operands=[table, index]
    ).results[0]
    return ir.Operation.create(
        "arith.index_cast", results=[ir.IndexType.get()], operands=[value]
    ).results[0]


def walk_on_curve(gpu_module: ir.Operation, func: ir.Operation) -> bool:
    """``func``'s block ids replaced by the tile its linear id takes on the curve, the
    tables added to ``gpu_module``. ``False``, with nothing changed, for a grid of one
    line, whose only order is the linear one. :class:`ValueError` for a grid this can't
    read: no known size, or a third dim in use."""
    if "known_grid_size" not in func.attributes:
        raise ValueError("the kernel's grid has no known size")
    grid_x, grid_y, grid_z = ir.DenseI32ArrayAttr(func.attributes["known_grid_size"])
    if grid_z != 1:
        raise ValueError(f"a 3D grid: {[grid_x, grid_y, grid_z]}")
    if grid_x == 1 or grid_y == 1:
        return False
    if grid_x * grid_y > MAX_WORKGROUPS:
        raise ValueError(
            f"a grid of {grid_x}x{grid_y} workgroups: its tables hold at most "
            f"{MAX_WORKGROUPS}"
        )
    # With one workgroup in z, its block id is 0 either way.
    old = [op for op in _block_ids(func) if _dim(op) in DIMS]

    table_x, table_y = curve_tables(grid_x, grid_y)
    size = grid_x * grid_y
    symbol = ir.StringAttr(func.attributes["sym_name"]).value
    name_x, name_y = f"{symbol}_sfc_x", f"{symbol}_sfc_y"
    _table(gpu_module, name_y, table_y)
    _table(gpu_module, name_x, table_x)

    first = next(iter(func.regions[0].blocks[0].operations)).operation
    with ir.InsertionPoint(first) as point:
        ids = {dim: ir.Operation.parse(f"%id = gpu.block_id {dim}") for dim in DIMS}
        for op in ids.values():
            point.insert(op)
        x, y = (ids[dim].results[0] for dim in DIMS)
        linear = _binary("arith.addi", x, _binary("arith.muli", y, _constant(grid_x)))
        walked = {
            "x": _looked_up(name_x, size, linear),
            "y": _looked_up(name_y, size, linear),
        }
    for op in old:
        op.results[0].replace_all_uses_with(walked[_dim(op)])
        op.erase()
    return True
