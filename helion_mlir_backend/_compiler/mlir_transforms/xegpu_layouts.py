"""XeGPU layouts derived from the IR: ``annotate_layouts`` on a ``gpu.func``.

Reads only the kernel's own IR (tile shapes, def-use chains, memory accesses)
and a few hints the IR carries (:data:`HINTS_ATTR`), and writes workgroup-level
anchors, ``xegpu.prefetch_nd`` ops and the kernel's thread count. There are no
kernel classes. A layout is legal or not by what the ops require:

1. **Dim classes** (:class:`DimClasses`): a union-find over ``(value, dim)``. Two
   dims are one class when an op requires them to be distributed alike: an
   elementwise op's operands, a DPAS's ``A[m, k] B[k, n] C[m, n]``, a transpose's
   permutation, a broadcast's aligned dims, loop-carried values, the kept dims
   of a reduction, and accesses of one buffer at one offset.
2. **Pinned classes**: a contracted dim must be whole on each subgroup, and so
   must a reduced dim unless a cross-subgroup reduction is wanted (upstream has
   one, through SLM and ``gpu.barrier``; not used here). The rest are free.
3. **Grids** (:func:`legal_grids`): the output's two dims are the grid's axes.
   A grid ``(fx, fy)`` is legal if its factors divide every value of their class,
   leave DPAS and 2D block sizes on each subgroup, and fit the thread limit.
4. **Preference**: hints, then a parameter source: ``"selector"`` (lighthouse's
   parameter selection, for matmuls it can size) or ``"rule"``
   (:func:`preferred_grid`).
5. **Check** (:func:`check_layouts`): every DPAS and memory op must end up with a
   layout the hardware can take. Upstream's lowering does not refuse an
   unanchored op: ``gpu-lower-to-xevm-pipeline`` runs ``xegpu-propagate-layout``,
   which fills in defaults.

An op this module does not know, on tile vectors, is a refusal
(:class:`LayoutError`): a poor tile is slow, but a poor layout can give wrong
rows. Elementwise kernels keep ``xegpu_elementwise``'s layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from operator import itemgetter
from typing import TYPE_CHECKING

from lighthouse.dialects.transform.transform_ext.utils.select_xegpu_gemm_params import (
    select_xegpu_gemm_params,
)
from mlir import ir
from mlir.passmanager import PassManager

from .xegpu_elementwise import BLOCK_COLS
from .xegpu_elementwise import MAX_SG_ELEMENTS
from .xegpu_elementwise import MAX_SUBGROUPS
from .xegpu_elementwise import SUBGROUP_SIZE
from .xegpu_elementwise import block_load_tile
from .xegpu_elementwise import block_store_tile
from .xegpu_elementwise import elementwise_layout
from .xegpu_elementwise import find_tiles

if TYPE_CHECKING:
    from collections.abc import Sequence

HINTS_ATTR = "xegpu_layouts.hints"
"""Dictionary on a ``gpu.func``, read and removed by :func:`annotate_layouts`:

- ``source``: string, ``"rule"`` (the default) or ``"selector"``;
- ``sg_tile``: ``[rows, cols]``; ``[rows, 0]``: all subgroups on the rows;
- ``load_tiles``: ``[a_m, a_k, b_k, b_n]`` of DPAS operands, ``[m, n]`` of other
  tiles;
- ``prefetch_tiles``: ``[a_m, a_k, b_k, b_n]``, as the operands are stored;
- ``prefetch_depths``: ``[a, b]``, 0 for no prefetch.

The arrays are ``DenseI64ArrayAttr``."""
PREFETCH_ATTR = "xegpu_layouts.prefetch"
"""Set by :func:`annotate_layouts` on a ``xegpu.load_nd`` to prefetch: its index
in the plan's ``prefetches``, for the schedule that inserts it."""

# The target: a mirror of xegpu::uArch (uArch/IntelGpuXe2.h, uArchBase.h), which
# has no Python bindings, for Xe2 with the large register file.
DPAS_ROWS = (8, 4, 2, 1)
"""``getSupportedM``: 1..8 for 16-bit floats. Powers of two here: the layouts
need rows that divide a subgroup tile evenly."""
DPAS_COLS = 16
"""``getSupportedN``, and K of a 16-bit A or B (two 8-element pairs)."""
LOAD_MAX_ROWS = 32
LOAD_MAX_COLS = 32
"""A 2D block load of 16-bit elements: at most 32 rows of 64 bytes."""
TRANSPOSED_LOAD = (16, 16)
"""A transposed 2D block load of a DPAS's B (32-bit pairs: 8 wide)."""
TRANSPOSED_LOADS = (TRANSPOSED_LOAD, (32, 16))
"""``getBlockWidthHeightCount`` with transpose: 16 or 32 rows, 8 pairs wide."""
PREFETCH_ROWS = (8, 32)
PREFETCH_COLS = (16, 32)
PREFETCH_INST = (8, 16)
MAX_DPAS_PER_THREAD = 16
"""Lighthouse's ``dpas_exec_size``."""
REGISTER_BYTES = 256 * 64
"""The large register file."""
SHARED_PREFETCH = (16, 32)
"""The prefetch tile of an operand all subgroups read whole."""

_ANCHORED = frozenset(
    {"xegpu.dpas", "xegpu.load_nd", "xegpu.store_nd", "xegpu.prefetch_nd"}
)
_ELEMENTWISE_PREFIXES = ("arith.", "math.")
_NO_TILES = frozenset({"gpu.return", "scf.yield", "scf.for", "xegpu.create_nd_tdesc"})


class LayoutError(ValueError):
    """No legal layout, or IR this module does not reason about."""


def _shape(value: ir.Value) -> tuple[int, ...] | None:
    value_type = value.type
    if isinstance(value_type, ir.VectorType):
        return tuple(ir.VectorType(value_type).shape)
    return None


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


class DimClasses:
    """A union-find over ``(value, dim)`` of a kernel's tile vectors."""

    def __init__(self) -> None:
        self.parent: dict[tuple[ir.Value, int], tuple[ir.Value, int]] = {}
        self.contracted: set[tuple[ir.Value, int]] = set()
        self.reduced: set[tuple[ir.Value, int]] = set()

    def find(self, key: tuple[ir.Value, int]) -> tuple[ir.Value, int]:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def tie(self, a: ir.Value, a_dim: int, b: ir.Value, b_dim: int) -> None:
        sizes = (_shape(a)[a_dim], _shape(b)[b_dim])
        if sizes[0] != sizes[1]:
            raise LayoutError(
                f"dims of sizes {sizes[0]} and {sizes[1]} would have to be "
                "distributed alike"
            )
        self.parent[self.find((a, a_dim))] = self.find((b, b_dim))

    def tie_all(self, a: ir.Value, b: ir.Value) -> None:
        for dim in range(len(_shape(a))):
            self.tie(a, dim, b, dim)

    def pinned(self, value: ir.Value, dim: int) -> bool:
        root = self.find((value, dim))
        return root in {self.find(key) for key in self.contracted | self.reduced}

    def same(self, a: tuple[ir.Value, int], b: tuple[ir.Value, int]) -> bool:
        return self.find(a) == self.find(b)


def _tie_elementwise(classes: DimClasses, op: ir.Operation) -> None:
    values = [v for v in [*op.operands, *op.results] if _shape(v)]
    for value in values[1:]:
        classes.tie_all(values[0], value)


def _tie_broadcast(classes: DimClasses, op: ir.Operation) -> None:
    source, result = op.operands[0], op.results[0]
    source_shape = _shape(source)
    if source_shape is None:
        return
    shift = len(_shape(result)) - len(source_shape)
    for dim, size in enumerate(source_shape):
        if size != 1:
            classes.tie(source, dim, result, dim + shift)


def _tie_shape_cast(classes: DimClasses, op: ir.Operation) -> None:
    source, result = op.operands[0], op.results[0]
    kept = [
        [dim for dim, size in enumerate(_shape(value)) if size != 1]
        for value in (source, result)
    ]
    if [_shape(source)[d] for d in kept[0]] != [_shape(result)[d] for d in kept[1]]:
        raise LayoutError(f"'{op.name}' that does more than add or drop unit dims")
    for source_dim, result_dim in zip(*kept, strict=True):
        classes.tie(source, source_dim, result, result_dim)


def _tie_reduction(classes: DimClasses, op: ir.Operation) -> None:
    source, acc, result = op.operands[0], op.operands[1], op.results[0]
    reduced = set(ir.DenseI64ArrayAttr(op.attributes["reduction_dims"]))
    kept = [dim for dim in range(len(_shape(source))) if dim not in reduced]
    classes.reduced.update((source, dim) for dim in reduced)
    if _shape(result) is not None:
        for result_dim, dim in enumerate(kept):
            classes.tie(source, dim, result, result_dim)
            classes.tie(acc, result_dim, result, result_dim)


def _tie_for(classes: DimClasses, op: ir.Operation) -> None:
    inits = list(op.operands)[3:]
    body = op.regions[0].blocks[0]
    yielded = list(body.operations)[-1].operands
    iter_args = list(body.arguments)[1:]
    for init, arg, value, result in zip(
        inits, iter_args, yielded, op.results, strict=True
    ):
        if _shape(init) is None:
            continue
        classes.tie_all(init, arg)
        classes.tie_all(arg, value)
        classes.tie_all(value, result)


def _tie_dpas(classes: DimClasses, op: ir.Operation) -> None:
    a, b = op.operands[0], op.operands[1]
    result = op.results[0]
    classes.tie(a, 0, result, 0)
    classes.tie(b, 1, result, 1)
    classes.tie(a, 1, b, 0)
    classes.contracted.add((a, 1))
    if len(op.operands) > 2:
        classes.tie_all(op.operands[2], result)


def _tie_accesses(classes: DimClasses, accesses: list[_Access]) -> None:
    """Accesses of one buffer in one tile shape, at one offset per dim (or a loop
    variable in both, walking the same dim), are the same subgroup's data: a tile
    a subgroup stores must be the one it loads, and no subgroup waits for
    another's store inside a workgroup."""
    by_source: dict[ir.Value, list[_Access]] = {}
    for access in accesses:
        by_source.setdefault(access.source, []).append(access)
    for group in by_source.values():
        for other in group[1:]:
            first = group[0]
            if _shape(first.value) != _shape(other.value):
                continue
            for dim in range(len(_shape(first.value))):
                if _same_offset(first.offsets, other.offsets, dim):
                    classes.tie(first.value, dim, other.value, dim)


def _same_offset(a: Sequence[object], b: Sequence[object], dim: int) -> bool:
    if len(a) != len(b):
        return False
    left, right = a[dim], b[dim]
    if isinstance(left, int) or isinstance(right, int):
        return left == right
    return left == right or (_is_induction(left) and _is_induction(right))


def _is_induction(value: ir.Value) -> bool:
    if not isinstance(value, ir.BlockArgument):
        return False
    return (
        value.arg_number == 0
        and value.owner.owner is not None
        and value.owner.owner.name == "scf.for"
    )


@dataclass
class _Access:
    """A ``load_nd`` or ``store_nd`` and its tile."""

    op: ir.Operation
    value: ir.Value
    source: ir.Value
    offsets: list[object]


def _access(op: ir.Operation) -> _Access:
    if op.name == "xegpu.load_nd":
        value, tdesc, dynamic = op.results[0], op.operands[0], list(op.operands)[1:]
    else:
        value, tdesc, dynamic = op.operands[0], op.operands[1], list(op.operands)[2:]
    desc = _defining(tdesc)
    if desc is None or desc.name != "xegpu.create_nd_tdesc":
        raise LayoutError(f"'{op.name}' of a descriptor that is not created here")
    remaining = iter(dynamic)
    dynamic_marker = ir.ShapedType.get_dynamic_size()
    offsets: list[object] = []
    for offset in ir.DenseI64ArrayAttr(op.attributes["const_offsets"]):
        offsets.append(next(remaining) if offset == dynamic_marker else offset)
    return _Access(op, value, desc.operands[0], offsets)


@dataclass
class KernelIR:
    """What :func:`analyze` reads off a ``gpu.func``."""

    func: ir.Operation
    classes: DimClasses
    dpas: list[ir.Operation]
    loads: list[_Access]
    stores: list[_Access]
    reductions: list[ir.Operation]
    scatter_stores: list[ir.Operation]
    primary: ir.Value | None
    """The value that sets the grid's axes: the last 2D tile stored, else the
    first reduction's source."""


def analyze(func: ir.Operation) -> KernelIR:
    """The dim classes and tile accesses of the kernel ``func``; a
    :class:`LayoutError` on an op it does not know."""
    classes = DimClasses()
    dpas: list[ir.Operation] = []
    loads: list[_Access] = []
    stores: list[_Access] = []
    reductions: list[ir.Operation] = []
    scatter_stores: list[ir.Operation] = []
    for op in _walk(func):
        name = op.name
        if name == "xegpu.dpas":
            _tie_dpas(classes, op)
            dpas.append(op)
        elif name == "xegpu.load_nd":
            loads.append(_access(op))
        elif name == "xegpu.store_nd":
            stores.append(_access(op))
        elif name in ("xegpu.load", "xegpu.store"):
            _tie_gather(classes, op)
            if name == "xegpu.store":
                scatter_stores.append(op)
        elif name == "vector.transpose":
            source, result = op.operands[0], op.results[0]
            for dim, from_dim in enumerate(
                ir.DenseI64ArrayAttr(op.attributes["permutation"])
            ):
                classes.tie(source, from_dim, result, dim)
        elif name == "vector.multi_reduction":
            _tie_reduction(classes, op)
            reductions.append(op)
        elif name == "vector.broadcast":
            _tie_broadcast(classes, op)
        elif name == "vector.shape_cast":
            _tie_shape_cast(classes, op)
        elif name == "scf.for":
            _tie_for(classes, op)
        elif name.startswith(_ELEMENTWISE_PREFIXES):
            _tie_elementwise(classes, op)
        elif name not in _NO_TILES and any(
            _shape(value) for value in [*op.operands, *op.results]
        ):
            if name == "vector.contract":
                raise LayoutError(
                    "a 'vector.contract' that is no matmul (e.g. a row's dot product "
                    "with itself): no XeGPU op takes it"
                )
            if name != "vector.step":
                raise LayoutError(f"'{name}' on tile vectors")
    _tie_accesses(classes, [*loads, *stores])
    primary = next(
        (a.value for a in reversed(stores) if len(_shape(a.value)) == 2),
        reductions[0].operands[0] if reductions else None,
    )
    return KernelIR(
        func, classes, dpas, loads, stores, reductions, scatter_stores, primary
    )


def _tie_gather(classes: DimClasses, op: ir.Operation) -> None:
    """A gather's or scatter's data, offsets and mask are distributed alike."""
    values = [v for v in [*op.operands, *op.results] if _shape(v)]
    for value in values[1:]:
        classes.tie_all(values[0], value)


# --- Grids ---------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Subgroups along the primary tile's rows and columns."""

    rows: int
    cols: int

    @property
    def subgroups(self) -> int:
        return self.rows * self.cols


@dataclass
class Axes:
    """The primary tile's dims and what may be distributed."""

    kernel: KernelIR
    rows: int
    cols: int
    free_rows: bool
    free_cols: bool

    def tile(self, grid: Grid) -> tuple[int, int]:
        return self.rows // grid.rows, self.cols // grid.cols


def axes_of(kernel: KernelIR) -> Axes:
    primary = kernel.primary
    if primary is None or len(_shape(primary)) != 2:
        raise LayoutError("no 2D tile sets the kernel's grid")
    rows, cols = _shape(primary)
    classes = kernel.classes
    return Axes(
        kernel,
        rows,
        cols,
        not classes.pinned(primary, 0),
        not classes.pinned(primary, 1),
    )


def _on_axes(kernel: KernelIR, value: ir.Value) -> tuple[bool, bool]:
    """Whether ``value`` has dims of the primary tile's rows and columns."""
    classes, primary = kernel.classes, kernel.primary
    dims = range(len(_shape(value)))
    return (
        any(classes.same((value, dim), (primary, 0)) for dim in dims),
        any(classes.same((value, dim), (primary, 1)) for dim in dims),
    )


def _divisors(size: int) -> list[int]:
    return [d for d in range(1, size + 1) if size % d == 0]


def legal_grids(axes: Axes) -> list[Grid]:
    """Every grid the kernel's IR allows: factors on its free dims, dividing them,
    leaving a whole DPAS (or 2D block) on each subgroup, within the thread limit."""
    grids = []
    has_dpas = bool(axes.kernel.dpas)
    # A subgroup stores what it owns: a tile stored whole by each of several
    # subgroups would be written by all of them.
    on_axes = [_on_axes(axes.kernel, a.value) for a in axes.kernel.stores]
    free_rows = axes.free_rows and all(rows for rows, _ in on_axes)
    free_cols = axes.free_cols and all(cols for _, cols in on_axes)
    for fx in _divisors(axes.rows) if free_rows else [1]:
        for fy in _divisors(axes.cols) if free_cols else [1]:
            if fx * fy > MAX_SUBGROUPS:
                continue
            sg_rows, sg_cols = axes.rows // fx, axes.cols // fy
            if sg_cols % BLOCK_COLS:
                continue
            if has_dpas and not any(sg_rows % r == 0 for r in DPAS_ROWS):
                continue
            if fx > 1 and sg_rows % 2 and sg_rows != 1:
                continue
            grids.append(Grid(fx, fy))
    return grids


TARGET_SG_ROWS = 32
"""What :func:`preferred_grid` looks for: 32 rows per subgroup. A subgroup reads a
tile's rows from A's strided rows, and 64 of them stalled the memory pipe on the
B50 at K = 8192 (SendStall 41 % against 6 %, L1 2)."""


OUTPUT_BOUND_SG_TILE = (8, 64)
"""What :func:`preferred_grid` looks for in a kernel whose output traffic dominates
(:func:`output_bound`): each subgroup stores 8 rows of 128 bytes, even with fewer
subgroups. Measured on the B50, 32768x32768x64 at 64x128, device time: 8x64 15395
us; 8x128 16317; 16x32 17276; 8x32 18042; 16x16 21851."""


def output_bound(kernel: KernelIR) -> bool:
    """Whether the kernel writes at least as many output elements as it reads
    operand elements, ``M * N >= K * (M + N)``: a DPAS's loop bound for K and the
    stored buffer's last two dims for M and N, when all are static. The same test
    picks a GEMM's grid order (``xpu_params.columns_first``)."""
    if not kernel.dpas or not kernel.stores:
        return False
    source = kernel.stores[-1].source.type
    if not isinstance(source, ir.MemRefType) or source.rank < 2:
        return False
    m, n = ir.MemRefType(source).shape[-2:]
    loop = kernel.dpas[0].parent
    if loop is None or loop.name != "scf.for":
        return False
    bound = _defining(loop.operands[1])
    if bound is None or bound.name != "arith.constant":
        return False
    k = ir.IntegerAttr(bound.attributes["value"]).value
    return min(m, n, k) > 0 and m * n >= k * (m + n)


def preferred_grid(axes: Axes, grids: Sequence[Grid]) -> Grid:
    """The rule's pick among legal ``grids``. A kernel bound by its output takes
    :data:`OUTPUT_BOUND_SG_TILE` if legal. Else: tiles of whole DPAS depth (at least 8
    rows, where the kernel has that many); most subgroups whose accumulators leave
    registers for the operands (at most :data:`MAX_SG_ELEMENTS` elements); then the
    row count closest to :data:`TARGET_SG_ROWS`; then the most columns."""
    if not grids:
        raise LayoutError(
            f"no legal subgroup grid for a {axes.rows}x{axes.cols} tile: columns "
            f"must leave a multiple of {BLOCK_COLS} per subgroup"
        )
    if output_bound(axes.kernel):
        for grid in grids:
            if axes.tile(grid) == OUTPUT_BOUND_SG_TILE:
                return grid
    fitting = [
        grid
        for grid in grids
        if axes.tile(grid)[0] * axes.tile(grid)[1] <= MAX_SG_ELEMENTS
    ] or list(grids)

    def rank(grid: Grid) -> tuple[bool, int, float, int]:
        rows, cols = axes.tile(grid)
        shallow = rows < DPAS_ROWS[0] and axes.rows >= DPAS_ROWS[0]
        return (shallow, -grid.subgroups, abs(math.log2(rows / TARGET_SG_ROWS)), -cols)

    return min(fitting, key=rank)


# --- Layouts of values -----------------------------------------------------------


@dataclass(frozen=True)
class ValueLayout:
    sg_layout: tuple[int, ...]
    sg_data: tuple[int, ...]


def value_layout(kernel: KernelIR, grid: Grid, value: ir.Value) -> ValueLayout:
    """``value``'s subgroup layout on ``grid``. A dim of the primary tile's class
    takes that axis's factor; the other axis is a broadcast on a remaining dim
    (the whole dim on each subgroup: a DPAS's K), and a value on neither axis's
    distributed dims is shared whole, ``[1, 1]``."""
    primary = kernel.primary
    classes = kernel.classes
    shape = _shape(value)
    rank = len(shape)
    factors: list[int | None] = [None] * rank
    data = list(shape)
    used = {"rows": False, "cols": False}
    for dim in range(rank):
        for axis, axis_dim, factor in (
            ("rows", 0, grid.rows),
            ("cols", 1, grid.cols),
        ):
            if classes.same((value, dim), (primary, axis_dim)):
                factors[dim] = factor
                used[axis] = factor > 1
                if shape[dim] % factor:
                    raise LayoutError(
                        f"{factor} subgroups do not divide a dim of {shape[dim]}"
                    )
                data[dim] = shape[dim] // factor
    if not any(used.values()) and grid.subgroups > 1:
        return ValueLayout((1,) * rank, tuple(shape))
    # The axis this value is not distributed on is a broadcast on the other dim.
    for dim in range(rank):
        if factors[dim] is None:
            distributed = next((d for d in range(rank) if factors[d] is not None), None)
            if distributed is None:
                factors[dim] = 1
                continue
            on_rows = classes.same((value, distributed), (primary, 0))
            factors[dim] = grid.cols if on_rows else grid.rows
    return ValueLayout(tuple(factors), tuple(data))


def _layout_text(
    layout: ValueLayout,
    inst_data: Sequence[int] | None = None,
    order: Sequence[int] | None = None,
) -> str:
    fields_ = [
        f"sg_layout = {list(layout.sg_layout)}",
        f"sg_data = {list(layout.sg_data)}",
    ]
    if inst_data is not None:
        fields_.append(f"inst_data = {list(inst_data)}")
    if order is not None:
        fields_.append(f"order = {list(order)}")
    return f"#xegpu.layout<{', '.join(fields_)}>"


# --- Roles of loaded tiles ----------------------------------------------------------


@dataclass
class Role:
    """How a loaded tile is used."""

    a: list[ir.Operation] = field(default_factory=list)
    b: list[ir.Operation] = field(default_factory=list)
    other: bool = False
    transposed: bool = False


def load_role(load: ir.Value) -> Role:
    """The DPAS operands (through ``vector.transpose`` and ``arith.truncf``) and
    other users of ``load``."""
    role = Role()
    pending = [load]
    while pending:
        value = pending.pop()
        for use in value.uses:
            owner = use.owner
            if owner.name == "xegpu.dpas" and use.operand_number == 0:
                role.a.append(owner.operation)
            elif owner.name == "xegpu.dpas" and use.operand_number == 1:
                role.b.append(owner.operation)
            elif owner.name == "vector.transpose":
                role.transposed = True
                pending.append(owner.results[0])
            elif owner.name == "arith.truncf":
                pending.append(owner.results[0])
            else:
                role.other = True
    return role


# --- The plan -------------------------------------------------------------------


@dataclass(frozen=True)
class Hints:
    source: str = "rule"
    sg_tile: tuple[int, ...] | None = None
    load_tiles: tuple[int, ...] | None = None
    prefetch_tiles: tuple[int, ...] | None = None
    prefetch_depths: tuple[int, ...] | None = None

    @classmethod
    def of(cls, func: ir.Operation) -> Hints:
        if HINTS_ATTR not in func.attributes:
            return cls()
        hints = ir.DictAttr(func.attributes[HINTS_ATTR])
        values: dict[str, object] = {}
        if "source" in hints:
            values["source"] = ir.StringAttr(hints["source"]).value
        for key in ("sg_tile", "load_tiles", "prefetch_tiles", "prefetch_depths"):
            if key in hints:
                values[key] = tuple(ir.DenseI64ArrayAttr(hints[key]))
        return cls(**values)


def _selector(dpas: ir.Operation, sg_tile: Sequence[int] | None) -> dict | None:
    """Lighthouse's parameters for ``dpas``, or ``None`` where it has none."""
    acc_shape = _shape(dpas.results[0])
    k_tile = _shape(dpas.operands[0])[1]
    try:
        return select_xegpu_gemm_params(
            dpas.opview,
            wg_tile=tuple(acc_shape),
            k_tile=k_tile,
            sg_tile=tuple(sg_tile) if sg_tile is not None else None,
        )
    except (KeyError, ValueError, StopIteration, NotImplementedError):
        return None


def _grid_of_tile(axes: Axes, sg_tile: Sequence[int]) -> Grid:
    rows, cols = sg_tile
    if rows == 0 or rows > axes.rows or axes.rows % rows:
        raise LayoutError(
            f"subgroup tile {list(sg_tile)} does not divide the {axes.rows} rows "
            "of the kernel's tile"
        )
    fy = 1
    if cols != 0:
        if axes.cols % cols:
            raise LayoutError(
                f"subgroup tile {list(sg_tile)} does not divide the {axes.cols} "
                "columns of the kernel's tile"
            )
        fy = axes.cols // cols
    grid = Grid(axes.rows // rows, fy)
    if grid.rows > 1 and not axes.free_rows:
        raise LayoutError(f"subgroup tile {list(sg_tile)}: the rows cannot be split")
    if grid.cols > 1 and not axes.free_cols:
        raise LayoutError(
            f"subgroup tile {list(sg_tile)}: the columns are contracted or reduced, "
            f"so the tile is [rows, 0] or [rows, {axes.cols}]"
        )
    if grid.subgroups > MAX_SUBGROUPS:
        raise LayoutError(
            f"subgroup tile {list(sg_tile)}: {grid.subgroups} subgroups exceed the "
            f"maximum of {MAX_SUBGROUPS}"
        )
    return grid


REDUCTION_SG_ROWS = 8
"""Rows per subgroup of a kernel that reduces rows and has no DPAS: lighthouse's
reduction class's, whose tiles and registers were measured."""
ROW_KERNEL_SG_ROWS = 16
"""Rows per subgroup of a kernel with DPAS and row reductions: lighthouse's
attention form (two 8-row DPAS blocks)."""


def _rows_only_grid(axes: Axes, rows: int) -> Grid:
    """All subgroups on the rows: ``rows`` each, or fewer that divide them, within
    the thread limit."""
    for candidate in range(min(rows, axes.rows), 0, -1):
        if axes.rows % candidate == 0 and candidate in (32, 16, 8, 4, 2, 1):
            fx = axes.rows // candidate
            if fx <= MAX_SUBGROUPS:
                return Grid(fx, 1)
    raise LayoutError(
        f"{axes.rows} rows do not split into at most {MAX_SUBGROUPS} subgroups"
    )


def _check_registers(kernel: KernelIR, grid: Grid) -> None:
    """A forced grid's DPAS tiles fit a subgroup: at most :data:`MAX_DPAS_PER_THREAD`
    parallel DPAS, and A, B and C in the register file."""
    for dpas in kernel.dpas:
        a, b, c = (
            value_layout(kernel, grid, value).sg_data
            for value in (dpas.operands[0], dpas.operands[1], dpas.results[0])
        )
        parallel = (c[0] // DPAS_ROWS[0]) * (c[1] // DPAS_COLS)
        if parallel > MAX_DPAS_PER_THREAD:
            raise LayoutError(
                f"a subgroup tile of {list(c)}: {parallel} parallel DPAS, the "
                f"hardware has {MAX_DPAS_PER_THREAD}"
            )
        bytes_ = 2 * (a[0] * a[1] + b[0] * b[1]) + 4 * c[0] * c[1]
        if bytes_ > REGISTER_BYTES:
            raise LayoutError(
                f"a subgroup tile of {list(c)}: its tiles take {bytes_} bytes of "
                f"{REGISTER_BYTES} in registers"
            )


def pick_grid(axes: Axes, hints: Hints) -> tuple[Grid, list[dict | None]]:
    """The kernel's grid and, with the selector, its parameters per DPAS."""
    kernel = axes.kernel
    if hints.sg_tile is not None:
        grid = _grid_of_tile(axes, hints.sg_tile)
        if hints.sg_tile[1] != 0:
            _check_registers(kernel, grid)
        tile = axes.tile(grid)
        forced = tile if hints.sg_tile[1] != 0 else None
        if hints.source == "selector" and kernel.dpas and not kernel.reductions:
            return grid, [_selector(d, forced) for d in kernel.dpas]
        return grid, [None] * len(kernel.dpas)
    if kernel.reductions:
        rows = ROW_KERNEL_SG_ROWS if kernel.dpas else REDUCTION_SG_ROWS
        return _rows_only_grid(axes, rows), [None] * len(kernel.dpas)
    if hints.source == "selector" and kernel.dpas:
        first = _selector(kernel.dpas[0], None)
        if first is not None:
            tile = (first["sg_m"], first["sg_n"])
            grid = _grid_of_tile(axes, tile)
            return grid, [first, *(_selector(d, tile) for d in kernel.dpas[1:])]
    return preferred_grid(axes, legal_grids(axes)), [None] * len(kernel.dpas)


def _dpas_rows(sg_rows: int) -> int:
    rows = next((r for r in DPAS_ROWS if sg_rows % r == 0), None)
    if rows is None:
        raise LayoutError(f"no DPAS fits a subgroup tile of {sg_rows} rows")
    return rows


def _largest_rows(size: int, limit: int) -> int:
    return next(r for r in (32, 16, 8, 4, 2, 1) if r <= limit and size % r == 0)


RULE_LOAD = (16, LOAD_MAX_COLS)
"""What the rule loads of a DPAS operand: 16 rows of 64 bytes, half a 2D block
read's limit. Measured on the B50, device time, 3 processes each (A, B tile; hand
is ``GEMM_SQUARE``, ``GEMM_STANDARD``): 4096^3 at 128x512: 32x32, 32x32 2921 us;
hand 2762; 16x32, 16x32 2582. 2048x4096x8192 at 256x256: 32x32, 32x32 2872; hand
2778; 16x32, 16x32 2687. Why reads of half the size win is not established."""


def _rule_load(data: Sequence[int]) -> tuple[int, int]:
    """The rule's tile of a DPAS operand whose subgroup tile is ``data``."""
    cols = RULE_LOAD[1] if data[1] % RULE_LOAD[1] == 0 else DPAS_COLS
    return _largest_rows(data[0], RULE_LOAD[0]), cols


def _selector_a_load(data: Sequence[int]) -> tuple[int, int]:
    """Where the selector has no load tiles (lighthouse's attention form)."""
    return _largest_rows(data[0], DPAS_COLS), DPAS_COLS


def _selector_b_load(data: Sequence[int]) -> tuple[int, int]:
    if data[0] % LOAD_MAX_ROWS == 0 and data[1] % LOAD_MAX_COLS == 0:
        return LOAD_MAX_ROWS, LOAD_MAX_COLS
    return DPAS_COLS, DPAS_COLS


def _check_block(tile: Sequence[int], data: Sequence[int], what: str) -> None:
    if any(d % t for d, t in zip(data, tile, strict=True)):
        raise LayoutError(f"{what} {list(tile)} does not divide its tile {list(data)}")
    if tile[0] > LOAD_MAX_ROWS or tile[1] > LOAD_MAX_COLS:
        raise LayoutError(
            f"{what} {list(tile)}: 2D block loads are at most "
            f"{LOAD_MAX_ROWS}x{LOAD_MAX_COLS}"
        )


def _prefetch_tile(shape: Sequence[int], threads: int) -> tuple[int, int]:
    """Lighthouse's order among legal prefetch tiles of a ``shape`` tile: most
    threads, up to the kernel's, then the most balanced grid."""
    found = []
    for rows in range(PREFETCH_ROWS[0], PREFETCH_ROWS[1] + 1):
        for cols in range(PREFETCH_COLS[0], PREFETCH_COLS[1] + 1):
            if shape[0] % rows or shape[1] % cols:
                continue
            grid = (shape[0] // rows, shape[1] // cols)
            if grid[0] * grid[1] > threads:
                continue
            found.append(((rows, cols), grid[0] * grid[1], -abs(grid[0] - grid[1])))
    if not found:
        raise LayoutError(f"no prefetch tile fits a tile of {list(shape)}")
    found.sort(key=itemgetter(1, 2), reverse=True)
    return found[0][0]


@dataclass
class Plan:
    """What :func:`plan_layouts` decided: anchors and prefetches."""

    grid: Grid
    threads: int
    anchors: list[tuple[ir.Operation, str, str]] = field(default_factory=list)
    """``(op, attribute, layout text)``."""
    prefetches: list[tuple[ir.Operation, list[int]]] = field(default_factory=list)
    """``(load, [depth, sg_layout..., sg_data..., inst_data...])``, in the order
    to insert them."""
    cleanup: bool = False
    """Clean the kernel up afterwards."""
    licm: bool = False
    """Hoist out of the DPAS loops afterwards (what prefetches leave in them)."""


def plan_layouts(kernel: KernelIR, hints: Hints) -> Plan:
    """The anchors of a kernel with a 2D tile: see the module docstring."""
    axes = axes_of(kernel)
    grid, selected = pick_grid(axes, hints)
    plan = Plan(grid, grid.subgroups * SUBGROUP_SIZE)
    by_dpas = dict(zip(kernel.dpas, selected, strict=True))
    for dpas in kernel.dpas:
        _plan_dpas(kernel, grid, dpas, plan)
    _plan_stores(kernel, grid, plan)
    done: set[ir.Operation] = set()
    for dpas in kernel.dpas:
        for index, operand in enumerate(dpas.operands[:2]):
            access = _load_of(kernel, operand)
            if access is None or access.op in done:
                continue
            done.add(access.op)
            _plan_operand_load(kernel, grid, hints, access, index, by_dpas[dpas], plan)
    for access in kernel.loads:
        if access.op not in done:
            _plan_other_load(kernel, grid, hints, access, plan)
    plan.cleanup = bool(plan.prefetches)
    plan.licm = bool(plan.prefetches) and not kernel.reductions
    _check_hints_apply(kernel, hints)
    return plan


def _check_hints_apply(kernel: KernelIR, hints: Hints) -> None:
    """A hint with nothing to apply to is a mistake, not a preference."""
    operands = {
        load.op
        for dpas in kernel.dpas
        for operand in dpas.operands[:2]
        if (load := _load_of(kernel, operand)) is not None
    }
    in_loops = [
        access
        for access in kernel.loads
        if access.op in operands and access.op.parent.name == "scf.for"
    ]
    if hints.prefetch_tiles is not None or hints.prefetch_depths is not None:
        if not in_loops:
            raise LayoutError(
                "prefetch tiles and depths apply to loads of DPAS operands in a "
                "K loop; this kernel has none"
            )
    if hints.load_tiles is not None:
        others = [access for access in kernel.loads if access.op not in operands]
        if len(hints.load_tiles) == 4 and not operands:
            raise LayoutError(
                "load tiles [a_m, a_k, b_k, b_n] apply to DPAS operands; this kernel "
                "has none"
            )
        if len(hints.load_tiles) == 2 and not others:
            raise LayoutError(
                "a load tile [m, n] applies to tiles that are no DPAS operands; "
                "this kernel has none"
            )


def _load_of(kernel: KernelIR, operand: ir.Value) -> _Access | None:
    value = operand
    while True:
        op = _defining(value)
        if op is None:
            return None
        if op.name == "xegpu.load_nd":
            return next(a for a in kernel.loads if a.op == op)
        if op.name not in ("vector.transpose", "arith.truncf"):
            return None
        value = op.operands[0]


def _plan_dpas(kernel: KernelIR, grid: Grid, dpas: ir.Operation, plan: Plan) -> None:
    a, b, result = dpas.operands[0], dpas.operands[1], dpas.results[0]
    a_layout = value_layout(kernel, grid, a)
    b_layout = value_layout(kernel, grid, b)
    c_layout = value_layout(kernel, grid, result)
    rows = _dpas_rows(c_layout.sg_data[0])
    if c_layout.sg_data[1] % DPAS_COLS or a_layout.sg_data[1] % DPAS_COLS:
        raise LayoutError(
            f"DPAS tiles {list(a_layout.sg_data)}, {list(c_layout.sg_data)}: columns "
            f"and K must be multiples of {DPAS_COLS} (no partial tiles)"
        )
    plan.anchors.extend(
        [
            (
                dpas,
                "layout_a",
                _layout_text(a_layout, (rows, DPAS_COLS), _operand_order(a)),
            ),
            (
                dpas,
                "layout_b",
                _layout_text(b_layout, (DPAS_COLS, DPAS_COLS), _operand_order(b)),
            ),
            (dpas, "layout_cd", _layout_text(c_layout, (rows, DPAS_COLS))),
        ]
    )


def _operand_order(value: ir.Value) -> tuple[int, int] | None:
    """A DPAS operand through a ``vector.transpose`` of a loaded tile has its lanes
    in the transposed order: the load's block reads it that way."""
    op = _defining(value)
    if op is None or op.name != "vector.transpose":
        return None
    source = _defining(op.operands[0])
    return (1, 0) if source is not None and source.name == "xegpu.load_nd" else None


def _plan_stores(kernel: KernelIR, grid: Grid, plan: Plan) -> None:
    for access in kernel.stores:
        layout = value_layout(kernel, grid, access.value)
        if len(layout.sg_data) != 2:
            raise LayoutError(f"a {len(layout.sg_data)}D tile store_nd")
        plan.anchors.append(
            (
                access.op,
                "layout",
                _layout_text(layout, block_store_tile(layout.sg_data)),
            )
        )
    primary = kernel.primary
    for op in kernel.scatter_stores:
        _plan_row_store(kernel, grid, op, primary, plan)


def _plan_row_store(
    kernel: KernelIR, grid: Grid, op: ir.Operation, primary: ir.Value, plan: Plan
) -> None:
    """A ``[rows]`` or ``[rows, 1]`` scatter store of a row result (or any store of
    a vector on the rows axis), as a slice of the primary tile's layout or with
    its reduced dim's data set to 1."""
    shape = _shape(op.operands[0])
    tile = value_layout(kernel, grid, primary)
    classes = kernel.classes
    value = op.operands[0]
    if len(shape) == 1 and classes.same((value, 0), (primary, 0)):
        text = _layout_text(tile)
        plan.anchors.append((op, "layout", f"#xegpu.slice<{text}, dims = [1]>"))
    elif len(shape) == 2 and shape[1] == 1 and classes.same((value, 0), (primary, 0)):
        keep = ValueLayout(tile.sg_layout, (tile.sg_data[0], 1))
        plan.anchors.append((op, "layout", _layout_text(keep)))


def _plan_operand_load(
    kernel: KernelIR,
    grid: Grid,
    hints: Hints,
    access: _Access,
    index: int,
    selected: dict | None,
    plan: Plan,
) -> None:
    value = access.value
    layout = value_layout(kernel, grid, value)
    role = load_role(value)
    if role.a and role.b:
        raise LayoutError("a loaded tile used as both DPAS operands")
    if role.b and role.other:
        raise LayoutError(
            "a DPAS B tile with other users: its VNNI load gives them wrong elements"
        )
    # Hints are in the layout's [m, k] / [k, n] form, then the source's.
    forced: tuple[int, int] | None = None
    if hints.load_tiles is not None and len(hints.load_tiles) == 4:
        forced = hints.load_tiles[2 * index], hints.load_tiles[2 * index + 1]
    elif selected is not None:
        forced = (
            (selected["load_a_m"], selected["load_a_k"])
            if index == 0
            else (selected["load_b_k"], selected["load_b_n"])
        )
    if role.transposed:
        # A hint for this role goes to the role's plain loads if it has some (K and
        # V of attention): a transposed read has its own tile.
        plain = any(
            not load_role(other.value).transposed
            for other in kernel.loads
            if other.op != access.op and _operand_index(other, kernel) == index
        )
        tile = forced if forced is not None and not plain else TRANSPOSED_LOAD
        if tuple(tile) not in TRANSPOSED_LOADS:
            raise LayoutError(
                f"a transposed load takes {[list(t) for t in TRANSPOSED_LOADS]} "
                f"tiles, not {list(tile)}"
            )
        order: tuple[int, int] | None = (0, 1)
    elif role.other:
        # Its other users take 16-column reads: see LOAD_COLS.
        tile = block_load_tile(layout.sg_data, (value_bits(value),))
        order = None
    else:
        order = None
        if forced is not None:
            tile = forced
        elif hints.source == "rule" and not _has_shared_operand(kernel, grid):
            tile = _rule_load(layout.sg_data)
            if output_bound(kernel):
                # Operands are little of the traffic, and DPAS-sized reads win: on
                # L1 7, B50, device time, A, B tiles 8x16, 16x16 14277 us; 8x16,
                # 16x32 14896; 8x32, 16x32 15388.
                limit = DPAS_ROWS[0] if index == 0 else DPAS_COLS
                tile = (min(tile[0], limit), DPAS_COLS)
            elif any(_operand_order(dpas.operands[0]) for dpas in role.b):
                # B next to a transposed A: on 16, B50, device time, B 16x16 2608
                # us, 32x32 2561, 16x32 2833.
                tile = (min(tile[0], DPAS_COLS), DPAS_COLS)
        elif index == 0:
            tile = _selector_a_load(layout.sg_data)
        else:
            tile = _selector_b_load(layout.sg_data)
    _check_block(tile, layout.sg_data, "load tile")
    plan.anchors.append((access.op, "layout", _layout_text(layout, tile, order)))
    _plan_prefetch(kernel, grid, hints, access, layout, index, selected, plan)


def _has_shared_operand(kernel: KernelIR, grid: Grid) -> bool:
    """Whether a DPAS operand is read whole by every subgroup (``[1, 1]``). Those
    kernels keep lighthouse's attention loads (:func:`_selector_a_load`,
    :func:`_selector_b_load`): on 97, B50, device time, Q, V tiles 16x16, 32x32 1820
    us; 16x32, 16x32 1876; 16x32, 32x32 1929; 8x16, 32x32 1896."""
    return any(
        value_layout(kernel, grid, dpas.operands[1]).sg_layout == (1, 1)
        for dpas in kernel.dpas
    )


def _operand_index(access: _Access, kernel: KernelIR) -> int | None:
    """Which DPAS operand (0: A, 1: B) the load ``access`` feeds."""
    for dpas in kernel.dpas:
        for index, operand in enumerate(dpas.operands[:2]):
            load = _load_of(kernel, operand)
            if load is not None and load.op == access.op:
                return index
    return None


def value_bits(value: ir.Value) -> int:
    return ir.VectorType(value.type).element_type.width


def _plan_prefetch(
    kernel: KernelIR,
    grid: Grid,
    hints: Hints,
    access: _Access,
    layout: ValueLayout,
    index: int,
    selected: dict | None,
    plan: Plan,
) -> None:
    """A load of a DPAS operand in a loop over K steps, with its tile shared by all
    subgroups (``[1, 1]``) or distributed."""
    parent = access.op.parent
    if parent is None or parent.name != "scf.for":
        return
    depth = 1
    if selected is not None:
        depth = selected["prefetch_a_nb" if index == 0 else "prefetch_b_nb"]
    if hints.prefetch_depths is not None:
        depth = hints.prefetch_depths[index]
    if depth == 0:
        return
    shape = _shape(access.value)
    shared = layout.sg_layout == (1,) * len(shape)
    tile: tuple[int, int]
    if hints.prefetch_tiles is not None:
        tile = (
            hints.prefetch_tiles[2 * index],
            hints.prefetch_tiles[2 * index + 1],
        )
    elif selected is not None:
        tile = (
            (selected["prefetch_a_m"], selected["prefetch_a_k"])
            if index == 0
            else (selected["prefetch_b_k"], selected["prefetch_b_n"])
        )
    elif shared and all(
        s % t == 0 for s, t in zip(shape, SHARED_PREFETCH, strict=True)
    ):
        tile = SHARED_PREFETCH
    else:
        tile = _prefetch_tile(shape, grid.subgroups)
    if any(s % t for s, t in zip(shape, tile, strict=True)):
        raise LayoutError(f"prefetch tile {list(tile)} does not divide {list(shape)}")
    if not (
        PREFETCH_ROWS[0] <= tile[0] <= PREFETCH_ROWS[1]
        and PREFETCH_COLS[0] <= tile[1] <= PREFETCH_COLS[1]
    ):
        raise LayoutError(
            f"prefetch tile {list(tile)}: 2D block prefetches are "
            f"{PREFETCH_ROWS[0]}..{PREFETCH_ROWS[1]} rows of "
            f"{PREFETCH_COLS[0]}..{PREFETCH_COLS[1]} columns"
        )
    inst = tile if shared else PREFETCH_INST
    plan.prefetches.append(
        (
            access.op,
            [
                depth,
                *(s // t for s, t in zip(shape, tile, strict=True)),
                *tile,
                *inst,
            ],
        )
    )


def _plan_other_load(
    kernel: KernelIR, grid: Grid, hints: Hints, access: _Access, plan: Plan
) -> None:
    value = access.value
    layout = value_layout(kernel, grid, value)
    forced = None
    if hints.load_tiles is not None and len(hints.load_tiles) == 2:
        forced = hints.load_tiles
    tile = block_load_tile(layout.sg_data, (value_bits(value),), forced)
    plan.anchors.append((access.op, "layout", _layout_text(layout, tile)))


def plan_elementwise(func: ir.Operation, hints: Hints) -> Plan | None:
    """``xegpu_elementwise``'s layout, if ``func`` is an elementwise kernel."""
    tiles = find_tiles(func)
    if isinstance(tiles, str):
        return None
    if hints.prefetch_tiles is not None or hints.prefetch_depths is not None:
        raise LayoutError(
            "prefetch tiles and depths apply to loads of DPAS operands in a K loop; "
            "this kernel has none"
        )
    load_tile = hints.load_tiles
    if load_tile is not None and len(load_tile) not in (1, 2):
        raise LayoutError(
            "load tiles [a_m, a_k, b_k, b_n] apply to DPAS operands; this kernel has "
            "none"
        )
    layout = elementwise_layout(
        tiles.wg_tile,
        tiles.load_bits,
        tiles.store_bits,
        sg_tile=hints.sg_tile,
        load_tile=load_tile,
        contiguous=tiles.contiguous,
    )
    grid = Grid(layout.subgroups, 1)
    plan = Plan(grid, layout.threads, cleanup=True)
    sg = ValueLayout(layout.sg_layout, layout.sg_tile)
    for ops, inst in (
        (tiles.loads, layout.load_tile),
        (tiles.stores, layout.store_tile),
    ):
        for op in ops:
            plan.anchors.append((op, "layout", _layout_text(sg, inst)))
    return plan


# --- Writing it -------------------------------------------------------------------


def set_threads(module: ir.Operation, func: ir.Operation, threads: int) -> None:
    """The kernel's thread count in the IR: ``known_block_size`` on ``func``, and
    the threads operand of the ``gpu.launch_func`` ops of it in ``module``."""
    func.attributes["known_block_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
    symbol = ir.StringAttr(func.attributes["sym_name"]).value
    module_op = func.parent.operation
    module_name = ir.StringAttr(module_op.attributes["sym_name"]).value
    reference = f"@{module_name}::@{symbol}"
    index_type = ir.IndexType.get()

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.launch_func" and str(op.attributes["kernel"]) == reference:
            segments = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
            threads_operand = segments[0] + 3  # after async deps, blocks x, y, z
            with ir.InsertionPoint(op):
                constant = ir.Operation.create(
                    "arith.constant",
                    results=[index_type],
                    attributes={"value": ir.IntegerAttr.get(index_type, threads)},
                )
            op.operands[threads_operand] = constant.results[0]
        return ir.WalkResult.ADVANCE

    module.walk(visit, ir.WalkOrder.PRE_ORDER)


def annotate_layouts(module: ir.Operation, func: ir.Operation) -> Plan:
    """Write the layouts of the kernel ``func`` of ``module``: anchors, thread
    count, and :data:`PREFETCH_ATTR` numbering the loads to prefetch (the plan's
    ``prefetches``, for the caller's schedule; :func:`clear_tags` removes it).
    :class:`LayoutError` if none is legal."""
    hints = Hints.of(func)
    if hints.source not in ("rule", "selector", "upstream"):
        raise LayoutError(f"unknown layout source {hints.source!r}")
    plan = plan_elementwise(func, hints)
    if plan is None:
        kernel = analyze(func)
        plan = plan_layouts(kernel, hints)
        if hints.source == "upstream":
            plan = Plan(plan.grid, _upstream_threads(kernel, plan.threads))
    if HINTS_ATTR in func.attributes:
        del func.attributes[HINTS_ATTR]
    if hints.source == "upstream":
        # A baseline: only the count; the lowering's propagation decides the rest.
        plan = Plan(plan.grid, plan.threads)
    i64 = ir.IntegerType.get_signless(64)
    for op, name, text in plan.anchors:
        op.attributes[name] = ir.Attribute.parse(text)
    for number, (op, _) in enumerate(plan.prefetches):
        op.attributes[PREFETCH_ATTR] = ir.IntegerAttr.get(i64, number)
    set_threads(module, func, plan.threads)
    return plan


def _upstream_threads(kernel: KernelIR, fallback: int) -> int:
    """The baseline's thread count: the most subgroups whose tile DPAS and 2D blocks
    can take, pinned dims or not, as upstream's propagation distributes across
    them (a reduction through SLM and a barrier)."""
    rows, cols = _shape(kernel.primary)
    counts = [
        fx * fy
        for fx in _divisors(rows)
        for fy in _divisors(cols)
        if fx * fy <= MAX_SUBGROUPS
        and (cols // fy) % BLOCK_COLS == 0
        and (rows // fx) % DPAS_ROWS[0] == 0
    ]
    return max(counts) * SUBGROUP_SIZE if counts else fallback


def clear_tags(func: ir.Operation) -> None:
    """Remove :data:`PREFETCH_ATTR` from the loads of ``func``."""
    for op in _walk(func):
        if PREFETCH_ATTR in op.attributes:
            del op.attributes[PREFETCH_ATTR]


# --- The check -------------------------------------------------------------------

_LAYOUT_ATTRS: dict[str, tuple[str, ...]] = {
    "xegpu.dpas": ("layout_a", "layout_b", "layout_cd"),
    "xegpu.load_nd": ("layout",),
    "xegpu.store_nd": ("layout",),
    "xegpu.prefetch_nd": ("layout",),
    "xegpu.load": ("layout",),
    "xegpu.store": ("layout",),
}


def check_layouts(func: ir.Operation, *, require_all: bool = True) -> list[str]:
    """What is wrong with the anchors of ``func``, as messages: a DPAS or memory op
    without a layout (``require_all``), or a layout whose subgroup tile does not
    fit its value, its grid the kernel's subgroups, or its ``inst_data`` the
    subgroup tile."""
    problems: list[str] = []
    block = _known_block(func)
    subgroups = block // SUBGROUP_SIZE if block is not None else None

    def visit(op: ir.Operation) -> ir.WalkResult:
        names = _LAYOUT_ATTRS.get(op.name)
        if names is None:
            return ir.WalkResult.ADVANCE
        for name in names:
            if name not in op.attributes:
                if require_all:
                    problems.append(f"'{op.name}' has no {name}")
                continue
            text = str(op.attributes[name])
            # A prefetch tile's grid may exceed the subgroups: what they do not
            # reach is not prefetched, which is slow, not wrong (lighthouse's
            # selector does this on 6, 7 and 9).
            problems.extend(
                f"'{op.name}' {name}: {problem}"
                for problem in _check_attr(
                    text, None if op.name == "xegpu.prefetch_nd" else subgroups
                )
            )
        return ir.WalkResult.ADVANCE

    func.walk(visit, ir.WalkOrder.PRE_ORDER)
    return problems


def _known_block(func: ir.Operation) -> int | None:
    if "known_block_size" not in func.attributes:
        return None
    return ir.DenseI32ArrayAttr(func.attributes["known_block_size"])[0]


def _ints(text: str, key: str) -> list[int] | None:
    start = text.find(f"{key} = [")
    if start < 0:
        return None
    start += len(key) + 4
    end = text.index("]", start)
    return [int(part) for part in text[start:end].split(",")]


def _check_attr(text: str, subgroups: int | None) -> list[str]:
    problems = []
    sg_layout, sg_data, inst = (
        _ints(text, k) for k in ("sg_layout", "sg_data", "inst_data")
    )
    if sg_layout is None or sg_data is None:
        return [f"no subgroup layout in {text}"]
    if subgroups is not None and math.prod(sg_layout) > subgroups:
        problems.append(f"{sg_layout} is more than the kernel's {subgroups} subgroups")
    if inst is not None and any(s % i for s, i in zip(sg_data, inst, strict=False)):
        problems.append(f"inst_data {inst} does not divide sg_data {sg_data}")
    return problems


def check_module(module: ir.Module, names: Sequence[str]) -> list[str]:
    """What is wrong with the layouts of the GPU modules ``names`` of ``module``:
    their anchors as written (DPAS, ``load_nd``, ``store_nd``, ``prefetch_nd``
    must have one), then, on a copy, after the lowering's own subgroup propagation
    has completed the rest: every memory op must have a layout then, and a
    warning or error of the pass is a problem."""
    if not names:
        return []
    problems: list[str] = []
    for gpu_module in _gpu_modules(module, names):
        for func in _funcs(gpu_module):
            problems.extend(check_anchors(func))
    if problems:
        return problems
    copy = ir.Module.parse(str(module), module.context)
    diagnostics: list[str] = []

    def on_diagnostic(diagnostic: ir.Diagnostic) -> bool:
        if diagnostic.severity in (
            ir.DiagnosticSeverity.ERROR,
            ir.DiagnosticSeverity.WARNING,
        ):
            diagnostics.append(diagnostic.message)
        return True

    handler = module.context.attach_diagnostic_handler(on_diagnostic)
    try:
        passes = PassManager.parse(
            "builtin.module(gpu.module(xegpu-propagate-layout{layout-kind=subgroup}))",
            module.context,
        )
        passes.run(copy.operation)
    except ir.MLIRError as exc:
        diagnostics.append(str(exc))
    finally:
        handler.detach()
    problems.extend(diagnostics)
    for gpu_module in _gpu_modules(copy, names):
        for func in _funcs(gpu_module):
            problems.extend(check_layouts(func))
    return problems


def _gpu_modules(module: ir.Module, names: Sequence[str]) -> list[ir.Operation]:
    return [
        op.operation
        for op in module.body.operations
        if op.operation.name == "gpu.module"
        and ir.StringAttr(op.operation.attributes["sym_name"]).value in names
    ]


def _funcs(gpu_module: ir.Operation) -> list[ir.Operation]:
    return [
        op.operation
        for op in gpu_module.regions[0].blocks[0].operations
        if op.operation.name == "gpu.func"
    ]


def check_anchors(func: ir.Operation) -> list[str]:
    """:func:`check_layouts`, requiring a layout on what must be anchored here:
    the propagation's defaults are for ``xegpu.load`` and ``xegpu.store`` only."""
    problems = check_layouts(func, require_all=False)
    for op in _walk(func):
        if op.name in _ANCHORED:
            problems.extend(
                f"'{op.name}' has no {name}"
                for name in _LAYOUT_ATTRS[op.name]
                if name not in op.attributes
            )
    return problems
