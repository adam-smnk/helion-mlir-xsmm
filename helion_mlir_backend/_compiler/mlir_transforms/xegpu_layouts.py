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
   must a reduced dim, but in a one-row kernel (:func:`plan_row_reduction`), whose
   columns are split across the subgroups with upstream's cross-subgroup reduction
   through SLM and ``gpu.barrier``. The rest are free.
3. **Grids** (:func:`legal_grids`): the output's two dims are the grid's axes.
   A grid ``(fx, fy)`` is legal if its factors divide every value of their class,
   leave DPAS and 2D block sizes on each subgroup, and fit the thread limit.
4. **Preference**: legal hints, then a rule (:func:`preferred_grid`, :func:`_rule_load`).
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
from dataclasses import replace
import logging
import math
from operator import itemgetter
from typing import TYPE_CHECKING

from mlir import ir
from mlir.passmanager import PassManager

from ..mlir.support.debug import LOWERING_LOG_ATTR
from .xegpu_elementwise import BLOCK_COLS
from .xegpu_elementwise import MAX_SG_ELEMENTS
from .xegpu_elementwise import MAX_SUBGROUPS
from .xegpu_elementwise import SUBGROUP_SIZE
from .xegpu_elementwise import block_load_tile
from .xegpu_elementwise import block_store_tile
from .xegpu_elementwise import elementwise_layout
from .xegpu_elementwise import find_tiles
from .xegpu_elementwise import is_contiguous_1d
from .xegpu_tile_forms import buffer_root
from .xegpu_tile_forms import expand_elementwise_contractions
from .xegpu_tile_forms import hoist_transposes
from .xegpu_tile_forms import split_reads
from .xegpu_tile_forms import stored_buffers

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

_HINT_KEYS = ("sg_tile", "load_tiles", "prefetch_tiles", "prefetch_depths")
HINTS_ATTR = "xegpu_layouts.hints"
"""Dictionary on a ``gpu.func``, read and removed by :func:`annotate_layouts`:

- ``sg_tile``: ``[rows, cols]``; ``[rows, 0]``: all subgroups on the rows; ``[1, cols]``
  of a one-row tile (:func:`plan_row_reduction`);
- ``load_tiles``: ``[a_m, a_k, b_k, b_n]`` of DPAS operands, ``[m, n]`` of other
  tiles;
- ``prefetch_tiles``: ``[a_m, a_k, b_k, b_n]``, as the operands are stored;
- ``prefetch_depths``: ``[a, b]``, 0 for no prefetch.

The arrays are ``DenseI64ArrayAttr``. They are hints: :func:`plan_kernel` drops one
that doesn't fit, with a log line; a legal one is honoured."""
PREFETCH_ATTR = "xegpu_layouts.prefetch"
"""Set by :func:`annotate_layouts` on a ``xegpu.load_nd`` to prefetch: its index
in the plan's ``prefetches``, for the schedule that inserts it."""

# The target: a mirror of xegpu::uArch (uArch/IntelGpuXe2.h, uArchBase.h), which
# has no Python bindings, for Xe2 with the large register file.
DPAS_ROWS = (8, 4, 2, 1)
"""``getSupportedM``: 1..8 for 16-bit floats. Powers of two here: the layouts
need rows that divide a subgroup tile evenly."""
MIN_DPAS_ROWS = 2
"""The fewest rows of a subgroup's DPAS this module lays out. A 1-row DPAS loses the row
dim in the lowering: ``ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED`` and an abort at launch
(attention of 3, 5, 7, 9 or 15 query rows on the B50)."""
DPAS_COLS = 16
"""``getSupportedN``, and K of a 16-bit A or B (two 8-element pairs)."""
BLOCK_ROWS = (1, 2, 4, 8, 16, 32)
"""``getBlockWidthHeightCount``: the rows of a 2D block read."""
VNNI_MIN_ROWS = 16
"""``getBlockWidthHeightCount`` with transform, 16-bit elements: 16 or 32 rows."""
LOAD_MAX_ROWS = 32
LOAD_MAX_COLS = 32
"""A 2D block load of 16-bit elements: at most 32 rows of 64 bytes."""
TRANSPOSED_LOAD = (16, 16)
"""The one transposed 2D block load of a DPAS operand (32-bit pairs: 8 wide). Upstream's
``getBlockWidthHeightCount`` also allows 32 rows, and every row is wrong on the B50 whatever
the anchor says (normalized error 1.3 against 2.9e-3 for 16 rows;
:data:`.xevm_lane_reductions._WRONG_TRANSPOSED_READ`). 16 rows are also the choice on speed:
with the order put right by hand, 32 rows gained nothing on the B50, and neither did the read
without those moves, on wrong rows."""
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


class HintError(LayoutError):
    """A hint that can't apply, doesn't fit the IR or is illegal: the annotator drops
    it and logs why (:func:`plan_kernel`); the kernel doesn't fail for it."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(reason)
        self.key = key


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


def _tie_accesses(
    classes: DimClasses, accesses: list[_Access], func: ir.Operation
) -> None:
    """Accesses of one buffer in one tile shape, at one offset per dim (or a loop
    variable in both, walking the same dim), are the same subgroup's data: a tile
    a subgroup stores must be the one it loads, and no subgroup waits for
    another's store inside a workgroup. Two reads of one tile of a buffer the kernel
    doesn't store to (:func:`.xegpu_tile_forms.split_reads`) take the same data whichever
    subgroup reads it, so they needn't be distributed alike."""
    stored, unknown = stored_buffers(func)
    by_source: dict[ir.Value, list[_Access]] = {}
    for access in accesses:
        by_source.setdefault(access.source, []).append(access)
    for source, group in by_source.items():
        for other in group[1:]:
            first = group[0]
            if _shape(first.value) != _shape(other.value):
                continue
            if (
                not unknown
                and buffer_root(source) not in stored
                and _same_read(first, other)
            ):
                continue
            for dim in range(len(_shape(first.value))):
                if _same_offset(first.offsets, other.offsets, dim):
                    classes.tie(first.value, dim, other.value, dim)


def _same_read(a: _Access, b: _Access) -> bool:
    return (
        a.op.name == b.op.name == "xegpu.load_nd"
        and len(a.offsets) == len(b.offsets)
        and all(x == y for x, y in zip(a.offsets, b.offsets, strict=True))
    )


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
                raise LayoutError(_contract_refusal(op))
            if name != "vector.step":
                raise LayoutError(f"'{name}' on tile vectors")
    _tie_accesses(classes, [*loads, *stores], func)
    primary = next(
        (a.value for a in reversed(stores) if len(_shape(a.value)) == 2),
        reductions[0].operands[0] if reductions else None,
    )
    return KernelIR(
        func, classes, dpas, loads, stores, reductions, scatter_stores, primary
    )


def _contract_refusal(op: ir.Operation) -> str:
    message = "a 'vector.contract' that is no matmul: no XeGPU op takes it"
    if len(_shape(op.operands[0]) or ()) == 1:
        return (
            f"{message}; a GEMM of M = 1 has a row for A: pad M to 2 rows in the model"
        )
    return message


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
    # A transposed read takes 16 rows of the tile as stored, whichever dim they are.
    transposed_on_rows = any(
        _on_axes(axes.kernel, a.value)[0]
        for a in axes.kernel.loads
        if load_role(a.value).transposed and _on_axes(axes.kernel, a.value)[0]
    )
    for fx in _divisors(axes.rows) if free_rows else [1]:
        for fy in _divisors(axes.cols) if free_cols else [1]:
            if fx * fy > MAX_SUBGROUPS:
                continue
            sg_rows, sg_cols = axes.rows // fx, axes.cols // fy
            if sg_cols % BLOCK_COLS:
                continue
            if has_dpas and not any(
                r >= MIN_DPAS_ROWS and sg_rows % r == 0 for r in DPAS_ROWS
            ):
                continue
            if transposed_on_rows and sg_rows % TRANSPOSED_LOAD[0]:
                continue
            if fx > 1 and sg_rows % 2 and sg_rows != 1:
                continue
            grids.append(Grid(fx, fy))
    return grids


MIN_SG_ELEMENTS = 32 * 32
MIN_FULL_SUBGROUPS = 16
"""A tile is at least :data:`MIN_SG_ELEMENTS` as long as :data:`MIN_FULL_SUBGROUPS` remain
(lighthouse's ``MIN_NB_THREADS``), and below that as many subgroups as fit."""
TARGET_SG_ROWS = 32
"""What :func:`preferred_grid` looks for: 32 rows per subgroup. A subgroup reads a
tile's rows from A's strided rows, and 64 of them stalled the memory pipe on the
B50 at K = 8192 (EU stall sampling, L1 2)."""


OUTPUT_BOUND_SG_TILE = (8, 64)
"""What :func:`preferred_grid` looks for in a kernel whose output traffic dominates
(:func:`output_bound`): each subgroup stores 8 rows of 128 bytes, even with fewer
subgroups. The fastest of the subgroup tiles tried on the B50, at 32768x32768x64."""
OUTPUT_BOUND_FACTOR = 32
"""How far output elements must exceed operand elements for that: on the B50, 8x64
subgroups lose to 32x32 below this factor and win or tie from it."""


def output_bound(kernel: KernelIR) -> bool:
    """Whether the kernel writes far more output elements than it reads operand
    elements, ``M * N >= OUTPUT_BOUND_FACTOR * K * (M + N)``: a DPAS's loop bound for K
    and the stored buffer's last two dims for M and N, when all are static."""
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
    return min(m, n, k) > 0 and m * n >= OUTPUT_BOUND_FACTOR * k * (m + n)


def preferred_grid(axes: Axes, grids: Sequence[Grid]) -> Grid:
    """The rule's pick among legal ``grids``. A kernel bound by its output takes
    :data:`OUTPUT_BOUND_SG_TILE` if legal. Else: tiles of :data:`MIN_SG_ELEMENTS` or more
    where that leaves :data:`MIN_FULL_SUBGROUPS`; tiles of whole DPAS depth (at least 8
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
    ]
    if not fitting:
        _log_line(
            axes.kernel.func,
            f"XeGPU layouts of {_func_name(axes.kernel.func)}: no subgroup tile of a "
            f"{axes.rows}x{axes.cols} workgroup tile leaves registers for the operands, "
            "so it will spill and run far slower: take smaller block sizes",
        )
        fitting = list(grids)
    # Tiles of 32x32 or more where that still leaves many threads: 128x128 at most
    # 16 subgroups, which beat 32 subgroups of 32x16 on the B50.
    fitting = [
        grid
        for grid in fitting
        if grid.subgroups >= MIN_FULL_SUBGROUPS
        and axes.tile(grid)[0] >= DPAS_ROWS[0]
        and axes.tile(grid)[0] * axes.tile(grid)[1] >= MIN_SG_ELEMENTS
    ] or fitting

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
    sg_tile: tuple[int, ...] | None = None
    load_tiles: tuple[int, ...] | None = None
    prefetch_tiles: tuple[int, ...] | None = None
    prefetch_depths: tuple[int, ...] | None = None

    @classmethod
    def of(cls, func: ir.Operation) -> Hints:
        if HINTS_ATTR not in func.attributes:
            return cls()
        hints = ir.DictAttr(func.attributes[HINTS_ATTR])
        return cls(
            **{
                key: tuple(ir.DenseI64ArrayAttr(hints[key]))
                for key in _HINT_KEYS
                if key in hints
            }
        )


def _grid_of_tile(axes: Axes, sg_tile: Sequence[int]) -> Grid:
    rows, cols = sg_tile
    if rows == 0 or rows > axes.rows or axes.rows % rows:
        raise HintError(
            "sg_tile",
            f"subgroup tile {list(sg_tile)} does not divide the {axes.rows} rows "
            "of the kernel's tile",
        )
    fy = 1
    if cols != 0:
        if axes.cols % cols:
            raise HintError(
                "sg_tile",
                f"subgroup tile {list(sg_tile)} does not divide the {axes.cols} "
                "columns of the kernel's tile",
            )
        fy = axes.cols // cols
    grid = Grid(axes.rows // rows, fy)
    if grid.rows > 1 and not axes.free_rows:
        raise HintError(
            "sg_tile", f"subgroup tile {list(sg_tile)}: the rows cannot be split"
        )
    if grid.cols > 1 and not axes.free_cols:
        raise HintError(
            "sg_tile",
            f"subgroup tile {list(sg_tile)}: the columns are contracted or reduced, "
            f"so the tile is [rows, 0] or [rows, {axes.cols}]",
        )
    if grid.subgroups > MAX_SUBGROUPS:
        raise HintError(
            "sg_tile",
            f"subgroup tile {list(sg_tile)}: {grid.subgroups} subgroups exceed the "
            f"maximum of {MAX_SUBGROUPS}",
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
    the thread limit, and enough for a DPAS (:data:`MIN_DPAS_ROWS`)."""
    least = MIN_DPAS_ROWS if axes.kernel.dpas else 1
    transposed_on_rows = any(
        _on_axes(axes.kernel, a.value)[0]
        for a in axes.kernel.loads
        if load_role(a.value).transposed
    )
    for candidate in (32, 16, 8, 4, 2, 1):
        if candidate > rows or candidate < least or axes.rows % candidate:
            continue
        if transposed_on_rows and candidate % TRANSPOSED_LOAD[0]:
            continue
        fx = axes.rows // candidate
        if fx <= MAX_SUBGROUPS:
            return Grid(fx, 1)
    raise LayoutError(
        f"{axes.rows} rows don't split into at most {MAX_SUBGROUPS} subgroups of at "
        f"least {least} rows each"
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
            raise HintError(
                "sg_tile",
                f"a subgroup tile of {list(c)}: {parallel} parallel DPAS, the "
                f"hardware has {MAX_DPAS_PER_THREAD}",
            )
        bytes_ = 2 * (a[0] * a[1] + b[0] * b[1]) + 4 * c[0] * c[1]
        if bytes_ > REGISTER_BYTES:
            raise HintError(
                "sg_tile",
                f"a subgroup tile of {list(c)}: its tiles take {bytes_} bytes of "
                f"{REGISTER_BYTES} in registers",
            )


def pick_grid(axes: Axes, hints: Hints) -> Grid:
    """The kernel's grid: a legal hint, else what its reductions need, else the rule."""
    kernel = axes.kernel
    if not axes.free_rows and axes.rows > 1:
        # A reduction over the rows as well: all rows on each subgroup, or SLM and a
        # barrier upstream's path sizes past what the hardware has.
        raise LayoutError(
            f"the kernel reduces or contracts its {axes.rows} rows, so no subgroup grid "
            "fits them"
        )
    if hints.sg_tile is not None:
        grid = _grid_of_tile(axes, hints.sg_tile)
        if hints.sg_tile[1] != 0:
            _check_registers(kernel, grid)
        return grid
    if kernel.reductions:
        rows = ROW_KERNEL_SG_ROWS if kernel.dpas else REDUCTION_SG_ROWS
        return _rows_only_grid(axes, rows)
    return preferred_grid(axes, legal_grids(axes))


def _dpas_rows(sg_rows: int) -> int:
    rows = next((r for r in DPAS_ROWS if r >= MIN_DPAS_ROWS and sg_rows % r == 0), None)
    if rows is None:
        raise LayoutError(f"no DPAS fits a subgroup tile of {sg_rows} rows")
    return rows


def _largest_rows(size: int, limit: int) -> int:
    return next(r for r in (32, 16, 8, 4, 2, 1) if r <= limit and size % r == 0)


RULE_LOAD = (16, LOAD_MAX_COLS)
"""What the rule loads of a DPAS operand: 16 rows of 64 bytes, half a 2D block read's
limit. Equal to the hand-picked tiles at the B50's 70 W cap over a sustained window; short
windows, before the cap settles, don't rank these."""


def _rule_load(data: Sequence[int]) -> tuple[int, int]:
    """The rule's tile of a DPAS operand whose subgroup tile is ``data``."""
    cols = RULE_LOAD[1] if data[1] % RULE_LOAD[1] == 0 else DPAS_COLS
    return _largest_rows(data[0], RULE_LOAD[0]), cols


def _shared_kernel_a_load(data: Sequence[int]) -> tuple[int, int]:
    """Next to an operand all subgroups read whole (:func:`_has_shared_operand`)."""
    return _largest_rows(data[0], DPAS_COLS), DPAS_COLS


def _shared_kernel_b_load(data: Sequence[int]) -> tuple[int, int]:
    if data[0] % LOAD_MAX_ROWS == 0 and data[1] % LOAD_MAX_COLS == 0:
        return LOAD_MAX_ROWS, LOAD_MAX_COLS
    return DPAS_COLS, DPAS_COLS


def _check_operand_tile(tile: Sequence[int], index: int, data: Sequence[int]) -> None:
    """What a DPAS operand's load must be, beyond dividing its tile. Each of these
    lowered and gave wrong rows or a failure on the B50 otherwise: A rows below the
    DPAS's (every row wrong), B rows below VNNI's 16 (undefined function in ``ocloc``),
    columns other than 16 or 32 (abort)."""
    if tile[1] not in (DPAS_COLS, LOAD_MAX_COLS):
        raise LayoutError(f"load tile {list(tile)}: 16 or 32 columns")
    if tile[0] not in BLOCK_ROWS:
        raise LayoutError(f"load tile {list(tile)}: rows are one of {list(BLOCK_ROWS)}")
    if index == 0 and tile[0] % _dpas_rows(data[0]):
        raise LayoutError(
            f"load tile {list(tile)}: fewer rows than the {_dpas_rows(data[0])} of its DPAS"
        )
    if index == 1 and tile[0] < VNNI_MIN_ROWS:
        raise LayoutError(f"load tile {list(tile)}: a B tile takes at least 16 rows")


def _check_block(tile: Sequence[int], data: Sequence[int], what: str) -> None:
    if any(d % t for d, t in zip(data, tile, strict=True)):
        raise LayoutError(f"{what} {list(tile)} does not divide its tile {list(data)}")
    if tile[0] > LOAD_MAX_ROWS or tile[1] > LOAD_MAX_COLS:
        raise LayoutError(
            f"{what} {list(tile)}: 2D block loads are at most "
            f"{LOAD_MAX_ROWS}x{LOAD_MAX_COLS}"
        )


def _prefetch_tile(
    shape: Sequence[int], threads: int, shared: bool
) -> tuple[int, int] | None:
    """Lighthouse's order among legal prefetch tiles of a ``shape`` tile: most
    threads, up to the kernel's, then the most balanced grid. Whole blocks of its
    ``inst_data``: the tile itself if shared, else :data:`PREFETCH_INST`. ``None`` if
    none fits: no prefetch, which is slower and not wrong."""
    found = []
    for rows in range(PREFETCH_ROWS[0], PREFETCH_ROWS[1] + 1):
        for cols in range(PREFETCH_COLS[0], PREFETCH_COLS[1] + 1):
            if (
                shape[0] % rows
                or shape[1] % cols
                or not _prefetch_blocks((rows, cols), shared)
            ):
                continue
            grid = (shape[0] // rows, shape[1] // cols)
            if grid[0] * grid[1] > threads:
                continue
            found.append(((rows, cols), grid[0] * grid[1], -abs(grid[0] - grid[1])))
    if not found:
        return None
    found.sort(key=itemgetter(1, 2), reverse=True)
    return found[0][0]


def _prefetch_blocks(tile: Sequence[int], shared: bool) -> bool:
    """Whether ``tile`` is whole 2D block prefetches."""
    if shared:
        return tile[0] in BLOCK_ROWS and tile[1] in (DPAS_COLS, LOAD_MAX_COLS)
    return not (tile[0] % PREFETCH_INST[0] or tile[1] % PREFETCH_INST[1])


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
    reductions_to_separate: list[ir.Operation] = field(default_factory=list)
    """Reductions :func:`annotate_layouts` gives :func:`_separate_reduction_source`."""


def plan_layouts(kernel: KernelIR, hints: Hints) -> Plan:
    """The anchors of a kernel with a 2D tile: see the module docstring."""
    axes = axes_of(kernel)
    grid = pick_grid(axes, hints)
    plan = Plan(grid, grid.subgroups * SUBGROUP_SIZE)
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
            _plan_operand_load(kernel, grid, hints, access, index, plan)
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
            raise HintError(
                "prefetch_tiles"
                if hints.prefetch_tiles is not None
                else "prefetch_depths",
                "prefetch tiles and depths apply to loads of DPAS operands in a "
                "K loop; this kernel has none",
            )
    if hints.load_tiles is not None:
        others = [access for access in kernel.loads if access.op not in operands]
        if len(hints.load_tiles) == 4 and not operands:
            raise HintError(
                "load_tiles",
                "load tiles [a_m, a_k, b_k, b_n] apply to DPAS operands; this kernel "
                "has none",
            )
        if len(hints.load_tiles) == 2 and not others:
            raise HintError(
                "load_tiles",
                "a load tile [m, n] applies to tiles that are no DPAS operands; "
                "this kernel has none",
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
    plan: Plan,
) -> None:
    value = access.value
    layout = value_layout(kernel, grid, value)
    role = load_role(value)
    if value_bits(value) != 16:
        raise LayoutError(
            f"a DPAS operand read as {value_bits(value)}-bit elements and narrowed: its "
            "tile has no block read; read a bf16 or f16 tile, or copy it to one first"
        )
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
    if role.transposed:
        # A hint for this role goes to the role's plain loads if it has some (K and
        # V of attention): a transposed read has its own tile.
        plain = any(
            not load_role(other.value).transposed
            for other in kernel.loads
            if other.op != access.op and _operand_index(other, kernel) == index
        )
        tile = forced if forced is not None and not plain else TRANSPOSED_LOAD
        if tuple(tile) != TRANSPOSED_LOAD:
            raise HintError(
                "load_tiles",
                f"a transposed load takes {list(TRANSPOSED_LOAD)} tiles, not "
                f"{list(tile)}: other tiles lower to wrong rows",
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
        elif not _has_shared_operand(kernel, grid):
            tile = _rule_load(layout.sg_data)
            if output_bound(kernel):
                # Operands are little of the traffic, and DPAS-sized reads win (L1 7,
                # B50).
                limit = DPAS_ROWS[0] if index == 0 else DPAS_COLS
                tile = (min(tile[0], limit), DPAS_COLS)
            elif any(_operand_order(dpas.operands[0]) for dpas in role.b):
                # B next to a transposed A: on 16, B50, device time, B 16x16 2608
                # us, 32x32 2561, 16x32 2833.
                tile = (min(tile[0], DPAS_COLS), DPAS_COLS)
        elif index == 0:
            tile = _shared_kernel_a_load(layout.sg_data)
        else:
            tile = _shared_kernel_b_load(layout.sg_data)
    try:
        _check_block(tile, layout.sg_data, "load tile")
        if not role.transposed and not role.other:
            _check_operand_tile(tile, index, layout.sg_data)
    except LayoutError as exc:
        if hints.load_tiles is None:
            raise
        raise HintError("load_tiles", str(exc)) from exc
    plan.anchors.append((access.op, "layout", _layout_text(layout, tile, order)))
    _plan_prefetch(grid, hints, access, layout, index, plan)


def _has_shared_operand(kernel: KernelIR, grid: Grid) -> bool:
    """Whether a DPAS operand is read whole by every subgroup (``[1, 1]``). Those
    kernels keep lighthouse's attention loads (:func:`_shared_kernel_a_load`,
    :func:`_shared_kernel_b_load`): on 97, B50, device time, Q, V tiles 16x16, 32x32 1820
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
    grid: Grid,
    hints: Hints,
    access: _Access,
    layout: ValueLayout,
    index: int,
    plan: Plan,
) -> None:
    """A load of a DPAS operand in a loop over K steps, with its tile shared by all
    subgroups (``[1, 1]``) or distributed."""
    parent = access.op.parent
    if parent is None or parent.name != "scf.for":
        return
    depth = 1
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
    elif shared and all(
        s % t == 0 for s, t in zip(shape, SHARED_PREFETCH, strict=True)
    ):
        tile = SHARED_PREFETCH
    else:
        tile = _prefetch_tile(shape, grid.subgroups, shared)
        if tile is None:
            return
    if any(s % t for s, t in zip(shape, tile, strict=True)):
        raise HintError(
            "prefetch_tiles",
            f"prefetch tile {list(tile)} does not divide {list(shape)}",
        )
    if not (
        PREFETCH_ROWS[0] <= tile[0] <= PREFETCH_ROWS[1]
        and PREFETCH_COLS[0] <= tile[1] <= PREFETCH_COLS[1]
    ):
        raise HintError(
            "prefetch_tiles",
            f"prefetch tile {list(tile)}: 2D block prefetches are "
            f"{PREFETCH_ROWS[0]}..{PREFETCH_ROWS[1]} rows of "
            f"{PREFETCH_COLS[0]}..{PREFETCH_COLS[1]} columns",
        )
    if not _prefetch_blocks(tile, shared):
        raise HintError(
            "prefetch_tiles",
            f"prefetch tile {list(tile)}: not whole 2D block prefetches",
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
    transposed = load_role(value).transposed
    try:
        tile = block_load_tile(layout.sg_data, (value_bits(value),), forced)
        if transposed and tuple(tile) != TRANSPOSED_LOAD:
            # 1, 2, 4 or 8 rows abort in upstream's layout code (XeGPULayoutImpl.cpp).
            raise ValueError(
                f"a transposed load takes {list(TRANSPOSED_LOAD)} tiles, not {list(tile)}"
            )
    except ValueError as exc:
        if forced is None:
            raise LayoutError(str(exc)) from exc
        raise HintError("load_tiles", str(exc)) from exc
    order = (0, 1) if transposed else None
    plan.anchors.append((access.op, "layout", _layout_text(layout, tile, order)))


# --- One row per workgroup ---------------------------------------------------------

ROW_SPLIT_SUBGROUPS = (32, 16, 1)
"""What a reduced dim can be split over, in order of preference. A subgroup reads the
other subgroups' partial results as one vector and reduces it across its lanes: upstream
lowers that for whole lanes only, 32 or 16 elements (:mod:`.xevm_lane_reductions` refuses
reductions over part of a subgroup), and 1 subgroup has nothing to read."""
ROW_MIN_SG_COLS = 32
"""A subgroup's columns, where that leaves a split at all. 32 subgroups of one 64-byte
gather per K tile stall, on the B50 and the B70."""
ROW_REDUCTION_KINDS = frozenset({"add", "maximumf", "minimumf"})
"""The kinds whose partial results through SLM are checked, on the B50 and the B70."""
ROW_REDUCTION_BITS = 32
"""A reduction's elements here: upstream's SLM reads and writes are checked on f32."""
ROW_LANE_TILE = (SUBGROUP_SIZE,)
"""The ``inst_data`` of every access: one element per lane.

A tile that reaches a reduction has no choice: the cross-lane reduction forces
``lane_data = [1]``, and a gather of more elements per lane under that consumer takes
wrong elements silently (:func:`.xegpu_elementwise.chunk_1d`). The other tiles could take
more, and don't: a division by the row result, a uniform value, then is a vector ``fdiv``,
which IGC emits as a bare ``math.inv``, and a subnormal divisor gives inf
(``1e-40 * randn`` through an L1 norm). The wider gathers would gain little."""
_ROW_MEMORY_FORMS = frozenset(
    {
        "memref.subview",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.extract_strided_metadata",
        "memref.extract_aligned_pointer_as_index",
    }
)
_ROW_VECTOR_OPS = frozenset(
    {
        "xegpu.load",
        "xegpu.store",
        "vector.multi_reduction",
        "vector.broadcast",
        "vector.step",
        "scf.for",
        "scf.yield",
    }
)


def _row_refusal(kernel: KernelIR) -> str | None:
    """Why ``kernel``'s reductions of one-row tiles can't be split across subgroups:
    the ops :func:`plan_row_reduction` has reasoned about, nothing else."""
    tile = _shape(kernel.primary)
    for op in _walk(kernel.func):
        name = op.name
        if name in ("xegpu.load", "xegpu.store"):
            if not is_contiguous_1d(op):
                return f"a '{name}' of a one-row tile that isn't whole columns"
        elif name == "vector.multi_reduction":
            source = op.operands[0]
            kind = str(op.attributes["kind"]).removeprefix("#vector.kind<").rstrip(">")
            if kind not in ROW_REDUCTION_KINDS:
                return f"a '{kind}' reduction across subgroups"
            if ir.VectorType(source.type).element_type.width != ROW_REDUCTION_BITS:
                return f"a reduction across subgroups of {source.type}"
            if _shape(op.results[0]) is not None:
                return "a reduction of a one-row tile that keeps a dim"
            # One SLM array per reduction, and no barrier after its reads: the next
            # iteration's stores would race with a slower subgroup's reads.
            if op.parent.operation.name == "scf.for":
                return "a reduction across subgroups inside a loop"
        elif op.regions and name != "scf.for":
            return f"'{name}' in a kernel with a reduction across subgroups"
        elif name == "memref.store" or (
            name.startswith("memref.")
            and name not in _ROW_MEMORY_FORMS
            and any(isinstance(v.type, ir.VectorType) for v in op.results)
        ):
            return f"'{name}' of a row result: only tiles are stored"
        elif name == "arith.constant":
            attr = op.attributes["value"]
            if isinstance(attr, ir.DenseElementsAttr) and not (
                ir.DenseElementsAttr(attr).is_splat
            ):
                return "a constant that isn't a splat on a one-row tile"
        elif not name.startswith(("arith.", "math.")) and name not in _ROW_VECTOR_OPS:
            if any(_shape(v) for v in [*op.operands, *op.results]):
                return f"'{name}' on a one-row tile"
        for value in [*op.operands, *op.results]:
            shape = _shape(value)
            if shape == ():
                return "a row result carried or stored as a 0-D vector"
            if shape not in (None, tile):
                return f"a vector of shape {list(shape)} next to tiles of {list(tile)}"
    return None


def _row_subgroups(columns: int, hints: Hints) -> int:
    """How many subgroups split a one-row tile of ``columns``: the first of
    :data:`ROW_SPLIT_SUBGROUPS` leaving each a multiple of 16 elements and no more
    than :data:`MAX_SG_ELEMENTS`, and at least :data:`ROW_MIN_SG_COLS` if some split
    does; or a hint's."""
    legal = [
        s
        for s in ROW_SPLIT_SUBGROUPS
        if columns % (s * BLOCK_COLS) == 0 and columns // s <= MAX_SG_ELEMENTS
    ]
    if hints.sg_tile is not None:
        sg_columns = hints.sg_tile[-1]
        if (
            sg_columns == 0
            or any(rows != 1 for rows in hints.sg_tile[:-1])
            or columns % sg_columns
            or columns // sg_columns not in legal
        ):
            raise HintError(
                "sg_tile",
                f"subgroup tile {list(hints.sg_tile)} of a one-row tile of {columns} "
                f"columns: it takes {[columns // s for s in legal]}",
            )
        return columns // sg_columns
    if not legal:
        raise LayoutError(
            f"no subgroup split of a one-row tile of {columns} columns: "
            f"{list(ROW_SPLIT_SUBGROUPS)} subgroups of a multiple of {BLOCK_COLS} and at "
            f"most {MAX_SG_ELEMENTS} columns"
        )
    return next(
        (s for s in legal if s > 1 and columns // s >= ROW_MIN_SG_COLS), legal[0]
    )


def plan_row_reduction(kernel: KernelIR, hints: Hints) -> Plan | None:
    """A kernel that reduces one-row tiles: a workgroup per row, its columns split
    across the subgroups (:data:`ROW_SPLIT_SUBGROUPS`). ``None`` if ``kernel`` isn't one:
    its reductions have 1D sources (the unit row dim folded) and all its tiles are 1D
    gathers and scatters. :class:`LayoutError` for one this module can't reason about.

    Only such a kernel gets here. A 2D tile keeps its pinned reduced dims. Each
    subgroup reduces its columns, writes the result to SLM at its id and waits at a
    ``gpu.barrier``; then all read the same partial results and reduce them the same
    way, so all hold the same bits. The SLM array comes from the ``memref.alloca`` of
    upstream's distribution, which the lowering makes a global of the workgroup address
    space.

    Why split: time on the B50 and B70 follows the threads per row, not the tile's
    size (``helion_mlir_xpu_utils.normalization``)."""
    primary = kernel.primary
    if (
        not kernel.reductions
        or kernel.dpas
        or kernel.loads
        or kernel.stores
        or len(_shape(primary) or ()) != 1
    ):
        return None
    refusal = _row_refusal(kernel)
    if refusal is not None:
        raise LayoutError(f"a reduction across subgroups: {refusal}")
    if hints.prefetch_tiles is not None or hints.prefetch_depths is not None:
        raise HintError(
            "prefetch_tiles" if hints.prefetch_tiles is not None else "prefetch_depths",
            "prefetch tiles and depths apply to loads of DPAS operands in a K loop; "
            "this kernel has none",
        )
    columns = _shape(primary)[0]
    subgroups = _row_subgroups(columns, hints)
    sg = ValueLayout((subgroups,), (columns // subgroups,))
    _check_row_load_tile(hints)
    plan = Plan(Grid(subgroups, 1), subgroups * SUBGROUP_SIZE)
    for op in _walk(kernel.func):
        if op.name in ("xegpu.load", "xegpu.store"):
            plan.anchors.append((op, "layout", _layout_text(sg, ROW_LANE_TILE)))
    plan.reductions_to_separate = list(kernel.reductions)
    return plan


def _check_row_load_tile(hints: Hints) -> None:
    if hints.load_tiles is not None and hints.load_tiles[-1] != ROW_LANE_TILE[0]:
        raise HintError(
            "load_tiles",
            f"load tile {list(hints.load_tiles)} of a one-row tile: only "
            f"{list(ROW_LANE_TILE)}",
        )


def _separate_reduction_source(reduction: ir.Operation) -> None:
    """A bitcast round trip between ``reduction`` and its source, for
    ``xegpu-sg-to-lane-distribute`` to lower it to a scalar: a workaround, removed by its
    own canonicalization (no code is left).

    The pass leaves such a reduction illegal when its source has a layout of its own,
    as an ``scf.for`` result has: ``agent-wiki/upstream-repro/xpu-reduction-of-loop-result``
    in helion-mlir-dev. Remove it when ``repro.mlir`` there lowers."""
    source = reduction.operands[0]
    source_type = ir.VectorType(source.type)
    integers = ir.VectorType.get(
        source_type.shape,
        ir.IntegerType.get_signless(source_type.element_type.width),
    )
    with ir.InsertionPoint(reduction):
        to_int = ir.Operation.create(
            "arith.bitcast", results=[integers], operands=[source]
        )
        back = ir.Operation.create(
            "arith.bitcast", results=[source_type], operands=[to_int.results[0]]
        )
    reduction.operands[0] = back.results[0]


def plan_elementwise(func: ir.Operation, hints: Hints) -> Plan | None:
    """``xegpu_elementwise``'s layout, if ``func`` is an elementwise kernel."""
    tiles = find_tiles(func)
    if isinstance(tiles, str):
        return None
    if hints.prefetch_tiles is not None or hints.prefetch_depths is not None:
        raise HintError(
            "prefetch_tiles" if hints.prefetch_tiles is not None else "prefetch_depths",
            "prefetch tiles and depths apply to loads of DPAS operands in a K loop; "
            "this kernel has none",
        )
    load_tile = hints.load_tiles
    if load_tile is not None and len(load_tile) not in (1, 2):
        raise HintError(
            "load_tiles",
            "load tiles [a_m, a_k, b_k, b_n] apply to DPAS operands; this kernel has "
            "none",
        )

    def layout_with(sg_tile: object, tile: object) -> object:
        return elementwise_layout(
            tiles.wg_tile,
            tiles.load_bits,
            tiles.store_bits,
            sg_tile=sg_tile,
            load_tile=tile,
            contiguous=tiles.contiguous,
        )

    try:
        layout = layout_with(hints.sg_tile, load_tile)
    except ValueError as exc:
        if load_tile is not None:
            raise HintError("load_tiles", str(exc)) from exc
        if hints.sg_tile is not None:
            raise HintError("sg_tile", str(exc)) from exc
        raise LayoutError(str(exc)) from exc
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


def plan_kernel(func: ir.Operation, hints: Hints) -> Plan:
    """The kernel's plan. Hints are hints: one that can't apply, doesn't fit the IR or
    is illegal is dropped with a log line, and the rule decides instead. A kernel with
    no legal layout without hints is a :class:`LayoutError`."""
    while True:
        try:
            plan = None if _reads_transposed(func) else plan_elementwise(func, hints)
            _refuse_transposes_the_lowering_aborts_on(func)
            if plan is None:
                kernel = analyze(func)
                plan = plan_row_reduction(kernel, hints) or plan_layouts(kernel, hints)
        except HintError as exc:
            if getattr(hints, exc.key) is None:
                raise LayoutError(str(exc)) from exc
            _log_dropped(func, exc.key, getattr(hints, exc.key), str(exc))
            hints = replace(hints, **{exc.key: None})
            continue
        except LayoutError as exc:
            # Not traced to one hint, e.g. a forced grid that a value doesn't divide:
            # without any, the error is the kernel's own.
            forced = [key for key in _HINT_KEYS if getattr(hints, key) is not None]
            if not forced:
                raise
            for key in forced:
                _log_dropped(func, key, getattr(hints, key), str(exc))
            hints = replace(hints, **dict.fromkeys(forced))
            continue
        return plan


def _func_name(func: ir.Operation) -> str:
    return ir.StringAttr(func.attributes["sym_name"]).value


def _log_dropped(func: ir.Operation, key: str, value: object, reason: str) -> None:
    _log_line(
        func,
        f"XeGPU layouts of {_func_name(func)}: dropped the hint {key}={list(value)}: "
        f"{reason}",
    )


def _log_line(func: ir.Operation, line: str) -> None:
    """A warning, here and, for a caller in another process, on the module
    (:data:`LOWERING_LOG_ATTR`). Once: planning may retry without a hint."""
    root = func
    while root.parent is not None:
        root = root.parent.operation
    attributes = root.attributes
    lines = (
        list(ir.ArrayAttr(attributes[LOWERING_LOG_ATTR]))
        if LOWERING_LOG_ATTR in attributes
        else []
    )
    new = ir.StringAttr.get(line)
    if new in lines:
        return
    log.warning("%s", line)
    attributes[LOWERING_LOG_ATTR] = ir.ArrayAttr.get([*lines, new])


def _reads_transposed(func: ir.Operation) -> bool:
    """A transposed read: an elementwise kernel's single layout can't take one, so its
    tiles get laid out one by one."""
    return any(
        op.name == "vector.transpose"
        and (source := _defining(op.operands[0])) is not None
        and source.name == "xegpu.load_nd"
        for op in _walk(func)
    )


def _refuse_transposes_the_lowering_aborts_on(func: ir.Operation) -> None:
    """A transposed tile outside the two forms that work, a loaded tile and a broadcast
    row: anything else asserts in upstream's lane distribution of its SLM path
    (``computeDistributedCoordsForMatrixOp``), on any layout path."""
    for op in _walk(func):
        if op.name != "vector.transpose":
            continue
        source = _defining(op.operands[0])
        name = source.name if source is not None else ""
        if name in ("vector.broadcast", "xegpu.load_nd"):
            continue
        raise LayoutError(
            f"a 'vector.transpose' of a value from '{name or 'a block argument'}': "
            "the lowering can't distribute it across subgroups"
        )


def prepare_tiles(func: ir.Operation) -> None:
    """The forms :mod:`.xegpu_tile_forms` rewrites to ones this module lays out."""
    expand_elementwise_contractions(func)
    hoist_transposes(func)
    split_reads(func, refuse=LayoutError)


def annotate_layouts(module: ir.Operation, func: ir.Operation) -> Plan:
    """Write the layouts of the kernel ``func`` of ``module``: anchors, thread
    count, and :data:`PREFETCH_ATTR` numbering the loads to prefetch (the plan's
    ``prefetches``, for the caller's schedule; :func:`clear_tags` removes it).
    :class:`LayoutError` if none is legal."""
    hints = Hints.of(func)
    prepare_tiles(func)
    plan = plan_kernel(func, hints)
    if HINTS_ATTR in func.attributes:
        del func.attributes[HINTS_ATTR]
    i64 = ir.IntegerType.get_signless(64)
    for op, name, text in plan.anchors:
        op.attributes[name] = ir.Attribute.parse(text)
    for reduction in plan.reductions_to_separate:
        _separate_reduction_source(reduction)
    for number, (op, _) in enumerate(plan.prefetches):
        op.attributes[PREFETCH_ATTR] = ir.IntegerAttr.get(i64, number)
    set_threads(module, func, plan.threads)
    return plan


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
