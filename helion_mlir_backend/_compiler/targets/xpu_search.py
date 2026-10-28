"""The XPU target's autotuning search space: block sizes the XeGPU pipeline
compiles, and the ``xegpu_*`` config keys as per-kernel fragments.

Each GPU kernel (a root grid loop of the Helion kernel, in program order) is
classified from Helion's facts on the kernel, before any code generation:
a GEMM (a ``MatmulFact`` on its grid), a 2D row reduction (a
``ReductionDescriptor`` in its graphs, one grid dim of rows), else elementwise
(1D or 2D tiles). Its memory accesses give its element sizes, its *column*
blocks (the blocks of the contiguous, stride-1 dim of a tile) and its *leading*
blocks (the blocks of all but the last two dims of a rank-3+ tile, e.g. a batch
dim). The pipeline folds unit tile dims away, so leading blocks are 1 and the
kernel is classified by the rank of its tiles without them.

Block sizes (:func:`constrain_block_sizes`) are limited to what Helion's
power-of-two ranges can express: each at most the largest power of two dividing
its (static) size, so every sampled size divides it (no masking); column blocks
at least 16 (a 2D block load or store row); leading blocks 1; per class, the
bounds below. What a
range cannot express -- the workgroup tiles lighthouse's matmul cost model
accepts, the subgroups an elementwise tile needs, forced parameters that do not
fit the sampled block sizes -- is rejected before lowering by the target's
checks (``xpu_checks``, ``xpu_params.check_params``) with an ``MLIRBackendError``
the autotuner skips.

The default config takes Helion's default block sizes in those ranges, a GEMM's
workgroup and K tiles from lighthouse's parameter selection for its sizes, and
infers every XeGPU parameter. Given configs are used as they are.

The ``xegpu_*`` keys (:func:`xegpu_fragments`) become
:class:`KernelChoicesFragment` s: one value per GPU kernel, chosen among a few
fixed candidates of its class, ``None`` (lighthouse's inference) first and
default. A candidate that does not fit the sampled block sizes is rejected like
an invalid block size. ``xegpu_prefetch_tiles`` is not searched: its valid tiles
depend too much on the workgroup tile.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING

import helion
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
from helion.language import _tracing_ops
from lighthouse.schedule.xegpu import matmul_constraints
from lighthouse.schedule.xegpu.xegpu_parameter_selector import XeGPUParameterSelector

from ..mlir_transforms.xegpu_elementwise import BLOCK_COLS
from ..mlir_transforms.xegpu_elementwise import MAX_SG_ELEMENTS
from ..mlir_transforms.xegpu_elementwise import MAX_SUBGROUPS
from ..mlir_transforms.xegpu_elementwise import load_cols
from .xpu_params import LOAD_TILES
from .xpu_params import PREFETCH_DEPTHS
from .xpu_params import REDUCTION_SG_ROWS
from .xpu_params import SG_TILES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_fragment import FragmentFingerprint
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec
    from helion.autotuner.config_spec import MatmulFact
    from helion.autotuner.config_spec import MemoryOpFact

GEMM = "gemm"
REDUCTION = "reduction"
ELEMENTWISE = "elementwise"
OTHER = "other"
"""A kernel outside the classes (several matmuls, rank-3 tiles that are not
unit on their leading dims, a column reduction, ...): its block sizes get the
generic bounds only, its parameters stay inferred."""

MIN_WG_TILE = 32
"""Smallest matmul M/N block size searched: lighthouse's subgroup tiles are 16
(small dims) or 32..128."""
MAX_WG_TILE = 1024
"""Largest matmul M/N block size searched: lighthouse's cost model takes
workgroup tiles of at most 64K elements (32 subgroups of up to 32x64), whose
other side is at least 64."""
DPAS_K = matmul_constraints.DPAS.K
MAX_K_TILE = 64
"""Largest K block size searched: the K tiles of lighthouse's cost model (16, 32,
64); larger ones exceed its register budget."""
MAX_REDUCTION_ROWS = REDUCTION_SG_ROWS * MAX_SUBGROUPS
"""Rows of a reduction workgroup: at most 32 subgroups of lighthouse's 8 rows."""
MAX_REDUCTION_COLS = MAX_SG_ELEMENTS // REDUCTION_SG_ROWS
"""Columns of a reduction's tiles: an 8-row subgroup holds at most
:data:`MAX_SG_ELEMENTS` of each carried tile."""
MAX_ELEMENTWISE_TILE = MAX_SUBGROUPS * MAX_SG_ELEMENTS
"""Elements of an elementwise workgroup tile: 32 subgroups of at most 2048."""

GEMM_SG_TILES = ((32, 32), (32, 64), (64, 32))
"""Subgroup tiles searched for GEMM kernels: the ones lighthouse's cost model
picks most (its options are 32..128 per side)."""
PREFETCH_DEPTH_CHOICES = ((2, 2),)
"""GEMM prefetch depths searched besides lighthouse's 1 (its cost model expands
configs up to depth 2)."""
ELEMENTWISE_SG_TILES = ((16, 32), (32, 32), (32, 64))
"""Subgroup tiles searched for 2D elementwise kernels: 512 to 2048 elements, a
multiple of 16 columns (inferred: closest to 32x32)."""
REDUCTION_SG_TILES = ((16, 0), (32, 0))
"""Subgroup rows searched for reduction kernels (inferred: 8)."""
SEARCH_LOAD_ROWS = (8, 32)
"""Rows of the block load tiles searched for elementwise and reduction kernels
(at the widest block load columns of their element size)."""


@dataclass(frozen=True)
class SearchKernel:
    """What the search space knows of one GPU kernel before code generation."""

    kind: str
    """:data:`GEMM`, :data:`REDUCTION`, :data:`ELEMENTWISE` or :data:`OTHER`."""
    grid_block_ids: tuple[int, ...]
    column_block_ids: frozenset[int] = frozenset()
    """Blocks tiling the contiguous dim of a tile the kernel loads or stores."""
    leading_block_ids: frozenset[int] = frozenset()
    """Blocks tiling a dim before the last two of a tile the kernel loads or
    stores (block size 1: folded away)."""
    load_bits: tuple[int, ...] = ()
    """Element sizes of the tensors the kernel loads."""
    rank: int = 0
    """Rank of the tiles an elementwise kernel stores, without leading dims."""
    matmul: MatmulFact | None = None
    transpose_a: bool = False
    transpose_b: bool = False
    reduced_block_ids: tuple[int, ...] = ()
    """The blocks tiling a reduction's reduced dim."""


def search_kernels(config_spec: ConfigSpec, device_ir: DeviceIR) -> list[SearchKernel]:
    """The GPU kernels of a Helion kernel, in program order."""
    reductions = (
        config_spec.reduction_kernel_fact.reductions
        if config_spec.reduction_kernel_fact is not None
        else ()
    )
    kernels = []
    for root_id, grid in zip(device_ir.root_ids, device_ir.grid_block_ids, strict=True):
        graphs = _graphs_of(device_ir, root_id)
        accesses = [
            fact for fact in config_spec.memory_op_facts if fact.graph_id in graphs
        ]
        columns = frozenset(
            block_id
            for fact in accesses
            if (block_id := _column_block(fact)) is not None
        )
        leading = frozenset(
            block_id for fact in accesses for block_id in _leading_blocks(fact)
        )
        rows = [block_id for block_id in grid if block_id not in leading]
        load_bits = tuple(
            sorted(
                {
                    fact.dtype.itemsize * 8
                    for fact in accesses
                    if fact.kind == "load" and fact.dtype is not None
                }
            )
        )
        common = {
            "grid_block_ids": tuple(grid),
            "column_block_ids": columns,
            "leading_block_ids": leading,
            "load_bits": load_bits,
        }
        matmuls = [
            fact
            for fact in config_spec.matmul_facts
            if {fact.m_block_id, fact.n_block_id} & set(grid)
        ]
        reduced = [
            fact
            for fact in reductions
            if fact.graph_id in graphs and fact.category in SIZED_REDUCTION_CATEGORIES
        ]
        if matmuls:
            if len(matmuls) > 1:
                kernels.append(SearchKernel(OTHER, **common))
                continue
            (matmul,) = matmuls
            transpose_a, transpose_b = _transposes(matmul, accesses)
            kernels.append(
                SearchKernel(
                    GEMM,
                    matmul=matmul,
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                    **common,
                )
            )
        elif reduced:
            # One grid dim of rows (else e.g. a column reduction).
            row_reduction = len(rows) == 1 and rows[0] not in columns
            kind = REDUCTION if row_reduction else OTHER
            reduced_ids = tuple(dict.fromkeys(fact.block_id for fact in reduced))
            kernels.append(SearchKernel(kind, reduced_block_ids=reduced_ids, **common))
        else:
            rank = max(
                (_rank(fact, leading) for fact in accesses if fact.kind == "store"),
                default=0,
            )
            # Tiles of the grid only (an inner tiled loop is e.g. an einsum's K
            # loop, which has no matmul fact).
            inner = {
                block_id
                for fact in accesses
                for block_id in fact.subscript_block_ids
                if block_id is not None and block_id not in grid
            }
            kind = ELEMENTWISE if rank in (1, 2) and not inner else OTHER
            kernels.append(SearchKernel(kind, rank=rank, **common))
    return kernels


def _graphs_of(device_ir: DeviceIR, root_id: int) -> set[int]:
    """The device graphs a root graph runs: itself and its loop and branch bodies."""
    found: set[int] = set()
    pending = [root_id]
    while pending:
        graph_id = pending.pop()
        if graph_id in found or not 0 <= graph_id < len(device_ir.graphs):
            continue
        found.add(graph_id)
        for node in device_ir.graphs[graph_id].graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if _tracing_ops.is_for_loop_target(node.target):
                bodies = node.args[:1]
            elif node.target is _tracing_ops._if:
                bodies = node.args[1:3]
            else:
                continue
            pending.extend(body for body in bodies if isinstance(body, int))
    return found


def _leading_blocks(fact: MemoryOpFact) -> list[int]:
    """The blocks tiling the dims of a rank-3+ access before its last two."""
    return [
        block_id
        for block_id in fact.subscript_block_ids[: max(fact.ndim - 2, 0)]
        if block_id is not None
    ]


def _rank(fact: MemoryOpFact, leading: frozenset[int]) -> int:
    """The rank of an access's tiles without its leading (unit) block dims."""
    return fact.ndim - sum(
        block_id in leading for block_id in fact.subscript_block_ids[: fact.ndim - 2]
    )


def _column_block(fact: MemoryOpFact) -> int | None:
    """The block tiling the contiguous dim of an access (``None``: none, e.g. a
    full slice)."""
    if fact.ndim < 1 or not fact.subscript_block_ids:
        return None
    strides = fact.subscript_strides
    position = strides.index(1) if 1 in strides else len(fact.subscript_block_ids) - 1
    return fact.subscript_block_ids[position]


def _transposes(
    matmul: MatmulFact, accesses: Sequence[MemoryOpFact]
) -> tuple[bool, bool]:
    """Whether A is stored ``[K, M]`` and B ``[N, K]``: their contiguous dim is M
    and K."""
    columns = {
        fact.matmul_operand: _column_block(fact)
        for fact in accesses
        if fact.matmul_operand is not None
    }
    return (
        columns.get("lhs") == matmul.m_block_id is not None,
        columns.get("rhs") == matmul.k_block_id is not None,
    )


# --- Block sizes -----------------------------------------------------------------


def constrain_block_sizes(
    config_spec: ConfigSpec, kernels: Sequence[SearchKernel]
) -> None:
    """Narrow the block sizes to what the kernels compile (module docstring)."""
    block_sizes = config_spec.block_sizes
    valid = set(block_sizes.valid_block_ids())

    def spec(block_id: int | None) -> BlockSizeSpec | None:
        if block_id is None or block_id not in valid:
            return None
        return block_sizes.block_id_lookup(block_id)

    def bound(block_id: int | None, low: int = 1, high: int | None = None) -> None:
        if (block := spec(block_id)) is None:
            return
        block.update_min(min(low, block.max_size))
        if high is not None:
            block.update_max(max(high, block.min_size))

    for block in block_sizes:
        size = block.size_hint
        if size > 0:
            # Powers of two up to this one divide the size: no partial tiles.
            block.update_max(max(size & -size, block.min_size))
    for kernel in kernels:
        for block_id in kernel.column_block_ids:
            bound(block_id, BLOCK_COLS)
        for block_id in kernel.leading_block_ids - kernel.column_block_ids:
            # Unit: the tile is its lower-rank kernel (real 3D tiles fail).
            bound(block_id, high=1)
        if kernel.kind == GEMM and kernel.matmul is not None:
            matmul = kernel.matmul
            bound(matmul.m_block_id, MIN_WG_TILE, MAX_WG_TILE)
            bound(matmul.n_block_id, MIN_WG_TILE, MAX_WG_TILE)
            bound(matmul.k_block_id, DPAS_K, MAX_K_TILE)
            k_block = spec(matmul.k_block_id)
            if k_block is not None and matmul.static_k is not None:
                # Below K: the K loop must stay.
                below_k = 1 << ((matmul.static_k - 1).bit_length() - 1)
                if matmul.static_k > 1 and below_k >= k_block.min_size:
                    k_block.update_max(below_k)
        elif kernel.kind == REDUCTION:
            for block_id in kernel.grid_block_ids:
                if block_id not in kernel.leading_block_ids:
                    bound(block_id, REDUCTION_SG_ROWS, MAX_REDUCTION_ROWS)
            for block_id in kernel.reduced_block_ids:
                bound(block_id, BLOCK_COLS, MAX_REDUCTION_COLS)
        elif kernel.kind == ELEMENTWISE:
            # A 1D tile up to 32 subgroups of 2048; a 2D one at most 2048 per
            # side (the subgroup limit on their product is checked).
            high = MAX_ELEMENTWISE_TILE if kernel.rank == 1 else MAX_SG_ELEMENTS
            for block_id in kernel.grid_block_ids:
                if block_id not in kernel.leading_block_ids:
                    bound(block_id, high=high)


def default_block_sizes(
    config_spec: ConfigSpec, kernels: Sequence[SearchKernel]
) -> list[int] | None:
    """Default block sizes that lower: Helion's defaults (inside the narrowed
    ranges), with each GEMM's workgroup and K tiles from lighthouse's parameter
    selection for its sizes; ``None`` if that changes nothing."""
    block_sizes = config_spec.block_sizes
    defaults = list(config_spec.default_config().config.get("block_sizes", []))
    if len(defaults) != len(block_sizes):
        return None
    changed = False
    for kernel in kernels:
        params = _gemm_params(kernel)
        if params is None:
            continue
        matmul = kernel.matmul
        assert matmul is not None
        for block_id, size in (
            (matmul.m_block_id, params["wg_m"]),
            (matmul.n_block_id, params["wg_n"]),
            (matmul.k_block_id, params["k_tile"]),
        ):
            if block_id is None:
                continue
            index = block_sizes.block_id_to_index(block_id)
            block = block_sizes[index]
            if block.min_size <= size <= block.max_size and defaults[index] != size:
                defaults[index] = size
                changed = True
    return defaults if changed else None


def _gemm_params(kernel: SearchKernel) -> dict[str, int] | None:
    """Lighthouse's parameters for a GEMM kernel's sizes (``None``: not a GEMM
    with static sizes, or none fit)."""
    matmul = kernel.matmul
    if kernel.kind != GEMM or matmul is None:
        return None
    mnk = (matmul.static_m, matmul.static_n, matmul.static_k)
    if None in mnk:
        return None
    try:
        return XeGPUParameterSelector().get_parameters_dict(
            mnk, transpose_a=kernel.transpose_a, transpose_b=kernel.transpose_b
        )
    except ValueError:
        return None


# --- xegpu_* fragments --------------------------------------------------------------


def xegpu_fragments(
    kernels: Sequence[SearchKernel],
) -> dict[str, KernelChoicesFragment]:
    """The ``xegpu_*`` keys searched, with each kernel's candidates (``None``
    first); keys whose candidates are all ``None`` are left out."""
    per_key: dict[str, list[tuple[object, ...]]] = {
        SG_TILES: [],
        LOAD_TILES: [],
        PREFETCH_DEPTHS: [],
    }
    for kernel in kernels:
        candidates = _candidates(kernel)
        for key, choices in per_key.items():
            values = [list(value) for value in candidates.get(key, ())]
            choices.append((None, *values))
    return {
        key: KernelChoicesFragment(tuple(choices))
        for key, choices in per_key.items()
        if any(len(kernel_choices) > 1 for kernel_choices in choices)
    }


def _candidates(kernel: SearchKernel) -> dict[str, list[tuple[int, ...]]]:
    """A kernel's candidate values besides ``None``, by key."""
    if kernel.kind == GEMM:
        return {
            SG_TILES: list(GEMM_SG_TILES),
            PREFETCH_DEPTHS: list(PREFETCH_DEPTH_CHOICES),
        }
    if kernel.kind == REDUCTION:
        cols = max(load_cols(kernel.load_bits))
        return {
            SG_TILES: list(REDUCTION_SG_TILES),
            LOAD_TILES: [(rows, cols) for rows in SEARCH_LOAD_ROWS],
        }
    if kernel.kind == ELEMENTWISE and kernel.rank == 2:
        cols = max(load_cols(kernel.load_bits))
        return {
            SG_TILES: list(ELEMENTWISE_SG_TILES),
            LOAD_TILES: [(rows, cols) for rows in SEARCH_LOAD_ROWS],
        }
    return {}


@dataclass
class KernelChoicesFragment(ConfigSpecFragment):
    """A per-kernel config value: one of ``choices[i]`` for GPU kernel ``i``.

    Like Helion's ``ListOf(EnumFragment)``, but with a choice set per position
    (each kernel's class has its own candidates). Values are lists, as configs
    hold them after a JSON round trip; the first choice of each kernel is the
    default."""

    choices: tuple[tuple[object, ...], ...]

    def default(self) -> list[object]:
        return [choices[0] for choices in self.choices]

    def random(self) -> list[object]:
        return [random.choice(choices) for choices in self.choices]

    def _current(self, value: object) -> list[object]:
        """``value`` as one valid choice per kernel (others: the default)."""
        items = list(value) if isinstance(value, (list, tuple)) else []
        result = []
        for index, choices in enumerate(self.choices):
            item = _plain(items[index]) if index < len(items) else None
            result.append(item if item in choices else choices[0])
        return result

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        """The values differing from ``current`` in one kernel's choice."""
        base = self._current(current)
        neighbors: list[object] = []
        for index, choices in enumerate(self.choices):
            for choice in choices:
                if choice != base[index]:
                    neighbor = list(base)
                    neighbor[index] = choice
                    neighbors.append(neighbor)
        return neighbors

    def differential_mutation(self, a: object, b: object, c: object) -> list[object]:
        """Per kernel, like ``EnumFragment``: keep ``a``'s choice where ``b`` and
        ``c`` agree, else take one of theirs that differs from it."""
        a, b, c = self._current(a), self._current(b), self._current(c)
        result = []
        for index, choices in enumerate(self.choices):
            if b[index] == c[index]:
                result.append(a[index])
                continue
            options = [v for v in (b[index], c[index]) if v != a[index]]
            result.append(random.choice(options) if options else random.choice(choices))
        return result

    def dim(self) -> int:
        return sum(len(choices) for choices in self.choices)

    def encode(self, value: object) -> list[float]:
        """One-hot per kernel."""
        encoded: list[float] = []
        for item, choices in zip(self._current(value), self.choices, strict=True):
            encoded.extend(1.0 if choice == item else 0.0 for choice in choices)
        return encoded

    def fingerprint(self) -> FragmentFingerprint:
        return (
            "kernel_choices",
            *(repr(choices) for choices in self.choices),
        )


def _plain(value: object) -> object:
    """A config entry as a list (tuples as lists), ``None`` as is."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def search_space(config_spec: ConfigSpec, device_ir: DeviceIR) -> None:
    """Narrow ``config_spec``'s block sizes for the XPU, default them to sizes
    that lower, and add the ``xegpu_*`` fragments (as backend tunables: Helion
    searches them and keeps their defaults in every config)."""
    kernels = search_kernels(config_spec, device_ir)
    constrain_block_sizes(config_spec, kernels)
    defaults = default_block_sizes(config_spec, kernels)
    if defaults is not None:
        config_spec.compiler_default_config = helion.Config(block_sizes=defaults)
    config_spec.backend_tunable_fragments = {
        **config_spec.backend_tunable_fragments,
        **xegpu_fragments(kernels),
    }
