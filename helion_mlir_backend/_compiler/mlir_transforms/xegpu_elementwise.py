"""``infer_xegpu_elementwise_layout``: the workgroup-level XeGPU layout of an
elementwise kernel, read off the kernel's memory accesses.

Lighthouse's XeGPU flow knows GEMM, attention and reduction kernels; an
elementwise kernel (any fused unary/binary math, casts, several inputs and
outputs) has no anchor op. What it has are tile accesses: every input is loaded
and every output stored as one workgroup tile (the Helion grid tile). The op
recognizes the kernel by that property, not by the ops computed in between:

- the *workgroup tile* is the shape every tile store writes
  (``vector.transfer_write`` into a buffer before the XeGPU conversion,
  ``xegpu.store_nd``, or ``xegpu.store`` for 1D tiles, after it); all stores
  agree and there is no contraction or reduction;
- the *tile loads and stores* are the accesses of that shape (``xegpu.load_nd``,
  ``prefetch_nd``, ``store_nd``; ``xegpu.load``/``store`` for 1D tiles;
  transfers before the conversion): the ops to anchor;
- the subgroup, load and store tiles come from :func:`elementwise_layout` for
  the workgroup tile and the accessed element types, or are forced.

The results feed lighthouse's ops: ``compute_num_threads`` (``gpu.launch``
threads) and ``xegpu.set_anchor_layout`` (``sg_layout``/``sg_data`` shared by all
tile accesses, ``inst_data`` the load tile on loads and the store tile on
stores; lighthouse's legacy ``elemwise_schedule`` annotates its loads and stores
the same way with one tile for both).

``map_elementwise_grid`` maps an elementwise kernel's grid loop innermost dim
first (grid x = the column tiles), so that the workgroups running at the same
time stream neighbouring memory (see :func:`map_grid_innermost_first`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

SUBGROUP_SIZE = 16
"""Work items per subgroup (SIMD16)."""
MAX_SUBGROUPS = 32
"""Subgroups per workgroup (large register file: 32 threads of 256 registers)."""
BLOCK_COLS = 16
"""Columns of a 2D block store of 16- or 32-bit elements (the minimum of a 2D
block load: one per work item); the elements of a 1D gather or scatter."""
STORE_ROWS = (8, 4, 2, 1)
"""Rows of a 2D block store (at most 8, any element size)."""
LOAD_ROWS = (32, 16, 8, 4, 2, 1)
"""Rows of a 2D block load (at most 32)."""
LOAD_COLS = 16
"""Columns of the 2D block loads of elementwise and reduction tiles, of any
element size: one per work item.

A block load row may hold 64 bytes, but a 16-bit ``8x32`` (``16b_8r32x1c``)
read gives work item ``k`` columns ``2k`` and ``2k + 1``, while upstream's
``xegpu-sg-to-lane-distribute`` (llvm-project 79ab4054) distributes the
``inst_data = [8, 32]`` tile with ``lane_data = [1, 1]`` (columns ``k`` and
``16 + k``): the elementwise ops and 16-column stores then take the elements
in the wrong order (a ReLU gives ``out[0, j] == relu(x[0, 2j])``, a row sum is
wrong). The array-length rewrite (``XeGPUArrayLengthOptimization``) that
splits such reads into 16-column blocks fires only when every user is a
block-aligned ``vector.extract_strided_slice``, as for a GEMM's A tile (DPAS
operands; B uses VNNI transform reads), so GEMMs are right and elementwise and
reduction tiles are not."""
ELEMENT_BITS = frozenset({16, 32})
"""Element sizes of the tiles (f16, bf16, f32, i16, i32): 2D block loads and
stores of both share the store's 16 columns."""
MAX_LANE_ELEMENTS = 16
"""Elements per work item of a load tile (``inst_data``): at most 16, 256 per
tile. The ops on a loaded tile are unrolled to its load tile, one vector per
work item, and LLVM's SPIR-V backend (verified at llvm-project 79ab4054 on an
Arc B50) cannot legalize float extensions and truncations of vectors of more
than 16 elements (``G_FPEXT``/``G_FPTRUNC`` of ``<64 x s16>``: "unable to
legalize instruction"), which every bf16 tile computed in f32 has; vectors of
more than 16 components need ``SPV_EXT_long_vector``."""
LANE_ACCESS_BYTES = 16
"""Bytes one work item reads or writes per gather/scatter instruction of a 1D
tile (``xegpu.load``/``store`` with ``lane_data = [chunk]``, upstream's
``LoadGatherInstruction``/``StoreScatterInstruction.getMaxLaneAccessSizeBytes``):
a subgroup's instruction then covers 256 contiguous bytes, whole 64-byte cache
lines.

Why it matters: a 2D block store of 16-bit elements writes rows of at most 16
elements (XeVM ``blockstore2d``: ``tile_width`` 16 for 8/16/32-bit), 32 bytes, half
a cache line, and bf16 elementwise kernels were bound by it on the Arc B50
(2026-10-31, 4096x393216 relu, column-fastest grid: f32 loads with bf16 8x16
stores 167 GB/s, bf16 16x16 loads with f32 stores 230 GB/s, f32 both ways 197,
the read+write copy ceiling 198). Gathers and scatters of one element per work
item (32 bytes per bf16 instruction) measured the same 165 GB/s; with 16 bytes
per work item 196 GB/s (99 %), with 4 or 8 bytes the same."""
TARGET_SG_ELEMENTS = 32 * 32
"""Default subgroup tile size (lighthouse kernel_bench's elementwise 32x32)."""
MAX_SG_ELEMENTS = 2048
"""Largest subgroup tile: 8 KB of f32 values (math runs in f32), half of a
subgroup's large register file (256 registers of 64 bytes); larger tiles spill."""

_STORES = frozenset({"vector.transfer_write", "xegpu.store_nd", "xegpu.store"})
_LOADS = frozenset(
    {"vector.transfer_read", "xegpu.load_nd", "xegpu.prefetch_nd", "xegpu.load"}
)
_NOT_ELEMENTWISE = frozenset(
    {"vector.contract", "vector.multi_reduction", "vector.reduction", "xegpu.dpas"}
)


@dataclass(frozen=True)
class ElementwiseLayout:
    """A workgroup-level XeGPU layout of the tiles of an elementwise kernel."""

    wg_tile: tuple[int, ...]
    sg_tile: tuple[int, ...]
    load_tile: tuple[int, ...]
    """``inst_data`` of the tile loads (and prefetches)."""
    store_tile: tuple[int, ...]
    """``inst_data`` of the tile stores."""

    @property
    def sg_layout(self) -> tuple[int, ...]:
        return tuple(w // s for w, s in zip(self.wg_tile, self.sg_tile, strict=True))

    @property
    def subgroups(self) -> int:
        return math.prod(self.sg_layout)

    @property
    def threads(self) -> int:
        return self.subgroups * SUBGROUP_SIZE


def _largest_dividing(candidates: Sequence[int], size: int) -> int | None:
    return next((c for c in candidates if size % c == 0), None)


def default_sg_tile(wg_tile: Sequence[int]) -> tuple[int, ...] | None:
    """The subgroup tile of ``wg_tile``: of the tiles that divide it in at most
    :data:`MAX_SUBGROUPS` subgroups, have a multiple of 16 innermost elements
    and at most :data:`MAX_SG_ELEMENTS`, the one closest to :data:`TARGET_SG_ELEMENTS` elements, then the
    squarest, then the widest; ``None`` if none fits."""
    options = [
        [d for d in range(1, w + 1) if w % d == 0 and (i > 0 or d % BLOCK_COLS == 0)]
        for i, w in enumerate(reversed(wg_tile))
    ][::-1]
    candidates = [
        sizes
        for sizes in _product(options)
        if math.prod(w // s for w, s in zip(wg_tile, sizes, strict=True))
        <= MAX_SUBGROUPS
        and math.prod(sizes) <= MAX_SG_ELEMENTS
    ]
    if not candidates:
        return None

    def cost(sizes: tuple[int, ...]) -> tuple[float, float, int]:
        return (
            abs(math.log2(math.prod(sizes) / TARGET_SG_ELEMENTS)),
            abs(math.log2(sizes[-1] / sizes[0])),
            -sizes[-1],
        )

    return min(candidates, key=cost)


def _product(options: list[list[int]]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = [()]
    for values in options:
        result = [(*prefix, value) for prefix in result for value in values]
    return result


def elementwise_layout(
    wg_tile: Sequence[int],
    load_bits: Sequence[int],
    store_bits: Sequence[int],
    sg_tile: Sequence[int] | None = None,
    load_tile: Sequence[int] | None = None,
    contiguous: bool = False,
) -> ElementwiseLayout:
    """The layout of the tiles of an elementwise kernel whose workgroup tile is
    ``wg_tile``, loading tiles of ``load_bits``- and storing tiles of
    ``store_bits``-bit elements; a forced ``sg_tile``/``load_tile`` is checked,
    a missing one inferred. ``contiguous``: every 1D tile access covers
    neighbouring elements (offsets ``base + vector.step``).

    - Tiles are 1D or 2D of 16- or 32-bit elements, with a multiple of 16
      innermost elements.
    - The subgroup tile divides the workgroup tile in 1..32 subgroups and holds
      at most :data:`MAX_SG_ELEMENTS` elements (inferred: :func:`default_sg_tile`).
    - 2D: the store tile is the largest 2D block store dividing the subgroup
      tile, ``[rows <= 8, 16]``; the load tile a 2D block load dividing it,
      ``[rows <= 32, 16]`` with rows a power of two (:data:`LOAD_COLS`), at most
      :data:`MAX_LANE_ELEMENTS` elements per work item; inferred: the most
      rows.
    - 1D: loads and stores are gathers and scatters of 16 elements, one per
      work item; of :func:`chunk_1d` elements per work item when ``contiguous``
      (the same for loads and stores). A forced ``load_tile`` ``[16 * c]``
      sets the chunk ``c`` (a power of two up to the inferred one) of both.

    Raises ``ValueError`` naming the violated constraint.
    """
    wg_tile = tuple(wg_tile)
    rank = len(wg_tile)
    if rank not in (1, 2):
        raise ValueError(f"{rank}D tiles: elementwise kernels take 1D or 2D tiles")
    unsupported = sorted({*load_bits, *store_bits} - ELEMENT_BITS)
    if unsupported:
        raise ValueError(
            f"{unsupported[0]}-bit elements: XeGPU loads and stores of elementwise "
            "tiles take 16- or 32-bit elements"
        )
    if wg_tile[-1] % BLOCK_COLS:
        raise ValueError(
            f"the workgroup tile {list(wg_tile)} has {wg_tile[-1]} innermost "
            f"elements, not a multiple of {BLOCK_COLS} (a block load or store row)"
        )
    if sg_tile is None:
        sg_tile = default_sg_tile(wg_tile)
        if sg_tile is None:
            raise ValueError(
                f"no subgroup tile of at most {MAX_SG_ELEMENTS} elements divides "
                f"the workgroup tile {list(wg_tile)} in at most {MAX_SUBGROUPS} "
                "subgroups: smaller block sizes"
            )
    sg_tile = tuple(sg_tile)
    if len(sg_tile) != rank or 0 in sg_tile:
        raise ValueError(f"subgroup tiles of {rank}D tiles are {rank} positive ints")
    if any(w % s for w, s in zip(wg_tile, sg_tile, strict=True)):
        raise ValueError(
            f"subgroup tile {list(sg_tile)} does not divide the workgroup tile "
            f"{list(wg_tile)}"
        )
    if sg_tile[-1] % BLOCK_COLS:
        raise ValueError(
            f"subgroup tile {list(sg_tile)}: the innermost size is a multiple of "
            f"{BLOCK_COLS}"
        )
    if math.prod(sg_tile) > MAX_SG_ELEMENTS:
        raise ValueError(
            f"subgroup tile {list(sg_tile)}: more than {MAX_SG_ELEMENTS} elements "
            "spill out of the register file"
        )
    subgroups = math.prod(w // s for w, s in zip(wg_tile, sg_tile, strict=True))
    if subgroups > MAX_SUBGROUPS:
        raise ValueError(f"{subgroups} subgroups exceed the maximum of {MAX_SUBGROUPS}")
    if rank == 1:
        chunk = chunk_1d(sg_tile, (*load_bits, *store_bits)) if contiguous else 1
        if load_tile is not None:
            # Forced: [16 x c], c a power of two up to the inferred chunk (also
            # the stores' tile: loads and stores share it).
            chunks = [c for c in (16, 8, 4, 2, 1) if c <= chunk]
            if tuple(load_tile) not in [(SUBGROUP_SIZE * c,) for c in chunks]:
                raise ValueError(
                    f"load tile {list(load_tile)}: this kernel's 1D tiles take "
                    f"{[SUBGROUP_SIZE * c for c in chunks]} elements per gather "
                    "(16 x a power of two, at most 16 bytes per work item, "
                    "dividing the subgroup tile; one element per work item for "
                    "tiles not known contiguous)"
                )
            chunk = load_tile[0] // SUBGROUP_SIZE
        store_tile: tuple[int, ...] = (SUBGROUP_SIZE * chunk,)
        return ElementwiseLayout(wg_tile, sg_tile, store_tile, store_tile)
    store_tile = block_store_tile(sg_tile)
    load_tile = block_load_tile(sg_tile, load_bits, load_tile)
    return ElementwiseLayout(wg_tile, sg_tile, load_tile, store_tile)


def chunk_1d(sg_tile: Sequence[int], bits: Sequence[int]) -> int:
    """Elements per work item of the gathers and scatters of a contiguous 1D
    subgroup tile ``sg_tile`` accessing ``bits``-bit elements: one chunk for all
    of them, :data:`LANE_ACCESS_BYTES` of the widest element (8 bf16, 4 f32),
    halved until 16 chunks divide the subgroup tile.

    Loads and stores share the chunk on purpose. Upstream XeGPU completes the
    lane layout of a gather from its consumer: a gather of ``[16 * n]`` whose
    consumer has ``lane_data = [1]`` gets ``vector<n>`` offsets per work item
    ``n``-strided, which ``XeGPUToXeVM`` (llvm-project 79ab4054) coalesces into one
    access at the first offset anyway: wrong elements, silently (a bf16 relu
    with ``[128]`` loads and ``[16]`` stores: 70 % of the outputs wrong on the
    B50)."""
    chunk = max(1, LANE_ACCESS_BYTES * 8 // max(bits, default=16))
    chunk = min(chunk, MAX_LANE_ELEMENTS)
    while chunk > 1 and sg_tile[0] % (SUBGROUP_SIZE * chunk):
        chunk //= 2
    return chunk


def block_store_tile(sg_tile: Sequence[int]) -> tuple[int, int]:
    """The 2D block store tile (``inst_data``) of a subgroup tile ``sg_tile``
    whose columns are a multiple of 16: the largest ``[rows <= 8, 16]``
    dividing it."""
    rows = _largest_dividing(STORE_ROWS, sg_tile[0])
    assert rows is not None
    return (rows, BLOCK_COLS)


def block_load_tile(
    sg_tile: Sequence[int],
    load_bits: Sequence[int],
    load_tile: Sequence[int] | None = None,
) -> tuple[int, int]:
    """The 2D block load tile (``inst_data``) of a subgroup tile ``sg_tile`` of
    ``load_bits``-bit elements; a forced ``load_tile`` is checked, a missing one
    inferred.

    ``[rows, cols]``, dividing ``sg_tile``: rows a power of two of at most 32,
    :data:`LOAD_COLS` columns, at most :data:`MAX_LANE_ELEMENTS`
    elements per work item; inferred: the most rows. Raises ``ValueError`` naming
    the violated constraint (also when no such tile divides ``sg_tile``).
    """
    max_elements = SUBGROUP_SIZE * MAX_LANE_ELEMENTS
    if load_tile is None:
        load_rows = _largest_dividing(
            [r for r in LOAD_ROWS if r * LOAD_COLS <= max_elements], sg_tile[0]
        )
        if load_rows is None or sg_tile[1] % LOAD_COLS:
            raise ValueError(
                f"no 2D block load of {max(load_bits, default=16)}-bit elements "
                f"divides the subgroup tile {list(sg_tile)}"
            )
        load_tile = (load_rows, LOAD_COLS)
    load_tile = tuple(load_tile)
    if (
        len(load_tile) != 2
        or load_tile[0] not in LOAD_ROWS
        or load_tile[1] != LOAD_COLS
    ):
        raise ValueError(
            f"load tile {list(load_tile)}: 2D block loads of "
            f"{max(load_bits, default=16)}-bit elements take [rows, cols] tiles with "
            f"rows in {sorted(LOAD_ROWS)} and cols in [{LOAD_COLS}]"
        )
    if math.prod(load_tile) > max_elements:
        raise ValueError(
            f"load tile {list(load_tile)}: more than {MAX_LANE_ELEMENTS} elements "
            f"per work item ({max_elements} per tile) do not compile to SPIR-V "
            "(float conversions of wider vectors)"
        )
    if any(s % i for s, i in zip(sg_tile, load_tile, strict=True)):
        raise ValueError(
            f"load tile {list(load_tile)} does not divide the subgroup tile "
            f"{list(sg_tile)}"
        )
    return load_tile


# --- The kernel's tiles in the IR ----------------------------------------------


def _is_tdesc(value_type: ir.Type) -> bool:
    return str(value_type).startswith("!xegpu.tensor_desc")


def _tile_type(op: ir.Operation) -> ir.ShapedType | None:
    """The vector or tensor descriptor type of the tile ``op`` accesses in
    memory (``None``: not a tile access)."""
    name = op.name
    if name == "vector.transfer_write":
        if not isinstance(op.operands[1].type, ir.MemRefType):
            return None
        return ir.ShapedType(op.operands[0].type)
    if name == "vector.transfer_read":
        if not isinstance(op.operands[0].type, ir.MemRefType):
            return None
        return ir.ShapedType(op.results[0].type)
    if name in ("xegpu.load_nd", "xegpu.store_nd", "xegpu.prefetch_nd"):
        tdesc = next(value.type for value in op.operands if _is_tdesc(value.type))
        return ir.ShapedType(tdesc)
    if name == "xegpu.load":
        return ir.ShapedType(op.results[0].type)
    if name == "xegpu.store":
        return ir.ShapedType(op.operands[0].type)
    return None


def _defining_op(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    return owner.operation if isinstance(owner, ir.OpView) else None


def _is_step(value: ir.Value) -> bool:
    op = _defining_op(value)
    return op is not None and op.name == "vector.step"


def _is_splat_constant(op: ir.Operation) -> bool:
    attr = op.attributes["value"]
    return (
        isinstance(attr, ir.DenseElementsAttr) and ir.DenseElementsAttr(attr).is_splat
    )


def _is_splat_index(value: ir.Value) -> bool:
    """A vector of one index: ``vector.broadcast`` of a scalar or a splat
    constant."""
    op = _defining_op(value)
    if op is None:
        return False
    if op.name == "vector.broadcast":
        return not isinstance(op.operands[0].type, ir.VectorType)
    return op.name == "arith.constant" and _is_splat_constant(op)


def _contiguous_offsets(value: ir.Value) -> bool:
    """``value`` is ``vector.step`` or ``base + vector.step`` (``base`` a
    splat): offsets of neighbouring elements."""
    if _is_step(value):
        return True
    op = _defining_op(value)
    if op is None or op.name != "arith.addi":
        return False
    lhs, rhs = op.operands
    return (_is_step(lhs) and _is_splat_index(rhs)) or (
        _is_step(rhs) and _is_splat_index(lhs)
    )


def _all_true(value: ir.Value) -> bool:
    op = _defining_op(value)
    if op is None or op.name != "arith.constant":
        return False
    attr = op.attributes["value"]
    if isinstance(attr, ir.DenseElementsAttr):
        attr = ir.DenseElementsAttr(attr)
        return attr.is_splat and ir.BoolAttr(attr.get_splat_value()).value
    return isinstance(attr, ir.BoolAttr) and attr.value


_OFFSETS_OPERAND = {"xegpu.load": 1, "xegpu.store": 2}
"""The offsets operand of a gather (``source, offsets, mask``) and a scatter
(``value, dest, offsets, mask``) with offsets (no tensor descriptor)."""


def _gather_scatter_operands(op: ir.Operation) -> tuple[ir.Value, ir.Value] | None:
    """The offsets and mask of a 1D gather or scatter with offsets; ``None`` for
    other forms (e.g. on a tensor descriptor)."""
    index = _OFFSETS_OPERAND.get(op.name)
    if index is None or len(op.operands) < index + 2:
        return None
    if _is_tdesc(op.operands[index - 1].type):
        return None
    return op.operands[index], op.operands[index + 1]


def _is_contiguous_1d(op: ir.Operation) -> bool:
    """A 1D tile access ``op`` covers neighbouring elements: a gather or
    scatter at :func:`_contiguous_offsets` under an all-true constant mask."""
    operands = _gather_scatter_operands(op)
    if operands is None:
        return False
    offsets, mask = operands
    return _contiguous_offsets(offsets) and _all_true(mask)


def _is_index_vector(value: ir.Value) -> bool:
    return isinstance(value.type, ir.VectorType) and isinstance(
        ir.VectorType(value.type).element_type, ir.IndexType
    )


def _only_offsets(value: ir.Value, accesses: list[ir.Operation]) -> bool:
    """Every use of ``value`` is the offsets operand of one of ``accesses``,
    directly or through ``arith.addi`` of index vectors."""
    for use in value.uses:
        user = use.owner.operation
        if user.name == "arith.addi" and _is_index_vector(user.results[0]):
            if not _only_offsets(user.results[0], accesses):
                return False
        elif not (
            user in accesses and use.operand_number == _OFFSETS_OPERAND[user.name]
        ):
            return False
    return True


def _chunkable_1d(target: ir.Operation, accesses: list[ir.Operation]) -> bool:
    """The 1D tile ``accesses`` of the kernel ``target`` may take several
    neighbouring elements per work item: each is contiguous
    (:func:`_is_contiguous_1d`), and every other op on vectors is an
    elementwise ``arith``/``math`` op, a splat constant, a scalar broadcast, or
    part of the offsets (``vector.step`` and index additions used only as
    offsets).

    Anything else keeps one element per work item, the form upstream lowers
    for any vector: e.g. a non-splat constant (``tile.index`` folds to an
    arange) is distributed with the chunked lane layout into per-work-item
    vectors of more than 16 components read with dynamic ``extractelement``,
    which LLVM's SPIR-V backend crashes on (``Index < this->size()``)."""
    if not accesses or not all(_is_contiguous_1d(op) for op in accesses):
        return False
    fits = [True]

    def visit(op: ir.Operation) -> ir.WalkResult:
        values = [*op.operands, *op.results]
        if not any(isinstance(v.type, ir.VectorType) for v in values):
            return ir.WalkResult.ADVANCE
        name = op.name
        if name in _OFFSETS_OPERAND:
            ok = op in accesses
        elif name == "arith.constant":
            ok = _is_splat_constant(op)
        elif name == "vector.broadcast":
            ok = not isinstance(op.operands[0].type, ir.VectorType)
        elif name == "vector.step":
            ok = _only_offsets(op.results[0], accesses)
        elif name.startswith(("arith.", "math.")):
            ok = not any(_is_index_vector(v) for v in values) or (
                name == "arith.addi" and _only_offsets(op.results[0], accesses)
            )
        else:
            ok = False
        if not ok:
            fits[0] = False
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    target.walk(visit, ir.WalkOrder.PRE_ORDER)
    return fits[0]


@dataclass
class ElementwiseTiles:
    """The tile accesses of an elementwise kernel."""

    wg_tile: tuple[int, ...]
    loads: list[ir.Operation]
    stores: list[ir.Operation]
    load_bits: list[int]
    store_bits: list[int]
    contiguous: bool = False
    """1D tiles that may take several neighbouring elements per work item
    (:func:`_chunkable_1d`; only gathers and scatters are checked, so ``False``
    before the XeGPU conversion)."""
    converted: bool = False
    """The accesses are XeGPU ops (else vector transfers: before the XeGPU
    conversion, where only the subgroup layout, the thread count, is used)."""


def find_tiles(target: ir.Operation) -> ElementwiseTiles | str:
    """The tile accesses of the elementwise kernel ``target`` (a ``gpu.launch``
    or ``gpu.func``); else why it is not an elementwise kernel."""
    accesses: list[tuple[ir.Operation, ir.ShapedType]] = []
    blocker: list[str] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in _NOT_ELEMENTWISE:
            blocker.append(op.name)
            return ir.WalkResult.INTERRUPT
        if op.name in _STORES or op.name in _LOADS:
            tile_type = _tile_type(op)
            if tile_type is not None:
                accesses.append((op, tile_type))
        return ir.WalkResult.ADVANCE

    target.walk(visit, ir.WalkOrder.PRE_ORDER)
    if blocker:
        return f"'{blocker[0]}' is not elementwise"
    shapes = {tuple(t.shape) for op, t in accesses if op.name in _STORES}
    if len(shapes) != 1:
        return f"tile stores of {len(shapes)} shapes (one expected)"
    (wg_tile,) = shapes
    tiles = [(op, t) for op, t in accesses if tuple(t.shape) == wg_tile]
    loads = [(op, t) for op, t in tiles if op.name in _LOADS]
    stores = [(op, t) for op, t in tiles if op.name in _STORES]
    return ElementwiseTiles(
        wg_tile=wg_tile,
        loads=[op for op, _ in loads],
        stores=[op for op, _ in stores],
        load_bits=sorted({t.element_type.width for _, t in loads}),
        store_bits=sorted({t.element_type.width for _, t in stores}),
        contiguous=len(wg_tile) == 1
        and len(tiles) == len(accesses)
        and _chunkable_1d(target, [op for op, _ in tiles]),
        converted=any(op.name.startswith("xegpu.") for op, _ in tiles),
    )


# --- The transform op ------------------------------------------------------------


def _params(values: Sequence[int]) -> list[ir.IntegerAttr]:
    i64 = ir.IntegerType.get_signless(64)
    return [ir.IntegerAttr.get(i64, value) for value in values]


@transform_op(modifies_payload=False)
class InferXeGPUElementwiseLayoutOp(
    HelionTransformDialect.Operation, name="infer_xegpu_elementwise_layout"
):
    """The XeGPU layout of the elementwise kernel ``target`` (one ``gpu.launch``
    or ``gpu.func``): see the module docstring.

    Results: ``loads`` and ``stores``, the tile accesses (ops whose memory tile
    is the workgroup tile); params holding one i64 per dim: ``wg_tile``,
    ``sg_tile`` (= ``sg_data``), ``sg_layout``, ``load_tile`` and ``store_tile``
    (``inst_data`` of the loads and the stores). The optional ``force_sg_tile``
    and ``force_load_tile`` params (one i64 per dim) force those tiles.

    A silenceable failure if ``target`` is not an elementwise kernel (a
    contraction or reduction, tile stores of several shapes) or no layout fits.
    """

    target: ext.Operand[transform.AnyOpType]
    force_sg_tile: ext.Operand[transform.AnyParamType] | None = None
    force_load_tile: ext.Operand[transform.AnyParamType] | None = None
    loads: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    stores: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    wg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    sg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    sg_layout: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    load_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    store_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @staticmethod
    def run(
        op: "InferXeGPUElementwiseLayoutOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        if len(targets) != 1:
            return DiagnosedSilenceableFailure.SilenceableFailure
        tiles = find_tiles(targets[0].operation)
        if isinstance(tiles, str):
            return DiagnosedSilenceableFailure.SilenceableFailure

        def forced(handle: ir.Value | None) -> list[int] | None:
            if handle is None:
                return None
            values = []
            for attr in state.get_params(handle):
                if isinstance(attr, ir.ArrayAttr):
                    values.extend(ir.IntegerAttr(a).value for a in attr)
                else:
                    values.append(ir.IntegerAttr(attr).value)
            return values

        try:
            layout = elementwise_layout(
                tiles.wg_tile,
                tiles.load_bits,
                tiles.store_bits,
                sg_tile=forced(op.force_sg_tile),
                load_tile=forced(op.force_load_tile),
                # Before the XeGPU conversion only the thread count is used: a
                # forced chunk is checked when the gathers are anchored.
                contiguous=tiles.contiguous or not tiles.converted,
            )
        except ValueError as exc:
            op.location.emit_error(f"infer_xegpu_elementwise_layout: {exc}")
            return DiagnosedSilenceableFailure.SilenceableFailure
        results.set_ops(op.loads, tiles.loads)
        results.set_ops(op.stores, tiles.stores)
        results.set_params(op.wg_tile, _params(layout.wg_tile))
        results.set_params(op.sg_tile, _params(layout.sg_tile))
        results.set_params(op.sg_layout, _params(layout.sg_layout))
        results.set_params(op.load_tile, _params(layout.load_tile))
        results.set_params(op.store_tile, _params(layout.store_tile))
        return DiagnosedSilenceableFailure.Success


def _param(values: Sequence[int] | None) -> ir.Value | None:
    """One param holding ``values`` (``None`` for none)."""
    if values is None:
        return None
    return transform.ParamConstantOp(
        transform.AnyParamType.get(), ir.ArrayAttr.get(_params(values))
    ).param


def infer_xegpu_elementwise_layout(
    target: ir.Value,
    sg_tile: Sequence[int] | None = None,
    load_tile: Sequence[int] | None = None,
) -> InferXeGPUElementwiseLayoutOp:
    """snake_case wrapper to create an InferXeGPUElementwiseLayoutOp."""
    return InferXeGPUElementwiseLayoutOp(
        target=target,
        force_sg_tile=_param(sg_tile),
        force_load_tile=_param(load_tile),
    )


# --- The grid order ----------------------------------------------------------------

MAPPING_ATTR = "mapping"
"""The ``scf.parallel`` attribute ``gpu-map-parallel-loops`` sets and
``convert-parallel-loops-to-gpu`` reads (``gpu::getMappingAttrName``)."""
_BLOCK_IDS = ("block_x", "block_y", "block_z")


def map_grid_innermost_first(
    loop: ir.Operation, *, elementwise_only: bool = True
) -> bool:
    """Map the grid loop ``loop`` (a top-level 2D or 3D ``scf.parallel``) of an
    elementwise kernel (:func:`find_tiles`; with ``elementwise_only=False`` any
    kernel, for the GEMMs ``xpu_params.columns_first`` picks) to the GPU grid
    innermost dim first:
    its last dim to block x, the one before to block y (then z), as
    ``gpu-map-parallel-loops{mapping-policy=innermost-first}`` does, for this
    loop only. Returns whether it did; other loops (already mapped, nested, of
    one or more than three dims, not elementwise) are left alone.

    Why: workgroups are dispatched with block x fastest, and
    ``gpu-map-parallel-loops`` maps the outermost dim (the rows) to x, so the
    workgroups running at the same time are a column of tiles, each row of a
    tile in another DRAM page (Arc B50, 2026-10-31: a bf16 relu at 4096x393216,
    rows 768 KiB apart, with 128x256 tiles 40 GB/s rows-first, 115 GB/s
    columns-first; 1-row tiles of 4096 columns with 16-byte accesses 116
    against 196 GB/s; the read+write copy ceiling is 198). Columns first, they
    stream neighbouring tiles of the same rows. The order changes no result.
    Other kernels keep the rows-first order (GEMM configs were tuned for it)
    unless the caller maps them (``elementwise_only=False``)."""
    loop = loop.operation
    if loop.name != "scf.parallel" or MAPPING_ATTR in loop.attributes:
        return False
    parent = loop.parent
    while parent is not None:
        if parent.operation.name == "scf.parallel":
            return False
        parent = parent.operation.parent
    rank = len(loop.regions[0].blocks[0].arguments)
    if rank not in (2, 3):
        return False
    if elementwise_only and isinstance(find_tiles(loop), str):
        return False
    attrs = [
        ir.Attribute.parse(
            f"#gpu.loop_dim_map<processor = {_BLOCK_IDS[rank - 1 - dim]}, "
            "map = (d0) -> (d0), bound = (d0) -> (d0)>"
        )
        for dim in range(rank)
    ]
    loop.attributes[MAPPING_ATTR] = ir.ArrayAttr.get(attrs)
    return True


@transform_op(modifies_payload=True)
class MapElementwiseGridOp(
    HelionTransformDialect.Operation, name="map_elementwise_grid"
):
    """:func:`map_grid_innermost_first` on each target ``scf.parallel``
    (others are left alone)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MapElementwiseGridOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            map_grid_innermost_first(target.operation)
        return DiagnosedSilenceableFailure.Success


def map_elementwise_grid(target: ir.Value) -> MapElementwiseGridOp:
    """snake_case wrapper to create a MapElementwiseGridOp."""
    return MapElementwiseGridOp(target=target)


@transform_op(modifies_payload=True)
class MapGridColumnsFirstOp(
    HelionTransformDialect.Operation, name="map_grid_columns_first"
):
    """:func:`map_grid_innermost_first` on each target ``scf.parallel`` of any
    kernel class (``elementwise_only=False``); the caller picks the kernels
    (GEMMs whose output store dominates, ``xpu_params.columns_first``)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MapGridColumnsFirstOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            map_grid_innermost_first(target.operation, elementwise_only=False)
        return DiagnosedSilenceableFailure.Success


def map_grid_columns_first(target: ir.Value) -> MapGridColumnsFirstOp:
    """snake_case wrapper to create a MapGridColumnsFirstOp."""
    return MapGridColumnsFirstOp(target=target)
