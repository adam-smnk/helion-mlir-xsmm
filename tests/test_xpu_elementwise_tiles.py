"""Elementwise kernels on the XPU target, IR only: the per-work-item chunk of
contiguous 1D tiles (``find_tiles`` / ``elementwise_layout``) and the grid
order (``map_grid_innermost_first``), on hand-written IR: the exact patterns
they match and the IR they leave alone."""

from __future__ import annotations

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import MAPPING_ATTR
from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import (
    elementwise_layout,
)
from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import find_tiles
from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import (
    map_grid_innermost_first,
)

# --- 1D tiles: contiguous or not ---------------------------------------------------

_GATHER = """
module attributes {{gpu.container_module}} {{
  gpu.module @m {{
    gpu.func @k(%src: i64, %dst: i64, %base: index, %flag: vector<1024xi1>) kernel {{
      %step = vector.step : vector<1024xindex>
      %b = vector.broadcast %base : index to vector<1024xindex>
      %true = arith.constant dense<true> : vector<1024xi1>
      %zero = arith.constant dense<0.0> : vector<1024xbf16>
{defs}
      %v = xegpu.load %src[{load}], {mask} : i64, vector<1024xindex>, vector<1024xi1> -> vector<1024xbf16>
      %r = arith.maximumf %v, %zero : vector<1024xbf16>
{body}
      xegpu.store {value}, %dst[{store}], %true : vector<1024xbf16>, i64, vector<1024xindex>, vector<1024xi1>
      gpu.return
    }}
  }}
}}"""
_DEFS = {
    "%contiguous": "%contiguous = arith.addi %b, %step : vector<1024xindex>",
    "%swapped": "%swapped = arith.addi %step, %b : vector<1024xindex>",
    "%doubled": "%two = arith.constant dense<2> : vector<1024xindex>\n"
    "      %doubled = arith.muli %step, %two : vector<1024xindex>",
}
_DEFS["%strided"] = (
    _DEFS["%doubled"]
    + "\n      %strided = arith.addi %b, %doubled : vector<1024xindex>"
)


def _tiles(
    load: str,
    store: str,
    mask: str = "%true",
    body: str = "",
    value: str = "%r",
) -> object:
    defs = "\n".join(
        f"      {_DEFS[name]}" for name in dict.fromkeys((load, store)) if name in _DEFS
    )
    text = _GATHER.format(
        defs=defs, load=load, store=store, mask=mask, body=body, value=value
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        (func,) = [
            op.operation
            for op in module.body.operations[0].regions[0].blocks[0].operations
            if op.operation.name == "gpu.func"
        ]
        return find_tiles(func)


@pytest.mark.parametrize(
    ("load", "store"),
    [("%contiguous", "%contiguous"), ("%swapped", "%step"), ("%step", "%contiguous")],
    ids=["base+step", "step+base", "step"],
)
def test_contiguous_1d_tiles(load: str, store: str) -> None:
    tiles = _tiles(load, store)
    assert tiles.wg_tile == (1024,)
    assert tiles.contiguous


_ARANGE = (
    "      %c = arith.constant dense<[{}]> : vector<1024xbf16>\n"
    "      %w = arith.addf %r, %c : vector<1024xbf16>"
).format(", ".join(str(float(i % 7)) for i in range(1024)))
_STEP_AS_DATA = (
    "      %i = arith.index_cast %step : vector<1024xindex> to vector<1024xi32>\n"
    "      %f = arith.sitofp %i : vector<1024xi32> to vector<1024xbf16>\n"
    "      %w = arith.addf %r, %f : vector<1024xbf16>"
)
_OFFSETS_AS_DATA = (
    "      %i = arith.index_cast %contiguous : vector<1024xindex> to vector<1024xi32>\n"
    "      %f = arith.sitofp %i : vector<1024xi32> to vector<1024xbf16>\n"
    "      %w = arith.addf %r, %f : vector<1024xbf16>"
)
_OTHER_SHAPE_LOAD = (
    "      %s16 = vector.step : vector<16xindex>\n"
    "      %t16 = arith.constant dense<true> : vector<16xi1>\n"
    "      %e = xegpu.load %src[%s16], %t16 : i64, vector<16xindex>, vector<16xi1> "
    "-> vector<16xbf16>"
)
_SHAPE_CAST = (
    "      %s = vector.shape_cast %r : vector<1024xbf16> to vector<8x128xbf16>\n"
    "      %w = vector.shape_cast %s : vector<8x128xbf16> to vector<1024xbf16>"
)


@pytest.mark.parametrize(
    ("load", "store", "mask", "body", "value"),
    [
        ("%strided", "%contiguous", "%true", "", "%r"),
        ("%contiguous", "%strided", "%true", "", "%r"),
        ("%doubled", "%contiguous", "%true", "", "%r"),
        ("%contiguous", "%contiguous", "%flag", "", "%r"),
        ("%contiguous", "%contiguous", "%true", _ARANGE, "%w"),
        ("%contiguous", "%contiguous", "%true", _STEP_AS_DATA, "%w"),
        ("%contiguous", "%contiguous", "%true", _SHAPE_CAST, "%w"),
        ("%contiguous", "%contiguous", "%true", _OFFSETS_AS_DATA, "%w"),
        ("%contiguous", "%contiguous", "%true", _OTHER_SHAPE_LOAD, "%r"),
    ],
    ids=[
        "strided-load",
        "strided-store",
        "step-times-2",
        "runtime-mask",
        "non-splat-constant",
        "step-as-data",
        "other-vector-op",
        "offsets-cast-to-data",
        "access-of-another-shape",
    ],
)
def test_other_1d_tiles_keep_one_element_per_work_item(
    load: str, store: str, mask: str, body: str, value: str
) -> None:
    """Any access that is not provably contiguous (strided offsets, a mask that
    may differ per element), and any kernel with other vector values than the
    tiles, their offsets and elementwise math (a non-splat constant such as
    ``tile.index``'s arange, ``vector.step`` or the offsets cast and used as
    data, another vector op, a load or store of another shape), keeps one
    element per work item."""
    tiles = _tiles(load, store, mask, body, value)
    assert tiles.wg_tile == (1024,)
    assert not tiles.contiguous


@pytest.mark.parametrize(
    ("wg_tile", "load_bits", "store_bits", "contiguous", "tile"),
    [
        # 16 bytes per work item of the widest element.
        ((8192,), (16,), (16,), True, (128,)),
        ((8192,), (32,), (32,), True, (64,)),
        # Loads and stores share the chunk (bf16 in, f32 out: 4).
        ((8192,), (16,), (32,), True, (64,)),
        ((8192,), (32,), (16,), True, (64,)),
        # Halved until 16 chunks divide the subgroup tile (48: one element).
        ((2048,), (16,), (16,), True, (128,)),
        ((64,), (16,), (16,), True, (64,)),
        ((48,), (16,), (16,), True, (16,)),
        # Not known contiguous: one element per work item.
        ((8192,), (16,), (16,), False, (16,)),
    ],
)
def test_1d_tile_chunks(
    wg_tile: tuple[int, ...],
    load_bits: tuple[int, ...],
    store_bits: tuple[int, ...],
    contiguous: bool,
    tile: tuple[int, ...],
) -> None:
    layout = elementwise_layout(wg_tile, load_bits, store_bits, contiguous=contiguous)
    assert layout.load_tile == layout.store_tile == tile


@pytest.mark.parametrize(
    ("wg_tile", "bits", "load_tile", "tile"),
    [
        ((8192,), (16,), (128,), (128,)),
        ((8192,), (16,), (32,), (32,)),
        ((8192,), (16,), (16,), (16,)),
        ((8192,), (32,), (64,), (64,)),
    ],
)
def test_forced_1d_load_tiles(
    wg_tile: tuple[int, ...],
    bits: tuple[int, ...],
    load_tile: tuple[int, ...],
    tile: tuple[int, ...],
) -> None:
    """A forced 1D load tile sets the chunk of the loads and the stores."""
    layout = elementwise_layout(
        wg_tile, bits, bits, load_tile=load_tile, contiguous=True
    )
    assert layout.load_tile == layout.store_tile == tile


@pytest.mark.parametrize(
    ("bits", "load_tile", "contiguous"),
    [
        ((16,), (256,), True),  # 16 bf16 per work item: more than 16 bytes
        ((32,), (128,), True),  # 8 f32 per work item
        ((16,), (48,), True),  # not 16 x a power of two
        ((16,), (8, 16), True),  # 2D
        ((16,), (128,), False),  # not known contiguous: one element
    ],
)
def test_invalid_forced_1d_load_tiles(
    bits: tuple[int, ...], load_tile: tuple[int, ...], contiguous: bool
) -> None:
    with pytest.raises(ValueError, match=r"load tile"):
        elementwise_layout(
            (8192,), bits, bits, load_tile=load_tile, contiguous=contiguous
        )


def test_2d_tiles_ignore_contiguity() -> None:
    layout = elementwise_layout((128, 256), (16,), (16,), contiguous=True)
    assert (layout.load_tile, layout.store_tile) == ((16, 16), (8, 16))


# --- The grid order -----------------------------------------------------------------

_ELEMENTWISE = """
      %v = vector.transfer_read %in[%i, %col], %pad {in_bounds = [true, true]} : memref<64x512xbf16>, vector<1x256xbf16>
      %r = arith.maximumf %v, %zero : vector<1x256xbf16>
      vector.transfer_write %r, %out[%i, %col] {in_bounds = [true, true]} : vector<1x256xbf16>, memref<64x512xbf16>"""
_REDUCTION = """
      %v = vector.transfer_read %in[%i, %col], %pad {in_bounds = [true, true]} : memref<64x512xbf16>, vector<1x256xbf16>
      %init = arith.constant dense<0.0> : vector<1xbf16>
      %s = vector.multi_reduction <add>, %v, %init [1] : vector<1x256xbf16> to vector<1xbf16>
      vector.transfer_write %s, %sums[%i] {in_bounds = [true]} : vector<1xbf16>, memref<64xbf16>"""
_CONTRACTION = """
      %a = vector.transfer_read %in[%i, %col], %pad {in_bounds = [true, true]} : memref<64x512xbf16>, vector<16x16xbf16>
      %acc = arith.constant dense<0.0> : vector<16x16xbf16>
      %c = vector.contract {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %a, %acc : vector<16x16xbf16>, vector<16x16xbf16> into vector<16x16xbf16>
      vector.transfer_write %c, %out[%i, %col] {in_bounds = [true, true]} : vector<16x16xbf16>, memref<64x512xbf16>"""


def _loop(rank: int, body: str, attrs: str = "", nested: bool = False) -> str:
    ivs = ["%i", "%j", "%k", "%l"][:rank]
    bounds = ["%c64", "%c2", "%c2", "%c2"][:rank]
    loop = (
        f"scf.parallel ({', '.join(ivs)}) = ({', '.join(['%c0'] * rank)}) to "
        f"({', '.join(bounds)}) step ({', '.join(['%c1'] * rank)}) {{\n"
        f"      %col = arith.muli %j, %c256 : index\n{body}\n"
        f"      scf.reduce\n    }}{attrs}"
    )
    if rank == 1:
        loop = loop.replace("%col = arith.muli %j", "%col = arith.muli %i")
    if nested:
        loop = (
            "scf.parallel (%o) = (%c0) to (%c2) step (%c1) {\n    "
            f"{loop}\n      scf.reduce\n    }}"
        )
    return f"""
module {{
  func.func @k(%in: memref<64x512xbf16>, %out: memref<64x512xbf16>, %sums: memref<64xbf16>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %pad = arith.constant 0.0 : bf16
    %zero = arith.constant dense<0.0> : vector<1x256xbf16>
    {loop}
    return
  }}
}}"""


def _map(text: str) -> tuple[list[bool], list[str | None]]:
    """``map_grid_innermost_first`` on every ``scf.parallel`` (outer first):
    whether it mapped each, and each one's mapping afterwards."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        loops: list[ir.Operation] = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "scf.parallel":
                loops.append(op)
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
        mapped = [map_grid_innermost_first(loop) for loop in loops]
        assert module.operation.verify()
        mappings = [
            str(loop.attributes[MAPPING_ATTR])
            if MAPPING_ATTR in loop.attributes
            else None
            for loop in loops
        ]
        return mapped, mappings


def _processors(mapping: str) -> list[str]:
    return [part.split(",")[0] for part in mapping.split("processor = ")[1:]]


@pytest.mark.parametrize(
    ("rank", "processors"),
    [(2, ["block_y", "block_x"]), (3, ["block_z", "block_y", "block_x"])],
)
def test_elementwise_grid_maps_innermost_first(
    rank: int, processors: list[str]
) -> None:
    mapped, (mapping,) = _map(_loop(rank, _ELEMENTWISE))
    assert mapped == [True]
    assert _processors(mapping) == processors
    assert mapping.count("map = (d0) -> (d0), bound = (d0) -> (d0)") == rank


_PRESET = (
    " {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), "
    "bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_y, "
    "map = (d0) -> (d0), bound = (d0) -> (d0)>]}"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_loop(2, _REDUCTION), [None]),
        (_loop(2, _CONTRACTION), [None]),
        (_loop(1, _ELEMENTWISE), [None]),
        (_loop(4, _ELEMENTWISE), [None]),
        (_loop(2, _ELEMENTWISE, attrs=_PRESET), ["block_x"]),
        (_loop(2, _ELEMENTWISE, nested=True), [None, None]),
    ],
    ids=["reduction", "contraction", "1d-grid", "4d-grid", "mapped", "nested"],
)
def test_other_grid_loops_are_left_alone(text: str, expected: list[str | None]) -> None:
    """Reductions and GEMMs keep the default (outermost-first) order; 1D and
    4D grids, loops that already have a mapping and nested loops are not
    touched."""
    mapped, mappings = _map(text)
    assert not any(mapped)
    assert [
        _processors(mapping)[0] if mapping else None for mapping in mappings
    ] == expected
