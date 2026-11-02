"""Inner loops bounded by an enclosing tile's position (non-rectangular iteration):
results on the CPU, including partial tiles, and which of them XPU takes as full."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
from mlir.dialects import arith
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend._compiler.mlir.analysis.index_facts import IndexFacts
from helion_mlir_backend._compiler.mlir.analysis.index_facts import index_facts
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _config(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


def _kernel(fn: object, *block_sizes: int) -> object:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=_config(*block_sizes)
    )


def up_to_tile_end(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def behind_window(
    x: torch.Tensor, window: hl.constexpr, tile: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(max((tm.begin - window) // tile * tile, 0), tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def zero_span(x: torch.Tensor, width: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.begin, tm.begin + width):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def capped(x: torch.Tensor, cap: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(min(tm.end, cap)):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def _tiles(rows: int, tile: int) -> list[tuple[int, int]]:
    return [(b, min(b + tile, rows)) for b in range(0, rows, tile)]


def _expected(x: torch.Tensor, tile: int, keys: object) -> torch.Tensor:
    """``keys(begin, end)``: the range of keys of the row tile ``begin, end``."""
    out = torch.empty_like(x)
    for begin, end in _tiles(x.size(0), tile):
        first, last = keys(begin, end)
        out[begin:end] = x[first:last].sum(0)
    return out


@pytest.mark.parametrize(
    ("rows", "row_tile", "key_tile"),
    [
        (64, 16, 8),  # every tile full
        (64, 16, 16),  # one key tile per row tile
        (64, 12, 8),  # the bound is no multiple of the key tile: partial key tiles
        (60, 16, 8),  # a partial row tile
    ],
)
def test_loop_up_to_the_tile_end(rows: int, row_tile: int, key_tile: int) -> None:
    x = torch.randn(rows, 16)
    check_kernel(
        _kernel(up_to_tile_end, row_tile, key_tile),
        lambda a: _expected(a, row_tile, lambda _, end: (0, end)),
        [x],
    )


@pytest.mark.parametrize(
    ("window", "tile", "row_tile"),
    [(16, 8, 16), (20, 8, 16), (8, 8, 16), (0, 8, 16), (24, 8, 8)],
)
def test_loop_from_a_value(window: int, tile: int, row_tile: int) -> None:
    """A lower bound that is a value, clamped to 0 and aligned down."""
    x = torch.randn(64, 16)
    check_kernel(
        _kernel(behind_window, row_tile, tile),
        lambda a, *_: _expected(
            a,
            row_tile,
            lambda begin, end: (max((begin - window) // tile * tile, 0), end),
        ),
        [x, hl.constexpr(window), hl.constexpr(tile)],
    )


def test_loop_to_a_minimum_runs_no_key_past_it() -> None:
    x = torch.randn(64, 16)
    check_kernel(
        _kernel(capped, 16, 8),
        lambda a, *_: _expected(a, 16, lambda _, end: (0, min(end, 24))),
        [x, hl.constexpr(24)],
    )


def behind_any_row(x: torch.Tensor, window: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.begin - window, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def test_a_lower_bound_that_may_be_negative_is_refused() -> None:
    """Regression: the first row tiles read before the tensor, with no error and
    wrong rows (the XPU refused it, as a partial tile)."""
    with pytest.raises(UnsupportedOperationError, match="before the tensor"):
        _kernel(behind_any_row, 16, 8)(torch.randn(64, 16), hl.constexpr(16))


def test_a_lower_bound_proven_in_the_tensor_is_not_refused() -> None:
    """Negative: from the second row tile on, the same window stays in the tensor."""
    x = torch.randn(64, 16)

    check_kernel(
        _kernel(behind_window, 16, 8),
        lambda a, *_: _expected(a, 16, lambda b, e: (max(b - 16, 0), e)),
        [x, hl.constexpr(16), hl.constexpr(8)],
    )


def test_a_loop_that_spans_nothing_runs_no_tile() -> None:
    """Regression: ``ZeroDivisionError`` in the proof of full tiles: the tile of a
    loop whose span is 0 has size 0."""
    x = torch.randn(32, 8)
    check_kernel(
        _kernel(zero_span, 8, 4),
        lambda a, *_: torch.zeros_like(a),
        [x, hl.constexpr(0)],
    )


def attention_like(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A matmul per key tile up to the row tile's end."""
    out = torch.empty_like(q)
    for tm in hl.tile(q.size(0)):
        acc = hl.zeros([tm, q.size(1)], dtype=torch.float32)
        for tn in hl.tile(tm.end):
            scores = hl.zeros([tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("md,nd->mn", q[tm, :], k[tn, :])
            acc = acc + torch.einsum("mn,nd->md", scores.to(k.dtype), k[tn, :])
        out[tm, :] = acc.to(out.dtype)
    return out


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("rows", "keys", "blocks"),
    [
        (512, 512, (128, 96)),  # the row tile's end is no multiple of the key tile
        (512, 256, (128, 64)),  # the row tiles pass the keys' extent
    ],
    ids=["no-multiple", "past-the-tensor"],
)
def test_xpu_refuses_tiles_it_cannot_prove_full(
    rows: int, keys: int, blocks: tuple[int, ...]
) -> None:
    """Negative: full tiles are proven, not assumed; no masked loads on XPU."""
    args = [
        xpu_tensor(rows, 64, dtype=torch.bfloat16),
        xpu_tensor(keys, 64, dtype=torch.bfloat16),
    ]
    with pytest.raises(UnsupportedOperationError, match="partial tiles"):
        lower_xpu(_kernel(attention_like, *blocks), args)


def behind_without_a_clamp(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Starts before K for the first row tiles."""
    out = torch.empty_like(q)
    for tm in hl.tile(q.size(0)):
        acc = hl.zeros([tm, q.size(1)], dtype=torch.float32)
        for tn in hl.tile(tm.begin - 128, tm.end):
            scores = hl.zeros([tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("md,nd->mn", q[tm, :], k[tn, :])
            acc = acc + torch.einsum("mn,nd->md", scores.to(k.dtype), k[tn, :])
        out[tm, :] = acc.to(out.dtype)
    return out


@pytest.mark.isolated
def test_xpu_refuses_a_lower_bound_that_may_be_negative() -> None:
    """Negative: multiples of the tile, but no proof that it stays in the tensor."""
    args = [xpu_tensor(512, 64, dtype=torch.bfloat16)] * 2
    with pytest.raises(UnsupportedOperationError, match="before the tensor"):
        lower_xpu(_kernel(behind_without_a_clamp, 128, 64), args)


# --- index_facts ---------------------------------------------------------------------


def _facts_of(build: object, leaf: IndexFacts) -> IndexFacts:
    """``build(position, index)``'s facts, ``position`` a value with facts ``leaf``."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        index = ir.IndexType.get()
        with ir.InsertionPoint(module.body):
            position = arith.ConstantOp(index, 12345).result

            def constant(value: int) -> ir.Value:
                return arith.ConstantOp(index, value).result

            result = build(position, constant)
            return index_facts(result, lambda v: leaf if v == position else None)


_LEAF = IndexFacts(64, 0, 448)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda p, c: arith.AddIOp(p, c(64)).result, IndexFacts(64, 64, 512)),
        (lambda p, c: arith.AddIOp(p, c(32)).result, IndexFacts(32, 32, 480)),
        (lambda p, c: arith.SubIOp(p, c(128)).result, IndexFacts(64, -128, 320)),
        (lambda p, c: arith.MulIOp(p, c(2)).result, IndexFacts(128, 0, 896)),
        (lambda p, c: arith.MulIOp(c(-1), p).result, IndexFacts(64, -448, 0)),
        (lambda p, c: arith.FloorDivSIOp(p, c(32)).result, IndexFacts(2, 0, 14)),
        (lambda p, c: arith.FloorDivSIOp(p, c(128)).result, IndexFacts(1, 0, 3)),
        (lambda p, c: arith.MaxSIOp(p, c(0)).result, IndexFacts(64, 0, 448)),
        (lambda p, c: arith.MinSIOp(p, c(96)).result, IndexFacts(32, 0, 96)),
        (
            lambda p, c: arith.MaxSIOp(arith.XOrIOp(p, c(64)).result, c(32)).result,
            IndexFacts(1, 32, None),
        ),
        (lambda p, c: arith.XOrIOp(p, c(1)).result, IndexFacts()),
    ],
    ids=[
        "add",
        "add-other",
        "sub",
        "mul",
        "mul-negative",
        "fdiv",
        "fdiv-more",
        "max",
        "min",
        "max-of-unknown",
        "other",
    ],
)
def test_index_facts(build: object, expected: IndexFacts) -> None:
    assert _facts_of(build, _LEAF) == expected
