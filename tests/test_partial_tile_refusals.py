"""Reductions at partial tiles that Helion leaves wrong, on every backend: refused
here, as a silent wrong result is the one outcome to avoid. Helion on Triton gives
the same wrong values on the B50 (2026-11-02)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError


def _minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def _multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a * b


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a + b


def mean_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.mean(-1, keepdim=True)
    return out


def var_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.var(-1, keepdim=True)
    return out


def std_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.std(-1, keepdim=True)
    return out


def reduce_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_minimum, tv, dim=1, keep_dims=True)
    return out


def reduce_mul(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_multiply, tv, dim=1, keep_dims=True)
    return out


def reduce_min_identity(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(
                _minimum, tv, dim=1, other=float("inf"), keep_dims=True
            )
    return out


def reduce_add(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_add, tv, dim=1, keep_dims=True)
    return out


def scan_min_reverse(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.associative_scan(_minimum, tv, dim=1, reverse=True)
    return out


def scan_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.associative_scan(_minimum, tv, dim=1)
    return out


def std_mean_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            std, mean = torch.std_mean(x[tm, tn], dim=-1, keepdim=True)
            out[tm, tn] = std + mean
    return out


def var_mean_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            var, mean = torch.var_mean(x[tm, tn], dim=-1, keepdim=True)
            out[tm, tn] = var + mean
    return out


def last_amax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        out[tm] = value.amax(-1)
    return out


def last_mean(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0
        out[tm] = value.mean(-1)
    return out


def last_sum_of_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = torch.exp(x[tm, tn])
        out[tm] = value.sum(-1)
    return out


def last_sum_of_shift(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] + 1.0
        out[tm] = value.sum(-1)
    return out


def last_sum_of_scaled(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0 - x[tm, tn]
        out[tm] = value.sum(-1)
    return out


def _logical_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(a, b)


def _bitwise_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a | b


def reduce_logical_or(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            tile = x[tm, tn]
            out[tm, tn] = (
                hl.reduce(_logical_or, tile > 1.5, dim=1, keep_dims=True).to(
                    torch.float32
                )
                + tile * 0.0
            )
    return out


def reduce_bitwise_or(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            tile = x[tm, tn]
            out[tm, tn] = (
                hl.reduce(_bitwise_or, tile > 1.5, dim=1, keep_dims=True).to(
                    torch.float32
                )
                + tile * 0.0
            )
    return out


def scan_sum_reverse(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            out[tm, tn] = hl.associative_scan(_add, x[tm, tn], dim=1, reverse=True)
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )(fn)


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (mean_tile, "over a partial tile"),
        (var_tile, "over a partial tile"),
        (std_tile, "over a partial tile"),
        (reduce_min, "other=0"),
        (reduce_mul, "other=0"),
        (scan_min_reverse, "reverse"),
        (std_mean_tile, "over a partial tile"),
        (var_mean_tile, "over a partial tile"),
        (last_amax, "after a loop"),
        (last_mean, "after a loop"),
        (last_sum_of_exp, "after a loop"),
        (last_sum_of_shift, "after a loop"),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_refused_at_a_partial_tile(fn: object, match: str) -> None:
    x = torch.randn(16, 60).abs() + 1.0
    with pytest.raises(UnsupportedOperationError, match=match):
        _kernel(fn)(x)


@pytest.mark.parametrize(
    "fn",
    [
        mean_tile,
        var_tile,
        std_tile,
        reduce_min,
        reduce_mul,
        scan_min_reverse,
        std_mean_tile,
        var_mean_tile,
        last_amax,
        last_mean,
        last_sum_of_exp,
        last_sum_of_shift,
    ],
    ids=lambda v: v.__name__,
)
def test_lowers_with_whole_tiles(fn: object) -> None:
    _kernel(fn)(torch.randn(16, 64).abs() + 1.0)


def mean_of_columns(x: torch.Tensor) -> torch.Tensor:
    """A mean over whole columns, with partial rows: right."""
    m, n = x.shape
    bm = hl.register_block_size(m)
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m, block_size=bm):
        out[tm, :] = x[tm, :].mean(-1, keepdim=True)
    return out


def test_mean_over_whole_columns_with_partial_rows() -> None:
    torch.manual_seed(0)
    x = torch.randn(60, 16)
    got = _kernel(mean_of_columns)(x).double()
    assert (got - x.double().mean(-1, keepdim=True)).abs().max() < 1e-5


def _chunk(x: torch.Tensor, op: object) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float64)
    for start in range(0, x.shape[1], 16):
        out[:, start : start + 16] = op(x.double()[:, start : start + 16])
    return out


@pytest.mark.parametrize(
    ("fn", "op"),
    [
        (reduce_min_identity, lambda v: v.amin(-1, keepdim=True)),
        (reduce_add, lambda v: v.sum(-1, keepdim=True)),
        (reduce_logical_or, lambda v: (v > 1.5).any(1, keepdim=True).double()),
        (reduce_bitwise_or, lambda v: (v > 1.5).any(1, keepdim=True).double()),
        (scan_sum_reverse, lambda v: v.flip(1).cumsum(1).flip(1)),
        (scan_min, lambda v: v.cummin(1).values),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_right_at_a_partial_tile(fn: object, op: object) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 60).abs() + 1.0
    got = _kernel(fn)(x).double()
    assert (got - _chunk(x, op)).abs().max() < 1e-5


def test_last_tile_sum_of_padding_that_stays_zero() -> None:
    x = torch.randn(16, 60).abs() + 1.0
    ref = x[:, 48:].double().sum(-1)
    assert (
        _kernel(last_sum_of_scaled)(x).double() - ref
    ).abs().max() / ref.max() < 1e-5


@pytest.mark.parametrize("fn", [mean_tile, reduce_min, scan_min_reverse, last_amax])
def test_refusal_names_static_shapes(fn: object) -> None:
    """Runtime bounds count every tile as partial, so divisible sizes don't help: the
    message has to say so."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[16, 16]),
    )
    with pytest.raises(UnsupportedOperationError) as refused:
        kernel(fn)(torch.randn(16, 64).abs() + 1.0)
    assert "static_shapes=True" in str(refused.value)
