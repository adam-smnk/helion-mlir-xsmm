"""Row-wise normalizations on Helion's MLIR backend: softmax and GroupNorm.

Each tile holds whole rows (softmax) or whole groups (GroupNorm), so the
statistics are reduced inside the tile, in f32, and the result is written in
one pass.
"""

from __future__ import annotations

import os
from typing import Callable

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

# f32 elements of a tile's rows: its intermediates stay in L2.
_TILE_ELEMENTS = 1 << 16


def _softmax_kernel(x: Tensor) -> Tensor:
    """Softmax over the last dim of ``[M, N]`` in passes over column chunks:
    row maxima, sums of exponentials, then the normalized output. Whole-row
    ops instead get the row reduction fused into, and recomputed by, every
    chunk of the store."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            e = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
            out[tile_m, tile_n] = (e * inv).to(x.dtype)
    return out


def _second_operand(y: Tensor, extra: Tensor) -> Tensor:
    return y


def _group_norm_kernel(
    x3: Tensor,
    weight: Tensor,
    bias: Tensor,
    extra: Tensor,
    eps: hl.constexpr,
    epilogue: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """``epilogue(GroupNorm(x), extra)`` of ``x3`` ``[M, G, C/G]``; ``weight``,
    ``bias`` and ``extra`` are per-channel ``[G, C/G]``."""
    m, groups, group_size = x3.shape
    hl.specialize(group_size)
    out = torch.empty_like(x3)
    for tile_m, tile_g in hl.tile([m, groups]):
        y = x3[tile_m, tile_g, :].to(torch.float32)
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = centered * torch.rsqrt(var + eps) * weight[tile_g, :].to(
            torch.float32
        ) + bias[tile_g, :].to(torch.float32)
        y = epilogue(y, extra[tile_g, :].to(torch.float32))
        out[tile_m, tile_g, :] = y.to(x3.dtype)
    return out


_KERNELS: dict[tuple, helion.Kernel] = {}


def _kernel(fn: Callable, block_sizes: list[int]) -> helion.Kernel:
    key = (fn.__name__, *block_sizes)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            fn, backend="mlir", config=helion.Config(block_sizes=block_sizes)
        )
    return _KERNELS[key]


def _threads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))


def _row_tile(rows: int, width: int, other_tiles: int = 1) -> int:
    """Rows per tile: within ``_TILE_ELEMENTS`` and leaving every thread a tile."""
    per_thread = -(-rows * other_tiles // _threads())
    return max(1, min(_TILE_ELEMENTS // width, per_thread))


def softmax(x: Tensor) -> Tensor:
    """Softmax over the last dim of a 2-D ``x``."""
    if x.dim() != 2:
        raise ValueError(f"softmax() expects a 2-D tensor, got {tuple(x.shape)}")
    m, n = map(int, x.shape)
    block_n = min(n, 128)
    # The column block is registered first.
    return _kernel(_softmax_kernel, [block_n, _row_tile(m, block_n)])(x.contiguous())


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    epilogue: Callable[[Tensor, Tensor], Tensor] = _second_operand,
    extra: Tensor | None = None,
) -> Tensor:
    """``epilogue(GroupNorm(x), extra)`` of ``x`` ``[M, C]``; ``extra`` is an
    optional per-channel ``[C]`` operand of ``epilogue`` (default: identity)."""
    if x.dim() != 2 or x.shape[1] % num_groups:
        raise ValueError(
            f"group_norm() expects [M, C] with C divisible by {num_groups} groups, "
            f"got {tuple(x.shape)}"
        )
    m, c = map(int, x.shape)
    group_size = c // num_groups
    groups_per_tile = max(1, min(num_groups, _TILE_ELEMENTS // 32 // group_size))
    while num_groups % groups_per_tile:
        groups_per_tile -= 1
    tile_m = _row_tile(m, groups_per_tile * group_size, num_groups // groups_per_tile)
    kernel = _kernel(_group_norm_kernel, [tile_m, groups_per_tile])
    channels = (num_groups, group_size)
    return kernel(
        x.contiguous().view(m, num_groups, group_size),
        weight.view(channels),
        bias.view(channels),
        (weight if extra is None else extra).view(channels),
        hl.constexpr(eps),
        epilogue,
    ).view(m, c)
