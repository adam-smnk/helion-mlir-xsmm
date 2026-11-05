"""Kernels that reduce one-row tiles, for the XPU layouts' row split
(:func:`~helion_mlir_backend._compiler.mlir_transforms.xegpu_layouts.plan_row_reduction`).
A block of 1 row folds the row dim: tiles are vectors, a row sum is a scalar."""

from __future__ import annotations

import helion.language as hl
import torch
from torch import Tensor


def l1_two_passes(x: Tensor) -> Tensor:
    """KernelBench 38: a sum over K tiles, then the output."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.abs(x[tile_m, tile_n])
        mean = total.sum(-1, keepdim=True) / n
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / mean
    return out


def l2_two_passes(x: Tensor) -> Tensor:
    """KernelBench 39: sums of squares, then the output."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            total = total + v * v
        inverse = torch.rsqrt(total.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inverse
    return out


def partial_sums_in_the_loop(x: Tensor) -> Tensor:
    """A reduction inside the K loop, added up over it: a 0-D vector carried."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.abs(x[tile_m, tile_n]).sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / total
    return out


def norm_of_each_tile(x: Tensor) -> Tensor:
    """A reduction inside the K loop with nothing carried: one SLM array, written and
    read on each iteration."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            out[tile_m, tile_n] = v / torch.abs(v).sum(-1, keepdim=True)
    return out


def standardize(x: Tensor) -> Tensor:
    """Two reductions, the second computed from the first, and a pass after each:
    ``(x - mean) / std`` of each row."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + x[tile_m, tile_n]
        mean = total.sum(-1, keepdim=True) / n
        squares = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n] - mean
            squares = squares + v * v
        std = torch.sqrt(squares.sum(-1, keepdim=True) / n)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean) / std
    return out


def minus_row_maximum(x: Tensor) -> Tensor:
    """A ``maximumf`` reduction: a row maximum that propagates NaN."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - row_top
    return out


def minus_row_minimum(x: Tensor) -> Tensor:
    """A ``minimumf`` reduction."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        low = hl.full([tile_m, block_n], float("inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            low = torch.minimum(low, x[tile_m, tile_n])
        row_low = torch.amin(low, dim=-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - row_low
    return out


def direct_sum_of_a_load(x: Tensor) -> Tensor:
    """A reduction of a load: no K loop carries the tile."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = x[tile_m, :].abs().sum(-1, keepdim=True)
        out[tile_m, :] = x[tile_m, :] / total
    return out


def bf16_with_f32_sums(x: Tensor) -> Tensor:
    """Tiles of bf16, the reduction in f32, the output in bf16."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.abs(x[tile_m, tile_n]).float()
        mean = total.sum(-1, keepdim=True) / n
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n].float() / mean).to(x.dtype)
    return out


def product(x: Tensor) -> Tensor:
    """A ``mul`` reduction: not among the kinds that are split."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.full([tile_m, block_n], 1.0, dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total * x[tile_m, tile_n]
        scale = torch.prod(total, dim=-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


def row_results(x: Tensor) -> Tensor:
    """A row result stored, no tile: nothing to split it over."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.abs(x[tile_m, tile_n])
        out[tile_m, :] = total.sum(-1, keepdim=True)
    return out


def tile_and_row_results(x: Tensor) -> tuple[Tensor, Tensor]:
    """A row result stored next to the tiles."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    sums = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.abs(x[tile_m, tile_n])
        row_sum = total.sum(-1, keepdim=True)
        sums[tile_m, :] = row_sum
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / row_sum
    return out, sums
