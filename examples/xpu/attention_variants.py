"""Forward attention variants on Intel Arc as fused flash-attention Helion kernels.

Each variant is a flash-attention kernel like ``llama_gqa_attention.py``'s with few
lines changed, written in full so that a diff between two kernels shows what a variant
takes. The same source also runs through Helion's Triton backend: rank-3 tiles and
``hl.dot``, which keeps matmul results in f32 there. Queries are not packed by head
group, as a mask depends on the query's row. Definitions and their
sources: ``README.md``. Run on a node with an Arc GPU:

    python examples/xpu/attention_variants.py check causal
    python examples/xpu/attention_variants.py check causal --vectorizer cmp
    python examples/xpu/attention_variants.py bench causal --seq 1024

``check`` compares a kernel to a float64 reference, per row and globally, on
NaN-poisoned outputs, on a second call and after other kernels ran. ``bench`` prints
the wall time per call, with the inputs out of the GPU's L2.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from typing import TYPE_CHECKING
from typing import NamedTuple

import helion
import helion.language as hl
import torch

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Callable


def plain(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
) -> torch.Tensor:
    """Flash attention of ``[B * H, S, D]``: each workgroup takes a tile of query rows of one head and runs over the K/V sequence in tiles with an online softmax. A matmul takes bf16 operands, its result f32 (DPAS); the probabilities are rounded to bf16 for ``P @ V``, the output once to bf16. The nearest kernel of every variant below."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def alibi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    heads: hl.constexpr,
) -> torch.Tensor:
    """ALiBi: ``score + (j - i) * 2 ** (-8 * (h + 1) / H)``; ``tb.index % heads`` is the head ``h``, ``-5.545... = -8 ln 2``."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        slope = torch.exp(
            -5.545177444479562 * ((tb.index % heads).to(torch.float32) + 1.0) / heads
        )
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            scores = scores + slope[:, None, None] * (j - i)
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def alibi_abs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    heads: hl.constexpr,
) -> torch.Tensor:
    """ALiBi as in the paper: ``score - |i - j| * 2 ** (-8 * (h + 1) / H)``."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        slope = torch.exp(
            -5.545177444479562 * ((tb.index % heads).to(torch.float32) + 1.0) / heads
        )
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            scores = scores - slope[:, None, None] * torch.abs(i - j)
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def softcap(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    cap: hl.constexpr,
) -> torch.Tensor:
    """Softcap: ``cap * tanh(score / cap)``."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = torch.tanh(scores * scale / cap) * cap
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def sigmoid_act(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
) -> torch.Tensor:
    """Sigmoid activation of the score: ``sigmoid(score)``, then the softmax."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = torch.sigmoid(scores * scale)
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def rel_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    slope: hl.constexpr,
) -> torch.Tensor:
    """Relative position bias: ``score + (i - j) * slope``."""
    streams, rows, dim = q.shape
    seq = k.size(1)
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            scores = scores + slope * (i - j)
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
) -> torch.Tensor:
    """Causal: keys ``j <= i``; K/V tiles past the query tile are skipped."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(tm.end):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            visible = j <= i
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def causal_strict(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
) -> torch.Tensor:
    """Keys strictly before the query, ``j < i``: row 0 sees no key, and gives NaN."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(tm.end):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            visible = j < i
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    window: hl.constexpr,
    kv_tile: hl.constexpr,
) -> torch.Tensor:
    """Sliding window: keys ``0 <= i - j <= window``; K/V tiles outside are skipped."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(max((tm.begin - window) // kv_tile * kv_tile, 0), tm.end):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            delta = i - j
            visible = (delta >= 0) & (delta <= window)
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def dilated_sw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    window: hl.constexpr,
    dilation: hl.constexpr,
    kv_tile: hl.constexpr,
) -> torch.Tensor:
    """Dilated sliding window: ``0 <= i - j <= window`` and ``(i - j) % dilation == 0``."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(max((tm.begin - window) // kv_tile * kv_tile, 0), tm.end):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            delta = i - j
            visible = (
                (delta >= 0)
                & (delta <= window)
                & (torch.floor(delta / dilation) * dilation == delta)
            )
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def block_diffusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    block: hl.constexpr,
) -> torch.Tensor:
    """Block diffusion: a query sees its block and earlier ones, ``j // block <= i // block``."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile((tm.end + block - 1) // block * block):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            visible = torch.floor(j / block) <= torch.floor(i / block)
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def document(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    documents: torch.Tensor,
) -> torch.Tensor:
    """Causal same-document: ``j <= i`` and ``documents[row stream, i] == documents[row stream, j]``."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(tm.end):
            scores = hl.dot(
                q_tile, k[tb, tn, :].transpose(1, 2), out_dtype=torch.float32
            )
            scores = scores * scale
            same = documents[tb, tm][:, :, None] == documents[tb, tn][:, None, :]
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            visible = same & (j <= i)
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[tb, tn, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def gemma2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    cap: hl.constexpr,
    group: hl.constexpr,
) -> torch.Tensor:
    """Gemma 2: grouped query heads, causal, softcap; ``k, v: [B * Hkv, S, D]``."""
    streams, rows, dim = q.shape
    out = torch.empty_like(q)
    for tb, tm in hl.tile([streams, rows]):
        kv = tb.begin // group
        q_tile = q[tb, tm, :]
        row_max = hl.full([tb, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, dim], dtype=torch.float32)
        for tn in hl.tile(tm.end):
            scores = hl.dot(
                q_tile,
                k[kv, tn, :][None, :, :].transpose(1, 2),
                out_dtype=torch.float32,
            )
            scores = torch.tanh(scores * scale / cap) * cap
            i = tm.index.to(torch.float32)[None, :, None]
            j = tn.index.to(torch.float32)[None, None, :]
            visible = j <= i
            scores = torch.where(
                visible,
                scores,
                hl.full([tb, tm, tn], float("-inf"), dtype=torch.float32),
            )
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + hl.dot(
                p.to(v.dtype), v[kv, tn, :][None, :, :], out_dtype=torch.float32
            )
            row_max = new_max
        out[tb, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


# --- Configs -------------------------------------------------------------------------

KV_TILE = 64
"""K/V rows per step: 32 DPAS calls per subgroup and matmul (2 x 8x16 blocks on 16
query rows), the tile of ``llama_gqa_attention`` and of lighthouse's attention layout."""

QUERY_TILE = 256
"""Query rows of a workgroup: 16 subgroups of 16 rows, 2 workgroups on an Xe-core of a
B50 (32 large-register threads). Every subgroup reads the K/V tile: subgroups of one
tile share it through the L1. Streams (batch times head) tile at 1: a grid dim."""

BLOCK_SIZES = [1, QUERY_TILE, KV_TILE]

SOFTCAP = 50.0
"""Gemma 2's ``attn_logit_softcapping``, the value of Triton-XPU's benchmark too."""

WINDOW = 256
"""Keys behind a query in the window kernels: 4 K/V tiles, so that skipping K/V tiles
shows at the shapes used (S >= 512)."""

DILATION = 2
DIFFUSION_BLOCK = 128
"""Smaller than a query tile, so a tile's rows see different keys: the mask matters."""

RELATIVE_SLOPE = 0.01
"""``rel_bias``: the slope is the kernel's constant (no source gives one)."""


class RowContext(NamedTuple):
    """Where a reference's score matrix ``[S, S]`` is: batch ``b`` of ``batch``, query
    head ``head`` of ``heads``, ``seq`` rows."""

    b: int
    batch: int
    head: int
    heads: int
    seq: int


class Variant(NamedTuple):
    """A kernel, the arguments after ``q, k, v, scale`` given ``batch, heads, seq,
    device``, and for the float64 reference: what maps the scores, the bias added to
    them and the visible mask, each from a :class:`RowContext`; then K/V heads."""

    fn: Callable[..., torch.Tensor]
    extra: Callable[[int, int, int, str], list[object]] = lambda *_: []
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    bias: Callable[[RowContext], torch.Tensor] | None = None
    mask: Callable[[RowContext], torch.Tensor] | None = None
    kv_heads: Callable[[int], int] = lambda heads: heads


def documents_of(batch: int, seq: int, device: str = "cpu") -> torch.Tensor:
    """Document ids ``[B, S]``, int32: per batch runs of lengths from 1 to about 2 K/V
    tiles and a few of length 1, so some rows see few keys."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.empty(batch, seq, dtype=torch.int32)
    for b in range(batch):
        lengths = torch.randint(1, 2 * KV_TILE, (seq,), generator=generator)
        lengths[::5] = 1
        runs = torch.repeat_interleave(torch.arange(seq), lengths)[:seq]
        ids[b] = runs.to(torch.int32)
    return ids.to(device)


def _positions(seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(seq)[:, None], torch.arange(seq)[None, :]


def _slope(head: int, heads: int) -> float:
    return 2.0 ** (-8.0 * (head + 1) / heads)


def _alibi_bias(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return (j - i) * _slope(ctx.head, ctx.heads)


def _alibi_abs_bias(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return -(i - j).abs() * _slope(ctx.head, ctx.heads)


def _relative_bias(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return (i - j) * RELATIVE_SLOPE


def _capped(scores: torch.Tensor) -> torch.Tensor:
    return SOFTCAP * torch.tanh(scores / SOFTCAP)


def _causal_mask(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return j <= i


def _strict_mask(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return j < i


def _window_mask(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return (i - j >= 0) & (i - j <= WINDOW)


def _dilated_mask(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return _window_mask(ctx) & ((i - j) % DILATION == 0)


def _diffusion_mask(ctx: RowContext) -> torch.Tensor:
    i, j = _positions(ctx.seq)
    return j // DIFFUSION_BLOCK <= i // DIFFUSION_BLOCK


def _document_mask(ctx: RowContext) -> torch.Tensor:
    ids = documents_of(ctx.batch, ctx.seq)[ctx.b]
    return _causal_mask(ctx) & (ids[:, None] == ids[None, :])


def _heads_arg(_batch: int, heads: int, _seq: int, _device: str) -> list[object]:
    return [hl.constexpr(heads)]


VARIANTS: dict[str, Variant] = {
    "plain": Variant(plain),
    "alibi": Variant(alibi, _heads_arg, bias=_alibi_bias),
    "alibi_abs": Variant(alibi_abs, _heads_arg, bias=_alibi_abs_bias),
    "softcap": Variant(softcap, lambda *_: [hl.constexpr(SOFTCAP)], score_fn=_capped),
    "sigmoid_act": Variant(sigmoid_act, score_fn=torch.sigmoid),
    "rel_bias": Variant(
        rel_bias, lambda *_: [hl.constexpr(RELATIVE_SLOPE)], bias=_relative_bias
    ),
    "causal": Variant(causal, mask=_causal_mask),
    "causal_strict": Variant(causal_strict, mask=_strict_mask),
    "sliding_window": Variant(
        sliding_window,
        lambda *_: [hl.constexpr(WINDOW), hl.constexpr(KV_TILE)],
        mask=_window_mask,
    ),
    "dilated_sw": Variant(
        dilated_sw,
        lambda *_: [
            hl.constexpr(WINDOW),
            hl.constexpr(DILATION),
            hl.constexpr(KV_TILE),
        ],
        mask=_dilated_mask,
    ),
    "block_diffusion": Variant(
        block_diffusion,
        lambda *_: [hl.constexpr(DIFFUSION_BLOCK)],
        mask=_diffusion_mask,
    ),
    "document": Variant(
        document,
        lambda batch, heads, seq, device: [
            documents_of(batch, seq, device).repeat_interleave(heads, 0)
        ],
        mask=_document_mask,
    ),
    "gemma2": Variant(
        gemma2,
        lambda *_: [hl.constexpr(SOFTCAP), hl.constexpr(2)],
        score_fn=_capped,
        mask=_causal_mask,
        kv_heads=lambda heads: heads // 2,
    ),
}


def make_kernel(
    name: str,
    *,
    backend: str = "mlir",
    block_sizes: list[int] | None = None,
    vectorizer: str = "off",
    **config: object,
) -> object:
    """The variant's kernel. ``xegpu_*`` and ``igc_vectorizer`` apply to ``mlir`` only."""
    options = dict(config)
    if backend == "mlir":
        options.setdefault("xegpu_nan_ignoring_maxima", [True])
        options.setdefault("igc_vectorizer", [vectorizer])
    return helion.kernel(
        VARIANTS[name].fn,
        backend=backend,
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes or BLOCK_SIZES, **options),
    )


def inputs_of(
    name: str, batch: int, heads: int, seq: int, dim: int, gain: float = 1.0
) -> list[torch.Tensor]:
    """``q, k, v``: randn on the CPU (``gain`` times, in Q and K), bf16."""
    generator = torch.Generator().manual_seed(0)
    kv_heads = VARIANTS[name].kv_heads(heads)
    q = torch.randn(batch, heads, seq, dim, generator=generator) * gain
    k = torch.randn(batch, kv_heads, seq, dim, generator=generator) * gain
    v = torch.randn(batch, kv_heads, seq, dim, generator=generator)
    return [x.to(torch.bfloat16) for x in (q, k, v)]


def call_args(
    name: str, batch: int, heads: int, seq: int, dim: int, device: str = "xpu"
) -> list[object]:
    """What comes after ``q, k, v`` in a call of the variant's kernel."""
    return [
        hl.constexpr(1.0 / math.sqrt(dim)),
        *VARIANTS[name].extra(batch, heads, seq, device),
    ]


def as_streams(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` as the kernels take it, ``[B * H, S, D]``; costs nothing."""
    return x.view(-1, x.size(2), x.size(3))


def attention(
    kernel: object,
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """The kernel on ``[B, H, S, D]`` inputs."""
    batch, heads, seq, dim = q.shape
    args = call_args(name, batch, heads, seq, dim, q.device.type)
    return kernel(as_streams(q), as_streams(k), as_streams(v), *args).view(q.shape)


def reference(
    name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """The variant with dense scores and an explicit mask, in float64 on the CPU."""
    batch, heads, seq, dim = q.shape
    variant = VARIANTS[name]
    group = heads // k.size(1)
    out = torch.empty(q.shape, dtype=torch.float64)
    for b in range(batch):
        for h in range(heads):
            ctx = RowContext(b, batch, h, heads, seq)
            scores = q[b, h].double() @ k[b, h // group].double().T / math.sqrt(dim)
            if variant.score_fn is not None:
                scores = variant.score_fn(scores)
            if variant.bias is not None:
                scores = scores + variant.bias(ctx)
            if variant.mask is not None:
                scores = scores.masked_fill(~variant.mask(ctx), -math.inf)
            out[b, h] = torch.softmax(scores, -1) @ v[b, h // group].double()
    return out


def visible_keys(name: str, batch: int, seq: int) -> torch.Tensor | None:
    """Visible keys per row ``[B, S]``, ``None`` without a mask."""
    mask = VARIANTS[name].mask
    if mask is None:
        return None
    contexts = [RowContext(b, batch, 0, 1, seq) for b in range(batch)]
    return torch.stack([mask(ctx).sum(-1) for ctx in contexts])


def errors(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Error normalized by ``max|ref|``, globally and per row; a row in ``ref`` that
    is all NaN must be all NaN in ``out``, and nowhere else may be NaN."""
    out = out.cpu().double()
    ref_nan = ref.isnan()
    extra_nan = int((out.isnan() & ~ref_nan).sum())
    lost_nan = int((~out.isnan() & ref_nan).sum())
    diff = (out - ref).abs().nan_to_num(0.0)
    finite = ref.nan_to_num(0.0).abs()
    row = diff.amax(-1) / finite.amax(-1).clamp_min(1e-300)
    return {
        "global_max": float(diff.max() / finite.max()),
        "global_mean": float(diff.mean() / finite.max()),
        "row_max": float(row.max()),
        "row_median": float(row.median()),
        "rows_over_1pct": float((row > 0.01).double().mean()),
        "extra_nan": extra_nan,
        "lost_nan": lost_nan,
    }


def _other_kernels(n: int) -> dict[str, Callable[[], object]]:
    """Torch kernels that leave different values in registers (stale reads)."""
    return {
        "sum_one": lambda: torch.ones(n, device="xpu").sum().item(),
        "sum_huge": lambda: torch.full((n,), 1e30, device="xpu").sum().item(),
        "sum_nan": lambda: torch.full((n,), float("nan"), device="xpu").sum().item(),
        "max_randn": lambda: torch.randn(n, device="xpu").max().item(),
        "mul": lambda: (torch.randn(n, device="xpu") * 2).cpu(),
        "matmul": lambda: (
            torch.randn(256, 256, device="xpu") @ torch.randn(256, 256, device="xpu")
        ).cpu(),
    }


def check(
    name: str,
    batch: int = 2,
    heads: int = 4,
    seq: int = 512,
    dim: int = 64,
    gain: float = 1.0,
    **kernel_options: object,
) -> dict[str, object]:
    """Errors against the reference on a NaN-poisoned output, and whether the bits are
    the same on a second call and after other kernels."""
    q, k, v = inputs_of(name, batch, heads, seq, dim, gain)
    ref = reference(name, q, k, v)
    device_inputs = [x.to("xpu") for x in (q, k, v)]
    kernel = make_kernel(name, **kernel_options)
    poison = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device="xpu")
    del poison
    torch.xpu.synchronize()
    first = attention(kernel, name, *device_inputs)
    torch.xpu.synchronize()
    result: dict[str, object] = errors(first, ref)
    result["second_call_equal"] = bool(
        torch.equal(
            attention(kernel, name, *device_inputs).nan_to_num(7.0),
            first.nan_to_num(7.0),
        )
    )
    wrong = []
    for other_name, other in _other_kernels(q.numel()).items():
        for _ in range(2):
            other()
            out = attention(kernel, name, *device_inputs)
            if not torch.equal(out.nan_to_num(7.0), first.nan_to_num(7.0)):
                wrong.append(other_name)
                break
    result["differs_after"] = wrong
    counts = visible_keys(name, batch, seq)
    if counts is not None:
        result["visible_min"] = int(counts.min())
        few = (counts <= 4)[:, None, :].expand(-1, heads, -1)
        if few.any():
            diff = (first.cpu().double() - ref).abs().nan_to_num(0.0)
            row = diff.amax(-1) / ref.nan_to_num(0.0).abs().amax(-1).clamp_min(1e-300)
            result["rows_few_keys"] = int(few.sum())
            result["few_keys_row_max"] = float(row[few].max())
    return result


def bench(
    name: str,
    batch: int = 4,
    heads: int = 16,
    seq: int = 1024,
    dim: int = 64,
    warmup: int = 20,
    calls: int = 100,
    **kernel_options: object,
) -> float:
    """Median wall time per synchronized call, us, after ``warmup`` calls; the inputs
    are out of the GPU's L2, filling a 256 MiB buffer before each call as AI-bench does."""
    inputs = [x.to("xpu") for x in inputs_of(name, batch, heads, seq, dim)]
    kernel = make_kernel(name, **kernel_options)
    flush = torch.empty(256 * 2**20, dtype=torch.int8, device="xpu")
    times = []
    with torch.no_grad():
        for i in range(warmup + calls):
            flush.zero_()
            torch.xpu.synchronize()
            start = time.perf_counter()
            attention(kernel, name, *inputs)
            torch.xpu.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1e6)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "bench"])
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--vectorizer", choices=["off", "cmp"], default="off")
    args = parser.parse_args()
    shape = {
        "batch": args.batch or (2 if args.command == "check" else 4),
        "heads": args.heads or (4 if args.command == "check" else 16),
        "seq": args.seq or (512 if args.command == "check" else 1024),
        "dim": args.dim,
    }
    if args.command == "check":
        result = check(
            args.variant, gain=args.gain, vectorizer=args.vectorizer, **shape
        )
        print(args.variant, args.vectorizer, result)
    else:
        time_us = bench(args.variant, vectorizer=args.vectorizer, **shape)
        print(f"{args.variant} {args.vectorizer} {time_us:.1f} us")


if __name__ == "__main__":
    main()
