"""Helion XPU kernels of KernelBench level 1 and level 2 problems, at the shapes
and dtypes of AI-bench's ``mlir-gpu-bench`` variant (``AI-bench/problems/specs/
KernelBench/level<n>/<name>.yaml``), for IR-level tests
(``test_xpu_kernelbench.py``).

The XeGPU pipeline compiles static, nicely shaped kernels: the block sizes below
divide every size they tile (no masking), so each problem's block sizes are
chosen per shape:

- Matmuls (bf16 operands, f32 accumulator, a bf16 result as PyTorch's matmul of
  bf16 tensors): block sizes ``[M tile, N tile, K tile]``, the workgroup tile and
  the K-loop step. 256x256 workgroups of 32 subgroups with K steps of 32, except
  where measured otherwise on the B50: 2 forces 32x64 subgroup tiles (the 64x32
  lighthouse infers stalls the memory pipe at K = 8192, ``GEMM_STANDARD``), and
  1 and 13 use 128x512 workgroups (``GEMM_SQUARE``), so the 256x256 rule no
  longer holds for them; lighthouse infers 32x64 for the transposed 16-18;
  a 128x128 workgroup tile for the
  256x256 output of 6_large_K (4 workgroups; the model splits K instead,
  :data:`GEMM_SPLIT_K`), :data:`GEMM_SMALL_K` for the output-bound 7 and 9
  (9's K padded from 32 to 64, as the model does). Transposed operands
  (16-18: A stored ``[K, M]``, B ``[N, K]``, as the specs give them) are loaded
  as transposed 16x16 tiles.
- Elementwise (bf16, 4096x393216): one-row workgroup tiles of 8192 columns
  (:data:`ELEMENTWISE_BLOCKS`, measured on the B50), computed in f32 like
  PyTorch's bf16 kernels: a 1D tile of contiguous elements, 8 subgroups of
  1024, gathers and scatters of 8 neighbouring bf16 (16 bytes) per work item;
  the grid runs the column tiles first.
- Row reductions (f32): ``[rows, column chunk]`` block sizes, 64 rows per
  workgroup (lighthouse's reduction class: 8 subgroups of 8 rows) and 64-column
  chunks; multi-pass over the chunks like AI-bench's CPU kernels. The models
  run them as three kernels instead (:data:`ROW_REDUCTIONS`,
  :data:`ROW_SEGMENT_COLS`): partials over contiguous row segments, the row
  results, an elementwise output kernel of one-row tiles.
- Rank 3 and 4 (batched matmuls, NCHW activations, a softmax over the last dim
  of ``[B, H, S, D]``, attention): block size 1 on the leading (batch, channel,
  head) dims, the 2D kernels' tiles on the last two. The pipeline folds unit
  tile dims, so each is its 2D kernel class, and the leading dims become
  workgroup grid dims (they multiply the workgroups: a small per-batch matrix
  still fills the GPU). Larger leading blocks (real 3D tiles) are rejected.
- Attention (97, bf16 ``[32, 32, 512, 64]``): written fused and tiled as flash
  attention (online softmax over K/V tiles), block sizes ``[1, 1, query rows,
  K/V rows]``; see :data:`ATTENTION_BLOCKS` for the hardware rationale.
- Level 2 (``LEVEL2_PROBLEMS``): ``nn.Linear`` (``x @ weight.T + bias``, the
  weight stored ``[out, in]``) and its whole elementwise epilogue fused in one
  GEMM kernel, in f32 on the accumulator, rounded to bf16 once at the store;
  see :data:`LINEAR_BLOCKS` for the hardware rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

BF16 = torch.bfloat16
F32 = torch.float32


def _kernel(
    block_sizes: list[int], **xegpu: object
) -> Callable[[Callable[..., object]], object]:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes, **xegpu),
    )


# --- Matmuls --------------------------------------------------------------------


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_split_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """f32 partial products ``[S, M, N]`` of ``a: [M, S, Kc]`` and
    ``b: [S, Kc, N]``, contiguous views of ``[M, K]`` and ``[K, N]`` operands
    (:func:`split_k_views`): chunk ``s`` of a K split into ``S`` chunks of
    ``Kc``. A chunk of ``a`` is the tile ``a[tm, s, tk]``, rows ``K`` apart:
    passing a strided ``[S, M, Kc]`` view instead would copy all of A on every
    call (kernel arguments are contiguous)."""
    m, splits, kc = a.size()
    _, _, n = b.size()
    out = torch.empty([splits, m, n], dtype=torch.float32, device=a.device)
    for ts, tm, tn in hl.tile([splits, m, n]):
        acc = hl.zeros([ts, tm, tn], dtype=torch.float32)
        for tk in hl.tile(kc):
            chunk = a[tm, ts.begin, tk].unsqueeze(0)
            acc = hl.dot(chunk, b[ts, tk, tn], acc=acc)
        out[ts, tm, tn] = acc
    return out


def _sum_splits(parts: torch.Tensor) -> torch.Tensor:
    """``parts.sum(0)`` of f32 ``[S, M, N]`` partial products, rounded to bf16
    (in split order): the second kernel of a split K."""
    splits, m, n = parts.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=parts.device)
    for tm, tn in hl.tile([m, n]):
        acc = parts[0, tm, tn]
        for index in hl.static_range(1, splits):
            acc = acc + parts[index, tm, tn]
        out[tm, tn] = acc.to(torch.bfloat16)
    return out


def split_k_views(
    a: torch.Tensor, b: torch.Tensor, splits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``a: [M, K]`` and ``b: [K, N]`` as ``splits`` K chunks, the operands of
    :func:`_matmul_split_k`: the contiguous views ``[M, S, K/S]`` and
    ``[S, K/S, N]``."""
    m, k = a.size()
    _, n = b.size()
    return a.view(m, splits, k // splits), b.view(splits, k // splits, n)


def _matmul_transposed_a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a.T @ b`` of ``a: [K, M]`` and ``b: [K, N]``."""
    k, m = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b.T`` of ``a: [M, K]`` and ``b: [N, K]``."""
    m, k = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_both(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a.T @ b.T`` of ``a: [K, M]`` and ``b: [N, K]``."""
    k, m = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


# --- Elementwise ------------------------------------------------------------------


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.01)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def selu(x: torch.Tensor) -> torch.Tensor:
    return torch.selu(x)


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.hardsigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.abs(x))


def elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x, alpha=1.0)


def hardtanh(x: torch.Tensor) -> torch.Tensor:
    return F.hardtanh(x, min_val=-1.0, max_val=1.0)


# The kernels apply the functions above to every element, in f32 (Helion kernels
# cannot close over a function: one kernel per function).


def _relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = relu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = leaky_relu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = sigmoid(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = tanh(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _swish(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = swish(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _selu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = selu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = hardsigmoid(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _softplus(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = softplus(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _softsign(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = softsign(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _elu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = elu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _hardtanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = hardtanh(x[tile].to(torch.float32)).to(x.dtype)
    return out


# --- Row reductions -------------------------------------------------------------


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """Row softmax in three passes over column chunks: maxima, sums of
    exponentials, then the output."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n] - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - row_top) * inv
    return out


def _log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row log-softmax: ``(x - max) - log(sum(exp(x - max)))``, in three
    passes. ``x - max`` first: ``x - (max + log(sum))`` rounds to the ulp of
    ``max``."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n] - row_top)
        log_total = torch.log(total.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - row_top) - log_total
    return out


def _l1_norm(x: torch.Tensor) -> torch.Tensor:
    """``x / mean(|x|, dim=1, keepdim=True)`` (KernelBench 38), in two passes.
    Subnormal inputs (``1e-40 * randn``) give inf: the row mean comes out 0
    on the B50 (the three kernels compute it)."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        mean = acc.sum(-1, keepdim=True) / n
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / mean
    return out


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """``x / norm(x, p=2, dim=1, keepdim=True)`` (KernelBench 39), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            acc = acc + v * v
        inv = torch.rsqrt(acc.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inv
    return out


# --- Row reductions in three kernels (ROW_SEGMENT_COLS) ------------------------
# 1. Per-segment partials of x.view(M * S, N / S).


def _segment_abs_sums(x: torch.Tensor) -> torch.Tensor:
    """``sum(|x|)`` of each row of ``x`` (a segment), as an f32 column."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        out[tile_m, :] = acc.sum(-1, keepdim=True)
    return out


def _segment_square_sums(x: torch.Tensor) -> torch.Tensor:
    """``sum(x * x)`` of each row of ``x`` (a segment), as an f32 column."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            acc = acc + v * v
        out[tile_m, :] = acc.sum(-1, keepdim=True)
    return out


def _segment_softmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The maximum ``t`` of each row of ``x`` (a segment) and its
    ``sum(exp(x - t))``, as f32 columns, in one pass.

    Each element of the carried tile keeps a running maximum ``top`` and sum
    ``total`` of ``exp(v - top)`` over its column chunks (an online softmax): a
    larger ``v`` rescales the sum by ``e = exp(top - v)`` and adds 1, any other
    adds ``e = exp(v - top)``; ``e = exp(-|v - top|)`` either way, one ``exp``
    per element. ``top`` starts at -FLT_MAX, not -inf, so ``v - top`` is never
    ``-inf - -inf``: -inf elements add ``exp(-inf) = 0``, a row of -inf ends
    with the sum 0 (NaN outputs, as torch's). A NaN makes its sum NaN (the
    comparison is false), and the row's results NaN. The row's maximum
    rescales the per-element sums once; the ``where`` (sums are never
    negative; NaN passes) keeps the product and the sum a reduction rather
    than a contraction."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    tops = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    sums = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], -3.4028234663852886e38, dtype=torch.float32)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            e = torch.exp(-torch.abs(v - top))
            total = torch.where(v > top, total * e + 1.0, total + e)
            top = torch.maximum(top, v)
        row_top = torch.amax(top, dim=-1, keepdim=True)
        scaled = torch.where(
            total < 0.0,
            hl.zeros([tile_m, block_n], dtype=torch.float32),
            total * torch.exp(top - row_top),
        )
        tops[tile_m, :] = row_top
        sums[tile_m, :] = torch.sum(scaled, dim=-1, keepdim=True)
    return tops, sums


# 2. Row results from the [M, S] partials.


def _means(parts: torch.Tensor, n: hl.constexpr) -> torch.Tensor:
    """``sum / n`` of each row of the partial sums ``parts`` (the output
    divides by it: ``n / sum`` overflows for tiny inputs, e.g. ``1e-40 *
    randn``)."""
    m, s = parts.shape
    block_s = hl.register_block_size(s)
    out = torch.empty([m, 1], dtype=torch.float32, device=parts.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            acc = acc + parts[tile_m, tile_s]
        out[tile_m, :] = acc.sum(-1, keepdim=True) / n
    return out


def _inverse_norms(parts: torch.Tensor) -> torch.Tensor:
    """``rsqrt(sum)`` of each row of the partial sums of squares ``parts``."""
    m, s = parts.shape
    block_s = hl.register_block_size(s)
    out = torch.empty([m, 1], dtype=torch.float32, device=parts.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            acc = acc + parts[tile_m, tile_s]
        out[tile_m, :] = torch.rsqrt(acc.sum(-1, keepdim=True))
    return out


def _softmax_rows(
    tops: torch.Tensor, sums: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row maximum ``t`` and ``1 / sum(sums * exp(tops - t))`` of segment
    maxima and sums ``[M, S]`` (see :func:`_segment_softmax`)."""
    m, s = tops.shape
    block_s = hl.register_block_size(s)
    row_tops = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    inverses = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_s], float("-inf"), dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            top = torch.maximum(top, tops[tile_m, tile_s])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            part = sums[tile_m, tile_s]
            total = total + torch.where(
                part < 0.0,
                hl.zeros([tile_m, block_s], dtype=torch.float32),
                part * torch.exp(tops[tile_m, tile_s] - row_top),
            )
        row_tops[tile_m, :] = row_top
        inverses[tile_m, :] = 1.0 / total.sum(-1, keepdim=True)
    return row_tops, inverses


def _log_softmax_rows(
    tops: torch.Tensor, sums: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row maximum ``t`` and ``log(sum(sums * exp(tops - t)))`` of
    segment maxima and sums ``[M, S]``, two columns: the output subtracts them
    one at a time (:func:`_log_softmax_out`)."""
    m, s = tops.shape
    block_s = hl.register_block_size(s)
    row_tops = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    log_sums = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_s], float("-inf"), dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            top = torch.maximum(top, tops[tile_m, tile_s])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            part = sums[tile_m, tile_s]
            total = total + torch.where(
                part < 0.0,
                hl.zeros([tile_m, block_s], dtype=torch.float32),
                part * torch.exp(tops[tile_m, tile_s] - row_top),
            )
        row_tops[tile_m, :] = row_top
        log_sums[tile_m, :] = torch.log(total.sum(-1, keepdim=True))
    return row_tops, log_sums


# 3. The output: elementwise one-row tiles into a preallocated ``out``.


def _scale_rows(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out = x * scale`` of a row scale ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale[tile_m, :]
    return out


def _divide_rows(
    x: torch.Tensor, divisor: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out = x / divisor`` of a row divisor ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] / divisor[tile_m, :]
    return out


def _log_softmax_out(
    x: torch.Tensor, tops: torch.Tensor, log_sums: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out = (x - tops) - log_sums`` of row columns ``[M, 1]``: ``x - tops``
    first, exact near the maximum (``x - (tops + log_sums)`` rounds to the ulp
    of ``tops``)."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = (x[tile_m, tile_n] - tops[tile_m, :]) - log_sums[
            tile_m, :
        ]
    return out


def _softmax_out(
    x: torch.Tensor, tops: torch.Tensor, inverses: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out = exp(x - tops) * inverses`` of row columns ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = (
            torch.exp(x[tile_m, tile_n] - tops[tile_m, :]) * (inverses[tile_m, :])
        )
    return out


def row_segments(shape: tuple[int, ...]) -> int | None:
    """The number of row segments ``S`` of the three-kernel row reductions of
    an f32 ``[M, N]`` input (the first of :data:`ROW_SEGMENT_COLS` dividing
    ``N`` into a multiple of 16 segments), or ``None`` (the single kernels):
    no such segments, ``M`` not a multiple of 8, fewer than 16 rows (a
    one-workgroup row results kernel fails to lower) or more than 2^24
    segment rows (a 2D block surface). Mirrors AI-bench's ``row_segments``."""
    m, n = shape
    if m % 8 or m < 16:
        return None
    for cols in ROW_SEGMENT_COLS:
        if n % cols == 0 and (n // cols) % 16 == 0 and m * (n // cols) <= 1 << 24:
            return n // cols
    return None


def rows_block(rows: int) -> int:
    """Workgroup rows of a reduction over ``rows``: the largest of 64 to 8
    dividing them into at least two workgroups, or 8."""
    return next((b for b in (64, 32, 16, 8) if rows % b == 0 and rows // b >= 2), 8)


ROW_REDUCTIONS = {
    "23_Softmax": (_segment_softmax, _softmax_rows, _softmax_out),
    "24_LogSoftmax": (_segment_softmax, _log_softmax_rows, _log_softmax_out),
    "38_L1Norm_": (_segment_abs_sums, _means, _divide_rows),
    "39_L2Norm_": (_segment_square_sums, _inverse_norms, _scale_rows),
}
"""The three kernels of each row reduction (partials, row results, output);
``_means`` takes ``N`` too."""


def row_reduction_kernels(
    name: str, shape: tuple[int, int]
) -> tuple[tuple[object, ...], tuple[list[int], ...]]:
    """The three kernels of the row reduction ``name`` (:data:`ROW_REDUCTIONS`)
    for an f32 input of ``shape`` with segments (:func:`row_segments`), and
    their block sizes: ``[64, rows]`` over the ``[M * S, N / S]`` view,
    ``[S block, rows]`` over ``[M, S]``, and the one-row output tiles."""
    segments = row_segments(shape)
    assert segments is not None, shape
    m, n = shape
    block_s = next(b for b in (64, 32, 16) if segments % b == 0)
    blocks = (
        [64, rows_block(m * segments)],
        [block_s, rows_block(m)],
        [1, next(b for b in range(min(8192, n), 0, -16) if n % b == 0)],
    )
    return ROW_REDUCTIONS[name], blocks


def row_reduction(name: str, x: torch.Tensor) -> torch.Tensor:
    """The row reduction ``name`` of an f32 ``[M, N]`` ``x`` in its three
    kernels, as AI-bench's models run it: the output allocated first, the
    partials of the contiguous segment view, the row results, the output."""
    fns, blocks = row_reduction_kernels(name, tuple(x.shape))
    partials, rows, apply = (
        _kernel(list(b))(fn) for fn, b in zip(fns, blocks, strict=True)
    )
    m, n = x.shape
    segments = row_segments(tuple(x.shape))
    out = torch.empty_like(x)
    parts = partials(x.view(m * segments, n // segments))
    parts = parts if isinstance(parts, tuple) else (parts,)
    extra = (n,) if name == "38_L1Norm_" else ()
    results = rows(*(p.view(m, segments) for p in parts), *extra)
    results = results if isinstance(results, tuple) else (results,)
    return apply(x, *results, out)


# --- Rank 3 and 4 ----------------------------------------------------------------


def _batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.bmm(a, b)`` of ``[B, M, K]`` and ``[B, K, N]`` (KernelBench 3)."""
    batch, m, k = a.size()
    _, _, n = b.size()
    out = torch.empty([batch, m, n], dtype=a.dtype, device=a.device)
    for tb, tm, tn in hl.tile([batch, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tb, tm, tk], b[tb, tk, tn], acc=acc)
        out[tb, tm, tn] = acc.to(a.dtype)
    return out


def _tensor_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.matmul(a, b)`` of ``[N, M, K]`` and ``[K, L]`` (KernelBench 10):
    ``B`` is shared by every batch (an einsum: ``hl.dot`` takes operands of one
    rank)."""
    batch, m, k = a.size()
    _, n = b.size()
    out = torch.empty([batch, m, n], dtype=a.dtype, device=a.device)
    for tb, tm, tn in hl.tile([batch, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + torch.einsum("bmk,kn->bmn", a[tb, tm, tk], b[tk, tn])
        out[tb, tm, tn] = acc.to(a.dtype)
    return out


def _softmax_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim of ``[B, H, S, D]`` (attention scores), in
    three passes over column chunks like :func:`_softmax`."""
    batch, heads, rows, cols = x.shape
    block_n = hl.register_block_size(cols)
    out = torch.empty_like(x)
    for tb, th, tm in hl.tile([batch, heads, rows]):
        top = hl.full([tb, th, tm, block_n], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(cols, block_size=block_n):
            top = torch.maximum(top, x[tb, th, tm, tn])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tb, th, tm, block_n], dtype=torch.float32)
        for tn in hl.tile(cols, block_size=block_n):
            total = total + torch.exp(x[tb, th, tm, tn] - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tn in hl.tile(cols, block_size=block_n):
            out[tb, th, tm, tn] = torch.exp(x[tb, th, tm, tn] - row_top) * inv
    return out


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v)`` of ``[B, H, S, D]`` (no mask,
    scale ``1/sqrt(D)``, KernelBench 97) as flash attention: each workgroup
    takes a tile of query rows of one (batch, head) and runs over the K/V
    sequence in tiles, with an online softmax (running row maxima and sums,
    the output rescaled when the maximum grows). The matmuls take bf16
    operands and accumulate in f32 (DPAS); the softmax is f32; ``P`` is
    rounded to V's dtype for ``P @ V``, the output to Q's. Also run on the CPU
    target (numerics) and with other shapes and block sizes (``test_xpu_ir``).

    The tiles are 4D with unit batch and head blocks, so the contractions are
    einsums (``hl.dot`` takes 2D or 3D operands). An einsum of bf16 tiles
    returns bf16; added to an f32 accumulator it accumulates in f32 (the
    backend fuses ``acc + einsum`` into one contraction), so ``Q @ K^T``
    accumulates into zeros."""
    batch, heads, m, d = q.shape
    n = k.size(2)
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    for tb, th, tm in hl.tile([batch, heads, m]):
        q_tile = q[tb, th, tm, :]
        row_max = hl.full([tb, th, tm], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tb, th, tm], dtype=torch.float32)
        acc = hl.zeros([tb, th, tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            scores = hl.zeros([tb, th, tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("bhmd,bhnd->bhmn", q_tile, k[tb, th, tn, :])
            scores = scores * scale
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, :, None] + torch.einsum(
                "bhmn,bhnd->bhmd", p.to(v.dtype), v[tb, th, tn, :]
            )
            row_max = new_max
        out[tb, th, tm, :] = (acc / row_sum[:, :, :, None]).to(out.dtype)
    return out


# --- Level 2: nn.Linear + elementwise epilogue -------------------------------------
#
# ``epilogue(x @ weight.T + bias)`` of bf16 ``x: [M, K]``, ``weight: [N, K]`` (as
# ``nn.Linear`` stores it: the GEMM reads B transposed) and ``bias: [N]``, in one
# kernel per problem (a Helion kernel cannot close over an epilogue): the f32
# accumulator plus the bias row (broadcast over the tile's rows), the epilogue in
# f32, one rounding to bf16 at the store. The models' scalar inits are
# ``hl.constexpr`` arguments, as AI-bench's level 1 XPU kernels take them.


def _linear_subtract_multiply_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract_value: hl.constexpr,
    multiply_value: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 9: ``relu((linear(x) - subtract) * multiply)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - subtract_value) * multiply_value).to(x.dtype)
    return out


def _linear_multiply_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: hl.constexpr,
    negative_slope: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 12: ``leaky_relu(linear(x) * multiplier)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = (acc + bias[tn]) * multiplier
        out[tm, tn] = F.leaky_relu(y, negative_slope=negative_slope).to(x.dtype)
    return out


def _linear_mish_mish(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """KernelBench level 2 29: ``mish(mish(linear(x)))``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = F.mish(F.mish(y)).to(x.dtype)
    return out


def _linear_scaling_residual_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 40: ``y * scaling_factor + y`` of ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (y * scaling_factor + y).to(x.dtype)
    return out


def _linear_swish_scaling(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 59: ``y * sigmoid(y) * scaling_factor`` of
    ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (y * torch.sigmoid(y) * scaling_factor).to(x.dtype)
    return out


def _linear_relu_divide(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 63: ``relu(linear(x)) / divisor``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (torch.relu(y) / divisor).to(x.dtype)
    return out


def _linear_sigmoid_scaling_residual_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 70: ``sigmoid(y) * scaling_factor + y`` of
    ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (torch.sigmoid(y) * scaling_factor + y).to(x.dtype)
    return out


def _linear_swish_divide_clamp_tanh_clamp(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """KernelBench level 2 81: ``clamp(tanh(clamp(swish(y) / 2, -1, 1)), -1, 1)``
    of ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        y = torch.clamp(y * torch.sigmoid(y) / 2.0, min=-1.0, max=1.0)
        out[tm, tn] = torch.clamp(torch.tanh(y), min=-1.0, max=1.0).to(x.dtype)
    return out


# --- Problems ---------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """A KernelBench problem at its ``mlir-gpu-bench`` variant."""

    name: str
    """The KernelBench file name, without ``.py``."""
    kind: str
    """``matmul``, ``elementwise``, ``reduction``, ``attention`` or ``linear``
    (level 2: ``nn.Linear`` and an epilogue)."""
    fn: Callable[..., object]
    shapes: tuple[tuple[int, ...], ...]
    """The input shapes, in argument order."""
    dtype: torch.dtype
    block_sizes: list[int]
    grid: tuple[int, ...]
    """The workgroup grid: the output size over the workgroup tile, grid x
    first (elementwise kernels: the innermost dim first)."""
    scalars: tuple[float, ...] = ()
    """``hl.constexpr`` arguments after the tensors (a model's scalar inits)."""
    xegpu: dict[str, object] = field(default_factory=dict, compare=False)
    """``xegpu_*`` config keys besides the block sizes (default: inferred)."""
    kernel: object = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        kernel = _kernel(self.block_sizes, **self.xegpu)(self.fn)
        object.__setattr__(self, "kernel", kernel)

    def args(self, *tensors: object) -> list[object]:
        """``tensors`` (one per shape) and the scalars, as the kernel takes them."""
        return [*tensors, *(hl.constexpr(value) for value in self.scalars)]


_GEMM_BLOCKS = [256, 256, 32]
GEMM_SQUARE = (
    [128, 512, 32],
    {"xegpu_sg_tiles": [[32, 64]], "xegpu_load_tiles": [[32, 32, 16, 32]]},
)
"""1_Square and 13_Matmul_for_symmetric_matrices (both 4096^3) on the Arc B50
(16 Xe cores, one 32-subgroup workgroup per core, 18 MB L2, about 220 GB/s
DRAM): ``(block sizes, xegpu config)``. Keep in sync with ``GEMM_SQUARE`` in
AI-bench's ``backends/utils/helion_mlir_xpu_utils/matmul.py``.

- 128x512 workgroups, not 256x256 (the reason is DRAM traffic per wave).
  Workgroups are dispatched in linear order with the first grid dim (M)
  fastest, so a wave of 16 workgroups (one per Xe core) of 256x256 tiles is a
  full column of M blocks and re-reads all of A: about 544 MB of reads per
  GEMM, 2.4 ms at the measured 220 GB/s, of 3.9 ms. A wave of 128x512 tiles
  covers half of M, which halves A's traffic per wave (about 288 MB per GEMM).
  The price is more L2-to-core traffic: a workgroup reads 128x32 + 32x512 bf16
  (40 KB) per K step for 4.2 Mflop, 102 flop per byte against 128 for 256x256.
  Forcing 32x64 subgroup tiles on 256x256 workgroups does not help (34.3
  TFLOPS): it is the workgroup shape.
- Subgroup tiles 32x64 (4x8 subgroups, 128 of the 256 registers for the
  accumulator). Lighthouse's inference also gives 32x64 on 128x512
  workgroups (only on 256x256 it picks 64x32); the tile is pinned so the
  config does not depend on the selector. Forcing 64x32 costs about 3 % at the
  70 W cap (41.6 against 42.8 TFLOPS device time, at a 5 % lower clock).
- Loads: A 32x32 (a subgroup's whole K step, 64 bytes wide, in one block read),
  B 16x32, as lighthouse's tuned parameters for this shape. A 8x32 / B 32x32
  and A 32x32 / B 32x32 measure the same within noise (42.6, 42.6, 42.2
  TFLOPS). Prefetches are inferred.
- B50-specific: N must be a multiple of 512 and a 16-workgroup wave must cover
  half of M. On a card with 32 Xe cores (B70) a 32-workgroup wave of 128x512
  tiles is again a full M column (32 M blocks); the portable fix is a grouped
  workgroup walk (plan P3).
- Measured 2026-10-30 on the B50 (device time, interleaved; efficiency =
  TFLOPS / (16 Xe cores x 2048 flop/clk x sampled clock), as the card runs at
  its 70 W limit): 42.6 TFLOPS (94 %) against 34.5 (64 %) for the inferred
  256x256 / 64x32; torch 46.0 (95 %). Bit-identical to ``torch.matmul``.
"""
GEMM_STANDARD = (
    [256, 256, 32],
    {
        "xegpu_sg_tiles": [[32, 64]],
        "xegpu_load_tiles": [[8, 32, 32, 32]],
        "xegpu_prefetch_tiles": [[16, 16, 16, 16]],
        "xegpu_prefetch_depths": [[1, 1]],
    },
)
"""2_Standard (M 2048, N 4096, K 8192) on the Arc B50: ``(block sizes, xegpu
config)``. Keep in sync with ``GEMM_STANDARD`` in AI-bench's
``backends/utils/helion_mlir_xpu_utils/matmul.py``.

- 256x256 workgroups (128 of them: 8 waves of 16 over the Xe cores).
- Subgroup tiles 32x64 (8x4 subgroups) in place of the 64x32 lighthouse
  infers: 64x32 collapses here (23.5 TFLOPS, 51 %) against 42.8 (94 %) with
  32x64. The cause is a memory-pipe stall with 64 strided A rows per subgroup
  at K = 8192 (EU stall sampling: SendStall 41 % against 6 %); it is not shown
  to be a pitch effect alone.
- Loads A 8x32, B 32x32 (lighthouse's tuned parameters for this shape) and
  prefetches of 16x16 at depth 1 (also lighthouse's). In one interleaved run:
  32x64 with inferred loads and prefetches 42.8, with the loads alone 42.6,
  with the loads and the 16x16 prefetch tiles 43.4: the gain is from the
  prefetch tiles. Prefetch depth 2 is equal to depth 1 (43.5 against 43.4).
- B50-specific, like the tiles of ``GEMM_SQUARE``: 128 workgroups are 8 waves
  of 16 on 16 Xe cores.
- Measured 2026-10-30 on the B50 (device time, interleaved; efficiency =
  TFLOPS / (16 Xe cores x 2048 flop/clk x sampled clock), as the card runs at
  its 70 W limit): 43.4-43.6 TFLOPS (95 %); torch 46.6 (98 %). Bit-identical
  to ``torch.matmul``.
"""

GEMM_SMALL_K = ([64, 128, 32], {"xegpu_sg_tiles": [[8, 64]]})
"""7_Matmul_with_small_K_dimension_ (32768x32768x64) and, with K zero-padded
from 32 to 64 by the model, 9_Tall_skinny_matrix_multiplication_ on the Arc
B50: ``(block sizes, xegpu config)``. Keep in sync with ``GEMM_SMALL_K`` in
AI-bench's ``backends/utils/helion_mlir_xpu_utils/matmul.py`` (the full
rationale and the measurements).

- Output-store bound (2 GiB of bf16 written, 8 MiB of operands read), so the
  grid runs columns first (``xpu_params.columns_first``): 53.6 -> 140.9 GB/s.
- 64x128 workgroups of 16 subgroups of 8x64: each subgroup writes 8 rows of
  128 bytes, the 16-bit block stores being 32 bytes wide (256x256 with the
  inferred 64x32: 86 GB/s columns first). 64 rows are lighthouse's minimum
  (16 prefetching subgroups at a K step of 32).
- Measured 2026-10-31 on the B50 (e2e): 15.1 ms, 142.7 GB/s (torch 18.4 ms,
  117 GB/s; was 43.5 ms). Bit-identical to ``torch.matmul``.
"""
GEMM_SPLIT_K = (
    32,
    [1, 256, 256, 32],
    {
        "xegpu_sg_tiles": [[32, 64]],
        "xegpu_load_tiles": [[32, 32, 16, 32]],
        "xegpu_prefetch_depths": [[4, 2]],
    },
    [8, 256],
)
"""6_Matmul_with_large_K_dimension_ (256x256x524288) on the Arc B50 as a split
K: ``(splits, GEMM block sizes, its xegpu config, reduction block sizes)``.
Keep in sync with ``GEMM_SPLIT_K`` in AI-bench's
``backends/utils/helion_mlir_xpu_utils/matmul.py`` (full rationale).

- 32 K chunks of 16384 (contiguous views, :func:`split_k_views`): 32
  workgroups of 256x256 (32 subgroups of 32x64, A read 32x32 per block read,
  prefetched 4 K steps ahead, B 2) write f32 partial sums ``[32, 256, 256]``;
  ``_sum_splits`` adds them (8-row tiles, 32 workgroups).
- 3.33 ms e2e against 9.95 unsplit (4 workgroups of 128x128 on 16 Xe cores);
  torch 3.16 ms.
"""

ELEMENTWISE_BLOCKS = [1, 8192]
"""Elementwise block sizes ``[rows, cols]`` of the bf16 4096x393216 problems
(19-32) on the Arc B50, measured 2026-10-31. Keep in sync with
``ELEMENTWISE_BLOCKS`` in AI-bench's
``backends/utils/helion_mlir_xpu_utils/elementwise.py``.

These kernels move bytes (2 per element read, 2 written) and the bound is the
DRAM: the read+write copy ceiling of the card is 198 GB/s, torch reaches it.
What decides the bandwidth:

- The 16-bit store width. A 2D block store of bf16 writes rows of 16
  elements, 32 bytes: half a 64-byte cache line (XeVM ``blockstore2d`` takes
  ``tile_width`` 16 for 16-bit elements). With 2D tiles that is the limit
  (f32 loads + bf16 8x16 stores 167 GB/s; bf16 loads + f32 stores 230; f32
  both ways 197). A one-row tile folds to a 1D tile of contiguous elements,
  which the target loads and stores with gathers and scatters of 8 bf16 per
  work item (``inst_data = [128]``, ``lane_data = [8]``): 16 bytes per work item,
  256 contiguous bytes per subgroup instruction, as torch's vectorized
  kernels do. 196 GB/s (99 %); one element per work item gave 165.
- Rows per workgroup. Rows are 768 KiB apart; a tile of R rows touches R DRAM
  pages at once (with the column-fastest grid, 2D tiles: 2 rows 172 GB/s, 8
  rows 149, 128 rows 115). One row is the minimum.
- The grid order (backend: ``map_elementwise_grid``): column tiles first, so
  the workgroups resident at a time stream one contiguous range of a row
  (1-row tiles of 4096 and 16384 columns: 196 GB/s columns-first, 116 and
  170 rows-first).
- 8192 columns: 8 subgroups of 1024 elements (64 GRFs of f32 math, a small
  part of the large register file), 16 KB read and 16 KB written per
  workgroup, 48 x 4096 = 196608 workgroups. 1024 to 32768 columns measured
  the same for relu (195.7-196.4 GB/s); 8192 divides the spec's 393216
  columns and any multiple of 8192 (8192x8192 too).
- 29_Softplus stays below the ceiling for another reason, a gap of the math
  lowering: its f32 ``log1p`` becomes a scalar ``__spirv_ocl_log1pf`` call per
  element, and the kernel runs at about 147 GB/s (at a power-limited clock,
  for every tile width tried, 1024-16384 columns) where torch's softplus
  reaches 191 GB/s.
"""
_ELEMENTWISE_SHAPE = (4096, 393216)
_NCHW_BLOCKS = [1, 1, 128, 256]
"""2D tiles with leading unit dims (the NCHW form of 19, an IR test of 2D
elementwise tiles): 32 subgroups of 32x32, 16x16 block loads, 8x16 stores."""
_REDUCTION_BLOCKS = [64, 64]
ROW_SEGMENT_COLS = (2048, 1024, 512, 256, 128)
"""Row segment lengths of the three-kernel f32 row reductions (23, 24, 38, 39:
:data:`ROW_REDUCTIONS`, :func:`row_segments`), in order of preference,
measured on the Arc B50 2026-11-01. Keep in sync with ``ROW_SEGMENT_COLS`` in
AI-bench's ``backends/utils/helion_mlir_xpu_utils/normalization.py`` (the
full rationale).

- The single kernels (``PROBLEMS``) store their output in 2D blocks of 8 rows
  (rows 256 KiB-1.5 MiB apart: ~112 GB/s read+write, as the 2D elementwise
  tiles). The output kernel takes the one-row tiles of the elementwise
  problems instead (1D, columns first): 195 GB/s, the copy ceiling.
- Their reductions read 8-row tiles of rows that far apart: 165-180 GB/s for
  any tile shape. Contiguous segments of 128-2048 columns (a view
  ``[M * S, N / S]``, no copy) read at 200-208 GB/s (222 GB/s ceiling).
- Softmax and log-softmax take the segment maximum and sum in one pass (a
  running maximum and sum per element, one ``exp`` each): two reads and a
  write, where the single kernel reads three times.

4096x393216 softmax 196 -> 98 ms (torch 130); 8192x65536 L2 norm 48.5 ->
33.6 ms (torch 31.7).
"""
ATTENTION_BLOCKS = [1, 1, 128, 64]
"""Flash-attention block sizes ``[batch, head, query rows, K/V rows]``, picked
from the hardware (Xe2, large register file) for a head dim of 64:

- Batch and head blocks 1: each workgroup works on one (batch, head) matrix
  (unit tile dims are folded; the grid is B x H x S/128 = 32x32x4).
- 128 query rows per workgroup: 8 subgroups of 16 rows (lighthouse's attention
  layout: subgroup tiles of 16 rows, two 8-row DPAS blocks), 128 threads of
  16 lanes. Every subgroup reads the whole K and V tiles, so more rows per
  workgroup share each K/V tile through the L1 cache; 8 subgroups (of at
  most 32) still leave room for several workgroups per Xe core, and
  32x32x4 = 4096 workgroups fill the GPU.
- K/V tiles of 64 rows: per subgroup, ``Q @ K^T`` is 16x64 (2x4 DPAS 8x16
  blocks, 4 K steps of 16 over the head dim) and ``P @ V`` 16x64 over 64 K/V
  rows (4 K steps): 32 DPAS each per step. In 64-byte registers Q (16x64
  bf16) takes 32, the scores and the accumulator (16x64 f32) 64 each: 160 of
  the 256-register file, plus the K or V tile (64x64 bf16, 128 registers)
  would be 288 if it stayed resident whole. Whether the compiler streams it
  (block reads per DPAS step) or spills is open until measured on hardware;
  K/V tiles of 32 (64 registers) are the fallback. 64 is also lighthouse's
  attention reduction tile, and whole 32x32 V blocks.
- 2D block loads: Q in 16x16 blocks, V in 32x32 blocks (the VNNI-transformed
  B operand), K transposed (``Q @ K^T`` of K stored ``[seq, head dim]``) in
  16x8 blocks of 32-bit pairs; the output is stored in 8x16 blocks.
"""


_MATMULS = {
    "": _matmul,
    "a": _matmul_transposed_a,
    "b": _matmul_transposed_b,
    "ab": _matmul_transposed_both,
}


def _gemm(
    name: str,
    mnk: tuple[int, int, int],
    transposed: str = "",
    blocks: list[int] = _GEMM_BLOCKS,
    xegpu: dict[str, object] | None = None,
    columns_first: bool = False,
) -> Problem:
    """A matmul of ``M x K`` and ``K x N`` operands, stored ``[K, M]`` /
    ``[N, K]`` when ``transposed`` names them; ``xegpu`` are the config keys
    besides the block sizes (default: inferred). ``columns_first``: the grid
    runs the N tiles first (x), as the backend maps output-bound GEMMs."""
    m, n, k = mnk
    a = (k, m) if "a" in transposed else (m, k)
    b = (n, k) if "b" in transposed else (k, n)
    grid = (m // blocks[0], n // blocks[1])
    if columns_first:
        grid = grid[::-1]
    return Problem(
        name,
        "matmul",
        _MATMULS[transposed],
        (a, b),
        BF16,
        blocks,
        grid,
        xegpu=xegpu or {},
    )


def _unary(name: str, fn: Callable[..., object]) -> Problem:
    # The grid runs the column tiles first (x), the rows second.
    grid = tuple(
        size // block
        for size, block in zip(_ELEMENTWISE_SHAPE, ELEMENTWISE_BLOCKS, strict=True)
    )[::-1]
    shapes = (_ELEMENTWISE_SHAPE,)
    return Problem(name, "elementwise", fn, shapes, BF16, ELEMENTWISE_BLOCKS, grid)


def _rows(name: str, fn: Callable[..., object], shape: tuple[int, int]) -> Problem:
    grid = (shape[0] // _REDUCTION_BLOCKS[0],)
    return Problem(name, "reduction", fn, (shape,), F32, _REDUCTION_BLOCKS, grid)


PROBLEMS: tuple[Problem, ...] = (
    _gemm(
        "1_Square_matrix_multiplication_",
        (4096, 4096, 4096),
        blocks=GEMM_SQUARE[0],
        xegpu=GEMM_SQUARE[1],
    ),
    _gemm(
        "2_Standard_matrix_multiplication_",
        (2048, 4096, 8192),
        blocks=GEMM_STANDARD[0],
        xegpu=GEMM_STANDARD[1],
    ),
    _gemm(
        "6_Matmul_with_large_K_dimension_", (256, 256, 524288), blocks=[128, 128, 32]
    ),
    _gemm(
        "7_Matmul_with_small_K_dimension_",
        (32768, 32768, 64),
        blocks=GEMM_SMALL_K[0],
        xegpu=GEMM_SMALL_K[1],
        columns_first=True,
    ),
    # K = 32 zero-padded to 64 by the model (AI-bench's pad_k).
    _gemm(
        "9_Tall_skinny_matrix_multiplication_",
        (32768, 32768, 64),
        blocks=GEMM_SMALL_K[0],
        xegpu=GEMM_SMALL_K[1],
        columns_first=True,
    ),
    _gemm(
        "13_Matmul_for_symmetric_matrices",
        (4096, 4096, 4096),
        blocks=GEMM_SQUARE[0],
        xegpu=GEMM_SQUARE[1],
    ),
    _gemm("16_Matmul_with_transposed_A", (2048, 4096, 8192), "a"),
    _gemm("17_Matmul_with_transposed_B", (2048, 4096, 8192), "b"),
    _gemm("18_Matmul_with_transposed_both", (2048, 4096, 8192), "ab"),
    _unary("19_ReLU", _relu),
    _unary("20_LeakyReLU", _leaky_relu),
    _unary("21_Sigmoid", _sigmoid),
    _unary("22_Tanh", _tanh),
    _unary("25_Swish", _swish),
    _unary("27_SELU_", _selu),
    _unary("28_HardSigmoid", _hardsigmoid),
    _unary("29_Softplus", _softplus),
    _unary("30_Softsign", _softsign),
    _unary("31_ELU", _elu),
    _unary("32_HardTanh", _hardtanh),
    _rows("23_Softmax", _softmax, (4096, 393216)),
    _rows("24_LogSoftmax", _log_softmax, (4096, 393216)),
    _rows("38_L1Norm_", _l1_norm, (32768, 65536)),
    _rows("39_L2Norm_", _l2_norm, (32768, 65536)),
)
"""The problems, in KernelBench order by class (``(M, N, K)`` for matmuls)."""

RANK3_PROBLEMS: tuple[Problem, ...] = (
    # bench-gpu shapes (no mlir-gpu-bench variant), bf16 for the spec's f16 (the
    # same DPAS). Per batch the matrix is small (512x1024: lighthouse alone
    # picks 128x128 workgroups to fill the GPU), but 128 batches give 1024
    # workgroups, so the 2D GEMMs' 256x256x32 tile (32 subgroups of 32x64)
    # keeps its operand reuse.
    Problem(
        "3_Batched_matrix_multiplication",
        "matmul",
        _batched_matmul,
        ((128, 512, 2048), (128, 2048, 1024)),
        BF16,
        [1, *_GEMM_BLOCKS],
        (128, 2, 4),
    ),
    # bench-gpu shapes, bf16: 1024x768 per batch, L = 768 = 3 x 256 (192
    # workgroups over 16 batches).
    Problem(
        "10_3D_tensor_matrix_multiplication",
        "matmul",
        _tensor_matmul,
        ((16, 1024, 2048), (2048, 768)),
        BF16,
        [1, *_GEMM_BLOCKS],
        (16, 4, 3),
    ),
    # 19_ReLU over an NCHW activation (no 4D level 1 spec): 16x64 images of
    # 256x256, 2D 128x256 tiles (_NCHW_BLOCKS). W = 256 is one tile, so the
    # grid is 3D (a launch has three grid dims; a fourth becomes a loop in each
    # workgroup), innermost first: H tiles, C, N.
    Problem(
        "19_ReLU_NCHW",
        "elementwise",
        _relu,
        ((16, 64, 256, 256),),
        BF16,
        _NCHW_BLOCKS,
        (2, 64, 16),
    ),
    # 23_Softmax over the attention scores of 97_ScaledDotProductAttention's
    # mlir-gpu-bench shapes (B=32, H=32, S=512: [32, 32, 512, 512], f32): the 2D
    # softmax's 64 rows x 64-column chunks (the column block comes first).
    Problem(
        "23_Softmax_last_dim_4D",
        "reduction",
        _softmax_last_dim,
        ((32, 32, 512, 512),),
        F32,
        [_REDUCTION_BLOCKS[1], 1, 1, _REDUCTION_BLOCKS[0]],
        (32, 32, 8),
    ),
    # 97_ScaledDotProductAttention at mlir-gpu-bench shapes, written as flash
    # attention (AI-bench lowers unfused SDPA; lighthouse fuses it in its WG
    # tiling, which Helion's grid replaces).
    Problem(
        "97_ScaledDotProductAttention",
        "attention",
        attention,
        ((32, 32, 512, 64),) * 3,
        BF16,
        ATTENTION_BLOCKS,
        (32, 32, 4),
    ),
)
"""Rank-3+ problems: block size 1 on the leading dims (folded away)."""


LINEAR_BLOCKS = [256, 256, 32]
"""Level 2 GEMM block sizes ``[M tile, N tile, K tile]``, picked from the hardware
(Xe2: DPAS 8x16x16 bf16 -> f32, 16-lane subgroups, at most 32 subgroups per
workgroup with the large register file, 2D block reads of at most 32 rows x 64
bytes) and lighthouse's GEMM parameters for these shapes (its selector's
workgroup and subgroup tiles; the K step and A loads of AI-bench's tuned
``kb_params_level2-*.json`` for the lighthouse path):

- 256x256 workgroups of 32 subgroups (inferred ``[8, 4]`` subgroups of 32x64:
  4x4 DPAS blocks). Per subgroup the f32 accumulator takes 128 of the 256
  64-byte registers, a K step's A (32x32) and B (32x64) bf16 tiles 32 + 64:
  the large register file is needed, and 32 subgroups of it fill one Xe core
  (8 vector engines x 4 threads). Per K step a workgroup reads 2x256x32 bf16
  for 256x256x32 MACs: 128 flops per byte. The grid of 1024x8192 outputs is
  4x32 (128 workgroups: 4 waves on B70's 32 Xe cores), of 16384x4096 64x16.
- K steps of 32: two DPAS K steps per load, and an A tile of 32x32 bf16 (64
  bytes wide) per subgroup, read with one 2D block read (``xegpu_load_tiles``
  ``[32, 32, ...]``, as lighthouse's tuned parameters; inferred would be eight
  8x16 reads). B (the weight, stored ``[N, K]``) is read transposed in 16x16
  tiles (32-bit pairs: the transposed block read takes 32-bit elements, 8
  wide), lighthouse's choice.
- The epilogue runs once on the subgroup's 32x64 f32 accumulator in registers
  (small next to the 8192/32 = 256 K steps, but its temporaries share the 32
  registers the K loop leaves); the bias row is a gather of the subgroup's 64
  columns (4 elements per lane), once per workgroup.
"""
LINEAR_SKINNY_BLOCKS = [128, 256, 32]
"""59's ``M = 128`` (128 x 32768 x 32768): one 128-row workgroup tile, so the
weight (2 GiB of bf16, the dominant traffic at 128 flops per byte) is read
exactly once over the 1x128 grid (128 workgroups). 32 subgroups of 16x64
(inferred ``[8, 4]``, as lighthouse's selector and tuned parameters), A tiles
16x32 in one 2D block read (the tuned parameters read 8x16)."""
_A_LOAD = {"xegpu_load_tiles": [[32, 32, 16, 16]]}
_SKINNY_A_LOAD = {"xegpu_load_tiles": [[16, 32, 16, 16]]}


def _linear(
    name: str,
    fn: Callable[..., object],
    mnk: tuple[int, int, int],
    scalars: tuple[float, ...] = (),
) -> Problem:
    """``fn`` of ``x: [M, K]``, ``weight: [N, K]`` and ``bias: [N]``."""
    m, n, k = mnk
    blocks, xegpu = (
        (LINEAR_SKINNY_BLOCKS, _SKINNY_A_LOAD) if m < 256 else (LINEAR_BLOCKS, _A_LOAD)
    )
    shapes = ((m, k), (n, k), (n,))
    # A dim of one tile is no grid dim (59: 1x128 launches 128 workgroups).
    grid = tuple(tiles for tiles in (m // blocks[0], n // blocks[1]) if tiles > 1)
    return Problem(name, "linear", fn, shapes, BF16, blocks, grid, scalars, xegpu)


_LINEAR_MNK = (1024, 8192, 8192)

LEVEL2_PROBLEMS: tuple[Problem, ...] = (
    _linear(
        "9_Matmul_Subtract_Multiply_ReLU",
        _linear_subtract_multiply_relu,
        _LINEAR_MNK,
        (2.0, 1.5),
    ),
    _linear(
        "12_Gemm_Multiply_LeakyReLU",
        _linear_multiply_leaky_relu,
        _LINEAR_MNK,
        (2.0, 0.1),
    ),
    _linear("29_Matmul_Mish_Mish", _linear_mish_mish, _LINEAR_MNK),
    _linear(
        "40_Matmul_Scaling_ResidualAdd",
        _linear_scaling_residual_add,
        (16384, 4096, 4096),
        (0.5,),
    ),
    _linear(
        "59_Matmul_Swish_Scaling",
        _linear_swish_scaling,
        (128, 32768, 32768),
        (2.0,),
    ),
    _linear("63_Gemm_ReLU_Divide", _linear_relu_divide, _LINEAR_MNK, (2.0,)),
    _linear(
        "70_Gemm_Sigmoid_Scaling_ResidualAdd",
        _linear_sigmoid_scaling_residual_add,
        _LINEAR_MNK,
        (2.0,),
    ),
    _linear(
        "81_Gemm_Swish_Divide_Clamp_Tanh_Clamp",
        _linear_swish_divide_clamp_tanh_clamp,
        _LINEAR_MNK,
    ),
)
"""KernelBench level 2 problems with an ``mlir-gpu-bench`` variant: ``nn.Linear``
(``(M, N, K)`` = batch, out, in features) and its epilogue, the spec's scalar
inits as ``hl.constexpr``. bf16 like the variant (also the weight and bias)."""
