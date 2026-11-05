"""Kernels of the partial-tile tests: outputs are arguments, so that tests can
place them in guarded buffers. Each class's forms, with references in float64."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32


def scale(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = (x[tm, tn].to(torch.float32) * 2.0 + 1.0).to(x.dtype)
    return o


def scale_rank3(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tb, tm, tn in hl.tile(x.size()):
        o[tb, tm, tn] = (x[tb, tm, tn].to(torch.float32) * 2.0 + 1.0).to(x.dtype)
    return o


def gemm(a: torch.Tensor, b: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        o[tm, tn] = acc.to(a.dtype)
    return o


def gemm_transposed_b(
    a: torch.Tensor, b: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    n, _ = b.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
        o[tm, tn] = acc.to(a.dtype)
    return o


def gemm_tanh(a: torch.Tensor, b: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        o[tm, tn] = torch.tanh(acc).to(a.dtype)
    return o


def batched(a: torch.Tensor, b: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.size()
    _, _, n = b.size()
    for tb, tm, tn in hl.tile([batch, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tb, tm, tk], b[tb, tk, tn], acc=acc)
        o[tb, tm, tn] = acc.to(a.dtype)
    return o


def row_sum(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    for tm in hl.tile(m):
        total = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            total = total + x[tm, tn].sum(-1)
        o[tm] = total
    return o


def row_max(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    for tm in hl.tile(m):
        top = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tm, tn].amax(-1))
        o[tm] = top
    return o


def softmax_running(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """KernelBench's softmax form: running values over a partial tile keep their
    elements past its end (``loops._keep_past_tile``)."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
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
            o[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - row_top) * inv
    return o


def softmax_masked(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """The same, with its padded lanes masked by the author."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            s = torch.where(
                tile_n.index[None, :] < n,
                x[tile_m, tile_n],
                hl.full([tile_m, tile_n], float("-inf"), dtype=torch.float32),
            )
            top = torch.maximum(top, s)
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            s = torch.where(
                tile_n.index[None, :] < n,
                x[tile_m, tile_n],
                hl.full([tile_m, tile_n], float("-inf"), dtype=torch.float32),
            )
            total = total + torch.exp(s - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            o[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - row_top) * inv
    return o


def running_sum(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """A running value of raw tiles, then a sum."""
    m, n = x.size()
    block_n = hl.register_block_size(n)
    for tm in hl.tile(m):
        total = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            total = total + x[tm, tn]
        o[tm] = total.sum(-1)
    return o


def mask_passing_padding(
    x: torch.Tensor, m: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    """Padding (``m == 0``) passes ``m >= 0``: the running value must not take it."""
    r, c = x.shape
    block_n = hl.register_block_size(c)
    for tm in hl.tile(r):
        top = hl.full([tm, block_n], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(c, block_size=block_n):
            s = torch.where(
                m[tm, tn] >= 0.0,
                x[tm, tn],
                hl.full([tm, tn], float("-inf"), dtype=torch.float32),
            )
            top = torch.maximum(top, s)
        o[tm] = torch.amax(top, dim=-1)
    return o


def mask_on_rows(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """A mask of the rows only: the running value must not take padded columns."""
    r, c = x.shape
    block_n = hl.register_block_size(c)
    for tm in hl.tile(r):
        top = hl.full([tm, block_n], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(c, block_size=block_n):
            s = torch.where(
                tm.index[:, None] < r,
                x[tm, tn],
                hl.full([tm, tn], float("-inf"), dtype=torch.float32),
            )
            top = torch.maximum(top, s)
        o[tm] = torch.amax(top, dim=-1)
    return o


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    batch, heads, m, d = q.shape
    n = k.size(2)
    scale_d = 1.0 / math.sqrt(d)
    for tb, th, tm in hl.tile([batch, heads, m]):
        q_tile = q[tb, th, tm, :]
        top = hl.full([tb, th, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum_ = hl.zeros([tb, th, tm], dtype=torch.float32)
        acc = hl.zeros([tb, th, tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            scores = hl.zeros([tb, th, tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("bhmd,bhnd->bhmn", q_tile, k[tb, th, tn, :])
            scores = scores * scale_d
            new_max = torch.maximum(top, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, :, None])
            alpha = torch.exp(top - new_max)
            row_sum_ = row_sum_ * alpha + p.sum(-1)
            acc = acc * alpha[:, :, :, None] + torch.einsum(
                "bhmn,bhnd->bhmd", p.to(v.dtype), v[tb, th, tn, :]
            )
            top = new_max
        o[tb, th, tm, :] = (acc / row_sum_[:, :, :, None]).to(o.dtype)
    return o


def sibling_gemms(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            g = torch.addmm(g, x[tm, tk], gate[tn, tk].T)
            u = torch.addmm(u, x[tm, tk], up[tn, tk].T)
        o[tm, tn] = (g / (1.0 + torch.exp(-g)) * u).to(x.dtype)
    return o


def gemm_row_norm(x: torch.Tensor, w: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(0)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        o[tm, :] = (acc / torch.abs(acc).sum(-1, keepdim=True)).to(x.dtype)
    return o


def where_mask(x: torch.Tensor, m: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = torch.where(
            m[tm, tn] > 0.0, x[tm, tn], hl.full([tm, tn], -1.0, dtype=x.dtype)
        )
    return o


def load_mask(x: torch.Tensor, m: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = hl.load(x, [tm, tn], extra_mask=m[tm, tn] > 0.0)
    return o


def _d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").to(torch.float64)


@dataclass(frozen=True)
class Case:
    """``fn(*inputs, output)``: inputs ``(shape, dtype, kind)``, kind ``n`` random
    normal, ``neg`` all negative (a padded zero would win a maximum), ``mask`` a
    random +-1."""

    fn: object
    inputs: tuple
    output: tuple
    blocks: list[int]
    reference: object
    bound: float | None = None
    config: dict = field(default_factory=dict)

    def kernel(self, **extra: object) -> helion.Kernel:
        return helion.kernel(
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=self.blocks, **self.config, **extra),
        )(self.fn)


def _scale_ref(x: torch.Tensor) -> torch.Tensor:
    return _d(x) * 2.0 + 1.0


def _gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _d(a) @ _d(b)


def _gemm_t_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _d(a) @ _d(b).T


def _gemm_tanh_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.tanh(_d(a) @ _d(b))


def _sum_ref(x: torch.Tensor) -> torch.Tensor:
    return _d(x).sum(-1)


def _max_ref(x: torch.Tensor) -> torch.Tensor:
    return _d(x).amax(-1)


def running_rows(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """A running maximum over row chunks: padded rows of the last chunk stay out."""
    r, c = x.size()
    block_m = hl.register_block_size(r)
    for tn in hl.tile(c):
        top = hl.full([block_m, tn], float("-inf"), dtype=F32)
        for tm in hl.tile(r, block_size=block_m):
            top = torch.maximum(top, x[tm, tn])
        for tm in hl.tile(r, block_size=block_m):
            o[tm, tn] = top
    return o


def _running_rows_ref(x: torch.Tensor) -> torch.Tensor:
    x = _d(x)
    r, c = x.shape
    top = torch.full((64, c), -torch.inf, dtype=torch.float64)
    for i in range(0, r, 64):
        p = x[i : i + 64]
        top[: p.shape[0]] = torch.maximum(top[: p.shape[0]], p)
    o = torch.empty(r, c, dtype=torch.float64)
    for i in range(0, r, 64):
        o[i : i + 64] = top[: min(64, r - i)]
    return o


def _softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(_d(x), -1)


def _masked_max_ref(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    lowest = torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.where(_d(m) >= 0, _d(x), lowest).amax(-1)


def _attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(_d(q), _d(k), _d(v))


def _sibling_ref(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(_d(x) @ _d(gate).T) * (_d(x) @ _d(up).T)


def _row_norm_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    acc = _d(x) @ _d(w).T
    return acc / acc.abs().sum(-1, keepdim=True)


def _where_ref(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.where(_d(m) > 0, _d(x), torch.tensor(-1.0, dtype=torch.float64))


def _load_ref(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.where(_d(m) > 0, _d(x), torch.tensor(0.0, dtype=torch.float64))


GEMM_BLOCKS = [256, 256, 32]


CPU_ONLY = {
    "running_mask_on_rows": Case(
        mask_on_rows, (((128, 252), F32, "neg"),), ((128,), F32), [64, 64], _max_ref
    ),
}
"""Row and column index masks in one kernel assert in upstream's
``WgToSgConvertLayoutOp`` on the XPU, also with whole tiles (LLVM 79ab4054)."""


def _gemm_case(m: int, k: int, n: int, fn: object = gemm, **config: object) -> Case:
    ops = ((m, k), BF16, "n"), ((k, n), BF16, "n")
    ref = _gemm_tanh_ref if fn is gemm_tanh else _gemm_ref
    return Case(fn, ops, ((m, n), BF16), GEMM_BLOCKS, ref, None, config)


def _gemm_t_case(m: int, k: int, n: int) -> Case:
    ops = ((m, k), BF16, "n"), ((n, k), BF16, "n")
    return Case(gemm_transposed_b, ops, ((m, n), BF16), GEMM_BLOCKS, _gemm_t_ref)


def _attention_case(seq: int, kv: int, blocks: list[int]) -> Case:
    q = ((2, 2, seq, 64), BF16, "n")
    k = ((2, 2, kv, 64), BF16, "n")
    return Case(attention, (q, k, k), ((2, 2, seq, 64), BF16), blocks, _attention_ref)


def _elementwise(sh: tuple[int, ...], dtype: torch.dtype, blocks: list[int]) -> Case:
    return Case(scale, ((sh, dtype, "n"),), (sh, dtype), blocks, _scale_ref)


def _masked(fn: object, sh: tuple[int, int], dtype: torch.dtype, ref: object) -> Case:
    ops = (sh, dtype, "n"), (sh, F32, "mask")
    return Case(fn, ops, (sh, dtype), [64, 32], ref)


def _rows(fn: object, sh: tuple[int, int], ref: object) -> Case:
    return Case(fn, ((sh, F32, "neg"),), ((sh[0],), F32), [64, 64], ref)


def _softmax(sh: tuple[int, int], dtype: torch.dtype) -> Case:
    return Case(
        softmax_masked, ((sh, dtype, "neg"),), (sh, dtype), [64, 64], _softmax_ref
    )


CASES = {
    # Elementwise: partial rows, columns, both; f32 and bf16; one-row tiles at
    # any row size; leading unit dims.
    "scale_f32_rows": _elementwise((253, 512), F32, [64, 128]),
    "scale_f32_cols": _elementwise((256, 508), F32, [64, 128]),
    "scale_f32_both": _elementwise((253, 480), F32, [64, 128]),
    "scale_bf16_rows": _elementwise((240, 512), BF16, [64, 128]),
    "scale_bf16_cols": _elementwise((256, 504), BF16, [64, 128]),
    "scale_bf16_both": _elementwise((253, 480), BF16, [64, 128]),
    "scale_one_row_f32": _elementwise((64, 16347), F32, [1, 8192]),
    "scale_one_row_bf16": _elementwise((64, 16347), BF16, [1, 8192]),
    "scale_rank3": Case(
        scale_rank3,
        (((3, 253, 480), BF16, "n"),),
        ((3, 253, 480), BF16),
        [1, 64, 128],
        _scale_ref,
    ),
    # Plane bases 48 bytes off 64: buffer_alignment calls that undefined.
    "scale_rank3_planes": Case(
        scale_rank3,
        (((3, 253, 248), BF16, "n"),),
        ((3, 253, 248), BF16),
        [1, 64, 128],
        _scale_ref,
    ),
    # GEMMs: M, N, K, all; transposed B; an epilogue; batched; small.
    "gemm_m": _gemm_case(509, 256, 512),
    "gemm_n": _gemm_case(512, 256, 480),
    "gemm_k": _gemm_case(512, 240, 512),
    "gemm_all": _gemm_case(509, 240, 480),
    "gemm_t_m": _gemm_t_case(509, 256, 512),
    "gemm_t_n": _gemm_t_case(512, 256, 480),
    "gemm_t_k": _gemm_t_case(512, 240, 512),
    "gemm_t_all": _gemm_t_case(509, 240, 480),
    "gemm_tanh_all": _gemm_case(509, 240, 480, gemm_tanh),
    "gemm_small": Case(
        gemm,
        (((120, 120), BF16, "n"), ((120, 120), BF16, "n")),
        ((120, 120), BF16),
        [64, 64, 32],
        _gemm_ref,
    ),
    "gemm_batched": Case(
        batched,
        (((3, 509, 240), BF16, "n"), ((3, 240, 480), BF16, "n")),
        ((3, 509, 480), BF16),
        [1, 256, 256, 32],
        _gemm_ref,
    ),
    # K short of a K tile, and K no multiple of 32 (rows of a multiple of 16 bytes).
    "gemm_k_40": _gemm_case(512, 40, 512),
    "gemm_k_56": _gemm_case(512, 56, 512),
    "gemm_k_248": _gemm_case(512, 248, 512),
    "gemm_ragged_m_n_k_small": _gemm_case(257, 40, 264),
    # A GEMM and a row norm, sibling GEMMs: forms of IR-derived layouts.
    "gemm_row_norm_m": Case(
        gemm_row_norm,
        (((509, 2048), BF16, "n"), ((256, 2048), BF16, "n")),
        ((509, 256), BF16),
        [128, 32],
        _row_norm_ref,
    ),
    "gemm_row_norm_k": Case(
        gemm_row_norm,
        (((512, 2040), BF16, "n"), ((256, 2040), BF16, "n")),
        ((512, 256), BF16),
        [128, 32],
        _row_norm_ref,
    ),
    "gemm_row_norm_both": Case(
        gemm_row_norm,
        (((509, 2040), BF16, "n"), ((256, 2040), BF16, "n")),
        ((509, 256), BF16),
        [128, 32],
        _row_norm_ref,
    ),
    "sibling_gemms_m": Case(
        sibling_gemms,
        (
            ((1021, 2048), BF16, "n"),
            ((1024, 2048), BF16, "n"),
            ((1024, 2048), BF16, "n"),
        ),
        ((1021, 1024), BF16),
        [256, 256, 32],
        _sibling_ref,
    ),
    "sibling_gemms_n": Case(
        sibling_gemms,
        (
            ((1024, 2048), BF16, "n"),
            ((1016, 2048), BF16, "n"),
            ((1016, 2048), BF16, "n"),
        ),
        ((1024, 1016), BF16),
        [256, 256, 32],
        _sibling_ref,
    ),
    "sibling_gemms_k": Case(
        sibling_gemms,
        (
            ((1024, 2040), BF16, "n"),
            ((1024, 2040), BF16, "n"),
            ((1024, 2040), BF16, "n"),
        ),
        ((1024, 1024), BF16),
        [256, 256, 32],
        _sibling_ref,
    ),
    # Row reductions.
    "row_sum_cols": _rows(row_sum, (128, 252), _sum_ref),
    "row_sum_rows": _rows(row_sum, (125, 256), _sum_ref),
    "row_max_cols": _rows(row_max, (128, 252), _max_ref),
    "row_max_rows": _rows(row_max, (125, 256), _max_ref),
    "row_max_both": _rows(row_max, (125, 252), _max_ref),
    "softmax_f32_rows": _softmax((125, 256), F32),
    "softmax_f32_cols": _softmax((128, 252), F32),
    "softmax_f32_both": _softmax((125, 252), F32),
    "softmax_bf16_cols": _softmax((128, 248), BF16),
    "softmax_bf16_both": _softmax((125, 248), BF16),
    "running_rows": Case(
        running_rows,
        (((252, 128), F32, "neg"),),
        ((252, 128), F32),
        [64, 64],
        _running_rows_ref,
    ),
    # Flash attention: partial query rows, K/V rows, both, short sequences.
    "attention_q": _attention_case(253, 256, [1, 1, 128, 64]),
    "attention_kv": _attention_case(256, 251, [1, 1, 128, 64]),
    "attention_both": _attention_case(253, 251, [1, 1, 128, 64]),
    "attention_96": _attention_case(96, 96, [1, 1, 128, 64]),
    "attention_97": _attention_case(97, 97, [1, 1, 64, 64]),
    "attention_64": _attention_case(64, 64, [1, 1, 32, 32]),
    # Value and memory masks on partial tiles.
    "where_cols": _masked(where_mask, (128, 248), BF16, _where_ref),
    "where_rows": _masked(where_mask, (125, 256), F32, _where_ref),
    "load_mask_cols": _masked(load_mask, (128, 252), F32, _load_ref),
    "load_mask_rows": _masked(load_mask, (125, 256), BF16, _load_ref),
}
"""Every class, ragged; rows of a multiple of 16 bytes where 2D."""

UNALIGNED = {
    "scale_f32": _elementwise((256, 505), F32, [64, 128]),
    "scale_bf16": _elementwise((256, 505), BF16, [64, 128]),
    "gemm_k": _gemm_case(512, 251, 512),
    "gemm_n": _gemm_case(512, 256, 507),
    "row_max": _rows(row_max, (128, 251), _max_ref),
}
"""Partial tiles on rows no 2D block access can take: refused."""

CASES.update(
    {
        "softmax_running_f32_cols": Case(
            softmax_running,
            (((128, 252), F32, "neg"),),
            ((128, 252), F32),
            [64, 64],
            _softmax_ref,
        ),
        "softmax_running_f32_both": Case(
            softmax_running,
            (((125, 252), F32, "neg"),),
            ((125, 252), F32),
            [64, 64],
            _softmax_ref,
        ),
        "softmax_running_bf16_both": Case(
            softmax_running,
            (((125, 248), BF16, "neg"),),
            ((125, 248), BF16),
            [64, 64],
            _softmax_ref,
        ),
        "running_sum_cols": Case(
            running_sum, (((128, 252), F32, "neg"),), ((128,), F32), [64, 64], _sum_ref
        ),
        "running_mask_passing_padding": Case(
            mask_passing_padding,
            (((128, 252), F32, "neg"), ((128, 252), F32, "mask")),
            ((128,), F32),
            [64, 64],
            _masked_max_ref,
        ),
    }
)
