"""Kernels lighthouse's layout annotation has no form for, that layouts derived
from the IR lay out (``xegpu_layouts="ir"``): sibling GEMMs, a GEMM next to a row
reduction, attention of other head dims and query rows, GEMMs of few rows, chained
GEMMs in registers and through a buffer."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import TYPE_CHECKING

import helion.language as hl
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

F32 = torch.float32
BF16 = torch.bfloat16


def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(x @ gate^T) * (x @ up^T)``: sibling GEMMs sharing ``x``."""
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=F32)
        u = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            g = torch.addmm(g, x[tm, tk], gate[tn, tk].T)
            u = torch.addmm(u, x[tm, tk], up[tn, tk].T)
        out[tm, tn] = (g / (1.0 + torch.exp(-g)) * u).to(x.dtype)
    return out


def swiglu_ref(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(x @ gate.T) * (x @ up.T)


def gemm_row_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w^T``, each row divided by its L1 norm: a row reduction after a GEMM."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        out[tm, :] = (acc / torch.abs(acc).sum(-1, keepdim=True)).to(x.dtype)
    return out


def gemm_row_norm_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    acc = x @ w.T
    return acc / acc.abs().sum(-1, keepdim=True)


def gemm_softmax(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``softmax(x @ w^T, -1)``, a maximum and a sum after a GEMM."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        e = torch.exp(acc - acc.amax(-1, keepdim=True))
        out[tm, :] = (e / e.sum(-1, keepdim=True)).to(x.dtype)
    return out


def gemm_softmax_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x @ w.T, -1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flash attention of ``[H, S, D]`` (``kernelbench_xpu.attention``), on any
    head dim and V dim."""
    heads, m, d = q.shape
    n = k.size(1)
    d_value = v.size(2)
    scale = 1.0 / math.sqrt(d)
    out = torch.empty([heads, m, d_value], dtype=q.dtype, device=q.device)
    for th, tm in hl.tile([heads, m]):
        # The running maximum's start: see ``kernelbench_xpu.attention``.
        row_max = hl.full([th, tm], -3.4028234663852886e38, dtype=F32)
        row_sum = hl.zeros([th, tm], dtype=F32)
        acc = hl.zeros([th, tm, d_value], dtype=F32)
        for tn in hl.tile(n):
            scores = hl.zeros([th, tm, tn], dtype=F32)
            scores = scores + torch.einsum("hmd,hnd->hmn", q[th, tm, :], k[th, tn, :])
            scores = scores * scale
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + torch.einsum(
                "hmn,hnd->hmd", p.to(v.dtype), v[th, tn, :]
            )
            row_max = new_max
        out[th, tm, :] = (acc / row_sum[:, :, None]).to(out.dtype)
    return out


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    return torch.softmax(scores, -1) @ v


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w^T``."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        out[tm, tn] = acc.to(x.dtype)
    return out


def linear_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w.T


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``: a B that is not transposed."""
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a @ b


def elementwise_plus_transposed(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        out[tm, tn] = (a[tm, tn].to(F32) + z[tn, tm].T.to(F32)).to(a.dtype)
    return out


def tile_plus_its_transpose(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        t = a[tm, tn].to(F32)
        out[tm, tn] = (t + t.T).to(a.dtype)
    return out


def transposed_both(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x.T @ w.T`` of ``x: [K, M]``, ``w: [N, K]``."""
    k, m = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tk, tm].T, w[tn, tk].T)
        out[tm, tn] = acc.to(x.dtype)
    return out


def transposed_both_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.T @ w.T


def gemm_plus_transposed(
    x: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``x @ w^T + z^T``: a transposed read that is no DPAS operand."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        out[tm, tn] = (acc + z[tn, tm].T.to(F32)).to(x.dtype)
    return out


def gemm_plus_transposed_ref(
    x: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    return x @ w.T + z.T


def gemm_norm_over_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``acc / acc.abs().sum(-1).sum(0)``: a reduction over the rows as well."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        out[tm, :] = (acc / acc.abs().sum(-1).sum(0)).to(x.dtype)
    return out


def chain_in_registers(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """``x @ w1^T @ w2``, the intermediate rounded to bf16 in registers, stepping
    over ``w1``'s rows: both DPAS in one loop."""
    m = x.size(0)
    n = w1.size(0)
    o = w2.size(1)
    out = torch.empty([m, o], dtype=x.dtype, device=x.device)
    for tm, to in hl.tile([m, o]):
        acc = hl.zeros([tm, to], dtype=F32)
        for tn in hl.tile(n):
            mid = hl.zeros([tm, tn], dtype=F32)
            mid = mid + torch.einsum("md,nd->mn", x[tm, :], w1[tn, :])
            acc = acc + torch.einsum("mn,no->mo", mid.to(x.dtype), w2[tn, to])
        out[tm, to] = acc.to(x.dtype)
    return out


def chain_through_buffer(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """:func:`chain_in_registers` with the intermediate stored and loaded back."""
    m = x.size(0)
    n = w1.size(0)
    o = w2.size(1)
    mid = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, o], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, o], dtype=F32)
        for tn in hl.tile(n):
            scores = hl.zeros([tm, tn], dtype=F32)
            scores = scores + torch.einsum("md,nd->mn", x[tm, :], w1[tn, :])
            mid[tm, tn] = scores.to(x.dtype)
            acc = acc + torch.einsum("mn,no->mo", mid[tm, tn], w2[tn, :])
        out[tm, :] = acc.to(x.dtype)
    return out


def chain_ref(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return (x @ w1.T) @ w2


def scaled(x: torch.Tensor) -> torch.Tensor:
    """An elementwise kernel with 2D tiles."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(list(x.size())):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def scaled_ref(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


def row_sums(x: torch.Tensor) -> torch.Tensor:
    """A reduction kernel: each row less its sum."""
    m, n = x.size()
    out = torch.empty_like(x)
    block = hl.register_block_size(n)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, block], dtype=F32)
        for tn in hl.tile(n, block_size=block):
            acc = acc + x[tm, tn].to(F32)
        total = acc.sum(-1, keepdim=True)
        for tn in hl.tile(n, block_size=block):
            out[tm, tn] = (x[tm, tn].to(F32) - total).to(x.dtype)
    return out


def row_sums_ref(x: torch.Tensor) -> torch.Tensor:
    return x - x.sum(-1, keepdim=True)


@dataclass(frozen=True)
class Form:
    fn: Callable[..., torch.Tensor]
    ref: Callable[..., torch.Tensor]
    blocks: list[int]
    shapes: tuple[tuple[int, ...], ...]
    refusal: str
    """What lighthouse's path says, on a new form."""
    sg_layout: tuple[int, int]
    """Of the output tile under the rule."""
    extra: dict[str, object] = field(default_factory=dict)


_ROW_REDUCTION = "a matmul and a row reduction in one grid loop"

FORMS: dict[str, Form] = {
    "sibling_gemms_m64": Form(
        swiglu,
        swiglu_ref,
        [64, 256, 32],
        ((64, 2048), (1024, 2048), (1024, 2048)),
        "",
        (2, 8),
    ),
    "sibling_gemms_m2048": Form(
        swiglu,
        swiglu_ref,
        [256, 256, 32],
        ((1024, 2048), (1024, 2048), (1024, 2048)),
        "",
        (8, 4),
    ),
    "row_norm_n256": Form(
        gemm_row_norm,
        gemm_row_norm_ref,
        [128, 32],
        ((512, 2048), (256, 2048)),
        _ROW_REDUCTION,
        (8, 1),
    ),
    "row_norm_n64": Form(
        gemm_row_norm,
        gemm_row_norm_ref,
        [128, 32],
        ((512, 2048), (64, 2048)),
        _ROW_REDUCTION,
        (8, 1),
    ),
    "softmax_n64": Form(
        gemm_softmax,
        gemm_softmax_ref,
        [128, 32],
        ((512, 2048), (64, 2048)),
        _ROW_REDUCTION,
        (8, 1),
    ),
    "attention_head32": Form(
        attention,
        attention_ref,
        [1, 512, 64],
        ((32, 512, 32), (32, 512, 32), (32, 512, 32)),
        "attention with head dim 32",
        (32, 1),
    ),
    "attention_head128": Form(
        attention,
        attention_ref,
        [1, 512, 64],
        ((32, 512, 128), (32, 512, 128), (32, 512, 128)),
        "attention with head dim 128",
        (32, 1),
    ),
    "attention_value32": Form(
        attention,
        attention_ref,
        [1, 512, 64],
        ((32, 512, 64), (32, 512, 64), (32, 512, 32)),
        "attention with head dim 64 .V: 32",
        (32, 1),
    ),
    "attention_rows4": Form(
        attention,
        attention_ref,
        [1, 4, 64],
        ((64, 4, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (1, 1),
    ),
    "attention_rows8": Form(
        attention,
        attention_ref,
        [1, 8, 64],
        ((64, 8, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (1, 1),
    ),
    "attention_rows24": Form(
        attention,
        attention_ref,
        [1, 24, 64],
        ((64, 24, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (3, 1),
    ),
    "attention_rows2": Form(
        attention,
        attention_ref,
        [1, 2, 64],
        ((64, 2, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (1, 1),
    ),
    "attention_rows6": Form(
        attention,
        attention_ref,
        [1, 6, 64],
        ((64, 6, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (3, 1),
    ),
    "attention_rows12": Form(
        attention,
        attention_ref,
        [1, 12, 64],
        ((64, 12, 64), (64, 512, 64), (64, 512, 64)),
        "not a multiple of the 16-row subgroups",
        (3, 1),
    ),
    "transposed_both_16_rows": Form(
        transposed_both,
        transposed_both_ref,
        [32, 128, 32],
        ((256, 64), (256, 256)),
        "",
        (2, 8),
    ),
    "gemm_plus_transposed": Form(
        gemm_plus_transposed,
        gemm_plus_transposed_ref,
        [256, 256, 32],
        ((512, 256), (512, 256), (512, 512)),
        "",
        (8, 4),
    ),
    "gemm_m32": Form(
        linear,
        linear_ref,
        [32, 256, 32],
        ((32, 2048), (1024, 2048)),
        "did not return any valid configurations",
        (2, 16),
    ),
    "gemm_m8": Form(
        linear,
        linear_ref,
        [8, 256, 32],
        ((8, 2048), (1024, 2048)),
        "did not return any valid configurations",
        (1, 16),
    ),
    "chain_in_registers": Form(
        chain_in_registers,
        chain_ref,
        [64, 128, 32],
        ((256, 64), (256, 64), (256, 256)),
        "no row reduction",
        (4, 8),
    ),
    "chain_through_buffer": Form(
        chain_through_buffer,
        chain_ref,
        [64, 32],
        ((256, 64), (256, 64), (256, 256)),
        "",
        (8, 1),
    ),
}

ELEMENTWISE_FORM = Form(scaled, scaled_ref, [128, 256], ((512, 1024),), "", (4, 8))
REDUCTION_FORM = Form(row_sums, row_sums_ref, [64, 64], ((256, 1024),), "", (8, 1))
"""Not in :data:`FORMS`: lighthouse's path takes these. For the hints tests."""
GEMM_FORM = Form(
    linear, linear_ref, [256, 256, 32], ((512, 2048), (1024, 2048)), "", (8, 4)
)
PLAIN_B_FORM = Form(
    matmul, matmul_ref, [256, 256, 32], ((512, 2048), (2048, 1024)), "", (8, 4)
)
GEMM_FORM_48 = Form(
    linear, linear_ref, [384, 256, 32], ((768, 2048), (1024, 2048)), "", (8, 4)
)
