"""Regressions of contractions the CPU opt pipeline got wrong or rejected:
operand ops fused into a contraction that raising it to ``linalg.contract``
dropped (``split_contraction_operands``), and dot products of a 1x1 output
tile vectorized to an invalid ``vector.contract`` (``vectorize_linalg``).
Also attention whose head dim is not a power of two, which Helion's
power-of-two factory padding broke (``test_slice_sizes.py``)."""

from __future__ import annotations

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

import helion_mlir_backend  # noqa: F401

# Copied from AI-bench's
# backends/helion_mlir/cpu/KernelBench/level1/97_ScaledDotProductAttention.py
# (backend tests do not import AI-bench).

# Growth of a row maximum (in scaled logits) below which the stale maximum is
# kept: exponentials stay below e^8, well within f32 and bf16 range.
_RESCALE_THRESHOLD = 8.0


def _dot_nt(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b.T`` of ``[M, K]`` and ``[N, K]``; bf16 operands in AMX's VNNI
    layouts, ``[M, K/2, 2]`` and ``[K/2, N, 2]``."""
    if not vnni:
        return torch.einsum("mk,nk->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0), b.size(1) // 2, 2).permute(1, 0, 2)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _dot_nn(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``, VNNI layouts as in ``_dot_nt``."""
    if not vnni:
        return torch.einsum("mk,kn->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0) // 2, 2, b.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    vnni: hl.constexpr,
) -> torch.Tensor:
    """Softmax attention of ``[BH, S, D]`` heads with logits scaled by ``scale``,
    online over key blocks: the running row maxima rescale the sums and the
    output accumulator, only when one of them grows past ``_RESCALE_THRESHOLD``."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty_like(q)
    for h in hl.grid(heads):
        for tile_m in hl.tile(seq):
            q_tile = q[h, tile_m, :]
            # Finite: Helion's value range analysis of the growth takes -inf - -inf.
            top = hl.full([tile_m], -1.0e30, dtype=torch.float32)
            total = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, dim], dtype=torch.float32)
            for tile_n in hl.tile(seq):
                scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                scores = scores + _dot_nt(q_tile, k[h, tile_n, :], vnni)
                new_top = torch.maximum(top, torch.amax(scores, dim=-1) * scale)
                if torch.amax(new_top - top) > _RESCALE_THRESHOLD:
                    alpha = torch.exp(top - new_top)
                    total = total * alpha
                    acc = acc * alpha[:, None]
                    top = new_top
                p = torch.exp(scores * scale - top[:, None])
                total = total + p.sum(-1)
                acc = acc + _dot_nn(p.to(q.dtype), v[h, tile_n, :], vnni)
            out[h, tile_m, :] = (acc / total[:, None]).to(q.dtype)
    return out


def _attention_kernel(block_m: int, block_n: int) -> helion.Kernel:
    return helion.kernel(
        _attention,
        backend="mlir",
        config=helion.Config(block_sizes=[block_m, block_n]),
    )


def _normalized_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _check_attention(
    heads: int,
    seq: int,
    dim: int,
    dtype: torch.dtype,
    blocks: tuple[int, int],
    *,
    vnni: bool = False,
) -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(heads, seq, dim).to(dtype) for _ in range(3))
    out = _attention_kernel(*blocks)(
        q, k, v, hl.constexpr(dim**-0.5), hl.constexpr(vnni)
    )
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert _normalized_error(out, ref) < tolerance


def test_attention_bf16_without_vnni() -> None:
    """The ``p.to(bf16)`` rounding fused into the P@V contraction was dropped
    when raised, leaving an f32 x bf16 contraction that failed to vectorize.
    AI-bench's 97 at B=1, H=2, S=64, D=32 with ``vnni=False``."""
    _check_attention(2, 64, 32, torch.bfloat16, (64, 64))


def test_attention_f32_sequence_length_1() -> None:
    """1x1 score tiles: the QK^T contraction folds to a dot product, which
    vectorized to a ``vector.contract`` of a ``vector<f32>`` accumulator.
    AI-bench's 97 at B=1, H=2, S=1, D=32 in f32 (blocks 1, 1)."""
    _check_attention(2, 1, 32, torch.float32, (1, 1))


def test_attention_bf16_vnni_head_dim_40() -> None:
    """The ``acc`` factory padded D=40 to 64, so ``q_tile.size(1) // 2`` in
    the VNNI reshape was 32: ``aten.view`` of a 64x40 tile to [64, 32, 2]
    failed. 97's VNNI kernel at heads=2, S=64, D=40 unpadded (blocks 64, 64)."""
    _check_attention(2, 64, 40, torch.bfloat16, (64, 64), vnni=True)


def test_attention_f32_head_dim_7() -> None:
    """``acc`` padded D=7 to 8, which failed to broadcast with the 7 wide
    P@V product in Helion's tracing. AI-bench's 97 at B=1, H=2, S=16, D=7
    in f32 (blocks 16, 16)."""
    _check_attention(2, 16, 7, torch.float32, (16, 16))


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 16]))
def exp_bmm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    h, m, k = a.shape
    n = b.size(2)
    hl.specialize(k)
    out = torch.empty([h, m, n], dtype=torch.float32, device=a.device)
    for g in hl.grid(h):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            acc = acc + torch.einsum("mk,kn->mn", torch.exp(a[g, tm, :]), b[g, :, tn])
            out[g, tm, tn] = acc
    return out


def test_exp_fused_into_batched_contraction() -> None:
    """The exp fused into the contraction of 3-D slices was dropped when
    raised: the kernel silently computed ``a @ b``."""
    torch.manual_seed(0)
    a, b = torch.randn(2, 32, 16), torch.randn(2, 16, 32)
    torch.testing.assert_close(
        exp_bmm_kernel(a, b), torch.exp(a) @ b, atol=1e-4, rtol=1e-4
    )


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[16, 16, 16])
)
def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


def test_dot_of_a_1x1_output_tile() -> None:
    """A 1x16 by 16x1 product: one 1x1 output tile, folded to a dot product."""
    torch.manual_seed(0)
    x, y = torch.randn(1, 16), torch.randn(16, 1)
    torch.testing.assert_close(dot_kernel(x, y), x @ y, atol=1e-5, rtol=1e-5)
