"""``helion_mlir_utils`` prepacked GEMMs: the f32 rows kernel, the prepared
call that :func:`linear` caches, and the small-M VNNI panel choice."""

from __future__ import annotations

import importlib

from helion_mlir_utils.linear import linear
from helion_mlir_utils.matmul import identity_epilogue
import pytest
import torch

# The package exports functions named like these modules.
mm = importlib.import_module("helion_mlir_utils.matmul")


def _old_blocked(
    a: torch.Tensor, b4: torch.Tensor, n: int, bias: torch.Tensor | None, epilogue
) -> torch.Tensor:
    """The f32 path before the rows kernel: A packed per call, blocked kernel."""
    m, k = a.shape
    m_pad, k_pad = mm._round_up(m, mm.BLOCK_M), mm._round_up(k, mm.BLOCK_K)
    blocks_n = int(b4.shape[0])
    a4 = mm.pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    if bias is None:
        out4 = mm._matmul_blocked_kernel(a4, b4, epilogue)
    else:
        n_pad = blocks_n * mm.BLOCK_N
        bias3 = mm._pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(
            blocks_n, 1, mm.BLOCK_N
        )
        out4 = mm._matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    return out4.reshape(m_pad, blocks_n * mm.BLOCK_N)[:m, :n]


@pytest.mark.parametrize(
    ("m", "k", "n", "bias"),
    [
        (64, 256, 96, True),  # whole row tiles: one call
        (33, 256, 64, True),  # ragged M: bulk rows + a padded rest
        (130, 96, 70, False),  # ragged M and N
        (17, 64, 32, False),  # fewer rows than one tile: the rest only
        (40, 100, 64, True),  # K % 32 != 0: the blocked path
    ],
)
def test_f32_rows_kernel_matches_blocked_path(
    m: int, k: int, n: int, bias: bool
) -> None:
    torch.manual_seed(m + k + n)
    a = torch.randn(m, k)
    w = torch.randn(n, k)
    b = torch.randn(n) if bias else None
    b4 = mm.pack_b_blocked_t(w)
    out = mm.matmul_prepacked_b(a, b4, n, b, torch.relu)
    assert torch.equal(out, _old_blocked(a, b4, n, b, torch.relu))
    if k % mm.BLOCK_K == 0:
        assert mm._ROWS_KERNELS, "the rows kernel ran"
    expected = torch.relu(a @ w.T + (0 if b is None else b))
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_reuses_its_call_until_the_weight_changes(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(128, 64).to(dtype)
    x = torch.randn(32, 128, dtype=dtype)
    with torch.no_grad():
        y, cache = linear(x, layer, torch.relu)
        call = cache.call
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache = linear(x2, layer, torch.relu, cache=cache)
        assert cache.call is call
        assert torch.equal(
            y2,
            mm.matmul_prepacked_b(x2, cache.packed_weight, 64, cache.bias, torch.relu),
        )
        # Another epilogue: a new call on the same packed weight.
        _, cache2 = linear(x2, layer, identity_epilogue, cache=cache)
        assert cache2.call is not call
        assert cache2.packed_weight is cache.packed_weight
        # A weight mutated in place: repacked, and the output follows.
        layer.weight.mul_(-1.0)
        y3, cache3 = linear(x2, layer, torch.relu, cache=cache)
        assert cache3.packed_weight is not cache.packed_weight
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(
            y3.float(),
            torch.relu(x2.float() @ layer.weight.float().T + layer.bias.float()),
            atol=tol,
            rtol=tol,
        )


def test_linear_call_falls_back_for_other_activations() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(64, 64)
    with torch.no_grad():
        _, cache = linear(torch.randn(32, 64), layer, torch.relu)
        xt = torch.randn(64, 32).t()  # same shape, not contiguous
        y, _ = linear(xt, layer, torch.relu, cache=cache)
    torch.testing.assert_close(y, torch.relu(layer(xt)), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("case", ["bias_length", "bias_rank", "dtype"])
def test_prepared_call_raises_like_matmul_prepacked_b(case: str) -> None:
    a = torch.randn(32, 64)
    b4 = mm.pack_b_blocked_t(torch.randn(64, 64))
    bias = None
    if case == "bias_length":
        bias = torch.randn(32)
    elif case == "bias_rank":
        bias = torch.randn(64, 1)
    else:
        a = a.bfloat16()
    with pytest.raises(ValueError) as general:
        mm.matmul_prepacked_b(a, b4, 64, bias, torch.relu)
    with pytest.raises(ValueError) as prepared:
        mm.prepacked_b_call(a, b4, 64, bias, torch.relu)(a)
    assert str(prepared.value) == str(general.value)


def test_small_m_vnni_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    # 128 rows: 64-row x 128-column tiles (64 of them), not 128 x 64.
    panel = mm.vnni_panel(128, 4096)
    assert panel == 128
    assert mm._prepacked_tile_m(128, 4096 // panel, panel) == 64
    # The N = 2048 layer keeps 64-column panels (a tie: the narrower one).
    assert mm.vnni_panel(128, 2048) == 64
    # From 256 rows on, the rule for shared rows is unchanged.
    assert mm.vnni_panel(256, 4096) == 64
    assert mm.vnni_panel(1024, 8192) == 256
