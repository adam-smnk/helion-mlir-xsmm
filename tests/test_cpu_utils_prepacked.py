"""``helion_mlir_utils`` prepacked GEMMs: the f32 rows kernel, the prepared
call that :func:`linear` caches, and the small-M VNNI panel choice."""

from __future__ import annotations

import importlib
import os

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


@pytest.mark.parametrize("post_bias", [False, True])
def test_linear_affine_reuses_its_call(
    monkeypatch: pytest.MonkeyPatch, post_bias: bool
) -> None:
    """bf16 ``linear_affine`` (L2/33, L2/39) prepares its call once, like
    ``linear``: later calls skip ``matmul_prepacked_b``'s per-call staging and
    give the same bits."""
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(1)
    dtype = torch.bfloat16
    layer = torch.nn.Linear(128, 64).to(dtype)
    scale = torch.randn(64, dtype=dtype)
    shift = torch.randn(64, dtype=dtype) if post_bias else None
    x = torch.randn(32, 128, dtype=dtype)
    with torch.no_grad():
        y, cache = linear_affine(x, layer, scale, shift, torch.relu)
        assert cache.call is not None
        general = mm.matmul_prepacked_b(
            x, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
        )
        assert torch.equal(y, general)

        def no_general(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("the prepared call took the general path")

        monkeypatch.setattr(mm, "matmul_prepacked_b", no_general)
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache2 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache)
        assert cache2.call is cache.call
        monkeypatch.undo()
        assert torch.equal(
            y2,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
            ),
        )
        # Another epilogue, then another thread count: each rebuilds the
        # call, which stays bit-equal to the general path.
        y5, cache5 = linear_affine(
            x2, layer, scale, shift, identity_epilogue, cache=cache2
        )
        assert cache5.call is not cache2.call
        assert torch.equal(
            y5,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, identity_epilogue, scale, shift
            ),
        )
        threads = os.environ.get("OMP_NUM_THREADS")
        monkeypatch.setenv("OMP_NUM_THREADS", "3" if threads != "3" else "5")
        y6, cache6 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache5)
        assert cache6.call is not cache5.call
        assert cache6.call[1] == os.environ["OMP_NUM_THREADS"]
        assert torch.equal(
            y6,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
            ),
        )
        # Another activation layout takes the general path, same result.
        xt = torch.randn(128, 32, dtype=dtype).t()
        y3, _ = linear_affine(xt, layer, scale, shift, torch.relu, cache=cache)
        assert torch.equal(
            y3, linear_affine(xt.contiguous(), layer, scale, shift, torch.relu)[0]
        )
        # A scale mutated in place: a new cache, and the output follows.
        scale.mul_(-1.0)
        y4, cache4 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache)
        assert cache4.key != cache.key
        expected = x2.float() @ layer.weight.float().T + layer.bias.float()
        expected = expected.to(dtype).float() * scale.float()
        if shift is not None:
            expected = expected + shift.float()
        err = (y4.float() - torch.relu(expected)).abs().max() / expected.abs().max()
        assert err < 2e-2


def test_linear_affine_f32_keeps_the_general_path() -> None:
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(2)
    layer = torch.nn.Linear(64, 64)
    scale = torch.randn(64)
    x = torch.randn(32, 64)
    with torch.no_grad():
        y, cache = linear_affine(x, layer, scale, None, torch.relu)
        y2, _ = linear_affine(x, layer, scale, None, torch.relu, cache=cache)
    assert torch.equal(y, y2)
    torch.testing.assert_close(y, torch.relu(layer(x) * scale), atol=1e-4, rtol=1e-4)


def test_row_sum_binds_once_per_shape_and_epilogue() -> None:
    red = importlib.import_module("helion_mlir_utils.reduction")

    def half(t: torch.Tensor) -> torch.Tensor:
        return t * 0.5

    torch.manual_seed(3)
    x = torch.randn(40, 96, dtype=torch.bfloat16)
    red._ROW_SUM_BOUND.clear()
    out = red.row_sum(x, half)
    assert len(red._ROW_SUM_BOUND) == 1
    out2 = red.row_sum(x.clone(), half)
    assert len(red._ROW_SUM_BOUND) == 1
    assert torch.equal(out, out2)
    # The unbound kernel call (the previous path) gives the same bits.
    (block_sizes,) = {key[0] for key in red._ROW_SUM_BOUND}
    assert torch.equal(out, red._ROW_SUM_KERNELS[block_sizes](x, half))
    red.row_sum(x, identity_epilogue)
    red.row_sum(torch.randn(8, 96, dtype=torch.bfloat16), half)
    assert len(red._ROW_SUM_BOUND) == 3
    ref = x.float().sum(1, keepdim=True) * 0.5
    torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=1e-2)


def test_row_sum_cache_evicts_least_recently_used() -> None:
    """Past ``_ROW_SUM_BOUND_SIZE`` keys the least recently used bound kernel
    goes, a key in use stays, and evicted keys rebind with the right result."""
    red = importlib.import_module("helion_mlir_utils.reduction")
    size = red._ROW_SUM_BOUND_SIZE

    def scale_by(factor: float):
        def epilogue(t: torch.Tensor) -> torch.Tensor:
            return t * factor

        return epilogue

    torch.manual_seed(4)
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    epilogues = [scale_by(float(i + 1)) for i in range(size + 1)]
    red._ROW_SUM_BOUND.clear()
    outs = {}
    for i, epilogue in enumerate(epilogues):
        # Key 0, the oldest, is used again before every new key: the new key
        # past the size evicts key 1, not key 0 (as first-in first-out would).
        if i:
            red.row_sum(x, epilogues[0])
        outs[i] = red.row_sum(x, epilogue)
    assert len(red._ROW_SUM_BOUND) == size
    keys = [key[-1] for key in red._ROW_SUM_BOUND]
    assert epilogues[0] in keys
    assert epilogues[1] not in keys  # the least recently used
    ref = x.float().sum(1, keepdim=True)
    for i in (0, 1, 2, size):
        out = red.row_sum(x, epilogues[i])
        assert torch.equal(out, outs[i])
        torch.testing.assert_close(out.float(), ref * (i + 1.0), atol=0.5, rtol=2e-2)


def _native_extension() -> bool:
    from helion_mlir_backend._compiler import native

    return native.module() is not None


@pytest.mark.skipif(not _native_extension(), reason="no native extension")
@pytest.mark.parametrize(
    ("dtype", "affine"),
    [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_prepared_linear_calls_run_natively(dtype: torch.dtype, affine: bool) -> None:
    """The prepared call of ``linear`` / ``linear_affine`` is the backend's
    native prepared run: later calls with the same activation layout run
    natively, bit-equal to the general path; others take the Python call."""
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(2)
    layer = torch.nn.Linear(128, 64).to(dtype)
    scale = torch.randn(64, dtype=dtype)
    shift = torch.randn(64, dtype=dtype)

    def run(x: torch.Tensor, cache: object) -> tuple[torch.Tensor, object]:
        if affine:
            return linear_affine(x, layer, scale, shift, torch.relu, cache=cache)
        return linear(x, layer, torch.relu, cache=cache)

    def general(x: torch.Tensor, cache: object) -> torch.Tensor:
        extra = (scale, shift) if affine else ()
        return mm.matmul_prepacked_b(
            x, cache.packed_weight, 64, cache.bias, torch.relu, *extra
        )

    with torch.no_grad():
        x = torch.randn(32, 128, dtype=dtype)
        _, cache = run(x, None)
        prepared = cache.call[2]
        assert type(prepared).__name__ == "PreparedRun"
        _, cache = run(x, cache)  # (the kernel's first call records its plan)
        before = prepared.native_calls
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache = run(x2, cache)
        assert prepared.native_calls == before + 1
        assert y2.shape == (32, 64) and y2.is_contiguous()
        assert torch.equal(y2, general(x2, cache))
        # A non-contiguous activation of the same shape: the Python call.
        xt = torch.randn(128, 32, dtype=dtype).t()
        y3, cache = run(xt, cache)
        assert torch.equal(y3, general(xt.contiguous(), cache))


@pytest.mark.skipif(not _native_extension(), reason="no native extension")
def test_prepared_call_keeps_the_closure_for_padded_columns() -> None:
    # N = 70 is padded to whole panels: the result is a column slice, which
    # the native prepared run does not make.
    a = torch.randn(32, 64)
    b4 = mm.pack_b_blocked_t(torch.randn(70, 64))
    call = mm.prepacked_b_call(a, b4, 70, None, torch.relu)
    assert type(call).__name__ == "function"
    assert torch.equal(call(a), mm.matmul_prepacked_b(a, b4, 70, None, torch.relu))
