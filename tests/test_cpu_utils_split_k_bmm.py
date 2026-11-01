"""``helion_mlir_utils`` f32 GEMM paths: split-K for few output tiles, the
batched kernel of ``bmm`` (batch as a grid dim), and their gates."""

from __future__ import annotations

import importlib

import pytest
import torch

# The package exports functions named like its modules.
mm = importlib.import_module("helion_mlir_utils.matmul")


def _error(out: torch.Tensor, expected: torch.Tensor) -> float:
    return ((out.double() - expected).abs().max() / expected.abs().max()).item()


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[object]:
    """Calls of ``mm.<name>`` (still run), recorded by their return value."""
    calls: list[object] = []
    original = getattr(mm, name)

    def wrapper(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(mm, name, wrapper)
    return calls


@pytest.mark.parametrize(
    ("m", "n", "k", "split"),
    [
        (64, 128, 4096, True),  # one 64x128 tile: K split for the threads
        (96, 128, 4096, False),  # rows not whole 64-row tiles: one K loop
    ],
    ids=["split", "ragged_rows"],
)
def test_split_k_gate(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int, split: bool
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    calls = _spy(monkeypatch, "_matmul_split_k")
    a, b = torch.randn(m, k), torch.randn(k, n)
    out = mm.matmul(a, b)
    assert (len(calls) == 1) is split
    assert out.shape == (m, n)
    assert _error(out, a.double() @ b.double()) < 1e-5


def test_split_k_splits_fill_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """S x tiles fill the threads, K chunks stay at least 512 long, and only
    the first level of a nested ``OMP_NUM_THREADS`` list counts."""
    args = (False, False, None, None, None, mm.identity_epilogue)
    monkeypatch.setenv("OMP_NUM_THREADS", "64,8")
    assert mm._split_k(256, 256, 8192, *args) == 8
    assert mm._split_k(64, 128, 2048, *args) == 4
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert mm._split_k(64, 128, 4096, *args) == 2
    # Not split: transposed, an epilogue, or enough tiles already.
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert (
        mm._split_k(256, 256, 8192, True, False, None, None, None, mm.identity_epilogue)
        == 0
    )
    assert mm._split_k(256, 256, 8192, False, False, None, None, None, torch.relu) == 0
    assert mm._split_k(2048, 2048, 2048, *args) == 0


@pytest.mark.parametrize(
    ("shape", "direct"),
    [
        ((2, 160, 96, 224), True),  # 32-multiples: the batched kernel
        ((2, 65, 129, 129), False),  # no dividing tiles: per-slice fallback
    ],
    ids=["direct", "ragged_fallback"],
)
def test_bmm_f32_gate(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], direct: bool
) -> None:
    calls = _spy(monkeypatch, "_bmm_f32")
    batch, m, n, k = shape
    a, b = torch.randn(batch, m, k), torch.randn(batch, k, n)
    out = mm.bmm(a, b)
    assert len(calls) == 1
    assert (calls[0] is not None) is direct
    assert out.shape == (batch, m, n)
    assert _error(out, torch.bmm(a.double(), b.double())) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "b_shape", [(1, 64, 64), (3, 32, 64)], ids=["batch", "contracted_dim"]
)
def test_bmm_rejects_mismatched_operands(
    dtype: torch.dtype, b_shape: tuple[int, ...]
) -> None:
    """Raises like ``torch.bmm`` (it read past ``b`` and returned zeros)."""
    a = torch.randn(3, 64, 64, dtype=dtype)
    b = torch.randn(*b_shape, dtype=dtype)
    with pytest.raises(RuntimeError, match="batch2"):
        mm.bmm(a, b)
