"""``fold_region_inserts``: a grid loop's store at a fixed index stores that slice
alone: one row of a K/V cache, not the whole region the iteration owns."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

BF16 = torch.bfloat16
F32 = torch.float32


def put_row(x: torch.Tensor, cache: torch.Tensor, pos: int) -> torch.Tensor:
    b, h, d = x.shape
    for tb, th, td in hl.tile([b, h, d]):
        cache[tb, th, pos, td] = x[tb, th, td].to(F32).to(x.dtype)
    return cache


def put_two_rows(x: torch.Tensor, cache: torch.Tensor, pos: int) -> torch.Tensor:
    """Two inserts into the region: not the exact pattern."""
    b, h, d = x.shape
    for tb, th, td in hl.tile([b, h, d]):
        cache[tb, th, pos, td] = x[tb, th, td].to(F32).to(x.dtype)
        cache[tb, th, pos + 1, td] = x[tb, th, td].to(F32).to(x.dtype)
    return cache


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[1, 1, 64]),
    )


def _bf(*shape: int) -> torch.Tensor:
    return xpu_tensor(*shape, dtype=BF16)


@pytest.mark.isolated
def test_row_store_is_one_row_tile() -> None:
    ir = lower_xpu(_kernel(put_row), [_bf(8, 8, 64), _bf(8, 8, 512, 64), 511])
    assert "vector<64xbf16>" in ir
    assert "512x64" not in ir.split("gpu.func")[1].split("gpu.return")[0].replace(
        "memref<8x8x512x64xbf16>", ""
    )


@pytest.mark.isolated
def test_row_store_lowers_to_xevm() -> None:
    ir = lower_xpu(
        _kernel(put_row), [_bf(8, 8, 64), _bf(8, 8, 512, 64), 511], stop_at="xevm"
    )
    assert "xegpu." not in ir


@pytest.mark.isolated
def test_two_inserts_are_left_alone() -> None:
    with pytest.raises(UnsupportedOperationError, match="4D tiles on XPU"):
        lower_xpu(_kernel(put_two_rows), [_bf(8, 8, 64), _bf(8, 8, 512, 64), 511])
