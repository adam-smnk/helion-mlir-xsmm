"""Kernels of one tile on the XPU: a shape that fits one workgroup is one GPU
kernel with a grid of one, not host code touching device memory. A grid with
several tiles folds its unit dims as before."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor

BF16 = torch.bfloat16
F32 = torch.float32


def _kernel(fn: object, blocks: list[int], **config: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, **config),
    )(fn)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float32) * 2.0 + 1.0).to(x.dtype)
    return out


def row_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        top = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tm, tn].amax(-1))
        out[tm] = top
    return out


def softmax(x: torch.Tensor) -> torch.Tensor:
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


CASES = {
    "gemm_128": (gemm, [(128, 128), (128, 128)], BF16, [128, 128, 32]),
    "gemm_64": (gemm, [(64, 64), (64, 64)], BF16, [64, 64, 32]),
    "scale_128_256": (scale, [(128, 256)], F32, [128, 256]),
    "scale_16": (scale, [(16, 16)], F32, [16, 16]),
    "row_max": (row_max, [(64, 256)], F32, [64, 64]),
    "softmax": (softmax, [(64, 256)], F32, [64, 64]),
}


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(CASES))
def test_one_tile_is_one_gpu_kernel(name: str) -> None:
    fn, shapes, dtype, blocks = CASES[name]
    args = [xpu_tensor(*shape, dtype=dtype) for shape in shapes]
    text = lower_xpu(_kernel(fn, blocks), args, "xevm")
    assert text.count("gpu.launch_func") == 1


@pytest.mark.isolated
def test_several_tiles_fold_unit_dims() -> None:
    """A grid of (1, 2) stays a grid of 2 in x: no mapping, no moved dims."""
    args = [xpu_tensor(256, 64, dtype=BF16), xpu_tensor(64, 512, dtype=BF16)]
    text = lower_xpu(_kernel(gemm, [256, 256, 32]), args)
    assert "blocks in (%c2, %c1, %c1)" in text
    assert "mapping" not in text
