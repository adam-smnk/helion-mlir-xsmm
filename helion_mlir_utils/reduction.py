"""Matrix-vector product (gemv) on Helion's MLIR backend.

A memory-bound row reduction: each tile of rows multiplies its K-wide slices
by the vector's and sums them, in f32. No matrix unit: a gemv reads every
element of A once, so its speed is A's bandwidth.
"""

from __future__ import annotations

import os
from typing import Callable

import helion
import helion.language as hl
import torch
from torch import Tensor

from .matmul import identity_epilogue


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _matvec_kernel(a: Tensor, x: Tensor) -> Tensor:
    """``a @ x`` of ``[M, K]`` and ``[K]``. Products accumulate per K lane, summed
    once at the end: a sum per K tile vectorizes across rows, reading A by
    columns."""
    m, k = a.shape
    block_k = hl.register_block_size(k)
    out = torch.empty([m], dtype=a.dtype, device=a.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_k], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = acc + a[tile_m, tile_k].float() * x[tile_k].float()[None, :]
        out[tile_m] = acc.sum(-1).to(a.dtype)
    return out


def supports_matvec(a: Tensor, b: Tensor) -> bool:
    """Whether :func:`matvec` can handle this pair of operands."""
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.bfloat16):
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2 or b.shape[1] != 1:
        return False
    return a.shape[1] == b.shape[0]


def matvec(a: Tensor, b: Tensor) -> Tensor:
    """``A @ b`` for ``A: [M, K]``, ``b: [K, 1]`` -> ``[M, 1]``.

    Raises ``ValueError`` if unsupported (see :func:`supports_matvec`) instead
    of silently falling back to eager PyTorch -- callers that need a fallback
    must check :func:`supports_matvec` themselves and choose one explicitly.
    """
    if not supports_matvec(a, b):
        raise ValueError(
            f"matvec() does not support these operands: a.shape={tuple(a.shape)} "
            f"a.dtype={a.dtype} a.device={a.device}, b.shape={tuple(b.shape)} "
            f"b.dtype={b.dtype} b.device={b.device}. b must be K-by-1, dtype must be "
            "float32 or bfloat16, both on cpu. Check supports_matvec() before "
            "calling, or use torch.matmul directly for unsupported cases."
        )
    m, k = a.shape
    return _matvec_kernel(a, b.reshape(k)).reshape(m, 1)


def _row_sum_kernel(x: Tensor, epilogue: Callable[[Tensor], Tensor]) -> Tensor:
    """``epilogue`` of the f32 row sums of ``[M, N]``, as ``[M, 1]``."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty((m, 1), dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].to(torch.float32)
        out[tile_m, :] = epilogue(acc.sum(-1, keepdim=True)).to(x.dtype)
    return out


_ROW_SUM_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def row_sum(
    x: Tensor, epilogue: Callable[[Tensor], Tensor] = identity_epilogue
) -> Tensor:
    """``epilogue(x.sum(1, keepdim=True))`` of a 2-D ``x``, summed in f32; rows
    split evenly over the threads."""
    if x.dim() != 2:
        raise ValueError(f"row_sum() expects a 2-D tensor, got {tuple(x.shape)}")
    m, n = map(int, x.shape)
    threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    # The column block is registered first.
    block_sizes = (min(n, 512), max(1, min(32, -(-m // threads))))
    if block_sizes not in _ROW_SUM_KERNELS:
        _ROW_SUM_KERNELS[block_sizes] = helion.kernel(
            _row_sum_kernel,
            backend="mlir",
            config=helion.Config(block_sizes=list(block_sizes)),
        )
    return _ROW_SUM_KERNELS[block_sizes](x.contiguous(), epilogue)
