"""Block-packing kernels for the Helion MLIR backend.

Repacks the operands of an ``[M, K] x [K, N]`` matmul into panel-major layouts
so that each cache tile the matmul consumes is contiguous:

* ``pack_b_panels``: ``[K, N]`` -> ``[N/BN, K, BN]``. Panel ``j`` holds the full
  ``K`` extent of ``BN`` columns, so the ``32 x 16`` sub-block that the AMX VNNI
  repack reads has a row stride of ``BN`` instead of ``N``.
* ``pack_a_panels``: ``[M, K]`` -> ``[K/BK, M, BK]``. Panel ``kb`` holds ``BK``
  columns for the full ``M`` extent.

The ``A`` side is only needed for a fully blocked layout; ``a.view(M/BM, BM, K)``
is already free and block-row contiguous.

These use a three-dimensional ``[panel, K, block-width]`` tile with a unit
panel tile and an 8-row K sub-tile. The explicit permutation makes the value
shape match the output slice. Keeping the transpose tile small is intentional:
it limits the vector shuffle temporary to the working register tile instead of
materializing the entire 32-row block at once.

The packed operands are not used by helion_matmul_bf16.py yet: the packed-RHS
matmul formulation currently miscomputes, and whole-K matmul still hits the
deeper AMX reduction-cache-tile issue.
"""

from __future__ import annotations

import os
import statistics
import time

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 8, 32]),
)
def pack_b_panels(b3: Tensor) -> Tensor:
    """``[K, N/BN, BN]`` -> ``[N/BN, K, BN]``, one contiguous panel per column block."""
    k, nbn, bn = b3.shape
    out = torch.empty((nbn, k, bn), dtype=b3.dtype, device=b3.device)
    for j, tk, tn in hl.tile([nbn, k, bn]):
        out[j, tk, tn] = b3[tk, j, tn].permute(1, 0, 2)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 8, 32]),
)
def pack_a_panels(a3: Tensor) -> Tensor:
    """``[M, K/BK, BK]`` -> ``[K/BK, M, BK]``, one contiguous panel per K block."""
    m, nbk, bk = a3.shape
    out = torch.empty((nbk, m, bk), dtype=a3.dtype, device=a3.device)
    for kb, tm, tk in hl.tile([nbk, m, bk]):
        out[kb, tm, tk] = a3[tm, kb, tk].permute(1, 0, 2)
    return out


def pack_b(b: Tensor, block_n: int) -> Tensor:
    """Pack ``[K, N]`` into ``[N/block_n, K, block_n]``."""
    k, n = b.shape
    if n % block_n:
        raise ValueError(f"N={n} is not a multiple of block_n={block_n}")
    return pack_b_panels(b.view(k, n // block_n, block_n).contiguous())


def pack_a(a: Tensor, block_k: int) -> Tensor:
    """Pack ``[M, K]`` into ``[K/block_k, M, block_k]``."""
    m, k = a.shape
    if k % block_k:
        raise ValueError(f"K={k} is not a multiple of block_k={block_k}")
    return pack_a_panels(a.view(m, k // block_k, block_k).contiguous())


def _benchmark(name: str, operation: object, iters: int = 3) -> float:
    """Return the median per-call time in milliseconds."""
    for _ in range(1):
        operation()
    timings_ms = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iters):
            operation()
        timings_ms.append((time.perf_counter() - start) * 1_000 / iters)
    median_ms = statistics.median(timings_ms)
    print(f"{name:24s} {median_ms:8.3f} ms")
    return median_ms


def main() -> None:
    size = int(os.environ.get("HELION_PACK_SIZE", "512"))
    block_n = int(os.environ.get("HELION_PACK_BLOCK_N", "32"))
    block_k = int(os.environ.get("HELION_PACK_BLOCK_K", "32"))
    threads = int(os.environ.get("OMP_NUM_THREADS", "64"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)

    a = torch.randn((size, size), dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((size, size), dtype=torch.float32).to(torch.bfloat16)

    print(
        f"bf16 {size}x{size}; B panels of {block_n} cols ({size // block_n} panels), "
        f"A panels of {block_k} cols ({size // block_k} panels); threads={threads}"
    )

    packed_b = pack_b(b, block_n)
    torch.testing.assert_close(
        packed_b, b.view(size, size // block_n, block_n).permute(1, 0, 2).contiguous()
    )
    packed_a = pack_a(a, block_k)
    torch.testing.assert_close(
        packed_a, a.view(size, size // block_k, block_k).permute(1, 0, 2).contiguous()
    )
    print("packing numerics ok")

    b3 = b.view(size, size // block_n, block_n).contiguous()
    a3 = a.view(size, size // block_k, block_k).contiguous()
    pack_b_ms = _benchmark("pack B", lambda: pack_b_panels(b3))
    pack_a_ms = _benchmark("pack A", lambda: pack_a_panels(a3))

    bytes_moved = 2 * size * size * 2
    for name, ms in (("pack B", pack_b_ms), ("pack A", pack_a_ms)):
        print(f"{name:24s} {bytes_moved / (ms * 1e6):8.1f} GB/s")

    matmul_ms = 2 * size**3 / 1e9  # 1 TFLOP/s reference point, for scale only
    print(
        f"combined packing is {(pack_a_ms + pack_b_ms) / matmul_ms:.1%} "
        "of a 1 TFLOP/s matmul at this size"
    )


if __name__ == "__main__":
    main()
