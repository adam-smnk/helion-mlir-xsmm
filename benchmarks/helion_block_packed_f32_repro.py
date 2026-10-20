"""Small f32 block-packed matmul reproducer for the Helion MLIR backend.

This file is intentionally CPU-generic: it uses f32 inputs and does not depend on
AMX or bf16 lowering. It exercises the packed-RHS layout needed by the bf16 AMX
matmul work:

    B[K, N] -> packed_b[N / BN, K, BN]

and then computes each output column panel from that packed RHS:

    out_panel[j, M, BN] = A[M, K] @ packed_b[j, K, BN]

Run with, for example:

    OMP_NUM_THREADS=4 uv run python benchmarks/helion_block_packed_f32_repro.py

The defaults are deliberately small so this can run under `timeout 30` on a
non-AMX machine while debugging lowering/runtime bugs.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Callable


SIZE = int(os.environ.get("HELION_REPRO_SIZE", "128"))
BLOCK_M = int(os.environ.get("HELION_REPRO_BLOCK_M", "32"))
BLOCK_N = int(os.environ.get("HELION_REPRO_BLOCK_N", "32"))
BLOCK_K = int(os.environ.get("HELION_REPRO_BLOCK_K", "64"))
SAMPLES = 3
ITERS = 3


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, BLOCK_K, BLOCK_N]),
)
def pack_b_panels_f32(b3: Tensor) -> Tensor:
    """Pack ``B`` from ``[K, N/BN, BN]`` to ``[N/BN, K, BN]``."""
    k, n_panels, block_n = b3.shape
    out = torch.empty((n_panels, k, block_n), dtype=b3.dtype, device=b3.device)
    for panel in hl.grid(n_panels):
        for tile_k in hl.tile(k):
            out[panel, tile_k, :] = b3[tile_k, panel, :]
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[BLOCK_M, BLOCK_K]),
)
def matmul_packed_b_f32(a: Tensor, packed_b: Tensor) -> Tensor:
    """Compute panel-major ``[N/BN, M, BN]`` output from packed RHS panels."""
    m, k = a.shape
    n_panels, k2, block_n = packed_b.shape
    assert k == k2, "matmul dimension mismatch"

    out = torch.empty((n_panels, m, block_n), dtype=torch.float32, device=a.device)
    for panel in hl.grid(n_panels):
        for tile_m in hl.tile(m):
            acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], packed_b[panel, tile_k, :])
            out[panel, tile_m, :] = acc
    return out


def unpack_panel_major(panel_out: Tensor) -> Tensor:
    """Convert ``[N/BN, M, BN]`` back to row-major ``[M, N]`` on the host."""
    n_panels, m, block_n = panel_out.shape
    return panel_out.permute(1, 0, 2).contiguous().view(m, n_panels * block_n)


def benchmark(name: str, operation: Callable[[], object]) -> float:
    """Return the median per-call time in milliseconds."""
    for _ in range(1):
        operation()
    timings_ms = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(ITERS):
            operation()
        timings_ms.append((time.perf_counter() - start) * 1_000 / ITERS)
    median_ms = statistics.median(timings_ms)
    print(f"{name:24s} {median_ms:8.3f} ms")
    return median_ms


def main() -> None:
    if SIZE % BLOCK_N or SIZE % BLOCK_M or SIZE % BLOCK_K:
        raise ValueError("SIZE must be divisible by BLOCK_M, BLOCK_N, and BLOCK_K")

    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)

    a = torch.randn((SIZE, SIZE), dtype=torch.float32)
    b = torch.randn((SIZE, SIZE), dtype=torch.float32)
    b3 = b.view(SIZE, SIZE // BLOCK_N, BLOCK_N).contiguous()

    print(
        f"f32 packed-RHS matmul repro: {SIZE}x{SIZE}; "
        f"tiles={BLOCK_M}x{BLOCK_N}x{BLOCK_K}; threads={threads}"
    )

    packed_b = pack_b_panels_f32(b3)
    torch.testing.assert_close(
        packed_b,
        b3.permute(1, 0, 2).contiguous(),
        rtol=0,
        atol=0,
    )
    print("pack_b numerics ok")

    panel_result = matmul_packed_b_f32(a, packed_b)
    result = unpack_panel_major(panel_result)
    reference = torch.mm(a, b)
    try:
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
    except AssertionError as exc:
        diff = (result - reference).abs()
        mismatches = (~torch.isclose(result, reference, rtol=1e-4, atol=1e-4)).sum()
        print("packed matmul numerics FAILED")
        print(f"mismatched elements: {mismatches.item()} / {result.numel()}")
        print(f"max abs error: {diff.max().item():.6e}")
        message = next((line for line in str(exc).splitlines() if line), "")
        if message:
            print(message)
        sys.exit(1)
    print("packed matmul numerics ok")

    with torch.inference_mode():
        pack_ms = benchmark("pack B", lambda: pack_b_panels_f32(b3))
        matmul_ms = benchmark("packed matmul", lambda: matmul_packed_b_f32(a, packed_b))
        total_ms = benchmark(
            "pack + matmul",
            lambda: matmul_packed_b_f32(a, pack_b_panels_f32(b3)),
        )
        eager_ms = benchmark("torch.mm", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    print(f"pack B               {pack_ms:8.3f} ms")
    print(f"packed matmul        {flops / (matmul_ms * 1e6):8.1f} GFLOP/s")
    print(f"pack + matmul        {flops / (total_ms * 1e6):8.1f} GFLOP/s")
    print(f"torch.mm             {flops / (eager_ms * 1e6):8.1f} GFLOP/s")


if __name__ == "__main__":
    main()
