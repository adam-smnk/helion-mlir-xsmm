"""
Multi-Phase Kernel with MLIR Backend

Demonstrates the MLIR backend's `hl.barrier()` support: a kernel with
multiple top-level `hl.tile()` loops separated by an explicit phase boundary,
plus a host-computed tensor ("interop") that isn't one of the kernel's own
parameters.

Why hl.barrier() is required (not optional, not CPU-specific):
- Helion's frontend rejects a later top-level loop reading a tensor written
  by an earlier one unless hl.barrier() separates them (LoopDependencyError).
- In this backend, each hl.barrier()-separated phase compiles to its own
  private MLIR function; the module's entry function calls them in order.

Note on syntax: Helion does not allow any statement (other than
hl.barrier() itself) between two top-level device loops, so a host-computed
interop tensor that a later phase needs must be computed *before* the loops
that use it. `scale` is computed once, before phase 0, and consumed inside
phase 0's loop via `hl.load(scale, [])`.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

import helion_mlir_backend  # noqa: F401  (registers the "mlir" backend)


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[32, 32, 32, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def normalize_then_scale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two-phase kernel with a host-computed interop tensor.

    (host)   scale = x.mean() * 2.0     -- computed once, before phase 0
    Phase 0: mid = (x + y) * scale
    Phase 1: out = mid * 2.0
    """
    m, n = x.shape
    scale = x.mean() * 2.0
    mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        s = hl.load(scale, [])
        mid[tile_m, tile_n] = (x[tile_m, tile_n] + y[tile_m, tile_n]) * s

    # Phase boundary: phase 1's loop below reads `mid`, written above.
    hl.barrier()

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = mid[tile_m, tile_n] * 2.0

    return out


def main() -> None:
    """Run the multi-phase kernel and validate against eager PyTorch."""
    device = torch.device("cpu")
    m, n = 32, 32

    x = torch.randn(m, n, dtype=torch.float32, device=device)
    y = torch.randn(m, n, dtype=torch.float32, device=device)

    print("=" * 80)
    print("Multi-Phase Kernel Execution (hl.barrier() + host-tensor interop)")
    print("=" * 80)
    print(f"Input shapes: x={x.shape}, y={y.shape}")

    print('\nRunning kernel via direct backend="mlir" call...')
    actual = normalize_then_scale(x, y)

    scale = x.mean() * 2.0
    mid = (x + y) * scale
    expected = mid * 2.0

    max_err = torch.max(torch.abs(actual - expected)).item()
    status = "PASS" if max_err < 1e-4 else "FAIL"
    print(f"Result shape: {actual.shape}")
    print(f"Max abs error vs eager PyTorch: {max_err:.2e}  [{status}]")

    print("\n" + "=" * 80)
    print("Key observations:")
    print("=" * 80)
    print("- Two hl.tile() loops, separated by hl.barrier(), compile to two")
    print("  private MLIR functions called in order by the entry function.")
    print("- `scale` is a host-computed tensor (not a kernel parameter),")
    print("  recomputed by the host code on each call and passed to the entry.")
    print("- `mid` (phase 0's output) is threaded into phase 1 as an SSA value.")


if __name__ == "__main__":
    main()
