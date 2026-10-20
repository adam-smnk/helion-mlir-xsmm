"""
Matmul Kernel with MLIR Backend

Demonstrates a tiled matrix multiplication kernel that generates MLIR IR
using linalg.matmul operations in the Linalg-on-Tensors abstraction.

This example shows:
- Basic kernel structure with hl.tile() loops
- How to generate and inspect MLIR IR
- Integration with the MLIR backend
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend import generate_mlir


@helion.kernel(static_shapes=True)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Tiled matrix multiplication: C = A @ B

    Args:
        x: Input tensor of shape (M, K) with dtype float32
        y: Input tensor of shape (K, N) with dtype float32

    Returns:
        Output tensor of shape (M, N) with dtype float32

    The kernel uses tile-based computation where:
    - Outer loop iterates over (M, N) tile dimensions
    - Inner loop iterates over K reduction dimension
    - Each tile computes a partial result using linalg.matmul
    """
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, "Dimension mismatch: K must match"

    # Initialize output accumulator
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    # Outer loop: tile over output dimensions (M, N)
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulator for this tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        # Inner loop: tile over reduction dimension (K)
        for tile_k in hl.tile(k):
            # Accumulate: acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

        # Store result back to output
        out[tile_m, tile_n] = acc

    return out


def main() -> None:
    """Generate and display MLIR IR for the matmul kernel."""

    # Create test inputs
    device = torch.device("cpu")
    m, k, n = 64, 128, 96

    x = torch.randn(m, k, dtype=torch.float32, device=device)
    y = torch.randn(k, n, dtype=torch.float32, device=device)

    print("=" * 80)
    print("Matmul Kernel MLIR Generation Example")
    print("=" * 80)
    print(f"\nInput shapes: x={x.shape}, y={y.shape}")
    print(f"Output shape: ({m}, {n})")

    # Generate MLIR module
    print("\nGenerating MLIR IR...")
    module = generate_mlir(matmul_kernel, [x, y])

    # Display the generated MLIR
    print("\n" + "=" * 80)
    print("Generated MLIR IR:")
    print("=" * 80)
    print(str(module))

    print("\n" + "=" * 80)
    print("Key observations:")
    print("=" * 80)
    print("- scf.forall: Outer loop over output tile dimensions (m, n)")
    print("- scf.for: Inner loop over reduction dimension (k)")
    print("- linalg.matmul: Matrix multiplication for each tile")
    print("- tensor.extract_slice: Extract input tiles")
    print("- tensor.parallel_insert_slice: Store output tiles")


if __name__ == "__main__":
    main()
