"""
Fused Matmul + ReLU with MLIR Backend

Demonstrates a fused kernel that combines matrix multiplication and ReLU activation.
This example shows how complex operations are composed within the MLIR backend.

This example illustrates:
- Multiple nested tile loops (output dims, reduction dim)
- Composition of linalg.matmul and custom element-wise operations
- Instruction-level fusion opportunities
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend import generate_mlir


@helion.kernel(static_shapes=True)
def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fused matmul + ReLU: C = max(A @ B, 0)

    Args:
        x: Input tensor of shape (M, K)
        y: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with ReLU applied

    The kernel computes matrix multiplication with ReLU activation applied
    to each output tile, demonstrating kernel fusion.
    """
    m, k = x.shape
    k2, n = y.shape
    assert k == k2

    # Initialize output
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    # Outer loop over output dimensions
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulator for this output tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        # Inner loop over reduction dimension
        for tile_k in hl.tile(k):
            # Accumulate matmul
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

        # Apply ReLU to accumulated result
        out[tile_m, tile_n] = torch.relu(acc)

    return out


@helion.kernel(static_shapes=True)
def matmul_add(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Matmul + bias add: C = A @ B + bias

    Args:
        x: Input tensor of shape (M, K)
        y: Input tensor of shape (K, N)
        bias: Bias tensor of shape (N,)

    Returns:
        Output tensor of shape (M, N) with bias added

    Demonstrates how broadcasting bias across matrix multiplication results
    is handled in the MLIR backend.
    """
    m, k = x.shape
    k2, n = y.shape
    assert k == k2

    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

        # Add bias (broadcasted)
        out[tile_m, tile_n] = acc + bias[tile_n]

    return out


def main() -> None:
    """Generate and display MLIR IR for fused operations."""

    device = torch.device("cpu")
    m, k, n = 64, 128, 96

    x = torch.randn(m, k, dtype=torch.float32, device=device)
    y = torch.randn(k, n, dtype=torch.float32, device=device)
    bias = torch.randn(n, dtype=torch.float32, device=device)

    print("=" * 80)
    print("Fused Matmul + ReLU/Bias MLIR Generation Examples")
    print("=" * 80)

    # Example 1: Matmul + ReLU
    print("\n" + "-" * 80)
    print("1. Matmul + ReLU (max(A @ B, 0))")
    print("-" * 80)
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"Output shape: ({m}, {n})")

    module = generate_mlir(matmul_relu, [x, y])
    ir_str = str(module)

    print(f"\nGenerated IR size: {len(ir_str)} characters")
    print("Contains:")
    print(f"  - scf.forall: {('scf.forall' in ir_str)}")
    print(f"  - scf.for: {('scf.for' in ir_str)}")
    print(f"  - linalg.matmul: {('linalg.matmul' in ir_str)}")
    print(f"  - Custom ops (like max): {('linalg.generic' in ir_str)}")

    # Example 2: Matmul + Bias
    print("\n" + "-" * 80)
    print("2. Matmul + Bias (A @ B + bias)")
    print("-" * 80)
    print(f"Input shapes: x={x.shape}, y={y.shape}, bias={bias.shape}")
    print(f"Output shape: ({m}, {n})")

    module = generate_mlir(matmul_add, [x, y, bias])
    ir_str = str(module)

    print(f"\nGenerated IR size: {len(ir_str)} characters")
    print("Contains:")
    print(f"  - scf.forall: {('scf.forall' in ir_str)}")
    print(f"  - linalg.matmul: {('linalg.matmul' in ir_str)}")
    print(f"  - tensor.extract_slice: {('tensor.extract_slice' in ir_str)}")

    print("\n" + "=" * 80)
    print("Key observations:")
    print("=" * 80)
    print("- Fused operations are represented as sequential operations in MLIR")
    print("- Downstream compilers (Triton, MLIR's affine transform, etc.) can optimize")
    print("- Bias broadcasting is handled through slice indexing (e.g., bias[tile_n])")
    print(
        "- Operations remain in tensor abstraction (no bufferization in MLIR backend)"
    )


if __name__ == "__main__":
    main()
