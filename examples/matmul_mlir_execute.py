"""
Matmul Execution Example with Numerical Validation (CPU)

Demonstrates end-to-end flow for 128x128x128 matmul:
1. Define Helion kernel (backend="mlir")
2. Generate MLIR
3. Execute via lighthouse JIT (execute_mlir)
4. Execute via direct helion kernel call (MLIR backend integrated)
5. Validate numerical correctness
6. Report timing
"""

from __future__ import annotations

import time

import helion
import helion.language as hl
import torch

from helion_mlir_backend import generate_mlir


@helion.kernel(
    static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[32])
)
def matmul_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute C = A @ B for 128x128x128 matmul.

    Simple outer product accumulation without explicit tiling.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch"

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Single tile covering entire M dimension
    for tile_m in hl.tile(M):
        # Load A tile and compute matmul with full B
        A_tile = hl.load(A, [tile_m, slice(None)])
        B_full = hl.load(B, [slice(None), slice(None)])
        C_tile = A_tile @ B_full
        hl.store(C, [tile_m, slice(None)], C_tile)

    return C


def main() -> None:
    """Run end-to-end matmul with execution and validation."""
    device = torch.device("cpu")
    M, K, N = 128, 128, 128

    print("=" * 80)
    print("Matmul Execution Example (128x128x128, CPU)")
    print("=" * 80)

    # Generate test inputs
    print("\n[1/5] Preparing inputs...")
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    print(f"  A: {A.shape} {A.dtype} {A.device}")
    print(f"  B: {B.shape} {B.dtype} {B.device}")

    # Generate MLIR
    print("\n[2/5] Generating MLIR...")
    t0 = time.time()
    mlir_module = generate_mlir(matmul_kernel, [A, B])
    t_mlir_gen = time.time() - t0
    print(f"  MLIR generation time: {t_mlir_gen:.4f}s")
    print(f"  Module IR size: {len(str(mlir_module))} chars")

    # Execute via lighthouse
    print("\n[3/5] JIT compilation and execution...")
    t0 = time.time()
    try:
        from helion._compiler.backend_registry import get_backend_class

        backend_cls = get_backend_class("mlir")
        backend = backend_cls()
        C_jit = backend.execute_mlir(mlir_module, A, B, kernel_name="matmul_kernel")
        t_exec = time.time() - t0
        print(f"  JIT execution time: {t_exec:.4f}s")
    except Exception as exc:
        print("  ERROR: Execution failed")
        print(f"  {type(exc).__name__}: {exc}")
        print("\n[DIAGNOSTIC] Stopping here as requested.")
        print("  Summarize the error above and investigate lighthouse limitations.")
        return

    C_direct: torch.Tensor | None = None
    # Direct kernel call (MLIR backend integrated into helion runtime)
    print("\n[4/5] Direct kernel call via helion runtime...")
    t0 = time.time()
    try:
        C_direct = matmul_kernel(A, B)
        t_direct = time.time() - t0
        print(f"  Direct call time: {t_direct:.4f}s")
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        t_direct = float("nan")

    # Validate both execution paths against PyTorch reference
    print("\n[5/5] Numerical validation...")
    C_ref = A @ B

    def _check(name: str, C: torch.Tensor) -> None:
        max_err = torch.max(torch.abs(C - C_ref)).item()
        rel_err = (torch.norm(C - C_ref) / torch.norm(C_ref)).item()
        status = "✓ PASS" if max_err < 1e-5 or rel_err < 1e-6 else "✗ FAIL"
        print(f"  {name}: max_abs={max_err:.2e}  rel_l2={rel_err:.2e}  {status}")

    _check("execute_mlir ", C_jit)
    if C_direct is not None:
        _check("direct call  ", C_direct)

    # Summary
    print("\n[6/6] Performance Summary")
    print("=" * 80)
    print(f"  MLIR generation:   {t_mlir_gen:.4f}s")
    print(f"  execute_mlir:      {t_exec:.4f}s")
    print(f"  direct call:       {t_direct:.4f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
