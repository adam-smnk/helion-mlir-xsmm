"""
Inline MLIR in Helion kernels (MLIR backend)

Demonstrates ``helion_mlir_backend.inline_mlir``, which calls a hand-written MLIR
function on tiles inside a device loop:

1. A pure tensor function built from ``linalg.generic``: softplus with a scalar
   ``beta`` argument. Its dims are ``?``, so it works for any block size.
2. An ``mlir.ir.Module`` built with the MLIR Python bindings instead of text: a
   polynomial whose Horner steps are generated from a list of coefficients.
3. A matmul microkernel whose body works on memrefs: it reads the operand tiles
   in place (``bufferization.to_buffer ... read_only`` with a strided layout),
   accumulates into the accumulator tile's buffer with vector FMAs, and returns
   that buffer as the new accumulator. Its shapes are static, so it pairs with a
   fixed config.
4. The same microkernel taking its buffers with the identity layout instead: the
   results are the same, but bufferization now copies an operand tile on every
   step of the K loop.

Each call also gets a ``reference`` implementation, used by Helion's eager ref
mode and other backends.
"""

from __future__ import annotations

import math

import helion
import helion.language as hl
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import tensor
import mlir.ir as ir
from mlir.passmanager import PassManager
import torch

from helion_mlir_backend import generate_mlir
from helion_mlir_backend import inline_mlir

SOFTPLUS = """
#id = affine_map<(i, j) -> (i, j)>
func.func @softplus(%x: tensor<?x?xf32>, %beta: f32) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %one = arith.constant 1.0 : f32
    %scaled = arith.mulf %v, %beta : f32
    %exp = math.exp %scaled : f32
    %sum = arith.addf %exp, %one : f32
    %log = math.log %sum : f32
    %r = arith.divf %log, %beta : f32
    linalg.yield %r : f32
  } -> tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}
"""

# C[8x16] += A[8x8] @ B[8x16], one row of C per vector<16xf32>. Buffers are taken
# with a fully dynamic layout, which bufferization can always give without a copy,
# then cast to the unit inner stride that the vector reads and writes assume.
MICROKERNEL = """
!any_a = memref<8x8xf32, strided<[?, ?], offset: ?>>
!any_bc = memref<8x16xf32, strided<[?, ?], offset: ?>>
!a = memref<8x8xf32, strided<[?, 1], offset: ?>>
!bc = memref<8x16xf32, strided<[?, 1], offset: ?>>
func.func @microkernel(%acc: tensor<8x16xf32>, %a: tensor<8x8xf32>, %b: tensor<8x16xf32>)
    -> tensor<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  %a_any = bufferization.to_buffer %a read_only : tensor<8x8xf32> to !any_a
  %b_any = bufferization.to_buffer %b read_only : tensor<8x16xf32> to !any_bc
  %c_any = bufferization.to_buffer %acc : tensor<8x16xf32> to !any_bc
  %A = memref.cast %a_any : !any_a to !a
  %B = memref.cast %b_any : !any_bc to !bc
  %C = memref.cast %c_any : !any_bc to !bc
  scf.for %i = %c0 to %c8 step %c1 {
    %row = vector.transfer_read %C[%i, %c0], %zero {in_bounds = [true]} : !bc, vector<16xf32>
    %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%partial = %row) -> (vector<16xf32>) {
      %a_ik = memref.load %A[%i, %k] : !a
      %a_splat = vector.broadcast %a_ik : f32 to vector<16xf32>
      %b_row = vector.transfer_read %B[%k, %c0], %zero {in_bounds = [true]} : !bc, vector<16xf32>
      %next = vector.fma %a_splat, %b_row, %partial : vector<16xf32>
      scf.yield %next : vector<16xf32>
    }
    vector.transfer_write %sum, %C[%i, %c0] {in_bounds = [true]} : vector<16xf32>, !bc
  }
  %result = bufferization.to_tensor %c_any restrict writable : !any_bc to tensor<8x16xf32>
  return %result : tensor<8x16xf32>
}
"""

# The same microkernel taking its buffers with the identity layout (memref<8x8xf32>).
MICROKERNEL_IDENTITY_LAYOUT = MICROKERNEL.replace(", strided<[?, ?], offset: ?>", "")


def build_polynomial(coefficients: list[float]) -> ir.Module:
    """``sum(c[i] * x**i)`` on 2-D f32 tiles, as one ``linalg.generic`` in Horner form."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        dynamic = ir.ShapedType.get_dynamic_size()
        tile = ir.RankedTensorType.get([dynamic, dynamic], f32)
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(2))
        parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(tile, name="polynomial")
            def polynomial(x: ir.Value) -> ir.Value:
                index = ir.IndexType.get()
                sizes = [tensor.dim(x, arith.constant(index, dim)) for dim in (0, 1)]
                init = tensor.EmptyOp(sizes, f32).result
                generic = linalg.GenericOp(
                    [tile],
                    [x],
                    [init],
                    ir.ArrayAttr.get([identity, identity]),
                    ir.ArrayAttr.get([parallel, parallel]),
                )
                body = generic.regions[0].blocks.append(f32, f32)
                with ir.InsertionPoint(body):
                    value = body.arguments[0]
                    result = arith.constant(f32, coefficients[-1])
                    for coefficient in reversed(coefficients[:-1]):
                        product = arith.mulf(result, value)
                        result = arith.addf(product, arith.constant(f32, coefficient))
                    linalg.YieldOp([result])
                return generic.result

    return module


# The Taylor series of exp to degree 8: within 3e-6 of exp on [-1, 1].
EXP_COEFFICIENTS = [1 / math.factorial(power) for power in range(9)]
EXP_POLYNOMIAL = build_polynomial(EXP_COEFFICIENTS)

# The bufferization stage of the lighthouse pipelines, after inlining.
BUFFERIZE = (
    "builtin.module(inline,canonicalize,eliminate-empty-tensors,"
    "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map "
    "bufferize-function-boundaries allow-return-allocs-from-loops},"
    "drop-equivalent-buffer-results,buffer-deallocation-pipeline,"
    "convert-bufferization-to-memref,cse,canonicalize)"
)


def softplus_reference(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.softplus(x, beta)


def exp_polynomial_reference(x: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(x, EXP_COEFFICIENTS[-1])
    for coefficient in reversed(EXP_COEFFICIENTS[:-1]):
        result = result * x + coefficient
    return result


def microkernel_reference(
    acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return acc + a @ b


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 32]))
def softplus(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(SOFTPLUS, [t, 2.0], t, reference=softplus_reference)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 32]))
def exp_polynomial(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(
            EXP_POLYNOMIAL, [t], t, reference=exp_polynomial_reference
        )
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 16, 8]))
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = inline_mlir(
                MICROKERNEL,
                [acc, a[tm, tk], b[tk, tn]],
                acc,
                reference=microkernel_reference,
            )
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 16, 8]))
def matmul_identity_layout(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = inline_mlir(
                MICROKERNEL_IDENTITY_LAYOUT,
                [acc, a[tm, tk], b[tk, tn]],
                acc,
                reference=microkernel_reference,
            )
        out[tm, tn] = acc
    return out


def buffer_traffic(kernel: helion.Kernel, args: list[torch.Tensor]) -> dict[str, str]:
    """Allocations / copies in the bufferized kernel, per K step and per output tile."""
    module = generate_mlir(kernel, args)
    with module.context:
        PassManager.parse(BUFFERIZE).run(module.operation)
    counts = {"scf.for": [0, 0], "scf.forall": [0, 0], "func.func": [0, 0]}

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in ("memref.alloc", "memref.copy"):
            loop = op.parent
            while loop.name not in counts:
                loop = loop.parent
            counts[loop.name][op.name == "memref.copy"] += 1
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return {
        "per K step": "{} / {}".format(*counts["scf.for"]),
        "per output tile": "{} / {}".format(*counts["scf.forall"]),
    }


def main() -> None:
    torch.manual_seed(0)

    x = torch.randn(100, 70)
    y = softplus(x)
    error = (y - torch.nn.functional.softplus(x, 2.0)).abs().max().item()
    print(f"softplus (linalg.generic snippet): max error {error:.2e}")

    x = torch.rand(100, 70) * 2 - 1
    error = (exp_polynomial(x) - torch.exp(x)).abs().max().item()
    print(f"exp polynomial (ir.Module built in Python): max error vs exp {error:.2e}")

    a, b = torch.randn(64, 48), torch.randn(48, 80)
    c = matmul(a, b)
    error = (c - a @ b).abs().max().item()
    print(f"matmul (memref/vector microkernel): max error {error:.2e}")

    calls = [
        line.strip()
        for line in str(generate_mlir(matmul, [a, b])).splitlines()
        if "func.call" in line
    ]
    print("\nThe kernel calls the microkernel before inlining:")
    print("\n".join(f"  {line}" for line in calls))

    # Compiling this variant also reports an InlineMLIRHint about its layout.
    c = matmul_identity_layout(a, b)
    error = (c - a @ b).abs().max().item()
    print(f"\nmatmul, identity-layout buffers: max error {error:.2e}")
    print("Allocations / copies after bufferization:")
    print(f"  {'to_buffer layout':<34}{'per K step':>12}{'per output tile':>18}")
    for label, kernel in (
        ("strided<[?, ?], offset: ?> + cast", matmul),
        ("identity", matmul_identity_layout),
    ):
        traffic = buffer_traffic(kernel, [a, b])
        print(
            f"  {label:<34}{traffic['per K step']:>12}{traffic['per output tile']:>18}"
        )
    print(
        "\n"
        "The identity layout copies the B tile (a strided view of b) into a new\n"
        "buffer on every K step. Per output tile, both allocate the accumulator\n"
        "and copy it into the output at the end."
    )


if __name__ == "__main__":
    main()
