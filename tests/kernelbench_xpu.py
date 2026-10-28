"""Helion XPU kernels of KernelBench level 1 problems, at the shapes and dtypes of
AI-bench's ``mlir-gpu-bench`` variant (``AI-bench/problems/specs/KernelBench/
level1/<name>.yaml``), for IR-level tests (``test_xpu_kernelbench.py``).

The XeGPU pipeline compiles static, nicely shaped kernels: the block sizes below
divide every size they tile (no masking), so each problem's block sizes are
chosen per shape:

- Matmuls (bf16 operands, f32 accumulator, a bf16 result as PyTorch's matmul of
  bf16 tensors): block sizes ``[M tile, N tile, K tile]``, the workgroup tile and
  the K-loop step. 256x256 workgroups (lighthouse's subgroup inference picks
  32 subgroups of 32x64) with K steps of 32; a 128x128 workgroup tile for the
  256x256 output of 6_large_K (4 workgroups), a 16 K step for K = 32
  (9_Tall_skinny; the K loop must stay, DPAS K is 16). Transposed operands
  (16-18: A stored ``[K, M]``, B ``[N, K]``, as the specs give them) are loaded
  as transposed 16x16 tiles.
- Elementwise (bf16, 4096x393216): ``[rows, cols]`` workgroup tiles of 128x256
  (lighthouse kernel_bench's elementwise default), computed in f32 like
  PyTorch's bf16 kernels; inferred subgroup tiles of 32x32 (32 subgroups), 32x32
  bf16 block loads, 8x16 block stores.
- Row reductions (f32): ``[rows, column chunk]`` block sizes, 64 rows per
  workgroup (lighthouse's reduction class: 8 subgroups of 8 rows) and 64-column
  chunks; multi-pass over the chunks like AI-bench's CPU kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

BF16 = torch.bfloat16
F32 = torch.float32


def _kernel(block_sizes: list[int]) -> Callable[[Callable[..., object]], object]:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes),
    )


# --- Matmuls --------------------------------------------------------------------


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a.T @ b`` of ``a: [K, M]`` and ``b: [K, N]``."""
    k, m = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b.T`` of ``a: [M, K]`` and ``b: [N, K]``."""
    m, k = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_both(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a.T @ b.T`` of ``a: [K, M]`` and ``b: [N, K]``."""
    k, m = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


# --- Elementwise ------------------------------------------------------------------


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.01)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def selu(x: torch.Tensor) -> torch.Tensor:
    return torch.selu(x)


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.hardsigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.abs(x))


def elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x, alpha=1.0)


def hardtanh(x: torch.Tensor) -> torch.Tensor:
    return F.hardtanh(x, min_val=-1.0, max_val=1.0)


# The kernels apply the functions above to every element, in f32 (Helion kernels
# cannot close over a function: one kernel per function).


def _relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = relu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = leaky_relu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = sigmoid(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = tanh(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _swish(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = swish(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _selu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = selu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = hardsigmoid(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _softplus(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = softplus(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _softsign(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = softsign(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _elu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = elu(x[tile].to(torch.float32)).to(x.dtype)
    return out


def _hardtanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = hardtanh(x[tile].to(torch.float32)).to(x.dtype)
    return out


# --- Row reductions -------------------------------------------------------------


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """Row softmax in three passes over column chunks: maxima, sums of
    exponentials, then the output."""
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


def _log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row log-softmax: ``x - max - log(sum(exp(x - max)))``, in three passes."""
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
        shift = row_top + torch.log(total.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - shift
    return out


def _l1_norm(x: torch.Tensor) -> torch.Tensor:
    """``x / mean(|x|, dim=1, keepdim=True)`` (KernelBench 38), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        inv = n / acc.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inv
    return out


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """``x / norm(x, p=2, dim=1, keepdim=True)`` (KernelBench 39), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            acc = acc + v * v
        inv = torch.rsqrt(acc.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inv
    return out


# --- Problems ---------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """A KernelBench level 1 problem at its ``mlir-gpu-bench`` variant."""

    name: str
    """The KernelBench file name, without ``.py``."""
    kind: str
    """``matmul``, ``elementwise`` or ``reduction``."""
    fn: Callable[..., object]
    shapes: tuple[tuple[int, ...], ...]
    """The input shapes, in argument order."""
    dtype: torch.dtype
    block_sizes: list[int]
    grid: tuple[int, ...]
    """The workgroup grid: the output size over the workgroup tile."""
    kernel: object = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel", _kernel(self.block_sizes)(self.fn))


_GEMM_BLOCKS = [256, 256, 32]
_ELEMENTWISE_BLOCKS = [128, 256]
_ELEMENTWISE_SHAPE = (4096, 393216)
_REDUCTION_BLOCKS = [64, 64]


_MATMULS = {
    "": _matmul,
    "a": _matmul_transposed_a,
    "b": _matmul_transposed_b,
    "ab": _matmul_transposed_both,
}


def _gemm(
    name: str,
    mnk: tuple[int, int, int],
    transposed: str = "",
    blocks: list[int] = _GEMM_BLOCKS,
) -> Problem:
    """A matmul of ``M x K`` and ``K x N`` operands, stored ``[K, M]`` /
    ``[N, K]`` when ``transposed`` names them."""
    m, n, k = mnk
    a = (k, m) if "a" in transposed else (m, k)
    b = (n, k) if "b" in transposed else (k, n)
    grid = (m // blocks[0], n // blocks[1])
    return Problem(name, "matmul", _MATMULS[transposed], (a, b), BF16, blocks, grid)


def _unary(name: str, fn: Callable[..., object]) -> Problem:
    grid = tuple(
        size // block
        for size, block in zip(_ELEMENTWISE_SHAPE, _ELEMENTWISE_BLOCKS, strict=True)
    )
    shapes = (_ELEMENTWISE_SHAPE,)
    return Problem(name, "elementwise", fn, shapes, BF16, _ELEMENTWISE_BLOCKS, grid)


def _rows(name: str, fn: Callable[..., object], shape: tuple[int, int]) -> Problem:
    grid = (shape[0] // _REDUCTION_BLOCKS[0],)
    return Problem(name, "reduction", fn, (shape,), F32, _REDUCTION_BLOCKS, grid)


PROBLEMS: tuple[Problem, ...] = (
    _gemm("1_Square_matrix_multiplication_", (4096, 4096, 4096)),
    _gemm("2_Standard_matrix_multiplication_", (2048, 4096, 8192)),
    _gemm(
        "6_Matmul_with_large_K_dimension_", (256, 256, 524288), blocks=[128, 128, 32]
    ),
    _gemm("7_Matmul_with_small_K_dimension_", (32768, 32768, 64)),
    _gemm(
        "9_Tall_skinny_matrix_multiplication_",
        (32768, 32768, 32),
        blocks=[256, 256, 16],
    ),
    _gemm("13_Matmul_for_symmetric_matrices", (4096, 4096, 4096)),
    _gemm("16_Matmul_with_transposed_A", (2048, 4096, 8192), "a"),
    _gemm("17_Matmul_with_transposed_B", (2048, 4096, 8192), "b"),
    _gemm("18_Matmul_with_transposed_both", (2048, 4096, 8192), "ab"),
    _unary("19_ReLU", _relu),
    _unary("20_LeakyReLU", _leaky_relu),
    _unary("21_Sigmoid", _sigmoid),
    _unary("22_Tanh", _tanh),
    _unary("25_Swish", _swish),
    _unary("27_SELU_", _selu),
    _unary("28_HardSigmoid", _hardsigmoid),
    _unary("29_Softplus", _softplus),
    _unary("30_Softsign", _softsign),
    _unary("31_ELU", _elu),
    _unary("32_HardTanh", _hardtanh),
    _rows("23_Softmax", _softmax, (4096, 393216)),
    _rows("24_LogSoftmax", _log_softmax, (4096, 393216)),
    _rows("38_L1Norm_", _l1_norm, (32768, 65536)),
    _rows("39_L2Norm_", _l2_norm, (32768, 65536)),
)
"""The problems, in KernelBench order by class (``(M, N, K)`` for matmuls)."""
