"""The XPU target on hardware: kernels compiled to device binaries and run on an
Intel GPU through the Level Zero runtime, against eager PyTorch on the same XPU.

Every test is marked ``xpu_runtime``: it runs only where ``torch.xpu`` finds a
device and the MLIR bindings can emit and run XPU binaries (an LLVM with the
SPIR-V target and the Level Zero runner); elsewhere it is skipped with the
missing piece as the reason (``tests/conftest.py``).

Inputs and outputs are torch XPU tensors: their USM pointers go into the
kernels' memref descriptors (no host copies). Errors are the largest absolute
error normalized by ``max|ref|``; the bounds follow from the output dtype's
rounding (unit roundoff 2**-8 for bf16, 2**-11 for f16: two roundings of nearly
the same f32 value differ by at most twice that, one ulp), since the kernels and
eager PyTorch round the same f32 math at different points or in another order.
Exact ops (ReLU, HardTanh, maxima, copies) are compared bit for bit.

Shapes are small and aligned: the hand-picked block sizes divide them (no
masking on XPU), with at least two workgroups per tiled dim and a K loop.
"""

from __future__ import annotations

import itertools
import math
import re
import string
from typing import TYPE_CHECKING
from unittest import mock

import helion
import helion.language as hl
import pytest
import torch
import torch.nn.functional as F

from tests import kernelbench_xpu as kb
from tests import test_xevm_bf16_extensions as bf16_kernels
from tests import test_xpu_ir as ir_kernels
from tests import xpu_math_kernels as mk
from tests.test_xpu_kernelbench import _reference as level2_reference

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.targets.xpu import XpuTarget

if TYPE_CHECKING:
    from collections.abc import Callable

# A kernel that faults loses the device for the rest of its process: each test
# runs in a process of its own.
pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated]

BF16 = torch.bfloat16
F16 = torch.float16
F32 = torch.float32

# Normalized error bounds by output dtype: two roundings of nearly the same f32
# value differ by at most one ulp (twice the unit roundoff, 2**-8 bf16 and 2**-11
# f16), doubled for the f32 math's own order differences; f32 outputs differ by
# accumulation order.
_BOUND = {BF16: 2**-7, F16: 2**-10, F32: 1e-5}
DEVICE = "xpu"


def normalized_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.float()
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _check(out: torch.Tensor, ref: torch.Tensor, bound: float | None = None) -> None:
    assert out.device.type == "xpu"
    assert out.shape == ref.shape
    assert torch.isfinite(out.float()).all()
    if bound == 0.0:
        # An exact op: bit for bit (also signed zeros).
        expected = ref.to(out.dtype)
        assert torch.equal(
            out.view(torch.int16 if out.element_size() == 2 else torch.int32),
            expected.view(torch.int16 if out.element_size() == 2 else torch.int32),
        )
        return
    error = normalized_error(out, ref)
    limit = _BOUND[out.dtype] if bound is None else bound
    assert error <= limit, f"normalized error {error:.3g} > {limit:.3g}"


# Each test imports this module in its own process: the same inputs every run.
torch.manual_seed(0)


def _randn(*shape: int, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=DEVICE)


@pytest.fixture(autouse=True)
def _full_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binaries and execution: no stop point, the XPU target by device."""
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


def _kernel(fn: Callable[..., object], block_sizes: list[int], **xegpu: object):
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes, **xegpu),
    )(fn)


# --- KernelBench XPU configs (tests/kernelbench_xpu.py) on the device ----------

# Small shapes the problems' block sizes divide: (M, N, K) for matmuls (K keeps
# a K loop; 7 and 9 keep their spec K), else the input shapes.
_MATMUL_MNK = {
    "1_Square_matrix_multiplication_": (512, 512, 256),
    "2_Standard_matrix_multiplication_": (512, 512, 256),
    "6_Matmul_with_large_K_dimension_": (256, 256, 4096),
    "7_Matmul_with_small_K_dimension_": (512, 512, 64),
    "9_Tall_skinny_matrix_multiplication_": (512, 512, 32),  # K padded to 64
    "13_Matmul_for_symmetric_matrices": (512, 512, 256),
    "16_Matmul_with_transposed_A": (512, 512, 256),
    "17_Matmul_with_transposed_B": (512, 512, 256),
    "18_Matmul_with_transposed_both": (512, 512, 256),
}
_ELEMENTWISE_SHAPE = (16, 16384)
"""Two column tiles of ``ELEMENTWISE_BLOCKS`` (one row, 8192 columns) per row,
16 rows: both grid dims (the columns first) have more than one tile."""
_REDUCTION_SHAPE = (128, 256)
_ELEMENTWISE_REFERENCES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "19_ReLU": kb.relu,
    "20_LeakyReLU": kb.leaky_relu,
    "21_Sigmoid": kb.sigmoid,
    "22_Tanh": kb.tanh,
    "25_Swish": kb.swish,
    "27_SELU_": kb.selu,
    "28_HardSigmoid": kb.hardsigmoid,
    "29_Softplus": kb.softplus,
    "30_Softsign": kb.softsign,
    "31_ELU": kb.elu,
    "32_HardTanh": kb.hardtanh,
}
_REDUCTION_REFERENCES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "23_Softmax": lambda x: torch.softmax(x, dim=1),
    "24_LogSoftmax": lambda x: torch.log_softmax(x, dim=1),
    "38_L1Norm_": lambda x: x / torch.mean(torch.abs(x), dim=1, keepdim=True),
    "39_L2Norm_": lambda x: x / torch.norm(x, p=2, dim=1, keepdim=True),
}


def _matmul_operands(
    name: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The operands as the problem stores them (A ``[K, M]`` when transposed, B
    ``[N, K]``), and ``A @ B`` in eager torch."""
    m, n, k = _MATMUL_MNK[name]
    transposed_a = "transposed_A" in name or "transposed_both" in name
    transposed_b = "transposed_B" in name or "transposed_both" in name
    a = _randn(k, m, dtype=dtype) if transposed_a else _randn(m, k, dtype=dtype)
    b = _randn(n, k, dtype=dtype) if transposed_b else _randn(k, n, dtype=dtype)
    return a, b, (a.T if transposed_a else a) @ (b.T if transposed_b else b)


_PAD_K = {"9_Tall_skinny_matrix_multiplication_": 64}
"""K padded with zeros before the kernel, as the model does (AI-bench's
``pad_k``)."""


@pytest.mark.parametrize("name", list(_MATMUL_MNK))
def test_kernelbench_matmul(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    a, b, ref = _matmul_operands(name, problem.dtype)
    if name in _PAD_K:
        pad = _PAD_K[name] - a.shape[1]
        a, b = F.pad(a, (0, pad)), F.pad(b, (0, 0, 0, pad))
    _check(problem.kernel(a, b), ref)


def test_matmul_k_step_16() -> None:
    """K = 32 in K steps of 16 on 256x256 workgroups (9_Tall_skinny's config
    before its K was padded, ``GEMM_SMALL_K``)."""
    a, b = _randn(512, 32), _randn(32, 512)
    _check(_kernel(kb._matmul, [256, 256, 16])(a, b), a @ b)


def test_small_k_gemm_columns_first_bit_exact() -> None:
    """``GEMM_SMALL_K`` with its columns-first grid (``columns_first``: the
    output dominates) on 32x16 workgroups of 64x128 (2048x2048x64): bit for
    bit ``torch.matmul``, as at the spec shape (the grid order moves no
    arithmetic)."""
    a, b = _randn(2048, 64), _randn(64, 2048)
    kernel = _kernel(kb._matmul, kb.GEMM_SMALL_K[0], **kb.GEMM_SMALL_K[1])
    _check(kernel(a, b), a @ b, bound=0.0)


def test_split_k_gemm() -> None:
    """``GEMM_SPLIT_K`` (6_large_K: 32 K chunks of the strided A view, f32
    partial sums, ``_sum_splits``) at 256x256x32768: as close to the f32
    result as ``torch.matmul`` (one bf16 rounding of a sum in another order)."""
    splits, blocks, xegpu, reduce_blocks = kb.GEMM_SPLIT_K
    a, b = _randn(256, 32768), _randn(32768, 256)
    gemm = _kernel(kb._matmul_split_k, blocks, **xegpu)
    reduce = _kernel(kb._sum_splits, reduce_blocks)
    parts = gemm(*kb.split_k_views(a, b, splits))
    ref32 = a.float() @ b.float()
    # The f32 partial sums against the f32 chunk products.
    chunks = [
        a[:, i * 1024 : (i + 1) * 1024].float() @ b[i * 1024 : (i + 1) * 1024].float()
        for i in range(splits)
    ]
    _check(parts, torch.stack(chunks), bound=1e-5)
    out = reduce(parts)
    _check(out, ref32)
    assert normalized_error(out, ref32) <= 2 * normalized_error(a @ b, ref32) + 2**-9


_GEMM_AT_SCALE = {
    "1_Square_matrix_multiplication_": (1024, 1024, 1024),
    "13_Matmul_for_symmetric_matrices": (1024, 1024, 1024),
    "2_Standard_matrix_multiplication_": (512, 1024, 8192),
}
"""The hand-picked GEMMs on a longer K loop and more workgroups than the small
shapes: 1 and 13 are 8x2 = 16 workgroups of 128x512 (exactly one wave on the
B50's 16 Xe cores), 2 is 2x4 = 8 workgroups of 256x256 with its spec K."""
_BIT_EXACT_VS_TORCH = {
    "1_Square_matrix_multiplication_",
    "13_Matmul_for_symmetric_matrices",
}
"""Verified bit for bit against ``torch.matmul`` at the shapes above (two seeds).
2 is not: at 512x1024x8192 torch picks another kernel (and K order) than the
spec shape's, so it is compared with the same kernel under the inferred config."""


@pytest.mark.parametrize("name", list(_GEMM_AT_SCALE))
def test_kernelbench_hand_picked_gemm_at_scale(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    m, n, k = _GEMM_AT_SCALE[name]
    a, b = _randn(m, k), _randn(k, n)
    out = problem.kernel(a, b)
    if name in _BIT_EXACT_VS_TORCH:
        _check(out, a @ b, bound=0.0)
        return
    _check(out, a @ b)
    # The same K step, so the same accumulation order: the config changes
    # the tiling and the loads, not a bit of the result.
    inferred = _kernel(problem.fn, problem.block_sizes)(a, b)
    _check(out, inferred, bound=0.0)


_EXACT = {"19_ReLU", "32_HardTanh", "19_ReLU_NCHW"}
"""Problems computing an exact op (no rounding): bit for bit."""
_GUARDED_ELEMENTWISE = {
    "22_Tanh": r"math\.tanh rounded to bf16",
    "30_Softsign": r"math\.absf of a bf16 value",
}
"""Problems the XPU target rejects (``xpu_checks.check_math``): computed on the
device they are wrong (22_Tanh: zeros) or not separable from wrong forms
(30_Softsign's abs of bf16 values)."""


def _poison_output(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """NaN-fill and free a block of the output's size: the caching allocator
    hands it to the next allocation of that size (the kernel's output), so an
    element the kernel does not write stays NaN (``_check`` rejects it).
    Returns the block's address, for the caller to check the output got it."""
    poison = torch.full(shape, float("nan"), dtype=dtype, device=DEVICE)
    address = poison.data_ptr()
    del poison
    return address


@pytest.mark.parametrize("name", list(_ELEMENTWISE_REFERENCES))
def test_kernelbench_elementwise(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    # N(0, 3): both sides of every kink (hardtanh, hardsigmoid, ELU, SELU).
    x = _randn(*_ELEMENTWISE_SHAPE) * 3
    bound = 0.0 if name in _EXACT else None
    if name in _GUARDED_ELEMENTWISE:
        with pytest.raises(UnsupportedOperationError, match=_GUARDED_ELEMENTWISE[name]):
            problem.kernel(x)
        return
    poisoned = _poison_output(x.shape, x.dtype)
    out = problem.kernel(x)
    assert out.data_ptr() == poisoned
    _check(out, _ELEMENTWISE_REFERENCES[name](x), bound)


def _relu_to_f32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.shape):
        out[tile] = torch.relu(x[tile].to(torch.float32))
    return out


def _relu_to_bf16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for tile in hl.tile(x.shape):
        out[tile] = torch.relu(x[tile].to(torch.float32)).to(torch.bfloat16)
    return out


@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [(F32, F32), (BF16, F32), (F32, BF16)],
    ids=["f32", "bf16-to-f32", "f32-to-bf16"],
)
def test_one_row_elementwise_tiles(dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    """One-row tiles (``ELEMENTWISE_BLOCKS``) of other element types: 1D
    gathers and scatters of 16 bytes of the widest element per work item, the
    same chunk for the loads and the stores (4: a bf16 load chunk of 8 under
    f32 stores of 4 would be miscompiled by upstream), columns first."""
    x = _randn(*_ELEMENTWISE_SHAPE, dtype=dtype) * 3
    poisoned = _poison_output(x.shape, out_dtype)
    fn = _relu_to_f32 if out_dtype == F32 else _relu_to_bf16
    out = _kernel(fn, kb.ELEMENTWISE_BLOCKS)(x)
    assert out.dtype == out_dtype
    assert out.data_ptr() == poisoned
    _check(out, torch.relu(x.float()), 0.0)


def _relu_1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.relu(x[tile])
    return out


def _relu_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.relu(x[tile])
    return out


def _relu_in_place_2d(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        x[tile] = torch.relu(x[tile])
    return x


@pytest.mark.parametrize(
    ("fn", "block_sizes", "shape"),
    [
        (_relu_1d, [8192], (16 * 16384,)),
        (_relu_2d, kb.ELEMENTWISE_BLOCKS, _ELEMENTWISE_SHAPE),
        (_relu_2d, [128, 256], (256, 512)),
    ],
    ids=["1d", "one-row", "2d"],
)
def test_misaligned_views_are_passed_aligned(
    fn: Callable[..., torch.Tensor], block_sizes: list[int], shape: tuple[int, ...]
) -> None:
    """A bf16 view 2 bytes past an allocation (``buf[1:]``): the target passes
    a 64-byte aligned copy (``XpuTarget.buffer_alignment``). Regression: the
    1D gathers and scatters of 8 elements per work item (vector loads of ABI
    alignment 16) computed 75 % of the outputs wrong, and the 2D block loads
    (64-byte aligned base) too."""
    buf = _randn(math.prod(shape) + 1)
    x = buf[1:].view(shape)
    assert x.data_ptr() % 64 == 2
    _check(_kernel(fn, block_sizes)(x), torch.relu(x), 0.0)


def test_misaligned_inout_is_staged_and_copied_back() -> None:
    """An in-place kernel on a misaligned view: the aligned copy it writes is
    copied back into the view; the elements around it keep their values."""
    buf = _randn(math.prod(_ELEMENTWISE_SHAPE) + 2)
    expected = buf.clone()
    expected[1:-1] = torch.relu(expected[1:-1])
    x = buf[1:-1].view(_ELEMENTWISE_SHAPE)
    _kernel(_relu_in_place_2d, kb.ELEMENTWISE_BLOCKS)(x)
    _check(buf, expected, 0.0)


_MATH_DTYPES = {"bf16": BF16, "f16": F16, "f32": F32}
_KNOWN_WRONG: dict[str, set[tuple[str, str]]] = {
    **{
        name: {(i, o) for i in _MATH_DTYPES for o in ("bf16", "f16")}
        for name in (
            "exp2",
            "log",
            "log2",
            "log1p",
            "expm1",
            "sin",
            "cos",
            "tanh",
            "pow",
        )
    },
    **{
        name: {("bf16", o) for o in _MATH_DTYPES} | {("f32", "bf16")}
        for name in ("abs", "abs_raw")
    },
    "abs_p1": {("bf16", "bf16"), ("bf16", "f16")},
    "abs_x2": {("bf16", "bf16"), ("bf16", "f16")},
    "atan": {("bf16", o) for o in _MATH_DTYPES},
    # The reviewer's rows (bf16 and f16 inputs measured).
    **{
        name: {(i, o) for i in ("bf16", "f16") for o in ("bf16", "f16")}
        for name in ("tan", "acos", "atanh", "sinh")
    },
    "cosh": {(i, o) for i in ("bf16", "f16") for o in ("bf16", "f16")}
    | {("bf16", "f32")},
    **{
        name: {("bf16", o) for o in _MATH_DTYPES}
        for name in ("abs_neg", "abs_mul1", "atan_mul1")
    },
}
"""``(input, output)`` dtypes each kernel computed wrong on the B50 without the
math check (llvm-project 79ab4054, /swtools/intel-gpu/latest IGC): these must
be rejected."""
# Rounded f32 math: half a 16-bit ulp of the output twice over; f32 native
# functions (``native_exp``, OpenCL tanh) are accurate to a few f32 ulps.
_MATH_BOUND = {BF16: 2**-7, F16: 2**-10, F32: 1e-4}


@pytest.mark.parametrize("name", list(mk.EXPRS))
def test_math_functions(name: str) -> None:
    """``out = f(x.float())`` for every input and output dtype: right, or
    rejected by the XPU math check; the forms known to compute wrong values on
    the device are rejected."""
    kernel = getattr(mk, f"k_{name}")
    rejected = set()
    for (i, in_dtype), (o, out_dtype) in itertools.product(
        _MATH_DTYPES.items(), _MATH_DTYPES.items()
    ):
        x = (torch.rand(256, 512, device=DEVICE) * 4 - 2).to(in_dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=DEVICE)
        v, tile = x.float(), slice(None)  # noqa: F841 - read by the expression
        ref = eval(mk.EXPRS[name])
        try:
            result = kernel(x, out)
        except UnsupportedOperationError:
            rejected.add((i, o))
            continue
        error = normalized_error(result, ref)
        assert error <= _MATH_BOUND[out_dtype], f"{name} {i}->{o}: {error:.3g}"
    assert _KNOWN_WRONG.get(name, set()) <= rejected, rejected


@pytest.mark.parametrize("name", list(_REDUCTION_REFERENCES))
def test_kernelbench_reduction(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    x = _randn(*_REDUCTION_SHAPE, dtype=F32)
    _check(problem.kernel(x), _REDUCTION_REFERENCES[name](x))


_RANK3_CASES: dict[
    str, tuple[tuple[tuple[int, ...], ...], Callable[..., torch.Tensor]]
] = {
    "3_Batched_matrix_multiplication": (((2, 256, 256), (2, 256, 512)), torch.bmm),
    "10_3D_tensor_matrix_multiplication": (
        ((2, 512, 256), (256, 512)),
        lambda a, b: torch.einsum("bmk,kn->bmn", a, b),
    ),
    "19_ReLU_NCHW": (((2, 2, 256, 512),), torch.relu),
    "23_Softmax_last_dim_4D": (((2, 2, 128, 256),), lambda x: torch.softmax(x, -1)),
    "97_ScaledDotProductAttention": (
        ((2, 2, 256, 64),) * 3,
        F.scaled_dot_product_attention,
    ),
}


@pytest.mark.parametrize("name", list(_RANK3_CASES))
def test_kernelbench_rank3(name: str) -> None:
    problem = {p.name: p for p in kb.RANK3_PROBLEMS}[name]
    shapes, reference = _RANK3_CASES[name]
    args = [_randn(*shape, dtype=problem.dtype) for shape in shapes]
    bound = 0.0 if name in _EXACT else None
    if problem.kind == "attention":
        # P is rounded to bf16 for P @ V (as eager's flash attention does), then
        # the output: a few bf16 ulps.
        bound = 2**-6
    _check(problem.kernel(*args), reference(*args), bound)


_LEVEL2_MNK = {"59_Matmul_Swish_Scaling": (128, 512, 256)}


@pytest.mark.parametrize("dtype", [BF16, F16], ids=["bf16", "f16"])
@pytest.mark.parametrize("name", [p.name for p in kb.LEVEL2_PROBLEMS])
def test_kernelbench_level2(name: str, dtype: torch.dtype) -> None:
    """``epilogue(x @ weight.T + bias)``: bias row and epilogue in f32 on the
    accumulator, as the KernelBench model in f32 from the same inputs."""
    problem = {p.name: p for p in kb.LEVEL2_PROBLEMS}[name]
    kernel = _kernel(problem.fn, problem.block_sizes, **problem.xegpu)
    m, n, k = _LEVEL2_MNK.get(name, (512, 512, 256))
    x = _randn(m, k, dtype=dtype)
    weight = (torch.rand(n, k, dtype=dtype, device=DEVICE) * 2 - 1) / 4
    bias = torch.rand(n, dtype=dtype, device=DEVICE) * 2 - 1
    out = kernel(*problem.args(x, weight, bias))
    assert out.dtype == dtype
    linear = F.linear(x.float(), weight.float(), bias.float())
    _check(out, level2_reference(name)(linear, *problem.scalars))


# --- bf16 rounding (xevm_bf16_extensions) --------------------------------------

_BF16_SPECIALS = [
    float("inf"),
    float("-inf"),
    float("nan"),
    0.0,
    -0.0,
    1.0,
    -1.0,
    3.0,
    3.3895313892515355e38,  # the largest bf16
    1.1754943508222875e-38,  # the smallest normal
    9.183549615799121e-41,  # the smallest subnormal
    -2.5,
    0.5,
    1.0078125,
    -7.0,
    65280.0,
]


def _bf16_operands() -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 operands: N(0, 1), the first row starting with special values
    (infinities, NaN, signed zeros, the extremes, subnormals) against each
    other, so overflow, NaN propagation and exact results are covered too."""
    x, y = _randn(256, 512), _randn(256, 512)
    specials = torch.tensor(_BF16_SPECIALS, dtype=BF16, device=DEVICE)
    n = len(_BF16_SPECIALS)
    x[0, : n * n] = specials.repeat_interleave(n)
    y[0, : n * n] = specials.repeat(n)
    return x, y


def _assert_bits(out: torch.Tensor, ref: torch.Tensor) -> None:
    """``out`` is ``ref`` bit for bit, NaNs where ``ref`` has NaNs (any NaN):
    the kernel wrote every element of its NaN-filled output."""
    assert out.dtype == ref.dtype == BF16
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), nan)
    bits, expected = out.view(torch.int16)[~nan], ref.view(torch.int16)[~nan]
    wrong = (bits != expected).sum().item()
    assert wrong == 0, f"{wrong} of {out.numel()} elements differ from torch"


@pytest.mark.parametrize("op", list(bf16_kernels.BF16_OPS))
def test_bf16_arithmetic_rounds_to_nearest_even(op: str) -> None:
    """One rounding of f32 math on bf16 values (``x + y`` of bf16 tensors, and
    ``(x.float() + y.float()).to(bf16)``) is torch's, bit for bit, in one-row
    and 2D tiles. Regression: IGC shrank the f32 add, sub and mul between the
    two conversions to bf16 arithmetic, which truncates: 27 % of adds and 47 %
    of muls were one ulp off (``xevm_bf16_extensions``)."""
    bf16_fn, f32_fn, reference = bf16_kernels.BF16_OPS[op]
    x, y = _bf16_operands()
    ref = reference(x, y)
    for block_sizes in ([1, 256], [32, 256]):
        for fn in (bf16_fn, f32_fn):
            out = torch.full_like(x, float("nan"))
            bf16_kernels.bf16_kernel(fn, block_sizes)(x, y, out)
            _assert_bits(out, ref)


def test_bf16_scalar_multiplication_rounds_to_nearest_even() -> None:
    """KernelBench level 1 ``5_Matrix_scalar_multiplication`` in bf16 (``A *
    s``, ``s`` = 3.14 rounded to bf16): torch's result bit for bit (it was
    truncated, see above)."""
    a, _ = _bf16_operands()
    out = torch.full_like(a, float("nan"))
    bf16_kernels.scalar_mul(a, 3.140625, out)
    _assert_bits(out, a * 3.140625)


def test_bf16_division_rounds_to_nearest_even() -> None:
    """bf16 division (not shrunk by IGC, so right before the workaround too) is
    torch's bit for bit on finite operands; f32 division of special values
    (subnormals, infinities) differs from torch's on the device, so these are
    left out."""
    x, y = _randn(256, 512), _randn(256, 512)
    ref = x / y
    for fn in (bf16_kernels._bf16_div, bf16_kernels._f32_div):
        out = torch.full_like(x, float("nan"))
        bf16_kernels.bf16_kernel(fn, [32, 256])(x, y, out)
        _assert_bits(out, ref)


# --- Backend kernels of the IR tests (tests/test_xpu_ir.py) --------------------


@pytest.mark.parametrize("dtype", [F16, BF16], ids=["f16", "bf16"])
def test_gemm_f32_output(dtype: torch.dtype) -> None:
    a, b = _randn(512, 256, dtype=dtype), _randn(256, 512, dtype=dtype)
    out = ir_kernels.matmul(a, b)
    assert out.dtype == F32
    _check(out, a.float() @ b.float(), 1e-5)


def test_gemm_relu_epilogue() -> None:
    a, b = _randn(512, 256, dtype=F16), _randn(256, 512, dtype=F16)
    _check(ir_kernels.matmul_relu(a, b), torch.relu(a.float() @ b.float()), 1e-5)


@pytest.mark.parametrize("dtype", [F16, BF16], ids=["f16", "bf16"])
def test_gemm_runtime_scalar(dtype: torch.dtype) -> None:
    a, b = _randn(512, 256, dtype=dtype), _randn(256, 512, dtype=dtype)
    ref = a.float() @ b.float()
    _check(ir_kernels.matmul_scaled(a, b, 0.5), ref * 0.5, 1e-5)
    # Another value of the same compiled kernel: scalars are passed per call.
    _check(ir_kernels.matmul_scaled(a, b, -3.0), ref * -3.0, 1e-5)


def test_gemm_transposed_b() -> None:
    a, b = _randn(512, 256, dtype=F16), _randn(512, 256, dtype=F16)
    _check(ir_kernels.matmul_transposed_b(a, b), a.float() @ b.float().T, 1e-5)


def test_two_phase_gemm() -> None:
    """``hl.barrier`` phases: two launches in order; the second reads the
    first's f16 result."""
    a, b, c = (_randn(512, 512, dtype=F16) / 4 for _ in range(3))
    tmp = (a.float() @ b.float()).half()
    _check(ir_kernels.two_phase_matmul(a, b, c), tmp.float() @ c.float(), 1e-3)


def _addmm_c(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``c + a @ b``: the accumulator starts from the existing C tile."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=c.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = c[tm, tn].to(torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(c.dtype)
    return out


def _addmm_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``a @ b + bias`` with a bias row added after the K loop."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = (acc + bias[tn][None, :].to(torch.float32)).to(a.dtype)
    return out


_F32_C = pytest.mark.xfail(
    strict=True,
    reason="open: a C tile of the accumulator's dtype (f32) read as the K loop's "
    "init is not writable, so bufferization copies it into a gpu.alloc inside the "
    "kernel, which the XeVM lowering cannot call ('mgpuMemAlloc' does not "
    "reference a symbol)",
)


@pytest.mark.parametrize(
    "dtype",
    [F16, BF16, pytest.param(F32, marks=_F32_C)],
    ids=["f16", "bf16", "f32"],
)
def test_gemm_accumulates_into_existing_c(dtype: torch.dtype) -> None:
    """A GEMM reading an existing C (lighthouse's numpy example path drops C):
    C is not lost. Regression: C's tile is block-read in the kernel's entry
    block, into an alloca the SPIR-V backend sized to 4 bytes (zeros) before
    ``xevm_alloca_arrays``."""
    kernel = _kernel(_addmm_c, [256, 256, 32])
    a, b = _randn(512, 256, dtype=BF16), _randn(256, 512, dtype=BF16)
    c = _randn(512, 512, dtype=dtype) * 16
    out = kernel(c, a, b)
    ref = c.float() + a.float() @ b.float()
    _check(out, ref)
    # Without C the error would be about max|c| / max|ref|.
    assert normalized_error(out, a.float() @ b.float()) > 0.1


@pytest.mark.parametrize("dtype", [F16, BF16], ids=["f16", "bf16"])
def test_gemm_bias_row(dtype: torch.dtype) -> None:
    kernel = _kernel(_addmm_bias, [256, 256, 32])
    a, b = _randn(512, 256, dtype=dtype), _randn(256, 512, dtype=dtype)
    bias = _randn(512, dtype=dtype) * 8
    _check(kernel(a, b, bias), a.float() @ b.float() + bias.float())


def test_elementwise_kernels() -> None:
    x = _randn(256, 512, dtype=F32)
    _check(ir_kernels.relu(x), torch.relu(x), 0.0)
    xb = _randn(256, 512)
    _check(
        ir_kernels.gelu_tanh_cast(xb), (torch.tanh(xb) * torch.exp(xb)).float(), 2**-7
    )
    y = _randn(256, 512)
    first, second = ir_kernels.two_in_two_out(xb, y)
    s = torch.sigmoid(xb + y)
    _check(first, s)
    _check(second, torch.exp(s) * xb)
    bias = _randn(512)
    _check(ir_kernels.bias_relu(xb, bias), torch.relu(xb + bias[None, :]))
    v = _randn(8192, dtype=F32)
    _check(ir_kernels.sigmoid_1d(v), torch.sigmoid(v))
    scaled = ir_kernels.scale_shift(x, 0.5, 3)
    _check(scaled, x * 0.5 + 3)


@pytest.mark.parametrize("dtype", [F32, BF16, F16], ids=["f32", "bf16", "f16"])
def test_reduction_kernels(dtype: torch.dtype) -> None:
    """Row results stored as rows (``[rows]``, regression: pairs of equal wrong
    values before ``keep_row_dims``) and as a ``keepdim`` column; 16-bit tiles
    (regression: 32-column loads gave wrong sums)."""
    x = _randn(128, 256, dtype=dtype)
    _check(ir_kernels.softmax(x), torch.softmax(x.float(), -1))
    _check(ir_kernels.row_sums(x), x.float().sum(-1))
    _check(ir_kernels.row_maxima(x), x.amax(-1, keepdim=True), 0.0)


_RANK3_KERNELS: dict[
    str,
    tuple[
        Callable[..., torch.Tensor],
        list[int],
        list[tuple[int, ...]],
        torch.dtype,
        Callable[..., torch.Tensor],
    ],
] = {
    "relu_3d": (ir_kernels.relu_nd, [1, 128, 256], [(2, 256, 512)], BF16, torch.relu),
    "relu_4d": (
        ir_kernels.relu_nd,
        [1, 1, 128, 256],
        [(2, 3, 256, 256)],
        F32,
        torch.relu,
    ),
    "batched_matmul": (
        ir_kernels.batched_matmul,
        [1, 256, 256, 32],
        [(2, 512, 256), (2, 256, 512)],
        F16,
        lambda a, b: a.float() @ b.float(),
    ),
    "tensor_matmul": (
        ir_kernels.tensor_matmul,
        [1, 256, 256, 32],
        [(2, 512, 256), (256, 512)],
        BF16,
        lambda a, b: a.float() @ b.float(),
    ),
    "head_row_sums": (
        ir_kernels.head_row_sums,
        [64, 1, 1, 64],
        [(2, 4, 128, 256)],
        F32,
        lambda x: x.sum(-1, keepdim=True),
    ),
}


@pytest.mark.parametrize("name", list(_RANK3_KERNELS))
def test_rank3_kernels(name: str) -> None:
    fn, block_sizes, shapes, dtype, reference = _RANK3_KERNELS[name]
    args = [_randn(*shape, dtype=dtype) for shape in shapes]
    out = _kernel(fn, block_sizes)(*args)
    bound = 0.0 if name.startswith("relu") else 1e-5 if out.dtype == F32 else None
    _check(out, reference(*args), bound)


def test_attention_other_blocks() -> None:
    """97's flash attention at other block sizes: 256 query rows, 32-row K/V
    tiles (K stored transposed is outside the attention form, test_xpu_ir)."""
    q, k, v = (_randn(2, 2, 512, 64) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    _check(_kernel(kb.attention, [1, 1, 256, 32])(q, k, v), ref, 2**-6)


# --- Runtime: USM interop, synchronization, device index, caching ----------------


def _spy_entry_arguments() -> tuple[list[list[object]], object]:
    """A patch of ``CompiledEntry.__call__`` recording the values it passes."""
    calls: list[list[object]] = []
    original = execution.CompiledEntry.__call__

    def spy(self: execution.CompiledEntry, values: list[object]) -> None:
        calls.append(list(values))
        original(self, values)

    return calls, mock.patch.object(execution.CompiledEntry, "__call__", spy)


def test_usm_pointers_go_straight_to_the_kernel() -> None:
    """XPU tensors (contiguous, also a view at a storage offset) reach the entry
    as themselves: their USM pointers in the descriptors, no host copies."""
    storage = _randn(1024, 256, dtype=F16)
    a = storage[512:]  # a contiguous view at an offset
    b = _randn(256, 512, dtype=F16)
    calls, patch = _spy_entry_arguments()
    with (
        patch,
        mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("cpu()")),
        mock.patch.object(torch.Tensor, "numpy", side_effect=AssertionError("numpy()")),
    ):
        out = ir_kernels.matmul(a, b)
    (values,) = calls
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    assert all(t.device.type == "xpu" for t in tensors)
    pointers = {t.data_ptr() for t in tensors}
    assert a.data_ptr() in pointers and b.data_ptr() in pointers
    assert out.data_ptr() in pointers
    assert a.data_ptr() == storage.data_ptr() + 512 * 256 * 2
    _check(out, a.float() @ b.float(), 1e-5)


def _relu_in_place(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        x[tile] = torch.relu(x[tile])
    return x


def test_in_place_update_of_a_torch_tensor() -> None:
    """The kernel writes the caller's XPU tensor in place, through its pointer."""
    x = _randn(256, 512)
    expected = torch.relu(x)
    pointer = x.data_ptr()
    out = _kernel(_relu_in_place, [128, 256])(x)
    assert x.data_ptr() == pointer
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    assert out.data_ptr() == pointer


def test_strided_input_is_staged_on_the_device() -> None:
    """A transposed view is made contiguous on the XPU (a device copy)."""
    base = _randn(512, 256, dtype=F16)
    b = _randn(256, 512, dtype=F16)
    a = base.T.contiguous().T  # strided (256 x 512 stored transposed)
    assert not a.is_contiguous()
    calls, patch = _spy_entry_arguments()
    with patch:
        out = ir_kernels.matmul(a, b)
    assert all(v.device.type == "xpu" for v in calls[0] if isinstance(v, torch.Tensor))
    _check(out, a.float() @ b.float(), 1e-5)


def test_torch_ops_consume_the_output_immediately() -> None:
    """The entry synchronizes its stream before returning, and waits for
    torch's queue first: torch ops right before and after see consistent data."""
    a, b = _randn(512, 256, dtype=F16), _randn(256, 512, dtype=F16)
    for _ in range(3):
        # Produced by torch just before the call (not synchronized by the test).
        a2 = a * 2
        out = ir_kernels.matmul(a2, b)
        # Consumed by torch right after, then by another kernel.
        doubled = out * 0.5 + 1.0
        again = ir_kernels.relu(doubled)
        ref = (a.float() @ b.float()) + 1.0
        _check(doubled, ref, 1e-5)
        _check(again, torch.relu(ref), 1e-5)


def test_kernel_chain_without_host_sync() -> None:
    """Outputs feed the next kernel directly (no torch op between)."""
    a, b = _randn(512, 256, dtype=F16), _randn(256, 512, dtype=F16)
    out = ir_kernels.relu(ir_kernels.matmul(a, b))
    _check(out, torch.relu(a.float() @ b.float()), 1e-5)


def test_device_index() -> None:
    """``xpu:0`` named explicitly runs on the runtime's default device; tensors
    on two devices are rejected before the launch."""
    a = torch.randn(512, 256, dtype=F16, device="xpu:0")
    b = torch.randn(256, 512, dtype=F16, device="xpu:0")
    out = ir_kernels.matmul(a, b)
    assert out.device == torch.device("xpu:0")
    _check(out, a.float() @ b.float(), 1e-5)
    if torch.xpu.device_count() < 2:
        return
    with torch.xpu.device(1):
        a1, b1 = a.to("xpu:1"), b.to("xpu:1")
        out1 = ir_kernels.matmul(a1, b1)
    assert out1.device == torch.device("xpu:1")
    _check(out1, a1.float() @ b1.float(), 1e-5)
    with pytest.raises(ValueError, match="several devices"):
        XpuTarget().prepare_call([a, b1])


def test_repeated_calls_reuse_the_jit_cache() -> None:
    """Calls with new inputs of the same shapes reuse the compiled kernel (one
    JIT); each call's result is right."""
    kernel = _kernel(ir_kernels.matmul_relu.fn, [256, 256, 32])
    a, b = _randn(512, 256, dtype=F16), _randn(256, 512, dtype=F16)
    with mock.patch.object(
        execution, "_compile_entry", wraps=execution._compile_entry
    ) as compile_entry:
        for step in range(4):
            a2, b2 = a * (step + 1), b + step / 4
            _check(kernel(a2, b2), torch.relu(a2.float() @ b2.float()), 1e-5)
    assert compile_entry.call_count <= 1
    # Another kernel object with the same source and config hits the in-process
    # JIT cache (equal module text).
    again = _kernel(ir_kernels.matmul_relu.fn, [256, 256, 32])
    with mock.patch.object(
        execution, "_compile_entry", wraps=execution._compile_entry
    ) as compile_entry:
        _check(again(a, b), torch.relu(a.float() @ b.float()), 1e-5)
    assert compile_entry.call_count == 0


def test_hardware_name_is_the_gpu() -> None:
    assert XpuTarget().hardware_name() == torch.xpu.get_device_name()
    XpuTarget().validate_environment(None)


def _unescape(text: str) -> bytes:
    """The bytes of an MLIR string literal's body."""
    out = bytearray()
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            pair = text[i + 1 : i + 3]
            if len(pair) == 2 and all(c in string.hexdigits for c in pair):
                out.append(int(pair, 16))
                i += 3
                continue
            out.append(ord(text[i + 1]))
            i += 2
            continue
        out.append(ord(text[i]))
        i += 1
    return bytes(out)


def _objects(text: str) -> list[list[bytes]]:
    """Per ``gpu.binary``, the device binaries (zebin ELF bytes) of its objects."""
    binaries = []
    for line in text.splitlines():
        if not line.lstrip().startswith("gpu.binary"):
            continue
        blobs = re.findall(r'"(\\7FELF(?:[^"\\]|\\.)*)"', line)
        binaries.append([_unescape(blob) for blob in blobs])
    return binaries


def test_one_binary_per_kernel() -> None:
    """Each GPU module compiles to one object, for its one XeVM target (two
    targets compiled every kernel twice)."""
    args = [_randn(512, 512, dtype=F16) / 4 for _ in range(3)]
    text = str(lower_mlir(ir_kernels.two_phase_matmul, args, stop_at="binary"))
    binaries = _objects(text)
    assert len(binaries) == 2, text[:2000]
    assert [len(objects) for objects in binaries] == [1, 1]


def test_gemm_binary_has_the_large_register_file() -> None:
    """1_Square's GEMM (32x64 subgroup tiles of f32 accumulators) is compiled
    with the large register file (``.ze_info``: 256 GRFs) and does not spill.
    Without ``gpu-module-to-binary``'s ``opts`` it gets 128 GRFs and spills
    (7 KB; 1_Square drops from 33 to 7 TFLOPS)."""
    problem = {p.name: p for p in kb.PROBLEMS}["1_Square_matrix_multiplication_"]
    args = [_randn(512, 256), _randn(256, 512)]
    ((zebin,),) = _objects(str(lower_mlir(problem.kernel, args, stop_at="binary")))
    assert zebin.startswith(b"\x7fELF")
    assert re.findall(rb"grf_count:\s*(\d+)", zebin) == [b"256"], zebin[-2000:]
    assert not re.search(rb"spill_size:\s*[1-9]", zebin)
