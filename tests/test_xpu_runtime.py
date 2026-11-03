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
from tests import test_nan_minmax as nan_kernels
from tests import test_xevm_bf16_extensions as bf16_kernels
from tests import test_xpu_ir as ir_kernels
from tests import xpu_math_full_kernels as full_kernels
from tests import xpu_math_kernels as mk
from tests.harness import load_example
from tests.test_xpu_kernelbench import _reference as level2_reference

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.targets import xpu as xpu_target
from helion_mlir_backend._compiler.targets.xpu import XpuTarget

if TYPE_CHECKING:
    from collections.abc import Callable
    import pathlib

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
# Rounded f32 math: half a 16-bit ulp of the output twice over; f32 native
# functions (``native_exp``, OpenCL tanh) are accurate to a few f32 ulps.
_MATH_BOUND = {BF16: 2**-7, F16: 2**-10, F32: 1e-4}


@pytest.mark.parametrize("name", list(mk.EXPRS))
def test_math_functions(name: str) -> None:
    """``out = f(x.float())`` on x in [-2, 2], error normalized by ``max|ref|``, for
    every input and output dtype, in 2D tiles: compound forms (GELU, Mish, shared
    operands) and none rejected by the XPU math check. IGC computed most of the
    16-bit forms wrong before ``EnableVectorizer=0``, ``xevm_bf16_extensions``
    and ``xevm_fabs`` (zeros, garbage or ``abs`` returning its operand). Full ranges,
    1D tiles and the input's own dtype: :func:`test_math_every_pattern`."""
    kernel = getattr(mk, f"k_{name}")
    for (i, in_dtype), (o, out_dtype) in itertools.product(
        _MATH_DTYPES.items(), _MATH_DTYPES.items()
    ):
        x = (torch.rand(256, 512, device=DEVICE) * 4 - 2).to(in_dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=DEVICE)
        v, tile = x.float(), slice(None)  # noqa: F841 - read by the expression
        ref = eval(mk.EXPRS[name])
        result = kernel(x, out)
        error = normalized_error(result, ref)
        assert error <= _MATH_BOUND[out_dtype], f"{name} {i}->{o}: {error:.3g}"


_FULL_TILES = {"1d": ((16, 8192), [1, 8192]), "2d": ((256, 512), [128, 256])}
_FULL_CASES = (
    ("cast", "1d", "bf16", "bf16"),
    ("cast", "1d", "bf16", "f32"),
    ("cast", "1d", "f16", "f16"),
    ("cast", "1d", "f32", "bf16"),
    ("nat", "1d", "bf16", "bf16"),
    ("nat", "1d", "f16", "f16"),
    ("cast", "2d", "bf16", "bf16"),
)
"""The forms of the reviewer's 638-case matrix most likely to differ: 7 of the 22
per function. The rest (f16 in, f32 out, f32 in, f16 out, ... and 2D in full) gave
the same or smaller errors when it ran."""
_FULL_F32_ULPS = {"exp": 60, "softplus": 60}
"""f32 outputs through the fast native ``exp``: up to 60 ulps, as measured; others 5."""
_FULL_F16_EXP_ULPS = 10
"""``exp`` in f16 itself (native): 10 ulps at x = 10.375; others 1. Not attributed."""


def _full_patterns(dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    """Every 16-bit pattern of ``dtype``, repeated; f32: every bf16 value, then
    random bit patterns."""
    n = shape[0] * shape[1]
    bits = (torch.arange(n) % 65536).to(torch.int32).to(torch.uint16).view(torch.int16)
    if dtype != F32:
        return bits.reshape(shape).view(dtype).clone()
    values = bits[:65536].view(BF16).float()
    generator = torch.Generator().manual_seed(1)
    rest = torch.randint(
        -(2**31), 2**31, (n - 65536,), dtype=torch.int32, generator=generator
    )
    return torch.cat([values, rest.view(F32)]).reshape(shape)


def _ordered(value: torch.Tensor) -> torch.Tensor:
    """Bit patterns in value order, so that the difference counts ulps."""
    size = value.element_size()
    raw = value.view({2: torch.int16, 4: torch.int32}[size]).to(torch.int64)
    return torch.where(raw < 0, -(raw & ((1 << (8 * size - 1)) - 1)), raw)


@pytest.mark.parametrize("name", list(full_kernels.FN))
def test_math_every_pattern(name: str) -> None:
    """Each math function on every bf16 or f16 pattern (±0, ±inf, NaNs, subnormals,
    every magnitude), against fp64 rounded to the output's dtype: NaNs and infinities
    exactly where the reference has them, finite values within 1 ulp of a 16-bit
    output and a few of an f32 one. Rejected on 2ec9e93, or computed wrong."""
    expression = full_kernels.FN[name].format(v="v")
    for kind, tile, i, o in _FULL_CASES:
        shape, blocks = _FULL_TILES[tile]
        case = f"{name} {kind} {tile} {i}->{o}"
        x = _full_patterns(_MATH_DTYPES[i], shape)
        out = torch.empty(shape, dtype=_MATH_DTYPES[o], device=DEVICE)
        fn = getattr(full_kernels, f"{kind}_{name}")
        result = _kernel(fn, blocks)(x.to(DEVICE), out).cpu()
        v = x.to(torch.float64)
        ref = eval(expression, {"torch": torch, "F": F, "v": v}).to(_MATH_DTYPES[o])
        nans = ref.float().isnan()
        assert result.float().isnan().equal(nans), case
        assert result.float().isinf().equal(ref.float().isinf()), case
        ulps = (_ordered(result) - _ordered(ref)).abs()[~nans].max().item()
        if o == "f32":
            bound = _FULL_F32_ULPS.get(name, 5)
        elif (kind, i, name) == ("nat", "f16", "exp"):
            bound = _FULL_F16_EXP_ULPS
        else:
            bound = 1
        assert ulps <= bound, f"{case}: {ulps} ulps"


def _pow_native(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.pow(x[tile], 2.5)
    return out


def _pow_reference(v: torch.Tensor) -> torch.Tensor:
    return torch.pow(v, 2.5)


def _exp2_native(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.exp2(x[tile])
    return out


@pytest.mark.parametrize(
    ("fn", "reference"),
    [
        (_pow_native, _pow_reference),
        (_exp2_native, torch.exp2),
    ],
    ids=["pow", "exp2"],
)
@pytest.mark.parametrize(
    ("shape", "blocks"), list(_FULL_TILES.values()), ids=list(_FULL_TILES)
)
def test_bf16_powf_with_a_constant(fn, reference, shape, blocks) -> None:
    """``powf`` with a constant operand on bf16, no cast (a constant extended to
    f32 stayed at tile size through XeGPU's distribution: an invalid op, reported
    as an internal error): bit for bit the fp64 reference."""
    x = _full_patterns(BF16, shape)
    out = torch.empty(shape, dtype=BF16, device=DEVICE)
    result = _kernel(fn, blocks)(x.to(DEVICE), out)
    _same_bits(result, reference(x.to(torch.float64)).to(BF16))


def _every_bf16_pattern() -> torch.Tensor:
    """All 65536 bf16 bit patterns (±0, ±inf, NaNs, subnormals, every magnitude),
    four times over: the 1D tiles' spec-like shape."""
    bits = torch.arange(_ELEMENTWISE_SHAPE[0] * _ELEMENTWISE_SHAPE[1]) % 65536
    bits = bits.to(torch.uint16).reshape(_ELEMENTWISE_SHAPE)
    return bits.view(torch.int16).to(DEVICE).view(BF16)


def _same_bits(out: torch.Tensor, ref: torch.Tensor) -> None:
    """``out`` and ``ref`` are the same values bit for bit; NaNs of any payload
    and sign are the same value."""
    out, ref = out.cpu(), ref.cpu()
    assert out.isnan().equal(ref.isnan())
    differ = (out.view(torch.int16) != ref.view(torch.int16)) & ~out.isnan()
    assert not differ.any(), (
        f"{differ.sum().item()} patterns differ, first inputs "
        f"{differ.nonzero()[:4].tolist()}"
    )


_BF16_ROUNDED_ONCE = {
    "22_Tanh": torch.tanh,
    "29_Softplus": F.softplus,
    "30_Softsign": lambda v: v / (1 + torch.abs(v)),
}
"""KernelBench problems of the IGC fixes: the fp64 result rounded to bf16 is what
one rounding of f32 math gives. (torch's 30_Softsign rounds ``1 + |x|`` to bf16
first: 1 ulp off this in 1.7 % of the patterns.)"""


@pytest.mark.parametrize("name", list(_BF16_ROUNDED_ONCE))
def test_elementwise_every_bf16_pattern(name: str) -> None:
    """The KernelBench kernel at its elementwise tiles on every bf16 pattern,
    poisoned output: bit for bit the fp64 reference."""
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    x = _every_bf16_pattern()
    poisoned = _poison_output(x.shape, x.dtype)
    out = problem.kernel(x)
    assert out.data_ptr() == poisoned
    _same_bits(out, _BF16_ROUNDED_ONCE[name](x.cpu().to(torch.float64)).to(BF16))


def _abs_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.abs(x[tile].to(F32)).to(x.dtype)
    return out


def test_abs_every_bf16_pattern() -> None:
    """``abs`` rounded straight to bf16 (IGC's no-op ``mov (abs):uw``): only the
    sign bit changes, NaNs included."""
    x = _every_bf16_pattern()
    out = _kernel(_abs_kernel, kb.ELEMENTWISE_BLOCKS)(x)
    _same_bits(out, (x.view(torch.int16) & 0x7FFF).view(BF16))


def _tanh_f32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.tanh(x[tile])
    return out


def _log1p_f32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.log1p(x[tile])
    return out


_F32_SPECIALS = (
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    1e-45,  # the smallest subnormals
    -1e-45,
    1e-39,
    -1e-39,
    1e-38,
    1e-20,
    1e-4,
    3.9e-4,  # tanh's tiny threshold
    4.1e-4,
    0.0078,  # the native log's worst in log1p's Kahan form
    0.5,
    1.0,
    -0.5,
    -1.0,  # log1p: -inf
    -1.5,  # log1p: NaN
    7.9,  # tanh's clamp
    8.1,
    20.0,
    1e30,
    -1e30,
    3.4e38,
    -3.4e38,
)


@pytest.mark.parametrize(
    ("fn", "reference", "bound"),
    [
        (_tanh_f32, torch.tanh, 8),
        (_log1p_f32, torch.log1p, 8),
    ],
    ids=["tanh", "log1p"],
)
def test_approximated_f32_math(fn, reference, bound: int) -> None:
    """``xpu_math_approx`` on f32: special values and random values of every
    exponent, within a few ulps of the fp64 reference (4 for both in a sweep of
    2^32 / 7 patterns), NaNs where it has NaNs, exact at signed zeros and
    subnormals."""
    bits = torch.randint(-(2**31), 2**31, _ELEMENTWISE_SHAPE, dtype=torch.int32)
    x = bits.view(F32).clone()
    specials = torch.tensor(_F32_SPECIALS, dtype=F32)
    x.view(-1)[: len(specials)] = specials
    out = _kernel(fn, kb.ELEMENTWISE_BLOCKS)(x.to(DEVICE)).cpu()
    ref = reference(x.to(torch.float64)).to(F32)
    assert out.isnan().equal(ref.isnan())
    finite = ~ref.isnan()
    assert out[finite].isinf().equal(ref[finite].isinf())
    assert torch.equal(out[finite].signbit(), ref[finite].signbit())

    def ordered(value: torch.Tensor) -> torch.Tensor:
        raw = value.view(torch.int32).to(torch.int64)
        return torch.where(raw < 0, -(raw & 0x7FFFFFFF), raw)

    ulps = (ordered(out[finite]) - ordered(ref[finite])).abs()
    assert ulps.max().item() <= bound, ulps.max().item()
    tiny = finite & (x.abs() < 1e-30)
    assert torch.equal(out[tiny].view(torch.int32), x[tiny].view(torch.int32))


@pytest.mark.parametrize("name", list(_REDUCTION_REFERENCES))
def test_kernelbench_reduction(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    x = _randn(*_REDUCTION_SHAPE, dtype=F32)
    _check(problem.kernel(x), _REDUCTION_REFERENCES[name](x))


def _running_row_maximum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.full([tm], float("-inf"), dtype=x.dtype)
        for tn in hl.tile(n):
            r = torch.maximum(r, x[tm, tn].amax(-1))
        out[tm] = r
    return out


def _running_row_minimum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.full([tm], float("inf"), dtype=x.dtype)
        for tn in hl.tile(n):
            r = torch.minimum(r, x[tm, tn].amin(-1))
        out[tm] = r
    return out


def _running_row_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.zeros([tm], dtype=x.dtype)
        for tn in hl.tile(n):
            r = r + x[tm, tn].sum(-1)
        out[tm] = r
    return out


def _row_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].sum(-1)
    return out


def _row_sum_keepdim(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm, :] = x[tm, :].sum(-1, keepdim=True)
    return out


def _batched_row_maximum(x: torch.Tensor) -> torch.Tensor:
    b, m, _ = x.shape
    out = torch.empty([b, m], dtype=x.dtype, device=x.device)
    for tb, tm in hl.tile([b, m]):
        out[tb, tm] = x[tb, tm, :].amax(-1)
    return out


def _batched_running_row_maximum(x: torch.Tensor) -> torch.Tensor:
    b, m, n = x.shape
    out = torch.empty([b, m], dtype=x.dtype, device=x.device)
    for tb, tm in hl.tile([b, m]):
        r = hl.full([tb, tm], float("-inf"), dtype=x.dtype)
        for tn in hl.tile(n):
            r = torch.maximum(r, x[tb, tm, tn].amax(-1))
        out[tb, tm] = r
    return out


def _run_row_kernel(
    fn: Callable[[torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    blocks: list[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (_randn(*shape, dtype=F32) * 3).to(dtype)
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks),
    )
    return kernel(x), x


@pytest.mark.parametrize("dtype", [F32, BF16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    ("shape", "blocks"),
    [
        # The shapes and block sizes the wrong rows were found with, and a
        # smaller failing one.
        ((256, 512), [64, 64]),
        ((256, 128), [64, 64]),
        ((256, 256), [64, 128]),
        ((128, 128), [64, 64]),
        ((64, 64), [16, 16]),
    ],
    ids=str,
)
def test_running_row_maximum_as_a_vector(
    shape: tuple[int, ...], blocks: list[int], dtype: torch.dtype
) -> None:
    """A row value carried over column tiles, stored as ``[rows]``: bit for bit
    (it gave pairs of equal wrong values, 238 of 256 rows wrong)."""
    out, x = _run_row_kernel(_running_row_maximum, shape, blocks, dtype)
    _check(out, x.amax(-1), bound=0.0)


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize(
    "fn", [_running_row_maximum, _running_row_sum], ids=["maximum", "sum"]
)
def test_running_row_values_in_larger_subgroups(
    fn: Callable[[torch.Tensor], torch.Tensor], rows: int
) -> None:
    """Forced subgroups of 16 and 32 rows: the ``[rows]`` store failed to lower
    (``xevm.blockload2d``), stored as a column it is right."""
    x = _randn(128, 64, dtype=F32) * 3
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 16], xegpu_sg_tiles=[[rows, 0]]),
    )
    ref = x.amax(-1) if fn is _running_row_maximum else x.cpu().double().sum(-1)
    _check(
        kernel(x),
        ref.to(DEVICE, F32),
        bound=0.0 if fn is _running_row_maximum else None,
    )


@pytest.mark.parametrize(
    ("shape", "blocks"),
    [
        ((128, 64), [64, 16]),
        ((256, 256), [64, 128]),
        ((256, 512), [64, 64]),
        ((128, 128), [32, 64]),
        ((64, 64), [32, 32]),
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "fn",
    [_running_row_sum, _row_sum, _row_sum_keepdim],
    ids=["running", "direct", "keepdim"],
)
def test_bf16_row_sums(
    fn: Callable[[torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    blocks: list[int],
) -> None:
    """bf16 sums on the row path: within two ulps of the f64 reference
    (``_BOUND``); they reach 7.65e-3."""
    out, x = _run_row_kernel(fn, shape, blocks, BF16)
    ref = x.cpu().double().sum(-1).to(DEVICE, F32)
    _check(out, ref.reshape(out.shape))


def _running_keepdim_then_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.full([tm, 1], float("-inf"), dtype=x.dtype)
        for tn in hl.tile(n, block_size=16):
            r = torch.maximum(r, x[tm, tn].amax(-1, keepdim=True))
        for tn in hl.tile(n, block_size=16):
            out[tm, tn] = x[tm, tn] - r
    return out


def test_running_keepdim_maximum_then_tile() -> None:
    """A running row maximum broadcast over 2D tiles. Wrong in 7 of 8 rows of
    each subgroup with IGC's vector min/max match (only the first row written);
    the row layouts are right."""
    x = _randn(128, 64, dtype=F32) * 3
    kernel = helion.kernel(
        _running_keepdim_then_tile,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 16]),
    )
    _check(kernel(x), x - x.amax(-1, keepdim=True), bound=1e-5)


def _keepdim_running_then_tile_and_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    rows = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.full([tm, 1], float("-inf"), dtype=x.dtype)
        for tn in hl.tile(n, block_size=16):
            r = torch.maximum(r, x[tm, tn].amax(-1, keepdim=True))
        for tn in hl.tile(n, block_size=16):
            out[tm, tn] = x[tm, tn] - r
        rows[tm] = r.squeeze(-1)
    return out, rows


def _rows_running_then_tile_and_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    rows = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        r = hl.full([tm], float("-inf"), dtype=x.dtype)
        for tn in hl.tile(n, block_size=16):
            r = torch.maximum(r, x[tm, tn].amax(-1))
        rows[tm] = r
        k = r.unsqueeze(-1)
        for tn in hl.tile(n, block_size=16):
            out[tm, tn] = x[tm, tn] - k
    return out, rows


@pytest.mark.parametrize(
    "fn",
    [_keepdim_running_then_tile_and_rows, _rows_running_then_tile_and_rows],
    ids=["keepdim", "vector"],
)
def test_running_row_maximum_stored_and_broadcast(
    fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """A running row maximum stored as a vector and subtracted from 2D tiles
    (112 of 128 rows wrong with IGC's vector min/max match)."""
    x = _randn(128, 64, dtype=F32) * 3
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 16]),
    )
    out, rows = kernel(x)
    _check(rows, x.amax(-1), bound=0.0)
    _check(out, x - x.amax(-1, keepdim=True), bound=1e-5)


def test_running_row_minimum_as_a_vector() -> None:
    out, x = _run_row_kernel(_running_row_minimum, (256, 512), [64, 64], F32)
    _check(out, x.amin(-1), bound=0.0)


def test_running_row_sum_as_a_vector() -> None:
    out, x = _run_row_kernel(_running_row_sum, (256, 512), [64, 64], F32)
    _check(out, x.cpu().double().sum(-1).to(DEVICE, F32))


@pytest.mark.parametrize(
    ("fn", "blocks"),
    [(_batched_row_maximum, [1, 64]), (_batched_running_row_maximum, [1, 64, 16])],
    ids=["direct", "running"],
)
def test_rank3_row_maximum_as_a_matrix(
    fn: Callable[[torch.Tensor], torch.Tensor], blocks: list[int]
) -> None:
    """``out[tb, tm]`` stores rows into a collapsed unit tile of the buffer."""
    out, x = _run_row_kernel(fn, (8, 128, 64), blocks, F32)
    _check(out, x.amax(-1), bound=0.0)


_ROW_REDUCTION_SHAPE = (128, 16384)
"""Sixteen 1024-column segments per row (``row_segments``): 32 workgroups of
partials, two of row results, 2 x 128 one-row output tiles."""


def _row_reference(name: str, x: torch.Tensor) -> torch.Tensor:
    return _REDUCTION_REFERENCES[name](x)


@pytest.mark.parametrize("name", list(kb.ROW_REDUCTIONS))
def test_kernelbench_row_reduction_three_kernels(name: str) -> None:
    """The models' three-kernel row reductions against an f64 reference, as
    close as torch's f32 kernels (normalized error), the output allocated in a
    NaN-filled block (every element written)."""
    x = _randn(*_ROW_REDUCTION_SHAPE, dtype=F32) * 8
    poison = torch.full_like(x, float("nan"))
    del poison
    out = kb.row_reduction(name, x)
    ref = _row_reference(name, x.double())
    _check(out, ref)
    assert (
        normalized_error(out, ref)
        <= 4 * normalized_error(_row_reference(name, x), ref) + 1e-7
    )


@pytest.mark.parametrize("name", list(kb.ROW_REDUCTIONS))
def test_kernelbench_row_reduction_special_values(name: str) -> None:
    """NaN, -inf and +inf inputs, and rows of -inf (all, or one segment): the
    three kernels give NaN and infinities where torch does (the running
    maxima start at -FLT_MAX, ``_segment_softmax``), and close values
    elsewhere."""
    x = _randn(*_ROW_REDUCTION_SHAPE, dtype=F32)
    x[0, 5] = float("nan")
    x[1, :] = float("-inf")
    x[2, 100:200] = float("-inf")
    x[3, 7] = float("inf")
    x[4, :1024] = float("-inf")
    x[5, 9000] = float("nan")
    x[5, 9001] = float("-inf")
    out = kb.row_reduction(name, x)
    ref = _row_reference(name, x)
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    assert torch.equal(torch.isinf(out), torch.isinf(ref))
    finite = torch.isfinite(ref)
    assert torch.equal(
        out[~finite & ~torch.isnan(ref)], ref[~finite & ~torch.isnan(ref)]
    )
    rows = finite.all(1)
    _check(out[rows], ref[rows])


@pytest.mark.parametrize("offset", [1e4, -1e4, 1e6])
@pytest.mark.parametrize("form", ["three_kernels", "single_kernel"])
def test_log_softmax_offset_rows(form: str, offset: float) -> None:
    """Regression (review): log-softmax of rows offset far from 0, against f64.
    ``x - (max + log(sum))`` rounded to the ulp of ``max`` (2.9e-5 of
    ``max|ref|`` at +1e4, 1.9e-3 at 1e6); ``(x - max) - log(sum)`` is as
    close as torch's."""
    x = _randn(*_ROW_REDUCTION_SHAPE, dtype=F32) + offset
    if form == "three_kernels":
        out = kb.row_reduction("24_LogSoftmax", x)
    else:
        out = {p.name: p for p in kb.PROBLEMS}["24_LogSoftmax"].kernel(x)
    ref = torch.log_softmax(x.double(), dim=1)
    _check(out, ref, bound=1e-6)


def test_l1_norm_tiny_inputs() -> None:
    """Regression (review): the three-kernel L1 norm of subnormal inputs
    (``1e-40 * randn``) divides by the row mean, ``x / (sum / n)``; the former
    ``x * (n / sum)`` overflowed ``n / sum`` to inf. (The single-kernel
    fallback still gives inf there: its row mean of subnormals comes out 0 on
    the B50.)"""
    x = _randn(*_ROW_REDUCTION_SHAPE, dtype=F32) * 1e-40
    out = kb.row_reduction("38_L1Norm_", x)
    ref = x.double() / x.double().abs().mean(dim=1, keepdim=True)
    _check(out, ref)


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


@pytest.mark.parametrize("shape", list(ir_kernels.TILE_COPY_SHAPES))
@pytest.mark.parametrize("dtype", [BF16, F16, F32], ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("name", list(ir_kernels.TILE_COPIES))
def test_tile_copies(name: str, dtype: torch.dtype, shape: str) -> None:
    """Kernels storing their tiles unchanged (regression: bf16 ``a * 1.0`` in
    [32, 256] tiles did not lower), and ``a + 0.0`` next to them, into
    NaN-poisoned outputs: bit for bit, NaN, infinities and signed zeros
    included."""
    size = ir_kernels.TILE_COPY_SHAPES[shape][0]
    x = _randn(*size, dtype=dtype)
    flat = x.view(-1)
    flat[:4] = float("nan")
    flat[4:6] = float("inf")
    flat[6:8] = -0.0
    out = ir_kernels.tile_copy_kernel(name, shape)(x, torch.full_like(x, math.nan))
    nan_kernels.assert_same_values(out, x + 0.0 if name == "add0" else x)


def test_clamp_propagates_nan() -> None:
    """Regression (exact kernel and config): -1.0 at NaN inputs."""
    x = nan_kernels.nan_inputs(64, 256, dtype=BF16, device=DEVICE)
    out = nan_kernels.clamp_scaled(x, torch.full_like(x, math.nan))
    nan_kernels.assert_same_values(out, torch.clamp(x * 1.5, -1.0, 1.0))


@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "f32"])
def test_clamp_family_propagates_nan(dtype: torch.dtype) -> None:
    x = nan_kernels.nan_inputs(64, 512, dtype=dtype, device=DEVICE)
    outs = nan_kernels.clamp_family(x)
    for out, ref in zip(outs, nan_kernels.clamp_family_reference(x), strict=True):
        nan_kernels.assert_same_values(out, ref)


@pytest.mark.parametrize("dtype", [BF16, F32], ids=["bf16", "f32"])
def test_binary_minmax_propagates_nan(dtype: torch.dtype) -> None:
    x = nan_kernels.nan_inputs(64, 512, dtype=dtype, device=DEVICE)
    y = nan_kernels.nan_inputs(64, 512, dtype=dtype, device=DEVICE, seed=1)
    y[0, :16] = 1.0  # NaN in x only (row 0), in y only (row 3)
    y[3, :16] = float("nan")
    outs = nan_kernels.binary_minmax(x, y)
    for out, ref in zip(outs, nan_kernels.binary_minmax_reference(x, y), strict=True):
        nan_kernels.assert_same_values(out, ref)


@pytest.mark.parametrize("dtype", [BF16, F16, F32], ids=["bf16", "f16", "f32"])
def test_row_maxima_propagate_nan(dtype: torch.dtype) -> None:
    """Regression: a bf16 row ``amax`` ([64] tiles) of a row with NaN gave the
    row's largest number (lighthouse's ``maxnumf`` reductions). Rows with NaN
    first, inside and last, an all ``-inf`` row and one with ``inf``; also the
    tiled running maximum (``torch.maximum`` in a loop, then ``amax``)."""
    x = _randn(256, 512, dtype=dtype)
    x[0, :16] = float("nan")
    x[1, 3] = float("nan")
    x[2, 511] = float("nan")
    x[3] = -float("inf")
    x[4, 7] = float("inf")
    nan_kernels.assert_same_values(ir_kernels.row_amax(x), x.amax(-1))
    nan_kernels.assert_same_values(ir_kernels.row_maxima(x), x.amax(-1, keepdim=True))


def test_attention_row_maxima_after_other_kernels() -> None:
    """Regression: 97's flash attention (its config, B=H=2) with lighthouse's
    ``maxnumf`` row maxima gave rows of inf/NaN (row 4 of every 16) in calls
    right after some torch kernels (here: ``_check``'s reductions). IGC's vector
    min/max (see ``IGC_OPTIONS``) read their stale registers, so this needs those
    kernels to leave a huge value there; ``test_xpu_igc_workarounds`` has
    deterministic ones."""
    problem = next(p for p in kb.RANK3_PROBLEMS if p.name.startswith("97"))
    kernel = _kernel(problem.fn, problem.block_sizes)
    batch, heads, seq, dim = 2, 2, 512, 64
    generator = torch.Generator().manual_seed(0)
    for big in (30.0, 60.0, 90.0) * 2:
        q = torch.ones(batch, heads, seq, dim, dtype=BF16, device=DEVICE)
        k = (torch.randn(batch, heads, seq, dim, generator=generator) * 0.1).to(BF16)
        v = torch.randn(batch, heads, seq, dim, generator=generator).to(BF16)
        for t in range(seq // 64):
            k[:, :, t * 64 + (t * 7) % 64, :] = big / dim
        k, v = k.to(DEVICE), v.to(DEVICE)
        out = kernel(q, k, v)
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        _check(out, ref, 2**-6)
        assert torch.equal(kernel(q, k, v), out)


def test_device_row_maxima_get_nsz(monkeypatch: pytest.MonkeyPatch) -> None:
    """The binary's row maxima take the cheap form (``xevm_nsz_minmax``)."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")  # lower in-process
    counts: list[int] = []
    flag = xpu_target.nsz_minmax

    def record(module: object) -> int:
        counts.append(flag(module))
        return counts[-1]

    monkeypatch.setattr(xpu_target, "nsz_minmax", record)
    x = _randn(256, 512, dtype=F32)
    nan_kernels.assert_same_values(ir_kernels.row_amax(x), x.amax(-1))
    assert counts and counts[0] > 0


def _nan_rows(*shape: int) -> torch.Tensor:
    """Normal values; NaN first, inside and last in rows 0-2, a NaN next to
    an inf in row 3, a row of -inf (row 4) and an inf (row 5)."""
    x = _randn(*shape, dtype=F32) * 8
    x[0, 0] = x[1, shape[1] // 2 + 3] = x[2, -1] = float("nan")
    x[3, 5], x[3, 6] = float("nan"), float("inf")
    x[4] = -float("inf")
    x[5, 7] = float("inf")
    return x


def _same_nan_pattern(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), finite)
    _check(out[finite], ref[finite])


@pytest.mark.parametrize("name", list(kb.ROW_REDUCTIONS))
def test_row_reductions_propagate_nan(name: str) -> None:
    """The three-kernel row reductions with NaN and infinities in rows: the
    same NaN (and non-finite) pattern as torch, the rest as close."""
    x = _nan_rows(*_ROW_REDUCTION_SHAPE)
    _same_nan_pattern(kb.row_reduction(name, x), _row_reference(name, x))


@pytest.mark.parametrize("name", list(_REDUCTION_REFERENCES))
def test_reductions_propagate_nan(name: str) -> None:
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    x = _nan_rows(*_REDUCTION_SHAPE)
    _same_nan_pattern(problem.kernel(x), _REDUCTION_REFERENCES[name](x))


@pytest.mark.parametrize("dtype", [BF16, F16, F32], ids=["bf16", "f16", "f32"])
def test_reductions_of_mixed_zeros_give_a_zero(dtype: torch.dtype) -> None:
    """``nsz`` (``xevm_nsz_minmax``) gives up the sign of a zero over rows of mixed
    zeros, nothing else: the same values as torch, a NaN row stays NaN."""
    x = _randn(256, 512, dtype=dtype)
    x[6] = 0.0
    x[7] = -0.0
    x[8, ::2], x[8, 1::2] = 0.0, -0.0
    x[9, ::2], x[9, 1::2] = -0.0, 0.0
    x[10, 3] = float("nan")
    out, ref = ir_kernels.row_amax(x).cpu(), x.amax(-1).cpu()
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    keep = ~torch.isnan(ref)
    assert torch.equal(out[keep], ref[keep])  # == does not see the zero's sign


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


def _llama_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A small GQA shape of the example: 8 query heads on 2 K/V heads."""
    return _randn(2, 8, 256, 64), _randn(2, 2, 256, 64), _randn(2, 2, 256, 64)


def _llama_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


def test_llama_gqa_example() -> None:
    """The example at a small shape against torch and an f64 reference."""
    example = load_example("xpu/llama_gqa_attention.py")
    q, k, v = _llama_inputs()
    ref = _llama_torch(q, k, v)
    poison = torch.full(q.shape, float("nan"), dtype=BF16, device=DEVICE)
    del poison
    out = example.llama_attention(q, k, v)
    _check(out, ref, 2**-6)
    exact = _llama_torch(q.cpu().double(), k.cpu().double(), v.cpu().double())
    error = (out.cpu().double() - exact).abs().max() / exact.abs().max()
    torch_error = (ref.cpu().double() - exact).abs().max() / exact.abs().max()
    assert error <= 2 * torch_error, (error, torch_error)
    assert torch.equal(example.llama_attention(q, k, v), out)


def _other_kernels(n: int) -> dict[str, Callable[[], object]]:
    """Kernels to run right before an attention kernel: which of them runs
    selects what stale registers hold (``xpu_transforms.IGC_OPTIONS``), their data
    does not matter."""
    return {
        "none": lambda: None,
        "sum_zero": lambda: torch.zeros(n, device=DEVICE).sum().item(),
        "sum_one": lambda: torch.ones(n, device=DEVICE).sum().item(),
        "sum_huge": lambda: torch.full((n,), 1e30, device=DEVICE).sum().item(),
        "sum_nan": lambda: torch.full((n,), float("nan"), device=DEVICE).sum().item(),
        "sum_randn": lambda: torch.randn(n, device=DEVICE).sum().item(),
        "max_randn": lambda: torch.randn(n, device=DEVICE).max().item(),
        "small_sum": lambda: torch.ones(16, device=DEVICE).sum().item(),
        "abs": lambda: torch.randn(n, device=DEVICE).abs().cpu(),
        "mul": lambda: (torch.randn(n, device=DEVICE) * 2).cpu(),
        "bool_sum": lambda: (torch.randn(n, device=DEVICE) > 0).sum().item(),
        "int_sum": lambda: torch.ones(n, device=DEVICE, dtype=torch.int32).sum().item(),
        "matmul": lambda: (
            torch.randn(256, 256, device=DEVICE) @ torch.randn(256, 256, device=DEVICE)
        ).cpu(),
    }


def _large_scores(
    q_shape: tuple[int, ...], kv_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One large score per K/V tile."""
    generator = torch.Generator().manual_seed(0)
    k = torch.randn(*kv_shape, generator=generator) * 0.1
    v = torch.randn(*kv_shape, generator=generator)
    for tile in range(kv_shape[2] // 64):
        k[:, :, tile * 64 + (tile * 7) % 64, :] = 60.0 / kv_shape[3]
    q = torch.ones(*q_shape, dtype=BF16, device=DEVICE)
    return q, k.to(BF16).to(DEVICE), v.to(BF16).to(DEVICE)


def _wrong_after_other_kernels(
    call: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    **sdpa: bool,
) -> list[str]:
    """The kernels after which ``call`` gives other bits than on its first call,
    nonfinite rows or an error beyond a bf16 ulp's, in repeated calls."""
    q, k, v = inputs
    double = [x.cpu().double() for x in inputs]
    ref = F.scaled_dot_product_attention(*double, **sdpa)
    first = call(q, k, v)
    wrong = []
    for name, kernel in _other_kernels(q.numel()).items():
        for _ in range(3):
            kernel()
            out = call(q, k, v)
            error = (out.cpu().double() - ref).abs().max() / ref.abs().max()
            if not (torch.equal(out, first) and error <= 2**-6):
                wrong.append(name)
                break
    return wrong


_ATTENTION_CASES = {
    "llama": ((2, 8, 256, 64), (2, 2, 256, 64)),
    "llama-spec": ((4, 32, 512, 64), (4, 8, 512, 64)),
    "97": ((2, 2, 512, 64), (2, 2, 512, 64)),
    "97-b4h8": ((4, 8, 512, 64), (4, 8, 512, 64)),
}
_FORMS = {
    "shipped": {"xegpu_nan_ignoring_maxima": [True]},
    "vectorizer-off": {"xegpu_nan_ignoring_maxima": [True], "igc_vectorizer": ["off"]},
    "nan-propagating": {},
    "nan-propagating-off": {"igc_vectorizer": ["off"]},
}
"""Row maxima and IGC's vectorizer: what the example and 97 use, and the others."""


def _attention_call(
    case: str, form: str = "shipped"
) -> tuple[Callable[..., torch.Tensor], dict[str, bool]]:
    """The attention kernel of ``case`` in ``form``, and its SDPA's options. The
    kernels are new, so they are lowered under the current patches."""
    xegpu = _FORMS[form]
    if case.startswith("llama"):
        example = load_example("xpu/llama_gqa_attention.py")
        kernel = _kernel(example.gqa_attention, example.BLOCK_SIZES, **xegpu)

        def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return example.llama_attention(q, k, v, kernel=kernel)

        return call, {"enable_gqa": True}
    return _kernel(kb.attention, kb.ATTENTION_BLOCKS, **xegpu), {}


@pytest.mark.parametrize("form", list(_FORMS))
@pytest.mark.parametrize("case", list(_ATTENTION_CASES))
def test_attention_after_other_kernels(case: str, form: str) -> None:
    """Regression: the outputs of attention kernels do not depend on the kernel
    that ran before (stale registers: ``xpu_transforms.IGC_OPTIONS``), in each
    form, on normal and on large scores."""
    q_shape, kv_shape = _ATTENTION_CASES[case]
    call, sdpa = _attention_call(case, form)
    normal = _randn(*q_shape), _randn(*kv_shape), _randn(*kv_shape)
    assert _wrong_after_other_kernels(call, normal, **sdpa) == []
    large = _large_scores(q_shape, kv_shape)
    assert _wrong_after_other_kernels(call, large, **sdpa) == []


@pytest.mark.parametrize(
    ("prefix", "shapes"),
    [
        ("29_Softplus", ((128, 393216),)),
        ("29_Matmul_Mish_Mish", None),
        ("81_Gemm_Swish", None),
    ],
)
def test_vectorizer_gains_after_other_kernels(
    prefix: str, shapes: tuple[tuple[int, ...], ...] | None
) -> None:
    """The kernels other than attention that gain most from the vectorizer: the
    same bits as with it off, and stable after other kernels; a sentinel
    if their code changes. It cannot show that they avoid the bug: they do not take
    that path (the no-option control does not fail for them)."""
    problems = (*kb.PROBLEMS, *kb.LEVEL2_PROBLEMS)
    problem = next(p for p in problems if p.name.startswith(prefix))
    off = {**problem.xegpu, "igc_vectorizer": ["off"]}
    generator = torch.Generator().manual_seed(0)
    inputs = [
        torch.randn(*shape, generator=generator).to(problem.dtype).to(DEVICE)
        for shape in shapes or problem.shapes
    ]
    args = problem.args(*inputs)
    first = _kernel(problem.fn, problem.block_sizes, **problem.xegpu)(*args).cpu()
    reference = _kernel(problem.fn, problem.block_sizes, **off)(*args).cpu()
    assert torch.equal(first, reference)
    wrong = []
    kernel = _kernel(problem.fn, problem.block_sizes, **problem.xegpu)
    for name, other in _other_kernels(2**22).items():
        for _ in range(3):
            other()
            if not torch.equal(kernel(*args).cpu(), first):
                wrong.append(name)
                break
    assert wrong == []


def test_attention_matrix_finds_the_miscompile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check above tells the wrong form apart: NaN-ignoring row maxima built
    with IGC's vectorizer on get IGC 2.41's one-element min/max. How many cases of a
    matrix of kernels go wrong varies with what stale registers hold (108-154 of 156
    across runs, none with the options), so this asserts only that some do. If this
    fails after an IGC update, the miscompile is gone: then this test can go, not the
    one above."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    # In-process entries are cached by module text, which the option is not part of.
    monkeypatch.setattr(execution, "_JIT_CACHE", type(execution._JIT_CACHE)())
    for name in ("IGC_OPTIONS", "IGC_OPTIONS_OFF"):
        monkeypatch.setattr(
            xpu_target.xpu_transforms, name, "-ze-opt-large-register-file"
        )
    q_shape, kv_shape = _ATTENTION_CASES["97"]
    call, _ = _attention_call("97")
    inputs = (_randn(*q_shape), _randn(*kv_shape), _randn(*kv_shape))
    assert _wrong_after_other_kernels(call, inputs) != []


_INF = float("inf")
_NONFINITE = {"nan": float("nan"), "inf": _INF, "-inf": -_INF}
_NAN_CASES = {
    "gqa": ((2, 8, 256, 64), (2, 2, 256, 64)),
    "97": ((2, 2, 512, 64), (2, 2, 512, 64)),
}
_NAN_DTYPES = {"bf16": BF16, "f16": torch.float16}
_POSITIONS = {
    # A K/V row: first, a row inside a later tile for each K/V tile the kernels use
    # (32, 128), last.
    "first": 0,
    "tile-end": 127,
    "second-tile": 133,
    "last": 255,
}
_SPANS = {
    "row": slice(70, 71),
    "first-tile": slice(0, 128),
    "second-tile": slice(128, 256),
    "whole-head": slice(0, 256),
}


def _nonfinite_edits() -> dict[str, Callable[..., object]]:
    """``edit(q, k, v)`` by name: a NaN or an inf in one element of each operand, in
    whole K rows and tiles (queries of both signs: scores of both infinities and
    NaN), and K/V tiles of -inf scores (positive queries)."""
    edits: dict[str, Callable[..., object]] = {}

    def element(operand: int, row: int, value: float) -> Callable[..., object]:
        def edit(*tensors: torch.Tensor) -> None:
            tensors[operand][0, 1, row, 3] = value

        return edit

    def span(rows: slice, value: float) -> Callable[..., object]:
        def edit(_: torch.Tensor, k: torch.Tensor, __: torch.Tensor) -> None:
            k[1, 0, rows] = value

        return edit

    def minus_inf_tile(rows: slice) -> Callable[..., object]:
        def edit(q: torch.Tensor, k: torch.Tensor, _: torch.Tensor) -> None:
            q.abs_()
            k[0, 0, rows] = -_INF

        return edit

    for name, value in _NONFINITE.items():
        for operand_index, operand in enumerate("qkv"):
            for position, row in _POSITIONS.items():
                edits[f"{operand}-{name}-{position}"] = element(
                    operand_index, row, value
                )
        for rows_name, rows in _SPANS.items():
            edits[f"k-{name}-{rows_name}"] = span(rows, value)
    edits["minus-inf-first-tile"] = minus_inf_tile(slice(0, 128))
    edits["minus-inf-second-tile"] = minus_inf_tile(slice(128, 256))
    return edits


def _bits(out: torch.Tensor) -> bytes:
    return out.contiguous().view(torch.uint8).numpy().tobytes()


def _nonfinite_differences(
    out: torch.Tensor, on_device: torch.Tensor, reference: torch.Tensor
) -> str | None:
    """How ``out`` differs from torch on the device and in f64: other NaN or inf
    elements, or the others beyond a 16-bit rounding's error of f64."""
    out = out.float()
    for ref in (on_device.float(), reference):
        for test in (torch.isnan, torch.isposinf, torch.isneginf):
            if not torch.equal(test(out), test(ref)):
                return f"{test.__name__}: {int((test(out) ^ test(ref)).sum())} elements"
    finite = out.isfinite()
    if finite.any():
        error = (out[finite].double() - reference[finite]).abs().max()
        if error > 2**-6 * reference[finite].abs().max():
            return f"error {error}"
    return None


@pytest.mark.parametrize("dtype", list(_NAN_DTYPES))
@pytest.mark.parametrize("case", list(_NAN_CASES))
def test_attention_nonfinite(case: str, dtype: str) -> None:
    """A NaN or an inf anywhere (:func:`_nonfinite_edits`): the forms give the same
    bits (``xpu_params.NAN_IGNORING_MAXIMA``), with torch's
    nonfinite elements. Regression among them: -inf scores in the whole first K/V
    tile gave NaN rows in every form of the row maxima (``-inf - -inf`` while the
    running maximum was an initial -inf); they add nothing, as in torch."""
    name = "llama" if case == "gqa" else "97"
    kernels = {form: _attention_call(name, form)[0] for form in _FORMS}
    q_shape, kv_shape = _NAN_CASES[case]
    sdpa = {"enable_gqa": True} if q_shape != kv_shape else {}
    wrong = {}
    for edit_name, edit in _nonfinite_edits().items():
        generator = torch.Generator().manual_seed(0)
        q, k, v = (
            torch.randn(*shape, generator=generator).to(_NAN_DTYPES[dtype])
            for shape in (q_shape, kv_shape, kv_shape)
        )
        edit(q, k, v)
        reference = F.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), **sdpa
        )
        on_device = q.to(DEVICE), k.to(DEVICE), v.to(DEVICE)
        outs = {form: kernel(*on_device).cpu() for form, kernel in kernels.items()}
        out = outs["shipped"]
        if len({_bits(other) for other in outs.values()}) > 1:
            wrong[edit_name] = "the forms differ"
        torch_out = F.scaled_dot_product_attention(*on_device, **sdpa).cpu()
        difference = _nonfinite_differences(out, torch_out, reference)
        if difference is not None:
            wrong[edit_name] = difference
    assert wrong == {}


@pytest.mark.parametrize("dtype", list(_NAN_DTYPES))
@pytest.mark.parametrize("case", list(_NAN_CASES))
def test_attention_row_of_only_minus_inf_scores(case: str, dtype: str) -> None:
    """Holds a known difference. A row with only -inf scores ends in ``0 / 0``: NaN,
    as ``F.softmax(...) @ V`` gives, while torch's SDPA gives finite outputs here.
    Other rows are as torch's. If this fails the kernel has changed its contract:
    the example's README describes it."""
    name = "llama" if case == "gqa" else "97"
    q_shape, kv_shape = _NAN_CASES[case]
    generator = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(*shape, generator=generator).to(_NAN_DTYPES[dtype])
        for shape in (q_shape, kv_shape, kv_shape)
    )
    q.abs_()
    k[0, 0] = -_INF
    sdpa = {"enable_gqa": True} if q_shape != kv_shape else {}
    on_device = q.to(DEVICE), k.to(DEVICE), v.to(DEVICE)
    empty = slice(0, 4) if case == "gqa" else slice(0, 1)
    for form in _FORMS:
        out = _attention_call(name, form)[0](*on_device).cpu()
        assert out[0, empty].isnan().all()
        assert out.isfinite().sum() == out.numel() - out[0, empty].numel()
    reference = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), **sdpa
    )
    assert reference[0, empty].isfinite().all()
    torch_out = F.scaled_dot_product_attention(*on_device, **sdpa).cpu()
    assert torch_out[0, empty].float().isfinite().all()


@pytest.mark.parametrize(
    "fn",
    [ir_kernels.attention_and_row_maximum, ir_kernels.attention_and_output_maximum],
)
def test_key_keeps_nan_in_other_uses_of_a_maximum(fn: Callable[..., object]) -> None:
    """Regression (exact kernels): with ``xegpu_nan_ignoring_maxima`` a stored
    running maximum, or a maximum of the output row, was finite for a row with a
    NaN, where torch's is NaN (512 of 2048 rows, one NaN in K)."""
    generator = torch.Generator().manual_seed(0)
    shape = 2, 2, 512, 64
    q, k, v = (torch.randn(*shape, generator=generator).to(BF16) for _ in range(3))
    k[0, 1, 100, 3] = float("nan")
    kernel = _kernel(fn, kb.ATTENTION_BLOCKS, **_FORMS["shipped"])
    out, maximum = kernel(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE))
    expected_rows = out.cpu().isnan().any(-1, keepdim=True)
    assert expected_rows.any()
    assert torch.equal(maximum.cpu().isnan(), expected_rows)


@pytest.mark.parametrize("seq", [128, 256, 384, 512, 640, 768, 1024])
def test_attention_blocks_fit_the_shape(seq: int) -> None:
    """Regression: 97's bench blocks refused sequences that the earlier ones took
    (128, 640, 768). :func:`kernelbench_xpu.attention_blocks`, as AI-bench's model."""
    q, k, v = (_randn(2, 2, seq, 64) for _ in range(3))
    kernel = _kernel(kb.attention, kb.attention_blocks(2, 2, seq), **_FORMS["shipped"])
    _check(kernel(q, k, v), F.scaled_dot_product_attention(q, k, v), 2**-6)


@pytest.mark.parametrize(
    ("q_shape", "kv_shape"),
    [
        ((2, 4, 128, 64), (2, 1, 128, 64)),
        ((1, 2, 256, 64), (1, 1, 256, 64)),
        ((2, 4, 256, 64), (2, 1, 256, 64)),
        ((2, 8, 384, 64), (2, 2, 384, 64)),
        ((2, 2, 512, 64), (2, 2, 512, 64)),
        ((1, 1, 256, 64), (1, 1, 256, 64)),
    ],
)
def test_llama_example_other_shapes(
    q_shape: tuple[int, ...], kv_shape: tuple[int, ...]
) -> None:
    """Regression: block sizes that leave one workgroup were refused with "xegpu_*
    lists of 1 entries, but the kernel has 0 GPU kernel(s)"; here other block sizes
    take the shape."""
    example = load_example("xpu/llama_gqa_attention.py")
    q, k, v = _randn(*q_shape), _randn(*kv_shape), _randn(*kv_shape)
    _check(example.llama_attention(q, k, v), _llama_torch(q, k, v), 2**-6)


def test_attention_runtime_scale() -> None:
    """Regression (exact kernel, as lowered in ``test_xpu_ir``): a runtime float
    scale failed in XeVM. Right against SDPA, for two scales."""
    example = load_example("xpu/llama_gqa_attention.py")
    q, k, v = _llama_inputs()
    packed = q.view(2, 2, 1024, 64)
    kernel = _kernel(ir_kernels.attention_runtime_scale, example.BLOCK_SIZES)
    _check(kernel(packed, k, v, 0.125).view(q.shape), _llama_torch(q, k, v), 2**-6)
    doubled = F.scaled_dot_product_attention(
        q.cpu().double() * 2, k.cpu().double(), v.cpu().double(), enable_gqa=True
    )
    out = kernel(packed, k, v, 0.25).view(q.shape)
    error = (out.cpu().double() - doubled).abs().max() / doubled.abs().max()
    assert error <= 2**-6


def test_attention_head_index_from_outer_loop() -> None:
    """Regression (exact kernel, as lowered in ``test_xpu_ir``): ``h = th.begin //
    group`` above the K/V loop failed with "invalid block key". Unpacked query heads
    give torch's GQA."""
    q, k, v = _llama_inputs()
    out = _kernel(ir_kernels.attention_head_index, [256, 64])(q, k, v)
    _check(out, _llama_torch(q, k, v), 2**-6)


def test_attention_head_index_then_a_later_phase() -> None:
    """Regression (the head index and later phase of a Llama step's entry): host code
    after the loop made ``k_head`` an argument, which failed with "name 'k_head' is not
    defined"."""
    q, k, v = _llama_inputs()
    out = _kernel(ir_kernels.attention_head_index_then_phase, [256, 64, 1, 1, 256, 64])(
        q, k, v
    )
    _check(out, _llama_torch(q, k, v), 2**-6)


@pytest.mark.parametrize("expression", list(ir_kernels.TILE_EXPRESSIONS))
def test_tile_expression_in_an_inner_loop(
    expression: str, tmp_path: pathlib.Path
) -> None:
    """K/V heads computed with tile position arithmetic above the inner loop, against
    SDPA of those heads."""
    source, head_of = ir_kernels.TILE_EXPRESSIONS[expression]
    fn = ir_kernels.tile_expression_kernel(tmp_path, source)
    q, k, v = _llama_inputs()
    out = _kernel(fn, [256, 64])(q, k, v).cpu().double()
    exact = torch.stack(
        [
            F.scaled_dot_product_attention(
                q[:, head].cpu().double(),
                k[:, head_of(head)].cpu().double(),
                v[:, head_of(head)].cpu().double(),
            )
            for head in range(8)
        ],
        dim=1,
    )
    assert (out - exact).abs().max() <= 2**-6 * exact.abs().max()


# --- Runtime: USM interop, synchronization, device index, caching ----------------


def _spy_entry_arguments() -> tuple[list[list[object]], object]:
    """A patch of ``CompiledEntry`` recording the values each call passes, on the
    in-place path or the general one."""
    calls: list[list[object]] = []
    original = execution.CompiledEntry.__call__
    original_in_place = execution.CompiledEntry.call_in_place

    def spy(self: execution.CompiledEntry, values: list[object]) -> None:
        calls.append(list(values))
        original(self, values)

    def spy_in_place(
        self: execution.CompiledEntry, values: list[object]
    ) -> list[torch.Tensor] | None:
        inouts = original_in_place(self, values)
        if inouts is not None:
            calls.append(list(values))
        return inouts

    return calls, mock.patch.multiple(
        execution.CompiledEntry, __call__=spy, call_in_place=spy_in_place
    )


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


def test_mish_mish_binary_does_not_spill() -> None:
    """29_Matmul_Mish_Mish's nested polynomials, tile by tile: 92544 bytes of
    spill and 11 ms against 3.9 in vector-op order (``xevm_tile_order``). The
    binary keeps the 256 GRFs."""
    problem = {p.name: p for p in kb.LEVEL2_PROBLEMS}["29_Matmul_Mish_Mish"]
    inputs = [_randn(*shape, dtype=problem.dtype) for shape in problem.shapes]
    binary = lower_mlir(problem.kernel, list(problem.args(*inputs)), stop_at="binary")
    ((zebin,),) = _objects(str(binary))
    assert re.findall(rb"grf_count:\s*(\d+)", zebin) == [b"256"]
    assert not re.search(rb"spill_size:\s*[1-9]", zebin)


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


def _new_input(seed: int, m: int, n: int) -> torch.Tensor:
    """A new input per call: an output reuses a freed block, so a repeated input
    would pass with last call's right contents."""
    torch.manual_seed(seed)
    return torch.randn(m, n)


def _forward_shift_two_loops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and write overlap within the first loop, then an unrelated loop."""
    m, n = x.shape
    a = x.clone()
    w = a[: m - 1]
    v = a[1:]
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m - 1, n]):
        w[tm, tn] = v[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] + 1.0
    return a, out


@pytest.mark.parametrize("m", [2048, 4096])
def test_overlapping_views_in_one_loop_keep_the_clone(m: int) -> None:
    """Regression (review of the first view fix, which raced: up to 22000 elements
    wrong per call): such a view reads no earlier store: its value at the call."""
    kernel = _kernel(_forward_shift_two_loops, [1, 64, 1, 64])
    for seed in range(6):
        x = _new_input(seed, m, 1024)
        a, _ = kernel(x.to("xpu"))
        torch.xpu.synchronize()
        expected = x.clone()
        expected[:-1] = x[1:] * 2
        assert torch.equal(a.cpu(), expected)


def _every_other_column(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    a = torch.empty_like(x)
    v = a[:, ::2]
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


def _misaligned_view(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(-1)[4 : 4 + (m - 1) * n].view(m - 1, n)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m - 1, n]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


@pytest.mark.parametrize("fn", [_every_other_column, _misaligned_view])
def test_views_a_call_must_copy_are_refused(fn: Callable) -> None:
    """Regression: strided and misaligned views stay staged copies, so a read after
    the write would be stale: refused, on the device too."""
    kernel = _kernel(fn, [1, 64, 1, 64])
    with pytest.raises(UnsupportedOperationError, match="two kernel calls"):
        kernel(_new_input(0, 64, 128).to("xpu"))


def _view_of_output(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    a = torch.empty_like(x)
    v = a.view(m * 2, n // 2)
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


def test_view_across_phases() -> None:
    """Regression: the driver cloned the view before the call (it overlaps a written
    tensor), so the second launch read stale memory: every element wrong, silently."""
    kernel = _kernel(_view_of_output, [1, 64, 1, 64])
    for seed in range(4):
        x = _new_input(seed, 64, 128)
        out = kernel(x.to("xpu"))
        torch.xpu.synchronize()
        assert torch.equal(out.cpu(), (x * 2.0).view(128, 64) + 1.0)


def _caller_view(x: torch.Tensor, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(v)
    for tm, tn in hl.tile([m, n]):
        a[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = v[tm, tn] + 1.0
    return out


def test_callers_view_sees_the_write() -> None:
    """A caller's view is read like a host one: it sees earlier writes."""
    kernel = _kernel(_caller_view, [1, 64, 1, 64])
    for seed in range(4):
        x = _new_input(seed, 64, 128)
        a = torch.empty_like(x, device="xpu")
        out = kernel(x.to("xpu"), a, a.view(128, 64))
        torch.xpu.synchronize()
        assert torch.equal(out.cpu(), (x * 2.0).view(128, 64) + 1.0)


@pytest.mark.parametrize(
    ("name", "refusal"),
    [
        ("U2_range_read_then_store", "two kernel calls"),
        ("U5_dynamic_while_read_then_store", "bufferization"),
        ("U6_range_second_phase", "a tensor read after the kernel wrote"),
    ],
)
def test_next_iteration_reads_are_refused(name: str, refusal: str) -> None:
    """Regression: a nested loop's next iteration reads what this one stored, so the
    copy was stale: U2's every element wrong on the B50, silently. U5 doesn't reach
    the read order on XPU: it fails to lower, with no result to be wrong. U6 is refused
    for the read order itself."""
    from tests.view_read_cases import CASES

    case = CASES[name]
    kernel = _kernel(case["fn"], [1, 64] * (case["blocks"] // 2))
    with pytest.raises(Exception, match=refusal):
        kernel(_new_input(0, 64, 128).to("xpu"))


def test_call_right_with_work_outstanding_on_the_stream() -> None:
    """A kernel right after torch's unfinished work on its input. On this machine a
    call with no wait at all was right too (the driver seems to run queues in order,
    ``XpuTarget.prepare_call``), so removing the wait can't be a failing control."""

    def add_one(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.empty_like(x)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + 1.0
        return out

    kernel = _kernel(add_one, [1, 256])
    a = torch.rand(4096, 4096, device="xpu")
    for seed in range(6):
        torch.manual_seed(seed)
        y = torch.rand(64, 4096)
        x = y.to("xpu")
        x = x * 1.0
        _ = torch.matmul(a, a)  # outstanding on the stream when the kernel is called
        out = kernel(x)
        torch.xpu.synchronize()
        assert torch.equal(out.cpu(), y + 1.0)


def test_call_right_after_a_large_copy_to_the_device() -> None:
    """A copy may use another engine: the kernel reads it right after. No-wait isn't a
    failing control here either (see above)."""

    def add_one(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.empty_like(x)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + 1.0
        return out

    kernel = _kernel(add_one, [1, 8192])
    for seed in range(6):
        torch.manual_seed(seed)
        y = torch.rand(16384, 8192)
        out = kernel(y.to("xpu"))
        torch.xpu.synchronize()
        assert torch.equal(out.cpu(), y + 1.0)


@pytest.mark.parametrize("pos", [63, -1, -64])
def test_runtime_position_load_matches_torch(pos: int) -> None:
    from tests.test_runtime_index_bounds import get_row

    cache = torch.randn(8, 64, 16)
    out = _kernel(get_row, [1, 16])(cache.to("xpu"), pos)
    torch.xpu.synchronize()
    assert torch.equal(out.cpu(), cache[:, pos])


def test_runtime_position_out_of_bounds_raises() -> None:
    from tests.test_runtime_index_bounds import get_row

    with pytest.raises(IndexError, match="out of bounds"):
        _kernel(get_row, [1, 16])(torch.randn(8, 64, 16).to("xpu"), 64)


def test_sibling_gemms_swiglu() -> None:
    """Two GEMMs of one A tile and an epilogue in one grid loop, at SwiGLU's decode
    shape and at a tile with less register room, per row against float64, after
    another kernel ran (stale registers show here)."""
    from tests.test_xpu_sibling_gemms import swiglu

    for (m, n, k), blocks, extra in (
        ((64, 512, 256), [64, 256, 32], {"xegpu_sg_tiles": [[8, 64]]}),
        (
            (256, 512, 256),
            [128, 256, 32],
            {"xegpu_sg_tiles": [[32, 32]], "xegpu_load_tiles": [[32, 32, 16, 16]]},
        ),
    ):
        x = torch.randn(m, k).to(BF16)
        gate = (torch.randn(n, k) * 0.05).to(BF16)
        up = (torch.randn(n, k) * 0.05).to(BF16)
        ref = torch.nn.functional.silu(x.double() @ gate.double().T) * (
            x.double() @ up.double().T
        )
        kernel = _kernel(swiglu, blocks, **extra)
        out = kernel(x.to("xpu"), gate.to("xpu"), up.to("xpu"))
        torch.xpu.synchronize()
        error = (out.cpu().double() - ref).abs().amax(-1) / ref.abs().amax()
        assert error.max() < 4e-3


def test_row_store_after_other_kernels() -> None:
    """One row of a cache written, the other rows kept, in one launch among others:
    the K/V cache write of a decode step."""
    from tests.test_xpu_fold_region_inserts import put_row

    x = torch.randn(8, 8, 64).to(BF16)
    cache = torch.randn(8, 8, 512, 64).to(BF16)
    device_cache = cache.to("xpu")
    _kernel(put_row, [1, 1, 64])(x.to("xpu"), device_cache, 511)
    torch.xpu.synchronize()
    expected = cache.clone()
    expected[:, :, 511] = x
    assert torch.equal(device_cache.cpu(), expected)


@pytest.mark.parametrize("pos", [-1, -512, 511])
def test_row_store_at_a_runtime_position(pos: int) -> None:
    from tests.test_xpu_fold_region_inserts import put_row

    x = torch.randn(8, 8, 64).to(BF16)
    cache = torch.randn(8, 8, 512, 64).to(BF16)
    device_cache = cache.to("xpu")
    _kernel(put_row, [1, 1, 64])(x.to("xpu"), device_cache, pos)
    torch.xpu.synchronize()
    expected = cache.clone()
    expected[:, :, pos] = x
    assert torch.equal(device_cache.cpu(), expected)


@pytest.mark.parametrize("pos", [512, -513])
def test_row_store_out_of_bounds_raises(pos: int) -> None:
    from tests.test_xpu_fold_region_inserts import put_row

    cache = torch.zeros(8, 8, 512, 64, device="xpu").to(BF16)
    with pytest.raises(IndexError, match="out of bounds"):
        _kernel(put_row, [1, 1, 64])(
            torch.randn(8, 8, 64).to(BF16).to("xpu"), cache, pos
        )


def _add_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


def test_kernel_waits_for_a_copy_offloaded_from_the_host() -> None:
    """A non-blocking host-to-device copy runs on the copy engine, off the compute ring
    a kernel's launch waits on: with no wait in ``prepare_call``, 60 of 60 calls read
    the old data on the B50 (``UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`` hides it). The
    stream's ``query`` says busy, so the gate waits. A fresh value per trial, so
    stale data can't pass."""
    kernel = _kernel(_add_one, [1, 256])
    dst = torch.zeros((128, 1024, 1024), device="xpu")
    pinned = torch.zeros((128, 1024, 1024)).pin_memory()
    for trial in range(8):
        value = 1000.0 + trial
        pinned.fill_(value)
        dst.zero_()
        torch.xpu.synchronize()
        dst.copy_(pinned, non_blocking=True)
        out = kernel(dst[-1, -64:, :])
        torch.xpu.synchronize()
        assert (out.cpu() == value + 1).all()


def _plain_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.shape):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


def _phases(x: torch.Tensor, k: int) -> torch.Tensor:
    """Two launches through a view of a fresh tensor (the host writes nothing),
    with a runtime scalar: the second reads what the first stored."""
    m, n = x.shape
    q = torch.empty_like(x)
    view = q.view(m * 2, n // 2)
    out = torch.empty_like(view)
    for tm, tn in hl.tile([m, n]):
        q[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile([m * 2, n // 2]):
        out[tm, tn] = view[tm, tn] + k
    return out


def test_repeat_calls_take_the_in_place_path_with_fresh_values() -> None:
    """After the first call of an aliasing kernel no call packs its arguments anew,
    and each call computes from its own inputs, scalars and buffers: an earlier
    output is left as it was (its address differs), outputs are bit-exact against the
    general path."""
    kernel = _kernel(_phases, [1, 256, 1, 128])
    y = torch.randn(64, 256, dtype=F32, device="xpu")
    expected = lambda x, k: (x * 2.0).view(128, 128) + k  # noqa: E731
    first = kernel(y, 3)
    torch.xpu.synchronize()
    _check_exact = lambda out, ref: torch.equal(out.cpu(), ref.cpu())  # noqa: E731
    assert _check_exact(first, expected(y, 3))
    kept = first.clone()
    with mock.patch.object(
        execution, "_packed_args", wraps=execution._packed_args
    ) as general:
        for step in range(4):
            x = torch.randn(64, 256, dtype=F32, device="xpu")
            out = kernel(x, step)
            torch.xpu.synchronize()
            assert _check_exact(out, expected(x, step)), step
            assert _check_exact(first, kept)
            # The same input again: a stale buffer would not show otherwise.
            again = kernel(y, step)
            torch.xpu.synchronize()
            assert _check_exact(again, expected(y, step)), step
            assert _check_exact(first, kept)
    assert not general.called


def test_in_place_calls_wait_for_the_stream() -> None:
    kernel = _kernel(_plain_add, [1, 256])
    x = torch.randn(64, 256, dtype=F32, device="xpu")
    kernel(x)
    with (
        mock.patch.object(
            XpuTarget, "prepare_call", autospec=True, side_effect=XpuTarget.prepare_call
        ) as prepare,
        mock.patch.object(
            execution, "_packed_args", wraps=execution._packed_args
        ) as general,
    ):
        out = kernel(x)
    torch.xpu.synchronize()
    assert prepare.call_count == 1
    assert not general.called
    assert torch.equal(out.cpu(), x.cpu() + 1.0)


@pytest.mark.parametrize("form", ["misaligned", "strided", "negative", "aliased"])
def test_arguments_that_need_care_after_in_place_calls(form: str) -> None:
    """A kernel on the in-place path meets an argument it can't take as is: the
    general path staged it, right values, and the next plain call is right too."""

    def add_into(x: torch.Tensor, out: torch.Tensor) -> None:
        for tm, tn in hl.tile(x.shape):
            out[tm, tn] = x[tm, tn] + 1.0

    kernel = _kernel(add_into, [1, 64])
    base = torch.randn(64, 256, dtype=F32, device="xpu")
    out = torch.empty_like(base)
    kernel(base, out)
    if form == "misaligned":
        buf = torch.randn(64 * 256 + 1, dtype=F32, device="xpu")
        x = buf[1:].view(64, 256)
    elif form == "strided":
        x = torch.randn(128, 256, dtype=F32, device="xpu")[::2]
    elif form == "negative":
        x = torch._neg_view(base)
    else:
        x = out  # the input is the output: its value at the call
        out.copy_(base)
    ref = x.resolve_neg().clone() + 1.0
    with mock.patch.object(
        execution, "_packed_args", wraps=execution._packed_args
    ) as general:
        kernel(x, out)
    torch.xpu.synchronize()
    assert general.called, "right values alone don't show the call was refused"
    assert torch.equal(out.cpu(), ref.cpu())
    with mock.patch.object(
        execution, "_packed_args", wraps=execution._packed_args
    ) as general:
        kernel(base, out)
    torch.xpu.synchronize()
    assert not general.called
    assert torch.equal(out.cpu(), base.cpu() + 1.0)
