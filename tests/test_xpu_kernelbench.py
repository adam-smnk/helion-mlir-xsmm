"""KernelBench level 1 problems on the XPU target at the IR level, at the shapes
and dtypes of AI-bench's ``mlir-gpu-bench`` variant (fake XPU tensors: the large
shapes cost nothing). Each problem (``kernelbench_xpu.PROBLEMS``) lowers to
annotated XeGPU (``xegpu-wg``) with its kernel class's ops and one launch over
the expected workgroup grid, and on to LLVM/XeVM (``xevm``) without XeGPU ops
left. Rank-3+ problems (``RANK3_PROBLEMS``: block size 1 on the leading dims)
lower as their 2D kernel class: 2D XeGPU tiles, the leading dims as grid dims.
Attention (97, written as flash attention) lowers with lighthouse's attention
layout, both matmuls on DPAS; the same kernel source runs on the CPU target
against PyTorch's SDPA (no XPU needed). Level 2 problems (``LEVEL2_PROBLEMS``:
``nn.Linear`` with its epilogue fused in) lower as GEMMs with a transposed B,
the bias row and the epilogue on the f32 accumulator; their sources also run on
the CPU target against the KernelBench models' eager PyTorch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import helion
import pytest
import torch
import torch.nn.functional as F

from tests.harness import lower_xpu_stages
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import ATTENTION_BLOCKS
from tests.kernelbench_xpu import ELEMENTWISE_BLOCKS
from tests.kernelbench_xpu import LEVEL2_PROBLEMS
from tests.kernelbench_xpu import PROBLEMS
from tests.kernelbench_xpu import RANK3_PROBLEMS
from tests.kernelbench_xpu import Problem
from tests.kernelbench_xpu import attention

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Callable

_LAYOUT = re.compile(r"layout = #xegpu\.layout<([^>]*)>")
_TENSOR_DESC = re.compile(r"!xegpu\.tensor_desc<((?:\d+x)+)\w+")
_ALL = {
    problem.name: problem for problem in (*PROBLEMS, *RANK3_PROBLEMS, *LEVEL2_PROBLEMS)
}


def _args(problem: Problem) -> list[object]:
    return problem.args(
        *(xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes)
    )


def _anchor_layouts(text: str, op: str) -> list[str]:
    """The anchor layouts of the ``op`` ops in ``text`` (one per op)."""
    lines = [line for line in text.splitlines() if f"{op} " in line and "=" in line]
    return [match.group(1) for line in lines if (match := _LAYOUT.search(line))]


def _check_launch(text: str, grid: tuple[int, ...]) -> None:
    assert text.count("gpu.launch_func") == 1
    padded = [*grid, *[1] * (3 - len(grid))]
    assert f"known_grid_size = array<i32: {', '.join(map(str, padded))}>" in text


def _check_matmul(text: str) -> None:
    for op in ("xegpu.dpas", "xegpu.load_nd", "xegpu.prefetch_nd", "xegpu.store_nd"):
        assert op in text, op
    (store,) = _anchor_layouts(text, "xegpu.store_nd")
    assert "inst_data = [8, 16]" in store
    # bf16 out: the f32 accumulator is truncated before the store.
    assert "arith.truncf" in text
    assert re.search(r"xegpu\.store_nd [^\n]*vector<\d+x\d+xbf16>", text)


def _check_elementwise(text: str) -> None:
    assert "xegpu.dpas" not in text and "vector.multi_reduction" not in text
    if "xegpu.load_nd" not in text:
        _check_elementwise_1d(text)
        return
    loads = _anchor_layouts(text, "xegpu.load_nd")
    stores = _anchor_layouts(text, "xegpu.store_nd")
    assert len(loads) == len(stores) == 1
    # Inferred for a 128x256 bf16 tile (19_ReLU_NCHW): 32 subgroups of 32x32,
    # 16x16 block loads (16 columns, at most 16 elements per work item), 8x16
    # block stores.
    tile = "sg_layout = [4, 8], sg_data = [32, 32]"
    assert loads == [f"{tile}, inst_data = [16, 16]"]
    assert stores == [f"{tile}, inst_data = [8, 16]"]
    assert "known_block_size = array<i32: 512, 1, 1>" in text


def _check_elementwise_1d(text: str) -> None:
    # ELEMENTWISE_BLOCKS [1, 8192]: the unit row folds to a 1D tile of 8192
    # contiguous bf16, 8 subgroups of 1024, gathered and scattered 8 elements
    # (16 bytes) per work item: inst_data 16 x 8.
    layout = "sg_layout = [8], sg_data = [1024], inst_data = [128]"
    assert _anchor_layouts(text, "xegpu.load") == [layout]
    assert _anchor_layouts(text, "xegpu.store") == [layout]
    assert "known_block_size = array<i32: 128, 1, 1>" in text


def _check_reduction(text: str) -> None:
    assert "vector.multi_reduction" in text
    # Lighthouse's reduction layout (8 subgroups of 8 rows of the 64x64 f32
    # tiles), with 8x16 block loads and stores.
    tile = "sg_layout = [8, 1], sg_data = [8, 64]"
    loads = _anchor_layouts(text, "xegpu.load_nd")
    assert len(loads) >= 2
    assert set(loads) == {f"{tile}, inst_data = [8, 16]"}
    assert _anchor_layouts(text, "xegpu.store_nd") == [f"{tile}, inst_data = [8, 16]"]
    assert "known_block_size = array<i32: 128, 1, 1>" in text


def _check_attention(text: str) -> None:
    # Q @ K^T and P @ V, the online softmax's row maxima and sums.
    assert text.count("xegpu.dpas ") == 2
    assert text.count("vector.multi_reduction") == 2
    # Lighthouse's attention layout (annotate_attention): 8 subgroups of 16
    # query rows; every subgroup reads the whole 64x64 K and V tiles, K
    # transposed (order), V in 32x32 blocks, Q in 16x16 blocks.
    rows = "sg_layout = [8, 1], sg_data = [16, 64]"
    kv = "sg_layout = [1, 1], sg_data = [64, 64]"
    assert _anchor_layouts(text, "xegpu.load_nd") == [
        f"{rows}, inst_data = [16, 16]",
        f"{kv}, order = [0, 1]",
        f"{kv}, inst_data = [32, 32]",
    ]
    assert _anchor_layouts(text, "xegpu.store_nd") == [rows]
    # K and V prefetched one K/V step ahead, before and in the loop.
    prefetches = _anchor_layouts(text, "xegpu.prefetch_nd")
    assert len(prefetches) == 4
    assert set(prefetches) == {
        "sg_layout = [4, 2], sg_data = [16, 32], inst_data = [16, 32]"
    }
    assert "known_block_size = array<i32: 128, 1, 1>" in text
    # The softmax runs in f32, the output is stored as bf16.
    assert "math.exp" in text and "vector<128x64xf32>" in text
    assert re.search(r"xegpu\.store_nd [^\n]*vector<128x64xbf16>", text)


def _check_linear(text: str) -> None:
    """``nn.Linear`` and an epilogue: the GEMM with B (the weight) read
    transposed, A in one block load per subgroup tile (``xegpu_load_tiles``),
    the bias row gathered and broadcast over the rows, the epilogue in f32."""
    _check_matmul(text)
    sg_a = re.search(
        r"xegpu\.dpas [^\n]*layout_a = [^\n]*?sg_data = \[(\d+), (\d+)\]", text
    )
    assert sg_a is not None
    a_load, b_load = _anchor_layouts(text, "xegpu.load_nd")
    assert f"inst_data = [{sg_a[1]}, {sg_a[2]}]" in a_load
    assert "inst_data = [16, 16], order = [0, 1]" in b_load
    assert re.search(r"xegpu\.load [^\n]*-> vector<256xbf16>", text)
    assert re.search(
        r"vector\.broadcast [^\n]*vector<256xf32> to vector<\d+x256xf32>", text
    )
    # bf16 only at the loads and the final rounding.
    assert not re.search(r"(arith|math)\.(?!truncf|extf)\w+ [^\n]*xbf16>", text)
    assert "known_block_size = array<i32: 512, 1, 1>" in text


_DPAS_CALL = re.compile(
    r"llvm\.call spir_funccc @_Z\d+intel_sub_group_bf16_bf16_matrix_mad_k16"
)


def _block_read_rows(lowered: str) -> set[int]:
    """The rows of the 32-bit 16-column 2D block reads of lowered device code."""
    return {
        int(rows)
        for rows in re.findall(r"intel_sub_group_2d_block_read_32b_(\d+)r16x", lowered)
    }


_CHECKS = {
    "matmul": _check_matmul,
    "elementwise": _check_elementwise,
    "reduction": _check_reduction,
    "attention": _check_attention,
    "linear": _check_linear,
}


_REJECTED = {
    "22_Tanh": r"math\.tanh rounded to bf16",
    # abs of bf16 x: right on the device for softsign's shared operand, but not
    # separable from the wrong forms (xpu_checks.BF16_OPERAND_VERIFIED).
    "30_Softsign": r"math\.absf of a bf16 value",
}


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(_ALL))
def test_kernelbench_problem_lowers_for_xpu(name: str) -> None:
    problem = _ALL[name]
    if name in _REJECTED:
        # Computed wrong on the device (xpu_checks.check_math).
        with pytest.raises(UnsupportedOperationError, match=_REJECTED[name]):
            lower_xpu_stages(problem.kernel, _args(problem))
        return
    text, lowered = lower_xpu_stages(problem.kernel, _args(problem))
    _check_launch(text, problem.grid)
    _CHECKS[problem.kind](text)
    # Every tile is 2D, also of rank-3+ problems (unit leading dims folded);
    # the one-row elementwise tiles are 1D (gathers and scatters: no tensor
    # descriptors).
    ranks = {dims.count("x") for dims in _TENSOR_DESC.findall(text)}
    one_row = problem.kind == "elementwise" and problem.block_sizes[-2] == 1
    assert ranks == (set() if one_row else {2}), ranks
    assert "xegpu." not in lowered
    assert "llvm.func spir_funccc" in lowered
    assert re.search(r"gpu\.launch_func <%\w+ : !llvm\.ptr>", lowered)
    # No math on bf16 left for the device (no such OpenCL builtins).
    assert not re.search(r"llvm\.func @_Z\w+\([^)]*bf16", lowered)
    if problem.kind == "reduction":
        # The anchored loads: 8-row block reads, not one row each.
        assert _block_read_rows(lowered) == {8}
    if problem.kind == "attention":
        # Both matmuls on DPAS (8x16x16): per subgroup and K/V step, 2x4 blocks
        # of 16x64 times 4 K steps of 16, for Q @ K^T and for P @ V.
        assert len(_DPAS_CALL.findall(lowered)) == 2 * 8 * 4
        # K read transposed, V VNNI-transformed (the DPAS B operands).
        assert "intel_sub_group_2d_block_read_transpose_32b_16r8x1c" in lowered
        assert "intel_sub_group_2d_block_read_transform_16b_32r16x2c" in lowered
    if problem.kind == "linear":
        # Per subgroup and K step of 32: 32x64 (16x64 for 59's 128-row tiles)
        # in 8x16 DPAS blocks, two K steps of 16.
        sg_rows = problem.block_sizes[0] // 8
        assert len(_DPAS_CALL.findall(lowered)) == sg_rows // 8 * 4 * 2
        # A: the subgroup's whole 32-column tile in one 2D block read; the
        # weight ([N, K]) read transposed.
        assert f"intel_sub_group_2d_block_read_16b_{sg_rows}r16x2c" in lowered
        assert "intel_sub_group_2d_block_read_transpose_32b_16r8x1c" in lowered


_LAYOUT_32X64 = "sg_data = [32, 64], inst_data = [8, 16]"
_GEMM_PINS = {
    # GEMM_SQUARE: 128x512 workgroups of 4x8 subgroups of 32x64, A loaded 32x32
    # and B 16x32 per read (B's 32x32 tile is a [16, 32] block read), inferred
    # prefetches, one K step ahead.
    "1_Square_matrix_multiplication_": {
        "grid": (32, 8),
        "sg_layout": "[4, 8]",
        "a_load": "sg_data = [32, 32], inst_data = [32, 32]",
        "b_load": "sg_data = [32, 64], inst_data = [16, 32]",
        "prefetches": [
            "sg_layout = [16, 2], sg_data = [8, 16], inst_data = [8, 16]",
            "sg_layout = [2, 16], sg_data = [16, 32], inst_data = [8, 16]",
        ],
        "prefetch_count": 4,
        "block_reads": {"read_16b_32r16x2c", "read_transform_16b_16r16x2c"},
    },
    # GEMM_STANDARD: 256x256 workgroups of 8x4 subgroups of 32x64, A loaded
    # 8x32 and B 32x32, prefetches of 16x16, one K step ahead.
    "2_Standard_matrix_multiplication_": {
        "grid": (8, 16),
        "sg_layout": "[8, 4]",
        "a_load": "sg_data = [32, 32], inst_data = [8, 32]",
        "b_load": "sg_data = [32, 64], inst_data = [32, 32]",
        "prefetches": [
            "sg_layout = [16, 2], sg_data = [16, 16], inst_data = [8, 16]",
            "sg_layout = [2, 16], sg_data = [16, 16], inst_data = [8, 16]",
        ],
        "prefetch_count": 4,
        "block_reads": {"read_16b_8r16x2c", "read_transform_16b_32r16x2c"},
    },
}


_GEMM_PINS["13_Matmul_for_symmetric_matrices"] = _GEMM_PINS[
    "1_Square_matrix_multiplication_"
]


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(_GEMM_PINS))
def test_gemm_hand_picked_config(name: str) -> None:
    """1_Square, 13_Matmul_for_symmetric_matrices and 2_Standard run the
    hand-picked configs of ``GEMM_SQUARE`` and ``GEMM_STANDARD``: the subgroup
    layout (32x64, also where it is what lighthouse infers: 1 and 13 on 128x512
    workgroups), the block loads and prefetches, and the 2D block reads they
    lower to. For 2 the inferred 64x32 differs."""
    pin = _GEMM_PINS[name]
    problem = _ALL[name]
    text, lowered = lower_xpu_stages(problem.kernel, _args(problem))
    _check_launch(text, pin["grid"])
    layout = f"sg_layout = {pin['sg_layout']}"
    assert _anchor_layouts(text, "xegpu.store_nd") == [f"{layout}, {_LAYOUT_32X64}"]
    a_load, b_load = _anchor_layouts(text, "xegpu.load_nd")
    assert a_load == f"{layout}, {pin['a_load']}"
    assert b_load == f"{layout}, {pin['b_load']}"
    prefetches = _anchor_layouts(text, "xegpu.prefetch_nd")
    assert len(prefetches) == pin["prefetch_count"]
    assert set(prefetches) == set(pin["prefetches"])
    reads = set(
        re.findall(r"intel_sub_group_2d_block_(read\w*?_16b_\d+r16x2c)", lowered)
    )
    assert reads == pin["block_reads"]
    assert "known_block_size = array<i32: 512, 1, 1>" in text


@pytest.mark.isolated
def test_elementwise_hand_picked_config() -> None:
    """19_ReLU (as all of 19-32) runs ``ELEMENTWISE_BLOCKS``: one-row tiles of
    8192 columns, the column tiles first in the grid (48 of them per row, x),
    and per work item 16-byte loads and stores of 8 bf16 (the 2D tiles they
    replace stored 32-byte rows of 16 bf16)."""
    problem = _ALL["19_ReLU"]
    assert problem.block_sizes == ELEMENTWISE_BLOCKS == [1, 8192]
    text, lowered = lower_xpu_stages(problem.kernel, _args(problem))
    _check_launch(text, (48, 4096))
    _check_elementwise_1d(text)
    device = lowered[lowered.index("gpu.module") :]
    assert (
        len(re.findall(r"llvm\.load %\w+ : !llvm\.ptr<1> -> vector<8xbf16>", device))
        == 8
    )
    assert (
        len(
            re.findall(
                r"llvm\.store %\w+, %\w+ : vector<8xbf16>, !llvm\.ptr<1>", device
            )
        )
        == 8
    )
    assert "intel_sub_group_2d_block" not in device


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "block_sizes", [ATTENTION_BLOCKS, [1, 1, 32, 32]], ids=["xpu_blocks", "small"]
)
def test_attention_matches_sdpa_on_cpu(
    dtype: torch.dtype, block_sizes: list[int]
) -> None:
    """97's flash-attention source on the CPU target, at a small shape: the
    online softmax over K/V tiles gives ``scaled_dot_product_attention``."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=block_sizes),
    )(attention)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 128, 64, dtype=dtype) for _ in range(3))
    out = kernel(q, k, v)
    assert out.dtype == dtype
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    # bf16: P and the output are rounded to bf16 (8-bit mantissa).
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), expected, atol=tolerance, rtol=tolerance)


def _reference(name: str) -> Callable[..., torch.Tensor]:
    """The KernelBench level 2 model's forward after its ``nn.Linear``."""
    references: dict[str, Callable[..., torch.Tensor]] = {
        "9": lambda y, sub, mul: torch.relu((y - sub) * mul),
        "12": lambda y, mul, slope: F.leaky_relu(y * mul, slope),
        "29": lambda y: F.mish(F.mish(y)),
        "40": lambda y, scale: y * scale + y.clone().detach(),
        "59": lambda y, scale: y * torch.sigmoid(y) * scale,
        "63": lambda y, divisor: torch.relu(y) / divisor,
        "70": lambda y, scale: torch.sigmoid(y) * scale + y,
        "81": lambda y: torch.clamp(
            torch.tanh(torch.clamp(y * torch.sigmoid(y) / 2.0, -1.0, 1.0)), -1.0, 1.0
        ),
    }
    return references[name.split("_")[0]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", [problem.name for problem in LEVEL2_PROBLEMS])
def test_level2_matches_eager_on_cpu(name: str, dtype: torch.dtype) -> None:
    """A level 2 kernel's source on the CPU target, at a small shape and block
    sizes: ``epilogue(x @ weight.T + bias)`` as the KernelBench model computes it
    (in f32, from the same inputs)."""
    problem = _ALL[name]
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[32, 64, 32]),
    )(problem.fn)
    torch.manual_seed(0)
    # Linear outputs of about N(0, 2.3): every epilogue branch is taken (9:
    # ``relu(y - 2)`` is non-zero for 18% of them, 81 clamps).
    x = torch.randn(64, 256, dtype=dtype)
    weight = (torch.rand(128, 256, dtype=dtype) * 2 - 1) / 4
    bias = torch.rand(128, dtype=dtype) * 2 - 1
    out = kernel(*problem.args(x, weight, bias))
    assert out.dtype == dtype
    linear = F.linear(x.float(), weight.float(), bias.float())
    expected = _reference(name)(linear, *problem.scalars)
    # bf16: one rounding of the output (8-bit mantissa).
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), expected, atol=tolerance, rtol=tolerance)
