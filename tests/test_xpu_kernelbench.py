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

from tests.harness import lower_xpu
from tests.harness import lower_xpu_stages
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import ATTENTION_BLOCKS
from tests.kernelbench_xpu import ELEMENTWISE_BLOCKS
from tests.kernelbench_xpu import GEMM_SMALL_K
from tests.kernelbench_xpu import GEMM_SPLIT_K
from tests.kernelbench_xpu import LEVEL2_PROBLEMS
from tests.kernelbench_xpu import PROBLEMS
from tests.kernelbench_xpu import RANK3_PROBLEMS
from tests.kernelbench_xpu import ROW_REDUCTIONS
from tests.kernelbench_xpu import Problem
from tests.kernelbench_xpu import _matmul
from tests.kernelbench_xpu import _matmul_split_k
from tests.kernelbench_xpu import _segment_square_sums
from tests.kernelbench_xpu import _sum_splits
from tests.kernelbench_xpu import attention
from tests.kernelbench_xpu import row_reduction_kernels
from tests.kernelbench_xpu import row_segments
from tests.kernelbench_xpu import split_k_views

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


def _gemm_kernel(fn: Callable[..., object], blocks: list[int], **xegpu: object):
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, **xegpu),
    )(fn)


@pytest.mark.isolated
@pytest.mark.parametrize(
    "name", ["7_Matmul_with_small_K_dimension_", "9_Tall_skinny_matrix_multiplication_"]
)
def test_gemm_small_k_config(name: str) -> None:
    """7 and 9 (K padded to 64) run ``GEMM_SMALL_K``: 64x128 workgroups of 8x2
    subgroups of 8x64 (256 threads), stored in 8x16 blocks, the grid columns
    first (the output dominates: 256 N tiles in x, 512 M tiles in y)."""
    problem = _ALL[name]
    assert problem.block_sizes == GEMM_SMALL_K[0]
    assert problem.shapes == ((32768, 64), (64, 32768))
    text, lowered = lower_xpu_stages(problem.kernel, _args(problem))
    _check_launch(text, (256, 512))
    layout = "sg_layout = [8, 2], sg_data = [8, 64], inst_data = [8, 16]"
    assert _anchor_layouts(text, "xegpu.store_nd") == [layout]
    assert "known_block_size = array<i32: 256, 1, 1>" in text
    assert "intel_sub_group_2d_block_write_16b_8r16x1c" in lowered


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("mnk", "grid"),
    [
        # Writes 2^22 elements, reads 2^22: the boundary maps columns first.
        ((2048, 2048, 1024), (16, 32)),
        # Reads 2^23 elements (K = 2048) for 2^22 written: rows first.
        ((2048, 2048, 2048), (32, 16)),
    ],
)
def test_gemm_grid_order_follows_traffic(
    mnk: tuple[int, int, int], grid: tuple[int, int]
) -> None:
    """The same kernel and config (``GEMM_SMALL_K``) maps its grid columns
    first only while it writes at least as many elements as it reads
    (``xpu_params.columns_first``: M * N >= K * (M + N)); otherwise rows first
    (M tiles in x), the order of the operand-bound GEMMs (1, 2, 13, 16-18 keep
    their grids in ``test_kernelbench_problem_lowers_for_xpu``)."""
    m, n, k = mnk
    kernel = _gemm_kernel(_matmul, GEMM_SMALL_K[0], **GEMM_SMALL_K[1])
    args = [
        xpu_tensor(m, k, dtype=torch.bfloat16),
        xpu_tensor(k, n, dtype=torch.bfloat16),
    ]
    text, _ = lower_xpu_stages(kernel, args)
    _check_launch(text, grid)


@pytest.mark.isolated
def test_split_k_gemm_lowers() -> None:
    """6_large_K's split K (``GEMM_SPLIT_K``) over contiguous chunk views: the
    K chunk ``a[tm, s, tk]`` of ``A.view(M, S, K/S)`` (a unit dim inside the
    tile, ``fold_slice_unit_dims``) is read from a ``256x16384`` view whose
    rows are K apart (``rank_reduce_transfers``), and the kernel lowers as a
    256x256 GEMM per chunk (32 workgroups, 32x64 subgroup tiles, A read 32x32
    per block read, f32 stored in 8x16 blocks) with the rows-first grid (it
    reads far more than it writes)."""
    splits, blocks, xegpu, _ = GEMM_SPLIT_K
    a = xpu_tensor(256, 524288, dtype=torch.bfloat16)
    b = xpu_tensor(524288, 256, dtype=torch.bfloat16)
    kernel = _gemm_kernel(_matmul_split_k, blocks, **xegpu)
    text, lowered = lower_xpu_stages(kernel, list(split_k_views(a, b, splits)))
    _check_launch(text, (splits,))
    assert re.search(
        r"memref\.collapse_shape [^\n]*into memref<256x16384xbf16, "
        r"strided<\[524288, 1\], offset: \?>>",
        text,
    )
    (store,) = _anchor_layouts(text, "xegpu.store_nd")
    assert store == "sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]"
    assert re.search(r"xegpu\.store_nd [^\n]*vector<256x256xf32>", text)
    assert "intel_sub_group_2d_block_read_16b_32r16x2c" in lowered
    assert "intel_sub_group_2d_block_write_32b_8r16x1c" in lowered
    assert "xegpu." not in lowered


@pytest.mark.isolated
def test_sum_splits_lowers() -> None:
    """Regression (found by the 6_large_K split-K model): ``_sum_splits`` of
    ``[32, 256, 256]`` f32 partial sums with ``GEMM_SPLIT_K``'s 8x256 tiles. Its
    first load, ``parts[0, tm, tn]``, reads a tile at constant leading index 0,
    whose matrix view (``rank_reduce_transfers``) has the identity layout;
    the collapse_shape was built with the equal ``strided<[256, 1]>`` layout
    and failed to verify in 'gpu-outlining'."""
    splits, _, _, reduce_blocks = GEMM_SPLIT_K
    kernel = _gemm_kernel(_sum_splits, reduce_blocks)
    text, lowered = lower_xpu_stages(
        kernel, [xpu_tensor(splits, 256, 256, dtype=torch.float32)]
    )
    _check_launch(text, (32,))
    assert "memref.collapse_shape" in text
    assert "xegpu." not in lowered


_ROW_SHAPES = {
    "23_Softmax": (4096, 393216),
    "24_LogSoftmax": (4096, 393216),
    "38_L1Norm_": (32768, 65536),
    "39_L2Norm_": (32768, 65536),
}
"""The ``mlir-gpu-bench`` shapes of the row reductions."""


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(ROW_REDUCTIONS))
def test_row_reduction_three_kernels_lower(name: str) -> None:
    """The models' three kernels of each row reduction (``ROW_REDUCTIONS``) at
    the spec shape: partials of 2048-column segments in lighthouse's
    reduction layout (8x16 block reads over the ``[M * S, 2048]`` view, a
    workgroup per 64 segments), the row results over ``[M, S]``, and the output
    on one-row tiles of 8192 columns (1D gathers and scatters, the column
    tiles first; regression: its ``[M, 1]`` row column, a 0-D read per
    workgroup, crashed ``rank_reduce_transfers``)."""
    m, n = _ROW_SHAPES[name]
    segments = row_segments((m, n))
    assert segments == n // 2048
    fns, blocks = row_reduction_kernels(name, (m, n))
    kernels = [_gemm_kernel(fn, list(b)) for fn, b in zip(fns, blocks, strict=True)]
    view = xpu_tensor(m * segments, n // segments, dtype=torch.float32)
    text, lowered = lower_xpu_stages(kernels[0], [view])
    _check_launch(text, (m * segments // 64,))
    tile = "sg_layout = [8, 1], sg_data = [8, 64], inst_data = [8, 16]"
    assert set(_anchor_layouts(text, "xegpu.load_nd")) == {tile}
    assert "intel_sub_group_2d_block_read_32b_8r16x1c" in lowered
    assert "xegpu." not in lowered
    parts = [xpu_tensor(m, segments, dtype=torch.float32)]
    if "Softmax" in name:
        parts *= 2
    extra = [n] if name == "38_L1Norm_" else []
    text, lowered = lower_xpu_stages(kernels[1], [*parts, *extra])
    _check_launch(text, (m // 64,))
    assert "vector.multi_reduction" in text
    assert "xegpu." not in lowered
    x = xpu_tensor(m, n, dtype=torch.float32)
    column = xpu_tensor(m, 1, dtype=torch.float32)
    columns = [column, column] if "Softmax" in name else [column]
    text, lowered = lower_xpu_stages(kernels[2], [x, *columns, x])
    _check_launch(text, (n // 8192, m))
    layout = "sg_layout = [8], sg_data = [1024], inst_data = [16]"
    assert _anchor_layouts(text, "xegpu.load") == [layout]
    assert _anchor_layouts(text, "xegpu.store") == [layout]
    device = lowered[lowered.index("gpu.module") :]
    assert "intel_sub_group_2d_block" not in device
    assert "xegpu." not in lowered


@pytest.mark.parametrize(
    ("shape", "segments"),
    [
        ((4096, 393216), 192),
        ((32768, 65536), 32),
        ((64, 4096), 16),  # 256-column segments
        ((128, 1024), None),  # 8 segments of 128: too few block columns
        ((16, 65536), 32),  # two workgroups of 8 rows for the row results
        ((8, 65536), None),  # one: fails to lower, the single kernels instead
        ((100, 65536), None),  # rows not a multiple of 8
        ((1 << 21, 32768), None),  # 2^25 or more segment rows
    ],
)
def test_row_segments(shape: tuple[int, int], segments: int | None) -> None:
    assert row_segments(shape) == segments


@pytest.mark.parametrize(
    ("rows", "fails"),
    [(1 << 24, False), (25165824, True)],
    ids=["2^24_rows", "1.5x2^24_rows"],
)
def test_block_surface_rows_limit(rows: int, fails: bool) -> None:
    """Regression: 2D block reads of a surface of more than 2^24 rows wrap the
    payload's 24-bit height on hardware (B50: a row sum of ``25165824 x 64``
    f32 read rows from 2^23 on as zeros, no error); such kernels are refused
    (``xegpu_block_surfaces``) and 2^24 rows still lower."""
    kernel = _gemm_kernel(_segment_square_sums, [64, 64])
    args = [xpu_tensor(rows, 64, dtype=torch.float32)]
    if not fails:
        lower_xpu(kernel, args)
        return
    with pytest.raises(UnsupportedOperationError, match="2D block surface"):
        lower_xpu(kernel, args)


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
