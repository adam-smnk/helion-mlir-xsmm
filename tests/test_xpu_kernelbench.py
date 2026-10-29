"""KernelBench level 1 problems on the XPU target at the IR level, at the shapes
and dtypes of AI-bench's ``mlir-gpu-bench`` variant (fake XPU tensors: the large
shapes cost nothing). Each problem (``kernelbench_xpu.PROBLEMS``) lowers to
annotated XeGPU (``xegpu-wg``) with its kernel class's ops and one launch over
the expected workgroup grid, and on to LLVM/XeVM (``xevm``) without XeGPU ops
left. Rank-3+ problems (``RANK3_PROBLEMS``: block size 1 on the leading dims)
lower as their 2D kernel class: 2D XeGPU tiles, the leading dims as grid dims.
Attention (97, written as flash attention) lowers with lighthouse's attention
layout, both matmuls on DPAS; the same kernel source runs on the CPU target
against PyTorch's SDPA (no XPU needed).
"""

from __future__ import annotations

import re

import helion
import pytest
import torch
import torch.nn.functional as F

from tests.harness import lower_xpu
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import ATTENTION_BLOCKS
from tests.kernelbench_xpu import PROBLEMS
from tests.kernelbench_xpu import RANK3_PROBLEMS
from tests.kernelbench_xpu import Problem
from tests.kernelbench_xpu import attention

_LAYOUT = re.compile(r"layout = #xegpu\.layout<([^>]*)>")
_TENSOR_DESC = re.compile(r"!xegpu\.tensor_desc<((?:\d+x)+)\w+")
_ALL = {problem.name: problem for problem in (*PROBLEMS, *RANK3_PROBLEMS)}


def _args(problem: Problem) -> list[object]:
    return [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]


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
    loads = _anchor_layouts(text, "xegpu.load_nd")
    stores = _anchor_layouts(text, "xegpu.store_nd")
    assert len(loads) == len(stores) == 1
    # Inferred for a 128x256 bf16 tile: 32 subgroups of 32x32, 32x32 block
    # loads, 8x16 block stores.
    tile = "sg_layout = [4, 8], sg_data = [32, 32]"
    assert loads == [f"{tile}, inst_data = [32, 32]"]
    assert stores == [f"{tile}, inst_data = [8, 16]"]
    assert "known_block_size = array<i32: 512, 1, 1>" in text


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
}


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(_ALL))
def test_kernelbench_problem_lowers_for_xpu(name: str) -> None:
    problem = _ALL[name]
    text = lower_xpu(problem.kernel, _args(problem), "xegpu-wg")
    _check_launch(text, problem.grid)
    _CHECKS[problem.kind](text)
    # Every tile is 2D, also of rank-3+ problems (unit leading dims folded).
    ranks = {dims.count("x") for dims in _TENSOR_DESC.findall(text)}
    assert ranks == {2}, ranks
    lowered = lower_xpu(problem.kernel, _args(problem), "xevm")
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
