"""KernelBench level 1 problems on the XPU target at the IR level, at the shapes
and dtypes of AI-bench's ``mlir-gpu-bench`` variant (fake XPU tensors: the large
shapes cost nothing). Each problem (``kernelbench_xpu.PROBLEMS``) lowers to
annotated XeGPU (``xegpu-wg``) with its kernel class's ops and one launch over
the expected workgroup grid, and on to LLVM/XeVM (``xevm``) without XeGPU ops
left.
"""

from __future__ import annotations

import re

import pytest

from tests.harness import lower_xpu
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import PROBLEMS
from tests.kernelbench_xpu import Problem

_LAYOUT = re.compile(r"layout = #xegpu\.layout<([^>]*)>")


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
}


@pytest.mark.isolated
@pytest.mark.parametrize("name", [problem.name for problem in PROBLEMS])
def test_kernelbench_problem_lowers_for_xpu(name: str) -> None:
    (problem,) = [problem for problem in PROBLEMS if problem.name == name]
    text = lower_xpu(problem.kernel, _args(problem), "xegpu-wg")
    _check_launch(text, problem.grid)
    _CHECKS[problem.kind](text)
    lowered = lower_xpu(problem.kernel, _args(problem), "xevm")
    assert "xegpu." not in lowered
    assert "llvm.func spir_funccc" in lowered
    assert re.search(r"gpu\.launch_func <%\w+ : !llvm\.ptr>", lowered)
    # No math on bf16 left for the device (no such OpenCL builtins).
    assert not re.search(r"llvm\.func @_Z\w+\([^)]*bf16", lowered)
    if problem.kind == "reduction":
        # The anchored loads: 8-row block reads, not one row each.
        assert _block_read_rows(lowered) == {8}
