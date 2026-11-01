"""XeGPU parameters from the config (``xegpu_*`` keys), per GPU kernel, at the IR
level: no XPU, no SPIR-V target needed.

Each GPU kernel (top-level grid loop) gets its own forced parameters; ``None``
entries and kernels past the end of a list keep lighthouse's inference. Invalid
parameters fail before lowering with ``XeGPUConfigError``; the CPU target
ignores the keys.
"""

from __future__ import annotations

import re

import helion
from helion.autotuner import config_spec as helion_config_spec
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import lower_xpu_stages
from tests.harness import xpu_tensor

from helion_mlir_backend import compile_mlir
from helion_mlir_backend import generate_mlir
from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.support import MLIRBackendError
from helion_mlir_backend._compiler.mlir.support import XeGPUConfigError
from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import (
    elementwise_layout,
)
from helion_mlir_backend._compiler.mlir_transforms.xegpu_reduction import (
    reduction_tiles,
)
from helion_mlir_backend._compiler.targets import get_target
from helion_mlir_backend._compiler.targets import target_config_keys
from helion_mlir_backend._compiler.targets.xpu_params import PARAMS_ATTR
from helion_mlir_backend._compiler.targets.xpu_params import XEGPU_CONFIG_KEYS
from helion_mlir_backend._compiler.targets.xpu_params import ElementwiseShape
from helion_mlir_backend._compiler.targets.xpu_params import GemmShape
from helion_mlir_backend._compiler.targets.xpu_params import KernelParams
from helion_mlir_backend._compiler.targets.xpu_params import ReductionShape
from helion_mlir_backend._compiler.targets.xpu_params import attach_params
from helion_mlir_backend._compiler.targets.xpu_params import columns_first
from helion_mlir_backend._compiler.targets.xpu_params import take_params
from helion_mlir_backend._compiler.targets.xpu_params import xegpu_params

N = 1024
GEMM_BLOCKS = [256, 256, 32]
TWO_PHASE_BLOCKS = [256, 256, 32, 256, 256, 32]
SOFTMAX_BLOCKS = [64, 64]


def _kernel(fn: object) -> object:
    return helion.kernel(backend="mlir", static_shapes=True)(fn)


@_kernel
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


@_kernel
def two_phase_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``(a @ b) @ c`` in two phases: two GPU kernels."""
    m, k = a.size()
    _, n = b.size()
    tmp = torch.empty([m, n], dtype=torch.float16, device=a.device)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        tmp[tm, tn] = acc.to(torch.float16)
    hl.barrier()
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = torch.addmm(acc, tmp[tm, tk], c[tk, tn])
        out[tm, tn] = acc
    return out


@_kernel
def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row softmax in three passes over column tiles (max, sum of exps, output)."""
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


@_kernel
def softmax_in_f32(x: torch.Tensor) -> torch.Tensor:
    """Row softmax of any float type, computed in f32."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            v = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top) * inv
            out[tile_m, tile_n] = v.to(x.dtype)
    return out


@_kernel
def matmul_softmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row softmax of ``a @ b``: a GEMM kernel, then a reduction kernel."""
    m, k = a.size()
    _, n = b.size()
    scores = torch.empty([m, n], dtype=torch.float32, device=a.device)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    block_n = hl.register_block_size(n)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        scores[tm, tn] = acc
    hl.barrier()
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, scores[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(scores[tile_m, tile_n] - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = torch.exp(scores[tile_m, tile_n] - row_top) * inv
    return out


MATMUL_SOFTMAX_BLOCKS = [64, 256, 256, 32, 64]
"""block_n (softmax columns), the GEMM's M, N and K tiles, softmax rows."""


def _gemm_args() -> list[object]:
    return [xpu_tensor(N, N, dtype=torch.bfloat16) for _ in range(2)]


def _two_phase_args() -> list[object]:
    return [xpu_tensor(N, N, dtype=torch.float16) for _ in range(3)]


def _lower(
    kernel: object,
    args: list[object],
    blocks: list[int],
    stop_at: str = "xegpu-wg",
    **xegpu: object,
) -> str:
    config = helion.Config(block_sizes=blocks, **xegpu)
    return str(lower_mlir(kernel, args, config=config, target="xpu", stop_at=stop_at))


def _lower_stages(
    kernel: object, args: list[object], blocks: list[int], **xegpu: object
) -> tuple[str, str]:
    """:func:`_lower` to ``xegpu-wg`` and to ``xevm``, from one lowering."""
    config = helion.Config(block_sizes=blocks, **xegpu)
    return lower_xpu_stages(kernel, args, config=config)


@pytest.fixture(autouse=True)
def xpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


# --- IR checks ---------------------------------------------------------------


def _kernels(text: str) -> dict[str, str]:
    """The text of each ``gpu.module``, by name."""
    parts = re.split(r"(?=gpu\.module @)", text)[1:]
    return {re.match(r"gpu\.module @(\w+)", part).group(1): part for part in parts}


def _layout(sg_layout: list[int], sg_data: list[int], inst_data: list[int]) -> str:
    return f"sg_layout = {sg_layout}, sg_data = {sg_data}, inst_data = {inst_data}"


def _op_layouts(kernel: str, op: str) -> set[str]:
    """The anchor layouts of the ``op`` ops of a kernel."""
    return set(re.findall(rf"{re.escape(op)} [^\n]*#xegpu\.layout<([^>]*)>", kernel))


def _distribution(layouts: set[str]) -> set[str]:
    """``layouts`` without their ``inst_data`` (per-instruction tiles)."""
    return {layout.split(", inst_data")[0] for layout in layouts}


def _launch_threads(text: str) -> list[int]:
    """The thread count of each launch, in program order."""
    return [
        int(threads)
        for threads in re.findall(
            r"gpu\.launch_func\s+@\w+::@\w+ blocks in \([^)]*\) threads in \(%c(\d+),",
            text,
        )
    ]


def _known_threads(kernel: str) -> int:
    return int(re.search(r"known_block_size = array<i32: (\d+),", kernel).group(1))


def _check_xevm(text: str, launches: int) -> None:
    assert "xegpu." not in text
    assert "llvm.func spir_funccc" in text
    assert text.count("gpu.launch_func") == launches


def _block_reads(text: str) -> set[str]:
    """The 2D block reads of lowered device code, as ``<bits>b_<rows>r<cols>``."""
    return set(re.findall(r"intel_sub_group_2d_block_read_(\d+b_\d+r\d+)x", text))


DEFAULT_GEMM = {
    # wg 256x256, k 32: lighthouse infers sg [32, 64] (32 subgroups).
    "c": _layout([8, 4], [32, 64], [8, 16]),
    "a": _layout([8, 4], [32, 32], [8, 16]),
    "b": _layout([8, 4], [32, 64], [16, 16]),
    "prefetch_a": _layout([16, 2], [16, 16], [8, 16]),
    "prefetch_b": _layout([4, 8], [8, 32], [8, 16]),
}


# --- Forced parameters ---------------------------------------------------------


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("blocks", "sg_tile", "sg_layout", "threads"),
    [
        (GEMM_BLOCKS, [32, 64], [8, 4], 512),  # what lighthouse infers
        (GEMM_BLOCKS, [64, 32], [4, 8], 512),
        ([256, 128, 32], [32, 32], [8, 4], 512),
        ([128, 256, 32], [32, 64], [4, 4], 256),
    ],
    ids=["32x64", "64x32", "wg256x128-32x32", "wg128x256-32x64"],
)
def test_forced_subgroup_tile(
    blocks: list[int], sg_tile: list[int], sg_layout: list[int], threads: int
) -> None:
    text = _lower(matmul, _gemm_args(), blocks, xegpu_sg_tiles=[sg_tile])
    (kernel,) = _kernels(text).values()
    output = _layout(sg_layout, sg_tile, [8, 16])
    assert _op_layouts(kernel, "xegpu.store_nd") == {output}
    assert output in _op_layouts(kernel, "xegpu.dpas")
    # A loads (sg_m, k_tile) tiles, B (k_tile, sg_n) ones.
    assert _distribution(_op_layouts(kernel, "xegpu.load_nd")) == {
        f"sg_layout = {sg_layout}, sg_data = {[sg_tile[0], 32]}",
        f"sg_layout = {sg_layout}, sg_data = {[32, sg_tile[1]]}",
    }
    assert _known_threads(kernel) == threads
    assert _launch_threads(text) == [threads]


@pytest.mark.isolated
def test_forced_load_and_prefetch_tiles() -> None:
    text = _lower(
        matmul,
        _gemm_args(),
        GEMM_BLOCKS,
        xegpu_load_tiles=[[16, 16, 32, 16]],
        xegpu_prefetch_tiles=[[32, 16, 16, 32]],
        xegpu_prefetch_depths=[[2, 3]],
    )
    (kernel,) = _kernels(text).values()
    # The inferred subgroup tile (sg_layout, sg_data) stays; inst_data is forced.
    assert _op_layouts(kernel, "xegpu.load_nd") == {
        _layout([8, 4], [32, 32], [16, 16]),
        _layout([8, 4], [32, 64], [32, 16]),
    }
    # Prefetch tiles: A (256, 32) in 32x16 tiles, B (32, 256) in 16x32 tiles.
    assert _op_layouts(kernel, "xegpu.prefetch_nd") == {
        _layout([8, 2], [32, 16], [8, 16]),
        _layout([2, 8], [16, 32], [8, 16]),
    }
    # Prefetch depths: A and B prefetches ahead of the K loop.
    prologue = kernel[: kernel.index("scf.for")]
    assert prologue.count("xegpu.prefetch_nd %") == 2 + 3
    assert _launch_threads(text) == [512]


@pytest.mark.isolated
def test_parameters_per_kernel() -> None:
    """Each phase's kernel gets its own parameters; ``None`` keeps inference."""
    text, lowered = _lower_stages(
        two_phase_matmul,
        _two_phase_args(),
        [128, 256, 32, 256, 256, 32],
        xegpu_sg_tiles=[[32, 64], None],
        xegpu_load_tiles=[None, [16, 16, 32, 16]],
    )
    _check_xevm(lowered, launches=2)
    first, second = _kernels(text).values()
    # Kernel 0 (WG tile 128x256): forced subgroup tile, inferred load tiles; 16
    # subgroups.
    assert _op_layouts(first, "xegpu.store_nd") == {_layout([4, 4], [32, 64], [8, 16])}
    assert _distribution(_op_layouts(first, "xegpu.load_nd")) == {
        "sg_layout = [4, 4], sg_data = [32, 32]",
        "sg_layout = [4, 4], sg_data = [32, 64]",
    }
    # Kernel 1: inferred subgroup tile, forced load tiles; 32 subgroups.
    assert _op_layouts(second, "xegpu.store_nd") == {DEFAULT_GEMM["c"]}
    assert _op_layouts(second, "xegpu.load_nd") == {
        _layout([8, 4], [32, 32], [16, 16]),
        _layout([8, 4], [32, 64], [32, 16]),
    }
    assert _launch_threads(text) == [256, 512]
    assert (_known_threads(first), _known_threads(second)) == (256, 512)


@pytest.mark.isolated
def test_shorter_lists_and_none_entries_infer() -> None:
    """Without forced parameters the layouts are lighthouse's inference."""
    default = _lower(two_phase_matmul, _two_phase_args(), TWO_PHASE_BLOCKS)
    nones = _lower(
        two_phase_matmul,
        _two_phase_args(),
        TWO_PHASE_BLOCKS,
        xegpu_sg_tiles=[None, None],
        xegpu_prefetch_depths=[None],
    )
    assert nones == default
    for kernel in _kernels(default).values():
        assert _op_layouts(kernel, "xegpu.store_nd") == {DEFAULT_GEMM["c"]}
        assert _op_layouts(kernel, "xegpu.prefetch_nd") == {
            DEFAULT_GEMM["prefetch_a"],
            DEFAULT_GEMM["prefetch_b"],
        }
    # Only the second kernel is forced: the first one keeps its inference.
    second_only = _lower(
        two_phase_matmul,
        _two_phase_args(),
        TWO_PHASE_BLOCKS,
        xegpu_sg_tiles=[None, [64, 32]],
    )
    first, second = _kernels(second_only).values()
    assert first == next(iter(_kernels(default).values()))
    assert _op_layouts(second, "xegpu.store_nd") == {_layout([4, 8], [64, 32], [8, 16])}


@pytest.mark.isolated
@pytest.mark.parametrize(("rows", "threads"), [(16, 64), (64, 16), (None, 128)])
def test_reduction_subgroup_rows(rows: int | None, threads: int) -> None:
    """A reduction's subgroup tile is ``[rows, 0]`` (default: 8 rows)."""
    sg_tiles = [[rows, 0]] if rows is not None else [None]
    text, lowered = _lower_stages(
        softmax, [xpu_tensor(4096, 4096)], SOFTMAX_BLOCKS, xegpu_sg_tiles=sg_tiles
    )
    _check_xevm(lowered, launches=1)
    (kernel,) = _kernels(text).values()
    rows = rows or 8
    assert f"sg_layout = [{64 // rows}, 1], sg_data = [{rows}, 64]" in kernel
    assert _known_threads(kernel) == threads
    assert _launch_threads(text) == [threads]


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("dtype", "xegpu", "rows", "load_tile", "reads"),
    [
        (torch.float32, {}, 8, [8, 16], {"32b_8r16"}),
        (torch.float32, {"xegpu_sg_tiles": [[32, 0]]}, 32, [16, 16], {"32b_16r16"}),
        (
            torch.float32,
            {"xegpu_sg_tiles": [[32, 0]], "xegpu_load_tiles": [[16, 16]]},
            32,
            [16, 16],
            {"32b_16r16"},
        ),
        (
            torch.float32,
            {"xegpu_sg_tiles": [[32, 0]], "xegpu_load_tiles": [[8, 16]]},
            32,
            [8, 16],
            {"32b_8r16"},
        ),
        (torch.bfloat16, {}, 8, [8, 16], {"16b_8r16"}),
        (
            torch.bfloat16,
            {"xegpu_sg_tiles": [[32, 0]], "xegpu_load_tiles": [[16, 16]]},
            32,
            [16, 16],
            {"16b_16r16"},
        ),
    ],
    ids=[
        "f32-inferred",
        "f32-sg32",
        "f32-sg32-load-16x16",
        "f32-sg32-load-8x16",
        "bf16-inferred",
        "bf16-sg32-load-16x16",
    ],
)
def test_reduction_load_tiles(
    dtype: torch.dtype,
    xegpu: dict[str, object],
    rows: int,
    load_tile: list[int],
    reads: set[str],
) -> None:
    """A reduction's tile loads take its subgroup layout and a 2D block load
    tile (inferred: the largest of 16 columns and at most 16 elements per work
    item dividing the subgroup tile); its tile stores the 2D block store
    tile."""
    args = [xpu_tensor(4096, 4096, dtype=dtype)]
    text, lowered = _lower_stages(softmax_in_f32, args, SOFTMAX_BLOCKS, **xegpu)
    (kernel,) = _kernels(text).values()
    sg_layout, sg_data = [64 // rows, 1], [rows, 64]
    assert _op_layouts(kernel, "xegpu.load_nd") == {
        _layout(sg_layout, sg_data, load_tile)
    }
    assert _op_layouts(kernel, "xegpu.store_nd") == {
        _layout(sg_layout, sg_data, [8, 16])
    }
    _check_xevm(lowered, 1)
    assert _block_reads(lowered) == reads


@pytest.mark.parametrize(
    ("sg_data", "bits", "load_tile", "expected"),
    [
        ((8, 64), (32,), None, ((8, 16), (8, 16))),
        # At most 16 elements per work item: 16 rows of 16 columns.
        ((32, 64), (32,), None, ((16, 16), (8, 16))),
        ((64, 64), (32,), None, ((16, 16), (8, 16))),
        # 16 columns of 16-bit elements too (32-column loads came out wrong).
        ((8, 64), (16,), None, ((8, 16), (8, 16))),
        ((32, 64), (16,), None, ((16, 16), (8, 16))),
        ((8, 16), (16,), None, ((8, 16), (8, 16))),
        ((2, 64), (32,), None, ((2, 16), (2, 16))),
        ((32, 64), (16,), (16, 16), ((16, 16), (8, 16))),
    ],
)
def test_reduction_tile_inference(
    sg_data: tuple[int, int],
    bits: tuple[int, ...],
    load_tile: tuple[int, int] | None,
    expected: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    assert reduction_tiles(sg_data, bits, load_tile) == expected


@pytest.mark.parametrize(
    ("sg_data", "bits", "match"),
    [
        ((8, 24), (32,), r"has 24 columns, not a multiple of 16"),
        ((8, 64), (8,), r"8-bit elements"),
        ((8, 0, 64), (32,), r"3D subgroup tile"),
    ],
)
def test_reduction_tile_limits(
    sg_data: tuple[int, ...], bits: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        reduction_tiles(sg_data, bits)


@pytest.mark.parametrize(
    ("sg_data", "bits", "load_tile", "match"),
    [
        # 32-column loads of 16-bit elements came out wrong on hardware.
        ((8, 64), (16,), (8, 32), r"cols in \[16\]"),
        # Float conversions of more than 16 elements per work item do not
        # compile to SPIR-V.
        ((32, 64), (32,), (32, 16), r"more than 16 elements per work item"),
    ],
)
def test_reduction_forced_load_tile_limits(
    sg_data: tuple[int, int],
    bits: tuple[int, ...],
    load_tile: tuple[int, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        reduction_tiles(sg_data, bits, load_tile)


@pytest.mark.isolated
@pytest.mark.parametrize("forced", [False, True])
def test_kernels_of_different_classes(forced: bool) -> None:
    """A GEMM kernel and a reduction kernel in one Helion kernel, each set up by
    its own class (lighthouse's stages need every kernel in one class)."""
    xegpu = {"xegpu_sg_tiles": [[64, 32], [16, 0]]} if forced else {}
    args = [xpu_tensor(N, N, dtype=torch.float16) for _ in range(2)]
    text, lowered = _lower_stages(matmul_softmax, args, MATMUL_SOFTMAX_BLOCKS, **xegpu)
    _check_xevm(lowered, launches=2)
    gemm, reduction = _kernels(text).values()
    assert "xegpu.dpas" in gemm and "xegpu.dpas" not in reduction
    sg_tile, sg_layout, rows = (
        ([64, 32], [4, 8], 16) if forced else ([32, 64], [8, 4], 8)
    )
    assert _op_layouts(gemm, "xegpu.store_nd") == {_layout(sg_layout, sg_tile, [8, 16])}
    assert f"sg_layout = [{64 // rows}, 1], sg_data = [{rows}, 64]" in reduction
    assert _launch_threads(text) == [512, 64 // rows * 16]


@pytest.mark.isolated
@pytest.mark.parametrize(
    "xegpu",
    [
        {"xegpu_sg_tiles": [[64, 32]]},
        {
            "xegpu_sg_tiles": [[64, 32]],
            "xegpu_load_tiles": [[16, 16, 32, 32]],
            "xegpu_prefetch_tiles": [[32, 16, 16, 32]],
            "xegpu_prefetch_depths": [[2, 2]],
        },
    ],
    ids=["sg", "all"],
)
def test_forced_parameters_lower_to_xevm(xegpu: dict[str, object]) -> None:
    _check_xevm(_lower(matmul, _gemm_args(), GEMM_BLOCKS, "xevm", **xegpu), 1)


# --- Invalid parameters ----------------------------------------------------------


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("xegpu", "match"),
    [
        ({"xegpu_sg_tiles": [[48, 64]]}, r"sg_tile_m does not divide wg_tile_m"),
        ({"xegpu_sg_tiles": [[32, 32]]}, r"too many sg threads"),
        (
            {"xegpu_sg_tiles": [[64, 64]]},
            r"Number of parallel DPAS ops \(32\) exceeds",
        ),
        ({"xegpu_sg_tiles": [[0, 64]]}, r"subgroup tiles of a GEMM are positive"),
        (
            {"xegpu_load_tiles": [[64, 16, 16, 16]]},
            r"Load tile A \(64, 16\) does not divide the parent shape \(32, 32\)",
        ),
        (
            {"xegpu_sg_tiles": [[64, 32]], "xegpu_load_tiles": [[8, 16, 64, 16]]},
            r"Load tile B \(64, 16\) does not divide the parent shape \(32, 32\)",
        ),
        (
            {"xegpu_prefetch_tiles": [[4, 16, 16, 16]]},
            r"Prefetch tile A \(4, 16\) has too few rows",
        ),
        ({"xegpu_prefetch_depths": [[0, 1]]}, r"prefetch depths are positive"),
        ({"xegpu_sg_tiles": [[64]]}, r"xegpu_sg_tiles\[0\]=\[64\]: expected 2"),
        (
            {"xegpu_load_tiles": [[16, 16, 32]]},
            r"xegpu_load_tiles\[0\]=\[16, 16, 32\]: expected 4",
        ),
        ({"xegpu_sg_tiles": [64, 32]}, r"xegpu_sg_tiles\[0\]=64: expected 2"),
        ({"xegpu_sg_tiles": [[True, 32]]}, r"expected 2 non-negative ints"),
        ({"xegpu_sg_tiles": [[-8, 32]]}, r"expected 2 non-negative ints"),
        ({"xegpu_sg_tiles": "64x32"}, r"expected a list with one entry per GPU kernel"),
        (
            {"xegpu_sg_tiles": [None, [64, 32]]},
            r"lists of 2 entries, but the kernel has 1 GPU kernel",
        ),
    ],
    ids=[
        "sg-not-dividing",
        "sg-too-many-subgroups",
        "sg-too-many-dpas",
        "sg-zero",
        "load-a-too-big",
        "load-b-not-dividing",
        "prefetch-too-small",
        "prefetch-depth-zero",
        "sg-arity",
        "load-arity",
        "flat-list",
        "bool",
        "negative",
        "not-a-list",
        "too-many-entries",
    ],
)
def test_invalid_gemm_parameters(xegpu: dict[str, object], match: str) -> None:
    with pytest.raises(XeGPUConfigError, match=match) as info:
        _lower(matmul, _gemm_args(), GEMM_BLOCKS, **xegpu)
    error = info.value
    assert isinstance(error, MLIRBackendError) and isinstance(error, ValueError)
    # The autotuner skips such configs.
    assert MLIRBackend().classify_autotune_exception(error) == "warn"


@pytest.mark.isolated
def test_transposed_operands_follow_lighthouse_constraints() -> None:
    @_kernel
    def matmul_transposed_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.size()
        n, _ = b.size()
        out = torch.empty([m, n], dtype=torch.float32, device=a.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
            out[tm, tn] = acc
        return out

    args = [xpu_tensor(N, N, dtype=torch.float16) for _ in range(2)]
    with pytest.raises(XeGPUConfigError, match=r"transposed B.*must be \[16, 16\]"):
        _lower(
            matmul_transposed_b, args, GEMM_BLOCKS, xegpu_load_tiles=[[8, 16, 32, 16]]
        )
    text = _lower(
        matmul_transposed_b, args, GEMM_BLOCKS, xegpu_load_tiles=[[16, 16, 16, 16]]
    )
    (kernel,) = _kernels(text).values()
    assert "inst_data = [16, 16]" in kernel and "vector.transpose" in kernel


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("xegpu", "match"),
    [
        (
            {"xegpu_prefetch_tiles": [[8, 16, 16, 16]]},
            r"xegpu_prefetch_tiles apply to GEMM kernels only",
        ),
        (
            {"xegpu_prefetch_depths": [[1, 1]]},
            r"xegpu_prefetch_depths apply to GEMM kernels only",
        ),
        ({"xegpu_sg_tiles": [[8, 8]]}, r"is \[rows, 0\] with rows > 0"),
        ({"xegpu_sg_tiles": [[48, 0]]}, r"48 rows do not divide the workgroup's 64"),
        (
            {"xegpu_load_tiles": [[8, 16, 16, 16]]},
            r"the load tile of a reduction kernel is \[m, n\]",
        ),
        (
            {"xegpu_load_tiles": [[16, 16]]},
            r"load tile \[16, 16\] does not divide the subgroup tile \[8, 64\]",
        ),
        (
            {"xegpu_sg_tiles": [[32, 0]], "xegpu_load_tiles": [[32, 32]]},
            r"load tile \[32, 32\]: 2D block loads of 32-bit .* cols in \[16\]",
        ),
        (
            {"xegpu_sg_tiles": [[64, 0]], "xegpu_load_tiles": [[64, 16]]},
            r"load tile \[64, 16\]: .* rows in \[1, 2, 4, 8, 16, 32\]",
        ),
        (
            {"xegpu_sg_tiles": [[16, 0]], "xegpu_load_tiles": [[12, 16]]},
            r"load tile \[12, 16\]: .* rows in \[1, 2, 4, 8, 16, 32\]",
        ),
    ],
    ids=[
        "prefetch-tiles",
        "prefetch-depths",
        "two-dims",
        "not-dividing",
        "load-arity",
        "load-rows-not-dividing",
        "f32-load-cols",
        "load-rows-too-many",
        "load-rows-not-power-of-two",
    ],
)
def test_invalid_reduction_parameters(xegpu: dict[str, object], match: str) -> None:
    with pytest.raises(XeGPUConfigError, match=match) as info:
        _lower(softmax, [xpu_tensor(4096, 4096)], SOFTMAX_BLOCKS, **xegpu)
    assert "reduction of 64 rows per workgroup" in str(info.value)


# --- Elementwise kernels ---------------------------------------------------------


@_kernel
def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v * torch.sigmoid(v)).to(x.dtype)
    return out


@_kernel
def silu_1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * torch.sigmoid(x[tile])
    return out


ELEMENTWISE_BLOCKS = [128, 256]


def _elementwise_args(dtype: torch.dtype = torch.bfloat16) -> list[object]:
    return [xpu_tensor(4096, 4096, dtype=dtype)]


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("xegpu", "load", "store", "threads"),
    [
        (
            {},
            _layout([4, 8], [32, 32], [16, 16]),
            _layout([4, 8], [32, 32], [8, 16]),
            512,
        ),
        (
            {"xegpu_sg_tiles": [[32, 64]]},
            _layout([4, 4], [32, 64], [16, 16]),
            _layout([4, 4], [32, 64], [8, 16]),
            256,
        ),
        (
            {"xegpu_sg_tiles": [[16, 64]], "xegpu_load_tiles": [[8, 16]]},
            _layout([8, 4], [16, 64], [8, 16]),
            _layout([8, 4], [16, 64], [8, 16]),
            512,
        ),
        (
            {"xegpu_load_tiles": [[8, 16]]},
            _layout([4, 8], [32, 32], [8, 16]),
            _layout([4, 8], [32, 32], [8, 16]),
            512,
        ),
    ],
    ids=["inferred", "sg-32x64", "sg-16x64-load-8x16", "load-8x16"],
)
def test_elementwise_parameters(
    xegpu: dict[str, object], load: str, store: str, threads: int
) -> None:
    text, lowered = _lower_stages(
        silu, _elementwise_args(), ELEMENTWISE_BLOCKS, **xegpu
    )
    (kernel,) = _kernels(text).values()
    assert _op_layouts(kernel, "xegpu.load_nd") == {load}
    assert _op_layouts(kernel, "xegpu.store_nd") == {store}
    assert _known_threads(kernel) == threads
    assert _launch_threads(text) == [threads]
    _check_xevm(lowered, 1)


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("xegpu", "dtype", "match"),
    [
        (
            {"xegpu_prefetch_tiles": [[8, 16, 16, 16]]},
            torch.bfloat16,
            r"xegpu_prefetch_tiles apply to GEMM kernels only",
        ),
        (
            {"xegpu_prefetch_depths": [[1, 1]]},
            torch.bfloat16,
            r"xegpu_prefetch_depths apply to GEMM kernels only",
        ),
        (
            {"xegpu_load_tiles": [[8, 16, 16, 16]]},
            torch.bfloat16,
            r"the load tile of an elementwise kernel is \[m, n\]",
        ),
        (
            {"xegpu_load_tiles": [[64, 32]]},
            torch.bfloat16,
            r"load tile \[64, 32\]: 2D block loads of 16-bit elements",
        ),
        (
            {"xegpu_load_tiles": [[8, 32]]},
            torch.float32,
            r"load tile \[8, 32\]: 2D block loads of 32-bit elements .* cols in \[16\]",
        ),
        (
            {"xegpu_load_tiles": [[16, 32]]},
            torch.bfloat16,
            r"load tile \[16, 32\]: 2D block loads of 16-bit elements .* cols in \[16\]",
        ),
        (
            {"xegpu_load_tiles": [[32, 16]]},
            torch.bfloat16,
            r"load tile \[32, 16\]: more than 16 elements per work item",
        ),
        (
            {"xegpu_load_tiles": [[16, 16]], "xegpu_sg_tiles": [[8, 128]]},
            torch.bfloat16,
            r"load tile \[16, 16\] does not divide the subgroup tile \[8, 128\]",
        ),
        (
            {"xegpu_sg_tiles": [[48, 32]]},
            torch.bfloat16,
            r"subgroup tile \[48, 32\] does not divide the workgroup tile",
        ),
        (
            {"xegpu_sg_tiles": [[8, 16]]},
            torch.bfloat16,
            r"256 subgroups exceed the maximum of 32",
        ),
        (
            {"xegpu_sg_tiles": [[32, 8]]},
            torch.bfloat16,
            r"the innermost size is a multiple of 16",
        ),
        (
            {"xegpu_sg_tiles": [[64, 64]]},
            torch.bfloat16,
            r"more than 2048 elements spill",
        ),
    ],
    ids=[
        "prefetch-tiles",
        "prefetch-depths",
        "gemm-load-tiles",
        "load-rows",
        "f32-load-cols",
        "bf16-load-cols",
        "load-per-work-item",
        "load-not-dividing",
        "sg-not-dividing",
        "too-many-subgroups",
        "sg-cols",
        "sg-too-large",
    ],
)
def test_invalid_elementwise_parameters(
    xegpu: dict[str, object], dtype: torch.dtype, match: str
) -> None:
    with pytest.raises(XeGPUConfigError, match=match) as info:
        _lower(silu, _elementwise_args(dtype), ELEMENTWISE_BLOCKS, **xegpu)
    assert "elementwise kernel (WG tile 128x256)" in str(info.value)


@pytest.mark.isolated
def test_gemm_rejects_elementwise_load_tiles() -> None:
    with pytest.raises(XeGPUConfigError, match=r"load tiles of a GEMM are \[a_m"):
        _lower(matmul, _gemm_args(), GEMM_BLOCKS, xegpu_load_tiles=[[8, 16]])


@pytest.mark.isolated
def test_1d_elementwise_load_tile() -> None:
    """A 1D elementwise kernel takes a load tile ``[16 x chunk]`` (the chunk of
    its loads and stores; inferred: 16 bytes of f32, 4), nothing else."""
    args = [xpu_tensor(1 << 20)]
    text = _lower(silu_1d, args, [1024], xegpu_load_tiles=[[32]])
    layout = "sg_layout = [1], sg_data = [1024], inst_data = [32]"
    assert _op_layouts(text, "xegpu.load") == {layout}
    assert _op_layouts(text, "xegpu.store") == {layout}


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("xegpu", "match"),
    [
        (
            {"xegpu_load_tiles": [[16, 16]]},
            r"load tile of a 1D elementwise kernel is \[n\]",
        ),
        (
            {"xegpu_sg_tiles": [[1, 512]]},
            r"1D elementwise kernels take only a load tile",
        ),
        (
            {"xegpu_load_tiles": [[128]]},
            r"load tile \[128\]: this kernel's 1D tiles take",
        ),
        ({"xegpu_load_tiles": [[48]]}, r"load tile \[48\]"),
    ],
    ids=["2d-load-tile", "sg-tile", "too-wide", "not-16-x-pow2"],
)
def test_invalid_1d_elementwise_parameters(
    xegpu: dict[str, object], match: str
) -> None:
    with pytest.raises(XeGPUConfigError, match=match):
        _lower(silu_1d, [xpu_tensor(1 << 20)], [1024], **xegpu)


@pytest.mark.parametrize(
    ("wg_tile", "load_bits", "store_bits", "sg_tile", "load_tile", "store_tile"),
    [
        # Loads of 16 columns, at most 16 elements per work item.
        ((128, 256), (16,), (16,), (32, 32), (16, 16), (8, 16)),
        ((128, 256), (32,), (32,), (32, 32), (16, 16), (8, 16)),
        ((128, 256), (16,), (32,), (32, 32), (16, 16), (8, 16)),
        ((32, 256), (16,), (16,), (32, 32), (16, 16), (8, 16)),
        ((256, 256), (16,), (16,), (32, 64), (16, 16), (8, 16)),
        ((1, 4096), (16,), (16,), (1, 1024), (1, 16), (1, 16)),
        ((1, 16), (16,), (16,), (1, 16), (1, 16), (1, 16)),
        ((4, 16), (32,), (32,), (4, 16), (4, 16), (4, 16)),
        ((96, 48), (16,), (16,), (24, 48), (8, 16), (8, 16)),
        ((2048,), (32,), (32,), (1024,), (16,), (16,)),
    ],
)
def test_elementwise_layout_inference(
    wg_tile: tuple[int, ...],
    load_bits: tuple[int, ...],
    store_bits: tuple[int, ...],
    sg_tile: tuple[int, ...],
    load_tile: tuple[int, ...],
    store_tile: tuple[int, ...],
) -> None:
    layout = elementwise_layout(wg_tile, load_bits, store_bits)
    assert (layout.sg_tile, layout.load_tile, layout.store_tile) == (
        sg_tile,
        load_tile,
        store_tile,
    )
    assert 1 <= layout.subgroups <= 32
    assert layout.threads == layout.subgroups * 16


@pytest.mark.parametrize(
    ("wg_tile", "bits", "match"),
    [
        ((64, 8), (16,), r"8 innermost elements, not a multiple of 16"),
        ((64, 64), (64,), r"64-bit elements"),
        ((2, 4, 16), (16,), r"3D tiles"),
        ((512, 512), (16,), r"no subgroup tile of at most 2048 elements divides"),
    ],
)
def test_elementwise_layout_limits(
    wg_tile: tuple[int, ...], bits: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        elementwise_layout(wg_tile, bits, bits)


# --- Config keys, recording, other targets -------------------------------------


def test_config_keys_are_registered() -> None:
    assert target_config_keys() == XEGPU_CONFIG_KEYS == get_target("xpu").config_keys
    assert get_target("cpu").config_keys == frozenset()
    assert XEGPU_CONFIG_KEYS <= helion_config_spec.VALID_KEYS
    backend = MLIRBackend()
    assert all(backend.supports_config_key(key) for key in XEGPU_CONFIG_KEYS)
    # Searched per kernel: the XPU target adds them to a kernel's config spec
    # (``xpu_search``), not to Helion's global backend tunables.
    assert not XEGPU_CONFIG_KEYS & helion_config_spec.BACKEND_TUNABLE_KEYS


def test_parameters_are_recorded_on_the_module() -> None:
    config = helion.Config(
        block_sizes=GEMM_BLOCKS,
        xegpu_sg_tiles=[[64, 32]],
        xegpu_prefetch_depths=[None, (2, 1)],
    )
    params = xegpu_params(config)
    assert params == (
        KernelParams(sg_tile=(64, 32)),
        KernelParams(prefetch_depths=(2, 1)),
    )
    assert xegpu_params(helion.Config(block_sizes=GEMM_BLOCKS)) == ()
    module = generate_mlir(matmul, _gemm_args(), config=config)
    plain = str(module)
    get_target("cpu").configure(module, config)
    assert str(module) == plain
    get_target("xpu").configure(module, config)
    # Modules differing in parameters differ in text (the JIT cache key).
    assert PARAMS_ATTR in str(module) and str(module) != plain
    assert take_params(module) == params
    assert str(module) == plain
    assert take_params(module) == ()
    with ir.Context(), ir.Location.unknown():
        empty = ir.Module.create()
    attach_params(empty, ())
    assert PARAMS_ATTR not in str(empty)


def test_annotate_options_map_to_lighthouse() -> None:
    params = KernelParams(
        sg_tile=(64, 32),
        load_tiles=(8, 16, 16, 16),
        prefetch_tiles=(32, 16, 16, 32),
        prefetch_depths=(2, 3),
    )
    assert params.annotate_options() == {
        "sg_tile": (64, 32),
        "load_tile_a": (8, 16),
        "load_tile_b": (16, 16),
        "prefetch_tile_a": (32, 16),
        "prefetch_tile_b": (16, 32),
        "prefetch_a_nb": 2,
        "prefetch_b_nb": 3,
    }
    assert KernelParams().annotate_options() == {"sg_tile": None}


def _axpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2.0 + y[tile]
    return out


XEGPU_KEYS = {
    "xegpu_sg_tiles": [[64, 32]],
    "xegpu_load_tiles": [[8, 16, 16, 16]],
    "xegpu_prefetch_tiles": [None],
    "xegpu_prefetch_depths": [[1, 1]],
}


def test_cpu_target_ignores_xegpu_keys() -> None:
    """A config shared across devices: the CPU target ignores ``xegpu_*`` keys
    (even ones invalid on the XPU)."""
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    config = helion.Config(block_sizes=[32, 32], **XEGPU_KEYS)
    run = compile_mlir(helion.kernel(backend="mlir")(_axpy), [x, y], config=config)
    torch.testing.assert_close(run(x, y), x * 2.0 + y)
    # Through Helion's own config handling (normalization keeps the keys).
    kernel = helion.kernel(backend="mlir", config=config)(_axpy)
    torch.testing.assert_close(kernel(x, y), x * 2.0 + y)
    bound = kernel.bind((x, y))
    normalized = helion.Config(**config.config)
    bound.config_spec.normalize(normalized)
    assert {key: normalized[key] for key in XEGPU_KEYS} == XEGPU_KEYS


def test_xpu_block_size_bounds_for_autotuning() -> None:
    from helion_mlir_backend.api import _compile

    hf, _config, env = _compile(matmul, _gemm_args(), None)
    spec = env.config_spec
    get_target("xpu").adjust_config_spec(spec, hf.device_ir)
    m, n, k = spec.block_sizes
    assert (m.min_size, n.min_size, k.min_size) == (32, 32, 16)
    # K tiles stay below K (the K loop must remain) and within the K tiles of
    # lighthouse's cost model.
    assert k.max_size == 64


@pytest.mark.parametrize(
    ("mnk", "expected"),
    [
        ((32768, 32768, 64), True),  # 7_Matmul_with_small_K_dimension_
        ((32768, 32768, 32), True),  # 9_Tall_skinny (also padded to 64)
        ((8192, 8192, 512), True),
        ((4096, 4096, 2048), True),  # the boundary: as many written as read
        ((4096, 4096, 2049), False),
        ((4096, 4096, 4096), False),  # 1_Square, 13
        ((2048, 4096, 8192), False),  # 2, 16-18
        ((256, 256, 524288), False),  # 6_large_K (and its split chunks)
        ((256, 256, 16384), False),
    ],
)
def test_columns_first_rule(mnk: tuple[int, int, int], expected: bool) -> None:
    """``columns_first``: a GEMM's grid runs columns first exactly when it
    writes at least as many elements as it reads, M * N >= K * (M + N)."""
    shape = GemmShape(mnk, (256, 256), 32, transpose_a=False, transpose_b=False)
    assert columns_first(shape) is expected


def test_columns_first_rule_needs_a_known_gemm() -> None:
    """Unknown sizes and other kernel classes keep their order."""
    unknown = GemmShape(None, (256, 256), 32, transpose_a=False, transpose_b=False)
    assert not columns_first(unknown)
    dynamic = ir.ShapedType.get_dynamic_size()
    for mnk in ((dynamic, dynamic, 64), (32768, 32768, dynamic), (0, 0, 0)):
        shape = GemmShape(mnk, (256, 256), 32, transpose_a=False, transpose_b=False)
        assert not columns_first(shape), mnk
    assert not columns_first(None)
    assert not columns_first(ReductionShape(64, 64))
    assert not columns_first(ElementwiseShape((1, 8192), (16,), (16,)))
