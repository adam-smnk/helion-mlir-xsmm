"""The XPU target at the IR level: no XPU, no SPIR-V target needed.

Kernels on fake XPU tensors are lowered to annotated XeGPU (``xegpu-wg``) and on
to LLVM/XeVM without device binaries (``xevm``). Kernels outside what the XeGPU
pipeline compiles fail with ``UnsupportedOperationError`` before lowering.
"""

from __future__ import annotations

import contextlib
import io
import re

import helion
import helion.language as hl
from lighthouse.pipeline.stage import TransformStage
from mlir import ir
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend import compile_mlir
from helion_mlir_backend import generate_mlir
from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler import xpu_transforms
from helion_mlir_backend._compiler.mlir.support import LoweringStoppedError
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir.support import UnsupportedTargetError
from helion_mlir_backend._compiler.targets import XpuTarget
from helion_mlir_backend._compiler.targets import get_target
from helion_mlir_backend._compiler.targets.xpu import XpuLowering
from helion_mlir_backend._compiler.targets.xpu import _translated
from helion_mlir_backend._compiler.targets.xpu import bindings_emit_spirv

N = 1024
GEMM = helion.Config(block_sizes=[256, 256, 32])


def _kernel(config: helion.Config) -> object:
    return helion.kernel(backend="mlir", static_shapes=True, config=config)


@_kernel(GEMM)
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


@_kernel(GEMM)
def matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = torch.relu(acc)
    return out


@_kernel(GEMM)
def matmul_scaled(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc * scale
    return out


@_kernel(helion.Config(block_sizes=[64, 128]))
def scale_shift(x: torch.Tensor, alpha: float, beta: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * alpha + beta
    return out


@_kernel(helion.Config(block_sizes=[64, 128]))
def scale_in_f64(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """f64 math on f32 tiles: the kernel needs ``alpha`` as an f64."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile].to(torch.float64) * alpha).to(torch.float32)
    return out


@_kernel(GEMM)
def matmul_transposed_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b.T`` with ``b`` of shape (N, K): Helion emits a ``linalg.contract``
    reading ``b`` transposed."""
    m, k = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
        out[tm, tn] = acc
    return out


@_kernel(helion.Config(block_sizes=[256, 256, 32, 256, 256, 32]))
def two_phase_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``(a @ b) @ c``: the second product reads all of the first (``hl.barrier``)."""
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


@_kernel(helion.Config(block_sizes=[64, 64]))
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


@_kernel(helion.Config(block_sizes=[32, 256]))
def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.relu(x[tile])
    return out


@_kernel(helion.Config(block_sizes=[64, 128]))
def gelu_tanh_cast(x: torch.Tensor) -> torch.Tensor:
    """bf16 math (tanh, exp) on bf16 tiles, stored as f32."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.shape):
        v = x[tile]
        out[tile] = (torch.tanh(v) * torch.exp(v)).to(torch.float32)
    return out


@_kernel(helion.Config(block_sizes=[64, 128]))
def two_in_two_out(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bf16 math over two inputs, into two outputs."""
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        s = torch.sigmoid(x[tile] + y[tile])
        a[tile] = s
        b[tile] = torch.exp(s) * x[tile]
    return a, b


@_kernel(helion.Config(block_sizes=[64, 128]))
def bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """A row of ``bias`` broadcast over every row of the tile."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.shape):
        out[tm, tn] = torch.relu(x[tm, tn] + bias[tn][None, :])
    return out


@_kernel(helion.Config(block_sizes=[2048]))
def sigmoid_1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sigmoid(x[tile])
    return out


@_kernel(helion.Config(block_sizes=[64, 64]))
def row_sums(x: torch.Tensor) -> torch.Tensor:
    """Row sums, stored as a vector."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@_kernel(helion.Config(block_sizes=[64, 64]))
def row_maxima(x: torch.Tensor) -> torch.Tensor:
    """Row maxima, stored as a ``keepdim`` column."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        out[tile_m, :] = torch.amax(top, dim=-1, keepdim=True)
    return out


@_kernel(helion.Config(block_sizes=[256, 256, 32, 32]))
def two_matmuls(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], c[tk, tn])
        out[tm, tn] = acc
    return out


def _gemm_args(dtype: torch.dtype = torch.float16, size: int = N) -> list[object]:
    return [xpu_tensor(size, size, dtype=dtype), xpu_tensor(size, size, dtype=dtype)]


@pytest.fixture(autouse=True)
def xpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


# --- IR checks ---------------------------------------------------------------


def _check_gemm_kernel(text: str, grid: tuple[int, int], threads: int) -> None:
    """One annotated GEMM kernel per launch of ``grid`` workgroups of ``threads``."""
    for op in ("xegpu.dpas", "xegpu.load_nd", "xegpu.store_nd", "xegpu.prefetch_nd"):
        assert op in text, op
    assert "sg_layout" in text
    assert f"known_block_size = array<i32: {threads}, 1, 1>" in text
    assert f"known_grid_size = array<i32: {grid[0]}, {grid[1]}, 1>" in text
    assert re.search(r"gpu\.launch_func\s+@\w+::@\w+ blocks in", text)


def _host_function(text: str, name: str) -> str:
    """The body of the host function ``name`` (up to its first return)."""
    start = re.search(rf"(func\.func|llvm\.func) (public )?@{name}\(", text)
    assert start, f"no host function @{name}"
    end = text.index("return", start.end())
    return text[start.start() : end]


def _check_xevm(text: str, entry: str, launches: int) -> None:
    """Device code fully lowered; the host launches ``launches`` kernels in order on
    one stream, then waits for them."""
    assert "xegpu." not in text
    assert "llvm.func spir_funccc" in text
    host = _host_function(text, entry)
    streams = re.findall(r"(%\w+) = llvm\.call @mgpuStreamCreate\(\)", host)
    assert len(streams) == 1, host
    stream = streams[0]
    launched = [m.start() for m in re.finditer(r"gpu\.launch_func <", host)]
    on_stream = list(re.finditer(rf"gpu\.launch_func <{stream} : !llvm\.ptr>", host))
    assert len(launched) == len(on_stream) == launches, host
    sync = host.index(f"llvm.call @mgpuStreamSynchronize({stream})")
    assert sync > launched[-1]


def _launch_args(text: str, entry: str) -> list[list[str]]:
    """The ``helion.name`` of each entry argument a launch passes, per launch."""
    host = _host_function(text, entry)
    names = dict(re.findall(r"(%arg\d+): memref<[^{]*\{helion\.name = \"(\w+)\"", host))
    return [
        [names[arg] for arg in re.findall(r"%arg\d+", args) if arg in names]
        for args in re.findall(r"gpu\.launch_func [^\n]* args\(([^\n]*)\)", host)
    ]


# --- Supported kernels -------------------------------------------------------


@pytest.mark.isolated
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_lowers_to_xegpu(dtype: torch.dtype) -> None:
    text = lower_xpu(matmul, _gemm_args(dtype))
    _check_gemm_kernel(text, grid=(4, 4), threads=512)
    assert text.count("gpu.launch_func") == 1


@pytest.mark.isolated
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_lowers_to_xevm(dtype: torch.dtype) -> None:
    _check_xevm(lower_xpu(matmul, _gemm_args(dtype), "xevm"), "matmul", launches=1)


@pytest.mark.isolated
@pytest.mark.parametrize("stop_at", ["xegpu-wg", "xevm"])
def test_matmul_relu_epilogue(stop_at: str) -> None:
    text = lower_xpu(matmul_relu, _gemm_args(), stop_at)
    if stop_at == "xevm":
        _check_xevm(text, "matmul_relu", launches=1)
        return
    _check_gemm_kernel(text, grid=(4, 4), threads=512)
    # The epilogue runs in the kernel, on the accumulator before the store.
    kernel = text[text.index("gpu.module") :]
    assert re.search(r"arith\.maximumf|arith\.cmpf", kernel)


@pytest.mark.isolated
@pytest.mark.parametrize("stop_at", ["xegpu-wg", "xevm"])
def test_matmul_transposed_b(stop_at: str) -> None:
    text = lower_xpu(matmul_transposed_b, _gemm_args(), stop_at)
    if stop_at == "xevm":
        _check_xevm(text, "matmul_transposed_b", launches=1)
        return
    _check_gemm_kernel(text, grid=(4, 4), threads=512)
    # B tiles load as (N, K) tiles, transposed in registers for DPAS.
    assert "vector.transpose" in text
    assert "order = [0, 1]" in text


@pytest.mark.isolated
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_runtime_scalar_is_passed_by_value(dtype: torch.dtype) -> None:
    text = lower_xpu(matmul_scaled, [*_gemm_args(dtype), 0.5])
    _check_gemm_kernel(text, grid=(4, 4), threads=512)
    host = _host_function(text, "matmul_scaled")
    scalar = re.search(
        r"(%arg\d+): f64 \{helion\.name = \"scale\", helion\.role = \"scalar\"", host
    )
    assert scalar, host
    # The conversion to the accumulator's type runs on the host: the kernel takes
    # the f32 value.
    cast = re.search(rf"(%\w+) = arith\.truncf {scalar.group(1)} : f64 to f32", host)
    assert cast, host
    assert host.index(cast.group(0)) < host.index("gpu.launch_func")
    assert re.search(rf"args\([^\n]*{cast.group(1)} : f32", host)
    kernel = text[text.index("gpu.module") :]
    assert re.search(r"gpu\.func @matmul_scaled_kernel\([^)]*: f32", kernel)
    assert "f64" not in kernel


@pytest.mark.isolated
def test_runtime_scalar_kernel_lowers_to_xevm() -> None:
    text = lower_xpu(matmul_scaled, [*_gemm_args(torch.bfloat16), 0.5], "xevm")
    _check_xevm(text, "matmul_scaled", launches=1)
    kernel = re.search(
        r"llvm\.func spir_kernelcc @matmul_scaled_kernel\(([^)]*)\)", text
    )
    assert kernel, text
    assert ": f32" in kernel.group(1)
    assert "f64" not in text[text.index("gpu.module") :]


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("dtype", "element"), [(torch.float32, "f32"), (torch.bfloat16, "bf16")]
)
def test_float_and_int_scalars_reach_the_kernel_in_its_type(
    dtype: torch.dtype, element: str
) -> None:
    text = lower_xpu(scale_shift, [xpu_tensor(1024, 1024, dtype=dtype), 0.5, 3], "xevm")
    _check_xevm(text, "scale_shift", launches=1)
    host = _host_function(text, "scale_shift")
    assert re.search(r"%arg\d+: f64 \{helion\.name = \"alpha\"", host)
    assert re.search(r"%arg\d+: i64 \{helion\.name = \"beta\"", host)
    kernel = re.search(r"llvm\.func spir_kernelcc @scale_shift_kernel\(([^)]*)\)", text)
    assert kernel, text
    # Both scalars converted on the host (alpha truncated, beta converted to float).
    assert kernel.group(1).count(f": {element},") == 2, kernel.group(1)
    assert "f64" not in text[text.index("gpu.module") :]


@pytest.mark.isolated
def test_f64_kernel_argument_is_unsupported() -> None:
    with pytest.raises(UnsupportedOperationError, match="kernel argument") as info:
        lower_xpu(scale_in_f64, [xpu_tensor(1024, 1024), 0.5])
    assert "'alpha' (f64) is used by" in str(info.value)
    assert "no floats wider than 32 bits" in str(info.value)


@pytest.mark.isolated
def test_barrier_phases_launch_in_order_on_one_stream() -> None:
    args = [xpu_tensor(N, N, dtype=torch.float16) for _ in range(3)]
    text = lower_xpu(two_phase_matmul, args)
    _check_gemm_kernel(text, grid=(4, 4), threads=512)
    assert text.count("gpu.launch_func") == 2
    first, second = _launch_args(text, "two_phase_matmul")
    assert {"a", "b", "tmp"} <= set(first) and "c" not in first
    assert {"tmp", "c", "out"} <= set(second) and "a" not in second
    _check_xevm(lower_xpu(two_phase_matmul, args, "xevm"), "two_phase_matmul", 2)


@pytest.mark.isolated
@pytest.mark.parametrize("stop_at", ["xegpu-wg", "xevm"])
def test_softmax_row_reduction(stop_at: str) -> None:
    text = lower_xpu(softmax, [xpu_tensor(4096, 4096)], stop_at)
    if stop_at == "xevm":
        _check_xevm(text, "softmax", launches=1)
        return
    assert "known_grid_size = array<i32: 64, 1, 1>" in text
    assert "vector.multi_reduction" in text
    assert "xegpu.load_nd" in text and "xegpu.store_nd" in text
    assert "sg_layout" in text


def _layouts(text: str, op: str) -> list[str]:
    """The anchor layouts of the ``op`` ops in ``text``."""
    return re.findall(rf"{re.escape(op)} [^\n]*<\{{layout = (#xegpu\.[^}}]*)\}}>", text)


@pytest.mark.isolated
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_elementwise_kernel_lowers_to_xegpu(dtype: torch.dtype) -> None:
    text = lower_xpu(relu, [xpu_tensor(4096, 4096, dtype=dtype)])
    # 32x256 workgroup tiles: 8 subgroups of 32x32 (inferred), the widest block
    # loads (32 bf16 or 16 f32 columns), 8x16 block stores.
    load_cols = 32 if dtype == torch.bfloat16 else 16
    tile = "sg_layout = [1, 8], sg_data = [32, 32]"
    assert _layouts(text, "xegpu.load_nd") == [
        f"#xegpu.layout<{tile}, inst_data = [32, {load_cols}]>"
    ]
    assert _layouts(text, "xegpu.store_nd") == [
        f"#xegpu.layout<{tile}, inst_data = [8, 16]>"
    ]
    assert "known_block_size = array<i32: 128, 1, 1>" in text
    assert "known_grid_size = array<i32: 128, 16, 1>" in text
    _check_xevm(
        lower_xpu(relu, [xpu_tensor(4096, 4096, dtype=dtype)], "xevm"), "relu", 1
    )


def _bf16_natives(text: str) -> list[str]:
    """Device functions declared with bf16 operands (no such OpenCL builtins)."""
    return re.findall(r"llvm\.func @(_Z\w+)\([^)]*bf16", text)


@pytest.mark.isolated
def test_elementwise_bf16_math_and_casts() -> None:
    args = [xpu_tensor(1024, 1024, dtype=torch.bfloat16)]
    text = lower_xpu(gelu_tanh_cast, args)
    # bf16 loads, f32 stores: each with its own block tile.
    assert re.search(r"xegpu\.load_nd [^\n]*inst_data = \[32, 32\][^\n]*xbf16>", text)
    assert re.search(r"xegpu\.store_nd [^\n]*inst_data = \[8, 16\][^\n]*xf32>", text)
    lowered = lower_xpu(gelu_tanh_cast, args, "xevm")
    _check_xevm(lowered, "gelu_tanh_cast", 1)
    # bf16 math runs in f32 (the fast exp too: upstream's math-to-xevm would
    # call a bf16 native exp otherwise).
    assert _bf16_natives(lowered) == []
    assert "@_Z16__spirv_ocl_tanhf" in lowered
    assert "@_Z22__spirv_ocl_native_expf" in lowered


@pytest.mark.isolated
def test_elementwise_several_inputs_and_outputs() -> None:
    args = [xpu_tensor(1024, 1024, dtype=torch.bfloat16) for _ in range(2)]
    text = lower_xpu(two_in_two_out, args)
    assert len(_layouts(text, "xegpu.load_nd")) == 2
    assert len(_layouts(text, "xegpu.store_nd")) == 2
    assert "known_grid_size = array<i32: 16, 8, 1>" in text
    lowered = lower_xpu(two_in_two_out, args, "xevm")
    _check_xevm(lowered, "two_in_two_out", 1)
    assert _bf16_natives(lowered) == []


@pytest.mark.isolated
def test_elementwise_broadcast_row() -> None:
    args = [
        xpu_tensor(1024, 1024, dtype=torch.bfloat16),
        xpu_tensor(1024, dtype=torch.bfloat16),
    ]
    text = lower_xpu(bias_relu, args)
    # The 64x128 tiles are anchored; the bias row's layout is propagated.
    assert len(_layouts(text, "xegpu.load_nd")) == 1
    assert len(_layouts(text, "xegpu.store_nd")) == 1
    _check_xevm(lower_xpu(bias_relu, args, "xevm"), "bias_relu", 1)


@pytest.mark.isolated
def test_elementwise_1d_tiles_gather_and_scatter() -> None:
    args = [xpu_tensor(1 << 20)]
    text = lower_xpu(sigmoid_1d, args)
    layout = "#xegpu.layout<sg_layout = [2], sg_data = [1024], inst_data = [16]>"
    assert _layouts(text, "xegpu.load") == [layout]
    assert _layouts(text, "xegpu.store") == [layout]
    assert "known_block_size = array<i32: 32, 1, 1>" in text
    _check_xevm(lower_xpu(sigmoid_1d, args, "xevm"), "sigmoid_1d", 1)


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("name", "layout"),
    [
        (
            "row_sums",
            (
                "#xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [8, 64]>, "
                "dims = [1]>"
            ),
        ),
        ("row_maxima", "#xegpu.layout<sg_layout = [8, 1], sg_data = [8, 1]>"),
    ],
    ids=["vector", "keepdim-column"],
)
def test_row_reduction_results(name: str, layout: str) -> None:
    """Row results are stored with scatter stores, anchored with the reduction's
    layout; a ``keepdim`` column needs no temporary buffer."""
    kernel = {"row_sums": row_sums, "row_maxima": row_maxima}[name]
    args = [xpu_tensor(4096, 8192)]
    text = lower_xpu(kernel, args)
    assert _layouts(text, "xegpu.store") == [layout]
    assert "xegpu.store_nd" not in text and "gpu.alloc" not in text
    assert "known_block_size = array<i32: 128, 1, 1>" in text
    _check_xevm(lower_xpu(kernel, args, "xevm"), name, 1)


@pytest.mark.isolated
def test_cpu_tensors_lower_for_a_named_xpu_target() -> None:
    """Codegen is target-neutral: the IR of CPU (or meta) tensors lowers for the
    XPU when the target is named; running it needs XPU tensors."""
    cpu = [torch.randn(N, N, dtype=torch.float16) for _ in range(2)]
    _check_gemm_kernel(lower_xpu(matmul, cpu), grid=(4, 4), threads=512)
    meta = [torch.empty(N, N, dtype=torch.float16, device="meta") for _ in range(2)]
    _check_gemm_kernel(lower_xpu(matmul, meta), grid=(4, 4), threads=512)


@pytest.mark.isolated
def test_stop_point_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_XPU_STOP_AT", "xegpu-wg")
    text = str(lower_mlir(matmul, _gemm_args()))
    assert "xegpu.dpas" in text
    # An explicit stop point wins.
    assert "xegpu." not in str(lower_mlir(matmul, _gemm_args(), stop_at="xevm"))
    monkeypatch.setenv("HELION_MLIR_XPU_STOP_AT", "llvm")
    with pytest.raises(ValueError, match="HELION_MLIR_XPU_STOP_AT='llvm'"):
        lower_mlir(matmul, _gemm_args())


@pytest.mark.isolated
def test_stage_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_DUMP_STAGES", "1")
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")  # lower in-process
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        lower_xpu(matmul, _gemm_args())
    stages = re.findall(r"=== MLIR after XPU stage '([\w-]+)' ===", output.getvalue())
    assert stages == [
        "prepare",
        "cleanup",
        "vectorized",
        "bufferized",
        "host-legalized",
        "gpu-outlining",
        "xegpu-initial",
        "xegpu-wg",
    ]


@pytest.mark.isolated
def test_cpu_target_lowers_to_llvm() -> None:
    cpu = [torch.randn(64, 64), torch.randn(64, 64)]
    text = str(lower_mlir(matmul, cpu, config=helion.Config(block_sizes=[32, 32, 32])))
    assert "llvm.func @matmul" in text
    with pytest.raises(ValueError, match="stop point 'xevm'"):
        lower_mlir(matmul, cpu, stop_at="xevm")


# --- Unsupported kernels -----------------------------------------------------


@pytest.mark.isolated
def test_elementwise_tile_columns_must_fit_block_loads() -> None:
    config = helion.Config(block_sizes=[64, 8])
    with pytest.raises(
        UnsupportedOperationError,
        match=r"elementwise kernel \(WG tile 64x8\) on XPU(.|\n)*not a multiple of 16",
    ):
        lower_mlir(
            relu,
            [xpu_tensor(4096, 4096, dtype=torch.bfloat16)],
            config=config,
            target="xpu",
            stop_at="xegpu-wg",
        )


@pytest.mark.isolated
def test_partial_tiles_are_unsupported() -> None:
    with pytest.raises(UnsupportedOperationError, match="partial tiles on XPU"):
        lower_xpu(matmul, _gemm_args(size=1000))


@pytest.mark.isolated
def test_matmul_without_k_loop_is_unsupported() -> None:
    config = helion.Config(block_sizes=[256, 256, N])
    with pytest.raises(UnsupportedOperationError, match="without a K loop"):
        lower_mlir(matmul, _gemm_args(), config=config, target="xpu", stop_at="xevm")


@pytest.mark.isolated
def test_f32_matmul_is_unsupported() -> None:
    with pytest.raises(UnsupportedOperationError, match="f32 x f32 into f32"):
        lower_xpu(matmul, _gemm_args(torch.float32))


@pytest.mark.isolated
def test_two_matmuls_in_one_grid_loop_are_unsupported() -> None:
    args = [xpu_tensor(N, N, dtype=torch.float16) for _ in range(3)]
    with pytest.raises(UnsupportedOperationError, match="several matmuls"):
        lower_xpu(two_matmuls, args)


@pytest.mark.isolated
def test_dynamic_shapes_are_unsupported() -> None:
    dynamic = helion.kernel(backend="mlir", static_shapes=False, config=GEMM)(matmul.fn)
    # Real tensors: Helion keeps the sizes of fake ones static.
    args = [torch.randn(N, 512).half(), torch.randn(512, 2 * N).half()]
    with pytest.raises(UnsupportedOperationError, match="dynamic shapes on XPU"):
        lower_xpu(dynamic, args)


@pytest.mark.isolated
def test_cpu_pipelines_are_rejected_on_xpu() -> None:
    with pytest.raises(ValueError, match="unknown pipeline 'opt' for the MLIR target"):
        lower_mlir(matmul, _gemm_args(), pipeline="opt", stop_at="xevm")
    config = helion.Config(block_sizes=[256, 256, 32], mlir_pipeline="scalar")
    with pytest.raises(ValueError, match="unknown pipeline 'scalar'"):
        lower_mlir(matmul, _gemm_args(), config=config, stop_at="xevm")


# --- Runtime without an XPU ---------------------------------------------------


@pytest.mark.isolated
def test_running_needs_an_xpu_capable_llvm(monkeypatch: pytest.MonkeyPatch) -> None:
    if bindings_emit_spirv():
        pytest.skip("the MLIR bindings have the SPIR-V target")
    cpu = [torch.randn(N, N, dtype=torch.float16) for _ in range(2)]
    with pytest.raises(UnsupportedTargetError, match="no SPIR-V target"):
        compile_mlir(matmul, cpu, target="xpu")
    with pytest.raises(UnsupportedTargetError, match="no SPIR-V target"):
        lower_mlir(matmul, cpu, target="xpu")
    # A default stop point lowers, then stops before the JIT.
    monkeypatch.setenv("HELION_MLIR_XPU_STOP_AT", "xevm")
    with pytest.raises(LoweringStoppedError, match="stops at 'xevm'"):
        compile_mlir(matmul, cpu, target="xpu")


def test_xpu_target_without_an_xpu() -> None:
    target = get_target("xpu")
    assert isinstance(target, XpuTarget)
    assert target.pipelines == ("xegpu",)
    assert target.default_pipeline() == "xegpu"
    assert target.stop_points == ("xegpu-wg", "xevm", "binary")
    target.validate_environment("xegpu-wg")
    target.validate_environment("xevm")
    with pytest.raises(NotImplementedError, match="compiled for XPU tensors"):
        target.check_tensor("x", torch.empty(2))
    # No XPU tensors: nothing to synchronize.
    target.prepare_call([torch.tensor(1.0)])
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        assert target.hardware_name() == "xpu"


def test_lighthouse_kernel_class_failures_are_unsupported_operations() -> None:
    failure = RuntimeError(
        "Failed to apply named transform sequence.\nDiagnostic message at "
        "loc(unknown): outline_gpu_func: Could not apply any of the defined patterns."
    )
    error = _translated(failure, "gpu-outlining")
    assert isinstance(error, UnsupportedOperationError)
    assert "'outline_gpu_func' stage matched no GEMM" in str(error)
    other = _translated(RuntimeError("boom"), "xevm")
    assert other.__notes__ == ["Helion MLIR backend: failed in the XPU stage 'xevm'"]


def test_generate_mlir_for_fake_xpu_tensors() -> None:
    text = str(generate_mlir(matmul, _gemm_args()))
    assert "linalg.matmul" in text and "scf.forall" in text


# --- Host-code legalization ----------------------------------------------------


HOST_CODE = """
func.func @k(%out: memref<64x64xf32> {helion.name = "out", helion.role = "inout"},
             %x: memref<64x64xf32> {helion.name = "x", helion.role = "in"},
             %s: f64 {helion.name = "s", helion.role = "scalar"}) {
  %zero = arith.constant 0.0 : f32
  %tmp = gpu.alloc () : memref<64x64xf32>
  memref.copy %x, %tmp : memref<64x64xf32> to memref<64x64xf32>
  linalg.fill ins(%zero : f32) outs(%out : memref<64x64xf32>)
  scf.forall (%i) in (64) {
    %v = memref.load %tmp[%i, %i] : memref<64x64xf32>
    %t = arith.truncf %s : f64 to f32
    %r = arith.mulf %v, %t : f32
    memref.store %r, %out[%i, %i] : memref<64x64xf32>
  }
  gpu.dealloc %tmp : memref<64x64xf32>
  return
}
"""


def _legalize(source: str) -> str:
    with ir.Context() as context, ir.Location.unknown():
        module = ir.Module.parse(source)
        stage = TransformStage(xpu_transforms.legalize_host_ops(), context)
        XpuLowering([("host-legalized", stage)], dump=False).apply(module)
        module.operation.verify()
        return str(module)


def test_host_copies_and_fills_become_gpu_runtime_calls() -> None:
    text = _legalize(HOST_CODE)
    assert "memref.copy" not in text
    assert "linalg.fill" not in text
    assert re.search(r"gpu\.memcpy\s+%\w+, %arg1 :", text)
    assert re.search(r"gpu\.memset\s+%arg0, %\w+ : memref<64x64xf32>, f32", text)
    # The kernel stays; the runtime scalar's conversion moves out of it, so the
    # kernel takes an f32.
    assert "memref.store" in text
    assert text.index("arith.truncf %arg2 : f64 to f32") < text.index("scf.forall")


SCALAR_CODE = """
func.func @k(%out: memref<64xf32>, %vout: memref<64x4xf32>, %s: f64, %n: i64) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c7 = arith.constant 7 : i64
  %two = arith.constant 2.0 : f32
  scf.forall (%i) in (64) {
    %acc = scf.for %k = %c0 to %c64 step %c4 iter_args(%a = %two) -> (f32) {
      %t = arith.truncf %s : f64 to f32
      %f = arith.sitofp %n : i64 to f32
      %u = arith.mulf %t, %f : f32
      %ki = arith.index_cast %k : index to i64
      %kf = arith.sitofp %ki : i64 to f32
      %w = arith.mulf %u, %kf : f32
      %d = arith.divsi %c7, %n : i64
      %df = arith.sitofp %d : i64 to f32
      %b = arith.addf %a, %w : f32
      %c = arith.addf %b, %df : f32
      scf.yield %c : f32
    }
    memref.store %acc, %out[%i] : memref<64xf32>
    %t0 = arith.truncf %s : f64 to f32
    %v = vector.broadcast %t0 : f32 to vector<4xf32>
    vector.store %v, %vout[%i, %c0] : memref<64x4xf32>, vector<4xf32>
  }
  return
}
"""


def test_invariant_scalar_code_moves_out_of_the_kernels() -> None:
    text = _legalize(SCALAR_CODE)
    kernel = text[text.index("scf.forall") :]
    host = text[: text.index("scf.forall")]
    # Scalar code of the runtime scalars, also from the nested loop.
    for op in ("arith.truncf %arg2", "arith.sitofp %arg3", "arith.mulf %0, %1"):
        assert op in host, op
    # Code of the loops' induction variables, code that may trap (and what uses
    # it) and vector code stay.
    for op in ("arith.index_cast", "arith.divsi", "vector.broadcast", "arith.addf"):
        assert op in kernel, op


STRIDED_COPY = """
  %sub = memref.subview %x[0, 0] [64, 32] [1, 1]
      : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1]>>
  %sub2 = memref.subview %tmp[0, 0] [64, 32] [1, 1]
      : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1]>>
  memref.copy %sub, %sub2
      : memref<64x32xf32, strided<[64, 1]>> to memref<64x32xf32, strided<[64, 1]>>
"""


@pytest.mark.parametrize(
    ("edits", "match"),
    [
        (
            [
                (
                    "  gpu.dealloc",
                    "  memref.store %zero, %out[%c0, %c0] : memref<64x64xf32>\n  gpu.dealloc",
                )
            ],
            "'memref.store' outside the kernels accesses a device buffer",
        ),
        (
            [("gpu.alloc ()", "memref.alloc ()"), ("gpu.dealloc", "memref.dealloc")],
            "'memref.alloc' creates a host buffer",
        ),
        (
            [
                (
                    "    %t = arith.truncf %s : f64 to f32",
                    (
                        "    %ii = arith.index_cast %i : index to i64\n"
                        "    %if = arith.sitofp %ii : i64 to f64\n"
                        "    %p = arith.mulf %if, %s : f64\n"
                        "    %t = arith.truncf %p : f64 to f32"
                    ),
                )
            ],
            "'s' (f64) is used by 'arith.mulf' in a kernel",
        ),
        (
            [("  memref.copy %x, %tmp", STRIDED_COPY + "  memref.copy %x, %tmp")],
            "'memref.copy' outside the kernels accesses a device buffer",
        ),
    ],
    ids=["host-store", "host-alloc", "f64-kernel-argument", "strided-copy"],
)
def test_host_access_to_device_memory_is_unsupported(
    edits: list[tuple[str, str]], match: str
) -> None:
    source = HOST_CODE.replace(
        "  %zero = arith.constant 0.0 : f32",
        "  %zero = arith.constant 0.0 : f32\n  %c0 = arith.constant 0 : index",
    )
    for old, new in edits:
        assert old in source
        source = source.replace(old, new)
    with pytest.raises(UnsupportedOperationError, match=re.escape(match)):
        _legalize(source)
