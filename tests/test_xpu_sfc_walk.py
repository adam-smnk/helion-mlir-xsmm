"""``xegpu_sfc_walk`` and ``xegpu_store_write_back``, per kernel. The curve is a bijection
of the grid and a kernel's block ids go through it; write-back hints reach the kernel's
stores only. Kernels without the keys keep their IR. Results on the device are at the
end, on NaN-poisoned outputs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import MLIRBackendError
from helion_mlir_backend._compiler.mlir_transforms.xegpu_sfc_walk import curve_tables

if TYPE_CHECKING:
    from collections.abc import Callable


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


def _kernel(
    fn: Callable[..., torch.Tensor], blocks: list[int], **config: object
) -> object:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, **config),
    )


def _gemm_ir(stage: str = "xevm", **config: object) -> str:
    args = [
        xpu_tensor(512, 256, dtype=torch.bfloat16),
        xpu_tensor(256, 1024, dtype=torch.bfloat16),
    ]
    kernel = _kernel(gemm, [128, 256, 32], **config)
    return str(lower_mlir(kernel, args, target="xpu", stop_at=stage))


@pytest.mark.parametrize(
    ("grid_x", "grid_y"),
    [(2, 2), (16, 16), (8, 16), (16, 8), (32, 8), (3, 7), (7, 3), (2, 64), (5, 5)],
)
def test_the_curve_is_a_bijection_of_the_grid(grid_x: int, grid_y: int) -> None:
    xs, ys = curve_tables(grid_x, grid_y)
    assert sorted(zip(xs, ys, strict=True)) == [
        (x, y) for x in range(grid_x) for y in range(grid_y)
    ]


def test_a_power_of_two_grid_moves_to_a_neighbour_at_each_step() -> None:
    xs, ys = curve_tables(16, 16)
    steps = [
        abs(xs[i + 1] - xs[i]) + abs(ys[i + 1] - ys[i]) for i in range(len(xs) - 1)
    ]
    assert set(steps) == {1}


def test_without_the_key_the_kernel_keeps_its_ir() -> None:
    assert _gemm_ir() == _gemm_ir(xegpu_sfc_walk=[None])
    assert "sfc" not in _gemm_ir()


def test_the_tile_offsets_come_from_the_tables() -> None:
    """Grid 4 x 4: 512 rows in 128, 1024 columns in 256. The block ids reach the offsets
    only through the two table loads."""
    text = _gemm_ir(xegpu_sfc_walk=[True])
    assert len(re.findall(r"llvm\.mlir\.global private constant @\w+_sfc_[xy]", text)) == 2
    assert len(re.findall(r"llvm\.load %\d+ : !llvm\.ptr<1> -> i32", text)) == 2
    ids = re.findall(r"(%\d+) = llvm\.call spir_funccc @_Z12get_group_idj", text)
    assert ids
    for value in ids:
        uses = re.findall(rf"^.*{value}\b(?!\s*=).*$", text, flags=re.MULTILINE)
        assert uses, value
        assert all(re.search(r"llvm\.(add|mul) ", use) for use in uses), uses
    linear = re.findall(r"llvm\.mul (%\d+), (%\d+) overflow<nsw>", text)
    assert not [pair for pair in linear if pair[0] in ids], linear


def test_without_the_key_the_offsets_are_the_block_ids() -> None:
    """The check above, on the form it must tell apart."""
    text = _gemm_ir()
    ids = re.findall(r"(%\d+) = llvm\.call spir_funccc @_Z12get_group_idj", text)
    linear = re.findall(r"llvm\.mul (%\d+), (%\d+) overflow<nsw>", text)
    assert [pair for pair in linear if pair[0] in ids]
    assert not re.findall(r"llvm\.load %\d+ : !llvm\.ptr<1> -> i32", text)


def test_the_walk_is_not_a_layout_hint() -> None:
    plain = re.findall(r"#xegpu\.layout<[^>]*>", _gemm_ir("xegpu-wg"))
    walked = re.findall(r"#xegpu\.layout<[^>]*>", _gemm_ir("xegpu-wg", xegpu_sfc_walk=[True]))
    assert plain == walked


@pytest.mark.parametrize("key", ["xegpu_sfc_walk", "xegpu_store_write_back"])
@pytest.mark.parametrize("value", [1, "gilbert", False])
def test_an_invalid_entry_is_refused(key: str, value: object) -> None:
    with pytest.raises(MLIRBackendError, match=key):
        _gemm_ir(**{key: [value]})


def test_a_grid_of_one_line_has_nothing_to_walk() -> None:
    """One row of workgroups: the kernel keeps its IR."""
    args = [
        xpu_tensor(128, 256, dtype=torch.bfloat16),
        xpu_tensor(256, 1024, dtype=torch.bfloat16),
    ]
    plain = lower_mlir(_kernel(gemm, [128, 256, 32]), args, target="xpu", stop_at="xevm")
    walked = lower_mlir(
        _kernel(gemm, [128, 256, 32], xegpu_sfc_walk=[True]),
        args,
        target="xpu",
        stop_at="xevm",
    )
    assert "sfc" not in str(walked)
    assert re.sub(r"\s+", " ", str(plain).split("gpu.module")[1]) == re.sub(
        r"\s+", " ", str(walked).split("gpu.module")[1]
    )


def batched(a: torch.Tensor) -> torch.Tensor:
    batch, m, n = a.shape
    out = torch.empty_like(a)
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        out[tile_b, tile_m, tile_n] = torch.relu(a[tile_b, tile_m, tile_n])
    return out


def test_a_3d_grid_is_refused() -> None:
    args = [xpu_tensor(2, 256, 256, dtype=torch.float32)]
    kernel = _kernel(batched, [1, 64, 64], xegpu_sfc_walk=[True])
    with pytest.raises(MLIRBackendError, match="xegpu_sfc_walk.*3D grid"):
        lower_mlir(kernel, args, target="xpu", stop_at="xevm")


def test_more_workgroups_than_a_table_holds_are_refused() -> None:
    """Regression (review): the SPIR-V backend aborts the process past the limit, without
    naming the key. 512 x 256 workgroups: refused before it gets there."""
    args = [xpu_tensor(16384, 65536, dtype=torch.float32)]
    kernel = _kernel(relu_tiles, [32, 256], xegpu_sfc_walk=[True])
    with pytest.raises(MLIRBackendError, match="xegpu_sfc_walk.*at most 65532"):
        lower_mlir(kernel, args, target="xpu", stop_at="xevm")


def _stores(text: str) -> list[str]:
    return re.findall(r"xegpu\.store_nd [^\n]*|xegpu\.store [^\n]*", text)


def test_write_back_reaches_the_stores_only() -> None:
    """The hints as XeVM's lowering turns them into a cache control, on the store
    built-in; the loads keep theirs."""
    text = _gemm_ir(xegpu_store_write_back=[True])
    assert "cachecontrol" in text
    plain = _gemm_ir()
    assert ".str.cachecontrol" not in plain or plain.count("cachecontrol") < text.count(
        "cachecontrol"
    )
    assert text.count("2d_block_write") == plain.count("2d_block_write")
    assert text.count("2d_block_read") == plain.count("2d_block_read")


def two_kernels(x: torch.Tensor, tmp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        tmp[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
    hl.barrier()
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = tmp[tile_m, tile_n] * 2.0
    return out


@pytest.mark.parametrize(
    ("key", "marker"),
    [("xegpu_sfc_walk", "sfc_"), ("xegpu_store_write_back", "cachecontrol")],
)
def test_the_keys_bind_to_their_kernel(key: str, marker: str) -> None:
    args = [xpu_tensor(256, 512, dtype=torch.float32) for _ in range(3)]
    kernel = _kernel(two_kernels, [64, 128, 64, 128], **{key: [None, True]})
    text = str(lower_mlir(kernel, args, target="xpu", stop_at="xevm"))
    modules = re.split(r"\n\s*gpu\.module ", text)[1:]
    assert len(modules) == 2
    assert marker not in modules[0]
    assert marker in modules[1]


def relu_tiles(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])
    return out


def gemm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


def relu_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])
    return out


def _poisoned(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """A tile the walk misses stays NaN: a fresh output may reuse the buffer of the
    kernel run before, which already holds the right values."""
    return torch.full(shape, float("nan"), dtype=dtype, device="xpu")


@pytest.mark.xpu_runtime
@pytest.mark.parametrize(
    ("m", "k", "n", "blocks"),
    [
        (512, 256, 1024, [128, 256, 32]),  # 4 x 4
        (768, 256, 256, [256, 64, 32]),  # 3 x 4
        (1024, 256, 256, [128, 128, 32]),  # 8 x 2
        (256, 256, 1024, [64, 256, 32]),  # 4 x 4
    ],
)
def test_a_gemm_on_the_device_is_the_same(
    m: int, k: int, n: int, blocks: list[int]
) -> None:
    generator = torch.Generator().manual_seed(m + n)
    a = torch.randn(m, k, generator=generator, dtype=torch.float64).to(torch.bfloat16)
    b = torch.randn(k, n, generator=generator, dtype=torch.float64).to(torch.bfloat16)
    inputs = (a.to("xpu"), b.to("xpu"))
    plain = _kernel(gemm_into, blocks)(*inputs, _poisoned(m, n, dtype=a.dtype)).cpu()
    walked = _kernel(gemm_into, blocks, xegpu_sfc_walk=[True])(
        *inputs, _poisoned(m, n, dtype=a.dtype)
    ).cpu()
    assert torch.equal(plain.float(), walked.float())
    reference = (a.float() @ b.float()).to(a.dtype).float()
    assert ((plain.float() - reference).abs().max() / reference.abs().max()) < 1e-2


@pytest.mark.xpu_runtime
@pytest.mark.parametrize(
    ("m", "n", "blocks"),
    [(256, 512, [64, 128]), (192, 512, [64, 128]), (512, 128, [128, 64])],
)
def test_an_elementwise_grid_on_the_device_is_the_same(
    m: int, n: int, blocks: list[int]
) -> None:
    x = torch.randn(m, n, generator=torch.Generator().manual_seed(m), dtype=torch.float32)
    out = _kernel(relu_into, blocks, xegpu_sfc_walk=[True])(
        x.to("xpu"), _poisoned(m, n, dtype=x.dtype)
    ).cpu()
    assert torch.equal(out, torch.relu(x))


_BOTH = {"xegpu_sfc_walk": [True], "xegpu_store_write_back": [True]}


@pytest.mark.xpu_runtime
@pytest.mark.parametrize(
    ("m", "k", "n", "blocks", "offset"),
    [
        (512, 256, 1024, [128, 256, 32], 0),
        (512, 256, 1040, [128, 208, 32], 0),  # tiles end in the middle of a cache line
        (512, 256, 1024, [128, 256, 32], 8),  # a base off the cache line
    ],
)
def test_write_back_gemm_on_the_device_is_the_same(
    m: int, k: int, n: int, blocks: list[int], offset: int
) -> None:
    generator = torch.Generator().manual_seed(m + n)
    a = torch.randn(m, k, generator=generator, dtype=torch.float64).to(torch.bfloat16)
    b = torch.randn(k, n, generator=generator, dtype=torch.float64).to(torch.bfloat16)
    inputs = (a.to("xpu"), b.to("xpu"))
    plain = _kernel(gemm_into, blocks)(*inputs, _poisoned(m, n, dtype=a.dtype)).cpu()
    buffer = _poisoned(m * n + offset, 1, dtype=a.dtype).flatten()
    out = _kernel(gemm_into, blocks, **_BOTH)(
        *inputs, buffer[offset:].view(m, n)
    ).cpu()
    assert torch.equal(plain.float(), out.float())


@pytest.mark.xpu_runtime
@pytest.mark.parametrize("keys", [[True, None], [True, True]])
def test_the_next_kernel_on_the_stream_reads_a_write_back_output(
    keys: list[bool | None],
) -> None:
    x = torch.randn(512, 1024, generator=torch.Generator().manual_seed(1))
    kernel = _kernel(two_kernels, [64, 128, 64, 128], xegpu_store_write_back=keys)
    for _ in range(3):
        out = kernel(
            x.to("xpu"),
            _poisoned(512, 1024, dtype=x.dtype),
            _poisoned(512, 1024, dtype=x.dtype),
        ).cpu()
        assert torch.equal(out, (x + 1.0) * 2.0)
