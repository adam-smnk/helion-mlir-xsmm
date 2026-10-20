"""Ragged (boundary) tiles: a tile that extends past its loop or tensor.

Loads zero-pad the part outside, ``_mask_to`` replaces it by the reduction
identity, and stores write only the real part. When the block size divides the
extent the IR has no dynamic sizes.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import opt_pipeline
from tests.harness import run_direct

from helion_mlir_backend import generate_mlir


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


@_kernel(8, 16)
def add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + y[tm, tn]
    return out


@_kernel(4, 16)
def row_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@_kernel(8)
def row_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        shifted = torch.exp(row - row.amax(dim=-1, keepdim=True))
        out[tm, :] = shifted / shifted.sum(dim=-1, keepdim=True)
    return out


@_kernel(8)
def chunk_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    """Softmax over each tile of rows: padded rows must not count (``_mask_to``)."""
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        rows = x[tm, :]
        shifted = torch.exp(rows - rows.amax(dim=0, keepdim=True))
        out[tm, :] = shifted / shifted.sum(dim=0, keepdim=True)
    return out


@_kernel(16)
def offset_scale_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(3, x.size(0)):
        out[tile] = x[tile] * 2.0
    return out


@_kernel(32)
def read_past_end_kernel(a: torch.Tensor, n_pad: hl.constexpr) -> torch.Tensor:
    out = torch.empty((int(n_pad),), dtype=a.dtype, device=a.device)
    for tile in hl.tile(int(n_pad)):
        out[tile] = a[tile]
    return out


@_kernel(16)
def masked_load_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.load(x, [tile], extra_mask=tile.index % 2 == 0)
    return out


@_kernel(16)
def masked_store_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, -1.0)
    for tile in hl.tile(x.size(0)):
        hl.store(out, [tile], x[tile], extra_mask=tile.index % 2 == 0)
    return out


@_kernel(32, 32)
def scale_2d_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 3.0
    return out


@_kernel(16)
def store_then_load_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1.0
        out[tile] = x[tile] * 2.0
    return out


@_kernel(32, 32, 32)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


def _every_other(x: torch.Tensor, fill: float) -> torch.Tensor:
    out = torch.full_like(x, fill)
    out[::2] = x[::2]
    return out


@_kernel()
def pack_blocks_kernel(b: torch.Tensor) -> torch.Tensor:
    """``[K, N] -> [N/8, K/8, 8, 8]``, zero-padded: blocks past B's end read zeros."""
    k, n = b.shape
    kb, nb = (k + 7) // 8, (n + 7) // 8
    out = torch.empty((nb, kb, 8, 8), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 8, kb * 8], block_size=[8, 8]):
        out[tn.id, tk.id, :, :] = b[tk, tn]
    return out


@helion.kernel(
    backend="mlir", static_shapes=False, config=helion.Config(block_sizes=[8])
)
def pad_rows_kernel(x: torch.Tensor, rows: hl.constexpr) -> torch.Tensor:
    out = torch.empty((rows, x.size(1)), dtype=x.dtype, device=x.device)
    for tm in hl.tile(rows):
        out[tm, :] = x[tm, :]
    return out


def _packed_blocks(b: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(b, (0, -b.shape[1] % 8, 0, -b.shape[0] % 8))
    kb, nb = padded.shape[0] // 8, padded.shape[1] // 8
    return padded.reshape(kb, 8, nb, 8).permute(2, 0, 1, 3).contiguous()


@_kernel()
def pack_vnni_blocks_kernel(b: torch.Tensor) -> torch.Tensor:
    """``[K, N] -> [N/32, K/32, 16, 32, 2]``, zero-padded: K pairs innermost."""
    k, n = b.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b[tk, tn].reshape(tk.block_size // 2, 2, tn.block_size)
        out[tn.id, tk.id, :, :, :] = pairs.permute(0, 2, 1)
    return out


def _packed_vnni_blocks(b: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(b, (0, -b.shape[1] % 32, 0, -b.shape[0] % 32))
    kb, nb = padded.shape[0] // 32, padded.shape[1] // 32
    blocks = padded.reshape(kb, 16, 2, nb, 32)
    return blocks.permute(3, 0, 1, 4, 2).contiguous()


@pytest.mark.parametrize("shape", [(24, 32), (21, 30)], ids=["aligned", "padded"])
def test_packing_blocks_padded_in_kernel(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    check_kernel(pack_blocks_kernel, _packed_blocks, [torch.randn(shape)])


def test_tile_id_store_index_is_parallel() -> None:
    ir = str(generate_mlir(pack_blocks_kernel, [torch.randn(21, 30)]))
    assert "scf.forall" in ir


@_kernel(32, 64, 16, 24)
def subtile_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each grid tile of rows is split into row panels that each run over all of K."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tmo, tn in hl.tile([m, n]):
        for tm in hl.tile(tmo.begin, tmo.end):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
    return out


@_kernel(32, 16)
def overlapping_rows_kernel(x: torch.Tensor) -> torch.Tensor:
    """Every grid iteration writes all rows: not partitioned by the grid tile."""
    out = torch.zeros_like(x)
    for _tg in hl.tile(x.size(0)):
        for t in hl.tile(x.size(0)):
            out[t] = out[t] + x[t]
    return out


@pytest.mark.parametrize(
    "shape",
    [(64, 48, 128), (70, 45, 100), (20, 16, 30)],
    ids=["aligned", "partial", "single_partial_tile"],
)
def test_subtile_loop_stores(shape: tuple[int, int, int]) -> None:
    """A loop over one grid tile's range (``hl.tile(t.begin, t.end)``) with an
    inner block that does not divide it, and partial grid tiles."""
    torch.manual_seed(0)
    m, k, n = shape
    check_kernel(
        subtile_matmul_kernel,
        torch.matmul,
        [torch.randn(m, k), torch.randn(k, n)],
        atol=1e-4,
        rtol=1e-4,
    )


def test_subtile_loop_stores_are_parallel() -> None:
    ir = str(
        generate_mlir(
            subtile_matmul_kernel, [torch.randn(64, 48), torch.randn(48, 128)]
        )
    )
    assert "scf.forall" in ir


def test_loop_over_whole_range_stays_sequential() -> None:
    torch.manual_seed(0)
    x = torch.randn(40)
    assert "scf.forall" not in str(generate_mlir(overlapping_rows_kernel, [x]))
    check_kernel(overlapping_rows_kernel, lambda x: x * 2, [x])


def test_padded_rows_of_runtime_width() -> None:
    torch.manual_seed(0)
    x = torch.randn(13, 10)
    expected = torch.zeros(24, 10)
    expected[:13] = x
    torch.testing.assert_close(
        run_direct(pad_rows_kernel, [x, hl.constexpr(24)]), expected
    )


def test_combined_2d() -> None:
    torch.manual_seed(0)
    check_kernel(add_2d_kernel, torch.add, [torch.randn(20, 36), torch.randn(20, 36)])


def test_ragged_reduction_loop() -> None:
    torch.manual_seed(0)
    check_kernel(row_sum_kernel, lambda x: x.sum(dim=-1), [torch.randn(10, 20)])


def test_row_softmax_ragged_rows() -> None:
    torch.manual_seed(0)
    check_kernel(
        row_softmax_kernel, lambda x: torch.softmax(x, -1), [torch.randn(20, 16)]
    )


def test_softmax_over_ragged_tile() -> None:
    torch.manual_seed(0)
    x = -torch.rand(20, 6) - 1.0  # all negative, so zero padding would win amax

    def reference(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.softmax(chunk, 0) for chunk in x.split(8)])

    check_kernel(chunk_softmax_kernel, reference, [x])


def test_nonzero_begin() -> None:
    torch.manual_seed(0)
    x = torch.randn(40)

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[3:] = x[3:] * 2.0
        return out

    check_kernel(offset_scale_kernel, reference, [x])


def test_read_past_tensor_end_is_zero() -> None:
    """A tile read past the tensor's end reads zeros there."""
    torch.manual_seed(0)
    a = torch.randn(19)
    result = run_direct(read_past_end_kernel, [a, hl.constexpr(64)])
    expected = torch.zeros(64)
    expected[:19] = a
    torch.testing.assert_close(result, expected)


def test_extra_mask_on_load() -> None:
    torch.manual_seed(0)
    check_kernel(masked_load_kernel, lambda x: _every_other(x, 0.0), [torch.randn(40)])


def test_extra_mask_on_store() -> None:
    torch.manual_seed(0)
    check_kernel(
        masked_store_kernel, lambda x: _every_other(x, -1.0), [torch.randn(40)]
    )


def test_divisible_extent_has_static_ir() -> None:
    ir = str(generate_mlir(scale_2d_kernel, [torch.randn(64, 96)]))
    assert "tensor.pad" not in ir
    assert "affine.min" not in ir
    assert "?" not in ir


def test_read_after_write_of_partial_region() -> None:
    torch.manual_seed(0)
    check_kernel(store_then_load_kernel, lambda x: (x + 1.0) * 2.0, [torch.randn(40)])


@pytest.mark.isolated
def test_opt_pipeline() -> None:
    torch.manual_seed(0)
    x = torch.randn(70, 100)
    y = torch.randn(100, 50)
    with opt_pipeline():
        scaled = scale_2d_kernel(x)
        product = matmul_kernel(x, y)
    torch.testing.assert_close(scaled, x * 3.0)
    torch.testing.assert_close(product, x @ y, atol=1e-3, rtol=1e-3)


@pytest.mark.isolated
def test_opt_pipeline_padded_blocks() -> None:
    """A bufferized ``tensor.pad`` temporary read back whole had poison padding."""
    torch.manual_seed(0)
    b = torch.randn(45, 70)
    with opt_pipeline():
        packed = pack_blocks_kernel(b)
    torch.testing.assert_close(packed, _packed_blocks(b))


@pytest.mark.isolated
@pytest.mark.parametrize("shape", [(64, 96), (45, 70)], ids=["aligned", "padded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_opt_pipeline_vnni_blocks(shape: tuple[int, int], dtype: torch.dtype) -> None:
    """A tile split into K pairs moved innermost: an untiled vector transpose."""
    torch.manual_seed(0)
    b = torch.randn(shape).to(dtype)
    with opt_pipeline():
        packed = pack_vnni_blocks_kernel(b)
    torch.testing.assert_close(packed, _packed_vnni_blocks(b), atol=0, rtol=0)


def test_static_reshape_is_expand_shape() -> None:
    ir = str(generate_mlir(pack_vnni_blocks_kernel, [torch.randn(64, 96)]))
    assert "tensor.expand_shape" in ir
    assert "tensor.reshape" not in ir
