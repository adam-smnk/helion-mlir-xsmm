"""Partial tiles on the XPU, at the IR level: out-of-bounds transfers and block
accesses on the true surface; refusals for what can't be right.
``test_xpu_partial_tiles_runtime.py`` runs them on the device."""

from __future__ import annotations

import helion.language as hl
from mlir import ir
import pytest
import torch

from tests import xpu_partial_kernels as pk
from tests.harness import lower_xpu
from tests.harness import xpu_tensor
from tests.test_partial_tile_running_values import softmax as running_softmax

from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms import partial_tiles


def _args(case: pk.Case) -> list[torch.Tensor]:
    return [
        *(xpu_tensor(*shape, dtype=dtype) for shape, dtype, _ in case.inputs),
        xpu_tensor(*case.output[0], dtype=case.output[1]),
    ]


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(pk.CASES))
def test_partial_tile_lowers(name: str) -> None:
    case = pk.CASES[name]
    text = lower_xpu(case.kernel(), _args(case), "xevm")
    assert "gpu.launch_func" in text


_PARTIAL_GEMMS = ["gemm_m", "gemm_n", "gemm_k", "gemm_all", "gemm_t_k", "gemm_batched"]


@pytest.mark.isolated
@pytest.mark.parametrize("name", _PARTIAL_GEMMS)
def test_partial_gemm_needs_layouts_from_the_ir(name: str) -> None:
    """Not lighthouse's parameter error: the backend's refusal, naming the toggle."""
    case = pk.CASES[name]
    with pytest.raises(UnsupportedOperationError, match='xegpu_layouts="ir"') as info:
        lower_xpu(case.kernel(xegpu_layouts=["lighthouse"]), _args(case))
    assert "partial tiles on lighthouse" in str(info.value)


@pytest.mark.isolated
def test_whole_gemm_on_lighthouses_selection_lowers() -> None:
    args = [xpu_tensor(512, 256, dtype=pk.BF16), xpu_tensor(256, 512, dtype=pk.BF16)]
    args.append(xpu_tensor(512, 512, dtype=pk.BF16))
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(
            block_sizes=[256, 256, 32], xegpu_layouts=["lighthouse"]
        ),
    )(pk.gemm)
    assert "gpu.launch_func" in lower_xpu(kernel, args)


@pytest.mark.isolated
@pytest.mark.parametrize("name", ["scale_f32_both", "row_max_both", "attention_both"])
def test_other_classes_lower_on_lighthouses_annotation(name: str) -> None:
    case = pk.CASES[name]
    lower_xpu(case.kernel(xegpu_layouts=["lighthouse"]), _args(case), "xevm")


@pytest.mark.isolated
def test_k_below_its_tile_is_refused() -> None:
    """A K tile clamped to K leaves no K loop."""
    case = pk._gemm_case(512, 16, 512)
    with pytest.raises(UnsupportedOperationError, match="without a K loop"):
        lower_xpu(case.kernel(), _args(case))


@pytest.mark.isolated
def test_block_accesses_on_the_true_surface() -> None:
    """A partial tile's block ops are the whole tile's, on the tensor's own
    memref, with the bounds check on; nothing is padded or copied."""
    case = pk.CASES["scale_f32_both"]
    text = lower_xpu(case.kernel(), _args(case))
    assert "tensor.pad" not in text
    assert "memref.copy" not in text
    surfaces = [line for line in text.splitlines() if "create_nd_tdesc" in line]
    assert surfaces
    for line in surfaces:
        assert "memref<253x480xf32>" in line
        assert "tensor_desc<64x128xf32>" in line
        assert "boundary_check = false" not in line


@pytest.mark.isolated
def test_rank3_surface_is_one_plane() -> None:
    """A partial row count clips at the plane's end, not the next plane's."""
    case = pk.CASES["scale_rank3"]
    text = lower_xpu(case.kernel(), _args(case))
    surfaces = [line for line in text.splitlines() if "create_nd_tdesc" in line]
    assert surfaces
    for line in surfaces:
        assert "memref<253x480xbf16, strided<[480, 1], offset: ?>>" in line


@pytest.mark.isolated
def test_one_row_tiles_are_masked() -> None:
    """1D tiles have no bounds check in hardware: gathers and scatters are masked."""
    case = pk.CASES["scale_one_row_f32"]
    text = lower_xpu(case.kernel(), _args(case))
    assert "tensor_desc" not in text
    assert "arith.cmpi slt" in text


@pytest.mark.isolated
def test_row_results_store_is_masked() -> None:
    """A row result past the output's end is not stored."""
    case = pk.CASES["row_max_rows"]
    text = lower_xpu(case.kernel(), _args(case))
    assert "dense<true>" not in text.split("xegpu.store ")[0].splitlines()[-1]
    stores = [line for line in text.splitlines() if "xegpu.store " in line]
    assert len(stores) == 1
    assert "vector<64x1xi1>" in stores[0]


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(pk.UNALIGNED))
def test_unaligned_rows_are_refused(name: str) -> None:
    case = pk.UNALIGNED[name]
    with pytest.raises(UnsupportedOperationError, match="partial tiles on rows of"):
        lower_xpu(case.kernel(), _args(case))


@pytest.mark.isolated
def test_dynamic_shapes_stay_refused() -> None:
    """Dynamic memrefs never reach the transfers: upstream's surface check skips
    them, so a wrong pitch would be silent."""
    case = pk.CASES["scale_f32_cols"]
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=False,
        config=pk.helion.Config(block_sizes=case.blocks),
    )(case.fn)
    args = [torch.randn(256, 508), torch.randn(256, 508)]
    with pytest.raises(UnsupportedOperationError, match="dynamic shapes on XPU"):
        lower_xpu(kernel, args)


_PAD_TEMPLATE = """
#min = affine_map<(d0) -> (64, -d0 + 100)>
#rest = affine_map<(d0) -> (-d0 + 64)>
func.func @f(%x: tensor<100xf32>, %i: index) -> tensor<64xf32> {{
  %c0 = arith.constant {value} : f32
  %n = affine.min #min(%i)
  %h = affine.apply #rest(%n)
  %s = tensor.extract_slice %x[%i] [%n] [1] : tensor<100xf32> to tensor<?xf32>
  %p = tensor.pad %s low[0] high[%h] {{
  ^bb0(%j: index):
    tensor.yield %c0 : f32
  }} : tensor<?xf32> to tensor<64xf32>
  return %p : tensor<64xf32>
}}
"""


@pytest.mark.isolated
def test_zero_pad_becomes_an_out_of_bounds_read() -> None:
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(_PAD_TEMPLATE.format(value="0.0"))
        assert partial_tiles.to_transfers(module)
        text = str(module)
    assert "tensor.pad" not in text
    reads = [line for line in text.splitlines() if "vector.transfer_read" in line]
    assert len(reads) == 1
    assert "%arg0[%arg1]" in reads[0]
    assert "in_bounds" not in reads[0]  # false, the printed default


@pytest.mark.isolated
def test_other_pads_stay() -> None:
    """A pad of another value: a block read's zero fill is not it. Left for the
    gate to refuse."""
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(_PAD_TEMPLATE.format(value="1.0"))
        partial_tiles.to_transfers(module)
        text = str(module)
    assert "tensor.pad" in text
    assert "vector.transfer_read" not in text


_WRITE_ON_VIEW = """
#min = affine_map<(d0) -> (64, -d0 + {end})>
func.func @f(%m: memref<100x128xf32>, %v: vector<64x128xf32>, %i: index) {{
  %c0 = arith.constant 0 : index
  %n = affine.min #min(%i)
  %s = memref.subview %m[%i, 0] [%n, 128] [1, 1]
      : memref<100x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
  vector.transfer_write %v, %s[%c0, %c0] {{in_bounds = [false, true]}}
      : vector<64x128xf32>, memref<?x128xf32, strided<[128, 1], offset: ?>>
  return
}}
"""


@pytest.mark.isolated
def test_write_moves_to_the_buffer_when_bounds_agree() -> None:
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(_WRITE_ON_VIEW.format(end=100))
        partial_tiles.fold_tile_views(module)
        text = str(module)
    assert "vector.transfer_write %arg1, %arg0[" in text
    assert "in_bounds = [false, true]" in text


@pytest.mark.isolated
def test_write_stays_on_a_view_that_ends_earlier() -> None:
    """A view ending before the buffer: moved, the check would be against the
    wrong bound and elements past the view would be stored."""
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(_WRITE_ON_VIEW.format(end=90))
        partial_tiles.fold_tile_views(module)
        text = str(module)
    assert "vector.transfer_write %arg1, %0[" in text or "%subview[" in text


@pytest.mark.isolated
def test_no_extents_no_rewrite() -> None:
    """A kernel of whole tiles is left alone: spec shapes keep their IR."""
    source = """
    func.func @f(%x: tensor<128xf32>, %i: index) -> tensor<64xf32> {
      %s = tensor.extract_slice %x[%i] [64] [1] : tensor<128xf32> to tensor<64xf32>
      return %s : tensor<64xf32>
    }"""
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(source)
        before = str(module)
        assert not partial_tiles.to_transfers(module)
        assert str(module) == before


@pytest.mark.isolated
def test_running_value_keeps_its_old_elements() -> None:
    """A running sum's padded lanes keep the old value: ``insert_slice`` of a runtime
    size into a local value lowered as if it ran off a surface, adding padding."""
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 64]),
    )(running_softmax)
    text = lower_xpu(kernel, [xpu_tensor(128, 252)])
    total_loop = text.split("scf.for")[2]
    assert "arith.addf" in total_loop
    assert "arith.select" in total_loop.split("arith.addf")[1]


def stops_inside_the_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_m):
        for tile_n in hl.tile(248, block_size=block_n):
            y[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return y


@pytest.mark.isolated
def test_partial_tile_inside_the_surface_is_refused() -> None:
    """A full block access there would touch real elements past the tile: reads
    wouldn't be zero and writes wouldn't be dropped."""
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 64]),
    )(stops_inside_the_tensor)
    args = [xpu_tensor(128, 256), xpu_tensor(128, 256)]
    with pytest.raises(UnsupportedOperationError, match="partial tiles"):
        lower_xpu(kernel, args)


@pytest.mark.isolated
def test_running_rows_keep_their_selects() -> None:
    """Regression: the kept value's condition on rows went with the masks', and 12 of
    252 rows took padded rows."""
    case = pk.CASES["running_rows"]
    text = lower_xpu(case.kernel(), _args(case))
    loop = text.split("scf.for")[1]
    assert "arith.select" in loop


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("rows", "columns", "match"),
    [(128, 254, "partial tiles on rows of 254"), (125, 257, "multiple of 16")],
)
def test_unaligned_output_refused(rows: int, columns: int, match: str) -> None:
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 64]),
    )(wider_output)
    args = [xpu_tensor(rows, 252 if columns == 254 else 256), xpu_tensor(128, columns)]
    with pytest.raises(UnsupportedOperationError, match=match):
        lower_xpu(kernel, args, "xevm")


def wider_output(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = x[tm, tn] * 2.0 + 1.0
    return o


@pytest.mark.isolated
def test_partial_tile_of_integers_lowers() -> None:
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 64]),
    )(copied)
    args = [
        xpu_tensor(128, 252, dtype=torch.int32),
        xpu_tensor(128, 252, dtype=torch.int32),
    ]
    assert "gpu.launch" in lower_xpu(kernel, args)


def copied(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = x[tm, tn] + x[tm, tn]
    return o


def row_vector_broadcast(
    x: torch.Tensor, y: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = x[tm, tn] + y[tm, None]
    return o


@pytest.mark.isolated
def test_row_vector_at_partial_rows_refused() -> None:
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 128]),
    )(row_vector_broadcast)
    args = [xpu_tensor(125, 512), xpu_tensor(125), xpu_tensor(125, 512)]
    with pytest.raises(UnsupportedOperationError, match="XeGPU's gathers"):
        lower_xpu(kernel, args, "xevm")


def bool_output(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        o[tm, tn] = x[tm, tn] > 0.0
    return o


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((128, 252), r"(?s)rows of 252 elements.*252 bytes"),
        ((125, 256), "1-bit elements"),
    ],
    ids=["252", "256"],
)
def test_bool_output_refusal_says_what_is_true(
    shape: tuple[int, int], match: str
) -> None:
    """Regression: a region's 64 columns were taken for the surface's, and an i1's
    width for its storage: "rows of 64 elements, 8 bytes"."""
    kernel = pk.helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=pk.helion.Config(block_sizes=[64, 64]),
    )(bool_output)
    args = [xpu_tensor(*shape), xpu_tensor(*shape, dtype=torch.bool)]
    with pytest.raises(UnsupportedOperationError, match=match) as info:
        lower_xpu(kernel, args, "xevm")
    assert "rows of 64" not in str(info.value)
