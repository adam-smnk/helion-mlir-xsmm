"""One-row reductions split across subgroups, in the IR
(:func:`~helion_mlir_backend._compiler.mlir_transforms.xegpu_layouts.plan_row_reduction`):
the layout the rule picks, hints, and what is refused. The results on the device are in
``test_xpu_row_reduction_runtime``."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import helion
import lighthouse.dialects as lh_dialects
from mlir import ir
import pytest
import torch

from tests import xpu_row_reduction_kernels as rows
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms import xegpu_layouts as xl

if TYPE_CHECKING:
    from collections.abc import Callable


def _lower(
    fn: Callable[..., object],
    m: int,
    n: int,
    block_n: int | None,
    stage: str = "xegpu-wg",
    dtype: torch.dtype = torch.float32,
    **config: object,
) -> str:
    blocks = [1] if block_n is None else [block_n, 1]
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, **config),
    )
    return str(
        lower_mlir(kernel, [xpu_tensor(m, n, dtype=dtype)], target="xpu", stop_at=stage)
    )


def _layouts(text: str) -> list[tuple[str, str]]:
    return re.findall(
        r"(xegpu\.load|xegpu\.store) .*?layout = (#xegpu\.layout<[^>]*>)", text
    )


def _threads(text: str) -> int:
    return int(re.search(r"known_block_size = array<i32: (\d+),", text)[1])


def _slm_sizes(text: str) -> list[int]:
    """The bytes of each SLM array of a kernel lowered to ``xevm``."""
    return [
        int(size)
        for size in re.findall(
            r"llvm\.mlir\.global internal @__global_alloca\S* \{addr_space = 3 : i32\} : "
            r"!llvm\.array<1 x array<(\d+) x i8>>",
            text,
        )
    ]


@pytest.mark.parametrize(
    ("block_n", "subgroups", "sg_cols"),
    [(2048, 32, 64), (1024, 32, 32), (512, 16, 32), (256, 16, 16), (128, 1, 128)],
)
def test_the_rule_splits_the_columns(
    block_n: int, subgroups: int, sg_cols: int
) -> None:
    text = _lower(rows.l1_two_passes, 64, 2048, block_n)
    assert _threads(text) == subgroups * 16
    layouts = _layouts(text)
    assert {name for name, _ in layouts} == {"xegpu.load", "xegpu.store"}
    for _, layout in layouts:
        assert f"sg_layout = [{subgroups}], sg_data = [{sg_cols}]" in layout


def test_every_access_takes_one_element_per_lane() -> None:
    """A wider gather under the reduction's ``lane_data = [1]`` takes wrong elements; one
    in the output pass makes a division by the row result a vector ``fdiv``, which
    loses subnormal divisors (``test_a_subnormal_row_mean`` on the device)."""
    layouts = _layouts(_lower(rows.l1_two_passes, 64, 4096, 2048))
    assert [name for name, _ in layouts] == ["xegpu.load", "xegpu.load", "xegpu.store"]
    assert all("inst_data = [16]" in layout for _, layout in layouts)


def test_the_lowering_reduces_through_slm() -> None:
    """One SLM array of a partial result per subgroup, one barrier, all lanes reduced
    twice, no reduction left, and the workaround's casts gone."""
    text = _lower(rows.l1_two_passes, 64, 2048, 1024, stage="xevm")
    assert _slm_sizes(text) == [32 * 4]
    assert len(re.findall(r"llvm\.call spir_funccc @_Z7barrierj", text)) == 1
    # Two 16-lane vectors of the K loop and two read back: four chains of four.
    assert (
        len(re.findall(r"llvm\.call spir_funccc @_Z21sub_group_shuffle_xor", text))
        == 16
    )
    assert "multi_reduction" not in text
    assert "bitcast" not in text


def test_one_subgroup_needs_no_slm() -> None:
    text = _lower(rows.l1_two_passes, 64, 1024, 128, stage="xevm")
    assert _slm_sizes(text) == []
    assert "@_Z7barrierj" not in text


def test_two_reductions_take_two_arrays() -> None:
    text = _lower(rows.standardize, 64, 2048, 1024, stage="xevm")
    assert _slm_sizes(text) == [128, 128]


@pytest.mark.parametrize(
    "fn",
    [
        rows.l2_two_passes,
        rows.minus_row_maximum,
        rows.minus_row_minimum,
    ],
)
def test_forms_lower(fn: Callable[..., object]) -> None:
    assert _slm_sizes(_lower(fn, 64, 2048, 1024, stage="xevm")) == [128]


def test_a_reduction_of_a_load_lowers() -> None:
    assert _slm_sizes(
        _lower(rows.direct_sum_of_a_load, 64, 1024, None, stage="xevm")
    ) == [128]


def test_bf16_tiles_with_f32_sums_lower() -> None:
    text = _lower(
        rows.bf16_with_f32_sums, 64, 2048, 1024, stage="xevm", dtype=torch.bfloat16
    )
    assert _slm_sizes(text) == [128]


def test_sg_tile_hint_is_honoured() -> None:
    text = _lower(rows.l1_two_passes, 64, 2048, 1024, xegpu_sg_tiles=[[1, 64]])
    assert _threads(text) == 16 * 16
    assert "sg_layout = [16], sg_data = [64]" in _layouts(text)[0][1]


def test_a_sg_tile_hint_may_take_the_narrow_split() -> None:
    text = _lower(rows.l1_two_passes, 64, 2048, 512, xegpu_sg_tiles=[[1, 16]])
    assert _threads(text) == 32 * 16


@pytest.mark.parametrize("hint", [[1, 48], [1, 128], [2, 32], [1, 0]])
def test_a_sg_tile_hint_that_does_not_fit_is_dropped(hint: list[int]) -> None:
    """48 columns don't divide; 128 leave 8 subgroups, which upstream reduces across
    part of a subgroup; a row can't take 2 rows; 0 is no split."""
    text = _lower(rows.l1_two_passes, 64, 2048, 1024, xegpu_sg_tiles=[hint])
    assert _threads(text) == 32 * 16


@pytest.mark.parametrize("hint", [[32], [64], [24], [2, 16]])
def test_a_load_tile_hint_can_not_widen_the_gathers(hint: list[int]) -> None:
    layouts = _layouts(
        _lower(rows.l1_two_passes, 64, 4096, 2048, xegpu_load_tiles=[hint])
    )
    assert all("inst_data = [16]" in layout for _, layout in layouts)


@pytest.mark.parametrize(
    ("fn", "shape", "block_n", "match"),
    [
        (rows.product, (64, 2048), 1024, "'mul' reduction"),
        (rows.row_results, (64, 2048), 1024, "0-D vector"),
        (rows.tile_and_row_results, (64, 2048), 1024, "0-D vector"),
        (rows.partial_sums_in_the_loop, (64, 2048), 1024, "0-D vector"),
        (rows.norm_of_each_tile, (64, 2048), 1024, "inside a loop"),
        (rows.l1_two_passes, (64, 96), 24, "no subgroup split"),
    ],
)
def test_refused(
    fn: Callable[..., object], shape: tuple[int, int], block_n: int, match: str
) -> None:
    with pytest.raises(UnsupportedOperationError, match=match):
        _lower(fn, *shape, block_n)


def test_too_many_columns_for_the_registers_are_refused() -> None:
    with pytest.raises(UnsupportedOperationError, match="no subgroup split"):
        _lower(rows.l1_two_passes, 8, 262144, 131072)


def test_partial_k_tiles_are_refused() -> None:
    """Partial K tiles (2000 columns in 1024) leave no whole gathers."""
    with pytest.raises(UnsupportedOperationError, match="XeGPU|tile|subgroup"):
        _lower(rows.l1_two_passes, 64, 2000, 1024)


@pytest.mark.parametrize("m", [8192, 32768])
def test_the_spec_shapes_lower(m: int) -> None:
    for fn in (rows.l1_two_passes, rows.l2_two_passes):
        text = _lower(fn, m, 65536, 2048, stage="xevm")
        assert _slm_sizes(text) == [128]


def test_a_two_row_tile_keeps_its_rows() -> None:
    """Only a one-row tile is split: 2 rows keep pinned columns."""
    kernel = helion.kernel(
        rows.l1_two_passes,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 8]),
    )
    text = str(lower_mlir(kernel, [xpu_tensor(64, 2048)], target="xpu", stop_at="xevm"))
    assert "@_Z7barrierj" not in text
    assert _slm_sizes(text) == []


# --- The refusals, on the exact IR ------------------------------------------------------

_KERNEL = """
module attributes {{gpu.container_module}} {{
  gpu.module @m {{
    gpu.func @f(%src: i64, %dst: i64) kernel attributes {{known_block_size = array<i32: 16, 1, 1>}} {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %zero = arith.constant 0.0 : f32
      %init = arith.constant dense<0.0> : vector<64xf32>
      %mask = arith.constant dense<true> : vector<64xi1>
      %step = vector.step : vector<64xindex>
      %offsets = arith.constant dense<0> : vector<64xindex>
      {load}
      {body}
      gpu.return
    }}
  }}
}}
"""
_LOAD = (
    "%v = xegpu.load %src[%step], %mask : i64, vector<64xindex>, vector<64xi1> -> "
    "vector<64xf32>"
)
_STORE = (
    "xegpu.store %v, %dst[%step], %mask : vector<64xf32>, i64, vector<64xindex>, "
    "vector<64xi1>"
)
_SUM = "%s = vector.multi_reduction <add>, %v, %zero [0] : vector<64xf32> to f32"


def _plan(body: str, load: str = _LOAD) -> xl.Plan | None:
    module = ir.Module.parse(_KERNEL.format(load=load, body=body))
    gpu_module = next(iter(module.body.operations)).operation
    func = next(iter(gpu_module.regions[0].blocks[0].operations)).operation
    return xl.plan_row_reduction(xl.analyze(func), xl.Hints())


@pytest.fixture
def _context():
    with ir.Context() as context, ir.Location.unknown():
        lh_dialects.register_and_load()
        context.allow_unregistered_dialects = True
        yield context


@pytest.mark.usefixtures("_context")
def test_the_ir_of_a_row_kernel_is_taken() -> None:
    plan = _plan(f"{_SUM}\n{_STORE}")
    assert plan is not None
    assert plan.grid.rows == 1
    assert plan.grid.subgroups == 1
    assert len(plan.reductions_to_separate) == 1
    assert plan.threads == 16


@pytest.mark.usefixtures("_context")
def test_a_kernel_with_no_reduction_is_not_this_plans() -> None:
    assert _plan(_STORE) is None


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    ("load", "body", "match"),
    [
        (
            _LOAD.replace("%step", "%offsets"),
            f"{_SUM}\n{_STORE}",
            "isn't whole columns",
        ),
        (
            _LOAD,
            f"{_SUM}\n{_STORE.replace('%step', '%offsets')}",
            "isn't whole columns",
        ),
        (
            _LOAD,
            "%s = vector.multi_reduction <minsi>, %i, %c7 [0] : vector<64xi32> to i32",
            "'minsi' reduction",
        ),
        (
            "%v = xegpu.load %src[%step], %mask : i64, vector<64xindex>, vector<64xi1> "
            "-> vector<64xf16>",
            "%z = arith.constant 0.0 : f16\n"
            "%s = vector.multi_reduction <add>, %v, %z [0] : vector<64xf16> to f16",
            "reduction across subgroups of",
        ),
        (
            _LOAD,
            f"{_SUM}\n%b = arith.cmpf ogt, %s, %zero : f32\nscf.if %b {{\n}}\n{_STORE}",
            "'scf.if'",
        ),
        (
            _LOAD,
            f"scf.for %i = %c0 to %c4 step %c1 {{\n{_SUM}\n}}\n{_STORE}",
            "inside a loop",
        ),
        (_LOAD, f"{_SUM}\nmemref.store %s, %m[] : memref<f32>", "memref.store"),
        (
            _LOAD,
            f"{_SUM}\n%c = arith.constant dense<[1.0, 2.0]> : vector<2xf32>",
            "isn't a splat",
        ),
        (
            _LOAD,
            f"{_SUM}\n%w = vector.broadcast %s : f32 to vector<32xf32>",
            "shape",
        ),
        (
            _LOAD,
            f"{_SUM}\n%w = vector.interleave %v, %v : vector<64xf32> -> vector<128xf32>",
            "vector.interleave",
        ),
    ],
)
def test_what_it_has_not_reasoned_about_is_refused(
    load: str, body: str, match: str
) -> None:
    if "memref<f32>" in body:
        body = f"%m = memref.alloca() : memref<f32>\n{body}"
    if "minsi" in body:
        load = (
            "%v = xegpu.load %src[%step], %mask : i64, vector<64xindex>, vector<64xi1> "
            "-> vector<64xf32>\n%i = arith.bitcast %v : vector<64xf32> to vector<64xi32>\n"
            "%c7 = arith.constant 7 : i32"
        )
    with pytest.raises(xl.LayoutError, match=match):
        _plan(body, load)
