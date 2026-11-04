"""Each hook of ``tests/ir_invariants.py`` fires, and leaves declared forms alone."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests import ir_invariants
from tests import ir_query
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.targets import xpu as xpu_target

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get(ir_invariants.MODE_VARIABLE) == "off",
    reason="the hooks are off in this run",
)
_CONFIG = helion.Config(block_sizes=[32, 32])


@pytest.fixture(autouse=True)
def _failing_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests' violations must raise, also in a run in ``report:`` mode."""
    monkeypatch.setenv(ir_invariants.MODE_VARIABLE, "fail")


_WIDE = torch.float64


def _widens_by_a_name(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


def _declared(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float64) * 2.0).to(x.dtype)
    return out


def _times_two(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(fn, backend="mlir", static_shapes=True, config=_CONFIG)


def _tensor(device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if device == "xpu":
        return xpu_tensor(64, 64, dtype=dtype)
    return torch.zeros(64, 64, dtype=dtype)


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_undeclared_f64_kernel_fails_the_frontend(device: str) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        generate_mlir(_kernel(_widens_by_a_name), [_tensor(device)], target=device)


def test_compare_of_a_tile_widened_to_f64_is_flagged() -> None:
    """The IR torch-mlir made of ``x <= 0.1``: what the hook first found."""
    module = ir.Module.parse(
        """
func.func @f(%x: f32) -> i1 {
  %cst = arith.constant 1.000000e-01 : f64
  %wide = arith.extf %x : f32 to f64
  %le = arith.cmpf ole, %wide, %cst : f64
  return %le : i1
}""",
        ir.Context(),
    )
    assert ir_invariants.f64_ops(module) == [
        "%0 = arith.extf %arg0 : f32 to f64",
        "%1 = arith.cmpf ole, %0, %cst : f64",
    ]


def _word_in_a_comment(x: torch.Tensor) -> torch.Tensor:
    # float64 and double: words here are no declaration.
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


def _word_in_a_docstring(x: torch.Tensor) -> torch.Tensor:
    """Not a double precision kernel: float64 is only a word here."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


@pytest.mark.parametrize("fn", [_word_in_a_comment, _word_in_a_docstring])
def test_a_word_in_text_is_no_declaration(fn: object) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        generate_mlir(_kernel(fn), [_tensor("cpu")], target="cpu")


@pytest.mark.parametrize(
    ("helper", "found"),
    [
        ("_aten_var_correction_0", 0),
        ("_aten_var_mean_correction_0", 0),
        ("_aten_add_Tensor_0", 1),
    ],
)
def test_only_var_helpers_may_compute_in_f64(helper: str, found: int) -> None:
    module = ir.Module.parse(
        f"""
func.func private @{helper}(%x: f32) -> f64 {{
  %0 = arith.extf %x : f32 to f64
  return %0 : f64
}}""",
        ir.Context(),
    )
    assert len(ir_invariants.f64_ops(module)) == found


def test_scalar_numbers_narrowed_are_not_flagged() -> None:
    module = ir.Module.parse(
        """
func.func @f(%alpha: f64) -> f32 {
  %cst = arith.constant 1.000000e-01 : f64
  %0 = arith.truncf %alpha : f64 to f32
  %1 = arith.truncf %cst : f64 to f32
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}""",
        ir.Context(),
    )
    assert ir_invariants.f64_ops(module) == []


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_f64_in_the_source_or_arguments_is_declared(device: str) -> None:
    generate_mlir(_kernel(_declared), [_tensor(device)], target=device)
    generate_mlir(_kernel(_times_two), [_tensor(device, torch.float64)], target=device)


@pytest.mark.parametrize("device", ["cpu"])
def test_a_kernel_call_goes_through_the_f64_hook(device: str) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        _kernel(_widens_by_a_name)(_tensor(device))


@pytest.mark.isolated
def test_a_kernel_call_goes_through_the_f64_hook_in_an_isolated_child() -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        _kernel(_widens_by_a_name)(_tensor("cpu"))


@pytest.mark.xfail(strict=True, reason="a named exemption that doesn't fire fails")
@pytest.mark.expect_violation("layouts")
def test_an_exemption_that_does_not_fire_fails() -> None:
    pass


def test_unknown_modes_are_refused() -> None:
    assert ir_invariants.valid_mode("fail")
    assert ir_invariants.valid_mode("off")
    assert ir_invariants.valid_mode("report:/tmp/x")
    assert not ir_invariants.valid_mode("report:")
    assert not ir_invariants.valid_mode("Fail")
    assert not ir_invariants.valid_mode("")


def test_report_mode_records_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "findings.jsonl"
    monkeypatch.setenv(ir_invariants.MODE_VARIABLE, f"report:{report}")
    generate_mlir(_kernel(_widens_by_a_name), [_tensor("cpu")], target="cpu")
    finding = json.loads(report.read_text())
    assert finding["check"] == "f64-not-asked-for"
    assert finding["where"] == "frontend of _widens_by_a_name"
    assert "test_report_mode_records_and_goes_on" in finding["test"]
    assert any("arith.extf" in line for line in finding["first"])


_XEGPU = """
func.func @f(%src: memref<8x8xf32>) -> vector<8x8xf32> {
  %t = xegpu.create_nd_tdesc %src : memref<8x8xf32> -> !xegpu.tensor_desc<8x8xf32>
  %v = xegpu.load_nd %t[0, 0] : !xegpu.tensor_desc<8x8xf32> -> vector<8x8xf32>
  return %v : vector<8x8xf32>
}
"""


class _Keep:
    """A stage leaving the module as it is."""

    def apply(self, module: ir.Module) -> ir.Module:
        return module


def _xegpu_module() -> ir.Module:
    context = ir.Context()
    context.allow_unregistered_dialects = False
    return ir.Module.parse(_XEGPU, context)


def test_memory_op_without_layout_fails_the_xegpu_wg_stage() -> None:
    module = _xegpu_module()
    assert ir_invariants.ops_without_layout(module) == ["xegpu.load_nd without layout"]
    with pytest.raises(ir_invariants.IRInvariantError, match="layouts at stage"):
        xpu_target._apply_stage("xegpu-wg", _Keep(), module, False)


def test_gather_only_broadcast_needs_no_layout_others_do() -> None:
    gather = """
func.func @f(%p: i64, %i: vector<8xindex>, %m: vector<8xi1>) -> vector<8x8xf32> {{
  %0 = xegpu.load %p[%i], %m : i64, vector<8xindex>, vector<8xi1> -> vector<8xbf16>
  %1 = arith.extf %0 : vector<8xbf16> to vector<8xf32>
  {use}
  return %2 : vector<8x8xf32>
}}"""
    context = ir.Context()
    context.allow_unregistered_dialects = False
    broadcast = "%2 = vector.broadcast %1 : vector<8xf32> to vector<8x8xf32>"
    other = "%2 = ub.poison : vector<8x8xf32>\n  %3 = arith.addf %1, %1 : vector<8xf32>"
    good = ir.Module.parse(gather.format(use=broadcast), context)
    assert ir_invariants.ops_without_layout(good) == []
    bad = ir.Module.parse(gather.format(use=other), context)
    assert ir_invariants.ops_without_layout(bad) == ["xegpu.load without layout"]


def test_xegpu_op_after_xevm_fails_the_xevm_stage() -> None:
    module = _xegpu_module()
    assert ir_invariants.xegpu_ops_left(module) == [
        "xegpu.create_nd_tdesc",
        "xegpu.load_nd",
    ]
    with pytest.raises(ir_invariants.IRInvariantError, match="xegpu-left at stage"):
        xpu_target._apply_stage("xevm", _Keep(), module, False)
    # Not a concern before that stage.
    xpu_target._apply_stage("xegpu-initial", _Keep(), module, False)


@pytest.mark.parametrize(
    ("vector", "found"),
    [
        ("vector<16xf16>", []),
        ("vector<8xf16>", []),
        ("vector<4xf16>", ["llvm.intr.fabs vector<4xf16>"]),
        ("vector<32xf16>", ["llvm.intr.fabs vector<32xf16>"]),
        ("vector<16xf32>", ["llvm.intr.fabs vector<16xf32>"]),
    ],
)
def test_only_the_dumped_lane_counts_of_f16_fabs_are_allowed(
    vector: str, found: list[str]
) -> None:
    module = ir.Module.parse(
        f"""
llvm.func @f(%a: {vector}) -> () {{
  %0 = llvm.intr.fabs(%a) : ({vector}) -> {vector}
  llvm.return
}}""",
        ir.Context(),
    )
    assert ir_invariants.vector_compares_and_abs(module) == found


def test_vector_compares_and_abs_are_found_scalars_are_not() -> None:
    module = ir.Module.parse(
        """
llvm.func @f(%a: vector<4xf32>, %b: f32, %c: vector<4xi32>) -> () {
  %0 = llvm.fcmp "olt" %a, %a : vector<4xf32>
  %1 = llvm.fcmp "olt" %b, %b : f32
  %2 = llvm.intr.fabs(%a) : (vector<4xf32>) -> vector<4xf32>
  %3 = llvm.intr.fabs(%b) : (f32) -> f32
  %4 = llvm.icmp "slt" %c, %c : vector<4xi32>
  %5 = "llvm.intr.abs"(%c) <{is_int_min_poison = false}> : (vector<4xi32>) -> vector<4xi32>
  llvm.return
}
""",
        ir.Context(),
    )
    assert ir_invariants.vector_compares_and_abs(module) == [
        "llvm.fcmp vector<4xi1>",
        "llvm.icmp vector<4xi1>",
        "llvm.intr.abs vector<4xi32>",
        "llvm.intr.fabs vector<4xf32>",
    ]


@pytest.mark.isolated
def test_a_good_lowering_passes_every_stage_check() -> None:
    ran: list[str] = []

    def checked(stage: str) -> tuple[str, object]:
        check, findings = ir_invariants._STAGE_CHECKS[stage]

        def record(module: ir.Module) -> list[str]:
            ran.append(stage)
            return findings(module)

        return check, record

    with (
        patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}),
        patch.dict(
            ir_invariants._STAGE_CHECKS,
            {stage: checked(stage) for stage in ("xegpu-wg", "xevm")},
        ),
    ):
        args = [xpu_tensor(64, 64)]
        assert "llvm." in lower_xpu(_kernel(_times_two), args, "xevm")
    assert ran == ["xegpu-wg", "xevm"]


@pytest.mark.isolated
def test_xevm_final_is_after_the_workaround_stages() -> None:
    """The stop point runs the stages between ``xevm`` and the binary, without the
    binary (so with no SPIR-V target or device)."""
    seen: list[str] = []
    apply_stage = xpu_target._apply_stage

    def record(name: str, *rest: object) -> ir.Module:
        seen.append(name)
        return apply_stage(name, *rest)

    args = [xpu_tensor(64, 64)]
    with (
        patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}),
        patch.object(xpu_target, "_apply_stage", record),
    ):
        final = lower_xpu(_kernel(_times_two), args, "xevm-final")
    assert "llvm." in final
    assert "gpu.binary" not in final
    after_xevm = seen[seen.index("xevm") + 1 :]
    assert after_xevm[0] == "xevm-allocas"
    assert after_xevm[-1] == "xevm-lane-conversions"


def test_layout_of_reads_the_fields_of_one_attribute() -> None:
    module = ir_query.parse(
        """
func.func @f(%t: !xegpu.tensor_desc<8x8xf32>) -> vector<8x8xf32> {
  %v = xegpu.load_nd %t[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [4, 8], order = [0, 1]>}> : !xegpu.tensor_desc<8x8xf32> -> vector<8x8xf32>
  %w = xegpu.load_nd %t[0, 0] : !xegpu.tensor_desc<8x8xf32> -> vector<8x8xf32>
  return %v : vector<8x8xf32>
}"""
    )
    op = ir_query.ops_named(module, "xegpu.load_nd")
    assert ir_query.layout_of(op[0]) == ir_query.Layout((2, 1), (4, 8), None, (0, 1))
    assert ir_query.layout_of(op[1]) is None
    assert ir_query.layout_of(op[0], "layout_a") is None


def test_vector_compare_after_the_workaround_stages_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ir_invariants, "_WORKAROUND_STAGES", [("keep", _Keep())])
    module = ir.Module.parse(
        """
llvm.func @f(%a: vector<4xf32>) -> () {
  %0 = llvm.fcmp "olt" %a, %a : vector<4xf32>
  llvm.return
}""",
        ir.Context(),
    )
    with pytest.raises(ir_invariants.IRInvariantError, match="llvm.fcmp"):
        xpu_target._apply_stage("xevm", _Keep(), module, False)


@pytest.mark.isolated
def test_a_stop_at_xevm_remembers_the_workaround_stages() -> None:
    with patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}):
        lower_xpu(_kernel(_times_two), [xpu_tensor(64, 64)], "xevm")
    names = [name for name, _ in ir_invariants._WORKAROUND_STAGES]
    assert names[0] == "xevm-allocas"
    assert names[-1] == "xevm-lane-conversions"


@pytest.mark.isolated
def test_hooks_are_on_in_isolated_children() -> None:
    assert xpu_target._apply_stage is ir_invariants._checked_apply_stage
