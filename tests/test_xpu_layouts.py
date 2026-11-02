"""XeGPU layouts derived from the IR (``xegpu_layouts``): the toggle, the same
layouts as lighthouse's annotation where a kernel has them, the dim classes and
grids on small kernels, and what is refused."""

from __future__ import annotations

import re

import helion
import lighthouse.dialects as lh_dialects
from mlir import ir
import pytest

from tests import kernelbench_xpu as kb
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import XeGPUConfigError
from helion_mlir_backend._compiler.mlir_transforms import xegpu_layouts as xl
from helion_mlir_backend._compiler.targets.xpu_params import xegpu_params

PROBLEMS = (*kb.PROBLEMS, *kb.RANK3_PROBLEMS, *kb.LEVEL2_PROBLEMS)
HAS_EXPLICIT_FIELDS = {"97_ScaledDotProductAttention"}
"""Lighthouse's attention annotation leaves ``inst_data`` and ``order`` to the
propagation, whose values IR-derived anchors state: same otherwise."""


def _lower(problem: kb.Problem, stage: str = "xegpu-wg", **extra: object) -> str:
    config = helion.Config(
        block_sizes=list(problem.block_sizes),
        **{**getattr(problem, "xegpu", {}), **extra},
    )
    inputs = [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]
    if hasattr(problem, "args"):
        inputs = problem.args(*inputs)
    return str(
        lower_mlir(problem.kernel, inputs, config=config, target="xpu", stop_at=stage)
    )


def _kernels(text: str) -> str:
    """The GPU modules: the host's constants may be ordered differently."""
    return text[text.index("  gpu.module") :]


def _without_explicit_fields(text: str) -> str:
    text = re.sub(r", inst_data = \[\d+, \d+\]", "", text)
    return re.sub(r", order = \[\d+, \d+\]", "", text)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
def test_same_kernels_as_lighthouse(problem: kb.Problem) -> None:
    """The toggle's reproduction claim: with lighthouse's selector as the source
    and the same config, the same XeGPU, on every KernelBench kernel."""
    lighthouse = _kernels(_lower(problem))
    derived = _kernels(_lower(problem, xegpu_layouts=["ir-selector"]))
    if problem.name in HAS_EXPLICIT_FIELDS:
        lighthouse = _without_explicit_fields(lighthouse)
        derived = _without_explicit_fields(derived)
    assert derived == lighthouse


# --- The toggle ------------------------------------------------------------------


def test_toggle_values() -> None:
    params = xegpu_params(
        {"xegpu_layouts": ["ir", "lighthouse", None, "ir-selector", "ir-upstream"]}
    )
    assert [p.layout_source for p in params] == [
        "rule",
        None,
        None,
        "selector",
        "upstream",
    ]


@pytest.mark.parametrize(
    "value", ["ir", ["IR"], [1], [True]], ids=["not-a-list", "name", "int", "bool"]
)
def test_toggle_refuses(value: object) -> None:
    with pytest.raises(XeGPUConfigError, match="xegpu_layouts"):
        xegpu_params({"xegpu_layouts": value})


def test_toggle_is_part_of_the_module_text() -> None:
    """JIT keys: a kernel with another annotator is another module."""
    from helion_mlir_backend._compiler.targets.xpu_params import PARAMS_ATTR
    from helion_mlir_backend._compiler.targets.xpu_params import attach_params

    with ir.Context():
        module = ir.Module.parse("module {}")
        attach_params(module, xegpu_params({"xegpu_layouts": ["ir"]}))
        assert PARAMS_ATTR in str(module)
        assert "layouts" in str(module)


def test_lighthouse_value_is_the_default() -> None:
    problem = kb.PROBLEMS[0]
    assert _lower(problem) == _lower(problem, xegpu_layouts=["lighthouse"])


def test_forced_tiles_are_checked_from_the_ir() -> None:
    problem = kb.PROBLEMS[0]
    with pytest.raises(XeGPUConfigError, match="does not divide"):
        _lower(problem, xegpu_layouts=["ir"], xegpu_sg_tiles=[[33, 64]])


# --- Small kernels, parsed: nothing but the IR ---------------------------------------

_TDESC = "!xegpu.tensor_desc<{}xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>"
_FUNC = """
module attributes {{gpu.container_module}} {{
  func.func @host(%a: memref<1024x1024xbf16>, %b: memref<1024x1024xbf16>,
                  %c: memref<1024x1024xbf16>) {{
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    gpu.launch_func @m::@k blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%a : memref<1024x1024xbf16>, %b : memref<1024x1024xbf16>,
             %c : memref<1024x1024xbf16>)
    return
  }}
  gpu.module @m [#xevm.target<O = 3>] {{
    gpu.func @k(%a: memref<1024x1024xbf16>, %b: memref<1024x1024xbf16>,
                %c: memref<1024x1024xbf16>) kernel attributes {{
        known_block_size = array<i32: 16, 1, 1> {attrs} }} {{
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      {body}
      gpu.return
    }}
  }}
}}
"""


def _gemm_body(rows: int = 256, cols: int = 256, k: int = 32) -> str:
    a = _TDESC.format(f"{rows}x{k}")
    b = _TDESC.format(f"{k}x{cols}")
    c = _TDESC.format(f"{rows}x{cols}")
    return f"""
      %zero = arith.constant dense<0.0> : vector<{rows}x{cols}xf32>
      %da = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {a}
      %db = xegpu.create_nd_tdesc %b : memref<1024x1024xbf16> -> {b}
      %acc = scf.for %i = %c0 to %c1024 step %c32 iter_args(%x = %zero) -> (vector<{rows}x{cols}xf32>) {{
        %la = xegpu.load_nd %da[%c0, %i] : {a} -> vector<{rows}x{k}xbf16>
        %lb = xegpu.load_nd %db[%i, %c0] : {b} -> vector<{k}x{cols}xbf16>
        %d = xegpu.dpas %la, %lb, %x : vector<{rows}x{k}xbf16>, vector<{k}x{cols}xbf16>, vector<{rows}x{cols}xf32> -> vector<{rows}x{cols}xf32>
        scf.yield %d : vector<{rows}x{cols}xf32>
      }}
      %t = arith.truncf %acc : vector<{rows}x{cols}xf32> to vector<{rows}x{cols}xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {c}
      xegpu.store_nd %t, %dc[%c0, %c0] : vector<{rows}x{cols}xbf16>, {c}
    """


def _reduction_after_gemm(rows: int = 256, cols: int = 64, k: int = 32) -> str:
    """A GEMM, then a sum over its columns, then its output scaled by the sum."""
    gemm = _gemm_body(rows, cols, k)
    head, _, _ = gemm.partition("%t = arith.truncf")
    c = _TDESC.format(f"{rows}x{cols}")
    return f"""{head}
      %zr = arith.constant dense<0.0> : vector<{rows}xf32>
      %r = vector.multi_reduction <add>, %acc, %zr [1] : vector<{rows}x{cols}xf32> to vector<{rows}xf32>
      %bc = vector.broadcast %r : vector<{rows}xf32> to vector<{cols}x{rows}xf32>
      %tr = vector.transpose %bc, [1, 0] : vector<{cols}x{rows}xf32> to vector<{rows}x{cols}xf32>
      %q = arith.divf %acc, %tr : vector<{rows}x{cols}xf32>
      %t = arith.truncf %q : vector<{rows}x{cols}xf32> to vector<{rows}x{cols}xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {c}
      xegpu.store_nd %t, %dc[%c0, %c0] : vector<{rows}x{cols}xbf16>, {c}
    """


def _parse(body: str, hints: str = "") -> ir.Module:
    attrs = f', {xl.HINTS_ATTR} = {{{hints}}}' if hints else ""
    return ir.Module.parse(_FUNC.format(attrs=attrs, body=body))


def _func(module: ir.Module) -> ir.Operation:
    return next(
        op.operation
        for gpu_module in module.body.operations
        if gpu_module.operation.name == "gpu.module"
        for op in gpu_module.operation.regions[0].blocks[0].operations
        if op.operation.name == "gpu.func"
    )


@pytest.fixture(autouse=True)
def _context():
    with ir.Context() as context, ir.Location.unknown():
        lh_dialects.register_and_load()
        context.allow_unregistered_dialects = True
        yield context


def _anchors(module: ir.Module, name: str) -> list[str]:
    found: list[str] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == name:
            found.append(str(op))
        return ir.WalkResult.ADVANCE

    _func(module).walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def test_gemm_dim_classes_and_grid() -> None:
    module = _parse(_gemm_body())
    kernel = xl.analyze(_func(module))
    axes = xl.axes_of(kernel)
    assert (axes.free_rows, axes.free_cols) == (True, True)
    grids = xl.legal_grids(axes)
    assert xl.Grid(8, 4) in grids and xl.Grid(4, 8) in grids
    assert xl.preferred_grid(axes, grids) == xl.Grid(8, 4)  # 32x64
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert plan.threads == 512
    text = str(module)
    assert "known_block_size = array<i32: 512, 1, 1>" in text
    assert "threads in (%" in text
    a_load, b_load = _anchors(module, "xegpu.load_nd")
    assert "sg_layout = [8, 4], sg_data = [32, 32]" in a_load
    assert "sg_layout = [8, 4], sg_data = [32, 64]" in b_load
    (dpas,) = _anchors(module, "xegpu.dpas")
    assert "layout_cd = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64]" in dpas


def test_launch_func_gets_the_threads() -> None:
    module = _parse(_gemm_body())
    xl.annotate_layouts(module.operation, _func(module))
    host = str(module).split("gpu.module", 1)[0]
    assert "arith.constant 512" in host


def test_sg_tile_hint_overrides_the_rule() -> None:
    module = _parse(_gemm_body(), "sg_tile = array<i64: 64, 32>")
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert plan.grid == xl.Grid(4, 8)
    assert xl.HINTS_ATTR not in str(module)


@pytest.mark.parametrize(
    ("hint", "match"),
    [
        ("sg_tile = array<i64: 33, 64>", "does not divide"),
        ("sg_tile = array<i64: 8, 16>", "exceed"),
        ('source = "guess"', "unknown layout source"),
        (
            "sg_tile = array<i64: 64, 32>, load_tiles = array<i64: 64, 32, 16, 16>",
            "at most",
        ),
        ("load_tiles = array<i64: 7, 16, 16, 16>", "does not divide"),
        ("prefetch_tiles = array<i64: 7, 16, 16, 16>", "does not divide"),
    ],
)
def test_hints_refused(hint: str, match: str) -> None:
    module = _parse(_gemm_body(), hint)
    with pytest.raises(xl.LayoutError, match=match):
        xl.annotate_layouts(module.operation, _func(module))


def test_reduced_columns_pin_the_grid_to_rows() -> None:
    """A reduction over a GEMM's columns: all subgroups on the rows, no column
    split (it would need SLM and a barrier); ``[rows, cols]`` is refused if it
    splits them."""
    module = _parse(_reduction_after_gemm())
    kernel = xl.analyze(_func(module))
    axes = xl.axes_of(kernel)
    assert (axes.free_rows, axes.free_cols) == (True, False)
    assert all(grid.cols == 1 for grid in xl.legal_grids(axes))
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert plan.grid == xl.Grid(16, 1)
    assert plan.threads == 256
    module = _parse(_reduction_after_gemm(), "sg_tile = array<i64: 32, 32>")
    with pytest.raises(xl.LayoutError, match="contracted or reduced"):
        xl.annotate_layouts(module.operation, _func(module))


def test_unknown_op_on_tiles_is_refused() -> None:
    body = _gemm_body().replace(
        "%t = arith.truncf %acc :",
        '%u = "test.mystery"(%acc) : (vector<256x256xf32>) -> vector<256x256xf32>\n'
        "      %t = arith.truncf %u :",
    )
    module = _parse(body)
    with pytest.raises(xl.LayoutError, match="test.mystery"):
        xl.analyze(_func(module))


def test_dim_size_conflict_is_refused() -> None:
    classes = xl.DimClasses()
    module = _parse(_gemm_body(256, 512))
    func = _func(module)
    kernel = xl.analyze(func)
    (dpas,) = kernel.dpas
    with pytest.raises(xl.LayoutError, match="distributed alike"):
        classes.tie(dpas.results[0], 0, dpas.results[0], 1)


def test_partial_columns_are_refused() -> None:
    """Columns that leave no whole DPAS: no grid, not a silent layout."""
    module = _parse(_gemm_body(256, 24))
    with pytest.raises(xl.LayoutError):
        xl.annotate_layouts(module.operation, _func(module))


def test_shared_operand_has_no_grid() -> None:
    """A rows-only kernel: an operand on neither axis is read whole, ``[1, 1]``."""
    module = _parse(_reduction_after_gemm())
    func = _func(module)
    kernel = xl.analyze(func)
    (dpas,) = kernel.dpas
    layout = xl.value_layout(kernel, xl.Grid(16, 1), dpas.operands[1])
    assert layout == xl.ValueLayout((1, 1), (32, 64))
    layout = xl.value_layout(kernel, xl.Grid(16, 1), dpas.operands[0])
    assert layout == xl.ValueLayout((16, 1), (16, 32))


def test_unanchored_dpas_is_refused() -> None:
    """Upstream's lowering would complete it silently (its subgroup propagation
    defaults; gap L3, temp/megakernel/REPORT.md): the check does not."""
    module = _parse(_gemm_body())
    func = _func(module)
    xl.annotate_layouts(module.operation, func)
    for op in xl._walk(func):
        if op.name == "xegpu.dpas":
            del op.attributes["layout_a"]
    problems = xl.check_anchors(func)
    assert problems == ["'xegpu.dpas' has no layout_a"]


def test_check_layouts_refuses_more_subgroups_than_the_kernel() -> None:
    module = _parse(_gemm_body())
    func = _func(module)
    xl.annotate_layouts(module.operation, func)
    func.attributes["known_block_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])
    problems = xl.check_anchors(func)
    assert any("more than the kernel's 16 subgroups" in p for p in problems)


def test_check_layouts_refuses_inst_data_that_does_not_divide() -> None:
    module = _parse(_gemm_body())
    func = _func(module)
    xl.annotate_layouts(module.operation, func)
    for op in xl._walk(func):
        if op.name == "xegpu.store_nd":
            op.attributes["layout"] = ir.Attribute.parse(
                "#xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [7, 16]>"
            )
    assert any("does not divide" in p for p in xl.check_anchors(func))


@pytest.mark.parametrize(
    ("hint", "match"),
    [
        ("prefetch_depths = array<i64: 1, 1>", "apply to loads of DPAS operands"),
        ("prefetch_tiles = array<i64: 16, 16, 16, 16>", "apply to loads of DPAS"),
        ("load_tiles = array<i64: 16, 16, 16, 16>", "apply to DPAS operands"),
    ],
)
def test_hints_without_a_target_are_refused(hint: str, match: str) -> None:
    body = f"""
      %t = arith.constant dense<0.0> : vector<64x64xf32>
      %r0 = arith.constant dense<0.0> : vector<64xf32>
      %r = vector.multi_reduction <add>, %t, %r0 [1] : vector<64x64xf32> to vector<64xf32>
      %bc = vector.broadcast %r : vector<64xf32> to vector<64x64xf32>
      %tr = vector.transpose %bc, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
      %out = arith.truncf %tr : vector<64x64xf32> to vector<64x64xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {_TDESC.format("64x64")}
      xegpu.store_nd %out, %dc[%c0, %c0] : vector<64x64xbf16>, {_TDESC.format("64x64")}
    """
    module = _parse(body, hint)
    with pytest.raises(xl.LayoutError, match=match):
        xl.annotate_layouts(module.operation, _func(module))


def test_a_2d_load_tile_needs_a_tile_that_is_no_operand() -> None:
    module = _parse(_gemm_body(), "load_tiles = array<i64: 8, 16>")
    with pytest.raises(xl.LayoutError, match="no DPAS operands"):
        xl.annotate_layouts(module.operation, _func(module))


def test_prefetch_tile_limits() -> None:
    module = _parse(_gemm_body(), "prefetch_tiles = array<i64: 4, 16, 16, 16>")
    with pytest.raises(xl.LayoutError, match="2D block prefetches"):
        xl.annotate_layouts(module.operation, _func(module))


def test_prefetch_depth_zero_is_none() -> None:
    """Explicitly no prefetch of A, a form lighthouse's annotation refuses."""
    module = _parse(_gemm_body(), "prefetch_depths = array<i64: 0, 1>")
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert len(plan.prefetches) == 1


@pytest.mark.parametrize(
    ("hint", "k", "match"),
    [
        ("sg_tile = array<i64: 128, 256>", 32, "parallel DPAS"),
        ("sg_tile = array<i64: 64, 32>", 64, "in registers"),
    ],
)
def test_forced_tiles_must_fit_a_subgroup(hint: str, k: int, match: str) -> None:
    module = _parse(_gemm_body(256, 256, k), hint)
    with pytest.raises(xl.LayoutError, match=match):
        xl.annotate_layouts(module.operation, _func(module))
