"""XeGPU layouts derived from the IR (``xegpu_layouts``): the toggle, the same
layouts as lighthouse's annotation where a kernel has them, the dim classes and
grids on small kernels, and what is refused."""

from __future__ import annotations

import contextlib
import os
import re

import helion
import helion.language as hl
import lighthouse.dialects as lh_dialects
from mlir import ir
from mlir.passmanager import PassManager
import pytest
import torch

from tests import kernelbench_xpu as kb
from tests import xpu_layout_forms as forms
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir.support import XeGPUConfigError
from helion_mlir_backend._compiler.mlir_transforms import xegpu_layouts as xl
from helion_mlir_backend._compiler.targets.xpu_params import xegpu_params

PROBLEMS = (*kb.PROBLEMS, *kb.RANK3_PROBLEMS, *kb.LEVEL2_PROBLEMS)
DIFFERS_FROM_LIGHTHOUSE = {
    "6_Matmul_with_large_K_dimension_",
    "7_Matmul_with_small_K_dimension_",
    "9_Tall_skinny_matrix_multiplication_",
    "17_Matmul_with_transposed_B",
    "3_Batched_matrix_multiplication",
    "10_3D_tensor_matrix_multiplication",
    "40_Matmul_Scaling_ResidualAdd",
    "59_Matmul_Swish_Scaling",
    "97_ScaledDotProductAttention",
}
"""Where the rule decides a parameter these problems' keys leave open, measured (97:
explicit fields lighthouse leaves to the propagation, same lowered kernels)."""


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


@pytest.mark.parametrize(
    "problem",
    [p for p in PROBLEMS if p.name not in DIFFERS_FROM_LIGHTHOUSE],
    ids=lambda p: p.name,
)
def test_same_kernels_as_lighthouse(problem: kb.Problem) -> None:
    """With today's keys the annotators agree on all but :data:`DIFFERS_FROM_LIGHTHOUSE`."""
    lighthouse = _kernels(_lower(problem, xegpu_layouts=["lighthouse"]))
    assert _kernels(_lower(problem, xegpu_layouts=["ir"])) == lighthouse


# --- The toggle ------------------------------------------------------------------


def test_toggle_values() -> None:
    params = xegpu_params({"xegpu_layouts": ["ir", "lighthouse"]})
    assert [p.derives_layouts for p in params] == [True, False]


def test_default_layouts_follow_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion_mlir_backend._compiler.targets.xpu_params import DEFAULT_LAYOUTS_ATTR
    from helion_mlir_backend._compiler.targets.xpu_params import KernelParams
    from helion_mlir_backend._compiler.targets.xpu_params import attach_params
    from helion_mlir_backend._compiler.targets.xpu_params import take_params

    entry = xegpu_params({"xegpu_layouts": [None]})[0]
    for value, derived in (("ir", True), ("lighthouse", False)):
        monkeypatch.setenv("HELION_MLIR_XPU_LAYOUTS", value)
        assert entry.derives_layouts == derived
        assert KernelParams().derives_layouts == derived
        module = ir.Module.parse("module {}")
        attach_params(module, ())
        assert DEFAULT_LAYOUTS_ATTR in str(module)
        take_params(module)
        assert DEFAULT_LAYOUTS_ATTR not in str(module)
    monkeypatch.setenv("HELION_MLIR_XPU_LAYOUTS", "guess")
    with pytest.raises(XeGPUConfigError, match="HELION_MLIR_XPU_LAYOUTS"):
        _ = KernelParams().derives_layouts


@pytest.mark.parametrize(
    "value",
    ["ir", ["IR"], [1], [True], ["ir-selector"], ["ir-upstream"]],
    ids=["not-a-list", "name", "int", "bool", "selector", "upstream"],
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


def test_explicit_lighthouse_is_the_lighthouse_path() -> None:
    """The value reaches the same kernels as the environment's default for it."""
    problem = kb.PROBLEMS[0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HELION_MLIR_XPU_LAYOUTS", "lighthouse")
        by_default = _kernels(_lower(problem))
    assert by_default == _kernels(_lower(problem, xegpu_layouts=["lighthouse"]))


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
    attrs = f", {xl.HINTS_ATTR} = {{{hints}}}" if hints else ""
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


def test_reduced_columns_pin_the_grid_to_rows() -> None:
    """A reduction over a GEMM's columns: all subgroups on the rows, no column
    split (it would need SLM and a barrier)."""
    module = _parse(_reduction_after_gemm())
    kernel = xl.analyze(_func(module))
    axes = xl.axes_of(kernel)
    assert (axes.free_rows, axes.free_cols) == (True, False)
    assert all(grid.cols == 1 for grid in xl.legal_grids(axes))
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert plan.grid == xl.Grid(16, 1)
    assert plan.threads == 256
    # A hint that would split them is dropped.
    module = _parse(_reduction_after_gemm(), "sg_tile = array<i64: 32, 32>")
    assert xl.annotate_layouts(module.operation, _func(module)).grid == xl.Grid(16, 1)


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


def test_prefetch_depth_zero_is_none() -> None:
    """Explicitly no prefetch of A, a form lighthouse's annotation refuses."""
    module = _parse(_gemm_body(), "prefetch_depths = array<i64: 0, 1>")
    plan = xl.annotate_layouts(module.operation, _func(module))
    assert len(plan.prefetches) == 1


# --- New forms: refused by lighthouse's path, laid out from their IR --------------


def _form_kernel(form: forms.Form, **config: object) -> helion.Kernel:
    return helion.kernel(
        form.fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=form.blocks, **config),
    )


def _form_args(form: forms.Form) -> list[object]:
    return [xpu_tensor(*shape, dtype=torch.bfloat16) for shape in form.shapes]


@pytest.mark.parametrize("name", list(forms.FORMS))
def test_form_lowers_from_its_ir(name: str) -> None:
    form = forms.FORMS[name]
    kernel = _form_kernel(form, xegpu_layouts=["ir"])
    text = str(lower_mlir(kernel, _form_args(form), target="xpu", stop_at="xegpu-wg"))
    store = re.findall(r"xegpu\.store_nd [^\n]*sg_layout = \[(\d+), (\d+)\]", text)
    assert tuple(map(int, store[-1])) == form.sg_layout
    for op in ("xegpu.dpas", "xegpu.load_nd", "xegpu.store_nd"):
        assert all("layout" in line for line in text.splitlines() if f"{op} " in line)


@pytest.mark.parametrize("name", [n for n, f in forms.FORMS.items() if f.refusal])
def test_form_is_refused_by_lighthouses_path(name: str) -> None:
    form = forms.FORMS[name]
    with pytest.raises(
        (UnsupportedOperationError, XeGPUConfigError), match=form.refusal
    ):
        lower_mlir(
            _form_kernel(form, xegpu_layouts=["lighthouse"]),
            _form_args(form),
            target="xpu",
            stop_at="xegpu-wg",
        )


def test_dot_product_row_norm_is_refused_clearly() -> None:
    """``sum(a * a)`` vectorizes to a ``vector.contract`` of a row with itself, which
    no XeGPU op takes: refused, not a contraction laid out as a DPAS."""

    def row_l2(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        n = w.size(0)
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            acc = hl.zeros([tm, n], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
            out[tm, :] = (acc / torch.sqrt((acc * acc).sum(-1, keepdim=True))).to(
                x.dtype
            )
        return out

    kernel = helion.kernel(
        row_l2,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[128, 32], xegpu_layouts=["ir"]),
    )
    args = [
        xpu_tensor(512, 2048, dtype=torch.bfloat16),
        xpu_tensor(256, 2048, dtype=torch.bfloat16),
    ]
    with pytest.raises(UnsupportedOperationError, match="vector.contract"):
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")


def test_chain_through_buffer_keeps_its_store() -> None:
    """The intermediate of ``chain_through_buffer`` is forwarded in registers before
    XeGPU, and still stored: rows only, each subgroup storing what it owns."""
    form = forms.FORMS["chain_through_buffer"]
    kernel = _form_kernel(form, xegpu_layouts=["ir"])
    text = str(lower_mlir(kernel, _form_args(form), target="xpu", stop_at="xegpu-wg"))
    (stored,) = re.findall(
        r"xegpu\.store_nd [^\n]*sg_layout = (\[\d+, \d+\]), sg_data = (\[\d+, \d+\])[^\n]*vector<64x32",
        text,
    )
    assert stored == ("[8, 1]", "[8, 32]")


def test_buffer_accesses_are_tied() -> None:
    """A tile stored and loaded back at the same offsets is one class: a subgroup
    must load what it stored, with no barrier between subgroups."""
    tile = _TDESC.format("64x64")
    f32 = "vector<64x64xf32>"
    bf16 = "vector<64x64xbf16>"
    body = f"""
      %z = arith.constant dense<0.0> : {f32}
      %da = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {tile}
      %db = xegpu.create_nd_tdesc %b : memref<1024x1024xbf16> -> {tile}
      %scratch = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {tile}
      %la = xegpu.load_nd %da[%c0, %c0] : {tile} -> {bf16}
      %r = scf.for %i = %c0 to %c1024 step %c32 iter_args(%x = %z) -> ({f32}) {{
        %lb = xegpu.load_nd %db[%i, %c0] : {tile} -> {bf16}
        %d = xegpu.dpas %la, %lb, %x : {bf16}, {bf16}, {f32} -> {f32}
        scf.yield %d : {f32}
      }}
      %t = arith.truncf %r : {f32} to {bf16}
      xegpu.store_nd %t, %scratch[%c0, %c0] : {bf16}, {tile}
      %back = xegpu.load_nd %scratch[%c0, %c0] : {tile} -> {bf16}
      %e = arith.extf %back : {bf16} to {f32}
      %q = arith.addf %e, %e : {f32}
      %u = arith.truncf %q : {f32} to {bf16}
      xegpu.store_nd %u, %scratch[%c0, %c0] : {bf16}, {tile}
    """
    module = _parse(body)
    kernel = xl.analyze(_func(module))
    stored = next(a for a in kernel.stores)
    loaded = next(a for a in kernel.loads if a.source == stored.source)
    assert kernel.classes.same((stored.value, 0), (loaded.value, 0))
    assert kernel.classes.same((stored.value, 1), (loaded.value, 1))


def test_accesses_at_other_offsets_are_not_tied() -> None:
    """``x @ x^T``: one buffer read as A and B, at unrelated offsets. Its rows and
    its contracted dim stay apart, or every grid would be refused."""
    tile = _TDESC.format("256x32")
    b_tile = _TDESC.format("32x256")
    c_tile = _TDESC.format("256x256")
    body = f"""
      %z = arith.constant dense<0.0> : vector<256x256xf32>
      %row = arith.constant 256 : index
      %da = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {tile}
      %db = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {b_tile}
      %r = scf.for %i = %c0 to %c1024 step %c32 iter_args(%x = %z) -> (vector<256x256xf32>) {{
        %la = xegpu.load_nd %da[%c0, %i] : {tile} -> vector<256x32xbf16>
        %lb = xegpu.load_nd %db[%i, %row] : {b_tile} -> vector<32x256xbf16>
        %d = xegpu.dpas %la, %lb, %x : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %d : vector<256x256xf32>
      }}
      %t = arith.truncf %r : vector<256x256xf32> to vector<256x256xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {c_tile}
      xegpu.store_nd %t, %dc[%c0, %c0] : vector<256x256xbf16>, {c_tile}
    """
    module = _parse(body)
    axes = xl.axes_of(xl.analyze(_func(module)))
    assert (axes.free_rows, axes.free_cols) == (True, True)


_ROW_VECTOR_STORE = """
module attributes {gpu.container_module} {
  gpu.module @m [#xevm.target<O = 3>] {
    gpu.func @k(%src: memref<64x64xf32>, %dst: memref<64xf32>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %mask = arith.constant dense<true> : vector<64xi1>
      %c0 = arith.constant 0 : index
      %init = arith.constant dense<0xFF800000> : vector<64xf32>
      %t = xegpu.create_nd_tdesc %src : memref<64x64xf32> -> !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %v = xegpu.load_nd %t[%c0, %c0] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x16xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x16xf32>
      %r = vector.multi_reduction <maxnumf>, %v, %init [1] : vector<64x16xf32> to vector<64xf32>
      %idx = vector.step : vector<64xindex>
      %ptr = memref.extract_aligned_pointer_as_index %dst : memref<64xf32> -> index
      %p = arith.index_cast %ptr : index to i64
      xegpu.store %r, %p[%idx], %mask <{layout = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [8, 16]>, dims = [1]>}> : vector<64xf32>, i64, vector<64xindex>, vector<64xi1>
      gpu.return
    }
  }
}
"""


def test_layout_warnings_fail_the_lowering() -> None:
    """Upstream only warns on a load layout it cannot complete, then lowers wrong
    rows: with IR-derived layouts the warning is a failure."""
    from helion_mlir_backend._compiler.targets.xpu import _LayoutWarningsAreErrors

    class Warns:
        def apply(self, module: ir.Module) -> ir.Module:
            PassManager.parse(
                "builtin.module(gpu-lower-to-xevm-pipeline{xegpu-op-level=workgroup "
                "binary-format=skip})",
                module.context,
            ).run(module.operation)
            return module

    module = ir.Module.parse(_ROW_VECTOR_STORE)
    with pytest.raises(UnsupportedOperationError, match="lane layouts"):
        _LayoutWarningsAreErrors(Warns()).apply(module)


def test_rule_load_tiles() -> None:
    """The rule's operand loads, by subgroup tile: 16 rows of 32 columns where they
    divide it, smaller otherwise."""
    assert xl._rule_load((32, 32)) == (16, 32)
    assert xl._rule_load((32, 64)) == (16, 32)
    assert xl._rule_load((8, 32)) == (8, 32)
    assert xl._rule_load((16, 16)) == (16, 16)
    assert xl._rule_load((4, 16)) == (4, 16)


def test_rule_anchors_its_load_tiles() -> None:
    module = _parse(_gemm_body())
    xl.annotate_layouts(module.operation, _func(module))
    a_load, b_load = _anchors(module, "xegpu.load_nd")
    assert "sg_data = [32, 32], inst_data = [16, 32]" in a_load
    assert "sg_data = [32, 64], inst_data = [16, 32]" in b_load


def test_transposed_32_row_read_is_never_laid_out() -> None:
    """Upstream's table allows 32 rows, and it lowers with no diagnostic, but every row
    was wrong on the B50: the hint is dropped, and the read takes 16 rows."""
    a = _TDESC.format("256x32")
    b = _TDESC.format("256x32")
    c = _TDESC.format("256x256")
    body = f"""
      %z = arith.constant dense<0.0> : vector<256x256xf32>
      %da = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {a}
      %db = xegpu.create_nd_tdesc %b : memref<1024x1024xbf16> -> {b}
      %r = scf.for %i = %c0 to %c1024 step %c32 iter_args(%x = %z) -> (vector<256x256xf32>) {{
        %la = xegpu.load_nd %da[%c0, %i] : {a} -> vector<256x32xbf16>
        %lb = xegpu.load_nd %db[%c0, %i] : {b} -> vector<256x32xbf16>
        %tb = vector.transpose %lb, [1, 0] : vector<256x32xbf16> to vector<32x256xbf16>
        %d = xegpu.dpas %la, %tb, %x : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %d : vector<256x256xf32>
      }}
      %t = arith.truncf %r : vector<256x256xf32> to vector<256x256xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {c}
      xegpu.store_nd %t, %dc[%c0, %c0] : vector<256x256xbf16>, {c}
    """
    module = _parse(body, "load_tiles = array<i64: 16, 32, 16, 16>")
    xl.annotate_layouts(module.operation, _func(module))
    (_, transposed) = _anchors(module, "xegpu.load_nd")
    assert "inst_data = [16, 16], order = [0, 1]" in transposed
    module = _parse(body, "load_tiles = array<i64: 16, 32, 32, 16>")
    xl.annotate_layouts(module.operation, _func(module))
    (_, transposed) = _anchors(module, "xegpu.load_nd")
    assert "inst_data = [16, 16], order = [0, 1]" in transposed
    assert "inst_data = [32, 16]" not in str(module)


def test_output_bound_kernel_takes_wide_subgroups() -> None:
    """``M * N >= 32 * K * (M + N)``: fewer subgroups, each storing 8 rows of 128 bytes."""
    body = "%k = arith.constant 8 : index\n" + _gemm_body(64, 128, 32).replace(
        "%c1024 step", "%k step"
    )
    module = _parse(body)
    kernel = xl.analyze(_func(module))
    assert xl.output_bound(kernel)
    axes = xl.axes_of(kernel)
    assert xl.preferred_grid(axes, xl.legal_grids(axes)) == xl.Grid(8, 2)


def test_mildly_output_heavy_kernel_is_not_output_bound() -> None:
    """Factor 16 in the output's favour: 32x32 won there, so it isn't."""
    body = _gemm_body(64, 128, 32).replace("%c1024 step", "%c32 step")
    module = _parse(body)
    assert not xl.output_bound(xl.analyze(_func(module)))


def test_operand_bound_kernel_takes_all_subgroups() -> None:
    """Negative of the above: K dominates, so the grid fills the threads."""
    module = _parse(_gemm_body(64, 128, 32))
    kernel = xl.analyze(_func(module))
    assert not xl.output_bound(kernel)
    axes = xl.axes_of(kernel)
    assert xl.preferred_grid(axes, xl.legal_grids(axes)).subgroups == 32


def test_shared_operand_keeps_attention_loads() -> None:
    """Rows-only kernel, B shared: A 16x16 and B 32x32 tiles, not the GEMM rule's."""
    module = _parse(_reduction_after_gemm(256, 64, 32))
    xl.annotate_layouts(module.operation, _func(module))
    a_load, b_load = _anchors(module, "xegpu.load_nd")
    assert "inst_data = [16, 16]" in a_load
    assert "sg_layout = [1, 1], sg_data = [32, 64], inst_data = [32, 32]" in b_load


def test_b_next_to_a_transposed_a_loads_16_columns() -> None:
    a = _TDESC.format("32x256")
    b = _TDESC.format("32x256")
    c = _TDESC.format("256x256")
    body = f"""
      %z = arith.constant dense<0.0> : vector<256x256xf32>
      %da = xegpu.create_nd_tdesc %a : memref<1024x1024xbf16> -> {a}
      %db = xegpu.create_nd_tdesc %b : memref<1024x1024xbf16> -> {b}
      %r = scf.for %i = %c0 to %c1024 step %c32 iter_args(%x = %z) -> (vector<256x256xf32>) {{
        %la = xegpu.load_nd %da[%i, %c0] : {a} -> vector<32x256xbf16>
        %ta = vector.transpose %la, [1, 0] : vector<32x256xbf16> to vector<256x32xbf16>
        %lb = xegpu.load_nd %db[%i, %c0] : {b} -> vector<32x256xbf16>
        %d = xegpu.dpas %ta, %lb, %x : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %d : vector<256x256xf32>
      }}
      %t = arith.truncf %r : vector<256x256xf32> to vector<256x256xbf16>
      %dc = xegpu.create_nd_tdesc %c : memref<1024x1024xbf16> -> {c}
      xegpu.store_nd %t, %dc[%c0, %c0] : vector<256x256xbf16>, {c}
    """
    module = _parse(body)
    xl.annotate_layouts(module.operation, _func(module))
    a_load, b_load = _anchors(module, "xegpu.load_nd")
    assert "inst_data = [16, 16], order = [0, 1]" in a_load
    assert "sg_data = [32, 64], inst_data = [16, 16]" in b_load


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
def test_default_lowers_every_problem(problem: kb.Problem) -> None:
    """The default path, with each problem's config, reaches LLVM with no XeGPU."""
    text = _lower(problem, "xevm")
    assert "xegpu." not in text
    assert "llvm.func spir_funccc" in text


@pytest.mark.parametrize(
    "name",
    [
        "1_Square_matrix_multiplication_",
        "19_ReLU",
        "23_Softmax",
        "97_ScaledDotProductAttention",
    ],
)
def test_sampled_configs_are_laid_out_or_refused(name: str) -> None:
    """The autotuner's samples under the default: a layout, or a config error,
    later than lighthouse's (at the layout stage) but not an internal failure."""
    from tests.test_xpu_autotune import _samples
    from tests.test_xpu_autotune import _xpu_spec

    problem = next(p for p in PROBLEMS if p.name == name)
    spec = _xpu_spec(problem)
    inputs = [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]
    for config in [spec.default_config(), *_samples(spec, 6)]:
        with contextlib.suppress(XeGPUConfigError, UnsupportedOperationError):
            lower_mlir(
                problem.kernel, inputs, config=config, target="xpu", stop_at="xegpu-wg"
            )


# --- Hints: dropped when they don't fit, with a log line ----------------------------

_GEMM_FORM = forms.GEMM_FORM
_ROW_NORM = forms.FORMS["row_norm_n256"]
_ATTENTION = forms.FORMS["attention_head128"]
_ELEMENTWISE = forms.ELEMENTWISE_FORM
_REDUCTION = forms.REDUCTION_FORM

_PLAIN_B_FORM = forms.PLAIN_B_FORM

DROPPED_HINTS = {
    "load_a_24_rows": (
        forms.GEMM_FORM_48,
        "xegpu_load_tiles",
        [24, 16, 16, 16],
        "load_tiles",
    ),
    "prefetch_not_whole_blocks": (
        _GEMM_FORM,
        "xegpu_prefetch_tiles",
        [12, 16, 16, 16],
        "prefetch_tiles",
    ),
    # name: (form, config key, value, hint in the log, lighthouse's path refuses it too)
    "sg_not_dividing": (_GEMM_FORM, "xegpu_sg_tiles", [33, 64], "sg_tile"),
    "sg_too_many_subgroups": (_GEMM_FORM, "xegpu_sg_tiles", [8, 16], "sg_tile"),
    "sg_too_many_dpas": (_GEMM_FORM, "xegpu_sg_tiles", [128, 256], "sg_tile"),
    "sg_zero_rows": (_GEMM_FORM, "xegpu_sg_tiles", [0, 64], "sg_tile"),
    "sg_splits_reduced_columns": (_ROW_NORM, "xegpu_sg_tiles", [32, 32], "sg_tile"),
    "sg_attention_rows": (_ATTENTION, "xegpu_sg_tiles", [48, 0], "sg_tile"),
    "sg_elementwise": (_ELEMENTWISE, "xegpu_sg_tiles", [48, 64], "sg_tile"),
    "sg_reduction": (_REDUCTION, "xegpu_sg_tiles", [48, 0], "sg_tile"),
    "load_a_1_row": (_GEMM_FORM, "xegpu_load_tiles", [1, 32, 16, 16], "load_tiles"),
    "load_a_2_rows": (_GEMM_FORM, "xegpu_load_tiles", [2, 32, 16, 16], "load_tiles"),
    "load_a_4_rows": (_GEMM_FORM, "xegpu_load_tiles", [4, 32, 16, 16], "load_tiles"),
    "load_a_1_row_16_cols": (
        _GEMM_FORM,
        "xegpu_load_tiles",
        [1, 16, 16, 16],
        "load_tiles",
    ),
    "load_a_8_cols": (_GEMM_FORM, "xegpu_load_tiles", [8, 8, 16, 16], "load_tiles"),
    "load_b_1_row": (_PLAIN_B_FORM, "xegpu_load_tiles", [8, 16, 1, 16], "load_tiles"),
    "load_b_2_rows": (_PLAIN_B_FORM, "xegpu_load_tiles", [8, 16, 2, 16], "load_tiles"),
    "load_b_4_rows": (_PLAIN_B_FORM, "xegpu_load_tiles", [8, 16, 4, 16], "load_tiles"),
    "load_b_8_rows": (_PLAIN_B_FORM, "xegpu_load_tiles", [8, 16, 8, 16], "load_tiles"),
    "load_b_8_cols": (_PLAIN_B_FORM, "xegpu_load_tiles", [8, 16, 16, 8], "load_tiles"),
    "load_not_dividing": (
        _GEMM_FORM,
        "xegpu_load_tiles",
        [7, 16, 16, 16],
        "load_tiles",
    ),
    "load_too_big": (_GEMM_FORM, "xegpu_load_tiles", [64, 32, 16, 16], "load_tiles"),
    "load_two_entries_on_gemm": (_GEMM_FORM, "xegpu_load_tiles", [8, 16], "load_tiles"),
    "load_transposed_32_rows": (
        _GEMM_FORM,
        "xegpu_load_tiles",
        [16, 32, 32, 16],
        "load_tiles",
    ),
    "load_four_on_elementwise": (
        _ELEMENTWISE,
        "xegpu_load_tiles",
        [8, 16, 16, 16],
        "load_tiles",
    ),
    "load_four_on_reduction": (
        _REDUCTION,
        "xegpu_load_tiles",
        [8, 16, 16, 16],
        "load_tiles",
    ),
    "load_reduction_rows": (_REDUCTION, "xegpu_load_tiles", [3, 16], "load_tiles"),
    "prefetch_not_dividing": (
        _GEMM_FORM,
        "xegpu_prefetch_tiles",
        [7, 16, 16, 16],
        "prefetch_tiles",
    ),
    "prefetch_too_small": (
        _GEMM_FORM,
        "xegpu_prefetch_tiles",
        [4, 16, 16, 16],
        "prefetch_tiles",
    ),
    "prefetch_tile_on_reduction": (
        _REDUCTION,
        "xegpu_prefetch_tiles",
        [16, 16, 16, 16],
        "prefetch_tiles",
    ),
    "prefetch_depth_on_elementwise": (
        _ELEMENTWISE,
        "xegpu_prefetch_depths",
        [1, 1],
        "prefetch_depths",
    ),
}
"""Every way a hint was refused before; ``tests/test_xpu_layouts_runtime.py`` runs these
on the device."""


def _lower_hinted(form: forms.Form, stage: str = "xegpu-wg", **config: object) -> str:
    kernel = _form_kernel(form, **config)
    return str(lower_mlir(kernel, _form_args(form), target="xpu", stop_at=stage))


@pytest.mark.parametrize("name", list(DROPPED_HINTS))
def test_dropped_hint_gives_the_rules_layout_and_a_log_line(
    name: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    form, key, value, hint = DROPPED_HINTS[name]
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")  # lower in-process
    with caplog.at_level("WARNING", logger=xl.log.name):
        hinted = _lower_hinted(form, xegpu_layouts=["ir"], **{key: [value]})
    assert f"dropped the hint {hint}=" in caplog.text
    assert _kernels(hinted) == _kernels(_lower_hinted(form, xegpu_layouts=["ir"]))


@pytest.mark.parametrize("name", list(DROPPED_HINTS))
def test_dropped_hint_is_refused_on_lighthouses_path(name: str) -> None:
    """The strict checks stay on lighthouse's path. A form it has no class for is
    refused for that; every other case for the hint itself."""
    form, key, value, _ = DROPPED_HINTS[name]
    with pytest.raises((XeGPUConfigError, UnsupportedOperationError)):
        _lower_hinted(form, xegpu_layouts=["lighthouse"], **{key: [value]})


def test_legal_hint_is_honoured_without_a_log_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    with caplog.at_level("WARNING", logger=xl.log.name):
        text = _lower_hinted(
            _GEMM_FORM, xegpu_layouts=["ir"], xegpu_sg_tiles=[[64, 32]]
        )
    assert "dropped the hint" not in caplog.text
    assert "sg_layout = [4, 8], sg_data = [64, 32]" in text


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("xegpu_sg_tiles", [64]),
        ("xegpu_sg_tiles", [64.0, 32]),
        ("xegpu_load_tiles", [8, 16, 16]),
        ("xegpu_prefetch_depths", [1]),
        ("xegpu_prefetch_tiles", [8, 16]),
    ],
)
def test_malformed_keys_still_fail_under_the_annotator(key: str, value: list) -> None:
    """Typos are not hints."""
    with pytest.raises(XeGPUConfigError, match=key):
        _lower_hinted(_GEMM_FORM, xegpu_layouts=["ir"], **{key: [value]})


def test_an_illegal_layout_without_hints_is_still_refused() -> None:
    """Dropping hints doesn't make a kernel legal: the check after them stays strict."""
    module = _parse(_gemm_body(256, 24), "sg_tile = array<i64: 32, 24>")
    with pytest.raises(xl.LayoutError):
        xl.annotate_layouts(module.operation, _func(module))


def test_tiles_over_the_register_file_are_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A forced tile whose A, B and C don't fit in registers is dropped; the rule's
    grid stands."""
    hinted = _parse(_gemm_body(256, 256, 64), "sg_tile = array<i64: 64, 32>")
    with caplog.at_level("WARNING", logger=xl.log.name):
        plan = xl.annotate_layouts(hinted.operation, _func(hinted))
    assert "dropped the hint sg_tile=" in caplog.text
    assert "in registers" in caplog.text
    ruled = _parse(_gemm_body(256, 256, 64))
    assert plan.grid == xl.annotate_layouts(ruled.operation, _func(ruled)).grid
    assert _anchors(hinted, "xegpu.load_nd") == _anchors(ruled, "xegpu.load_nd")


_TRANSPOSED_32_ROWS = "transpose_32b_32r8x1c"
_TRANSPOSED_16_ROWS = "transpose_32b_16r8x1c"


def test_no_prefetch_of_level2_59_is_refused_on_lighthouses_path() -> None:
    """Depths of 0, the B50's config of 59, lower only under derived layouts: on
    lighthouse's path they are a clean refusal, as its parameters have no such value."""
    problem = next(p for p in PROBLEMS if p.name.startswith("59_"))
    no_prefetch = {"xegpu_prefetch_depths": [[0, 0]]}
    assert "xegpu.prefetch_nd" not in _lower(
        problem, xegpu_layouts=["ir"], **no_prefetch
    )
    with pytest.raises(XeGPUConfigError, match="prefetch depths are positive"):
        _lower(problem, xegpu_layouts=["lighthouse"], **no_prefetch)


@pytest.mark.parametrize("layouts", ["lighthouse", "ir"])
@pytest.mark.parametrize(
    "name",
    [
        "16_Matmul_with_transposed_A",
        "17_Matmul_with_transposed_B",
        "18_Matmul_with_transposed_both",
        "29_Matmul_Mish_Mish",
        "97_ScaledDotProductAttention",
    ],
)
def test_no_annotator_reads_transposed_tiles_in_32_rows(
    name: str, layouts: str
) -> None:
    """Those reads are wrong on every row on the B50, with no diagnostic: nothing
    a kernel with transposed operands lowers to may use them, whichever annotator."""
    problem = next(p for p in PROBLEMS if p.name == name)
    text = _lower(problem, "xevm", xegpu_layouts=[layouts])
    assert _TRANSPOSED_16_ROWS in text
    assert _TRANSPOSED_32_ROWS not in text


def test_lighthouses_path_refuses_a_transposed_32_row_hint() -> None:
    with pytest.raises(XeGPUConfigError, match="transposed"):
        _lower_hinted(
            _GEMM_FORM,
            xegpu_layouts=["lighthouse"],
            xegpu_load_tiles=[[16, 32, 32, 16]],
        )


@pytest.mark.parametrize("rows", [3, 5, 7, 9, 15])
def test_attention_rows_with_no_dpas_are_refused(rows: int) -> None:
    """Subgroups of 1 row abort at launch (``ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED``):
    a clean refusal, as lighthouse's path gives."""
    kernel = helion.kernel(
        forms.attention,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[1, rows, 64], xegpu_layouts=["ir"]),
    )
    args = [
        xpu_tensor(*shape, dtype=torch.bfloat16)
        for shape in ((64, rows, 64), (64, 512, 64), (64, 512, 64))
    ]
    with pytest.raises(UnsupportedOperationError, match="subgroups of"):
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")


def test_reduction_over_the_rows_is_refused() -> None:
    """Exact regression: a sum over columns and rows aborted on an SLM size."""
    kernel = helion.kernel(
        forms.gemm_norm_over_rows,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[128, 32], xegpu_layouts=["ir"]),
    )
    args = [
        xpu_tensor(512, 2048, dtype=torch.bfloat16),
        xpu_tensor(256, 2048, dtype=torch.bfloat16),
    ]
    with pytest.raises(UnsupportedOperationError, match="no subgroup grid"):
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")


def test_transposed_a_keeps_16_row_subgroups() -> None:
    """Exact regression: the rule's grid must leave a transposed read its 16 rows,
    where lighthouse lowers the kernel."""
    module_text = _lower_hinted(
        forms.FORMS["transposed_both_16_rows"], xegpu_layouts=["ir"]
    )
    (load, *_) = re.findall(r"xegpu\.load_nd [^\n]*order = \[0, 1\][^\n]*", module_text)
    assert "inst_data = [16, 16]" in load


def test_transposed_read_that_is_no_operand_has_its_order() -> None:
    """Exact regression: without ``order`` the lowering aborts."""
    text = _lower_hinted(
        forms.FORMS["gemm_plus_transposed"], xegpu_layouts=["ir"], stage="xevm"
    )
    assert "xegpu." not in text


def test_dropped_hint_reaches_the_parent_log(caplog: pytest.LogCaptureFixture) -> None:
    """In the default mode the lowering runs in a forked child, whose stderr is not
    shown: the line must still reach the caller's log."""
    assert os.environ.get("HELION_MLIR_COMPILE_TIMEOUT") != "0"
    with caplog.at_level("WARNING"):
        _lower_hinted(_GEMM_FORM, xegpu_layouts=["ir"], xegpu_sg_tiles=[[33, 64]])
    assert "dropped the hint sg_tile=" in caplog.text


def _strip_anchors(module: ir.Module) -> None:
    prefetches = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        for name in ("layout", "layout_a", "layout_b", "layout_cd"):
            if op.name.startswith("xegpu.") and name in op.attributes:
                del op.attributes[name]
        if op.name == "xegpu.prefetch_nd":
            prefetches.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    for op in prefetches:
        op.erase()


_TRANSPOSED_32_ROWS = """
gpu.module @m [#xevm.target<chip = "bmg">] {
  gpu.func @k(%p: memref<64x8xi32>) -> vector<32x8xi32> {
    %c0 = arith.constant 0 : index
    %d = xegpu.create_nd_tdesc %p : memref<64x8xi32> -> !xegpu.tensor_desc<32x8xi32,
      #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
    %v = xegpu.load_nd %d[%c0, %c0]
      <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1],
        order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32,
      #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
      -> vector<32x8xi32>
    gpu.return %v : vector<32x8xi32>
  }
}
"""


def test_transposed_32_row_read_is_cast_row_by_row() -> None:
    """Why 32 rows stay refused: the reduced form of the wrong rows. A lane's 16 values
    are cast to rows l and l + 16 one after the other, while the read delivers them
    column by column. If this fails, upstream changed the order: try 32 rows on the
    B50, and drop ``check_transposed_reads`` if they are right."""
    with ir.Context() as context, ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(_TRANSPOSED_32_ROWS)
        PassManager.parse(
            "builtin.module(gpu.module(xegpu-sg-to-lane-distribute))", context
        ).run(module.operation)
        text = str(module)
    assert "transpose = array<i64: 1, 0>" in text
    assert "vector.shape_cast %1 : vector<16xi32> to vector<2x8xi32>" in text


@pytest.mark.parametrize(
    ("name", "refused"),
    [
        ("16_Matmul_with_transposed_A", True),
        ("18_Matmul_with_transposed_both", True),
        ("17_Matmul_with_transposed_B", False),
    ],
)
def test_upstreams_default_32_row_transposed_read_is_caught(
    name: str, refused: bool
) -> None:
    """Exact regression, by what the check sees: with no anchors, upstream reads a
    transposed A in 32 rows, and 512 of 512 rows came out wrong on the B50 (max error
    1.4; 2.5e-3 with anchors), with no diagnostic. A transposed B stays at 16 rows."""
    from helion_mlir_backend._compiler.mlir_transforms.xevm_lane_reductions import (
        check_transposed_reads,
    )

    problem = next(p for p in PROBLEMS if p.name == name)
    module = ir.Module.parse(_lower(problem, xegpu_layouts=["lighthouse"]))
    _strip_anchors(module)
    PassManager.parse(
        "builtin.module(gpu-lower-to-xevm-pipeline{xegpu-op-level=workgroup "
        "binary-format=skip})",
        module.context,
    ).run(module.operation)
    if refused:
        with pytest.raises(UnsupportedOperationError, match="32 rows"):
            check_transposed_reads(module)
    else:
        check_transposed_reads(module)


@pytest.mark.parametrize(
    ("tile", "grid"),
    [
        ((128, 128), xl.Grid(4, 4)),  # at most 16 subgroups of 32x32
        ((256, 256), xl.Grid(8, 4)),  # 32 subgroups already give 32x64
        ((32, 256), xl.Grid(2, 16)),  # 32x32 would leave 8: as many as fit
    ],
)
def test_rule_keeps_tiles_of_32x32_while_16_subgroups_remain(
    tile: tuple[int, int], grid: xl.Grid
) -> None:
    module = _parse(_gemm_body(*tile, 32))
    axes = xl.axes_of(xl.analyze(_func(module)))
    assert xl.preferred_grid(axes, xl.legal_grids(axes)) == grid


def test_no_prefetch_tile_fits_means_no_prefetch() -> None:
    """Exact regression: a K=512 grid lighthouse lowers and ``ir`` refused. The tile 32x64
    with 1 subgroup row has no prefetch tile within the kernel's threads: slower, not
    wrong."""
    assert xl._prefetch_tile((512, 64), 2, False) is None
    assert xl._prefetch_tile((64, 64), 32, False) == (8, 16)


@pytest.mark.parametrize("blocks", [[32, 512, 64], [48, 512, 64], [512, 32, 64]])
def test_cells_without_a_prefetch_tile_lower(blocks: list[int]) -> None:
    kernel = helion.kernel(
        forms.matmul,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, xegpu_layouts=["ir"]),
    )
    m = 4 * blocks[0]
    n = 4 * blocks[1]
    args = [
        xpu_tensor(m, 512, dtype=torch.bfloat16),
        xpu_tensor(512, n, dtype=torch.bfloat16),
    ]
    assert "xegpu.dpas" in str(
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")
    )


def test_rule_prefetch_tile_is_whole_blocks() -> None:
    """Regression: a tile :data:`PREFETCH_INST` doesn't divide ended in "a kernel without
    valid XeGPU layouts" with a legal ``xegpu_sg_tiles=[[24, 32]]``."""
    assert xl._prefetch_tile((96, 64), 32, False) in {
        (8, 16),
        (16, 16),
        (8, 32),
        (16, 32),
        (24, 16),
        (24, 32),
        (32, 16),
        (32, 32),
    }
    assert xl._prefetch_blocks((12, 16), False) is False
    assert xl._prefetch_blocks((24, 32), True) is False


def test_tile_over_the_registers_warns_with_the_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No tile fits the registers (4096^3 at [512, 512, 32]: 134956 us against 7123 at
    [128, 128, 32] on the B50, correct, and lighthouse refuses): the rule goes on, and
    says why it will be slow."""
    module = _parse(_gemm_body(512, 512, 32))
    with caplog.at_level("WARNING", logger=xl.log.name):
        xl.annotate_layouts(module.operation, _func(module))
    assert "take smaller block sizes" in caplog.text


@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (forms.elementwise_plus_transposed, ((256, 256), (256, 256))),
        (forms.tile_plus_its_transpose, ((256, 256),)),
    ],
)
def test_transposes_the_lowering_aborts_on_are_refused(
    fn: object, shapes: tuple
) -> None:
    """Exact regression: an assert in upstream's lane distribution killed the lowering."""
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64], xegpu_layouts=["ir"]),
    )
    args = [xpu_tensor(*shape, dtype=torch.bfloat16) for shape in shapes]
    with pytest.raises(UnsupportedOperationError, match="vector.transpose"):
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")


def _computed_then_transposed(a, b, o):
    for tm, tn in hl.tile([a.size(1), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(0)):
            acc = torch.addmm(
                acc, (a[tk, tm].to(torch.float32) + 1.0).to(a.dtype).T, b[tk, tn]
            )
        o[tm, tn] = acc.to(a.dtype)
    return o


def _b_computed_then_transposed(a, b, o):
    for tm, tn in hl.tile([a.size(0), b.size(0)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(
                acc, a[tm, tk], (b[tn, tk].to(torch.float32) + 1.0).to(b.dtype).T
            )
        o[tm, tn] = acc.to(a.dtype)
    return o


@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (_computed_then_transposed, ((256, 512), (256, 512), (512, 512))),
        (_b_computed_then_transposed, ((512, 256), (512, 256), (512, 512))),
    ],
    ids=["a", "b"],
)
def test_computed_operand_then_transposed_is_refused(fn: object, shapes: tuple) -> None:
    """Exact regression: an assert in upstream's ``computeDistributedCoordsForMatrixOp``
    killed the lowering; the uncomputed ``a[tk, tm].T`` lays out."""
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[256, 256, 32], xegpu_layouts=["ir"]),
    )
    args = [xpu_tensor(*shape, dtype=torch.bfloat16) for shape in shapes]
    with pytest.raises(UnsupportedOperationError, match="vector.transpose"):
        lower_mlir(kernel, args, target="xpu", stop_at="xevm")
