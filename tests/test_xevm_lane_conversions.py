"""``xevm_lane_conversions``: in XeVM GPU modules, a narrowing vector conversion
of a vector built lane by lane becomes a conversion per lane, next to the lane's
value (an IGC vectorizer workaround); everything else is left alone. IR only: no
XPU or SPIR-V target needed. The device regressions are
``tests/test_xpu_runtime.py``'s ``test_math_every_pattern`` and
``tests/test_xpu_igc_workarounds.py``."""

from __future__ import annotations

import helion
from mlir import ir
import pytest
import torch

from tests.harness import lower_xpu
from tests.xpu_math_full_kernels import cast_f64_sin
from tests.xpu_math_full_kernels import cast_sin

from helion_mlir_backend._compiler.mlir_transforms.xevm_lane_conversions import (
    convert_lanes_where_computed,
)

_XEVM = '[#xevm.target<chip = "bmg">]'
_SIN = "_Z15__spirv_ocl_sinf"
_NATIVE = "_Z22__spirv_ocl_native_expf"
_OTHER_NATIVE = "_Z22__spirv_ocl_native_sinf"


def _module(body: str, target: str = _XEVM, args: str = "%a: f32") -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_funccc @{_SIN}(f32) -> f32
    llvm.func spir_funccc @{_NATIVE}(f32) -> f32
    llvm.func spir_funccc @{_OTHER_NATIVE}(f32) -> f32
    llvm.func spir_funccc @_Z15__spirv_ocl_sind(f64) -> f64
    llvm.func spir_kernelcc @k({args}, %p: !llvm.ptr<1>) {{
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c2 = llvm.mlir.constant(2 : i64) : i64
      %c3 = llvm.mlir.constant(3 : i64) : i64
      %u = llvm.mlir.poison : vector<2xf32>
      {body}
      llvm.return
    }}
  }}
  func.func @host(%h: vector<2xf32>, %q: !llvm.ptr) {{
    %t = llvm.fptrunc %h : vector<2xf32> to vector<2xbf16>
    llvm.store %t, %q : vector<2xbf16>, !llvm.ptr
    return
  }}
}}"""


def _call(name: str, operand: str = "%a") -> str:
    return f"{name} = llvm.call spir_funccc @{_SIN}({operand}) : (f32) -> f32"


def _chain(target: str = "bf16") -> str:
    return f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2x{target}>
      llvm.store %t, %p : vector<2x{target}>, !llvm.ptr<1>"""


def _rewrite(text: str) -> tuple[int, str]:
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(text)
        count = convert_lanes_where_computed(module)
        assert module.operation.verify()
        return count, str(module)


def _kernel(text: str) -> list[str]:
    body = text[text.index("spir_kernelcc") : text.index("func.func @host")]
    return [line.strip() for line in body.splitlines()]


def _ops(lines: list[str], op: str) -> list[int]:
    return [i for i, line in enumerate(lines) if f" {op} " in line]


@pytest.mark.parametrize("target", ["bf16", "f16"])
def test_each_lane_converts_next_to_its_value(target: str) -> None:
    count, text = _rewrite(_module(_chain(target)))
    assert count == 1
    lines = _kernel(text)
    calls, conversions = _ops(lines, "llvm.call"), _ops(lines, "llvm.fptrunc")
    assert [i + 1 for i in calls] == conversions
    assert all(f"f32 to {target}" in lines[i] for i in conversions)
    # The f32 chain is gone; a chain of the converted lanes reaches the store.
    assert "vector<2xf32>" not in "".join(
        line for line in lines if "llvm.insertelement" in line
    )
    inserts = _ops(lines, "llvm.insertelement")
    assert len(inserts) == 2 and inserts[0] > conversions[-1]
    store = next(line for line in lines if line.startswith("llvm.store"))
    assert f"vector<2x{target}>" in store
    # The host's conversion stays.
    assert "vector<2xf32> to vector<2xbf16>" in text


def test_lanes_written_out_of_order_keep_their_positions() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c1 : i64] : vector<2xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c0 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    lines = _kernel(text)
    conversions = _ops(lines, "llvm.fptrunc")
    results = [lines[i].split(" = ")[0] for i in conversions]
    inserts = [lines[i] for i in _ops(lines, "llvm.insertelement")]
    # The call first in the IR (position 1 in the source) is the second lane.
    assert f"insertelement {results[1]}," in inserts[0]
    assert f"insertelement {results[0]}," in inserts[1]


def test_a_later_insert_into_a_lane_wins() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      {_call("%s2")}
      %v2 = llvm.insertelement %s2, %v1[%c0 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v2 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    lines = _kernel(text)
    # Lane 0 is the third call's, and the first call's result is not converted.
    calls = _ops(lines, "llvm.call")
    assert len(calls) == 3 and len(_ops(lines, "llvm.fptrunc")) == 2
    assert " llvm.fptrunc " in lines[calls[1] + 1]
    assert " llvm.fptrunc " in lines[calls[2] + 1]
    assert " llvm.fptrunc " not in lines[calls[0] + 1]


def test_one_value_in_many_lanes_converts_once() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert len(_ops(_kernel(text), "llvm.fptrunc")) == 1


def test_a_chain_with_another_use_stays_for_that_use() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>
      llvm.store %v1, %p : vector<2xf32>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    lines = _kernel(text)
    assert len(_ops(lines, "llvm.fptrunc")) == 2
    assert (
        sum("vector<2xf32>" in line and "insertelement" in line for line in lines) == 2
    )


def test_slices_of_a_wider_vector_convert_each_lane_once() -> None:
    body = f"""%w = llvm.mlir.poison : vector<4xf32>
      {_call("%s0")}
      %v0 = llvm.insertelement %s0, %w[%c0 : i64] : vector<4xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<4xf32>
      {_call("%s2")}
      %v2 = llvm.insertelement %s2, %v1[%c2 : i64] : vector<4xf32>
      {_call("%s3")}
      %v3 = llvm.insertelement %s3, %v2[%c3 : i64] : vector<4xf32>
      %l = llvm.shufflevector %v3, %v3 [0, 1] : vector<4xf32>
      %t0 = llvm.fptrunc %l : vector<2xf32> to vector<2xbf16>
      llvm.store %t0, %p : vector<2xbf16>, !llvm.ptr<1>
      %h = llvm.shufflevector %v3, %v3 [2, 3] : vector<4xf32>
      %t1 = llvm.fptrunc %h : vector<2xf32> to vector<2xbf16>
      llvm.store %t1, %p : vector<2xbf16>, !llvm.ptr<1>
      %again = llvm.fptrunc %h : vector<2xf32> to vector<2xbf16>
      llvm.store %again, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 3
    lines = _kernel(text)
    calls, conversions = _ops(lines, "llvm.call"), _ops(lines, "llvm.fptrunc")
    assert [i + 1 for i in calls] == conversions
    assert not _ops(lines, "llvm.shufflevector")
    assert not any(
        "vector<4xf32>" in line and "insertelement" in line for line in lines
    )


def test_a_shuffle_reaches_the_second_operand() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%c1 : i64] : vector<2xf32>
      {_call("%s1")}
      %w0 = llvm.insertelement %s1, %u[%c0 : i64] : vector<2xf32>
      %w1 = llvm.insertelement %s1, %w0[%c1 : i64] : vector<2xf32>
      %l = llvm.shufflevector %v1, %w1 [0, 2] : vector<2xf32>
      %t = llvm.fptrunc %l : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    lines = _kernel(text)
    calls, conversions = _ops(lines, "llvm.call"), _ops(lines, "llvm.fptrunc")
    assert [i + 1 for i in calls] == conversions


def test_lanes_of_a_native_builtin_stay() -> None:
    """IGC inlines them without branches, and their vectorized conversion is right
    and cheaper (attention): the rewrite would keep them scalar."""
    native = f"llvm.call spir_funccc @{_NATIVE}(%a) : (f32) -> f32"
    body = f"""%s0 = {native}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %s1 = {native}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, _ = _rewrite(_module(body))
    assert count == 0


def test_lanes_of_another_native_builtin_convert() -> None:
    """Regression: the exemption was a prefix rule, measured only on ``native_exp``.
    A ``native_`` builtin whose inlined form has branches takes the miscompile."""
    native = f"llvm.call spir_funccc @{_OTHER_NATIVE}(%a) : (f32) -> f32"
    body = f"""%s0 = {native}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %s1 = {native}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, _ = _rewrite(_module(body))
    assert count == 1


def test_a_native_lane_next_to_a_branchy_one_converts() -> None:
    native = f"llvm.call spir_funccc @{_NATIVE}(%a) : (f32) -> f32"
    body = f"""%s0 = {native}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      {_call("%s1")}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, _ = _rewrite(_module(body))
    assert count == 1


def test_fast_math_flags_carry_over() -> None:
    body = f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 fastmath<nnan> : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    assert any(
        "fastmath<nnan>" in _kernel(text)[i]
        for i in _ops(_kernel(text), "llvm.fptrunc")
    )


def test_lane_values_from_other_blocks_and_arguments() -> None:
    body = f"""{_call("%s0")}
      llvm.br ^next
    ^next:
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %a, %v0[%c1 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 1
    lines = _kernel(text)
    conversions = _ops(lines, "llvm.fptrunc")
    calls = _ops(lines, "llvm.call")
    branch = next(i for i, line in enumerate(lines) if line.startswith("llvm.br"))
    # The argument's conversion is first in its block, before the call; the call's
    # stays in the call's block.
    assert conversions[0] < calls[0] < conversions[1] < branch


def test_an_argument_lane_converted_in_sibling_blocks_dominates_both() -> None:
    """Regression: the first of two conversions of one argument lane held the scalar
    conversion in its block, so the other one, in a sibling block, used a value it
    is not dominated by: IR that fails verification (``_rewrite`` checks it)."""
    body = """%v0 = llvm.insertelement %a, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %a, %v0[%c1 : i64] : vector<2xf32>
      llvm.cond_br %c, ^x, ^y
    ^x:
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>
      llvm.return
    ^y:
      %s = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %s, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, args="%a: f32, %c: i1"))
    assert count == 2
    lines = _kernel(text)
    conversions = _ops(lines, "llvm.fptrunc")
    assert len(conversions) == 1
    assert conversions[0] < next(
        i for i, line in enumerate(lines) if line.startswith("llvm.cond_br")
    )


def _f64_lanes(target: str) -> str:
    f64 = "llvm.call spir_funccc @_Z15__spirv_ocl_sind(%d) : (f64) -> f64"
    return f"""%w = llvm.mlir.poison : vector<2xf64>
      %d = llvm.fpext %a : f32 to f64
      %s0 = {f64}
      %v0 = llvm.insertelement %s0, %w[%c0 : i64] : vector<2xf64>
      %s1 = {f64}
      %v1 = llvm.insertelement %s1, %v0[%c1 : i64] : vector<2xf64>
      %t = llvm.fptrunc %v1 : vector<2xf64> to vector<2x{target}>
      llvm.store %t, %p : vector<2x{target}>, !llvm.ptr<1>"""


@pytest.mark.parametrize("target", ["f32", "bf16", "f16"])
def test_f64_lanes_convert_next_to_their_values(target: str) -> None:
    """``fptrunc <8 x double>`` takes the same wrong order as f32's."""
    count, text = _rewrite(_module(_f64_lanes(target)))
    assert count == 1
    lines = _kernel(text)
    calls, conversions = _ops(lines, "llvm.call"), _ops(lines, "llvm.fptrunc")
    assert [i + 1 for i in calls] == conversions
    assert all(f"f64 to {target}" in lines[i] for i in conversions)


def test_f64_narrowed_twice_takes_two_rounds() -> None:
    """Regression: widening the match alone left ``fptrunc <n x f32>`` of the f64
    conversions' vector, and the kernel's output stayed wrong."""
    body = f"""{_f64_lanes("f32")}
      %b = llvm.fptrunc %t : vector<2xf32> to vector<2xbf16>
      llvm.store %b, %p : vector<2xbf16>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body))
    assert count == 2
    lines = _kernel(text)
    assert not [line for line in lines if "fptrunc" in line and "vector<" in line]
    calls, conversions = _ops(lines, "llvm.call"), _ops(lines, "llvm.fptrunc")
    # Each lane: the call, to f32, to bf16.
    assert conversions == sorted(
        position for call in calls for position in (call + 1, call + 2)
    )


@pytest.mark.parametrize(
    ("body", "args"),
    [
        # A lane not written.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v0 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # A lane over a vector the chain does not build.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %b[%c0 : i64] : vector<2xf32>
      %t = llvm.fptrunc %v0 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32, %b: vector<2xf32>",
        ),
        # A position that is not a constant.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%i : i64] : vector<2xf32>
      %t = llvm.fptrunc %v1 : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32, %i: i64",
        ),
        # A slice with a lane no insert writes.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %l = llvm.shufflevector %v0, %v0 [0, 1] : vector<2xf32>
      %t = llvm.fptrunc %l : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # A poison lane in a slice.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%c1 : i64] : vector<2xf32>
      %l = llvm.shufflevector %v1, %v1 [0, -1] : vector<2xf32>
      %t = llvm.fptrunc %l : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # A vector that is not built by lanes.
        (
            """%t = llvm.fptrunc %b : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32, %b: vector<2xf32>",
        ),
        # Arithmetic between the chain and the conversion.
        (
            f"""{_call("%s0")}
      %v0 = llvm.insertelement %s0, %u[%c0 : i64] : vector<2xf32>
      %v1 = llvm.insertelement %s0, %v0[%c1 : i64] : vector<2xf32>
      %m = llvm.fmul %v1, %v1 : vector<2xf32>
      %t = llvm.fptrunc %m : vector<2xf32> to vector<2xbf16>
      llvm.store %t, %p : vector<2xbf16>, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # A scalar conversion is a lane's already.
        (
            f"""{_call("%s0")}
      %t = llvm.fptrunc %s0 : f32 to bf16
      llvm.store %t, %p : bf16, !llvm.ptr<1>""",
            "%a: f32",
        ),
        # An f64 vector not built by lanes.
        (
            """%t = llvm.fptrunc %w : vector<2xf64> to vector<2xf32>
      llvm.store %t, %p : vector<2xf32>, !llvm.ptr<1>""",
            "%a: f32, %w: vector<2xf64>",
        ),
        # Not a truncation.
        (
            """%v0 = llvm.insertelement %h, %w[%c0 : i64] : vector<2xbf16>
      %v1 = llvm.insertelement %h, %v0[%c1 : i64] : vector<2xbf16>
      %t = llvm.fpext %v1 : vector<2xbf16> to vector<2xf32>
      llvm.store %t, %p : vector<2xf32>, !llvm.ptr<1>""",
            "%a: f32, %h: bf16, %w: vector<2xbf16>",
        ),
    ],
    ids=[
        "partial",
        "over_another_vector",
        "computed_position",
        "slice_unwritten_lane",
        "slice_poison_lane",
        "not_built_by_lanes",
        "arithmetic_between",
        "scalar",
        "f64_not_built_by_lanes",
        "fpext",
    ],
)
def test_other_ir_is_untouched(body: str, args: str) -> None:
    text = _module(body, args=args)
    count, rewritten = _rewrite(text)
    assert count == 0
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        assert str(ir.Module.parse(text)) == rewritten


def test_non_xevm_gpu_modules_are_untouched() -> None:
    text = _module(_chain(), target="")
    count, rewritten = _rewrite(text)
    assert count == 0
    assert "vector<2xf32> to vector<2xbf16>" in rewritten


@pytest.mark.isolated
def test_builtin_kernel_is_rewritten() -> None:
    """The builtin's lanes, as the pipeline emits them: every 8 calls of ``sin``
    convert one by one (bf16 out); the f32 output keeps no conversion at all."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[1, 8192]),
    )(cast_sin)
    x = torch.zeros(16, 8192, dtype=torch.bfloat16)
    out = torch.zeros(16, 8192, dtype=torch.bfloat16)
    text = lower_xpu(kernel, [x, out], stop_at="xevm")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        assert convert_lanes_where_computed(module) > 0
        text = str(module)
    assert "vector<8xf32> to vector<8xbf16>" not in text
    assert text.count("f32 to bf16") == text.count("__spirv_ocl_sinf(") - 1


_F64_DTYPES = [
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
]


@pytest.mark.isolated
@pytest.mark.parametrize("dtypes", _F64_DTYPES, ids=str)
@pytest.mark.parametrize(
    ("shape", "blocks"),
    [((16, 8192), [1, 8192]), ((256, 512), [128, 256])],
    ids=["1d", "2d"],
)
def test_f64_kernel_keeps_no_vector_conversion(
    shape: tuple[int, int],
    blocks: list[int],
    dtypes: tuple[torch.dtype, torch.dtype],
) -> None:
    """Regression, the kernels the review found wrong on the device (up to 130037 of
    131072 elements off by over 1 %): none of their conversions stays a vector one."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks),
    )(cast_f64_sin)
    args = [torch.zeros(shape, dtype=dtypes[0]), torch.zeros(shape, dtype=dtypes[1])]
    text = lower_xpu(kernel, args, stop_at="xevm")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        assert convert_lanes_where_computed(module) > 0
        assert module.operation.verify()
        text = str(module)
    wrong = [
        line
        for line in text.splitlines()
        if "llvm.fptrunc" in line and ": vector<" in line
    ]
    assert wrong == []
