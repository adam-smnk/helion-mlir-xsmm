"""``compare_lanes``: a vector compare in an XeVM module becomes one scalar compare per lane, and
what it leaves alone. On the device, with what it works around: ``test_xpu_scalar_compares``."""

from __future__ import annotations

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir_transforms.xevm_lane_compares import (
    compare_lanes,
)
from helion_mlir_backend._compiler.targets.xpu import _narrows_f64

_XEVM = "[#xevm.target<O = 3>]"


def _module(body: str, args: str, target: str = _XEVM) -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_kernelcc @k({args}, %p: !llvm.ptr<1>) {{
      {body}
      llvm.return
    }}
  }}
  func.func @host(%h: vector<16xf32>, %q: !llvm.ptr) {{
    %c = llvm.fcmp "ogt" %h, %h : vector<16xf32>
    llvm.store %c, %q : vector<16xi1>, !llvm.ptr
    return
  }}
}}"""


def _rewrite(text: str) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = compare_lanes(module)
        assert module.operation.verify()
        return count, str(module)


def _kernel(text: str) -> str:
    return text[text.index("gpu.module") : text.index("func.func @host")]


@pytest.mark.parametrize(
    ("op", "predicate", "element", "lanes"),
    [
        ("llvm.fcmp", "ogt", "f32", 16),
        ("llvm.fcmp", "olt", "f32", 8),
        ("llvm.fcmp", "une", "f64", 16),
        ("llvm.icmp", "slt", "i32", 16),
        ("llvm.icmp", "eq", "i64", 4),
    ],
)
def test_vector_compare_goes_per_lane(
    op: str, predicate: str, element: str, lanes: int
) -> None:
    vector = f"vector<{lanes}x{element}>"
    body = f"""%r = {op} "{predicate}" %a, %b : {vector}
      llvm.store %r, %p : vector<{lanes}xi1>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, f"%a: {vector}, %b: {vector}"))
    assert count == 1
    kernel = _kernel(text)
    assert f": {vector}\n" not in kernel.split("llvm.store")[0].split(op)[-1]
    assert kernel.count(f'{op} "{predicate}"') == lanes
    assert kernel.count(f": {element}\n") >= lanes
    assert kernel.count("llvm.insertelement") == lanes
    assert kernel.count("llvm.extractelement") == 2 * lanes


def test_one_operand_constant_keeps_its_lanes() -> None:
    body = """%c = llvm.mlir.constant(dense<1.0> : vector<16xf32>) : vector<16xf32>
      %r = llvm.fcmp "ogt" %a, %c : vector<16xf32>
      llvm.store %r, %p : vector<16xi1>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, "%a: vector<16xf32>"))
    assert count == 1
    assert _kernel(text).count('llvm.fcmp "ogt"') == 16


@pytest.mark.parametrize(
    ("body", "args"),
    [
        (
            """%r = llvm.fcmp "ogt" %a, %a : f32
      llvm.store %r, %p : i1, !llvm.ptr<1>""",
            "%a: f32",
        ),
        (
            """%r = llvm.icmp "slt" %a, %a : i32
      llvm.store %r, %p : i1, !llvm.ptr<1>""",
            "%a: i32",
        ),
    ],
    ids=["scalar-fcmp", "scalar-icmp"],
)
def test_scalar_compares_are_left_alone(body: str, args: str) -> None:
    with ir.Context(), ir.Location.unknown():
        before = str(ir.Module.parse(_module(body, args)))
    count, text = _rewrite(_module(body, args))
    assert count == 0
    assert text == before


def test_other_gpu_modules_and_host_code_are_left_alone() -> None:
    body = """%r = llvm.fcmp "ogt" %a, %a : vector<16xf32>
      llvm.store %r, %p : vector<16xi1>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, "%a: vector<16xf32>", target=""))
    assert count == 0
    assert text.count("llvm.fcmp") == 2


@pytest.mark.parametrize(
    ("body", "args", "narrows"),
    [
        ("%r = llvm.fptrunc %a : f64 to f32", "%a: f64", True),
        (
            "%r = llvm.fptrunc %a : vector<8xf64> to vector<8xf32>",
            "%a: vector<8xf64>",
            True,
        ),
        ("%r = llvm.fptrunc %a : f32 to bf16", "%a: f32", False),
        (
            "%r = llvm.fptrunc %a : vector<8xf32> to vector<8xbf16>",
            "%a: vector<8xf32>",
            False,
        ),
        ("%r = llvm.fpext %a : f32 to f64", "%a: f32", False),
    ],
    ids=["scalar-f64", "vector-f64", "scalar-f32", "vector-f32", "widening"],
)
def test_narrowing_f64_turns_the_vectorizer_off(
    body: str, args: str, narrows: bool
) -> None:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_module(body, args))
        assert _narrows_f64(module.operation) is narrows
