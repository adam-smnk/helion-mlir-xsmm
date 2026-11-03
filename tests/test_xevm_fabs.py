"""``xevm_fabs``: f32 ``llvm.intr.fabs`` in XeVM GPU modules becomes integer ops
(a clear of the sign bit IGC cannot turn back into ``fabs``); everything else
is left alone. IR only. The device regressions are in
``tests/test_xpu_runtime.py`` (``test_math_functions``,
``test_elementwise_every_bf16_pattern``)."""

from __future__ import annotations

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir_transforms.xevm_fabs import fabs_as_integer_min

_XEVM = '[#xevm.target<chip = "bmg">]'


def _module(body: str, args: str, target: str = _XEVM) -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_kernelcc @k({args}, %p: !llvm.ptr<1>) {{
      {body}
      llvm.return
    }}
  }}
  func.func @host(%h: f32, %q: !llvm.ptr) {{
    %a = llvm.intr.fabs(%h) : (f32) -> f32
    llvm.store %a, %q : f32, !llvm.ptr
    return
  }}
}}"""


def _rewrite(text: str, *, apply: bool = True) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = fabs_as_integer_min(module) if apply else 0
        assert module.operation.verify()
        return count, str(module)


def _kernel(text: str) -> str:
    return text[text.index("gpu.module") : text.index("func.func @host")]


@pytest.mark.parametrize(
    ("value", "bits"),
    [("f32", "i32"), ("vector<8xf32>", "vector<8xi32>")],
    ids=["scalar", "vector"],
)
def test_f32_fabs_becomes_an_integer_minimum(value: str, bits: str) -> None:
    body = f"""%r = llvm.intr.fabs(%a) : ({value}) -> {value}
      llvm.store %r, %p : {value}, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, f"%a: {value}"))
    assert count == 1
    kernel = _kernel(text)
    assert "llvm.intr.fabs" not in kernel
    assert kernel.count("llvm.intr.umin") == 1
    assert kernel.count("llvm.xor") == 1
    assert kernel.count("llvm.bitcast") == 2
    assert f"-> {bits}" in kernel or f": {bits}" in kernel
    # The sign bit: the host's fabs stays.
    assert "-2147483648" in kernel
    assert "llvm.intr.fabs" in text


@pytest.mark.parametrize(
    ("body", "args"),
    [
        (
            """%r = llvm.intr.fabs(%a) : (f16) -> f16
      llvm.store %r, %p : f16, !llvm.ptr<1>""",
            "%a: f16",
        ),
        (
            """%r = llvm.fneg %a : f32
      llvm.store %r, %p : f32, !llvm.ptr<1>""",
            "%a: f32",
        ),
    ],
    ids=["f16", "other-op"],
)
def test_other_ops_are_untouched(body: str, args: str) -> None:
    source = _module(body, args)
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


def test_f64_fabs_becomes_an_integer_minimum() -> None:
    """IGC's ``llvm.fabs`` on a vector of f64 is no better than on f32."""
    body = """%r = llvm.intr.fabs(%a) : (vector<4xf64>) -> vector<4xf64>
      llvm.store %r, %p : vector<4xf64>, !llvm.ptr<1>"""
    count, text = _rewrite(_module(body, "%a: vector<4xf64>"))
    assert count == 1
    kernel = _kernel(text)
    assert "llvm.intr.fabs" not in kernel
    assert "llvm.intr.umin" in kernel
    assert "-9223372036854775808" in kernel


def test_non_xevm_gpu_modules_are_untouched() -> None:
    body = """%r = llvm.intr.fabs(%a) : (f32) -> f32
      llvm.store %r, %p : f32, !llvm.ptr<1>"""
    source = _module(body, "%a: f32", target="")
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]
