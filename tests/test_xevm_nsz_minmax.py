"""``xevm_nsz_minmax``: minima and maxima of XeVM GPU modules get ``nsz``;
everything else is left alone. IR only: no XPU or SPIR-V target needed."""

from __future__ import annotations

from mlir import ir

from helion_mlir_backend._compiler.mlir_transforms.xevm_nsz_minmax import nsz_minmax


def _module(body: str, target: str = '[#xevm.target<chip = "bmg">]') -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    llvm.func spir_kernelcc @k(%a: f32, %b: f32, %v: vector<1xf32>) {{
      {body}
      llvm.return
    }}
  }}
  llvm.func @host(%a: f32, %b: f32) -> f32 {{
    %h = llvm.intr.maximum(%a, %b) : (f32, f32) -> f32
    llvm.return %h : f32
  }}
}}"""


def _rewrite(text: str) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = nsz_minmax(module)
        assert module.operation.verify()
        return count, str(module)


def _parsed(text: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(text))


def test_device_minima_and_maxima_get_nsz() -> None:
    count, text = _rewrite(
        _module(
            "%0 = llvm.intr.maximum(%a, %b) : (f32, f32) -> f32\n"
            "%1 = llvm.intr.minimum(%a, %b) : (f32, f32) -> f32\n"
            "%2 = llvm.intr.maximum(%v, %v) : "
            "(vector<1xf32>, vector<1xf32>) -> vector<1xf32>"
        )
    )
    assert count == 3
    assert text.count("<nsz>") == 3
    # The host's keeps its signed-zero order.
    host = text[text.index("llvm.func @host") :]
    assert "fastmath" not in host and "llvm.intr.maximum" in host


def test_others_are_left_alone() -> None:
    body = (
        "%0 = llvm.intr.maxnum(%a, %b) : (f32, f32) -> f32\n"
        "%1 = llvm.intr.minnum(%a, %b) : (f32, f32) -> f32\n"
        "%2 = llvm.fadd %a, %b : f32\n"
        "%3 = llvm.intr.maximum(%a, %b) fastmath<nnan> : "
        "(f32, f32) -> f32"
    )
    text = _module(body)
    assert _rewrite(text) == (0, _parsed(text))


def test_modules_without_an_xevm_target_are_left_alone() -> None:
    text = _module("%0 = llvm.intr.maximum(%a, %b) : (f32, f32) -> f32", target="")
    assert _rewrite(text) == (0, _parsed(text))
