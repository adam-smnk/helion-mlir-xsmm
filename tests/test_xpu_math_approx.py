"""``xpu_math_approx``: f32 ``tanh`` and ``log1p`` in XeVM GPU modules become
branch-free ops, no builtin call per element; everything else is left alone.
IR only. The device regressions (every bf16 pattern, f32 special values) are in
``tests/test_xpu_runtime.py``."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests import kernelbench_xpu as kb
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend._compiler.mlir_transforms.xpu_math_approx import (
    approximate_math,
)
from helion_mlir_backend._compiler.mlir_transforms.xpu_math_approx import (
    fold_extended_splats,
)

_XEVM = '[#xevm.target<chip = "bmg">]'


def _module(function: str, value: str, target: str = _XEVM) -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    gpu.func @k(%a: {value}) kernel {{
      %r = {function} %a : {value}
      gpu.return
    }}
  }}
  func.func @host(%h: f32) -> f32 {{
    %r = math.tanh %h : f32
    return %r : f32
  }}
}}"""


def _rewrite(text: str, *, apply: bool = True) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = approximate_math(module) if apply else 0
        assert module.operation.verify()
        return count, str(module)


def _kernel(text: str) -> str:
    return text[text.index("gpu.module") : text.index("func.func @host")]


@pytest.mark.parametrize("value", ["f32", "vector<8192xf32>"], ids=["scalar", "vector"])
def test_tanh(value: str) -> None:
    count, text = _rewrite(_module("math.tanh", value))
    assert count == 1
    kernel = _kernel(text)
    assert "math.tanh" not in kernel
    assert kernel.count("math.fma") == 9
    assert kernel.count("arith.divf") == 1
    assert kernel.count("arith.select") == 3
    assert kernel.count("math.absf") == 1
    assert "scf." not in kernel and "cf." not in kernel
    assert "math.tanh" in text  # the host's


@pytest.mark.parametrize("value", ["f32", "vector<8192xf32>"], ids=["scalar", "vector"])
def test_log1p(value: str) -> None:
    count, text = _rewrite(_module("math.log1p", value))
    assert count == 1
    kernel = _kernel(text)
    assert "math.log1p" not in kernel
    # Polynomial: OpenCL's log runs at half the speed, the native one (afn) is
    # 76 ulps off near 1.
    assert "math.log " not in kernel
    assert "fastmath" not in kernel
    assert kernel.count("arith.bitcast") == 2
    assert kernel.count("math.fma") == 10
    assert kernel.count("arith.divf") == 1


@pytest.mark.parametrize(
    ("function", "value"),
    [
        ("math.tanh", "f16"),
        ("math.tanh", "f64"),
        ("math.log1p", "vector<8xf64>"),
        ("math.exp", "f32"),
        ("math.log", "vector<8xf32>"),
    ],
    ids=["tanh-f16", "tanh-f64", "log1p-f64", "exp", "log"],
)
def test_other_ops_are_untouched(function: str, value: str) -> None:
    source = _module(function, value)
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


def test_non_xevm_gpu_modules_are_untouched() -> None:
    source = _module("math.tanh", "f32", target="")
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("22_Tanh", "__spirv_ocl_tanh"),
        ("29_Softplus", "__spirv_ocl_log"),
    ],
)
def test_kernelbench_problems_call_no_builtin_per_element(
    name: str, builtin: str
) -> None:
    """The exact problems at their elementwise tiles: no ``tanh`` or ``log1p``
    calls left at ``xevm``; 29_Softplus keeps its native ``exp``."""
    problem = {p.name: p for p in kb.PROBLEMS}[name]
    x = xpu_tensor(16, 16384, dtype=torch.bfloat16)
    lowered = lower_xpu(problem.kernel, [x], "xevm")
    assert builtin not in lowered
    if name == "22_Tanh":
        assert "llvm.intr.fma" in lowered
    else:
        assert "__spirv_ocl_native_exp" in lowered


def _extended(constant: str, source: str, wide: str, target: str = _XEVM) -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    gpu.func @k() kernel {{
      %c = arith.constant {constant} : {source}
      %e = arith.extf %c : {source} to {wide}
      %r = math.exp %e : {wide}
      gpu.return
    }}
  }}
}}"""


def _fold(text: str) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = fold_extended_splats(module)
        assert module.operation.verify()
        return count, str(module)


@pytest.mark.parametrize("source", ["bf16", "f16"])
def test_extended_splats_fold(source: str) -> None:
    count, text = _fold(_extended("dense<2.5>", f"vector<8x{source}>", "vector<8xf32>"))
    assert count == 1
    assert "arith.extf" not in text
    assert "dense<2.500000e+00> : vector<8xf32>" in text


@pytest.mark.parametrize(
    ("constant", "source", "wide", "target"),
    [
        # Not a splat.
        ("dense<[1.0, 2.0]>", "vector<2xbf16>", "vector<2xf32>", _XEVM),
        # Another module, f32 to f64, a scalar.
        ("dense<2.5>", "vector<8xbf16>", "vector<8xf32>", ""),
        ("dense<2.5>", "vector<8xf32>", "vector<8xf64>", _XEVM),
        ("dense<2.5>", "vector<8xbf16>", "vector<8xf64>", _XEVM),
        ("dense<2.5>", "vector<8xf16>", "vector<8xf64>", _XEVM),
        ("2.5", "bf16", "f32", _XEVM),
    ],
    ids=["non-splat", "non-xevm", "f32-to-f64", "bf16-to-f64", "f16-to-f64", "scalar"],
)
def test_other_extensions_stay(
    constant: str, source: str, wide: str, target: str
) -> None:
    count, text = _fold(_extended(constant, source, wide, target))
    assert count == 0
    assert "arith.extf" in text


@pytest.mark.isolated
@pytest.mark.parametrize("expression", ["pow", "exp2"])
def test_bf16_powf_with_a_constant_lowers(expression: str) -> None:
    """``torch.pow(x, 2.5)`` and ``torch.exp2`` on bf16, no cast: on 8c12643 an
    invalid ``math.powf`` in XeGPU's distribution, reported as an internal error."""
    fn = _pow if expression == "pow" else _exp2
    kernel = helion.kernel(
        backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[128, 256])
    )(fn)
    args = [xpu_tensor(256, 512, dtype=torch.bfloat16)] * 2
    assert "__spirv_ocl_pow" in lower_xpu(kernel, args, "xevm")


def _pow(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.pow(x[tile], 2.5)
    return out


def _exp2(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.exp2(x[tile])
    return out
