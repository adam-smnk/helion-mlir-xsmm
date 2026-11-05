"""``nsz_reductions`` and ``xevm_nsz_minmax``: maximum and minimum reductions get ``nsz``,
XeGPU's lowering drops it, the stage puts it back; everything else is left alone. IR only:
no XPU or SPIR-V target needed."""

from __future__ import annotations

import re

import helion
import helion.language as hl
from mlir import ir
import torch

from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir_transforms.nsz_reductions import nsz_reductions
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


def amax_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - torch.amax(
                x[tile_m, tile_n], dim=-1, keepdim=True
            )
    return out


def _amax_ir(stage: str) -> str:
    kernel = helion.kernel(
        amax_rows,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64]),
    )
    args = [xpu_tensor(256, 512, dtype=torch.float32)]
    return str(lower_mlir(kernel, args, target="xpu", stop_at=stage))


_MAXIMA = r"llvm\.intr\.maximum[^\n]*"


def test_xegpu_still_drops_the_reductions_flag() -> None:
    """This stage's removal condition, as a test: it fails when upstream keeps the flag
    set by ``nsz_reductions``. Then remove ``xevm-nsz-minmax`` and this test."""
    maxima = re.findall(_MAXIMA, _amax_ir("xevm"))
    assert maxima
    assert not [line for line in maxima if "nsz" in line]


def test_the_stage_puts_the_flag_on_every_row_maximum() -> None:
    maxima = re.findall(_MAXIMA, _amax_ir("xevm-final"))
    assert maxima
    assert all("<nsz>" in line for line in maxima)


def test_only_maximum_and_minimum_reductions_of_kernels_get_the_flag() -> None:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            """
module attributes {gpu.container_module} {
  gpu.module @k {
    gpu.func @f(%v: vector<64xf32>, %a: f32) kernel {
      %0 = vector.multi_reduction <maximumf>, %v, %a [0] : vector<64xf32> to f32
      %1 = vector.multi_reduction <minimumf>, %v, %a [0] : vector<64xf32> to f32
      %2 = vector.multi_reduction <add>, %v, %a [0] : vector<64xf32> to f32
      %3 = vector.multi_reduction <maximumf>, %v, %a [0] : vector<64xf32> to f32
      gpu.return
    }
  }
  func.func @host(%v: vector<64xf32>, %a: f32) -> f32 {
    %0 = vector.multi_reduction <maximumf>, %v, %a [0] : vector<64xf32> to f32
    return %0 : f32
  }
}"""
        )
        kernel_ops = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "vector.multi_reduction":
                kernel_ops.append(op)
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        # A reduction with its own flags keeps them.
        kernel_ops[3].attributes["fastmath"] = ir.Attribute.parse("#arith.fastmath<nnan>")
        assert nsz_reductions(module) == 2
        assert module.operation.verify()
        text = str(module)
    host = text[text.index("func.func @host") :]
    assert "nsz" not in host
    assert text.count("nsz") == 2
    assert "nnan" in text
