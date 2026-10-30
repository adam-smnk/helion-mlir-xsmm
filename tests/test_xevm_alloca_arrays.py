"""``xevm_alloca_arrays``: constant-count allocas of XeVM GPU modules become
array allocas addressed by a GEP (SPIR-V backend workaround); everything else
is left alone. IR only: no XPU or SPIR-V target needed."""

from __future__ import annotations

from mlir import ir
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor
from tests.kernelbench_xpu import PROBLEMS

from helion_mlir_backend._compiler.mlir_transforms.xevm_alloca_arrays import (
    alloca_arrays,
)

_CALLEE = "llvm.func spir_funccc @_Z4fillPj(!llvm.ptr)"


def _module(body: str, target: str = '[#xevm.target<chip = "bmg">]') -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    {_CALLEE}
    llvm.func spir_kernelcc @k(%n: i32) {{
      {body}
      llvm.return
    }}
  }}
  func.func @host() {{
    %c = llvm.mlir.constant(8 : i32) : i32
    %h = llvm.alloca %c x i32 : (i32) -> !llvm.ptr
    return
  }}
}}"""


def _rewrite(text: str, *, apply: bool = True) -> tuple[int, str]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        count = alloca_arrays(module) if apply else 0
        assert module.operation.verify()
        return count, str(module)


def test_constant_count_alloca_becomes_an_array() -> None:
    count, text = _rewrite(
        _module(
            """%c8 = llvm.mlir.constant(8 : i32) : i32
      %a = llvm.alloca %c8 x i32 {alignment = 32 : i64} : (i32) -> !llvm.ptr
      llvm.call spir_funccc @_Z4fillPj(%a) : (!llvm.ptr) -> ()
      %v = llvm.load %a : !llvm.ptr -> vector<8xi32>"""
        )
    )
    assert count == 1
    kernel = text[text.index("gpu.module") : text.index("func.func @host")]
    assert (
        "x !llvm.array<8 x i32> {alignment = 32 : i64} : (i32) -> !llvm.ptr" in kernel
    )
    assert "llvm.mlir.constant(1 : i32) : i32" in kernel
    assert "llvm.getelementptr %" in kernel
    assert "[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>" in kernel
    # The call and the load use the GEP's address, not a counted alloca.
    assert " x i32 " not in kernel.replace("x !llvm.array<8 x i32>", "")
    # Host code keeps its counted alloca.
    host = text[text.index("func.func @host") :]
    assert "x i32 : (i32) -> !llvm.ptr" in host and "getelementptr" not in host


@pytest.mark.parametrize(
    "body",
    [
        # A runtime count.
        "%a = llvm.alloca %n x i32 : (i32) -> !llvm.ptr",
        # One element.
        """%c1 = llvm.mlir.constant(1 : i32) : i32
      %a = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr""",
        # Already an array.
        """%c1 = llvm.mlir.constant(1 : i32) : i32
      %a = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr""",
        # Two arrays: an array element type is left alone.
        """%c2 = llvm.mlir.constant(2 : i32) : i32
      %a = llvm.alloca %c2 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr""",
        # An attribute the rewrite would drop.
        """%c8 = llvm.mlir.constant(8 : i32) : i32
      %a = llvm.alloca inalloca %c8 x i32 : (i32) -> !llvm.ptr""",
    ],
    ids=["dynamic", "one", "array", "array_count", "inalloca"],
)
def test_other_allocas_are_untouched(body: str) -> None:
    source = _module(body)
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]
    assert "llvm.getelementptr" not in text


def test_address_space_is_kept() -> None:
    count, text = _rewrite(
        _module(
            """%c8 = llvm.mlir.constant(8 : i32) : i32
      %a = llvm.alloca %c8 x i16 {alignment = 16 : i64} : (i32) -> !llvm.ptr<5>
      %v = llvm.load %a : !llvm.ptr<5> -> vector<8xi16>"""
        )
    )
    assert count == 1
    kernel = text[text.index("gpu.module") : text.index("func.func @host")]
    assert (
        "x !llvm.array<8 x i16> {alignment = 16 : i64} : (i32) -> !llvm.ptr<5>"
        in kernel
    )
    assert "[0, 0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, !llvm.array<8 x i16>" in kernel


def test_non_xevm_gpu_modules_are_untouched() -> None:
    body = """%c8 = llvm.mlir.constant(8 : i32) : i32
      %a = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr"""
    source = _module(body, target="")
    count, text = _rewrite(source)
    assert count == 0
    assert text == _rewrite(source, apply=False)[1]


@pytest.mark.isolated
def test_elementwise_block_read_buffers_are_rewritten() -> None:
    """19_ReLU at ``xevm``: the 2D block reads' destination allocas (in the
    kernel's entry block) are counted allocas; after the rewrite none is left
    and the module verifies."""
    problem = next(p for p in PROBLEMS if p.name == "19_ReLU")
    text = lower_xpu(
        problem.kernel, [xpu_tensor(256, 512, dtype=torch.bfloat16)], "xevm"
    )
    kernel = text[text.index("gpu.module") :]
    assert "intel_sub_group_2d_block_read" in kernel
    count, rewritten = _rewrite(text)
    assert count >= 1
    kernel = rewritten[rewritten.index("gpu.module") :]
    assert "!llvm.array<" in kernel
    assert not [
        line
        for line in kernel.splitlines()
        if "llvm.alloca" in line and "!llvm.array" not in line
    ]
