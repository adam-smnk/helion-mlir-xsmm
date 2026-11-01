"""``xevm_tile_order``: the side-effect-free ops of a block depth first, chunk by
chunk; everything else keeps its order. The XPU target's binary stage applies it
only to modules that spill, and keeps it only if it spills less. IR and binaries;
the device regression is ``test_xpu_runtime.test_mish_mish_binary_does_not_spill``."""

from __future__ import annotations

import itertools
import os
import shutil
from unittest.mock import patch

from mlir import ir
import pytest

from tests import kernelbench_xpu as kb
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend._compiler.mlir_transforms.xevm_tile_order import (
    order_tile_chunks,
)
from helion_mlir_backend._compiler.mlir_transforms.xevm_tile_order import spill_bytes
from helion_mlir_backend._compiler.targets import xpu as xpu_target
from helion_mlir_backend._compiler.targets.xpu import bindings_emit_spirv

BINARIES = pytest.mark.skipif(
    not bindings_emit_spirv() or shutil.which("ocloc") is None,
    reason="needs the SPIR-V target and ocloc",
)
_XEVM = '[#xevm.target<chip = "bmg">]'
_F32 = "vector<8xf32>"
_CHUNKS = 4


def _chunks(depth: int = 6) -> str:
    """``_CHUNKS`` loads and stores of a chain of ``depth`` multiplications, in
    vector-op order: all chunks of one op, then the next op."""
    lines = []
    for c in range(_CHUNKS):
        lines.append(f"%v0_{c} = llvm.load %p : !llvm.ptr<1> -> {_F32}")
    for d in range(1, depth + 1):
        for c in range(_CHUNKS):
            lines.append(f"%v{d}_{c} = llvm.fmul %v{d - 1}_{c}, %v0_{c} : {_F32}")
    for c in range(_CHUNKS):
        lines.append(f"llvm.store %v{depth}_{c}, %p : {_F32}, !llvm.ptr<1>")
    return "\n      ".join(lines)


def _module(body: str, target: str = _XEVM, declarations: str = "") -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @k {target} {{
    {declarations}
    llvm.func spir_kernelcc @k(%p: !llvm.ptr<1>, %v: {_F32}) {{
      {body}
      llvm.return
    }}
  }}
}}"""


def _rewrite(text: str) -> tuple[int, str, str]:
    """The number of blocks ordered, and the module's text before and after."""
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(text)
        before = str(module)
        count = order_tile_chunks(module)
        assert module.operation.verify()
        return count, before, str(module)


def _kernel_lines(text: str) -> list[str]:
    lines = text[text.index("spir_kernelcc") :].splitlines()[1:]
    return [line.strip() for line in lines]


def test_chunks_are_computed_one_at_a_time() -> None:
    count, _, text = _rewrite(_module(_chunks()))
    assert count == 1
    lines = _kernel_lines(text)
    assert sum("llvm.fmul" in line for line in lines) == _CHUNKS * 6
    # Loads and stores keep their order.
    memory = [
        "load" if "llvm.load" in line else "store"
        for line in lines
        if "llvm.load" in line or "llvm.store" in line
    ]
    assert memory == ["load"] * _CHUNKS + ["store"] * _CHUNKS
    # Each store right after its own chain.
    stores = [i for i, line in enumerate(lines) if "llvm.store" in line]
    assert [b - a for a, b in itertools.pairwise(stores)] == [7] * (_CHUNKS - 1)


def test_non_xevm_modules_stay() -> None:
    count, before, after = _rewrite(_module(_chunks(), target=""))
    assert count == 0
    assert after == before


def _with_call_before_stores(call: str, declaration: str) -> str:
    lines = _chunks().splitlines()
    index = next(i for i, line in enumerate(lines) if "llvm.store" in line)
    lines.insert(index, f"      {call}")
    return _module("\n".join(lines), declarations=declaration)


@pytest.mark.parametrize(
    ("call", "declaration"),
    [
        (
            "llvm.call @opaque(%p) : (!llvm.ptr<1>) -> ()",
            "llvm.func @opaque(!llvm.ptr<1>)",
        ),
        # OpenCL's, but it writes memory: no result, a pointer operand.
        (
            "llvm.call @_Z__spirv_ocl_vstore(%v, %p) : (vector<8xf32>, !llvm.ptr<1>) -> ()",
            "llvm.func @_Z__spirv_ocl_vstore(vector<8xf32>, !llvm.ptr<1>)",
        ),
        # A result, and a pointer it writes through.
        (
            "%r = llvm.call @_Z__spirv_ocl_fract(%v, %p) : (vector<8xf32>, !llvm.ptr<1>) -> vector<8xf32>",
            "llvm.func @_Z__spirv_ocl_fract(vector<8xf32>, !llvm.ptr<1>) -> vector<8xf32>",
        ),
    ],
    ids=["opaque", "opencl-no-result", "opencl-pointer"],
)
def test_calls_with_effects_keep_their_order(call: str, declaration: str) -> None:
    """Not moved, nor memory accesses past them: a later load sees what they wrote."""
    count, _, text = _rewrite(_with_call_before_stores(call, declaration))
    assert count == 1
    lines = _kernel_lines(text)
    name = call.split("@")[1].split("(")[0]
    index = next(i for i, line in enumerate(lines) if f"@{name}" in line)
    assert all("llvm.load" not in line for line in lines[index:])
    assert all("llvm.store" not in line for line in lines[:index])


def test_opencl_math_moves() -> None:
    """An OpenCL function of values only is a value: computed right before its use."""
    body = f"""%l = llvm.load %p : !llvm.ptr<1> -> {_F32}
      %r = llvm.call @_Z__spirv_ocl_exp(%v) : ({_F32}) -> {_F32}
      llvm.store %l, %p : {_F32}, !llvm.ptr<1>
      llvm.store %r, %p : {_F32}, !llvm.ptr<1>"""
    declaration = f"llvm.func @_Z__spirv_ocl_exp({_F32}) -> {_F32}"
    _, _, text = _rewrite(_module(body, declarations=declaration))
    lines = _kernel_lines(text)
    call = next(i for i, line in enumerate(lines) if "llvm.call" in line)
    stores = [i for i, line in enumerate(lines) if "llvm.store" in line]
    assert stores[0] < call < stores[1]


def test_a_block_with_regions_stays() -> None:
    """A value may be used only inside a region: it must not move past its op."""
    body = f"""%x = llvm.fadd %v, %v : {_F32}
      %l = llvm.load %p : !llvm.ptr<1> -> {_F32}
      "test.region"() ({{
        llvm.store %x, %p : {_F32}, !llvm.ptr<1>
        "test.end"() : () -> ()
      }}) : () -> ()
      llvm.store %l, %p : {_F32}, !llvm.ptr<1>"""
    count, before, after = _rewrite(_module(body))
    assert count == 0
    assert after == before


def test_a_value_used_in_another_block_stays_available() -> None:
    body = _chunks().replace("llvm.store %v6_0, %p", "llvm.store %v5_0, %p", 1)
    body += f"""
      llvm.br ^next
    ^next:
      llvm.store %v6_0, %p : {_F32}, !llvm.ptr<1>"""
    count, _, text = _rewrite(_module(body))
    assert count == 2
    assert text.index("llvm.br ^bb1") < text.rindex("llvm.store")


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ("spill_size:      0\\0A", 0),
        ("spill_size:      4096\\0A  x: 1\\0Aspill_size: 8\\0A", 4104),
        ("no such key", 0),
    ],
    ids=["none", "sum", "absent"],
)
def test_spill_bytes(info: str, expected: int) -> None:
    text = f"""
module attributes {{gpu.container_module}} {{
  gpu.binary @k [#gpu.object<#xevm.target, "\\7FELF{info}">]
}}"""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        assert spill_bytes(module.operation) == expected


def _binary(name: str, problems: list[kb.Problem]) -> str:
    """Lowered in this process (no compile timeout), so the patches apply."""
    problem = {p.name: p for p in problems}[name]
    args = problem.args(*(xpu_tensor(*s, dtype=problem.dtype) for s in problem.shapes))
    with patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}):
        return lower_xpu(problem.kernel, list(args), "binary")


def _spy() -> tuple[list[int], object]:
    calls: list[int] = []
    real = xpu_target.order_tile_chunks

    def spy(module: object) -> int:
        calls.append(1)
        return real(module)

    return calls, spy


@BINARIES
@pytest.mark.isolated
def test_a_kernel_without_spill_is_not_ordered() -> None:
    """1_Square and elementwise Softplus, whose order is not touched."""
    calls, spy = _spy()
    with patch.object(xpu_target, "order_tile_chunks", spy):
        for name in ("1_Square_matrix_multiplication_", "29_Softplus"):
            _binary(name, list(kb.PROBLEMS))
    assert calls == []


@BINARIES
@pytest.mark.isolated
def test_a_spilling_kernel_is_ordered() -> None:
    """The exact problem: its first compile spills, the kept one does not."""
    calls, spy = _spy()
    spills: list[int] = []
    real = xpu_target.spill_bytes

    def record(module: object) -> int:
        spills.append(real(module))
        return spills[-1]

    with (
        patch.object(xpu_target, "order_tile_chunks", spy),
        patch.object(xpu_target, "spill_bytes", record),
    ):
        text = _binary("29_Matmul_Mish_Mish", list(kb.LEVEL2_PROBLEMS))
    assert calls == [1]
    assert spills[0] > 0
    assert spills[1] == 0
    assert spill_bytes_of(text) == 0


def spill_bytes_of(text: str) -> int:
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        return spill_bytes(ir.Module.parse(text).operation)


@BINARIES
@pytest.mark.isolated
def test_an_order_that_does_not_help_is_dropped() -> None:
    """With an order that changes nothing, the spill stays, and the first binary
    is the one returned (byte for byte: same input, same compile)."""
    with patch.object(xpu_target, "order_tile_chunks", lambda module: 0):
        kept = _binary("29_Matmul_Mish_Mish", list(kb.LEVEL2_PROBLEMS))
    assert spill_bytes_of(kept) > 0


@pytest.mark.isolated
def test_order_keeps_kernelbench_ir_valid() -> None:
    problem = {p.name: p for p in kb.LEVEL2_PROBLEMS}["29_Matmul_Mish_Mish"]
    args = problem.args(*(xpu_tensor(*s, dtype=problem.dtype) for s in problem.shapes))
    text = lower_xpu(problem.kernel, list(args), "xevm")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        assert order_tile_chunks(module) >= 1
        assert module.operation.verify()
