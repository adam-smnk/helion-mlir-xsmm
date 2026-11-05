"""``xevm_tile_order``: the side-effect-free ops of a block depth first, chunk by
chunk; everything else keeps its order. The XPU target's binary stage applies it
only to modules that spill, and keeps it only if it spills less. IR and binaries;
the device regression is ``test_xpu_runtime.test_mish_mish_binary_does_not_spill``."""

from __future__ import annotations

import itertools
import os
import shutil
from unittest.mock import patch

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch
import torch.nn.functional as F

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
_NEVER_FIRST = 10**12
"""``ORDER_FIRST_ABOVE`` for the compile-first path."""
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
    """The exact problem, compile first: its first binary spills, the kept one does
    not.

    ``spill_bytes`` reads ``spill_size`` from the binary, and no such text reads as
    no spill, as it would if the binary format changed: a kernel would then simply
    never be ordered. This test is what catches that, and it runs only where
    binaries build (``BINARIES``)."""
    calls, spy = _spy()
    spills: list[int] = []
    real = xpu_target.spill_bytes

    def record(module: object) -> int:
        spills.append(real(module))
        return spills[-1]

    with (
        patch.object(xpu_target, "ORDER_FIRST_ABOVE", _NEVER_FIRST),
        patch.object(xpu_target, "order_tile_chunks", spy),
        patch.object(xpu_target, "spill_bytes", record),
    ):
        text = _binary("29_Matmul_Mish_Mish", list(kb.LEVEL2_PROBLEMS))
    assert calls == [1]
    assert spills[0] > 0
    assert 0 in spills
    assert spill_bytes_of(text) == 0


@BINARIES
@pytest.mark.isolated
def test_long_math_is_compiled_ordered_first() -> None:
    """The exact problem: expected to spill, compiled ordered, no spill: no plain
    compile."""
    records: list[tuple[bool, list[str]]] = []
    with patch.object(xpu_target, "_compile", _recording_compile(records)):
        text = _binary("29_Matmul_Mish_Mish", list(kb.LEVEL2_PROBLEMS))
    assert [(ordered, len(binaries)) for ordered, binaries in records] == [(True, 1)]
    assert _binary_lines(text) == records[0][1]
    assert spill_bytes_of(text) == 0


def spill_bytes_of(text: str) -> int:
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        return spill_bytes(ir.Module.parse(text).operation)


def _binary_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "gpu.binary @" in line]


def _recording_compile(records: list[tuple[bool, list[str]]], fail: bool = False):
    """``_compile``, keeping each call's binaries as text: the same kernel lowered
    twice need not give the same bytes, so a binary is compared within one lowering."""
    real = xpu_target._compile

    def recording(to_binary, gpu_modules, *, ordered=False):
        if ordered and fail:
            raise RuntimeError("no compile today")
        binaries = real(to_binary, gpu_modules, ordered=ordered)
        if binaries:
            records.append(
                (ordered, [str(binary).strip() for binary in binaries.values()])
            )
        return binaries

    return recording


def _two_launches(x, o1, o2):
    for tile in hl.tile(x.shape):
        o1[tile] = F.mish(F.mish(F.mish(x[tile].to(torch.float32)))).to(x.dtype)
    hl.barrier()
    for tile in hl.tile(x.shape):
        o2[tile] = F.mish(o1[tile].to(torch.float32)).to(x.dtype)
    return o2


@BINARIES
@pytest.mark.isolated
@pytest.mark.parametrize("first", ["compile", "order"])
def test_only_the_spilling_kernel_is_ordered(first: str) -> None:
    """Two launches, one spilling (mish x3, then mish): only the spilling GPU module
    is compiled ordered, and the other keeps its plain binary (ordered, it would
    take the form that cost elementwise Mish time for nothing). ``compile``: both
    compiled plain, then the spilling one ordered; ``order``: predicted."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[256, 256, 128, 256]),
    )(_two_launches)
    args = [xpu_tensor(1024, 8192, dtype=torch.bfloat16)] * 3
    records: list[tuple[bool, list[str]]] = []
    threshold = _NEVER_FIRST if first == "compile" else xpu_target.ORDER_FIRST_ABOVE
    with (
        patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}),
        patch.object(xpu_target, "ORDER_FIRST_ABOVE", threshold),
        patch.object(xpu_target, "_compile", _recording_compile(records)),
    ):
        lines = _binary_lines(lower_xpu(kernel, args, "binary"))
    shapes = [(ordered, len(binaries)) for ordered, binaries in records]
    if first == "compile":
        assert shapes == [(False, 2), (True, 1)]
        plain, ordered = records[0][1], records[1][1]
        assert lines == [ordered[0], plain[1]]
        assert ordered[0] != plain[0]
    else:
        assert shapes == [(True, 1), (False, 1)]
        assert lines == [records[0][1][0], records[1][1][0]]


def _failing_order(module: object) -> int:
    raise RuntimeError("no order today")


@BINARIES
@pytest.mark.isolated
@pytest.mark.parametrize("first", ["compile", "order"])
@pytest.mark.parametrize("failure", ["order", "compile"])
def test_a_failed_ordered_compile_keeps_the_plain_binary(
    failure: str, first: str
) -> None:
    """The attempt fails in the reorder or in IGC, after a spill (``compile``) or
    in place of the first compile (``order``): a warning, and the plain binary."""
    records: list[tuple[bool, list[str]]] = []
    threshold = 0 if first == "order" else _NEVER_FIRST
    with (
        patch.object(xpu_target, "ORDER_FIRST_ABOVE", threshold),
        patch.object(xpu_target, "spill_bytes", lambda binary: 1),
        patch.object(
            xpu_target,
            "_compile",
            _recording_compile(records, fail=failure == "compile"),
        ),
        patch.object(
            xpu_target,
            "order_tile_chunks",
            _failing_order if failure == "order" else xpu_target.order_tile_chunks,
        ),
        pytest.warns(UserWarning, match="could not be compiled in chunk order"),
    ):
        text = _binary("1_Square_matrix_multiplication_", list(kb.PROBLEMS))
    assert [ordered for ordered, _ in records] == [False]
    assert _binary_lines(text) == records[0][1]


@BINARIES
@pytest.mark.isolated
@pytest.mark.parametrize("first", ["compile", "order"])
def test_an_order_that_does_not_help_is_dropped(first: str) -> None:
    """With an order that changes nothing the spill stays, and the plain binary is
    returned, also when the ordered one was compiled first."""
    records: list[tuple[bool, list[str]]] = []
    threshold = _NEVER_FIRST if first == "compile" else xpu_target.ORDER_FIRST_ABOVE
    with (
        patch.object(xpu_target, "ORDER_FIRST_ABOVE", threshold),
        patch.object(xpu_target, "order_tile_chunks", lambda module: 0),
        patch.object(xpu_target, "_compile", _recording_compile(records)),
    ):
        kept = _binary("29_Matmul_Mish_Mish", list(kb.LEVEL2_PROBLEMS))
    assert spill_bytes_of(kept) > 0
    plain = next(binaries for ordered, binaries in records if not ordered)
    assert _binary_lines(kept) == plain
    assert sorted(ordered for ordered, _ in records) == [False, True]


@pytest.mark.isolated
def test_order_keeps_kernelbench_ir_valid() -> None:
    problem = {p.name: p for p in kb.LEVEL2_PROBLEMS}["29_Matmul_Mish_Mish"]
    args = problem.args(*(xpu_tensor(*s, dtype=problem.dtype) for s in problem.shapes))
    text = lower_xpu(problem.kernel, list(args), "xevm")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        assert order_tile_chunks(module) >= 1
        assert module.operation.verify()
