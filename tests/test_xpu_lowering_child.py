"""A lowering process that ends in C ``exit`` (``LLVM ERROR``) leaves the parent's
GPU state alone and still reports its exit code and error."""

from __future__ import annotations

import ctypes
import functools
import multiprocessing
import os
from typing import TYPE_CHECKING
from unittest import mock

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.targets import xpu
from helion_mlir_backend._compiler.targets.cpu import CpuTarget

if TYPE_CHECKING:
    import pathlib

_FORK = multiprocessing.get_context("fork")


def _llvm_error(*_: object, code: int = 7) -> None:
    """What ``report_fatal_error`` does; ``PyDLL`` keeps the GIL, as bindings do.
    The default code tells the C exit from the status of the setup's ``_exit``."""
    os.write(2, b"LLVM ERROR: from the backend\n")
    ctypes.PyDLL(None).exit(code)


def _in_child(body: object) -> int:
    child = _FORK.Process(target=body)
    child.start()
    child.join(30)
    hung = child.is_alive()
    if hung:
        child.kill()
        child.join()
    assert not hung
    return child.exitcode


class _Lowering:
    def apply(self, _: ir.Module) -> ir.Module:
        _llvm_error()
        raise AssertionError("exit returned")


class _Target(xpu.XpuTarget):
    def __init__(self) -> None:
        pass

    def lowering(self, *_: object) -> _Lowering:
        return _Lowering()


def test_exit_keeps_the_code_and_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "60")
    with ir.Context():
        module = ir.Module.parse("module {}")
        with pytest.raises(RuntimeError) as info:
            execution._lower(module, "entry", _Target(), "pipeline")
    assert str(info.value).endswith("with exit code 1: LLVM ERROR: from the backend")


def _register_marks(path: pathlib.Path) -> ctypes.CFUNCTYPE:
    """A thread-local destructor and an exit handler that write to ``path``."""
    handler = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: path.write_text("ran"))
    libc = ctypes.PyDLL(None)
    address = ctypes.cast(handler, ctypes.c_void_p).value
    libc.__cxa_thread_atexit_impl.argtypes = [ctypes.c_void_p] * 3
    libc.__cxa_thread_atexit_impl(address, None, address)
    libc.__cxa_atexit.argtypes = [ctypes.c_void_p] * 3
    libc.__cxa_atexit(address, None, None)
    return handler


def test_exit_runs_no_destructors_or_handlers(tmp_path: pathlib.Path) -> None:
    marks = tmp_path / "ran"

    def body() -> None:
        keep = _register_marks(marks)  # noqa: F841 (C exit calls into it)
        _Target().enter_lowering_child()
        _llvm_error()

    assert _in_child(body) == 1
    assert not marks.exists()


@pytest.mark.isolated(timeout=90)
def test_without_the_setup_exit_runs_them(tmp_path: pathlib.Path) -> None:
    """The control of the test above: same child, no setup. Isolated: a real exit
    in a fork of a process that ran threads can wait forever on them (seen in an
    xdist worker)."""
    marks = tmp_path / "ran"

    def body() -> None:
        keep = _register_marks(marks)  # noqa: F841 (C exit calls into it)
        _llvm_error()

    assert _in_child(body) == 7
    assert marks.exists()


def test_a_libc_without_the_symbol_closes_device_descriptors() -> None:
    def body() -> None:
        device = os.open("/dev/null", os.O_RDONLY)
        other = os.open("/proc/self/stat", os.O_RDONLY)
        with (
            mock.patch.object(xpu, "_THREAD_ATEXIT", "no_such_symbol"),
            mock.patch.object(xpu, "_DEVICE_FDS", "/dev/null"),
        ):
            _Target().enter_lowering_child()
        with pytest.raises(OSError):
            os.fstat(device)
        os.fstat(other)
        os._exit(0)

    assert _in_child(body) == 0


@pytest.mark.isolated(timeout=90)
def test_the_cpu_target_changes_nothing() -> None:
    def body() -> None:
        CpuTarget().enter_lowering_child()
        _llvm_error()

    assert _in_child(body) == 7


def _good(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def _dying(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


def _kernel(function: object) -> object:
    return helion.kernel(
        function,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64]),
    )


@pytest.mark.xpu_runtime
@pytest.mark.isolated(timeout=120)
@pytest.mark.parametrize("kernel_first", [True, False], ids=["kernel_first", "fresh"])
def test_kernels_run_after_a_lowering_that_exits(
    kernel_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression test: without the fix, the good kernel stalls or the process
    aborts on ``L0 error 1879048193`` (the timeout catches a stall)."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "60")
    x = torch.randn(128, 256, dtype=torch.float32).to("xpu")
    if kernel_first:
        _kernel(_good)(x)
        torch.xpu.synchronize()
    with (
        mock.patch.object(xpu, "compare_lanes", functools.partial(_llvm_error, code=1)),
        pytest.raises(RuntimeError, match="with exit code 1: LLVM ERROR"),
    ):
        _kernel(_dying)(x)
    out = _kernel(_good)(x)
    torch.xpu.synchronize()
    assert torch.equal(out.cpu(), (x * 2.0).cpu())
