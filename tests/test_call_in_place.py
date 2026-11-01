"""The in-place call path (``CompiledEntry.call_in_place``).

A call whose arguments are all tensors of the entry's static shapes and dtypes,
contiguous, on the target's device and not aliasing an inout skips the general
path of ``driver.call_entry`` (checks, staging, hooks): its arguments are packed
once per entry. Every other call must take the general path, with its results
and errors unchanged.
"""

from __future__ import annotations

import threading

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import run_direct

from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir import driver
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def copy_plus_one(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        b[tile] = a[tile] + 1.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def write_both(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        a[tile] = a[tile] + 1.0
        b[tile] = b[tile] * 2.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def add_int(x: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + k
    return out


@helion.kernel(backend="mlir", static_shapes=False, config=_cfg(8))
def add_one_dynamic(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, 0, tn] + y[tm, 0, tn]
    return out


@pytest.fixture
def general_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The argument names the general path checked, in order."""
    seen: list[str] = []
    checked = driver._checked_tensor

    def spy(target, arg, value):
        seen.append(arg.name)
        return checked(target, arg, value)

    monkeypatch.setattr(driver, "_checked_tensor", spy)
    return seen


def test_static_contiguous_call_is_in_place(general_calls: list[str]) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    general_calls.clear()
    # Twice: the packed arguments are reused with new data pointers.
    for _ in range(2):
        x, y = torch.randn(16, 16), torch.randn(16, 16)
        torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    assert general_calls == []


def test_inout_parameter_is_written_in_place(general_calls: list[str]) -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    general_calls.clear()
    a, b = torch.randn(16), torch.zeros(16)
    assert run_direct(copy_plus_one, [a, b]) is None
    torch.testing.assert_close(b, a + 1.0)
    assert general_calls == []


def test_non_contiguous_input_takes_general_path(general_calls: list[str]) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    general_calls.clear()
    xt = x.t()
    torch.testing.assert_close(run_direct(scaled_add, [xt, y]), xt * 2.0 + y)
    assert general_calls


def test_unit_dim_with_other_stride_takes_general_path(
    general_calls: list[str],
) -> None:
    x, y = torch.randn(8, 1, 16), torch.randn(8, 1, 16)
    torch.testing.assert_close(run_direct(add_rows, [x, y]), (x + y)[:, 0])
    general_calls.clear()
    # Contiguous for torch, but not the strides the packed descriptor holds.
    odd = torch.randn(8, 16).as_strided((8, 1, 16), (16, 1, 1))
    assert odd.is_contiguous()
    torch.testing.assert_close(run_direct(add_rows, [odd, y]), (odd + y)[:, 0])
    assert general_calls


def test_input_aliasing_an_inout_takes_general_path(general_calls: list[str]) -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    general_calls.clear()
    x = torch.randn(16)
    expected = x + 1.0
    run_direct(copy_plus_one, [x, x])
    torch.testing.assert_close(x, expected)
    assert general_calls
    # A partial overlap too.
    general_calls.clear()
    buffer = torch.randn(24)
    a, b = buffer[:16], buffer[8:]
    expected = a + 1.0
    run_direct(copy_plus_one, [a, b])
    torch.testing.assert_close(b, expected)
    assert general_calls


def test_inouts_sharing_memory_are_still_rejected() -> None:
    a, b = torch.randn(16), torch.randn(16)
    run_direct(write_both, [a, b])
    x = torch.randn(16)
    with pytest.raises(UnsupportedOperationError, match="share memory"):
        run_direct(write_both, [x, x])


def test_wrong_dtype_reports_the_general_error() -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    entry = _entry(copy_plus_one, [a, b])
    with pytest.raises(ValueError, match="dtype"):
        driver.call_entry(entry, [b, a.double()])


def test_scalar_and_dynamic_entries_have_no_in_place_path(
    general_calls: list[str],
) -> None:
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(add_int, [x, 3]), x + 3)
    torch.testing.assert_close(run_direct(add_one_dynamic, [x]), x + 1.0)
    assert _entry(add_int, [x, 3])._static is None
    assert _entry(add_one_dynamic, [x])._static is None
    general_calls.clear()
    torch.testing.assert_close(run_direct(add_int, [x, 4]), x + 4)
    torch.testing.assert_close(run_direct(add_one_dynamic, [x]), x + 1.0)
    assert general_calls


def test_busy_packed_arguments_take_general_path(general_calls: list[str]) -> None:
    # The Python in-place path (no native EntryCall) packs into one buffer
    # per entry; the native paths pack on the stack and take no lock.
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    values = [{"x": x, "y": y}.get(arg.name, torch.empty(16, 16)) for arg in entry.args]
    static = entry._static
    native_call = entry._native
    object.__setattr__(entry, "_native", None)
    general_calls.clear()
    assert static.lock.acquire(blocking=False)
    try:
        assert entry.call_in_place(values) is None
        driver.call_entry(entry, values)
    finally:
        static.lock.release()
        object.__setattr__(entry, "_native", native_call)
    torch.testing.assert_close(values[0], x * 2.0 + y)
    assert general_calls


def test_concurrent_calls_are_correct() -> None:
    inputs = [(torch.randn(16, 16), torch.randn(16, 16)) for _ in range(8)]
    run_direct(scaled_add, list(inputs[0]))
    results: list[torch.Tensor | None] = [None] * len(inputs)

    def work(i: int) -> None:
        for _ in range(50):
            results[i] = run_direct(scaled_add, list(inputs[i]))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(inputs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for (x, y), result in zip(inputs, results, strict=True):
        torch.testing.assert_close(result, x * 2.0 + y)


def test_packed_descriptors_match_the_general_packing() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    out = torch.empty(16, 16)
    values = [
        out if arg.role == "inout" else {"x": x, "y": y}[arg.name] for arg in entry.args
    ]
    static = entry._static
    for slot, value in zip(static.slots, values, strict=True):
        static.buffer[slot] = static.buffer[slot + 1] = value.data_ptr()
    _, (general, _) = execution._packed_args(entry.args, values)
    descriptors = len(general) - 2 * len(values)
    assert list(static.buffer)[:descriptors] == list(general)[:descriptors]


def _entry(kernel: helion.Kernel, args: list[object]) -> execution.CompiledEntry:
    """The compiled entry ``kernel`` runs for ``args``."""
    run = kernel.bind(tuple(args))._run
    run = getattr(run, "general", run)  # the Python run behind a native KernelRun
    cells = dict(zip(run.__code__.co_freevars, run.__closure__, strict=True))
    return cells["entry"].cell_contents


def test_negative_view_input_is_read_negated() -> None:
    # Regression: a negative view (torch._neg_view) shares its storage with
    # the bit set; both call paths passed the storage as is (wrong sign).
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    neg = torch._neg_view(x)
    assert neg.is_neg()
    torch.testing.assert_close(run_direct(scaled_add, [neg, y]), -x * 2.0 + y)
    # The general path (a non-contiguous negative view).
    neg_t = torch._neg_view(x.t())
    torch.testing.assert_close(run_direct(scaled_add, [neg_t, y]), -x.t() * 2.0 + y)


def test_negative_view_inout_is_written_negated() -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    storage = torch.zeros(16)
    out = torch._neg_view(storage)
    run_direct(copy_plus_one, [a, out])
    torch.testing.assert_close(out, a + 1.0)
    torch.testing.assert_close(storage, -(a + 1.0))


def test_tensor_the_target_rejects_reports_the_device_error(
    general_calls: list[str],
) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    values = [{"x": x, "y": y}.get(arg.name, torch.empty(16, 16)) for arg in entry.args]
    general_calls.clear()
    # Static shape, dtype and strides match: only passes_in_place refuses it.
    values[0] = values[0].to("meta")
    with pytest.raises(NotImplementedError, match="only CPU is supported"):
        driver.call_entry(entry, values)
    assert general_calls
