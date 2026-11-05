"""``print`` in a kernel (``hl.device_print``): the printed values on each target, the kernel's
result unchanged, the refusals, and no change to a kernel without a print."""

from __future__ import annotations

import contextlib
import ctypes
import os
import re
import tempfile
from typing import TYPE_CHECKING
import warnings

import helion
from helion._compiler.backend_registry import get_backend_class
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import xpu_tensor

from helion_mlir_backend import api
from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms.prints import guard_prints
from helion_mlir_backend._compiler.mlir_transforms.prints import lower_prints_to_libc
from helion_mlir_backend._compiler.mlir_transforms.prints import (
    settle_prints_of_stored_buffers,
)
from helion_mlir_backend._compiler.targets import xpu as xpu_target

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

BLOCKS_2D = [32, 32]
BLOCKS_GEMM = [32, 32, 32]
XPU_BLOCKS_2D = [64, 64]
XPU_BLOCKS_GEMM = [64, 64, 32]


def _kernel(fn: Callable, blocks: list[int]) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )


@contextlib.contextmanager
def _stdout_lines() -> Iterator[list[str]]:
    """The lines written to fd 1 inside the block, by C or the driver, in the list after it.
    Works in an isolated child, whose fds aren't the ones pytest's fixtures capture."""
    lines: list[str] = []
    flush = ctypes.CDLL(None).fflush
    flush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+") as file:
        os.dup2(file.fileno(), 1)
        try:
            yield lines
        finally:
            flush(None)
            os.dup2(saved, 1)
            os.close(saved)
        file.seek(0)
        lines.extend(file.read().splitlines())


def _elements(lines: list[str], label: str) -> list[tuple[tuple[int, ...], float]]:
    """``label[i, j]: v`` lines; every line must be whole: threads must not split one."""
    elements = []
    for line in lines:
        match = re.fullmatch(rf"{re.escape(label)}\[([\d, ]+)\]: (-?[\d.]+)", line)
        assert match, f"not a whole line: {line!r}"
        where = tuple(int(i) for i in match[1].split(", "))
        elements.append((where, float(match[2])))
    return elements


def _check_positions(
    elements: list[tuple[tuple[int, ...], float]],
    expected: torch.Tensor,
    block: int,
    *,
    calls: int = 1,
) -> None:
    """Each position of a tile holds the values of that position in all the tiles, so a
    swapped index or a value printed for the wrong element shows: tiles run in any order,
    which is all a sorted comparison can't tell."""
    printed: dict[tuple[int, ...], list[float]] = {}
    for where, value in elements:
        printed.setdefault(where, []).append(value)
    positions = {(i, j) for i in range(block) for j in range(block)}
    assert set(printed) == positions
    tiles = torch.tensor(
        expected.double().unfold(0, block, block).unfold(1, block, block)
    )
    for (i, j), values in printed.items():
        want = sorted(tiles[..., i, j].flatten().tolist() * calls)
        assert sorted(values) == pytest.approx(want, abs=1e-6)


def _normalized_error(out: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.double()
    error = (out.double().cpu() - reference).abs().max() / reference.abs().max()
    return error.item()


def _x(size: int = 64) -> torch.Tensor:
    return torch.rand(size, size, generator=torch.Generator().manual_seed(1))


def _ab(size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    a = torch.rand(size, size, generator=generator, dtype=torch.float64)
    b = torch.rand(size, size, generator=generator, dtype=torch.float64)
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


def scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def scale_label(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("50% tile é")
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def scale_tile(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn] * 2.0
        print("v", v)
        out[tm, tn] = v
    return out


def chain(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn] * 2.0
        out[tm, tn] = torch.abs(v) + 1.0
    return out


def chain_tile(x: torch.Tensor) -> torch.Tensor:
    """The print is of a tile only computed in the kernel."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn] * 2.0
        print("v", v)
        out[tm, tn] = torch.abs(v) + 1.0
    return out


def two_tensors(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("both", v, v > 0.5)
        out[tm, tn] = v
    return out


def row_sums(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("sum", v.sum(0).sum(0))
        out[tm, tn] = v
    return out


def int_tile(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("i", v)
        out[tm, tn] = v
    return out


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gemm_k_label(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("k")
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gemm_input(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("in", a[tm, tk])
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gemm_k_label_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """No register-tiled consumer of the K loop: the label prints once per K tile."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("k")
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


def gemm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


def in_host_code(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.device_print("start", x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn]
    return out


def full_print(x: torch.Tensor) -> torch.Tensor:
    """The review's kernel, exact, printing wrong values on the B50: 11549 of 16384."""
    out = torch.full_like(x, 7.0)
    for tm, tn in hl.tile(x.size()):
        print("f", out[tm, tn])
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def in_place_before(x: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        print("b", x[tm, tn])
        x[tm, tn] = x[tm, tn] * 2.0
    return x


def in_place_after(x: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        x[tm, tn] = x[tm, tn] * 2.0
        print("a", x[tm, tn])
    return x


def read_and_accumulate(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm, tn in hl.tile(x.size()):
        print("z", out[tm, tn])
        out[tm, tn] = out[tm, tn] + x[tm, tn]
    return out


def input_and_output(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, -1.0)
    for tm, tn in hl.tile(x.size()):
        print("x", x[tm, tn])
        out[tm, tn] = x[tm, tn] * 2.0
        print("o", out[tm, tn])
    return out


def maybe_aliased(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(a.size()):
        print("p", a[tm, tn])
        b[tm, tn] = a[tm, tn] * 2.0
    return b


def gemm_stored(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
        print("o", out[tm, tn])
    return out


def gemm_store_in_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Later K tiles overwrite the tile the print reads, in the other threads' shares."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
            out[tm, tn] = acc
            print("o", out[tm, tn])
    return out


def l2_plain(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - 0.25) * 2.0).to(x.dtype)
    return out


def l2_k_acc(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
            print("acc", acc)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - 0.25) * 2.0).to(x.dtype)
    return out


def two_labels(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("one")
        print("two")
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def host_print(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    print("host", "only")
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def print_in_an_if(x: torch.Tensor) -> torch.Tensor:
    """A barrier here could be reached by some threads only, as far as the rule can tell."""
    out = torch.full_like(x, 7.0)
    for tm, tn in hl.tile(x.size()):
        if tm.begin == 0:
            print("i", out[tm, tn])
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def two_grid_loops(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, 7.0)
    for tm, tn in hl.tile(x.size()):
        print("a", out[tm, tn])
        out[tm, tn] = x[tm, tn] * 2.0
    hl.barrier()
    for tm, tn in hl.tile(x.size()):
        print("c", out[tm, tn])
        out[tm, tn] = out[tm, tn] + 1.0
    return out


def position(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("at", tm.begin)
        out[tm, tn] = x[tm, tn]
    return out


def position_tensor(x: torch.Tensor) -> torch.Tensor:
    """A tile position, which Helion's rule takes as a 0-d tensor (a computed one)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("at", torch.zeros([], dtype=torch.int64, device=x.device) + tm.begin)
        out[tm, tn] = x[tm, tn]
    return out


# --- The CPU target ----------------------------------------------------------------


def test_label_once_per_tile_and_result_unchanged() -> None:
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(scale_label, BLOCKS_2D)(x)
    assert lines == ["50% tile é"] * 4
    assert torch.equal(out, _kernel(scale, BLOCKS_2D)(x))


def test_tile_prints_its_values_on_whole_lines() -> None:
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(scale_tile, BLOCKS_2D)(x)
    elements = _elements(lines, "v")
    assert len(elements) == x.numel()
    _check_positions(elements, x * 2, 32)
    assert torch.equal(out, x * 2)


def test_each_tensor_prints_bool_as_zero_or_one() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(two_tensors, BLOCKS_2D)(x)
    elements = _elements(lines, "both")
    assert len(elements) == 2 * x.numel()
    values = [value for _, value in elements]
    integers = [v for line, v in zip(lines, values, strict=True) if "." not in line]
    assert len(integers) == x.numel()
    assert sum(integers) == (x > 0.5).sum().item()
    assert set(integers) == {0.0, 1.0}


def test_integer_tile() -> None:
    x = (_x() * 1000).to(torch.int32) - 500
    with _stdout_lines() as lines:
        _kernel(int_tile, BLOCKS_2D)(x)
    printed = sorted(int(value) for _, value in _elements(lines, "i"))
    assert printed == sorted(x.flatten().tolist())


def test_reduced_scalar() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(row_sums, BLOCKS_2D)(x)
    printed = sorted(float(line.split()[1]) for line in lines)
    tiles = [
        x[i : i + 32, j : j + 32].double().sum().item()
        for i in (0, 32)
        for j in (0, 32)
    ]
    assert printed == pytest.approx(sorted(tiles), rel=1e-5)


def test_scalar_pipeline() -> None:
    x = _x()
    kernel = helion.kernel(
        scale_tile,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=BLOCKS_2D, mlir_pipeline="scalar"),
    )
    with _stdout_lines() as lines:
        out = kernel(x)
    assert len(_elements(lines, "v")) == x.numel()
    assert torch.equal(out, x * 2)


def test_label_once_per_k_tile() -> None:
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_k_label_f32, BLOCKS_GEMM)(a, b)
    assert lines == ["k"] * (4 * 2)
    assert _normalized_error(out, a.double() @ b.double()) < 2**-7


def test_k_loop_may_repeat_but_the_result_stands() -> None:
    """A documented limit: fusion can copy a K loop with a print for a register-tiled
    consumer, so its lines repeat. Whole repeats only, and a right result."""
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_k_label, BLOCKS_GEMM)(a, b)
    assert lines
    assert set(lines) == {"k"}
    assert len(lines) % (4 * 2) == 0
    assert _normalized_error(out.to(torch.float32), a.double() @ b.double()) < 2**-7


def test_loaded_tile_in_the_k_loop() -> None:
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_input, BLOCKS_GEMM)(a, b)
    printed = {value for _, value in _elements(lines, "in")}
    assert printed <= {round(value, 6) for value in a.double().flatten().tolist()}
    assert len(printed) > 1
    assert _normalized_error(out.to(torch.float32), a.double() @ b.double()) < 2**-7


def test_print_in_a_k_loop_can_change_the_path_within_rounding() -> None:
    """The review's kernel, exact: this print in the K loop took an L2 from FMAs to AMX at
    these tiles on AMX hosts, changing a bf16 unit on some inputs, so its result can't be
    compared bit for bit; against fp64 it stays right."""
    torch.manual_seed(0)
    x, w = (
        torch.rand(128, 128, dtype=torch.float64).to(torch.bfloat16) for _ in range(2)
    )
    bias = torch.rand(128, dtype=torch.float64).to(torch.bfloat16)
    reference = torch.relu(((x.double() @ w.double().T + bias.double()) - 0.25) * 2.0)
    with _stdout_lines() as lines:
        out = _kernel(l2_k_acc, BLOCKS_GEMM)(x, w, bias)
    assert lines
    assert _normalized_error(out.to(torch.float32), reference) < 2**-7
    plain = _kernel(l2_plain, BLOCKS_GEMM)(x, w, bias)
    assert _normalized_error(plain.to(torch.float32), reference) < 2**-7


def test_cpu_prints_a_stored_buffer_before_its_store() -> None:
    """Tiles don't overlap and each thread runs a whole tile: right on the CPU."""
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(full_print, BLOCKS_2D)(x)
    assert {value for _, value in _elements(lines, "f")} == {7.0}
    assert torch.equal(out, x * 2)


def test_position_through_a_tensor_on_the_cpu() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(position_tensor, BLOCKS_2D)(x)
    assert sorted(lines) == ["at 0", "at 0", "at 32", "at 32"]


# --- Refusals ----------------------------------------------------------------------


@pytest.mark.helion_internals
def test_refused_in_host_code() -> None:
    with pytest.raises(UnsupportedOperationError, match=r"hl\.device_print"):
        _kernel(in_host_code, BLOCKS_2D)(_x())


@pytest.mark.helion_internals
def test_refused_non_tensor_argument() -> None:
    """Helion's rule, untouched: this backend doesn't override it."""
    with pytest.raises(Exception, match="only supports runtime tensor values"):
        lower_mlir(_kernel(position, BLOCKS_2D), [_x()], target="cpu")


def test_refused_computed_tile_on_the_xpu() -> None:
    kernel = _kernel(chain_tile, XPU_BLOCKS_2D)
    with pytest.raises(UnsupportedOperationError) as error:
        lower_mlir(kernel, [xpu_tensor(128, 128)], target="xpu", stop_at="xevm")
    text = str(error.value)
    assert "hl.device_print of a computed or partial tile on XPU" in text
    assert "print a full loaded tile of an input, or the output tile after" in text


def test_refused_partial_tile_on_the_xpu_names_it() -> None:
    kernel = _kernel(int_tile, XPU_BLOCKS_2D)
    with pytest.raises(UnsupportedOperationError, match="computed or partial tile"):
        lower_mlir(kernel, [xpu_tensor(100, 128)], target="xpu", stop_at="xevm")


STORED_BUFFER_WARNING = "may show wrong values"


def _lowered_and_warnings(fn: Callable, bf16: bool) -> tuple[str, list[str]]:
    """The kernel's IR at ``xegpu-wg``, and the messages of the warnings from lowering it.
    ``simplefilter("always")``: the project's ``ignore::UserWarning`` would otherwise leave
    the list empty whatever the backend does."""
    args = (
        [xpu_tensor(128, 128, dtype=torch.bfloat16) for _ in range(2)]
        if bf16
        else [xpu_tensor(128, 128)]
    )
    blocks = XPU_BLOCKS_GEMM if bf16 else XPU_BLOCKS_2D
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = str(
            lower_mlir(_kernel(fn, blocks), args, target="xpu", stop_at="xegpu-wg")
        )
    return text[text.index("gpu.module") :], [str(w.message) for w in caught]


def _stored_buffer_warnings(messages: list[str]) -> list[str]:
    return [message for message in messages if STORED_BUFFER_WARNING in message]


@pytest.mark.parametrize(
    ("fn", "bf16"),
    [
        (full_print, False),
        (in_place_before, False),
        (read_and_accumulate, False),
        (gemm_store_in_k, True),
    ],
    ids=["full", "in-place-before", "accumulate", "store-in-k"],
)
def test_print_that_can_see_other_threads_stores_gets_barriers(
    fn: Callable, bf16: bool
) -> None:
    text, messages = _lowered_and_warnings(fn, bf16)
    assert _stored_buffer_warnings(messages) == []
    assert text.count("gpu.barrier") == 2
    assert re.search(r"gpu\.barrier.*gpu\.printf.*gpu\.barrier", text, re.DOTALL)


def test_each_grid_loop_gets_its_own_barriers() -> None:
    kernel = _kernel(two_grid_loops, [64] * 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = str(
            lower_mlir(kernel, [xpu_tensor(128, 128)], target="xpu", stop_at="xegpu-wg")
        )
    assert _stored_buffer_warnings([str(w.message) for w in caught]) == []
    assert text.count("gpu.barrier") == 4


def test_warns_with_no_barrier_in_a_users_if() -> None:
    text, messages = _lowered_and_warnings(print_in_an_if, False)
    found = _stored_buffer_warnings(messages)
    assert len(found) == 1
    assert found[0].startswith("'i'")
    assert "control flow all threads may not run" in found[0]
    assert "Print the tile after its store" in found[0]
    assert "or print an input the kernel only reads" in found[0]
    assert "gpu.barrier" not in text


def test_warning_reaches_the_caller_with_pytest_warns() -> None:
    with pytest.warns(UserWarning, match=STORED_BUFFER_WARNING):
        lower_mlir(
            _kernel(print_in_an_if, XPU_BLOCKS_2D),
            [xpu_tensor(128, 128)],
            target="xpu",
            stop_at="xevm",
        )


@pytest.mark.parametrize(
    ("fn", "bf16"),
    [
        (in_place_after, False),
        (input_and_output, False),
        (scale_tile, False),
        (scale_label, False),
        (gemm_stored, True),
        (gemm_input, True),
    ],
    ids=[
        "in-place-after",
        "input-and-output",
        "tile-stored",
        "label",
        "gemm-stored",
        "k-input",
    ],
)
def test_nothing_added_for_inputs_labels_and_after_the_store(
    fn: Callable, bf16: bool
) -> None:
    text, messages = _lowered_and_warnings(fn, bf16)
    assert _stored_buffer_warnings(messages) == []
    assert "gpu.barrier" not in text


def test_warning_doesnt_change_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warned print stays as it is: the same module with or without the check."""

    def lowered() -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return str(
                lower_mlir(
                    _kernel(print_in_an_if, XPU_BLOCKS_2D),
                    [xpu_tensor(128, 128)],
                    target="xpu",
                    stop_at="xevm-final",
                )
            )

    with_check = lowered()
    monkeypatch.setattr(
        xpu_target, "settle_prints_of_stored_buffers", lambda module: []
    )
    assert lowered() == with_check
    assert "lowering_warnings" not in with_check


def test_cpu_has_no_such_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower_mlir(_kernel(full_print, BLOCKS_2D), [_x()], target="cpu")
    assert not [w for w in caught if STORED_BUFFER_WARNING in str(w.message)]


def test_cpu_lowers_a_print_before_the_store() -> None:
    text = str(lower_mlir(_kernel(full_print, BLOCKS_2D), [_x()], target="cpu"))
    assert "@printf" in text


# --- The XPU target's IR -----------------------------------------------------------


def _xegpu_lines(text: str) -> list[str]:
    """The ``xegpu`` ops without SSA names, which a print shifts."""
    return [
        re.sub(r"%[\w#]+", "%v", line.strip())
        for line in text.splitlines()
        if "xegpu." in line
    ]


@pytest.mark.parametrize(
    ("printing", "plain", "blocks", "bf16"),
    [
        (scale_label, scale, XPU_BLOCKS_2D, False),
        (scale_tile, scale, XPU_BLOCKS_2D, False),
        (gemm_k_label, gemm, XPU_BLOCKS_GEMM, True),
        (gemm_input, gemm, XPU_BLOCKS_GEMM, True),
    ],
    ids=["tile-label", "tile-stored", "k-label", "k-input"],
)
def test_xpu_print_is_guarded_and_keeps_the_xegpu_ops(
    printing: Callable, plain: Callable, blocks: list[int], bf16: bool
) -> None:
    def tensors() -> list[torch.Tensor]:
        if bf16:
            return [xpu_tensor(128, 128, dtype=torch.bfloat16) for _ in range(2)]
        return [xpu_tensor(128, 128)]

    text = str(
        lower_mlir(
            _kernel(printing, blocks), tensors(), target="xpu", stop_at="xegpu-wg"
        )
    )
    kernel = text[text.index("gpu.module") :]
    host = text[: text.index("gpu.module")]
    assert re.search(r"scf\.if %\w+ \{\s+gpu\.printf", kernel)
    assert "gpu.thread_id x" in kernel
    assert "gpu.thread_id" not in host
    plain_text = str(
        lower_mlir(_kernel(plain, blocks), tensors(), target="xpu", stop_at="xegpu-wg")
    )
    assert _xegpu_lines(text) == _xegpu_lines(plain_text)


def test_xpu_print_reaches_the_opencl_builtin() -> None:
    text = str(
        lower_mlir(
            _kernel(scale_label, XPU_BLOCKS_2D),
            [xpu_tensor(128, 128)],
            target="xpu",
            stop_at="xevm",
        )
    )
    assert "_Z6printfPU3AS2Kcz" in text
    assert "gpu.printf" not in text
    assert "_Z12get_local_idj" in text


EXPERIMENTAL = "(hl.device_print) is experimental"


def _experimental(recwarn: pytest.WarningsRecorder) -> list[str]:
    return [str(w.message) for w in recwarn.list if EXPERIMENTAL in str(w.message)]


def _lower_for(target: str, fn: Callable) -> None:
    if target == "xpu":
        args, blocks = [xpu_tensor(128, 128)], XPU_BLOCKS_2D
    else:
        args, blocks = [_x()], BLOCKS_2D
    lower_mlir(_kernel(fn, blocks), args, target=target, stop_at=_STOPS[target])


_STOPS = {"cpu": None, "xpu": "xegpu-wg"}


@pytest.mark.parametrize("target", ["cpu", "xpu"])
def test_experimental_warning_once_per_compile(
    target: str, recwarn: pytest.WarningsRecorder
) -> None:
    warnings.simplefilter("always")
    _lower_for(target, two_labels)
    messages = _experimental(recwarn)
    assert len(messages) == 1
    assert "two_labels" in messages[0]
    assert "hl.device_print" in messages[0]
    assert "'Printing in a kernel'" in messages[0]


@pytest.mark.parametrize("target", ["cpu", "xpu"])
def test_no_experimental_warning_without_a_print(
    target: str, recwarn: pytest.WarningsRecorder
) -> None:
    warnings.simplefilter("always")
    _lower_for(target, scale)
    assert _experimental(recwarn) == []


def test_host_print_is_no_kernel_print(recwarn: pytest.WarningsRecorder) -> None:
    warnings.simplefilter("always")
    _lower_for("cpu", host_print)
    assert _experimental(recwarn) == []


@pytest.mark.parametrize("target", ["cpu", "xpu"])
def test_every_compile_warns_under_the_default_filter(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python's default filter shows a text once per location: a plain ``warnings.warn``
    from one line stays silent for a second compile of the same kernel. No lowering process:
    forking one changes the filters, which happens to reset that memory."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _lower_for(target, two_labels)
        _lower_for(target, two_labels)
    assert len([w for w in caught if EXPERIMENTAL in str(w.message)]) == 2


def test_stored_buffer_warning_on_every_compile_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _lower_for("xpu", print_in_an_if)
        _lower_for("xpu", print_in_an_if)
    assert len([w for w in caught if STORED_BUFFER_WARNING in str(w.message)]) == 2


# --- A kernel without a print ------------------------------------------------------


def _generated(fn: Callable, blocks: list[int], x: torch.Tensor) -> str:
    """The module before any pipeline stage."""
    hf, config, env = api._compile(_kernel(fn, blocks), [x], None)
    with env:
        module = get_backend_class("mlir")().generate_mlir(hf, config, env)
    return str(module)


def test_generated_module_differs_by_the_print_alone() -> None:
    """A label emits one op and nothing else, so a kernel without one is untouched."""

    def k_printing(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tm, tn in hl.tile(x.size()):
            print("tile")
            out[tm, tn] = x[tm, tn] * 2.0
        return out

    def k_plain(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tm, tn in hl.tile(x.size()):
            out[tm, tn] = x[tm, tn] * 2.0
        return out

    x = _x()
    printing = _generated(k_printing, BLOCKS_2D, x)
    assert printing.count("gpu.printf") == 1
    without = "".join(
        line for line in printing.splitlines(True) if "gpu.printf" not in line
    )
    assert without.replace("k_printing", "k_plain") == _generated(k_plain, BLOCKS_2D, x)


def test_xevm_final_of_a_kernel_without_a_print_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lowered() -> str:
        return str(
            lower_mlir(
                _kernel(scale, XPU_BLOCKS_2D),
                [xpu_tensor(128, 128)],
                target="xpu",
                stop_at="xevm-final",
            )
        )

    with_steps = lowered()
    monkeypatch.setattr(xpu_target, "guard_prints", lambda module: 0)
    monkeypatch.setattr(
        xpu_target, "refuse_prints_needing_allocation", lambda module: None
    )
    assert lowered() == with_steps
    assert "printf" not in with_steps
    assert "thread_id" not in with_steps


NO_PRINT = """
module {
  func.func @f(%x: index) -> index { return %x : index }
}
"""

TWO_PRINTS = """
module {
  func.func @f(%x: f32, %y: i64) {
    gpu.printf "a %f\\n", %x : f32
    gpu.printf "a %f\\n", %x : f32
    gpu.printf "b %ld\\n", %y : i64
    return
  }
}
"""


@pytest.fixture
def _context():
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        yield context


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize("rewrite", [lower_prints_to_libc, guard_prints])
def test_rewrites_leave_a_module_without_a_print_alone(rewrite: Callable) -> None:
    module = ir.Module.parse(NO_PRINT)
    before = str(module)
    assert rewrite(module) == 0
    assert str(module) == before


@pytest.mark.usefixtures("_context")
def test_libc_calls_share_strings_and_pass_doubles() -> None:
    module = ir.Module.parse(TWO_PRINTS)
    assert lower_prints_to_libc(module) == 3
    module.operation.verify()
    text = str(module)
    assert "gpu.printf" not in text
    assert text.count("llvm.mlir.global") == 2
    assert text.count("llvm.call @printf") == 3
    assert text.count("llvm.call @fflush") == 3
    assert text.count("arith.extf %arg0 : f32 to f64") == 2


@pytest.mark.usefixtures("_context")
def test_guard_wraps_each_print() -> None:
    module = ir.Module.parse(TWO_PRINTS)
    assert guard_prints(module) == 3
    module.operation.verify()
    text = str(module)
    assert text.count("gpu.thread_id x") == 3
    assert text.count("arith.cmpi eq") == 3
    assert len(re.findall(r"scf\.if %\w+ \{\s+gpu\.printf", text)) == 3


STORE_THEN_READ = """
  %v = "test.tile"() : () -> vector<8x8xf32>
  vector.transfer_write %v, %a[%i, %j] {in_bounds = [true, true]} : vector<8x8xf32>, memref<64x64xf32>
"""
SECOND_STORE = STORE_THEN_READ.replace("%v", "%v2")
READ_TILE = """
  scf.for %p = %c0 to %c8 step %c1 {
    scf.for %q = %c0 to %c8 step %c1 {
      %r = arith.addi %i, %p : index
      %s = arith.addi %j, %q : index
      %e = memref.load %SOURCE[%r, %s] : memref<64x64xf32>
      gpu.printf "e %f\\n", %e : f32
    }
  }
"""


def _kernel_module(body: str, host_args: str = "%x, %x2") -> ir.Module:
    first, second = host_args.split(", ")
    return ir.Module.parse(f"""
module attributes {{gpu.container_module}} {{
  func.func @entry(%x: memref<64x64xf32>, %x2: memref<64x64xf32>, %n: index) {{
    %y = memref.cast %x : memref<64x64xf32> to memref<64x64xf32>
    gpu.launch_func @m::@k blocks in (%n, %n, %n) threads in (%n, %n, %n) args({first} : memref<64x64xf32>, {second} : memref<64x64xf32>)
    return
  }}
  gpu.module @m {{
    gpu.func @k(%a: memref<64x64xf32>, %b: memref<64x64xf32>) kernel {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c9 = arith.constant 9 : index
      %i = "test.begin"() : () -> index
      %j = "test.begin"() : () -> index
      {body}
      gpu.return
    }}
  }}
}}
""")


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    "body",
    [
        STORE_THEN_READ + READ_TILE.replace("%SOURCE", "%a"),
        READ_TILE.replace("%SOURCE", "%b"),
        READ_TILE.replace("%SOURCE", "%b") + STORE_THEN_READ,
        STORE_THEN_READ,
    ],
    ids=["after-its-store", "input", "input-before-another-store", "no-print"],
)
def test_store_rule_is_silent(body: str) -> None:
    module = _kernel_module(body)
    before = str(module)
    assert settle_prints_of_stored_buffers(module) == []
    assert str(module) == before


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    ("body", "host_args"),
    [
        (READ_TILE.replace("%SOURCE", "%a") + STORE_THEN_READ, "%x, %x2"),
        (
            STORE_THEN_READ + SECOND_STORE + READ_TILE.replace("%SOURCE", "%a"),
            "%x, %x2",
        ),
        (
            STORE_THEN_READ + READ_TILE.replace("%SOURCE", "%a") + SECOND_STORE,
            "%x, %x2",
        ),
        (
            STORE_THEN_READ
            + READ_TILE.replace("%SOURCE", "%a").replace("to %c8", "to %c9"),
            "%x, %x2",
        ),
        (
            STORE_THEN_READ
            + READ_TILE.replace("%SOURCE", "%a").replace(
                "arith.addi %i", "arith.addi %j"
            ),
            "%x, %x2",
        ),
        (
            "scf.for %t = %c0 to %c8 step %c1 {"
            + STORE_THEN_READ
            + READ_TILE.replace("%SOURCE", "%a")
            + "}",
            "%x, %x2",
        ),
        (
            STORE_THEN_READ.replace(
                "{in_bounds = [true, true]}",
                "{in_bounds = [true, true], permutation_map = "
                "affine_map<(d0, d1) -> (d1, d0)>}",
            )
            + READ_TILE.replace("%SOURCE", "%a"),
            "%x, %x2",
        ),
        (
            '"test.writes"(%a) : (memref<64x64xf32>) -> ()'
            + READ_TILE.replace("%SOURCE", "%a"),
            "%x, %x2",
        ),
        (STORE_THEN_READ + READ_TILE.replace("%SOURCE", "%b"), "%x, %y"),
    ],
    ids=[
        "before-the-store",
        "two-stores-before",
        "a-store-after",
        "outside-the-tile",
        "swapped-begins",
        "store-in-a-loop",
        "transposed-store",
        "unknown-op",
        "one-host-buffer",
    ],
)
def test_store_rule_warns(body: str, host_args: str) -> None:
    module = _kernel_module(body, host_args)
    messages = settle_prints_of_stored_buffers(module)
    assert len(messages) == 1
    assert "also stores" in messages[0]
    assert "gpu.barrier" not in str(module)


GRID_STORE = """
  %v = "test.tile"() : () -> vector<8x8xf32>
  vector.transfer_write %v, %a[%i, %j] {in_bounds = [true, true]} : vector<8x8xf32>, memref<64x64xf32>
"""


def _grid_module(
    body: str,
    *,
    begins: tuple[str, str] = ("x", "y"),
    step: int = 8,
    grid: str = "known_grid_size = array<i32: 2, 2, 1>",
) -> ir.Module:
    """A kernel whose tile is ``[block_id * step, ...]``, ``%i`` and ``%j``."""
    return ir.Module.parse(f"""
module attributes {{gpu.container_module}} {{
  func.func @entry(%x: memref<64x64xf32>, %x2: memref<64x64xf32>, %c: i1, %n: index) {{
    gpu.launch_func @m::@k blocks in (%n, %n, %n) threads in (%n, %n, %n) args(%x : memref<64x64xf32>, %x2 : memref<64x64xf32>, %c : i1, %n : index)
    return
  }}
  gpu.module @m {{
    gpu.func @k(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %u: i1, %n: index) kernel attributes {{{grid}}} {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %step = arith.constant {step} : index
      %bx = gpu.block_id {begins[0]}
      %by = gpu.block_id {begins[1]}
      %i = arith.muli %bx, %step : index
      %j = arith.muli %by, %step : index
      {body}
      gpu.return
    }}
  }}
}}
""")


GUARDED_READ = """
  %t = gpu.thread_id x
  %first = arith.cmpi eq, %t, %c0 : index
  scf.for %p = %c0 to %c8 step %c1 {
    scf.for %q = %c0 to %c8 step %c1 {
      %r = arith.addi %FIRST, %p : index
      %s = arith.addi %j, %q : index
      %e = memref.load %a[%r, %s] : memref<64x64xf32>
      scf.if %first {
        gpu.printf "e %f\\n", %e : f32
      }
    }
  }
"""
OWN_READ = GUARDED_READ.replace("%FIRST", "%i")


@pytest.mark.usefixtures("_context")
def test_barriers_go_around_the_prints_own_loops() -> None:
    module = _grid_module(OWN_READ + GRID_STORE)
    assert settle_prints_of_stored_buffers(module) == []
    module.operation.verify()
    text = str(module)
    assert text.count("gpu.barrier") == 2
    # Just outside the print's two loops, on both sides, and before the store.
    assert re.search(
        r"gpu\.barrier\s+scf\.for.*scf\.for.*gpu\.printf.*\}\s+\}\s+\}\s+gpu\.barrier",
        text,
        re.DOTALL,
    )
    assert text.index("vector.transfer_write") > text.rindex("gpu.barrier")


@pytest.mark.usefixtures("_context")
def test_barriers_in_a_loop_of_constant_bounds() -> None:
    module = _grid_module(
        "scf.for %k = %c0 to %c8 step %c1 {" + OWN_READ + GRID_STORE + "}"
    )
    assert settle_prints_of_stored_buffers(module) == []
    assert str(module).count("gpu.barrier") == 2


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    ("body", "options", "reason"),
    [
        ("scf.if %u {" + OWN_READ + "}" + GRID_STORE, {}, "control flow"),
        (
            "scf.for %k = %c0 to %n step %c1 {" + OWN_READ + GRID_STORE + "}",
            {},
            "control flow",
        ),
        (GUARDED_READ.replace("%FIRST", "%j") + GRID_STORE, {}, "another workgroup"),
        (OWN_READ + GRID_STORE.replace("%a[%i, %j]", "%a[%c0, %j]"), {}, "another"),
        (OWN_READ + GRID_STORE, {"begins": ("x", "x")}, "another workgroup"),
        (OWN_READ + GRID_STORE, {"step": 4}, "another workgroup"),
        (OWN_READ + GRID_STORE, {"grid": ""}, "another workgroup"),
        (
            OWN_READ
            + GRID_STORE
            + GRID_STORE.replace("%v", "%v2").replace("%a[%i, %j]", "%a[%j, %i]"),
            {},
            "another workgroup",
        ),
    ],
    ids=[
        "users-if",
        "runtime-loop",
        "read-swapped",
        "store-without-block-id",
        "grid-dim-unused",
        "step-below-tile",
        "no-known-grid",
        "second-store-elsewhere",
    ],
)
def test_no_barrier_unless_proven_right(
    body: str, options: dict[str, object], reason: str
) -> None:
    module = _grid_module(body, **options)
    messages = settle_prints_of_stored_buffers(module)
    assert len(messages) == 1
    assert reason in messages[0]
    assert "gpu.barrier" not in str(module)


# --- On the device -----------------------------------------------------------------


@pytest.mark.xpu_runtime
@pytest.mark.isolated
def test_label_and_values_on_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(variable, raising=False)
    x = _x(128).to("xpu")
    with _stdout_lines() as label_lines:
        out = _kernel(scale_label, XPU_BLOCKS_2D)(x)
        torch.xpu.synchronize()
    assert label_lines == ["50% tile é"] * 4
    plain = _kernel(scale, XPU_BLOCKS_2D)(x)
    torch.xpu.synchronize()
    assert torch.equal(out.cpu(), plain.cpu())
    with _stdout_lines() as lines:
        out = _kernel(scale_tile, XPU_BLOCKS_2D)(x)
        torch.xpu.synchronize()
    elements = _elements(lines, "v")
    assert len(elements) == x.numel()
    _check_positions(elements, x.cpu() * 2, 64)
    assert torch.equal(out.cpu(), plain.cpu())


@pytest.mark.xpu_runtime
@pytest.mark.isolated
def test_gemm_k_loop_on_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(variable, raising=False)
    a, b = (t.to("xpu") for t in _ab(128))
    with _stdout_lines() as lines:
        out = _kernel(gemm_k_label, XPU_BLOCKS_GEMM)(a, b)
        torch.xpu.synchronize()
    assert lines == ["k"] * (4 * 4)
    plain = _kernel(gemm, XPU_BLOCKS_GEMM)(a, b)
    torch.xpu.synchronize()
    assert torch.equal(out.to(torch.float32).cpu(), plain.to(torch.float32).cpu())


@pytest.mark.xpu_runtime
@pytest.mark.isolated
def test_before_the_store_forms_print_their_values_on_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The review's kernels, exact; without barriers most of these values were wrong."""
    for variable in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(variable, raising=False)
    x = _x(128)
    for fn, label, expected, result in (
        (full_print, "f", torch.full_like(x, 7.0), x * 2),
        (read_and_accumulate, "z", torch.zeros_like(x), x),
        (in_place_before, "b", x, x * 2),
    ):
        with _stdout_lines() as lines:
            out = _kernel(fn, XPU_BLOCKS_2D)(x.to("xpu").clone())
            torch.xpu.synchronize()
        _check_positions(_elements(lines, label), expected, 64)
        assert torch.equal(out.cpu(), result)


@pytest.mark.xpu_runtime
@pytest.mark.isolated
def test_allowed_stores_and_inputs_print_their_values_on_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input at the call, an output after its store (over a sentinel), and an input that
    is the output at the call, which is passed as a copy."""
    for variable in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(variable, raising=False)
    x = _x(128)
    with _stdout_lines() as lines:
        out = _kernel(input_and_output, XPU_BLOCKS_2D)(x.to("xpu"))
        torch.xpu.synchronize()
    inputs = [line for line in lines if line.startswith("x[")]
    stored = [line for line in lines if line.startswith("o[")]
    assert len(inputs) == len(stored) == x.numel()
    _check_positions(_elements(inputs, "x"), x, 64)
    _check_positions(_elements(stored, "o"), x * 2, 64)
    assert torch.equal(out.cpu(), x * 2)
    aliased = x.to("xpu")
    with _stdout_lines() as lines:
        out = _kernel(maybe_aliased, XPU_BLOCKS_2D)(aliased, aliased)
        torch.xpu.synchronize()
    _check_positions(_elements(lines, "p"), x, 64)
    assert torch.equal(out.cpu(), x * 2)
