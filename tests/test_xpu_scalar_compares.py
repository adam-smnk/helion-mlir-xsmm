"""``test_scalar_compares`` on the device. ``HELION_TEST_FULL_SWEEP=1``: every number and type."""

from __future__ import annotations

import helion
import pytest
import torch

from tests import scalar_compare_kernels as kernels
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated]


@pytest.fixture(autouse=True)
def _full_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[32, 32]),
    )


def _call(kernel: helion.Kernel, x: torch.Tensor, *scalars: object) -> object:
    out = kernel(x.to("xpu"), *scalars)
    torch.xpu.synchronize()
    return tuple(o.cpu() for o in out) if isinstance(out, tuple | list) else out.cpu()


def _run(fn: object, x: torch.Tensor) -> torch.Tensor:
    return _call(_kernel(fn), x)


def _same_outputs(fn: object, x: torch.Tensor, expected: list[torch.Tensor]) -> None:
    outputs = kernels.as_outputs(_run(fn, x))
    assert len(outputs) == len(expected)
    for out, want in zip(outputs, expected, strict=True):
        assert torch.equal(out, want)


def _same_or_refused(fn: object, x: torch.Tensor, expected: list[torch.Tensor]) -> None:
    """A tile type the XPU has no read for (8 bits) may be refused, never wrong. Found out without
    touching the device: a process that had one once hung at 100 % after such a refusal."""
    try:
        lower_mlir(
            _kernel(fn),
            [xpu_tensor(*x.shape, dtype=x.dtype)],
            target="xpu",
            stop_at="xegpu-wg",
        )
    except UnsupportedOperationError:
        return
    _same_outputs(fn, x, expected)


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_float_tile_as_torch(dtype: torch.dtype) -> None:
    """Every op against every number, packed in one kernel."""
    x = kernels.packed_input(kernels.NUMBERS, dtype)
    _same_outputs(
        kernels.compare_kernel(kernels.NUMBERS),
        x,
        kernels.all_compares(x, kernels.NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_float_tile_one_compare(dtype: torch.dtype, op: str) -> None:
    """Unpacked: fusion with the others can't be what hides a bug."""
    x = kernels.packed_input((0.1,), dtype)
    _same_outputs(
        kernels.compare_kernel((0.1,), (op,)), x, kernels.all_compares(x, (0.1,), (op,))
    )


@pytest.mark.parametrize("dtype", kernels.INT_TILES, ids=str)
def test_integer_tile_as_torch(dtype: torch.dtype) -> None:
    x = kernels.packed_input(kernels.INT_NUMBERS, dtype)
    _same_or_refused(
        kernels.compare_kernel(kernels.INT_NUMBERS),
        x,
        kernels.all_compares(x, kernels.INT_NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
@pytest.mark.parametrize("dtype", kernels.INT_TILES, ids=str)
def test_integer_tile_one_compare(dtype: torch.dtype, op: str) -> None:
    x = kernels.packed_input((-0.5, -3), dtype)
    _same_or_refused(
        kernels.compare_kernel((-0.5,), (op,)),
        x,
        kernels.all_compares(x, (-0.5,), (op,)),
    )


def test_bool_tile_as_torch() -> None:
    x = kernels.bool_input()
    _same_or_refused(
        kernels.compare_kernel(kernels.BOOL_NUMBERS),
        x,
        kernels.all_compares(x, kernels.BOOL_NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
def test_bool_tile_one_compare(op: str) -> None:
    x = kernels.bool_input()
    _same_or_refused(
        kernels.compare_kernel((0.5,), (op,)), x, kernels.all_compares(x, (0.5,), (op,))
    )


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
@pytest.mark.parametrize("name", list(kernels.COMBINED))
def test_combined_compares_as_torch(name: str, dtype: torch.dtype) -> None:
    """Boundary values, and the lane patterns of IGC's one-row ``fabs``, right in row 0 only:
    exact regressions, on one lowering."""
    fn, ref = kernels.COMBINED[name]
    kernel = _kernel(fn)
    for x in (
        kernels.boundary_input(1.0, dtype, (128, 256)),
        kernels.lane_input((128, 256), dtype, first_inside=True),
        kernels.lane_input((128, 256), dtype, first_inside=False),
    ):
        assert kernels.same(_call(kernel, x), ref(x))


@pytest.mark.parametrize("name", list(kernels.INT_ABS_KERNELS))
def test_int_abs_as_torch(name: str) -> None:
    """Exact regression: IGC's one-row ``llvm.abs.v16i32``, 30471 of 32768 wrong."""
    kernel = _kernel(kernels.INT_ABS_KERNELS[name])
    for x in (
        kernels.integer_input((128, 256)),
        (kernels.lane_input((128, 256), torch.float32, first_inside=True) * 50).to(
            torch.int32
        ),
        (kernels.lane_input((128, 256), torch.float32, first_inside=False) * 50).to(
            torch.int32
        ),
    ):
        assert torch.equal(_call(kernel, x), x.abs())


@pytest.mark.parametrize("name", list(kernels.F64_KERNELS))
def test_f64_math_narrowed(name: str) -> None:
    fn, ref = kernels.F64_KERNELS[name]
    x = kernels.boundary_input(1.0, torch.float32, (128, 256))
    assert kernels.same(_run(fn, x), ref(x))


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_runtime_number_as_torch(dtype: torch.dtype) -> None:
    kernel = _kernel(kernels.compare_runtime)
    for number in (0.1, -0.1):
        x = kernels.boundary_input(number, dtype)
        assert torch.equal(_call(kernel, x, number), (x <= number).float())


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_compare_with_other_users_as_torch(dtype: torch.dtype) -> None:
    x = kernels.boundary_input(0.1, dtype)
    c = x <= 0.1
    expected = torch.where(c, x, x * 0.0) + c.to(dtype)
    assert kernels.same(_run(kernels.compare_with_other_users, x), expected)
