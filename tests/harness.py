"""Shared helpers for MLIR backend tests."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch._subclasses import FakeTensorMode

from helion_mlir_backend import generate_mlir
from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    import helion


@contextlib.contextmanager
def use_pipeline(pipeline: str | None) -> Iterator[None]:
    """Default compiles in the block to ``pipeline`` (``None``: unchanged)."""
    if pipeline is None:
        yield
        return
    with patch.dict(os.environ, {"HELION_MLIR_PIPELINE": pipeline}):
        yield


def scalar_pipeline() -> contextlib.AbstractContextManager[None]:
    return use_pipeline("scalar")


def opt_pipeline() -> contextlib.AbstractContextManager[None]:
    return use_pipeline("opt")


_FAKE_MODE = FakeTensorMode()


def xpu_tensor(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A fake XPU tensor: generating and lowering XPU IR needs no XPU."""
    with _FAKE_MODE:
        return torch.empty(*shape, dtype=dtype, device="xpu")


def lower_xpu(
    kernel: helion.Kernel, args: list[object], stop_at: str = "xegpu-wg"
) -> str:
    """The IR of ``kernel`` for ``args`` lowered for the XPU to ``stop_at``."""
    return str(lower_mlir(kernel, list(args), target="xpu", stop_at=stop_at))


def execute_module(module: object, *tensors: torch.Tensor, kernel_name: str) -> object:
    """Run a module from ``generate_mlir``."""
    return MLIRBackend().execute_mlir(module, *tensors, kernel_name=kernel_name)


def run_generated(
    kernel: helion.Kernel,
    args: list[object],
    *,
    config: helion.Config | None = None,
) -> object:
    """Run ``kernel`` through the explicit ``generate_mlir``/``execute_mlir`` flow."""
    module = generate_mlir(kernel, list(args), config=config)
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    return execute_module(module, *tensors, kernel_name=kernel.fn.__name__)


def run_direct(
    kernel: helion.Kernel, args: list[object], *, pipeline: str | None = None
) -> object:
    """Run ``kernel`` through the direct ``@helion.kernel(backend="mlir")`` call path."""
    with use_pipeline(pipeline):
        return kernel(*args)


def check_kernel(
    kernel: helion.Kernel,
    reference: Callable[..., object],
    args: list[object],
    *,
    paths: tuple[str, ...] = ("direct",),
    config: helion.Config | None = None,
    pipeline: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Assert that ``kernel(*args)`` matches ``reference(*args)`` on every requested path.

    ``reference`` runs on clones so in-place kernels and references cannot see each
    other's writes. ``pipeline`` overrides the default lighthouse pipeline.
    """
    expected = reference(*_clone_args(args))
    for path in paths:
        with use_pipeline(pipeline):
            if path == "direct":
                actual = run_direct(kernel, _clone_args(args))
            elif path == "generated":
                actual = run_generated(kernel, _clone_args(args), config=config)
            else:
                raise ValueError(f"unknown execution path {path!r}")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=path)


def _clone_args(args: list[object]) -> list[object]:
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def blocked_matmul_cases() -> dict[str, tuple[helion.Kernel, list[object]]]:
    """Production bf16 blocked-matmul kernels from ``helion_mlir_utils`` with inputs."""
    import importlib

    cpu_matmul = importlib.import_module("helion_mlir_utils.matmul")

    def bf16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16)

    vector = [torch.randn(2, 1, 32) for _ in range(3)]
    return {
        "blocked_matmul": (
            cpu_matmul._matmul_blocked_kernel,
            [bf16(2, 2, 32, 32), bf16(2, 2, 32, 32), cpu_matmul.identity_epilogue],
        ),
        "blocked_matmul_bias": (
            cpu_matmul._matmul_blocked_kernel_bias,
            [
                bf16(2, 2, 32, 32),
                bf16(2, 2, 32, 32),
                vector[0],
                cpu_matmul.identity_epilogue,
            ],
        ),
        "blocked_matmul_affine": (
            cpu_matmul._matmul_blocked_kernel_affine,
            [
                bf16(2, 2, 32, 32),
                bf16(2, 2, 32, 32),
                *vector,
                cpu_matmul.identity_epilogue,
            ],
        ),
    }
