"""Public MLIR API for the external Helion MLIR backend package."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    import mlir.ir as ir


def generate_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
) -> ir.Module:
    """Lower a Helion kernel to an MLIR Linalg-on-Tensors module.

    The module holds one private tensor function per phase and a public
    memref-ABI entry function named after the kernel.
    """
    from helion._compiler.backend_registry import get_backend_class

    hf, config, env = _compile(kernel, args, config)
    with env:
        return get_backend_class("mlir")().generate_mlir(hf, config, env)


def compile_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
    pipeline: str | None = None,
) -> Callable[..., object]:
    """Compile a Helion kernel for ``args``' shapes and dtypes into a callable.

    The callable has the kernel's signature and Helion semantics: it runs the
    kernel's host code on each call and returns the kernel's return value.
    ``pipeline`` is ``"opt"`` or ``"scalar"`` (default: ``HELION_MLIR_PIPELINE``,
    else ``"opt"``).
    """
    from helion_mlir_backend._compiler.mlir.driver import compile_kernel

    hf, config, env = _compile(kernel, args, config)
    return compile_kernel(hf, config, env, pipeline=pipeline)


def _compile(
    kernel: object, args: list[object], config: object | None
) -> tuple[HostFunction, object, CompileEnvironment]:
    try:
        import mlir.ir  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mlir-python-bindings is required for MLIR code generation. "
            "Install it with: pip install mlir-python-bindings"
        ) from exc

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings
    import torch

    fn = getattr(kernel, "fn", None)
    if fn is None:
        raise ValueError(
            f"Expected a @helion.kernel-decorated function; got {type(kernel).__name__}"
        )

    settings: Settings = getattr(kernel, "settings", Settings())
    settings = dataclasses.replace(settings, backend="mlir")

    if config is None:
        # `@helion.kernel(config=...)` is normalized into `Kernel.configs`.
        kernel_configs = getattr(kernel, "configs", None)
        if kernel_configs:
            config = kernel_configs[0]

    device = next(
        (arg.device for arg in args if isinstance(arg, torch.Tensor)),
        torch.device("cpu"),
    )
    env = CompileEnvironment(device, settings)

    with env:
        fake_args = [
            env.to_fake(arg, ArgumentOrigin(name))
            for name, arg in zip(kernel.signature.parameters, args, strict=False)
        ]
        host_function = KernelCompiler(env).compile(fn, fake_args, {})
        if config is None:
            config = env.config_spec.default_config()
    return host_function, config, env
