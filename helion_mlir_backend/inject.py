"""Runtime backend registration helpers for external Helion MLIR backend."""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


def install() -> bool:
    """Register external MLIR backend into Helion registry.

    Returns True when registration is applied successfully.
    """
    try:
        from helion._compiler.backend_registry import register_compiler_backend
        from helion.runtime.kernel import BoundKernel

        from helion_mlir_backend._compiler.mlir.aten_bridge.original_args import (
            install_original_args_capture,
        )
        from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
        from helion_mlir_backend._compiler.mlir.driver import mlir_compile_config
        from helion_mlir_backend._compiler.mlir.trace_mode import install_trace_mode
    except ImportError as exc:
        log.debug("External MLIR backend registration unavailable: %s", exc)
        return False

    register_compiler_backend(MLIRBackend)
    _patch_bound_kernel(BoundKernel, MLIRBackend, mlir_compile_config)
    _allow_config_keys()
    _register_autotune_cache()
    _accept_mlir_module_globals()
    _keep_runtime_slice_sizes(MLIRBackend)
    install_trace_mode()
    install_original_args_capture()
    return True


def _accept_mlir_module_globals() -> None:
    """Let kernels read ``mlir.ir.Module`` globals (``inline_mlir`` sources).

    Helion rejects globals of unknown types; like Triton ``JITFunction`` globals,
    a module becomes its source text.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    import mlir.ir as ir

    from helion_mlir_backend._compiler.mlir.snippets import source_text

    if getattr(CompileEnvironment, "_helion_mlir_module_globals", False):
        return
    original = CompileEnvironment.to_fake

    def to_fake(self: CompileEnvironment, obj: object, origin: object) -> object:
        if isinstance(obj, ir.Module):
            return source_text(obj)
        return original(self, obj, origin)

    CompileEnvironment.to_fake = to_fake
    CompileEnvironment._helion_mlir_module_globals = True


def _keep_runtime_slice_sizes(MLIRBackend: type) -> None:
    """Give a ``:`` slice of a dim known only at run time a reduction block.

    The MLIR backend does not pad factory shapes to powers of two
    (``MLIRBackend.pad_factory_tensors_to_power_of_2``). On such backends
    Helion sizes ``:`` slices with ``try_concretize_symint``, which turns a
    dim of a ``static_shapes=False`` kernel into the example input's size
    without a guard: ``row / row.size(1)`` then divides by that size for
    every input of the shape bucket. A dim that is a compile-time constant
    (static or ``hl.specialize``) stays that constant; one known only at
    run time gets a reduction block, as with padding, whose size the kernel
    reads at run time.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.compile_environment import _has_unbacked
    import sympy
    import torch

    if getattr(CompileEnvironment, "_helion_mlir_runtime_slice_sizes", False):
        return
    original = CompileEnvironment.try_concretize_symint

    def try_concretize_symint(
        self: CompileEnvironment, size: int | torch.SymInt
    ) -> int | torch.SymInt:
        if not isinstance(self.backend, MLIRBackend) or not isinstance(
            size, torch.SymInt
        ):
            return original(self, size)
        expr = self.shape_env.replace(size._sympy_())
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if _has_unbacked(expr):
            return original(self, size)
        return self.allocate_reduction_dimension(size).var

    CompileEnvironment.try_concretize_symint = try_concretize_symint
    CompileEnvironment._helion_mlir_runtime_slice_sizes = True


def _register_autotune_cache() -> None:
    from helion import autotuner

    from helion_mlir_backend._compiler.mlir.autotune import AUTOTUNE_CACHE
    from helion_mlir_backend._compiler.mlir.autotune import CPU_AUTOTUNE_CACHE
    from helion_mlir_backend._compiler.mlir.autotune import MLIRAutotuneCache

    autotuner.cache_classes[AUTOTUNE_CACHE] = MLIRAutotuneCache
    autotuner.cache_classes[CPU_AUTOTUNE_CACHE] = MLIRAutotuneCache


def _allow_config_keys() -> None:
    """Let configs carry ``mlir_pipeline`` and the targets' lowering keys (e.g.
    ``xegpu_sg_tiles``): Helion accepts only keys in ``VALID_KEYS``. Backends
    that do not support them drop them from configs."""
    import helion.autotuner.config_spec as config_spec

    from helion_mlir_backend._compiler.mlir.support.debug import PIPELINE_CONFIG_KEY
    from helion_mlir_backend._compiler.targets import target_config_keys

    config_spec.VALID_KEYS = (
        config_spec.VALID_KEYS | {PIPELINE_CONFIG_KEY} | target_config_keys()
    )


def _patch_bound_kernel(
    BoundKernel: type,
    MLIRBackend: type,
    mlir_compile_config: Callable[..., object],
) -> None:
    """Patch BoundKernel.compile_config to route MLIR-backend kernels through lighthouse."""
    if getattr(BoundKernel, "_helion_mlir_compile_config_patched", False):
        return

    _original = BoundKernel.compile_config

    def _compile_config(
        self: object,
        config: object = None,
        *,
        allow_print: bool = True,
    ) -> object:
        if isinstance(self.env.backend, MLIRBackend):
            return mlir_compile_config(self, config, allow_print=allow_print)
        return _original(self, config, allow_print=allow_print)

    BoundKernel.compile_config = _compile_config
    BoundKernel._helion_mlir_compile_config_patched = True
