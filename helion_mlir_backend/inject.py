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
    _allow_pipeline_config_key()
    _register_cpu_autotune_cache()
    _accept_mlir_module_globals()
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


def _register_cpu_autotune_cache() -> None:
    from helion import autotuner

    from helion_mlir_backend._compiler.mlir.autotune import CPU_AUTOTUNE_CACHE
    from helion_mlir_backend._compiler.mlir.autotune import CpuAutotuneCache

    autotuner.cache_classes[CPU_AUTOTUNE_CACHE] = CpuAutotuneCache


def _allow_pipeline_config_key() -> None:
    """Let configs carry ``mlir_pipeline``: Helion accepts only keys in ``VALID_KEYS``."""
    import helion.autotuner.config_spec as config_spec

    from helion_mlir_backend._compiler.mlir.support.debug import PIPELINE_CONFIG_KEY

    config_spec.VALID_KEYS = config_spec.VALID_KEYS | {PIPELINE_CONFIG_KEY}


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
