"""Runtime backend registration helpers for external Helion MLIR backend."""

from __future__ import annotations

import logging
import sys
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
    _key_function_args_by_globals_and_defaults(MLIRBackend)
    _check_module_lookups(MLIRBackend)
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


class _SameObject:
    """Key part that matches only the very object it holds, unhashable or not.
    Holding it keeps its id from being reused while the key lives."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SameObject) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)

    def __repr__(self) -> str:
        return f"<{type(self.value).__name__} object>"


def _default_key(value: object) -> object:
    """Defaults are compiled in as constants, so equal values may share a key.
    Floats by their bits: 0.0 == -0.0, and NaN differs from itself."""
    if type(value) is float:
        return float, value.hex()
    if value is None or type(value) in (bool, int, str):
        return type(value), value
    return _SameObject(value)


def _key_function_args_by_globals_and_defaults(MLIRBackend: type) -> None:
    """Keep kernels from reusing a compile for a function that differs from it.

    Helion keys a function argument (an epilogue) by its code object and closure
    cells. Code objects compare by content, ignoring the file, so two modules'
    functions of the same text and line share a key, as do functions of one
    ``def`` with different defaults. The second call then ran the first one's
    compile: other module globals, or the first default, without an error.
    """
    import types

    from helion.runtime.kernel import _specialization_extractors as extractors

    original = extractors[types.FunctionType]
    if getattr(original, "_helion_mlir_globals_and_defaults", False):
        return
    backend_name = MLIRBackend().name

    def function_key(kernel: object, fn: types.FunctionType) -> object:
        key = original(kernel, fn)
        if kernel.settings.backend != backend_name:  # type: ignore[attr-defined]
            return key
        defaults = tuple(map(_default_key, fn.__defaults__ or ()))
        return key, _SameObject(fn.__globals__), defaults

    function_key._helion_mlir_globals_and_defaults = True  # type: ignore[attr-defined]
    extractors[types.FunctionType] = function_key


def _check_module_lookups(MLIRBackend: type) -> None:
    """Refuse a module global when ``sys.modules`` names another module.

    Helion finds a global's module through ``sys.modules``, by name. Its check
    that it found the right one is an ``assert``, gone under ``python -O``, and
    is skipped for a name it already resolved in this kernel; host code then
    reads the other module's value.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion._compiler.output_header import SOURCE_MODULE
    from helion._compiler.type_propagation import LocalScope
    from helion._compiler.variable_origin import AttributeOrigin
    from helion._compiler.variable_origin import GlobalOrigin

    from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

    if getattr(HostFunction, "_helion_mlir_checked_lookups", False):
        return

    def check(scope: dict[str, object], global_name: str) -> None:
        if not isinstance(CompileEnvironment.current().backend, MLIRBackend):
            return
        name = scope["__name__"]
        module = sys.modules.get(name)
        if module is None or vars(module) is not scope:
            raise UnsupportedOperationError(
                f"module global {global_name}",
                reason=f"sys.modules[{name!r}] is "
                f"{'missing' if module is None else 'another module'}; "
                "give each loaded module its own name",
            )

    global_scope_origin = HostFunction.global_scope_origin
    import_from_module = HostFunction.import_from_module
    maybe_get = LocalScope.maybe_get
    probing = [False]

    def checked_global_scope_origin(self: HostFunction, name: str) -> object:
        if probing[0]:
            # Assigning a local looks up its name first, only to compare types: no
            # global is read, and with no module loaded there is nothing to refuse.
            return AttributeOrigin(GlobalOrigin(SOURCE_MODULE), name)
        check(self.fn.__globals__, name)
        return global_scope_origin(self, name)

    def probing_maybe_get(self: LocalScope, name: str) -> object:
        probing[0] = True
        try:
            return maybe_get(self, name)
        finally:
            probing[0] = False

    def checked_import_from_module(
        self: HostFunction, module_scope: dict[str, object], name: str
    ) -> object:
        check(module_scope, name)
        return import_from_module(self, module_scope, name)

    LocalScope.maybe_get = probing_maybe_get
    HostFunction.global_scope_origin = checked_global_scope_origin
    HostFunction.import_from_module = checked_import_from_module
    HostFunction._helion_mlir_checked_lookups = True


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
