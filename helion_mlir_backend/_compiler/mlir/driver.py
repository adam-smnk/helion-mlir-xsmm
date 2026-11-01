"""Call path of a compiled kernel: host code, argument binding, entry call, return.

Every call runs the kernel's host code up to its device loops (``host_code.py``),
evaluates each entry argument's host expression in the host locals, calls the
memref-ABI entry (runtime scalars by value), then finishes the host code and returns its return value.
"""

from __future__ import annotations

import keyword
from typing import TYPE_CHECKING

import sympy
import torch

from .. import native
from ..execution import compile_entry
from ..targets import resolve_target
from .codegen import MLIRModuleBuilder
from .host_code import build_host_function
from .support import UnsupportedOperationError
from .support.debug import PIPELINE_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

    from ..execution import CompiledEntry
    from ..execution import EntryArg
    from ..targets import MLIRTarget


def mlir_compile_config(
    bound_kernel: BoundKernel,
    config: Config | None = None,
    *,
    allow_print: bool = True,
) -> Callable[..., object]:
    """``BoundKernel.compile_config`` for the MLIR backend (one module per config)."""
    if config is None:
        config = bound_kernel._require_implicit_config()
    config = bound_kernel._normalize_config(config)
    if (run := bound_kernel._compile_cache.get(config)) is not None:
        return run
    run = compile_kernel(bound_kernel.host_function, config, bound_kernel.env)
    bound_kernel._compile_cache[config] = run
    return run


def compile_kernel(
    hf: HostFunction,
    config: Config,
    env: CompileEnvironment,
    *,
    target: str | None = None,
    pipeline: str | None = None,
) -> Callable[..., object]:
    """A callable with the kernel's own signature and Helion call semantics.

    ``target`` names the device target (default: ``HELION_MLIR_TARGET``, else
    the one for ``env.device``). ``pipeline`` defaults to the config's
    ``mlir_pipeline``, then the target's default (``HELION_MLIR_PIPELINE``, else
    ``opt`` on the CPU). The target's config keys (e.g. ``xegpu_*`` on the XPU)
    apply to its lowering; other targets ignore them.
    """
    mlir_target = resolve_target(env.device, target)
    if pipeline is None:
        pipeline = config.get(PIPELINE_CONFIG_KEY)
    with env:
        builder = MLIRModuleBuilder(hf, config, env)
        module = builder.build()
        host_function = build_host_function(
            hf, builder.context.geometry.block_size, config
        )
        sizes = {
            name: builder.context.sizes.ref(name)
            for name in builder.context.signature.refs
        }
    mlir_target.configure(module, config)
    entry = compile_entry(module, hf.name, target=mlir_target, pipeline=pipeline)
    host_globals = host_function.__globals__
    arg_exprs = [
        compile(arg.name, f"<helion-mlir-arg:{arg.name}>", "eval") for arg in entry.args
    ]
    # Only sizes computed from symbols need checks at run time.
    symbolic_sizes = (
        sizes
        if any(expr.free_symbols for exprs in sizes.values() for expr in exprs)
        else None
    )

    # An argument named by a host local is looked up; others are evaluated.
    local_names = [
        arg.name
        if arg.name.isidentifier() and not keyword.iskeyword(arg.name)
        else None
        for arg in entry.args
    ]
    arg_sources = list(zip(local_names, arg_exprs, strict=True))

    def run(*args: object) -> object:
        host = host_function(*args)
        try:
            local_vars = next(host)
            call_entry(
                entry,
                [
                    local_vars[name]
                    if name is not None and name in local_vars
                    else eval(expr, host_globals, local_vars)
                    for name, expr in arg_sources
                ],
                symbolic_sizes,
            )
            next(host)
        except StopIteration as done:
            return done.value
        raise AssertionError("the host function yields once")

    return _native_run(run, entry, host_function, arg_sources, symbolic_sizes)


def _native_run(
    run: Callable[..., object],
    entry: CompiledEntry,
    host_function: Callable[..., object],
    arg_sources: list[tuple[str | None, object]],
    symbolic_sizes: object,
) -> Callable[..., object]:
    """``run``, or a native ``KernelRun`` in front of it if the kernel
    qualifies: its first call records a plan (:func:`_plan`); later calls
    whose arguments match it run natively, all others take ``run``. The first
    call decides: one that can't be planned leaves every call on ``run``."""
    native_call = getattr(entry, "_native", None)
    if (
        native_call is None
        or symbolic_sizes is not None
        or not getattr(host_function, "_helion_mlir_metadata_only", False)
        or any(name is None for name, _ in arg_sources)
        or (extension := native.module()) is None
    ):
        return run
    names = [name for name, _ in arg_sources]
    defaults = (
        *(host_function.__defaults__ or ()),
        *(host_function.__kwdefaults__ or {}).values(),
    )

    def planner(native_run: object, *args: object) -> object:
        return _plan(
            native_run, run, entry, native_call, host_function, names, defaults, args
        )

    return extension.KernelRun(run, planner)


_ScalarTypes = (int, float, bool, str, type(None))


def _plan(
    native_run: object,
    run: Callable[..., object],
    entry: CompiledEntry,
    native_call: object,
    host_function: Callable[..., object],
    names: list[str],
    defaults: tuple[object, ...],
    args: tuple[object, ...],
) -> object:
    """The first call: as ``run``, recording which argument or fresh host
    tensor each entry argument is and what is returned. Installs that plan in
    ``native_run`` if each entry argument and the result is exactly one
    argument (no two arguments the same object) or a fresh contiguous CPU
    tensor (not a default argument), tensor arguments are plain CPU tensors
    and other arguments immutable scalars, scalar ``hl.constexpr`` values or
    functions. Without a plan, every call takes ``run``."""
    try:
        params = _plan_params(args)
    except Exception:
        params = None
    if params is None:
        return run(*args)
    host = host_function(*args)
    try:
        local_vars = next(host)
        if not all(name in local_vars for name in names):
            # Not a host local after all (run evaluates it): no plan.
            host.close()
            return run(*args)
        values = [local_vars[name] for name in names]
        call_entry(entry, values)
        next(host)
    except StopIteration as done:
        result = done.value
    else:
        raise AssertionError("the host function yields once")
    plan = _plan_sources(args, values, result, defaults)
    if plan is not None:
        from ..targets.base import MLIRTarget

        hooks = (type(entry.target), MLIRTarget.prepare_call, MLIRTarget.finish_call)
        native_run.install(native_call, hooks, params, *plan)  # type: ignore[attr-defined]
    return result


def _plan_params(args: tuple[object, ...]) -> list[tuple[object, ...]] | None:
    """Per argument ``(kind, dtype or value, sizes, strides)``: 0 a plain CPU
    tensor, 1 an object compared by identity (a builtin function), 2 a scalar
    (type and ``==``), 3 an ``hl.constexpr`` of a scalar (its type, its
    value's type and ``==``: a helper may make a new one per call), 4 a
    function ``(4, function, cell kinds, cell contents)``: its code and its
    closure's contents when planned, each as 1-3 (as Helion keys functions:
    another instance of a model has its own epilogue closures); ``None``: no
    plan."""
    import types

    from helion.language.constexpr import ConstExpr

    params: list[tuple[object, ...]] = []
    for arg in args:
        if type(arg) in (torch.Tensor, torch.nn.Parameter):
            tensor: torch.Tensor = arg  # type: ignore[assignment]
            if tensor.device.type != "cpu" or tensor.layout != torch.strided:
                return None
            params.append((0, tensor.dtype, tuple(tensor.shape), tensor.stride()))
        elif type(arg) in _ScalarTypes:
            if arg != arg:  # NaN never matches: no plan
                return None
            params.append((2, arg, (), ()))
        elif type(arg) is ConstExpr and type(arg.value) in _ScalarTypes:  # type: ignore[attr-defined]
            if arg.value != arg.value:  # type: ignore[attr-defined]
                return None
            params.append((3, arg, (), ()))
        elif isinstance(arg, types.FunctionType):
            try:
                contents = tuple(cell.cell_contents for cell in arg.__closure__ or ())
            except ValueError:  # an empty cell
                return None
            params.append((4, arg, tuple(map(_value_kind, contents)), contents))
        elif isinstance(arg, types.BuiltinFunctionType):
            params.append((1, arg, (), ()))
        else:
            return None
    return params


def _value_kind(value: object) -> int:
    """How a closure cell's content is compared (as :func:`_plan_params`)."""
    from helion.language.constexpr import ConstExpr

    if type(value) in _ScalarTypes and value == value:
        return 2
    if type(value) is ConstExpr and type(value.value) in _ScalarTypes:  # type: ignore[attr-defined]
        return 3 if value.value == value.value else 1  # type: ignore[attr-defined]
    return 1


def _plan_sources(
    args: tuple[object, ...],
    values: list[object],
    result: object,
    defaults: tuple[object, ...],
) -> (
    tuple[
        list[tuple[int, int]],
        list[tuple[torch.dtype, tuple[int, ...]]],
        tuple[int, int],
    ]
    | None
):
    """``(sources, fresh, result)`` for ``KernelRun.install``: per entry
    argument and for the result, ``(0, argument index)`` or ``(1, fresh
    index)`` (``(0, -1)``: returns None); ``None``: no plan."""
    from ..execution import _contiguous_strides

    fresh: list[torch.Tensor] = []
    arg_storages = {
        arg.untyped_storage().data_ptr()
        for arg in args
        if isinstance(arg, torch.Tensor)
    }

    def source(value: object) -> tuple[int, int] | None:
        matches = [index for index, arg in enumerate(args) if value is arg]
        if matches:
            # The same object for two arguments: a later call may pass two.
            return (0, matches[0]) if len(matches) == 1 else None
        for index, tensor in enumerate(fresh):
            if value is tensor:
                return (1, index)
        if (
            type(value) is torch.Tensor
            and not any(value is default for default in defaults)
            and value.device.type == "cpu"
            and value.layout == torch.strided
            # The strides at::empty gives (torch's contiguity ignores the
            # strides of size-1 dims).
            and value.stride() == _contiguous_strides(value.shape)
            and value.storage_offset() == 0
            and value.untyped_storage().data_ptr() not in arg_storages
            and not value.requires_grad
        ):
            fresh.append(value)
            return (1, len(fresh) - 1)
        return None

    sources = [source(value) for value in values]
    result_source = (0, -1) if result is None else source(result)
    if result_source is None or any(s is None for s in sources):
        return None
    return (
        sources,  # type: ignore[return-value]
        [(t.dtype, tuple(t.shape)) for t in fresh],
        result_source,
    )


def call_entry(
    entry: CompiledEntry,
    values: list[object],
    sizes: dict[str, list[sympy.Expr]] | None = None,
) -> list[torch.Tensor]:
    """Call ``entry`` with one value per argument; returns the inout tensors.

    The entry's target checks each tensor (e.g. its device) and prepares and
    finishes the call. Tensors are passed contiguous (a copy for strided ones,
    copied back for inouts) and with their data aligned to the target's
    ``buffer_alignment`` (a copy likewise). An input that overlaps an inout is
    cloned so the ``restrict`` arguments hold; overlapping inouts are rejected. Runtime scalars
    are passed by value. ``sizes`` (each argument's size expressions) lets runtime
    sizes the kernel assumes equal be checked. Calls that need none of this
    (:meth:`CompiledEntry.call_in_place`) skip it: per-call host time is a large
    part of a small kernel's time.
    """
    if sizes is None and (inouts := entry.call_in_place(values)) is not None:
        return inouts
    target = entry.target
    tensors = [
        _checked_tensor(target, arg, value)
        for arg, value in zip(entry.args, values, strict=True)
        if arg.role != "scalar"
    ]
    if sizes is not None:
        _check_sizes(
            [
                (arg.name, value)
                for arg, value in zip(entry.args, values, strict=True)
                if arg.role != "scalar"
            ],
            sizes,
        )
    roles = [arg.role for arg in entry.args if arg.role != "scalar"]
    inouts = [t for t, role in zip(tensors, roles, strict=True) if role == "inout"]
    inout_spans = [_byte_span(t) for t in inouts if t.numel()]
    for i, first in enumerate(inout_spans):
        if any(_spans_overlap(first, second) for second in inout_spans[i + 1 :]):
            raise UnsupportedOperationError(
                "kernel writes two tensors that share memory",
                reason="each written tensor must be its own buffer",
            )
    staged = [
        _aligned(
            # A negative or conjugate view's storage holds the values unnegated
            # (unconjugated): the kernel gets a materialized copy (written back
            # for inouts), which never aliases an inout.
            tensor.resolve_conj().resolve_neg().contiguous()
            if tensor.is_neg() or tensor.is_conj()
            else tensor.clone(memory_format=torch.contiguous_format)
            if role == "in" and _overlaps_any(tensor, inout_spans)
            else tensor.contiguous(),
            target.buffer_alignment,
        )
        for tensor, role in zip(tensors, roles, strict=True)
    ]
    buffers = iter(staged)
    arguments = [
        value if arg.role == "scalar" else next(buffers)
        for arg, value in zip(entry.args, values, strict=True)
    ]
    target.prepare_call(staged)
    entry(arguments)
    target.finish_call(staged)
    for tensor, copy, role in zip(tensors, staged, roles, strict=True):
        if role == "inout" and copy is not tensor:
            tensor.copy_(copy)
    return inouts


def _aligned(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """``tensor``, or a copy of it whose data starts at a multiple of
    ``alignment`` bytes (the caching allocators return aligned blocks)."""
    if alignment <= 1 or not tensor.numel() or tensor.data_ptr() % alignment == 0:
        return tensor
    copy = tensor.clone(memory_format=torch.contiguous_format)
    if copy.data_ptr() % alignment:
        raise UnsupportedOperationError(
            f"cannot pass a {alignment}-byte aligned copy of a tensor",
            reason=f"the allocator returned data at {copy.data_ptr():#x}",
        )
    return copy


def _checked_tensor(target: MLIRTarget, arg: EntryArg, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"'{arg.name}' must be a torch.Tensor, got {type(value).__name__}"
        )
    target.check_tensor(arg.name, value)
    if (
        value.dim() != len(arg.shape)
        or value.dtype != arg.dtype
        or any(
            expected != -1 and expected != actual
            for expected, actual in zip(arg.shape, value.shape, strict=False)
        )
    ):
        expected = tuple("?" if size == -1 else size for size in arg.shape)
        raise ValueError(
            f"'{arg.name}' has shape {tuple(value.shape)} and dtype {value.dtype}; the "
            f"compiled kernel expects {expected} and {arg.dtype}"
        )
    return value


def _check_sizes(
    tensors: list[tuple[str, torch.Tensor]], sizes: dict[str, list[sympy.Expr]]
) -> None:
    """Runtime sizes with one symbol are equal, and sizes computed from symbols
    (``n // 2``) match them."""
    bound: dict[sympy.Symbol, int] = {}
    computed = []
    for name, tensor in tensors:
        for dim, expr in enumerate(sizes.get(name, ())):
            if not expr.free_symbols:
                continue
            actual = tensor.shape[dim]
            if isinstance(expr, sympy.Symbol):
                expected = bound.setdefault(expr, actual)
                if expected != actual:
                    raise ValueError(
                        f"'{name}' has size {actual} in dim {dim}, but the kernel was "
                        f"compiled for the same size as another argument's ({expected})"
                    )
            else:
                computed.append((name, dim, expr, actual))
    for name, dim, expr, actual in computed:
        if expr.free_symbols <= bound.keys():
            expected = int(expr.xreplace(bound))
            if expected != actual:
                raise ValueError(
                    f"'{name}' has size {actual} in dim {dim}; the kernel computes "
                    f"it as {expr} = {expected}"
                )


def _overlaps_any(tensor: torch.Tensor, spans: list[tuple[int, int]]) -> bool:
    if not spans or tensor.numel() == 0:
        return False
    span = _byte_span(tensor)
    return any(_spans_overlap(span, other) for other in spans)


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """The bytes ``t`` spans in memory; ``t`` is not empty."""
    start = t.data_ptr()
    if t.is_contiguous():
        return start, start + t.numel() * t.element_size()
    last = sum(
        (size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True)
    )
    return start, start + (last + 1) * t.element_size()


def prepared_call(
    bound_kernel: BoundKernel,
    args: Sequence[object],
    varying: Sequence[int],
    fallback: Callable[..., object],
    *,
    views: Sequence[Sequence[int] | None] | None = None,
    result_shape: Sequence[int] | None = None,
) -> Callable[..., object] | None:
    """``lambda *v: bound_kernel(*args with v at the varying positions)``,
    natively: a call checks the varying tensors (and everything else the
    native run checks: the bound tensors' metadata, their current data, that
    ``bound_kernel`` still runs the same compiled kernel) and calls the
    kernel; any other call returns ``fallback(*v)``. ``None`` if the kernel
    has no native run (not static, host code not metadata-only, no
    extension).

    ``views[k]`` (optional, per varying argument): the caller passes a
    contiguous tensor of that shape, standing for its contiguous view of the
    argument's shape (e.g. ``x`` for ``x.view(m, k // 2, 2)``).
    ``result_shape``: the result, a fresh tensor of the host code, is
    allocated with that shape (as ``result.view(result_shape)``). Neither may
    change the element count. The first native call happens after the kernel
    ran once (its plan is recorded on its first call)."""
    if bound_kernel._run is None:
        bound_kernel.ensure_config_exists(tuple(args))
    run = bound_kernel._run
    if native.module() is None or type(run).__name__ != "KernelRun":
        return None
    varying = list(varying)
    bound = tuple(None if i in varying else arg for i, arg in enumerate(args))
    return run.prepare(  # type: ignore[union-attr]
        bound_kernel,
        bound,
        varying,
        [None if v is None else tuple(v) for v in views]
        if views is not None
        else [None] * len(varying),
        None if result_shape is None else tuple(result_shape),
        fallback,
    )
