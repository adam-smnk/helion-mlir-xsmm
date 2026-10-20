"""Call path of a compiled kernel: host code, argument binding, entry call, return.

Every call runs the kernel's host code up to its device loops (``host_code.py``),
evaluates each entry argument's host expression in the host locals, calls the
memref-ABI entry, then finishes the host code and returns its return value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ..execution import compile_entry
from .codegen import MLIRModuleBuilder
from .host_code import build_host_function
from .support import UnsupportedOperationError
from .support.debug import PIPELINE_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

    from ..execution import CompiledEntry
    from ..execution import EntryArg


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
    pipeline: str | None = None,
) -> Callable[..., object]:
    """A callable with the kernel's own signature and Helion call semantics.

    ``pipeline`` defaults to the config's ``mlir_pipeline``, then
    ``HELION_MLIR_PIPELINE``, then ``opt``.
    """
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
    entry = compile_entry(module, hf.name, pipeline=pipeline)
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

    def run(*args: object) -> object:
        host = host_function(*args)
        try:
            local_vars = next(host)
            call_entry(
                entry,
                [eval(expr, host_globals, local_vars) for expr in arg_exprs],
                symbolic_sizes,
            )
            next(host)
        except StopIteration as done:
            return done.value
        raise AssertionError("the host function yields once")

    return run


def call_entry(
    entry: CompiledEntry,
    values: list[object],
    sizes: dict[str, list[sympy.Expr]] | None = None,
) -> list[torch.Tensor]:
    """Call ``entry`` with one value per argument; returns the inout tensors.

    Tensors are passed contiguous (a copy for strided ones, copied back for inouts).
    An input that overlaps an inout is cloned so the ``restrict`` arguments hold;
    overlapping inouts are rejected. ``sizes`` (each argument's size expressions)
    lets runtime sizes the kernel assumes equal be checked.
    """
    tensors = [
        _checked_tensor(arg, value)
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
        tensor.clone(memory_format=torch.contiguous_format)
        if role == "in" and _overlaps_any(tensor, inout_spans)
        else tensor.contiguous()
        for tensor, role in zip(tensors, roles, strict=True)
    ]
    scalars = [
        torch.tensor(value, dtype=arg.dtype)
        for arg, value in zip(entry.args, values, strict=True)
        if arg.role == "scalar"
    ]
    entry([*staged, *scalars])
    for tensor, copy, role in zip(tensors, staged, roles, strict=True):
        if role == "inout" and copy is not tensor:
            tensor.copy_(copy)
    return inouts


def _checked_tensor(arg: EntryArg, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"'{arg.name}' must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.device.type != "cpu":
        raise NotImplementedError(
            f"'{arg.name}' is on {value.device}; only CPU is supported"
        )
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
