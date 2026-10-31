"""Lower a generated module with lighthouse and JIT its memref-ABI entry function.

:func:`compile_entry` returns a :class:`CompiledEntry` that takes the entry's
arguments in order (tensors for memrefs, Python numbers for runtime scalars,
passed by value); :func:`lower_entry` only lowers, possibly to a stage before
the end of the target's pipeline (IR tests). The device target
(``targets/``) supplies the lighthouse lowering and the JIT runner. Failures keep
their exception type and get a note naming the stage. Lighthouse lowering runs in a forked process, killed after
the compile timeout; only the lowered module comes back. Compiled entries are
cached in-process by module text, target and pipeline, so equal modules (e.g.
configs that clamp to the same tiles) JIT once.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import io
import logging
import math
import multiprocessing
import os
import pickle
import tempfile
import threading
from typing import TYPE_CHECKING
import warnings

import mlir.ir as ir
from mlir.passmanager import PassManager
import torch

from helion_mlir_backend._compiler.mlir.in_place import update_carried_values_in_place
from helion_mlir_backend._compiler.mlir.support.debug import DebugOptions
from helion_mlir_backend._compiler.mlir.support.debug import compile_timeout
from helion_mlir_backend._compiler.mlir.support.errors import CompileTimeoutError
from helion_mlir_backend._compiler.mlir.support.errors import LoweringStoppedError
from helion_mlir_backend._compiler.mlir.support.type_utils import mlir_dtype_to_torch
from helion_mlir_backend._compiler.targets import resolve_target
from helion_mlir_backend._compiler.targets.base import MLIRTarget

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    from multiprocessing.connection import Connection
    from typing import BinaryIO

    from lighthouse.execution.runner import Runner

    from helion_mlir_backend._compiler.targets.base import Lowering

log = logging.getLogger(__name__)

_JIT_CACHE: OrderedDict[tuple[str, str, str], CompiledEntry] = OrderedDict()
_JIT_CACHE_SIZE = 128


def _dump_if(enabled: bool, label: str, module: ir.Module) -> None:
    if enabled:
        print(f"=== {label} ===", flush=True)
        print(module, flush=True)


def inline_module(module: ir.Module) -> ir.Module:
    """Inline phase and helper functions into the entry and update loop-carried
    values in place."""
    with module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(inline,canonicalize)").run(module.operation)
        update_carried_values_in_place(module)
        PassManager.parse("builtin.module(canonicalize)").run(module.operation)
    return module


@dataclass(frozen=True)
class EntryArg:
    """One argument of the entry function, from its ``helion.*`` argument attributes."""

    name: str
    """The host expression that produces the argument."""
    role: str
    """``in``, ``inout`` or ``scalar``."""
    tensor_param: int | None
    shape: tuple[int, ...]
    """``-1`` for a size only known at run time; ``()`` for a scalar."""
    dtype: torch.dtype


@dataclass(frozen=True)
class CompiledEntry:
    name: str
    args: tuple[EntryArg, ...]
    runner: Runner
    target: MLIRTarget
    """The target compiled for: its checks and hooks apply to every call."""

    def __post_init__(self) -> None:
        # The ctypes callable of the packed C interface, looked up once.
        object.__setattr__(self, "_func", self.runner.engine.lookup(self.name))
        object.__setattr__(self, "_static", _StaticCall.build(self.args))

    def call_in_place(self, values: Sequence[object]) -> list[torch.Tensor] | None:
        """Call the entry and return its inout tensors if every argument is a
        tensor that it can take as is -- the static shape and dtype, contiguous
        strides, no negative or conjugate bit,
        :meth:`MLIRTarget.passes_in_place` and a target without call
        hooks (:meth:`MLIRTarget.prepare_call`), no inout sharing memory
        with another argument -- else return ``None`` without calling it: the
        caller then takes the general path (``driver.call_entry``), which
        stages, checks and reports. Only the data pointers change between
        calls, so the arguments are packed once per entry."""
        static = self._static
        if static is None:
            return None
        # Checked per call, not once: entries are shared through the JIT
        # cache, and a target's hooks may be replaced after one was built.
        hooks = type(self.target)
        if (
            hooks.prepare_call is not MLIRTarget.prepare_call
            or hooks.finish_call is not MLIRTarget.finish_call
        ):
            return None
        passes = self.target.passes_in_place
        for value, shape, strides, dtype in zip(
            values, static.shapes, static.strides, static.dtypes, strict=True
        ):
            if not (
                isinstance(value, torch.Tensor)
                and value.dtype is dtype
                and value.shape == shape
                and value.stride() == strides
                and not value.is_neg()
                and not value.is_conj()
                and passes(value)
            ):
                return None
        pointers = [value.data_ptr() for value in values]  # type: ignore[union-attr]
        nbytes = static.nbytes
        for i in static.inouts:
            start, end = pointers[i], pointers[i] + nbytes[i]
            for j, other in enumerate(pointers):
                if j != i and other < end and start < other + nbytes[j]:
                    return None
        # One packed buffer per entry: a call from another thread meanwhile
        # takes the general path.
        if not static.lock.acquire(blocking=False):
            return None
        try:
            buffer = static.buffer
            for slot, pointer in zip(static.slots, pointers, strict=True):
                buffer[slot] = pointer
                buffer[slot + 1] = pointer
            self._func(static.packed)
        except Exception as exc:
            exc.add_note(f"Helion MLIR backend: failed while running '{self.name}'")
            raise
        finally:
            static.lock.release()
        return [values[i] for i in static.inouts]  # type: ignore[misc]

    def __call__(self, values: Sequence[torch.Tensor | float | int | bool]) -> None:
        """Call the entry with one value per argument: a tensor for a memref, a
        Python number for a runtime scalar."""
        packed, _storage = _packed_args(self.args, values)
        try:
            self._func(packed)
        except Exception as exc:
            exc.add_note(f"Helion MLIR backend: failed while running '{self.name}'")
            raise


_WORD = ctypes.sizeof(ctypes.c_int64)


class _StaticCall:
    """The packed arguments of an entry that takes only memrefs of static,
    non-empty shapes, built once: the descriptors as :func:`_packed_args`
    builds them for contiguous tensors, with the data pointers (``allocated``
    and ``aligned``) filled in per call."""

    def __init__(self, args: Sequence[EntryArg]) -> None:
        self.shapes = tuple(torch.Size(arg.shape) for arg in args)
        self.strides = tuple(_contiguous_strides(arg.shape) for arg in args)
        self.dtypes = tuple(arg.dtype for arg in args)
        self.nbytes = tuple(math.prod(arg.shape) * arg.dtype.itemsize for arg in args)
        self.inouts = tuple(i for i, arg in enumerate(args) if arg.role == "inout")
        words: list[int] = []
        slots: list[int] = []
        for shape, strides in zip(self.shapes, self.strides, strict=True):
            slots.append(len(words))
            words += (0, 0, 0, *shape, *strides)
        self.slots = tuple(slots)
        descriptors = len(words)
        self.buffer = (ctypes.c_int64 * (descriptors + 2 * len(args)))()
        base = ctypes.addressof(self.buffer)
        words += [base + _WORD * slot for slot in slots]
        words += [base + _WORD * (descriptors + i) for i in range(len(args))]
        self.buffer[:] = words
        self.packed = base + _WORD * (descriptors + len(args))
        self.lock = threading.Lock()

    @classmethod
    def build(cls, args: Sequence[EntryArg]) -> _StaticCall | None:
        if not args or any(
            arg.role == "scalar" or -1 in arg.shape or 0 in arg.shape for arg in args
        ):
            return None
        return cls(args)


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """The strides torch gives a contiguous tensor of ``shape``."""
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(size, 1)
    return tuple(reversed(strides))


# The C type of each runtime scalar type the entry takes by value, and the
# Python type its values are converted to (as ``torch.tensor(value, dtype)`` did).
_SCALAR_CTYPES: dict[torch.dtype, tuple[type, type]] = {
    torch.float64: (ctypes.c_double, float),
    torch.float32: (ctypes.c_float, float),
    torch.int64: (ctypes.c_int64, int),
    torch.int32: (ctypes.c_int32, int),
    torch.bool: (ctypes.c_bool, bool),
}


def _packed_args(
    args: Sequence[EntryArg], values: Sequence[torch.Tensor | float | int | bool]
) -> tuple[int, tuple[object, ...]]:
    """The address of the packed arguments of a C-interface entry -- one pointer
    per argument: to a pointer to its memref descriptor, or to the value of a
    runtime scalar -- and the storage they point to, which must outlive the call.

    The descriptors (``allocated``, ``aligned``, ``offset``, sizes, strides), the
    descriptor pointers and the packed arguments are all 64-bit words of one
    buffer; each scalar is a C value of its own type.
    """
    words: list[int] = []
    starts: list[int] = []
    scalars: dict[int, ctypes._SimpleCData] = {}
    for index, (arg, value) in enumerate(zip(args, values, strict=True)):
        if arg.role == "scalar":
            c_type, python_type = _SCALAR_CTYPES[arg.dtype]
            scalars[index] = c_type(python_type(value))
            continue
        starts.append(len(words))
        data = value.data_ptr()
        words += (data, data, 0, *value.shape, *value.stride())
    pointers = len(words)
    buffer = (ctypes.c_int64 * (pointers + len(starts) + len(args)))()
    base = ctypes.addressof(buffer)
    words += [base + _WORD * start for start in starts]
    descriptor = iter(range(pointers, pointers + len(starts)))
    words += [
        ctypes.addressof(scalars[index])
        if index in scalars
        else base + _WORD * next(descriptor)
        for index in range(len(args))
    ]
    buffer[:] = words
    return base + _WORD * (pointers + len(starts)), (buffer, scalars)


def entry_args(module: ir.Module, entry: str) -> tuple[EntryArg, ...]:
    """The entry function's arguments, read before lowering."""
    fn = next(
        (
            op
            for op in module.body.operations
            if "sym_name" in op.attributes
            and ir.StringAttr(op.attributes["sym_name"]).value == entry
        ),
        None,
    )
    if fn is None:
        raise ValueError(f"no function named '{entry}' in the module")
    arg_types = ir.FunctionType(
        ir.TypeAttr(fn.attributes["function_type"]).value
    ).inputs
    arg_attrs = fn.attributes.get("arg_attrs")
    if arg_attrs is None:
        raise ValueError(f"'{entry}' is not a Helion MLIR entry function")
    args = []
    for arg_type, attrs in zip(arg_types, ir.ArrayAttr(arg_attrs), strict=True):
        attrs = ir.DictAttr(attrs)
        role = ir.StringAttr(attrs["helion.role"]).value
        if role == "scalar":
            shape: tuple[int, ...] = ()
            element_type = arg_type
        else:
            memref = ir.MemRefType(arg_type)
            shape = tuple(
                -1 if ir.ShapedType.is_dynamic_size(size) else size
                for size in memref.shape
            )
            element_type = memref.element_type
        dtype = mlir_dtype_to_torch(str(element_type))
        if dtype is None or (role == "scalar" and dtype not in _SCALAR_CTYPES):
            kind = "scalar" if role == "scalar" else "memref"
            raise ValueError(f"'{entry}' takes a {kind} of unsupported {element_type}")
        args.append(
            EntryArg(
                ir.StringAttr(attrs["helion.name"]).value,
                role,
                ir.IntegerAttr(attrs["helion.param"]).value
                if "helion.param" in attrs
                else None,
                shape,
                dtype,
            )
        )
    return tuple(args)


def compile_entry(
    module: ir.Module,
    entry: str,
    *,
    target: MLIRTarget | None = None,
    pipeline: str | None = None,
) -> CompiledEntry:
    """Inline, lower and JIT-compile ``entry`` (consumes ``module``) for ``target``
    (default: :func:`resolve_target` for the CPU), or reuse the entry compiled from
    an identical module for the same target and pipeline. ``pipeline`` defaults to
    the target's. A target whose lowerings stop early by default (e.g.
    ``HELION_MLIR_XPU_STOP_AT``) lowers and raises :class:`LoweringStoppedError`."""
    if target is None:
        target = resolve_target()
    pipeline = target.resolve_pipeline(pipeline)
    debug = DebugOptions.from_env()
    stop_at = target.default_stop_at()
    if (
        debug.dump_ir
        or debug.dump_pre_lowering
        or debug.dump_lowered
        or debug.dump_stages
        or not target.is_full(stop_at)
    ):
        return _compile_entry(module, entry, target, pipeline, stop_at, debug)
    key = (hashlib.sha256(str(module).encode()).hexdigest(), target.name, pipeline)
    if (compiled := _JIT_CACHE.get(key)) is not None:
        _JIT_CACHE.move_to_end(key)
        return compiled
    compiled = _compile_entry(module, entry, target, pipeline, stop_at, debug)
    _JIT_CACHE[key] = compiled
    if len(_JIT_CACHE) > _JIT_CACHE_SIZE:
        _JIT_CACHE.popitem(last=False)
    return compiled


def lower_entry(
    module: ir.Module,
    entry: str,
    *,
    target: MLIRTarget | None = None,
    pipeline: str | None = None,
    stop_at: str | None = None,
) -> ir.Module:
    """Inline and lower ``entry`` (consumes ``module``) for ``target`` (default:
    the CPU) with ``pipeline`` (default: the target's), to the target's stage
    ``stop_at`` (default: :meth:`MLIRTarget.default_stop_at`; ``None`` there: the
    full lowering, the LLVM dialect on the CPU). Nothing is JIT-compiled."""
    if target is None:
        target = resolve_target()
    pipeline = target.resolve_pipeline(pipeline)
    if stop_at is None:
        stop_at = target.default_stop_at()
    target.check_stop_at(stop_at)
    if stop_at is None and target.stop_points:
        stop_at = target.stop_points[-1]
    target.validate_environment(stop_at)
    return _inline_and_lower(
        module, entry, target, pipeline, stop_at, DebugOptions.from_env()
    )


def _compile_entry(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None,
    debug: DebugOptions,
) -> CompiledEntry:
    full = target.is_full(stop_at)
    target.validate_environment(None if full else stop_at)
    args = entry_args(module, entry)
    lowered = _inline_and_lower(module, entry, target, pipeline, stop_at, debug)
    if not full:
        raise LoweringStoppedError(target.name, stop_at)
    with _stage("JIT compilation"):
        runner = target.runner(lowered)
    return CompiledEntry(entry, args, runner, target)


def _inline_and_lower(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None,
    debug: DebugOptions,
) -> ir.Module:
    _dump_if(debug.dump_ir, "MLIR before inlining", module)
    with _stage("inlining"):
        inline_module(module)
    _dump_if(debug.dump_pre_lowering, "MLIR before lighthouse lowering", module)
    until = "" if target.is_full(stop_at) else f" up to '{stop_at}'"
    with _stage(f"lowering with the lighthouse '{pipeline}' pipeline{until}"):
        lowered = _lower(module, entry, target, pipeline, stop_at)
    _dump_if(debug.dump_lowered, f"MLIR after lighthouse lowering{until}", lowered)
    return lowered


def _lower(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None = None,
) -> ir.Module:
    """Lower ``module`` with the target's lighthouse ``pipeline`` (to ``stop_at``),
    in a forked process under the compile timeout: native passes cannot be
    interrupted, only killed with their process. The lowering is built here and
    applied there."""
    with module.context, ir.Location.unknown():
        lowering = target.lowering(module, entry, pipeline, stop_at)
        timeout = compile_timeout()
        if timeout is None or "fork" not in multiprocessing.get_all_start_methods():
            return lowering.apply(module)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    with _anonymous_file() as stderr:
        child = multiprocessing.get_context("fork").Process(
            target=_lower_in_child,
            args=(lowering, module, sender, stderr.fileno()),
            daemon=True,
        )
        with warnings.catch_warnings():
            # The child uses only the module's context, which runs no threads.
            warnings.filterwarnings("ignore", "This process .* is multi-threaded")
            child.start()
        sender.close()
        error: Exception | None = None
        try:
            if receiver.poll(timeout):
                ok, result = receiver.recv()
                if not ok:
                    error = result
            else:
                child.kill()
                error = CompileTimeoutError(pipeline, timeout)
        except EOFError:
            error = RuntimeError("lighthouse lowering process died")
        finally:
            receiver.close()
            child.join()
        stderr.seek(0)
        output = stderr.read().decode(errors="replace").strip()
    if error is not None:
        if child.exitcode and not isinstance(error, CompileTimeoutError):
            error.add_note(f"exit code {child.exitcode}")
        if output:
            error.add_note(f"lighthouse output:\n{output}")
        raise error
    if output:
        log.debug("lighthouse output:\n%s", output)
    return ir.Module.parse(result, context=module.context)


def _anonymous_file() -> BinaryIO:
    """A file with no name, in memory where the OS supports it (Linux)."""
    if hasattr(os, "memfd_create"):
        return os.fdopen(os.memfd_create("lighthouse-stderr"), "w+b")
    return tempfile.TemporaryFile()


def _lower_in_child(
    lowering: Lowering, module: ir.Module, sender: Connection, stderr: int
) -> None:
    """Forked: apply the parent's lowering to the parent's module (both copied on
    write) and send back the lowered module as bytecode. What lighthouse prints
    goes to ``stderr``: diagnostics of a lowering that succeeds are not errors."""
    os.dup2(stderr, 2)
    # Freeing an MLIR context whose threads did not survive the fork waits forever.
    gc.disable()
    try:
        with module.context, ir.Location.unknown():
            lowered = lowering.apply(module)
        sender.send((True, _bytecode(lowered)))
    except Exception as exc:
        sender.send((False, _picklable(exc)))
    finally:
        sender.close()


def _bytecode(module: ir.Module) -> bytes:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return buffer.getvalue()


def _picklable(exc: Exception) -> Exception:
    try:
        return pickle.loads(pickle.dumps(exc))
    except Exception:
        fallback = RuntimeError(f"{type(exc).__name__}: {exc}")
        for note in getattr(exc, "__notes__", ()):
            fallback.add_note(note)
        return fallback


@contextmanager
def _stage(what: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        exc.add_note(f"Helion MLIR backend: failed while {what}")
        raise
