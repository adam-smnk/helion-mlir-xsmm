"""Lower a generated module with lighthouse and JIT its memref-ABI entry function.

:func:`compile_entry` returns a :class:`CompiledEntry` that takes the entry's
buffers in argument order. Failures keep their exception type and get a note
naming the stage. Lighthouse lowering runs in a forked process, killed after the
compile timeout; only the lowered module comes back. Compiled entries are cached
in-process by module text and pipeline, so equal modules (e.g. configs that clamp
to the same tiles) JIT once.
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
import multiprocessing
import os
import pickle
import tempfile
from typing import TYPE_CHECKING
import warnings

from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
import mlir.ir as ir
from mlir.passmanager import PassManager

from helion_mlir_backend._compiler.mlir.in_place import update_carried_values_in_place
from helion_mlir_backend._compiler.mlir.support.debug import PIPELINES
from helion_mlir_backend._compiler.mlir.support.debug import DebugOptions
from helion_mlir_backend._compiler.mlir.support.debug import compile_timeout
from helion_mlir_backend._compiler.mlir.support.debug import default_pipeline
from helion_mlir_backend._compiler.mlir.support.errors import CompileTimeoutError
from helion_mlir_backend._compiler.mlir.support.type_utils import mlir_dtype_to_torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection
    from typing import BinaryIO

    import torch

log = logging.getLogger(__name__)

_PIPELINE_FILES = {"opt": "./pipeline.yaml", "scalar": "./scalar.yaml"}
_JIT_CACHE: OrderedDict[tuple[str, str], CompiledEntry] = OrderedDict()
_JIT_CACHE_SIZE = 128


def _dump_if(enabled: bool, label: str, module: ir.Module) -> None:
    if enabled:
        print(f"=== {label} ===", flush=True)
        print(module, flush=True)


def pipeline_descriptor(pipeline: str | None = None) -> Descriptor:
    """The lighthouse pipeline by name (default: :func:`default_pipeline`)."""
    if pipeline is None:
        pipeline = default_pipeline()
    if pipeline not in _PIPELINE_FILES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {PIPELINES}")
    return Descriptor(_PIPELINE_FILES[pipeline], base_path=os.path.dirname(__file__))


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
    """``-1`` for a size only known at run time."""
    dtype: torch.dtype


@dataclass(frozen=True)
class CompiledEntry:
    name: str
    args: tuple[EntryArg, ...]
    runner: Runner

    def __post_init__(self) -> None:
        # The ctypes callable of the packed C interface, looked up once.
        object.__setattr__(self, "_func", self.runner.engine.lookup(self.name))

    def __call__(self, buffers: list[torch.Tensor]) -> None:
        packed, _buffer = _packed_memref_args(buffers)
        try:
            self._func(packed)
        except Exception as exc:
            exc.add_note(f"Helion MLIR backend: failed while running '{self.name}'")
            raise


_WORD = ctypes.sizeof(ctypes.c_int64)


def _packed_memref_args(buffers: list[torch.Tensor]) -> tuple[int, ctypes.Array]:
    """The address of the packed arguments of a C-interface entry -- one pointer
    per argument to a pointer to its memref descriptor -- and the buffer holding
    them, which must outlive the call.

    The descriptors (``allocated``, ``aligned``, ``offset``, sizes, strides), the
    descriptor pointers and the packed arguments are all 64-bit words of one
    buffer.
    """
    words: list[int] = []
    starts = []
    for tensor in buffers:
        starts.append(len(words))
        data = tensor.data_ptr()
        words += (data, data, 0, *tensor.shape, *tensor.stride())
    pointers = len(words)
    count = len(buffers)
    buffer = (ctypes.c_int64 * (pointers + 2 * count))()
    base = ctypes.addressof(buffer)
    words += [base + _WORD * start for start in starts]
    words += [base + _WORD * (pointers + i) for i in range(count)]
    buffer[:] = words
    return base + _WORD * (pointers + count), buffer


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
        memref = ir.MemRefType(arg_type)
        dtype = mlir_dtype_to_torch(str(memref.element_type))
        if dtype is None:
            raise ValueError(
                f"'{entry}' takes a memref of unsupported {memref.element_type}"
            )
        args.append(
            EntryArg(
                ir.StringAttr(attrs["helion.name"]).value,
                ir.StringAttr(attrs["helion.role"]).value,
                ir.IntegerAttr(attrs["helion.param"]).value
                if "helion.param" in attrs
                else None,
                tuple(
                    -1 if ir.ShapedType.is_dynamic_size(size) else size
                    for size in memref.shape
                ),
                dtype,
            )
        )
    return tuple(args)


def compile_entry(
    module: ir.Module, entry: str, *, pipeline: str | None = None
) -> CompiledEntry:
    """Inline, lower and JIT-compile ``entry`` (consumes ``module``), or reuse the
    entry compiled from an identical module with the same pipeline."""
    if pipeline is None:
        pipeline = default_pipeline()
    debug = DebugOptions.from_env()
    if debug.dump_ir or debug.dump_pre_lowering or debug.dump_lowered:
        return _compile_entry(module, entry, pipeline, debug)
    key = (hashlib.sha256(str(module).encode()).hexdigest(), pipeline)
    if (compiled := _JIT_CACHE.get(key)) is not None:
        _JIT_CACHE.move_to_end(key)
        return compiled
    compiled = _compile_entry(module, entry, pipeline, debug)
    _JIT_CACHE[key] = compiled
    if len(_JIT_CACHE) > _JIT_CACHE_SIZE:
        _JIT_CACHE.popitem(last=False)
    return compiled


def _compile_entry(
    module: ir.Module, entry: str, pipeline: str, debug: DebugOptions
) -> CompiledEntry:
    args = entry_args(module, entry)
    _dump_if(debug.dump_ir, "MLIR before inlining", module)
    with _stage("inlining"):
        inline_module(module)
    _dump_if(debug.dump_pre_lowering, "MLIR before lighthouse lowering", module)
    with _stage(f"lowering with the lighthouse '{pipeline}' pipeline"):
        lowered = _lower(module, entry, pipeline)
    _dump_if(debug.dump_lowered, "MLIR after lighthouse lowering", lowered)
    with _stage("JIT compilation"):
        runner = Runner(lowered, mem_manager_cls=TorchMemoryManager, shared_libs=[])
    return CompiledEntry(entry, args, runner)


def _lower(module: ir.Module, entry: str, pipeline: str) -> ir.Module:
    """Lower ``module`` with lighthouse, in a forked process under the compile
    timeout: native passes cannot be interrupted, only killed with their process."""
    with module.context, ir.Location.unknown():
        driver = BackendDriver(module, entry, result_to_args=False, benchmark=False)
        driver.add_stage(pipeline_descriptor(pipeline))
        timeout = compile_timeout()
        if timeout is None or "fork" not in multiprocessing.get_all_start_methods():
            return driver.apply(module)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    with _anonymous_file() as stderr:
        child = multiprocessing.get_context("fork").Process(
            target=_lower_in_child,
            args=(driver, module, sender, stderr.fileno()),
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
    driver: BackendDriver, module: ir.Module, sender: Connection, stderr: int
) -> None:
    """Forked: apply the parent's driver to the parent's module (both copied on
    write) and send back the lowered module as bytecode. What lighthouse prints
    goes to ``stderr``: diagnostics of a lowering that succeeds are not errors."""
    os.dup2(stderr, 2)
    # Freeing an MLIR context whose threads did not survive the fork waits forever.
    gc.disable()
    try:
        with module.context, ir.Location.unknown():
            lowered = driver.apply(module)
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
