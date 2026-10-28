"""The XPU target: lighthouse's XeGPU stages, the Level Zero runtime, XPU tensors.

Helion's grid loops already tile the workgroup grid (one ``scf.forall`` per root
loop, an ``scf.for`` K loop inside), so the pipeline skips lighthouse's workgroup
tiling and starts from Helion's tiles. Each root loop becomes one ``gpu.launch``;
the entry function stays on the host and launches the kernels in program order on
one stream, then synchronizes (``gpu-async-region``): phases of a kernel with
``hl.barrier()`` are separate launches. Runtime scalars are entry arguments by
value; the scalar code computed from them is hoisted out of the grid loops, so
each kernel gets the values it uses (e.g. an ``f32`` scale, never an ``f64``).

XeGPU parameters (subgroup, load and prefetch tiles, prefetch depths) come from
the config's ``xegpu_*`` keys, per GPU kernel (``xpu_params``), else from
lighthouse's inference. Each grid loop is tagged with its kernel index before
lowering, so the outlining and layout stages apply each kernel's parameters to
its own ``gpu.launch`` and ``gpu.func`` (lighthouse's stages apply one parameter
set to all kernels).

Only the IR stages run without an XPU-capable LLVM build: ``xegpu-wg`` (annotated
XeGPU) and ``xevm`` (LLVM/XeVM without device binaries). Binaries need the SPIR-V
target, execution the Level Zero runtime (``libmlir_levelzero_runtime.so``).
"""

from __future__ import annotations

import ctypes
import functools
import re
from typing import TYPE_CHECKING

from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.pipeline.driver import BackendDriver
from lighthouse.pipeline.stage import Stage
from lighthouse.pipeline.stage import TransformStage
from lighthouse.schedule import xegpu
from lighthouse.utils.mlir import get_mlir_library_path
import mlir.ir as ir
from mlir.passmanager import PassManager
import torch

from .. import xpu_transforms
from ..mlir.support.debug import XPU_STOP_POINTS
from ..mlir.support.debug import DebugOptions
from ..mlir.support.debug import xpu_stop_at
from ..mlir.support.errors import UnsupportedOperationError
from ..mlir.support.errors import UnsupportedTargetError
from ..mlir_transforms.legalize_host_ops import HOST_OPS_ERROR
from ..mlir_transforms.legalize_host_ops import KERNEL_ARGUMENT
from .base import MLIRTarget
from .xpu_checks import check_xegpu_capabilities
from .xpu_params import XEGPU_CONFIG_KEYS
from .xpu_params import KernelParams
from .xpu_params import attach_params
from .xpu_params import check_params
from .xpu_params import kernel_shape
from .xpu_params import take_params
from .xpu_params import xegpu_params
from .xpu_search import search_space

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_spec import ConfigSpec

_RUNTIME_LIB = "libmlir_levelzero_runtime.so"
_PIPELINES = ("xegpu",)
_CPU = "target='cpu' (CPU tensors)"
_NO_PATTERN = "Could not apply any of the defined patterns"
"""Lighthouse's failure when no kernel class (GEMM, attention, reduction, or the
backend's elementwise class) matches."""
_IR_ONLY = "HELION_MLIR_XPU_STOP_AT=xegpu-wg or xevm (IR only)"
_LLVM_BUILD = (
    "an LLVM built with -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV "
    "-DMLIR_ENABLE_LEVELZERO_RUNNER=1 and Python bindings, at the commit of the "
    "pinned bindings (PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core)"
)


class XpuLowering:
    """Named lighthouse stages applied in order (a ``Lowering``), each failure
    noted with its stage; host-code and kernel-class failures become
    :class:`UnsupportedOperationError`."""

    def __init__(self, stages: list[tuple[str, Stage]], dump: bool) -> None:
        self.stages = stages
        self.dump = dump

    def apply(self, module: ir.Module) -> ir.Module:
        for name, stage in self.stages:
            try:
                module = stage.apply(module)
            except Exception as exc:
                raise _translated(exc, name) from exc
            if self.dump:
                print(f"=== MLIR after XPU stage '{name}' ===", flush=True)
                print(module, flush=True)
        return module


def _translated(exc: Exception, stage: str) -> Exception:
    message = str(exc)
    if match := re.search(rf"{HOST_OPS_ERROR}: ([^;\n]+); ([^\n]+)", message):
        if match.group(1) == KERNEL_ARGUMENT:
            alternatives = ["f32 (or narrower) math on runtime scalars", _CPU]
        else:
            alternatives = ["keep host code to scalars and tensor allocation", _CPU]
        return UnsupportedOperationError(
            match.group(1), reason=match.group(2), alternatives=alternatives
        )
    if match := re.search(rf"([\w-]+): {_NO_PATTERN}", message):
        return UnsupportedOperationError(
            "kernel outside the XeGPU pipeline's kernel classes",
            reason=(
                f"lighthouse's '{match.group(1)}' stage matched no GEMM, attention "
                "reduction or elementwise pattern in a grid loop"
            ),
            alternatives=[_CPU],
        )
    exc.add_note(f"Helion MLIR backend: failed in the XPU stage '{stage}'")
    return exc


def _stages(
    entry: str, kernels: Sequence[KernelParams], stop_at: str
) -> list[tuple[str, Stage]]:
    """The XeGPU stage list of ``entry``, whose tagged grid loops are GPU kernels
    with parameters ``kernels``, up to ``stop_at``; lighthouse's dialects must
    be loaded (schedules use its transform ops)."""
    context = ir.Context.current
    schedules = [
        ("prepare", xpu_transforms.prepare_for_xegpu()),
        ("cleanup", xegpu.cleanup_schedule()),
        # No wg_tiling_schedule: Helion's grid loops are the workgroup tiles.
        ("vectorized", xegpu.vectorize_schedule()),
        ("bufferized", xegpu.bufferize_schedule()),
        ("host-legalized", xpu_transforms.legalize_host_ops()),
        # Lighthouse's outline_gpu_func and annotate_layouts schedules, per kernel.
        ("gpu-outlining", xpu_transforms.outline_kernels(entry, kernels)),
        ("xegpu-initial", xegpu.vector_to_xegpu_schedule()),
        ("xegpu-wg", xpu_transforms.annotate_kernels(entry, kernels)),
    ]
    if stop_at != "xegpu-wg":
        binary_format = "skip" if stop_at == "xevm" else "fatbin"
        schedules.append((stop_at, xpu_transforms.lower_to_xevm(binary_format)))
    return [(name, TransformStage(schedule, context)) for name, schedule in schedules]


@functools.cache
def bindings_emit_spirv() -> bool:
    """Whether the MLIR bindings can emit XPU binaries (have the SPIR-V target):
    the stock wheel cannot. Probed once, in a context of its own."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            "module attributes {gpu.container_module} {"
            ' gpu.module @probe [#xevm.target<chip = "bmg">] {} }'
        )
        try:
            PassManager.parse("builtin.module(gpu-module-to-binary)").run(
                module.operation
            )
        except ir.MLIRError:
            return False
    return True


def _runtime_library() -> str | None:
    """The Level Zero runtime library of the MLIR bindings, if there is one."""
    try:
        path = get_mlir_library_path() / _RUNTIME_LIB
    except (AssertionError, ValueError):
        return None
    return str(path) if path.is_file() else None


class XpuTarget(MLIRTarget):
    """Kernels compiled to Intel XPUs (XeGPU, Battlemage) through lighthouse.

    One pipeline, ``xegpu``; lowerings can stop at ``xegpu-wg``, ``xevm`` or
    ``binary`` (``HELION_MLIR_XPU_STOP_AT`` sets the default, else ``binary``).
    """

    name = "xpu"
    pipelines = _PIPELINES
    config_keys = XEGPU_CONFIG_KEYS
    stop_points = XPU_STOP_POINTS

    def __init__(self) -> None:
        self._device_index = 0
        """The device the Level Zero runtime launches on (its default: 0)."""

    def default_pipeline(self) -> str:
        return "xegpu"

    def default_stop_at(self) -> str | None:
        return xpu_stop_at()

    def configure(self, module: ir.Module, config: Mapping[str, object]) -> None:
        """Record the config's per-kernel ``xegpu_*`` parameters on ``module``."""
        attach_params(module, xegpu_params(config))

    def lowering(
        self,
        module: ir.Module,
        entry: str,
        pipeline: str,
        stop_at: str | None = None,
    ) -> XpuLowering:
        """The XeGPU stage list, after checking that ``module`` (inlined) is in
        what it compiles (:func:`check_xegpu_capabilities`) and that its
        recorded XeGPU parameters, forced or inferred, fit its kernels
        (:func:`check_params`): configs that cannot lower fail here, before any
        lighthouse stage. Tags
        each grid loop (GPU kernel) with its index for the per-kernel stages."""
        self.resolve_pipeline(pipeline)
        self.check_stop_at(stop_at)
        loops = check_xegpu_capabilities(module, entry)
        params = take_params(module)
        check_params(params, [kernel_shape(loop) for loop in loops])
        i64 = ir.IntegerType.get_signless(64)
        for index, loop in enumerate(loops):
            loop.attributes[xpu_transforms.KERNEL_ATTR] = ir.IntegerAttr.get(i64, index)
        kernels = [*params, *[KernelParams()] * (len(loops) - len(params))]
        # Loads lighthouse's dialects, which building the schedules needs.
        BackendDriver(module, entry, result_to_args=False, benchmark=False)
        stages = _stages(entry, kernels, stop_at or self.stop_points[-1])
        return XpuLowering(stages, DebugOptions.from_env().dump_stages)

    def runner(self, module: ir.Module) -> Runner:
        return Runner(
            module, mem_manager_cls=TorchMemoryManager, shared_libs=[_RUNTIME_LIB]
        )

    def check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "xpu":
            raise NotImplementedError(
                f"'{name}' is on {tensor.device}; the kernel was compiled for XPU "
                "tensors"
            )

    def prepare_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """Wait for torch's work on the buffers (e.g. host-code ``torch.zeros``):
        the kernels run on the Level Zero runtime's own queue."""
        devices = {t.device for t in buffers if t.device.type == "xpu"}
        if len(devices) > 1:
            raise ValueError(
                f"the kernel's tensors are on several devices: {sorted(map(str, devices))}"
            )
        for device in devices:
            torch.xpu.synchronize(device)
            index = device.index if device.index is not None else 0
            if index != self._device_index:
                self._set_runtime_device(index)

    def _set_runtime_device(self, index: int) -> None:
        library = _runtime_library()
        if library is None:
            raise RuntimeError(f"{_RUNTIME_LIB} not found")
        # The runtime the JIT loaded: dlopen returns the same library.
        ctypes.CDLL(library).mgpuSetDefaultDevice(ctypes.c_int32(index))
        self._device_index = index

    def hardware_name(self) -> str:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_name()
        return "xpu"

    def adjust_config_spec(self, config_spec: ConfigSpec, device_ir: DeviceIR) -> None:
        """The XPU search space (:func:`xpu_search.search_space`): block sizes
        that divide their sizes and fit each kernel class's XeGPU tiles, defaults
        that lower, and the ``xegpu_*`` keys as per-kernel fragments."""
        search_space(config_spec, device_ir)

    def validate_environment(self, stop_at: str | None = None) -> None:
        """Lowering to ``xegpu-wg`` or ``xevm`` needs nothing; binaries need
        bindings with the SPIR-V target, running also an XPU and the Level Zero
        runtime library."""
        if stop_at in ("xegpu-wg", "xevm"):
            return
        if not bindings_emit_spirv():
            raise UnsupportedTargetError(
                "XPU binaries",
                reason="the MLIR Python bindings have no SPIR-V target",
                alternatives=[_LLVM_BUILD, _IR_ONLY],
            )
        if stop_at is not None:
            return
        if _runtime_library() is None:
            raise UnsupportedTargetError(
                "running XPU kernels",
                reason=f"the MLIR bindings' libraries have no {_RUNTIME_LIB}",
                alternatives=[_LLVM_BUILD, _IR_ONLY],
            )
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise UnsupportedTargetError(
                "running XPU kernels",
                reason="torch finds no XPU (torch.xpu.is_available() is False)",
                alternatives=[_IR_ONLY],
            )
