"""The device-specific half of compiling and running a kernel.

Code generation is target-neutral: one Linalg-on-Tensors module serves every
device. An :class:`MLIRTarget` owns everything else -- the lighthouse lowering
pipelines, the JIT runner, the device checks and hooks around an entry call, and
what autotuning searches and caches. Exactly one target compiles a kernel.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Protocol

import mlir.ir as ir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_spec import ConfigSpec
    from lighthouse.execution.runner import Runner
    import torch


class Lowering(Protocol):
    """A lowering ready to apply (e.g. a lighthouse ``BackendDriver``)."""

    def apply(self, module: ir.Module) -> ir.Module: ...


class MLIRTarget(abc.ABC):
    """A device the MLIR backend compiles kernels for.

    Targets hold no per-kernel state: one instance serves every kernel.
    """

    name: ClassVar[str]
    """``cpu`` or ``xpu``; also part of JIT and autotune cache keys."""
    pipelines: ClassVar[tuple[str, ...]]
    """The names of the target's lighthouse pipelines."""

    @abc.abstractmethod
    def default_pipeline(self) -> str:
        """The pipeline of configs that name none (may read the environment)."""

    @abc.abstractmethod
    def lowering(self, module: ir.Module, entry: str, pipeline: str) -> Lowering:
        """The lowering of ``module`` with ``pipeline``, down to the LLVM dialect.

        Called in ``module``'s context. Unknown pipelines raise ``ValueError``. The
        result may be applied in a forked process: it must not need the threads of
        the calling one.
        """

    def lower(self, module: ir.Module, entry: str, pipeline: str) -> ir.Module:
        """``module`` lowered with ``pipeline``, in this process with no time limit."""
        with module.context, ir.Location.unknown():
            return self.lowering(module, entry, pipeline).apply(module)

    @abc.abstractmethod
    def runner(self, module: ir.Module) -> Runner:
        """The lighthouse JIT runner of a lowered module."""

    @abc.abstractmethod
    def check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Reject a tensor argument (named by its host expression) that the
        compiled kernel cannot access, e.g. one on another device."""

    def prepare_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """Before an entry call: make ``buffers`` ready for the kernel (default:
        nothing to do)."""
        return

    def finish_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """After an entry call: make the kernel's writes visible to the host
        (default: nothing to do)."""
        return

    @abc.abstractmethod
    def hardware_name(self) -> str:
        """The device model, keying autotuned configs."""

    def adjust_config_spec(self, config_spec: ConfigSpec, device_ir: DeviceIR) -> None:
        """Narrow the autotuning search space of a kernel to what the target
        compiles well, before a search (default: unchanged)."""
        return

    def validate_environment(self) -> None:
        """Raise if this process cannot compile and run kernels for the target
        (default: it can)."""
        return

    def __repr__(self) -> str:
        return f"<MLIR target {self.name!r}>"
