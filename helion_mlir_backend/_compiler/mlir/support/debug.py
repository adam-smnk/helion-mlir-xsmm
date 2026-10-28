"""Environment-driven debug options, read in one place."""

from __future__ import annotations

from dataclasses import dataclass
import os

_FALSE_VALUES = ("", "0", "false", "no")
PIPELINES = ("opt", "scalar")
"""The CPU target's lighthouse pipelines."""
TARGETS = ("auto", "cpu", "xpu")
"""``HELION_MLIR_TARGET`` values; ``auto`` follows the device."""
XPU_STOP_POINTS = ("xegpu-wg", "xevm", "binary")
"""Where the XPU lowering can stop: annotated XeGPU, LLVM/XeVM without a device
binary, and the full lowering with the binary."""
PIPELINE_CONFIG_KEY = "mlir_pipeline"
"""Config key naming the lighthouse pipeline (``opt``/``scalar``) of one config."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSE_VALUES


@dataclass(frozen=True)
class DebugOptions:
    """IR dump switches (``HELION_MLIR_DUMP_*``)."""

    dump_ir: bool
    dump_pre_lowering: bool
    dump_lowered: bool
    dump_stages: bool
    """The IR after each stage of a lowering built stage by stage (XPU)."""

    @classmethod
    def from_env(cls) -> DebugOptions:
        # Read per call: tests and users toggle these at runtime.
        return cls(
            dump_ir=_flag("HELION_MLIR_DUMP_IR"),
            dump_pre_lowering=_flag("HELION_MLIR_DUMP_PRE_LOWERING"),
            dump_lowered=_flag("HELION_MLIR_DUMP_LOWERED"),
            dump_stages=_flag("HELION_MLIR_DUMP_STAGES"),
        )


def default_pipeline() -> str:
    """``HELION_MLIR_PIPELINE`` (``opt`` or ``scalar``); ``opt`` if unset."""
    pipeline = os.environ.get("HELION_MLIR_PIPELINE", "").strip() or "opt"
    if pipeline not in PIPELINES:
        raise ValueError(
            f"HELION_MLIR_PIPELINE={pipeline!r}; expected one of {PIPELINES}"
        )
    return pipeline


def target_override() -> str | None:
    """``HELION_MLIR_TARGET`` (``cpu`` or ``xpu``); ``None`` if unset or ``auto``
    (the target follows the device of the kernel's tensors)."""
    target = os.environ.get("HELION_MLIR_TARGET", "").strip() or "auto"
    if target not in TARGETS:
        raise ValueError(f"HELION_MLIR_TARGET={target!r}; expected one of {TARGETS}")
    return None if target == "auto" else target


def xpu_stop_at() -> str | None:
    """``HELION_MLIR_XPU_STOP_AT`` (``xegpu-wg``, ``xevm`` or ``binary``): where
    XPU lowerings that name no stop point stop; ``None`` (the full lowering) if
    unset."""
    stop_at = os.environ.get("HELION_MLIR_XPU_STOP_AT", "").strip() or None
    if stop_at is not None and stop_at not in XPU_STOP_POINTS:
        raise ValueError(
            f"HELION_MLIR_XPU_STOP_AT={stop_at!r}; expected one of {XPU_STOP_POINTS}"
        )
    return stop_at


def compile_timeout() -> float | None:
    """Seconds a lighthouse lowering may take: ``HELION_MLIR_COMPILE_TIMEOUT``,
    30 if unset; ``None`` (no limit) for 0."""
    value = os.environ.get("HELION_MLIR_COMPILE_TIMEOUT", "").strip()
    timeout = float(value) if value else 30.0
    return timeout if timeout > 0 else None
