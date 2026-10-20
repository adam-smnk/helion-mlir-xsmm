"""Environment-driven debug options, read in one place."""

from __future__ import annotations

from dataclasses import dataclass
import os

_FALSE_VALUES = ("", "0", "false", "no")
PIPELINES = ("opt", "scalar")
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

    @classmethod
    def from_env(cls) -> DebugOptions:
        # Read per call: tests and users toggle these at runtime.
        return cls(
            dump_ir=_flag("HELION_MLIR_DUMP_IR"),
            dump_pre_lowering=_flag("HELION_MLIR_DUMP_PRE_LOWERING"),
            dump_lowered=_flag("HELION_MLIR_DUMP_LOWERED"),
        )


def default_pipeline() -> str:
    """``HELION_MLIR_PIPELINE`` (``opt`` or ``scalar``); ``opt`` if unset."""
    pipeline = os.environ.get("HELION_MLIR_PIPELINE", "").strip() or "opt"
    if pipeline not in PIPELINES:
        raise ValueError(
            f"HELION_MLIR_PIPELINE={pipeline!r}; expected one of {PIPELINES}"
        )
    return pipeline


def compile_timeout() -> float | None:
    """Seconds a lighthouse lowering may take: ``HELION_MLIR_COMPILE_TIMEOUT``,
    30 if unset; ``None`` (no limit) for 0."""
    value = os.environ.get("HELION_MLIR_COMPILE_TIMEOUT", "").strip()
    timeout = float(value) if value else 30.0
    return timeout if timeout > 0 else None
