"""Autotuning on the CPU: Helion's local autotune cache, keyed by the host CPU.

Helion's :class:`LocalAutotuneCache` derives its key from a GPU (or TPU/MPS)
device and has no CPU entry, so the MLIR backend uses this subclass instead.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path
import platform
import textwrap

from helion.autotuner.base_cache import LooseAutotuneCacheKey
from helion.autotuner.local_cache import LocalAutotuneCache

from .support.debug import default_pipeline

CPU_AUTOTUNE_CACHE = "MLIRCpuAutotuneCache"
"""Name under which the cache is registered in ``helion.autotuner.cache_classes``."""


class CpuAutotuneCache(LocalAutotuneCache):
    """Best configs on disk, keyed by the CPU model and the default pipeline."""

    def _generate_key(self) -> LooseAutotuneCacheKey:
        bound = self.kernel
        in_memory = bound.kernel._create_bound_kernel_cache_key(
            bound, tuple(self.args), bound.kernel._base_specialization_key(self.args)
        )
        source = textwrap.dedent(inspect.getsource(bound.kernel.fn))
        pipeline = default_pipeline()
        return LooseAutotuneCacheKey(
            specialization_key=in_memory.specialization_key,
            extra_results=in_memory.extra_results,
            kernel_source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            hardware=cpu_name(),
            runtime_name=f"lighthouse-{pipeline}",
            backend=bound.env.backend.name,
            config_spec_hash=bound.config_spec.structural_fingerprint_hash(
                advanced_controls_files=self.autotuner.settings.autotune_search_acf
                or None
            ),
            extra_cache_key=bound.extra_cache_key(),
            best_of_k=self.autotuner.settings.autotune_best_of_k,
        )


@functools.cache
def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()
