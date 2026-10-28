"""Transform schedules of the XPU (XeGPU) pipeline that lighthouse's XeGPU
schedules do not cover, built from lighthouse's and this backend's transform ops
(``mlir_transforms``). The stage list is in ``targets/xpu.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import apply_registered_pass
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir.dialects import transform
from mlir.dialects.transform import structured

from helion_mlir_backend._compiler import mlir_transforms as ht

if TYPE_CHECKING:
    from mlir import ir


def prepare_for_xegpu() -> ir.Module:
    """Schedule: Helion's tensor IR in the forms lighthouse's XeGPU stages match.

    - Asserts in linalg bodies (e.g. of ATen helpers) are removed: they block
      vectorization, and kernels cannot assert.
    - ``linalg.contract`` ops (e.g. a matmul of a transposed operand) become
      ``linalg.generic``: lighthouse matches named matmuls and generics only, and
      vectorizes a generic contraction to a ``vector.contract`` whose transposed
      operands load transposed.
    - Elementwise ops are fused (e.g. into a reduction's producers), unused
      symbols (inlined phase and helper functions) dropped, and the module
      cleaned up.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        module = apply_registered_pass(named_seq.bodyTarget, "symbol-dce")
        linalg_ops = lh_transform.match_op(
            module, structured.MatchInterfaceEnum.LinalgOp
        )
        transform_ext.remove_asserts(linalg_ops)
        contracts = lh_transform.match_op(module, "linalg.contract")
        structured.structured_generalize(anytype, contracts)
        module = apply_registered_pass(module, "linalg-fuse-elementwise-ops")
        lh_transform.cleanup(module)
        transform.yield_()
    return schedule


def legalize_host_ops() -> ir.Module:
    """Schedule: every function's host code, after bufferization, legalized for
    device buffers (``legalize_host_ops``: copies and fills as GPU runtime calls,
    a definite failure for other host accesses of device memory)."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.legalize_host_ops(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def lower_to_xevm(binary_format: str = "fatbin") -> ir.Module:
    """Schedule: annotated workgroup-level XeGPU lowered by upstream's
    ``gpu-lower-to-xevm-pipeline`` with the options of lighthouse's
    ``xegpu_to_binary`` (large register file); ``binary_format=skip`` stops
    before emitting device binaries (no SPIR-V target needed)."""
    with schedule_boilerplate() as (schedule, named_seq):
        apply_registered_pass(
            named_seq.bodyTarget,
            "gpu-lower-to-xevm-pipeline",
            options={
                "xegpu-op-level": "workgroup",
                "igc-cmd-options": "-ze-opt-large-register-file",
                "binary-format": binary_format,
            },
        )
        transform.yield_()
    return schedule
