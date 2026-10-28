"""Transform schedules of the XPU (XeGPU) pipeline that lighthouse's XeGPU
schedules do not cover, built from lighthouse's and this backend's transform ops
(``mlir_transforms``). The stage list is in ``targets/xpu.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import apply_registered_pass
from lighthouse.pipeline.helper import canonicalize
from lighthouse.pipeline.helper import match
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.schedule.xegpu.annotate_layouts_schedule import (
    analyze_and_annotate_gpu_func,
)
from lighthouse.schedule.xegpu.lowering_common import convert_to_gpu_launch
from lighthouse.schedule.xegpu.outline_gpu_func_schedule import (
    set_gpu_threads_attention,
)
from lighthouse.schedule.xegpu.outline_gpu_func_schedule import set_gpu_threads_gemm
from lighthouse.schedule.xegpu.outline_gpu_func_schedule import (
    set_gpu_threads_reduction,
)
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured

from helion_mlir_backend._compiler import mlir_transforms as ht

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion_mlir_backend._compiler.targets.xpu_params import KernelParams

KERNEL_ATTR = "helion.kernel"
"""Discardable attribute numbering a function's GPU kernels (its top-level
``scf.forall`` loops, then their ``scf.parallel`` and ``gpu.launch`` ops) in
program order."""


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


def kernel_module_name(entry: str, index: int) -> str:
    """The ``gpu.module`` that kernel ``index`` of ``entry`` is outlined to:
    the names upstream outlining gives in program order (``<entry>_kernel``,
    ``<entry>_kernel_0``, ...), set explicitly so the kernel is found by name."""
    return f"{entry}_kernel" if index == 0 else f"{entry}_kernel_{index - 1}"


def _i64_param(value: int) -> ir.Value:
    i64 = ir.IntegerType.get_signless(64)
    return transform.ParamConstantOp(
        transform.AnyParamType.get(), ir.IntegerAttr.get(i64, value)
    ).param


def _only(handle: ir.Value) -> ir.Value:
    """``handle``, failing unless it holds exactly one op."""
    return transform.split_handle([transform.AnyOpType.get()], handle)


def _kernel_ops(target: ir.Value, op_name: str, index: int) -> ir.Value:
    """The ``op_name`` ops in ``target`` tagged as kernel ``index``."""
    i64 = ir.IntegerType.get_signless(64)
    return match(
        target, ops={op_name}, op_attrs={KERNEL_ATTR: ir.IntegerAttr.get(i64, index)}
    )


def outline_kernels(entry: str, kernels: Sequence[KernelParams]) -> ir.Module:
    """Schedule: lighthouse's ``outline_gpu_func_schedule`` for ``entry``, kernel
    by kernel with each kernel's parameters.

    ``entry``'s top-level ``scf.forall`` loops are tagged ``helion.kernel = i``
    (one per kernel, ``kernels[i]`` its parameters). Each becomes an
    ``scf.parallel`` (which drops discardable attributes: the tag is set again),
    then a tagged ``gpu.launch`` (lighthouse's conversion, which keeps them). Each
    launch gets its thread count from lighthouse's kernel classes (attention,
    GEMM, reduction; a forced subgroup tile applies) and is outlined to the
    ``gpu.module`` named by :func:`kernel_module_name`. A missing or duplicated
    tag fails the schedule.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        module = named_seq.bodyTarget
        func = _only(
            match(
                module,
                ops={"func.func"},
                op_attrs={"sym_name": ir.StringAttr.get(entry)},
            )
        )
        for index in range(len(kernels)):
            forall = _only(_kernel_ops(func, "scf.forall", index))
            parallel = loop.loop_forall_to_parallel([anytype], forall)
            transform.annotate(parallel, KERNEL_ATTR, param=_i64_param(index))
        func = convert_to_gpu_launch(module, payload_func=func)
        for index, kernel in enumerate(kernels):
            func = _set_threads(func, index, kernel.sg_tile)
            launch = _only(_kernel_ops(func, "gpu.launch", index))
            name = ir.FlatSymbolRefAttr.get(kernel_module_name(entry, index))
            module_name = transform.ParamConstantOp(transform.AnyParamType.get(), name)
            transform.annotate(launch, "module", param=module_name.param)
        # The tail of lighthouse's `outline`.
        func = apply_registered_pass(func, "lower-affine")
        canonicalize(func)
        apply_registered_pass(func, "gpu-launch-sink-index-computations")
        module = apply_registered_pass(module, "gpu-kernel-outlining")
        transform.apply_cse(module)
        transform.yield_()
    return schedule


def _set_threads(
    func: ir.Value, index: int, sg_tile: tuple[int, ...] | None
) -> ir.Value:
    """Set the thread count of kernel ``index``'s ``gpu.launch`` in ``func`` the
    way its kernel class needs (lighthouse's alternatives, scoped to the
    function: it must be isolated from above); returns the new ``func``."""
    force = list(sg_tile) if sg_tile is not None else None
    alternatives = lh_transform.alternatives(
        func, num_alternatives=4, result_types=[func.type]
    )
    for region, set_threads in enumerate(
        (set_gpu_threads_attention, set_gpu_threads_gemm, set_gpu_threads_reduction)
    ):
        with alternatives.region(region) as scope:
            set_threads(_kernel_ops(scope, "gpu.launch", index), force_sg_tile=force)
            transform.yield_([scope])
    with alternatives.region(3) as scope:
        transform_ext.emit_definite_failure(
            scope,
            message="outline_gpu_func: Could not apply any of the defined patterns.",
        )
        transform.yield_([scope])
    return alternatives.results[0]


def annotate_kernels(entry: str, kernels: Sequence[KernelParams]) -> ir.Module:
    """Schedule: lighthouse's ``annotate_layouts_schedule``, ``gpu.func`` by
    ``gpu.func`` (found by :func:`kernel_module_name`) with each kernel's
    parameters."""
    with schedule_boilerplate() as (schedule, named_seq):
        for index, kernel in enumerate(kernels):
            gpu_module = _only(
                match(
                    named_seq.bodyTarget,
                    ops={"gpu.module"},
                    op_attrs={
                        "sym_name": ir.StringAttr.get(kernel_module_name(entry, index))
                    },
                )
            )
            gpu_func = _only(match(gpu_module, ops={"gpu.func"}))
            analyze_and_annotate_gpu_func(gpu_func, **kernel.annotate_options())
        transform.yield_()
    return schedule
