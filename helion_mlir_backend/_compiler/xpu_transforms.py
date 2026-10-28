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
from lighthouse.schedule.xegpu.annotate_layouts_schedule import annotate_attention
from lighthouse.schedule.xegpu.annotate_layouts_schedule import annotate_gemm
from lighthouse.schedule.xegpu.annotate_layouts_schedule import annotate_reduction
from lighthouse.schedule.xegpu.lowering_common import convert_to_gpu_launch
from lighthouse.schedule.xegpu.matmul_constraints import NB_WORKITEMS
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
from mlir.dialects.transform import tensor as tensor_transform
from mlir.dialects.transform import vector as vector_transform
from mlir.dialects.transform import xegpu

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
    - Unit dims of linalg ops are folded with reshapes (before lighthouse's
      cleanup folds them with slices): a ``keepdim`` reduction stored as a
      ``[rows, 1]`` column then reduces into a ``[rows]`` init, not a slice of a
      filled ``[rows, 1]`` temporary (which bufferizes to a scratch buffer the
      XeGPU conversion cannot read).
    - Unit tile dims are folded away entirely, so a grid loop with block size 1
      on its leading (e.g. batch or head) dims is its lower-rank kernel: the
      reshapes the folding leaves fold into rank-reducing tile loads and
      stores (``tensor.extract_slice``/``parallel_insert_slice``) and out of
      loop-carried accumulators (``fold_loop_unit_dims``); batched matmuls are
      generalized first (as lighthouse's cleanup does) so a unit batch dim
      folds, leaving a matmul.

    The XPU target applies this stage before its capability checks and
    parameter validation, which therefore see the rank-reduced kernels.
    """
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        module = apply_registered_pass(named_seq.bodyTarget, "symbol-dce")
        linalg_ops = lh_transform.match_op(
            module, structured.MatchInterfaceEnum.LinalgOp
        )
        transform_ext.remove_asserts(linalg_ops)
        contracts = lh_transform.match_op(
            module, ["linalg.contract", "linalg.batch_matmul"]
        )
        structured.structured_generalize(anytype, contracts)
        module = apply_registered_pass(module, "linalg-fuse-elementwise-ops")
        with ir.InsertionPoint(transform.apply_patterns(module).patterns):
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_reshapes()
            tensor_transform.apply_patterns_tensor_reassociative_reshape_folding()
        ht.fold_loop_unit_dims(module)
        with ir.InsertionPoint(transform.apply_patterns(module).patterns):
            tensor_transform.apply_patterns_tensor_reassociative_reshape_folding()
        lh_transform.cleanup(module)
        transform.yield_()
    return schedule


def legalize_host_ops() -> ir.Module:
    """Schedule: every function's host code, after bufferization, legalized for
    device buffers (``legalize_host_ops``: copies and fills as GPU runtime calls,
    a definite failure for other host accesses of device memory).

    First, the scalar code of the grid loops (the future kernels) that does not
    depend on them moves out (``hoist_invariant_scalars``): e.g. the conversion
    of an ``f64`` runtime scalar to an ``f32`` kernel's type runs on the host, so
    the kernel takes the ``f32`` value. A kernel that would still take a float
    wider than 32 bits (the XPU has no native ``f64``) fails.
    """
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        grid_loops = lh_transform.match_op(named_seq.bodyTarget, "scf.forall")
        ht.hoist_invariant_scalars(grid_loops)
        ht.legalize_host_ops(funcs, max_float_width=32)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def lower_to_xevm(binary_format: str = "fatbin") -> ir.Module:
    """Schedule: annotated workgroup-level XeGPU lowered by upstream's
    ``gpu-lower-to-xevm-pipeline`` with the options of lighthouse's
    ``xegpu_to_binary`` (large register file); ``binary_format=skip`` stops
    before emitting device binaries (no SPIR-V target needed). bf16 math is
    extended to f32 first (see below)."""
    with schedule_boilerplate() as (schedule, named_seq):
        # The pipeline converts fast (`afn`) math to native OpenCL calls before
        # it extends bf16 math to f32 (math-to-xevm takes any float type), which
        # leaves calls of bf16 natives that do not exist: extend first, with the
        # pipeline's own options (f16 and f32 math stay).
        gpu_modules = match(named_seq.bodyTarget, ops={"gpu.module"})
        apply_registered_pass(
            gpu_modules,
            "math-extend-to-supported-types",
            options={"extra-types": "f16", "target-type": "f32"},
        )
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
    GEMM, reduction), else as an elementwise kernel
    (``infer_xegpu_elementwise_layout``; forced subgroup and load tiles apply),
    and is outlined to the ``gpu.module`` named by :func:`kernel_module_name`. A
    missing or duplicated tag fails the schedule.

    First, transfers whose permutation map drops dims (e.g. a row vector written
    into a ``[rows, 1]`` column) are rewritten to minor-identity transfers of a
    reshaped vector, which the XeGPU conversion handles (scatter stores). Then
    transfers of tiles whose leading dims were unit (folded away) from
    higher-rank buffers move onto a view of the buffer with the tile's rank
    (``rank_reduce_transfers``), where lighthouse's analyses read the matrix the
    tile belongs to (e.g. one batch of a batched matmul's operands).
    """
    ht.HelionTransformDialect.load()
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
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            vector_transform.apply_patterns_vector_transfer_permutation_patterns()
        ht.rank_reduce_transfers(func)
        for index in range(len(kernels)):
            forall = _only(_kernel_ops(func, "scf.forall", index))
            parallel = loop.loop_forall_to_parallel([anytype], forall)
            transform.annotate(parallel, KERNEL_ATTR, param=_i64_param(index))
        func = convert_to_gpu_launch(module, payload_func=func)
        for index, kernel in enumerate(kernels):
            func = _set_threads(func, index, kernel)
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


def _set_threads(func: ir.Value, index: int, kernel: KernelParams) -> ir.Value:
    """Set the thread count of kernel ``index``'s ``gpu.launch`` in ``func`` the
    way its kernel class needs (lighthouse's alternatives, then elementwise
    kernels; scoped to the function: it must be isolated from above); returns
    the new ``func``."""
    force = list(kernel.sg_tile) if kernel.sg_tile is not None else None
    alternatives = lh_transform.alternatives(
        func, num_alternatives=5, result_types=[func.type]
    )
    for region, set_threads in enumerate(
        (set_gpu_threads_attention, set_gpu_threads_gemm, set_gpu_threads_reduction)
    ):
        with alternatives.region(region) as scope:
            set_threads(_kernel_ops(scope, "gpu.launch", index), force_sg_tile=force)
            transform.yield_([scope])
    with alternatives.region(3) as scope:
        launch = _only(_kernel_ops(scope, "gpu.launch", index))
        layout = _elementwise_layout(launch, kernel)
        threads = transform_ext.compute_num_threads(
            layout.wg_tile, layout.sg_tile, base=NB_WORKITEMS
        )
        xegpu.set_gpu_launch_threads(launch, threads=[threads, 1, 1])
        transform.yield_([scope])
    with alternatives.region(4) as scope:
        transform_ext.emit_definite_failure(
            scope,
            message="outline_gpu_func: Could not apply any of the defined patterns.",
        )
        transform.yield_([scope])
    return alternatives.results[0]


def _elementwise_layout(
    target: ir.Value, kernel: KernelParams
) -> ht.InferXeGPUElementwiseLayoutOp:
    """``infer_xegpu_elementwise_layout`` of ``target`` with ``kernel``'s forced
    subgroup and load tiles."""
    return ht.infer_xegpu_elementwise_layout(
        target, sg_tile=kernel.sg_tile, load_tile=kernel.block_load_tile()
    )


def annotate_kernels(entry: str, kernels: Sequence[KernelParams]) -> ir.Module:
    """Schedule: lighthouse's ``annotate_layouts_schedule``, ``gpu.func`` by
    ``gpu.func`` (found by :func:`kernel_module_name`) with each kernel's
    parameters, plus elementwise kernels (:func:`annotate_elementwise`)."""
    ht.HelionTransformDialect.load()
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
            _annotate_kernel(gpu_func, kernel)
        transform.yield_()
    return schedule


def _annotate_kernel(gpu_func: ir.Value, kernel: KernelParams) -> None:
    """Lighthouse's ``analyze_and_annotate_gpu_func`` (attention, GEMM and
    reduction kernels, in that order and as it builds them; reductions also get
    their row-result stores anchored), reductions storing only row results,
    elementwise kernels, then a definite failure."""
    options = kernel.annotate_options()
    sg_tile = options.pop("sg_tile")
    alternatives = lh_transform.alternatives(
        gpu_func,
        num_alternatives=6,
        result_types=[gpu_func.type],
    )
    with alternatives.region(0) as scope:
        dpas_ops = lh_transform.match_op(scope, "xegpu.dpas")
        dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_attention(scope, dpas_op, force_sg_tile=sg_tile)
        transform.yield_([scope])
    with alternatives.region(1) as scope:
        dpas_ops = lh_transform.match_op(scope, "xegpu.dpas")
        dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
        annotate_gemm(
            scope,
            dpas_op,
            force_sg_tile=sg_tile,
            force_load_tile_a=options.get("load_tile_a"),
            force_load_tile_b=options.get("load_tile_b"),
            force_prefetch_tile_a=options.get("prefetch_tile_a"),
            force_prefetch_tile_b=options.get("prefetch_tile_b"),
            force_prefetch_a_nb=options.get("prefetch_a_nb"),
            force_prefetch_b_nb=options.get("prefetch_b_nb"),
        )
        transform.yield_([scope])
    with alternatives.region(2) as scope:
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_reduction(scope, anchor_op, force_sg_tile=sg_tile)
        annotate_row_result_stores(scope, anchor_op, sg_tile)
        annotate_reduction_tiles(scope, anchor_op, kernel)
        transform.yield_([scope])
    with alternatives.region(3) as scope:
        # A reduction storing only its row results (no 2D tile store).
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_row_result_stores(scope, anchor_op, sg_tile, require=True)
        annotate_reduction_tiles(scope, anchor_op, kernel)
        transform.yield_([scope])
    with alternatives.region(4) as scope:
        annotate_elementwise(scope, kernel)
        transform.yield_([scope])
    with alternatives.region(5) as scope:
        transform_ext.emit_definite_failure(
            scope,
            message="annotate_layouts: Could not apply any of the defined patterns.",
        )
        transform.yield_([scope])


def annotate_row_result_stores(
    gpu_func: ir.Value,
    anchor_op: ir.Value,
    sg_tile: Sequence[int] | None,
    *,
    require: bool = False,
) -> None:
    """Anchor the row-result stores of the reduction ``anchor_op``
    (``match_row_result_stores``) with lighthouse's reduction layout: row
    vectors as its slice over the reduced dim, ``[rows, 1]`` columns with one
    column per subgroup. Lighthouse's 2D reductions only (``[rows, 0]``
    subgroup tiles). With ``require``, fails silenceably without such stores."""
    sg_layout, sg_data = _reduction_layout(anchor_op, sg_tile)
    stores = ht.match_row_result_stores(gpu_func, anchor_op, require=require)
    xegpu.set_anchor_layout(
        stores.sliced, sg_layout=sg_layout, sg_data=sg_data, slice_dims=[1]
    )
    xegpu.set_anchor_layout(
        stores.keepdim, sg_layout=sg_layout, sg_data=[sg_data[0], 1]
    )


def annotate_reduction_tiles(
    gpu_func: ir.Value, anchor_op: ir.Value, kernel: KernelParams
) -> None:
    """Anchor the tile loads (and prefetches) and stores of the reduction
    ``anchor_op``'s source tile shape (``infer_xegpu_reduction_tiles``) with
    lighthouse's reduction layout and their per-instruction tiles: the 2D block
    load tile (``kernel``'s forced one, else the largest fitting) on loads, the
    2D block store tile on stores. Lighthouse sets no ``inst_data``, and the
    loads would lower to one-row block reads. Lighthouse's 2D reductions only
    (``[rows, 0]`` subgroup tiles)."""
    sg_layout, sg_data = _reduction_layout(anchor_op, kernel.sg_tile)
    tiles = ht.infer_xegpu_reduction_tiles(
        gpu_func, anchor_op, sg_data, load_tile=kernel.block_load_tile()
    )
    for ops, inst_data in (
        (tiles.loads, tiles.load_tile),
        (tiles.stores, tiles.store_tile),
    ):
        xegpu.set_anchor_layout(
            ops, sg_layout=sg_layout, sg_data=sg_data, inst_data=inst_data
        )


def _reduction_layout(
    anchor_op: ir.Value, sg_tile: Sequence[int] | None
) -> tuple[list[ir.Value], list[ir.Value]]:
    """Lighthouse's ``sg_layout`` and ``sg_data`` params (one per dim) of the
    reduction ``anchor_op`` (as ``annotate_reduction`` computes them)."""
    wg_tile, sg, reduction_tile = transform_ext.infer_xegpu_reduction_params(
        anchor_op, force_sg_tile=list(sg_tile) if sg_tile is not None else None
    )
    return transform_ext.compute_sg_layout(wg_tile, sg, reduction_tile)


def annotate_elementwise(gpu_func: ir.Value, kernel: KernelParams) -> None:
    """Anchor the layout ``infer_xegpu_elementwise_layout`` infers for the
    elementwise ``gpu_func`` on its tile loads and stores: ``sg_layout = wg_tile
    / sg_tile`` and ``sg_data = sg_tile`` on all, ``inst_data`` the load tile on
    loads and the store tile on stores (lighthouse's legacy elementwise schedule
    anchors its loads and stores the same way). Fails silenceably for other
    kernels."""
    layout = _elementwise_layout(gpu_func, kernel)
    for tiles, inst_data in (
        (layout.loads, layout.load_tile),
        (layout.stores, layout.store_tile),
    ):
        xegpu.set_anchor_layout(
            tiles,
            sg_layout=layout.sg_layout,
            sg_data=layout.sg_tile,
            inst_data=inst_data,
        )
    lh_transform.cleanup(gpu_func)
