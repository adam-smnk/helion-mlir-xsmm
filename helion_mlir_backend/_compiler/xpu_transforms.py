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
    - A store at a fixed index stores that slice, not the whole region the
      iteration owns (``fold_region_inserts``: one row of a K/V cache).
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
      loop-carried accumulators (``fold_loop_unit_dims``), also unit dims
      inside a tile (``fold_slice_unit_dims``: e.g. one K chunk of a
      ``[M, S, K/S]`` view); batched matmuls are
      generalized first (as lighthouse's cleanup does) so a unit batch dim
      folds, leaving a matmul; so are named elementwise ops (a tile times a
      runtime scalar's fill), which the folding skips: their unit dims would
      otherwise reach bufferization as a scratch tile (``tile_width * v_blocks``
      in XeVM for a runtime attention scale).
    - Loop-carried tensors updated elementwise into a fresh tensor (e.g. the
      running sum of a flash-attention kernel) are updated in place
      (``carry_loop_inits``), so vectorization carries them as vectors.
    - A tile stored unchanged (a copy, or ``a * 1.0`` folded to one) is
      stored through a ``linalg.copy`` (``materialize_tile_copies``): the
      kernel has a linalg op to vectorize like an elementwise one.

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
            module, ["linalg.contract", "linalg.batch_matmul", "linalg.elementwise"]
        )
        structured.structured_generalize(anytype, contracts)
        ht.fold_region_inserts(module)
        module = apply_registered_pass(module, "linalg-fuse-elementwise-ops")
        with ir.InsertionPoint(transform.apply_patterns(module).patterns):
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_reshapes()
            tensor_transform.apply_patterns_tensor_reassociative_reshape_folding()
        ht.fold_loop_unit_dims(module)
        with ir.InsertionPoint(transform.apply_patterns(module).patterns):
            tensor_transform.apply_patterns_tensor_reassociative_reshape_folding()
        lh_transform.cleanup(module)
        # After the cleanup, which leaves the collapses of unit-dim slices.
        ht.fold_slice_unit_dims(module)
        lh_transform.cleanup(module)
        ht.carry_loop_inits(module)
        ht.materialize_tile_copies(module)
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


IGC_OPTIONS = r"-ze-opt-large-register-file\" \"-igc_opts \'VectorizerAllowCMP=0\'"
"""IGC options of a device binary by default: ``igc_vectorizer`` unset or ``"cmp"``.
Only ``gpu-module-to-binary``'s ``opts`` pass them to IGC (the XPU target's binary
stage); this LLVM never reads a target's ``cmd-options`` flag.

The spelling is for that path: the options are tokenized, so the quotes
``-igc_opts`` requires are escaped, then joined and wrapped in one pair of
double quotes, and the driver takes what is in a pair as one option: two options
in one pair silently lose the large register file (128 GRFs, with no error). The
escaped quotes close the first pair and open a second one.

- The large register file: 256 GRFs; without it GEMMs get 128 and spill.
- ``VectorizerAllowCMP=0`` keeps ``IGCVectorizer`` on without its vector compares,
  one of the two miscompiles below (IGC 2.41, B50). The other one, a vector ``fptrunc``
  before its operand, is avoided in the IR (:mod:`.mlir_transforms.xevm_lane_conversions`).

On an Arc B50 with IGC 2.41, every kernel benchmarked here gives correct results; the
two known bugs are avoided by this option and an IR rewrite, on that one IGC build.
Other kernels and other builds are not proven: ``agent-tools/igc_binary_check.py`` in
helion-mlir-dev checks a kernel's binaries, and ``igc_vectorizer="off"`` is the escape
hatch.

Device time with :data:`IGC_OPTIONS_OFF` against this, on the B50 at bench shapes:
29_Softplus +6.3 %, L2 29_Matmul_Mish_Mish +6.9 %, L2 81 +3.3 %, attention with
``maxnum`` row maxima +17 % on 97 at ``[1, 1, 256, 32]`` (3003 against 2561 us) and
+44 % on Llama GQA at ``[1, 1, 256, 64]`` (318 against 220), the rest about 1 % or
less either way. Compile time is the price: over the 37 benchmarked problems 23.4
against 19.4 s with :data:`IGC_OPTIONS_OFF` on the compile node, nearly all of it L2
29_Matmul_Mish_Mish (7.1 against 2.1 s).

The pass's partial switches (``-igc_opts``, comma separated), and what each avoids:

- ``VectorizerAllowCMP=0``: no vector compares. ``IGCVectorizer`` rebuilds scalar
  compares and selects as ``select <8 x i1> (fcmp a, b), a, b``, and
  ``CodeGenPatternMatch::MatchMinMax`` takes it for an integer min/max: its guard
  ``isFloatingPointTy()`` is false for ``<8 x float>``, and ``EmitMinMax`` has no
  vector path. Only element 0 is written; the rest read whatever the registers
  hold, also from earlier kernels. In what we tried it takes a row value compared
  with another one kept over a loop, or with a second row value: a running
  maximum, or a row sum through a clamp.
- ``VectorizerAllowFPTRUNC=0``: no vector ``fptrunc``. ``fptrunc <8 x float> %v`` is
  placed before the ``insertelement``s of ``%v``, a use before its def, where a scalar
  builtin's result goes straight into a conversion to 16 bits (``sin``, ``cos``, ``tan``,
  ``log``, ``log2``, ``expm1``, ``sinh``, ``cosh`` and ``acos`` of f32 rounded to bf16
  here; ``upstream-repro/igc-vector-minmax/fptrunc`` in the wiki, on a scalar ``tanh``
  builtin; Helion's ``tanh`` is a polynomial, ``xpu_math_approx``, and does not take
  it), or narrowing from f64: the output is whatever those registers hold, left by an
  earlier kernel, zero only if nothing has written them yet. Not set here, as it
  costs most of what that gains on attention (Llama GQA
  288 against 220 us): :mod:`.mlir_transforms.xevm_lane_conversions` avoids that form
  in the IR instead, for every kernel.
- ``VectorizerAllowSelect=0``: avoids the Triton compiler crash through ``ocloc``,
  as ``VectorizerAllowCMP=0`` also does; cause not investigated.
- ``VectorizerAllowFADD``, ``FSUB``, ``FMUL``, ``FDIV``, ``EXP2``, ``MAXNUM``,
  ``WAVEALL``, ``WAVEALLJoint``, ``UniformCMP``, ``UniformSelect``: no bug known in
  them, none tried on our kernels."""
IGC_OPTIONS_OFF = r"-ze-opt-large-register-file\" \"-igc_opts \'EnableVectorizer=0\'"
"""IGC options of a device binary with ``igc_vectorizer`` ``"off"``, the escape hatch:
``EnableVectorizer=0`` turns ``IGCVectorizer`` off, with the two miscompiles below
:data:`IGC_OPTIONS` and a compiler crash on a Triton kernel. See :data:`IGC_OPTIONS`."""
XEVM_TARGET = '#xevm.target<O = 3, chip = "bmg">'
"""The one XeVM target of every GPU module: lighthouse's (``O = 3``, ``bmg``)."""
_NO_MODULE = "^$"
"""A ``gpu.module`` name regex that matches no module (names are not empty)."""


def extend_math() -> ir.Module:
    """Schedule: every GPU module gets exactly one target, :data:`XEVM_TARGET`,
    and its 16-bit math is extended to f32.

    Lighthouse's vector-to-XeGPU stage attached the target, and
    :func:`lower_to_xevm`'s pipeline would attach a second one (whose
    ``cmd-options`` flag nothing reads), so every kernel would be compiled
    twice. The pipeline converts fast (`afn`) math to native OpenCL calls before
    it extends bf16 math to f32 (math-to-xevm takes any float type), which
    leaves calls of bf16 natives that do not exist: extend first, with the
    pipeline's own options (f16 and f32 math stay)."""
    with schedule_boilerplate() as (schedule, named_seq):
        gpu_modules = match(named_seq.bodyTarget, ops={"gpu.module"})
        targets = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get([ir.Attribute.parse(XEVM_TARGET)]),
        )
        transform.AnnotateOp(gpu_modules, "targets", param=targets)
        apply_registered_pass(
            gpu_modules,
            "math-extend-to-supported-types",
            options={"extra-types": "f16", "target-type": "f32"},
        )
        transform.yield_()
    return schedule


def lower_to_xevm() -> ir.Module:
    """Schedule: annotated workgroup-level XeGPU lowered by upstream's
    ``gpu-lower-to-xevm-pipeline`` (lighthouse's ``xegpu_to_binary``), stopping
    before device binaries (``binary-format=skip``: no SPIR-V target needed; the
    XPU target emits them in a stage of its own). :func:`extend_math` ran before;
    the pipeline attaches no target (a module matcher matching no name). The IGC
    options reach the binaries through ``gpu-module-to-binary``'s ``opts``."""
    with schedule_boilerplate() as (schedule, named_seq):
        apply_registered_pass(
            named_seq.bodyTarget,
            "gpu-lower-to-xevm-pipeline",
            options={
                "xegpu-op-level": "workgroup",
                "binary-format": "skip",
                "xevm-module-matcher": _NO_MODULE,
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


def outline_kernels(
    entry: str, kernels: Sequence[KernelParams], columns: Sequence[bool] = ()
) -> ir.Module:
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
    Elementwise kernels' grid loops are mapped innermost dim first
    (``map_elementwise_grid``: grid x = the column tiles), and so are those of
    kernel ``i`` if ``columns[i]`` (``map_grid_columns_first``: the GEMMs
    ``xpu_params.columns_first`` picks); the others keep
    ``gpu-map-parallel-loops``' outermost-first order.
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
            if index < len(columns) and columns[index]:
                ht.map_grid_columns_first(parallel)
            else:
                ht.map_elementwise_grid(parallel)
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
    if kernel.derives_layouts:
        # The layout stage sets the count it derives (xegpu_layouts.set_threads).
        launch = _only(_kernel_ops(func, "gpu.launch", index))
        xegpu.set_gpu_launch_threads(launch, threads=[NB_WORKITEMS, 1, 1])
        return func
    force = list(kernel.sg_tile) if kernel.sg_tile is not None else None
    alternatives = lh_transform.alternatives(
        func, num_alternatives=5, result_types=[func.type]
    )
    for region, set_threads, sg_tile in (
        (0, set_gpu_threads_attention, _attention_sg_tile(kernel.sg_tile)),
        (1, set_gpu_threads_gemm, force),
        (2, set_gpu_threads_reduction, force),
    ):
        with alternatives.region(region) as scope:
            set_threads(_kernel_ops(scope, "gpu.launch", index), force_sg_tile=sg_tile)
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


def _attention_sg_tile(sg_tile: Sequence[int] | None) -> list[int] | None:
    """A forced ``[rows, 0]`` subgroup tile (the reduction form) in the form
    lighthouse's attention class takes: the query rows last, as in its tiles
    of the full-rank attention op (``[0, 0, rows]``). Its thread count reads
    the last entry, and ``annotate_attention`` drops leading zeros and appends
    the untiled head dim (``[rows, 0]``)."""
    if sg_tile is None or len(sg_tile) != 2 or sg_tile[1] != 0:
        return list(sg_tile) if sg_tile is not None else None
    return [sg_tile[0]]


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
    parameters, plus elementwise kernels (:func:`annotate_elementwise`).
    Kernels with IR-derived layouts are left to :func:`finish_ir_layouts`."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        for index, kernel in enumerate(kernels):
            if kernel.derives_layouts:
                continue
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


def finish_ir_layouts(
    entry: str, finished: Sequence[tuple[int, ht.xegpu_layouts.Plan]]
) -> ir.Module:
    """Schedule, after :func:`~ht.xegpu_layouts.annotate_layouts` of the kernels
    ``finished`` (``(kernel index, plan)``): the plans' prefetches (upstream's
    ``insert_prefetch``, then their anchors, as lighthouse's annotation) and the
    clean-up; the entry's constants settle after the thread counts changed."""
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        for index, plan in finished:
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
            for number, (_, values) in enumerate(plan.prefetches):
                load = _only(
                    match(
                        gpu_func,
                        ops={"xegpu.load_nd"},
                        op_attrs={ht.xegpu_layouts.PREFETCH_ATTR: _i64_attr(number)},
                    )
                )
                depth, *rest = values
                sg_layout, sg_data, inst_data = rest[:2], rest[2:4], rest[4:6]
                desc = xegpu.insert_prefetch(load, nb_prefetch=depth)
                prefetches = transform.get_consumers_of_result(anytype, desc, 0)
                xegpu.set_anchor_layout(
                    prefetches,
                    sg_layout=sg_layout,
                    sg_data=sg_data,
                    inst_data=inst_data,
                )
            if plan.cleanup:
                lh_transform.cleanup(gpu_func)
            if plan.licm:
                dpas = lh_transform.match_op(gpu_func, "xegpu.dpas")
                loops = transform.get_parent_op(
                    anytype, dpas, op_name="scf.for", deduplicate=True
                )
                transform.apply_licm(loops)
                lh_transform.cleanup(gpu_func)
        entry_func = _only(
            match(
                named_seq.bodyTarget,
                ops={"func.func"},
                op_attrs={"sym_name": ir.StringAttr.get(entry)},
            )
        )
        lh_transform.cleanup(entry_func)
        transform.yield_()
    return schedule


def _i64_attr(value: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)


def _annotate_kernel(gpu_func: ir.Value, kernel: KernelParams) -> None:
    """Lighthouse's ``analyze_and_annotate_gpu_func`` (attention, GEMM and
    reduction kernels, in that order and as it builds them; reductions also get
    their row-result stores anchored), reductions storing only row results,
    elementwise kernels, then a definite failure."""
    options = kernel.annotate_options()
    sg_tile = options.pop("sg_tile")
    alternatives = lh_transform.alternatives(
        gpu_func,
        num_alternatives=7,
        result_types=[gpu_func.type],
    )
    with alternatives.region(0) as scope:
        dpas_ops = lh_transform.match_op(scope, "xegpu.dpas")
        dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_attention(scope, dpas_op, force_sg_tile=_attention_sg_tile(sg_tile))
        transform.yield_([scope])
    with alternatives.region(1) as scope:
        # Sibling GEMMs (xpu_params.is_sibling_gemms): the second DPAS is the
        # match; annotate_gemm lays out one DPAS, its loads and every store, so
        # it runs once per DPAS, on handles matched again after its cleanup.
        dpas_ops = lh_transform.match_op(scope, "xegpu.dpas")
        transform_ext.extract_handle(dpas_ops, 1, silenceable=True)
        for index in (0, 1):
            dpas_ops = lh_transform.match_op(scope, "xegpu.dpas")
            dpas_op = transform_ext.extract_handle(dpas_ops, index, silenceable=True)
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
    with alternatives.region(3) as scope:
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_reduction(scope, anchor_op, force_sg_tile=sg_tile)
        annotate_row_result_stores(scope, anchor_op, sg_tile)
        annotate_reduction_tiles(scope, anchor_op, kernel)
        transform.yield_([scope])
    with alternatives.region(4) as scope:
        # A reduction storing only its row results (no 2D tile store).
        anchor_op = lh_transform.match_op(scope, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_row_result_stores(scope, anchor_op, sg_tile, require=True)
        annotate_reduction_tiles(scope, anchor_op, kernel)
        transform.yield_([scope])
    with alternatives.region(5) as scope:
        annotate_elementwise(scope, kernel)
        transform.yield_([scope])
    with alternatives.region(6) as scope:
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
