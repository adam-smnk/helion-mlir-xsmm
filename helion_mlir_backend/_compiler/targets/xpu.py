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
target, execution the Level Zero runtime (``libmlir_levelzero_runtime.so``). The
``binary`` stop point lowers to ``xevm``, then rewrites the XeVM modules'
counted allocas (``xevm_alloca_arrays``, a SPIR-V backend workaround), the bf16
extensions (``xevm_bf16_extensions``) and f32 ``fabs`` (``xevm_fabs``, both IGC
workarounds), orders long tile math (``xevm_tile_order``), and emits the binaries: the ``xevm`` stop point's IR precedes
those rewrites.
"""

from __future__ import annotations

import ctypes
import functools
import operator
import os
import re
from typing import TYPE_CHECKING
import warnings

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
from ..mlir.support.debug import LOWERING_WARNINGS_ATTR
from ..mlir.support.debug import XPU_STOP_POINTS
from ..mlir.support.debug import DebugOptions
from ..mlir.support.debug import xpu_stop_at
from ..mlir.support.errors import UnsupportedOperationError
from ..mlir.support.errors import UnsupportedTargetError
from ..mlir_transforms import partial_tiles
from ..mlir_transforms import xegpu_layouts
from ..mlir_transforms.keep_row_dims import keep_row_dims
from ..mlir_transforms.legalize_host_ops import HOST_OPS_ERROR
from ..mlir_transforms.legalize_host_ops import KERNEL_ARGUMENT
from ..mlir_transforms.prints import guard_prints
from ..mlir_transforms.prints import prints_of_stored_buffers
from ..mlir_transforms.prints import warn_print_is_experimental
from ..mlir_transforms.prints import refuse_prints_needing_allocation
from ..mlir_transforms.row_maxima import restore_nan_propagating_maxima
from ..mlir_transforms.xegpu_attention_loops import copy_attention_layouts
from ..mlir_transforms.xegpu_block_surfaces import check_block_surfaces
from ..mlir_transforms.xevm_alloca_arrays import alloca_arrays
from ..mlir_transforms.xevm_bf16_extensions import bf16_extensions_as_shifts
from ..mlir_transforms.xevm_fabs import fabs_as_integer_min
from ..mlir_transforms.xevm_lane_compares import compare_lanes
from ..mlir_transforms.xevm_lane_conversions import convert_lanes_where_computed
from ..mlir_transforms.xevm_lane_reductions import check_lane_reductions
from ..mlir_transforms.xevm_lane_reductions import check_transposed_reads
from ..mlir_transforms.xevm_nsz_minmax import nsz_minmax
from ..mlir_transforms.xevm_tile_order import ORDER_FIRST_ABOVE
from ..mlir_transforms.xevm_tile_order import order_tile_chunks
from ..mlir_transforms.xevm_tile_order import spill_bytes
from ..mlir_transforms.xevm_tile_order import tile_math_grfs
from ..mlir_transforms.xevm_unit_clamps import unit_clamps_without_sat
from ..mlir_transforms.xpu_bool_casts import select_bool_casts
from ..mlir_transforms.xpu_math_approx import approximate_math
from ..mlir_transforms.xpu_math_approx import fold_extended_splats
from .base import MLIRTarget
from .base import exit_before_teardown
from .xpu_checks import check_partial_gemms
from .xpu_checks import check_xegpu_capabilities
from .xpu_params import XPU_CONFIG_KEYS
from .xpu_params import KernelParams
from .xpu_params import attach_params
from .xpu_params import check_params
from .xpu_params import columns_first
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
_DEVICE_FDS = "/dev/dri/"
_PIPELINES = ("xegpu",)
_CPU = "target='cpu' (CPU tensors)"
_NO_PATTERN = "Could not apply any of the defined patterns"
"""Lighthouse's failure when no kernel class (GEMM, attention, reduction, or the
backend's elementwise class) matches."""
_IR_ONLY = "HELION_MLIR_XPU_STOP_AT=xegpu-wg, xevm or xevm-final (IR only)"
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
            module = _apply_stage(name, stage, module, self.dump)
        return module


def _apply_stage(name: str, stage: Stage, module: ir.Module, dump: bool) -> ir.Module:
    """``stage`` applied to ``module``, its failure noted with its ``name``."""
    try:
        module = stage.apply(module)
    except Exception as exc:
        raise _translated(exc, name) from exc
    if dump:
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
    if "2D block surface pitch must be" in message:
        return UnsupportedOperationError(
            "a tensor's rows as a surface of 2D block accesses",
            reason="its row size in bytes is no multiple of 16, the pitch they need",
            alternatives=[
                "row sizes that are a multiple of 16 bytes",
                "one-row tiles of an elementwise kernel (block size 1 on rows)",
                _CPU,
            ],
        )
    if "failed to legalize operation 'xegpu.load'" in message:
        return UnsupportedOperationError(
            "a read as one of XeGPU's gathers",
            reason=(
                "the gather has no legal lowering here; a 1D tensor read per tile at "
                "a partial tile is such a form"
            ),
            alternatives=["block sizes that divide the sizes they tile", _CPU],
        )
    exc.add_note(f"Helion MLIR backend: failed in the XPU stage '{stage}'")
    return exc


def _stages(
    entry: str,
    kernels: Sequence[KernelParams],
    stop_at: str,
    columns: Sequence[bool] = (),
) -> list[tuple[str, Stage]]:
    """The XeGPU stage list of ``entry``, whose tagged grid loops are GPU kernels
    with parameters ``kernels``, up to ``stop_at``; ``columns[i]``: kernel
    ``i``'s grid is mapped columns first (:func:`columns_first`). Lighthouse's
    dialects must be loaded (schedules use its transform ops)."""
    context = ir.Context.current
    schedules = [
        # "prepare" ran before the capability checks (XpuTarget.lowering).
        # The entry by name: lighthouse otherwise finds it by a linalg.generic
        # or matmul, which a tile copy (a linalg.copy) does not have.
        ("cleanup", xegpu.cleanup_schedule(entry)),
        # No wg_tiling_schedule: Helion's grid loops are the workgroup tiles.
        ("vectorized", xegpu.vectorize_schedule(entry)),
        # Undoes lighthouse's maxnumf reductions: see row_maxima.
        ("nan-maxima", _NanMaxima(kernels)),
        ("bufferized", xegpu.bufferize_schedule(entry)),
        ("partial-tiles", _FoldTileViews()),
        ("host-legalized", xpu_transforms.legalize_host_ops()),
        ("grid-mappings", _StripGridMappings()),
        # Lighthouse's outline_gpu_func and annotate_layouts schedules, per kernel.
        ("gpu-outlining", xpu_transforms.outline_kernels(entry, kernels, columns)),
        # Prints a kernel can't do, or may do wrong: see prints.
        ("print-reads", _RefusePrintsNeedingAllocation()),
        # Row results stored as [rows, 1] columns: see keep_row_dims.
        ("row-results", _KeepRowDims()),
        # A SPIR-V backend abort on bool to float casts: see xpu_bool_casts.
        ("bool-casts", _BoolCasts()),
        ("xegpu-initial", xegpu.vector_to_xegpu_schedule()),
        ("xegpu-wg", xpu_transforms.annotate_kernels(entry, kernels)),
    ]
    if stop_at != "xegpu-wg":
        schedules.extend(
            [
                ("xevm-math", xpu_transforms.extend_math()),
                ("xevm-math-approx", _ApproximateMath()),
                ("xevm", xpu_transforms.lower_to_xevm()),
            ]
        )
    stages: list[tuple[str, Stage]] = [
        (name, s if isinstance(s, Stage) else TransformStage(s, context))
        for name, s in schedules
    ]
    # The annotated kernels' 2D block surfaces are checked against the
    # hardware limits (xegpu_block_surfaces), as part of the stage.
    index = next(i for i, (name, _) in enumerate(stages) if name == "xegpu-wg")
    stages[index] = (
        "xegpu-wg",
        _CheckBlockSurfaces(_IRLayouts(stages[index][1], entry, kernels)),
    )
    if stop_at != "xegpu-wg":
        # The reductions as lowered are checked (xevm_lane_reductions).
        index = next(i for i, (name, _) in enumerate(stages) if name == "xevm")
        lowering = _CheckLaneReductions(stages[index][1])
        if any(kernel.derives_layouts for kernel in kernels):
            lowering = _LayoutWarningsAreErrors(lowering)
        stages[index] = ("xevm", lowering)
    if stop_at in ("xevm-final", "binary"):
        # Between the LLVM/XeVM lowering and the binary: see xevm_alloca_arrays,
        # xevm_nsz_minmax, xevm_unit_clamps, xevm_bf16_extensions, xevm_fabs,
        # xevm_lane_compares and xevm_lane_conversions;
        # _ToBinary for xevm_tile_order.
        stages.extend(
            [
                ("xevm-allocas", _AllocaArrays()),
                ("xevm-nsz-minmax", _NszMinMax()),
                # Before the extensions: it looks through them.
                ("xevm-unit-clamps", _UnitClamps()),
                ("xevm-bf16-extensions", _Bf16Extensions()),
                ("xevm-fabs", _FabsAsIntegerMin()),
                ("xevm-lane-compares", _CompareLanes()),
                ("xevm-lane-conversions", _LaneConversions()),
            ]
        )
    if stop_at == "binary":
        stages.append(("binary", _ToBinary(_kernels_by_module(entry, kernels))))
    return stages


def _kernels_by_module(
    entry: str, kernels: Sequence[KernelParams]
) -> dict[str, KernelParams]:
    return {
        xpu_transforms.kernel_module_name(entry, index): kernel
        for index, kernel in enumerate(kernels)
    }


class _BoolCasts(Stage):
    """Bool to float casts as selects (:mod:`xpu_bool_casts`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        select_bool_casts(module)
        return module


class _NanMaxima(Stage):
    """Row maxima that propagate NaN, but what ``xegpu_nan_ignoring_maxima`` allows
    (:mod:`row_maxima`)."""

    def __init__(self, kernels: Sequence[KernelParams]) -> None:
        self.ignoring = frozenset(
            index
            for index, kernel in enumerate(kernels)
            if kernel.nan_ignoring_maxima is not None
        )

    def apply(self, module: ir.Module) -> ir.Module:
        restore_nan_propagating_maxima(
            module, self.ignoring, xpu_transforms.KERNEL_ATTR
        )
        return module


class _ToBinary(Stage):
    """The last pass of ``gpu-lower-to-xevm-pipeline`` (run with ``binary-format=
    skip`` before), with its options: device binaries (fatbin), each GPU module with
    its kernel's IGC options (``igc_vectorizer``). A GPU module that spills is
    compiled again with its tile math ordered chunk by chunk, and that binary is
    kept if it spills less; a failure there keeps the first binary
    (:mod:`xevm_tile_order`). A module of very long tile math is expected to spill:
    it is compiled ordered first, and plain only if that still spills. Decided per
    module: one kernel's spill does not change another's code."""

    def __init__(self, kernels: Mapping[str, KernelParams] | None = None) -> None:
        self.kernels = kernels or {}

    def apply(self, module: ir.Module) -> ir.Module:
        to_binary = _BinaryPasses(module.context, self.kernels)
        gpu_modules = [
            op.operation
            for op in module.body.operations
            if op.operation.name == "gpu.module"
        ]
        long_math = [m for m in gpu_modules if _expected_to_spill(m)]
        ordered = _compile_ordered(to_binary, long_math)
        plain = [
            gpu_module
            for gpu_module in gpu_modules
            if _symbol(gpu_module) not in ordered
            or spill_bytes(ordered[_symbol(gpu_module)])
        ]
        binaries = _compile(to_binary, plain)
        spilling = [
            gpu_module
            for gpu_module in plain
            if gpu_module not in long_math
            and spill_bytes(binaries[_symbol(gpu_module)])
        ]
        ordered.update(_compile_ordered(to_binary, spilling))
        for name, binary in ordered.items():
            if name not in binaries or spill_bytes(binary) < spill_bytes(
                binaries[name]
            ):
                binaries[name] = binary
        for gpu_module in gpu_modules:
            ir.InsertionPoint(gpu_module).insert(binaries[_symbol(gpu_module)])
            gpu_module.erase()
        return module


def _expected_to_spill(gpu_module: ir.Operation) -> bool:
    return tile_math_grfs(gpu_module) >= ORDER_FIRST_ABOVE


def _symbol(op: ir.Operation) -> str:
    return ir.StringAttr(op.attributes["sym_name"]).value


class _BinaryPasses:
    """``gpu-module-to-binary`` for each GPU module with its kernel's IGC options."""

    def __init__(
        self, context: ir.Context, kernels: Mapping[str, KernelParams]
    ) -> None:
        self.context = context
        self.kernels = kernels
        self.passes: dict[str, PassManager] = {}

    def run(self, operation: ir.Operation, name: str) -> None:
        kernel = self.kernels.get(name, KernelParams())
        options = (
            xpu_transforms.IGC_OPTIONS_OFF
            if kernel.vectorizer_off or _narrows_f64(operation)
            else xpu_transforms.IGC_OPTIONS
        )
        if options not in self.passes:
            self.passes[options] = PassManager.parse(
                f"builtin.module(gpu-module-to-binary{{format=fatbin opts={{{options}}}}})",
                self.context,
            )
        self.passes[options].run(operation)


def _narrows_f64(operation: ir.Operation) -> bool:
    """An ``fptrunc`` of f64, scalar or vector, in the module. ``IGCVectorizer`` makes
    ``fptrunc <8 x double>`` of the scalar ones too, and rows 4 to 7 of each 8 come out wrong,
    with no diagnostic (``torch.abs(t.to(f64)).to(f32)``, ``torch.sqrt(t.to(f64) * t.to(f64))``:
    half of the elements on the B50, IGC 2.41; right with the vectorizer off). No benchmark
    kernel has f64 in a GPU module. See ``agent-wiki/upstream-repro/igc-vector-fabs/``."""
    found: list[bool] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "llvm.fptrunc":
            source = op.operands[0].type
            element = (
                ir.VectorType(source).element_type
                if isinstance(source, ir.VectorType)
                else source
            )
            if isinstance(element, ir.F64Type):
                found.append(True)
                return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return bool(found)


def _compile(
    to_binary: _BinaryPasses,
    gpu_modules: Sequence[ir.Operation],
    *,
    ordered: bool = False,
) -> dict[str, ir.Operation]:
    """The ``gpu.binary`` of each of ``gpu_modules`` (left as they are), by name,
    each compiled in a module of its own."""
    binaries: dict[str, ir.Operation] = {}
    for gpu_module in gpu_modules:
        holder = ir.Module.create()
        holder.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
        ir.InsertionPoint(holder.body).insert(gpu_module.clone())
        if ordered:
            order_tile_chunks(holder)
        to_binary.run(holder.operation, _symbol(gpu_module))
        for op in holder.body.operations:
            if op.operation.name == "gpu.binary":
                binary = op.operation
                # Detached: it outlives the holder.
                binary.detach_from_parent()
                binaries[_symbol(binary)] = binary
                break
    return binaries


def _compile_ordered(
    to_binary: _BinaryPasses, gpu_modules: Sequence[ir.Operation]
) -> dict[str, ir.Operation]:
    """:func:`_compile` with the tile math ordered, or nothing if that fails: the
    caller then has, or compiles, the plain binary."""
    if not gpu_modules:
        return {}
    try:
        return _compile(to_binary, gpu_modules, ordered=True)
    except Exception as exc:
        names = [_symbol(gpu_module) for gpu_module in gpu_modules]
        warnings.warn(
            f"XPU kernels {names} could not be compiled in chunk order: {exc}",
            stacklevel=2,
        )
        return {}


class _FoldTileViews(Stage):
    """Out-of-bounds transfers of partial tiles on the true surface:
    :func:`partial_tiles.fold_tile_views`."""

    def apply(self, module: ir.Module) -> ir.Module:
        partial_tiles.fold_tile_views(module)
        return module


class _StripGridMappings(Stage):
    """The placeholder mappings that kept grid loops of one tile, right before
    outlining: :mod:`~..mlir.lowering.loops`."""

    def apply(self, module: ir.Module) -> ir.Module:
        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "scf.forall" and "mapping" in op.attributes:
                del op.attributes["mapping"]
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        return module


class _KeepRowDims(Stage):
    """``[rows]`` row-result stores as ``[rows, 1]`` (works around wrong
    scatter stores of row vectors on hardware, :mod:`keep_row_dims`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        keep_row_dims(module)
        return module


def _check_dpas_anchored(module: ir.Module) -> None:
    """A DPAS without a layout lowers with no diagnostic, upstream's propagation
    completes it, so an annotation that handled one DPAS of two is refused here."""
    unanchored: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "xegpu.dpas" and "layout_a" not in op.attributes:
            unanchored.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    if unanchored:
        raise UnsupportedOperationError(
            "a DPAS the layout annotation left without a layout on XPU",
            reason=f"no kernel class laid out {len(unanchored)} of the kernel's DPAS",
            alternatives=["one matmul per top-level hl.tile loop", _CPU],
        )


class _IRLayouts(Stage):
    """``stage`` (lighthouse's annotation), then the kernels whose layouts are
    derived from the IR (:mod:`xegpu_layouts`), each with its config as hints on
    its ``gpu.func``."""

    def __init__(
        self, stage: Stage, entry: str, kernels: Sequence[KernelParams]
    ) -> None:
        self.stage = stage
        self.entry = entry
        self.kernels = kernels

    def apply(self, module: ir.Module) -> ir.Module:
        module = self.stage.apply(module)
        # Lighthouse's kernels: the IR-derived ones have no anchors yet, so it leaves
        # them alone, and they anchor every K/V loop by its own tiles.
        copy_attention_layouts(module)
        funcs = {
            index: _kernel_func(
                module, xpu_transforms.kernel_module_name(self.entry, index)
            )
            for index, kernel in enumerate(self.kernels)
            if kernel.derives_layouts
        }
        if not funcs:
            return module
        finished: list[tuple[int, xegpu_layouts.Plan]] = []
        for index, func in funcs.items():
            kernel = self.kernels[index]
            func.attributes[xegpu_layouts.HINTS_ATTR] = _layout_hints(kernel)
            try:
                plan = xegpu_layouts.annotate_layouts(module.operation, func)
            except (xegpu_layouts.LayoutError, ValueError) as exc:
                raise _layout_failure(index, exc) from exc
            finished.append((index, plan))
        stage = TransformStage(
            xpu_transforms.finish_ir_layouts(self.entry, finished), module.context
        )
        module = stage.apply(module)
        for index in funcs:
            func = _kernel_func(
                module, xpu_transforms.kernel_module_name(self.entry, index)
            )
            xegpu_layouts.clear_tags(func)
        problems = xegpu_layouts.check_module(
            module,
            [
                _symbol(
                    _kernel_func(
                        module, xpu_transforms.kernel_module_name(self.entry, index)
                    ).parent.operation
                )
                for index in funcs
            ],
        )
        if problems:
            raise UnsupportedOperationError(
                "a kernel without valid XeGPU layouts",
                reason="; ".join(problems[:6]),
                alternatives=['xegpu_layouts="lighthouse"', _CPU],
            )
        return module


def _kernel_func(module: ir.Module, name: str) -> ir.Operation:
    for op in module.body.operations:
        if op.operation.name == "gpu.module" and _symbol(op.operation) == name:
            for nested in op.operation.regions[0].blocks[0].operations:
                if nested.operation.name == "gpu.func":
                    return nested.operation
    raise ValueError(f"no GPU kernel module named '{name}'")


def _layout_hints(kernel: KernelParams) -> ir.Attribute:
    hints: dict[str, ir.Attribute] = {}
    for key, value in (
        ("sg_tile", kernel.sg_tile),
        ("load_tiles", kernel.load_tiles),
        ("prefetch_tiles", kernel.prefetch_tiles),
        ("prefetch_depths", kernel.prefetch_depths),
    ):
        if value is not None:
            hints[key] = ir.DenseI64ArrayAttr.get(list(value))
    return ir.DictAttr.get(hints)


def _layout_failure(index: int, exc: Exception) -> Exception:
    return UnsupportedOperationError(
        f"GPU kernel {index}: no XeGPU layout derived from its IR",
        reason=str(exc),
        alternatives=['xegpu_layouts="lighthouse"', _CPU],
    )


class _LayoutWarningsAreErrors(Stage):
    """``stage``, failing on a layout warning from its passes. Upstream only warns
    when a layout a 2D block load cannot take reaches ``visitLoadNdOp``, then
    lowers wrong rows (``upstream-issues-xpu.md``). For kernels with IR-derived
    layouts, which must not depend on what the warning goes on to do."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage

    def apply(self, module: ir.Module) -> ir.Module:
        warnings_seen: list[str] = []

        def on_diagnostic(diagnostic: ir.Diagnostic) -> bool:
            if (
                diagnostic.severity == ir.DiagnosticSeverity.WARNING
                and "layout" in diagnostic.message
            ):
                warnings_seen.append(diagnostic.message)
                return True
            return False

        handler = module.context.attach_diagnostic_handler(on_diagnostic)
        try:
            module = self.stage.apply(module)
        finally:
            handler.detach()
        if warnings_seen:
            raise UnsupportedOperationError(
                "a kernel whose XeGPU layouts the lowering could not complete",
                reason="; ".join(sorted(set(warnings_seen))[:4]),
                alternatives=[_CPU],
            )
        return module


class _CheckBlockSurfaces(Stage):
    """``stage``, then a failure on 2D block descriptors of surfaces beyond the
    hardware limits (silently wrong rows, :mod:`xegpu_block_surfaces`) and on DPAS
    ops the layout annotation left out."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage

    def apply(self, module: ir.Module) -> ir.Module:
        module = self.stage.apply(module)
        check_block_surfaces(module)
        _check_dpas_anchored(module)
        return module


class _CheckLaneReductions(Stage):
    """``stage``, then a failure on row reductions over part of a subgroup
    (silently wrong rows, :mod:`xevm_lane_reductions`)."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage

    def apply(self, module: ir.Module) -> ir.Module:
        module = self.stage.apply(module)
        check_lane_reductions(module)
        check_transposed_reads(module)
        return module


class _AllocaArrays(Stage):
    """Constant-count allocas of the XeVM modules as array allocas (works around
    a SPIR-V backend miscompile, :mod:`xevm_alloca_arrays`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        alloca_arrays(module)
        return module


class _NszMinMax(Stage):
    """Device minima and maxima without the order of -0 and +0, much cheaper
    in SPIR-V (:mod:`xevm_nsz_minmax`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        nsz_minmax(module)
        return module


class _UnitClamps(Stage):
    """Compares with 1 that keep IGC from forming a saturation of a clamp to
    [0, 1], which gets NaN, -0 and bf16 wrong (:mod:`xevm_unit_clamps`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        unit_clamps_without_sat(module)
        return module


class _Bf16Extensions(Stage):
    """bf16-to-f32 extensions as integer shifts (works around IGC narrowing f32
    math on them to bf16 math, which truncates, :mod:`xevm_bf16_extensions`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        bf16_extensions_as_shifts(module)
        return module


class _FabsAsIntegerMin(Stage):
    """f32 ``fabs`` in integer ops (works around IGC's no-op bf16 ``fabs``,
    :mod:`xevm_fabs`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        fabs_as_integer_min(module)
        return module


class _RefusePrintsNeedingAllocation(Stage):
    """A print that needs a buffer in the kernel is refused; one that can see other threads'
    stores gets a warning (:mod:`prints`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        refuse_prints_needing_allocation(module)
        warnings_found = prints_of_stored_buffers(module)
        if warnings_found:
            module.operation.attributes[LOWERING_WARNINGS_ATTR] = ir.ArrayAttr.get(
                [ir.StringAttr.get(line) for line in warnings_found]
            )
        return module


class _CompareLanes(Stage):
    """Vector compares lane by lane (works around IGC's one-row ``fabs`` and ``abs``,
    :mod:`xevm_lane_compares`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        compare_lanes(module)
        return module


class _LaneConversions(Stage):
    """A narrowing conversion of a vector built by lanes, lane by lane (works around
    IGC's vectorized cast before its operand, :mod:`xevm_lane_conversions`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        convert_lanes_where_computed(module)
        return module


class _ApproximateMath(Stage):
    """``tanh`` and ``log1p`` as branch-free ops instead of a builtin call per
    element, and constants of extended math folded (:mod:`xpu_math_approx`)."""

    def apply(self, module: ir.Module) -> ir.Module:
        fold_extended_splats(module)
        approximate_math(module)
        return module


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


_device_of = operator.attrgetter("device")


def _runtime_library() -> str | None:
    """The Level Zero runtime library of the MLIR bindings, if there is one."""
    try:
        path = get_mlir_library_path() / _RUNTIME_LIB
    except (AssertionError, ValueError):
        return None
    return str(path) if path.is_file() else None


def _close_device_fds() -> None:
    """Closes this process's descriptors of GPU devices; the parent keeps its own."""
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{name}").startswith(_DEVICE_FDS):
                os.close(int(name))
        except OSError:
            pass  # The descriptor of the listing itself.


class XpuTarget(MLIRTarget):
    """Kernels compiled to Intel XPUs (XeGPU, Battlemage) through lighthouse.

    One pipeline, ``xegpu``; lowerings can stop at ``xegpu-wg``, ``xevm`` or
    ``binary`` (``HELION_MLIR_XPU_STOP_AT`` sets the default, else ``binary``).
    """

    name = "xpu"
    pipelines = _PIPELINES
    config_keys = XPU_CONFIG_KEYS
    stop_points = XPU_STOP_POINTS

    phases_see_aliases = True
    buffer_alignment = 64
    """Tensor data is passed 64-byte aligned (misaligned views, e.g.
    ``x[1:]``, as aligned copies; ``call_entry``). The kernels need it: a 2D
    block load or store takes a 64-byte aligned surface base address (Xe2
    ``uArch/IntelGpuXe2.h``; violated, the access is undefined: wrong data,
    no error), and the gathers and scatters of several elements per work item
    of 1D tiles become LLVM vector loads and stores without an alignment, so
    the vector type's ABI alignment (16 bytes for ``vector<8xbf16>``) is
    assumed. Measured on the Arc B50 (2026-10-31): a bf16 relu on a view 2
    bytes off computed 75 % of its outputs wrong in 1D and one-row tiles and
    also with 2D 128x256 tiles; torch's caching allocator returns aligned
    blocks, so only views pay for the copy."""

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

    def enter_lowering_child(self) -> None:
        """Keeps this forked process's exit away from the parent's GPU state.
        ``LLVM ERROR`` ends in C ``exit``, whose destructors (Level Zero objects of
        the runtime wrappers, the driver's own teardown) would run on device
        objects and descriptors shared with the parent: unbinds, closed buffers
        and destroyed queues, after which the parent's next kernel stalls or gets
        ``ZE_RESULT_ERROR_DEVICE_LOST``. A lowering never uses the device. Closing
        the descriptors also protects the parent, but the driver then aborts on
        the failed calls and the exit code is lost, so it is only the fallback."""
        if not exit_before_teardown():
            _close_device_fds()

    def lowering(
        self,
        module: ir.Module,
        entry: str,
        pipeline: str,
        stop_at: str | None = None,
    ) -> XpuLowering:
        """The XeGPU stage list after ``prepare``, which is applied to
        ``module`` (inlined) here: it folds unit tile dims, so the checks see
        each grid loop as its lower-rank kernel (block size 1 on leading dims).
        Then checks that ``module`` is in what the pipeline compiles
        (:func:`check_xegpu_capabilities`) and that its recorded XeGPU
        parameters, forced or inferred, fit its kernels (:func:`check_params`):
        configs that cannot lower fail here, before any other lighthouse
        stage. Tags each grid loop (GPU kernel) with its index for the
        per-kernel stages."""
        self.resolve_pipeline(pipeline)
        self.check_stop_at(stop_at)
        dump = DebugOptions.from_env().dump_stages
        # One line per workgroup: before outlining, which needs the guard's region.
        if guard_prints(module):
            warn_print_is_experimental(entry)
        # Loads lighthouse's dialects, which building the schedules needs.
        BackendDriver(module, entry, result_to_args=False, benchmark=False)
        if partial_tiles.to_transfers(module):
            PassManager.parse(
                "builtin.module(canonicalize,cse)", ir.Context.current
            ).run(module.operation)
        prepare = TransformStage(xpu_transforms.prepare_for_xegpu(), ir.Context.current)
        _apply_stage("prepare", prepare, module, dump)
        params = take_params(module)
        loops = check_xegpu_capabilities(
            module,
            entry,
            [kernel.derives_layouts for kernel in params],
            KernelParams().derives_layouts,
        )
        check_partial_gemms(
            loops,
            [
                (
                    params[index] if index < len(params) else KernelParams()
                ).derives_layouts
                for index in range(len(loops))
            ],
        )
        shapes = [kernel_shape(loop) for loop in loops]
        check_params(params, shapes)
        i64 = ir.IntegerType.get_signless(64)
        for index, loop in enumerate(loops):
            loop.attributes[xpu_transforms.KERNEL_ATTR] = ir.IntegerAttr.get(i64, index)
        kernels = [*params, *[KernelParams()] * (len(loops) - len(params))]
        columns = [columns_first(shape) for shape in shapes]
        stages = _stages(entry, kernels, stop_at or self.stop_points[-1], columns)
        return XpuLowering(stages, dump)

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

    hooks_in_place = True
    """The per-call Python a kernel's launch waits for grows with its arguments, and
    a layer's 39 cost more than 200 us on the general path, before its first launch."""

    def passes_in_place(self, tensor: torch.Tensor) -> bool:
        return tensor.is_xpu

    def prepare_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """Wait for torch's work on the current stream (e.g. host-code
        ``torch.zeros``): the kernels run on the Level Zero runtime's own queue.
        Work on another stream isn't waited for: as with torch's own ops, users
        synchronize across streams.

        A ``query`` is 1.4 us where an idle ``synchronize`` is 25.5, half of a tiny
        kernel's call. On the Arc B50, no wait at all still gave right results after
        compute work (180 of 180 calls after a matmul; the call took the matmul's full
        time): probably one ring per engine, in submission order (inferred). After a
        non-blocking copy from pinned host memory, on the copy engine, no wait read the
        old data in 60 of 60 calls, so this wait stays. Untested on B70."""
        devices = {
            device for device in set(map(_device_of, buffers)) if device.type == "xpu"
        }
        if not devices:
            return
        if len(devices) > 1:
            raise ValueError(
                "the kernel's tensors are on several devices: "
                f"{sorted(map(str, devices))}"
            )
        device = devices.pop()
        index = device.index if device.index is not None else 0
        if not torch.xpu.current_stream(index).query():
            torch.xpu.synchronize(device)
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
        """Lowering short of the binary needs nothing; binaries need
        bindings with the SPIR-V target, running also an XPU and the Level Zero
        runtime library."""
        if stop_at in ("xegpu-wg", "xevm", "xevm-final"):
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
