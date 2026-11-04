"""IR invariants checked on the lowerings the suite already does.

:func:`install` wraps the module builder (``MLIRModuleBuilder.build``, every kernel call
and ``generate_mlir``) and each XPU stage (``_apply_stage``), for tests only: ``tests/conftest.py`` sets
``HELION_MLIR_TEST_IR_CHECKS``. Its values:

- ``fail`` (the suite's default): a violation raises, failing the test;
- ``report:PATH``: each violation is one JSON line in ``PATH``, and the test goes on
  (one run lists them all);
- ``off``.

A kernel that means to compute in f64 says so: an f64 tensor argument, or ``float64`` or
``double`` as a name in its code (``torch.float64``, ``.double()``; a word in a comment or
docstring doesn't count). A lowering runs once per kernel and signature, so a declaration on
one test would depend on the order of tests.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

from mlir import ir

from tests.ir_query import walk

from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder
from helion_mlir_backend._compiler.targets import xpu as xpu_target

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

MODE_VARIABLE = "HELION_MLIR_TEST_IR_CHECKS"
MARK_STAGES_VARIABLE = "HELION_MLIR_TEST_MARK_STAGES"
EXPECTED_VARIABLE = "HELION_MLIR_TEST_EXPECTED_VIOLATION"
"""``check:path``: set by ``@pytest.mark.expect_violation(check)``: a violation of that check
is noted in ``path``, not raised, and the test fails if none is."""

XEGPU_MEMORY_OPS = {
    "xegpu.load_nd": ("layout",),
    "xegpu.store_nd": ("layout",),
    "xegpu.prefetch_nd": ("layout",),
    "xegpu.load": ("layout",),
    "xegpu.store": ("layout",),
    "xegpu.prefetch": ("layout",),
    "xegpu.dpas": ("layout_a", "layout_b", "layout_cd"),
}
"""The layout attributes each op must carry at ``xegpu-wg``."""

_F64_DECLARED = {"float64", "double"}
"""Names in the kernel's code (not in its comments or docstring) that ask for f64."""
_F64_HELPERS = ("_aten_var_correction_", "_aten_var_mean_correction_")
"""torch-mlir's decomposition of ``var`` and ``std`` computes in f64, whatever the tile's type;
the kernel's source can't say otherwise. Helion stops these kernels on the XPU before any IR."""
_F64_HARMLESS = {"arith.constant", "func.func", "func.return", "func.call"}
"""Only values: what uses them is what computes in f64."""


class IRInvariantError(AssertionError):
    """The IR breaks an invariant (a test-only failure)."""


def _element_type(value_type: ir.Type) -> ir.Type:
    return (
        ir.ShapedType(value_type).element_type
        if isinstance(value_type, ir.ShapedType)
        else value_type
    )


def _is_f64(value: ir.Value) -> bool:
    return isinstance(_element_type(value.type), ir.F64Type)


def _narrows_a_scalar(op: ir.Operation) -> bool:
    """A scalar parameter or constant taken out of f64: no data is widened."""
    return op.name == "arith.truncf" and not isinstance(
        op.operands[0].type, ir.ShapedType
    )


def _helper_ops(module: ir.Module) -> Iterator[ir.Operation]:
    """The ops under ``module``, but those of the functions ``_F64_HELPERS`` names."""
    for op in module.body.operations:
        operation = op.operation
        name = (
            str(operation.attributes["sym_name"])
            if operation.name == "func.func"
            else ""
        )
        if not name.strip('"').startswith(_F64_HELPERS):
            yield from walk(operation)


def f64_ops(module: ir.Module) -> list[str]:
    """The ops computing in f64 (a bare constant or scalar argument does not)."""
    if "f64" not in str(module):
        return []
    return [
        str(op).split("\n", 1)[0].strip()
        for op in _helper_ops(module)
        if op.name not in _F64_HARMLESS
        and not _narrows_a_scalar(op)
        and any(_is_f64(value) for value in (*op.operands, *op.results))
    ]


def _declares_f64(module: ir.Module, function: Callable[..., object] | None) -> bool:
    for op in walk(module):
        # The entry: ATen helpers of a widened tile take f64 tensors without a declaration.
        if op.name == "func.func" and "public" in str(op.attributes["sym_visibility"]):
            arguments = op.regions[0].blocks[0].arguments
            if any(
                isinstance(arg.type, ir.ShapedType) and _is_f64(arg)
                for arg in arguments
            ):
                return True
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return False
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute | ast.Name)
    }
    return not names.isdisjoint(_F64_DECLARED)


def xegpu_ops_left(module: ir.Module) -> list[str]:
    """XeGPU ops, which nothing lowers after the ``xevm`` stage."""
    return sorted({op.name for op in walk(module) if op.name.startswith("xegpu.")})


def _broadcast_gather(op: ir.Operation) -> bool:
    """A 1D ``xegpu.load`` only broadcast to a tile, possibly converted on the way (a
    row's bias): upstream propagates that layout from the broadcast, on purpose
    (``xegpu_layouts.check_anchors``)."""
    if op.name != "xegpu.load":
        return False
    pending, uses = [op.results[0]], 0
    while pending:
        for use in pending.pop().uses:
            uses += 1
            user = use.owner
            if user.name in ("arith.extf", "arith.truncf"):
                pending.append(user.results[0])
            elif user.name != "vector.broadcast":
                return False
    return uses > 0


def ops_without_layout(module: ir.Module) -> list[str]:
    """XeGPU memory ops and DPAS missing a layout: left to upstream's defaults, whose
    answers are not always right (agent-wiki xpu-layouts)."""
    return [
        f"{op.name} without {attribute}"
        for op in walk(module)
        if op.name in XEGPU_MEMORY_OPS and not _broadcast_gather(op)
        for attribute in XEGPU_MEMORY_OPS[op.name]
        if attribute not in op.attributes
    ]


def _vector_f16_fabs(op: ir.Operation) -> bool:
    """``vector<16xf16>`` or ``vector<8xf16>`` ``fabs``, exactly: ``xevm_fabs`` leaves it
    alone, and IGC scalarizes it, ``llvm.fabs.f16`` per lane in its final IR, and
    ``test_math_every_pattern``'s f16 kernels match fp64. This rests on IGC's dumps (IGC 2.41,
    helion-mlir-dev's agent-wiki/upstream-repro/igc-vector-fabs/f16-fabs-dumps.md), and
    ``test_vector_f16_fabs_stays_scalar_in_igc`` keeps it checked. Other lane counts fail the
    check:
    ``xevm_fabs`` reports ``llvm.fabs.v4f16`` wrong."""
    return op.name == "llvm.intr.fabs" and str(op.results[0].type) in (
        "vector<16xf16>",
        "vector<8xf16>",
    )


def vector_compares_and_abs(module: ir.Module) -> list[str]:
    """Vector ``llvm.fcmp``/``icmp``/``fabs``/``abs``: IGC's vectorizer gets them wrong
    (agent-wiki/upstream-repro/igc-vector-fabs/). Holds only after the ``xevm-*`` stages."""
    names = ("llvm.fcmp", "llvm.icmp", "llvm.intr.fabs", "llvm.intr.abs")
    return sorted(
        {
            f"{op.name} {op.results[0].type}"
            for op in walk(module)
            if op.name in names
            and isinstance(op.results[0].type, ir.ShapedType)
            and not _vector_f16_fabs(op)
        }
    )


_STAGE_CHECKS: dict[str, tuple[str, Callable[[ir.Module], list[str]]]] = {
    "xegpu-wg": ("layouts", ops_without_layout),
    "xevm": ("xegpu-left", xegpu_ops_left),
    "xevm-lane-conversions": ("vector-compare-or-abs", vector_compares_and_abs),
}


def _violation(check: str, where: str, findings: list[str]) -> None:
    expected, _, fired = os.environ.get(EXPECTED_VARIABLE, "").partition(":")
    if check == expected:
        with open(fired, "a") as log:
            log.write(f"{where}: {findings[:2]}\n")
        return
    mode = os.environ.get(MODE_VARIABLE, "off")
    if mode.startswith("report:"):
        line = json.dumps(
            {
                "check": check,
                "where": where,
                "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
                "count": len(findings),
                "first": findings[:4],
            }
        )
        with open(mode.removeprefix("report:"), "a") as log:
            log.write(line + "\n")
        return
    if mode == "fail":
        shown = "\n  ".join(findings[:8])
        raise IRInvariantError(
            f"{check} at {where}: {len(findings)} violations (tests/ir_invariants.py)"
            f":\n  {shown}"
        )


def _checked_build(self: MLIRModuleBuilder) -> ir.Module:
    module = _BUILD(self)
    definition = getattr(self.hf, "definition", None)
    function = getattr(definition, "fn", None)
    with module.context:
        if (found := f64_ops(module)) and not _declares_f64(module, function):
            name = getattr(definition, "name", "?")
            _violation("f64-not-asked-for", f"frontend of {name}", found)
    return module


def _check_after_workaround_stages(module: ir.Module) -> None:
    """A lowering that stops at ``xevm`` skips the ``xevm-*`` stages: they run on a copy,
    so IR-only tests check what would become the binary."""
    final = ir.Module.parse(str(module), module.context)
    for _, stage in _WORKAROUND_STAGES:
        final = stage.apply(final)
    if found := vector_compares_and_abs(final):
        _violation("vector-compare-or-abs", "after the xevm-* stages", found)


def _mark_stage(text: str) -> None:
    """Under ``MARK_STAGES_VARIABLE``, on stdout unbuffered, also from a forked lowering:
    what an isolated test's output shows if it hangs."""
    if os.environ.get(MARK_STAGES_VARIABLE):
        os.write(1, f"{text}\n".encode())


def _checked_apply_stage(
    name: str, stage: object, module: ir.Module, dump: bool
) -> ir.Module:
    _mark_stage(f"stage {name} starts")
    module = _APPLY_STAGE(name, stage, module, dump)
    _mark_stage(f"stage {name} ended")
    if name in _STAGE_CHECKS:
        check, findings = _STAGE_CHECKS[name]
        if found := findings(module):
            _violation(check, f"stage {name}", found)
    if name == "xevm" and _WORKAROUND_STAGES:
        _check_after_workaround_stages(module)
    return module


def _checked_stages(
    entry: str, kernels: object, stop_at: str, columns: object = ()
) -> list[tuple[str, object]]:
    """The stages of a lowering; remembers those a stop at ``xevm`` leaves out."""
    stages = _STAGES(entry, kernels, stop_at, columns)
    _WORKAROUND_STAGES.clear()
    if stop_at == "xevm":
        names = _STAGES(entry, kernels, "xevm-final", columns)
        index = next(i for i, (name, _) in enumerate(names) if name == "xevm")
        _WORKAROUND_STAGES.extend(names[index + 1 :])
    return stages


_BUILD = MLIRModuleBuilder.build
_APPLY_STAGE = xpu_target._apply_stage
_STAGES = xpu_target._stages
_WORKAROUND_STAGES: list[tuple[str, object]] = []


def valid_mode(mode: str) -> bool:
    return mode in ("fail", "off") or (
        mode.startswith("report:") and bool(mode.removeprefix("report:"))
    )


def install() -> None:
    """Hooks on, for this process and its forks (nothing to undo: tests only)."""
    if os.environ.get(MODE_VARIABLE, "off") == "off":
        return
    for patcher in (
        patch.object(MLIRModuleBuilder, "build", _checked_build),
        patch.object(xpu_target, "_apply_stage", _checked_apply_stage),
        patch.object(xpu_target, "_stages", _checked_stages),
    ):
        patcher.start()
