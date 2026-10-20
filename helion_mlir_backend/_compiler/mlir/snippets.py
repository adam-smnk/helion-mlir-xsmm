"""User MLIR functions called from device code by ``inline_mlir``.

A snippet is MLIR text (an ``mlir.ir.Module`` is printed with its locations)
holding ``func.func`` ops; the first one is the entry. :func:`validate` parses and
checks it once per text, :func:`check_call` checks a call site against it while
Helion traces the kernel, and :func:`define` clones it into a module under private
names unique to the snippet, for a ``func.call`` inlined with the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
from typing import TYPE_CHECKING

from helion import exc
from helion._compiler.compile_environment import warning
import mlir.ir as ir
import torch

from .support.type_utils import mlir_dtype_to_torch
from helion_mlir_backend.language import InlineMLIRHint

if TYPE_CHECKING:
    from collections.abc import Sequence

_SCALARS = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)
_HINTED: set[str] = set()


@dataclass(frozen=True)
class Param:
    """A parameter or result of the entry: a ranked tensor or a scalar."""

    type: str
    element: str
    # ``None`` for scalars; ``None`` items are dynamic dims.
    dims: tuple[int | None, ...] | None

    @property
    def is_tensor(self) -> bool:
        return self.dims is not None

    @property
    def is_float(self) -> bool:
        return self.element.startswith(("f", "bf"))


@dataclass(frozen=True)
class Signature:
    entry: str
    params: tuple[Param, ...]
    results: tuple[Param, ...]
    hints: tuple[str, ...]


def source_text(source: object) -> str:
    """The text of a snippet given as a string or an ``mlir.ir.Module``."""
    if isinstance(source, ir.Module):
        return source.operation.get_asm(enable_debug_info=True)
    if isinstance(source, str):
        return source
    raise exc.InvalidAPIUsage(
        f"inline_mlir source must be a str or an mlir.ir.Module, not {type(source).__name__}"
    )


def validate(text: str) -> Signature:
    """The snippet's entry signature; reports its hints once per process."""
    result = signature(text)
    if text not in _HINTED:
        _HINTED.add(text)
        for hint in result.hints:
            warning(InlineMLIRHint(hint))
    return result


def check_call(
    signature: Signature, args: Sequence[object], outputs: Sequence[object]
) -> None:
    """Raise unless ``args`` and ``outputs`` (fake values) fit the entry."""
    if len(args) != len(signature.params) or len(outputs) != len(signature.results):
        raise exc.InvalidAPIUsage(
            f"inline_mlir: @{signature.entry} takes {len(signature.params)} arguments "
            f"and returns {len(signature.results)} results; the call passes "
            f"{len(args)} args and {len(outputs)} output_like tensors"
        )
    for index, (param, arg) in enumerate(zip(signature.params, args, strict=True)):
        _check(signature.entry, f"argument {index}", param, arg)
    for index, (param, like) in enumerate(zip(signature.results, outputs, strict=True)):
        if not isinstance(like, torch.Tensor):
            raise exc.InvalidAPIUsage(
                f"inline_mlir: output_like[{index}] must be a tensor, not "
                f"{type(like).__name__}"
            )
        _check(signature.entry, f"result {index}", param, like)


def define(module: ir.Module, text: str) -> tuple[str, ir.FunctionType]:
    """Clone the snippet into ``module`` (once): the entry's symbol and type.

    Defined functions become private and prefixed per snippet; declarations
    keep their names, so they still bind to external symbols.
    """
    table = ir.SymbolTable(module.operation)
    prefix = "__inline_mlir_" + hashlib.sha256(text.encode()).hexdigest()[:16]
    entry = f"{prefix}_{signature(text).entry}"
    if entry not in table:
        snippet = ir.Module.parse(text, context=module.context)
        functions = [op.operation for op in snippet.body.operations]
        for function in functions:
            if _is_declaration(function):
                continue
            old = _symbol(function)
            ir.SymbolTable.replace_all_symbol_uses(
                old, f"{prefix}_{old}", snippet.operation
            )
            ir.SymbolTable.set_symbol_name(function, f"{prefix}_{old}")
            function.attributes["sym_visibility"] = ir.StringAttr.get("private")
        with ir.InsertionPoint.at_block_begin(module.body):
            for function in functions:
                if not _is_declaration(function) or _symbol(function) not in table:
                    function.clone()
    return entry, _function_type(ir.SymbolTable(module.operation)[entry])


@functools.cache
def signature(text: str) -> Signature:
    """Parse and check a snippet: its entry's parameters, results and hints."""
    from .codegen import _get_shared_mlir_context

    with _get_shared_mlir_context():
        try:
            snippet = ir.Module.parse(text)
        except ir.MLIRError as error:
            raise exc.InvalidAPIUsage(
                f"inline_mlir source is not valid MLIR:\n{error}"
            ) from None
        functions = [op.operation for op in snippet.body.operations]
        others = sorted({op.name for op in functions} - {"func.func"})
        if not functions or others:
            found = f"; found {', '.join(others)}" if others else ""
            raise exc.InvalidAPIUsage(
                f"inline_mlir source must hold func.func ops only{found}"
            )
        entry = functions[0]
        name = _symbol(entry)
        if _is_declaration(entry):
            raise exc.InvalidAPIUsage(f"inline_mlir: the entry @{name} has no body")
        function_type = _function_type(entry)
        params = tuple(
            _param(name, f"parameter {index}", value_type)
            for index, value_type in enumerate(function_type.inputs)
        )
        results = tuple(
            _param(name, f"result {index}", value_type)
            for index, value_type in enumerate(function_type.results)
        )
        if not results or not all(result.is_tensor for result in results):
            raise exc.InvalidAPIUsage(
                f"inline_mlir: @{name} must return one or more ranked tensors"
            )
        hints = [hint for function in functions for hint in _layout_hints(function)]
        return Signature(name, params, results, tuple(hints[:1]))


def _param(entry: str, what: str, value_type: ir.Type) -> Param:
    if isinstance(value_type, ir.RankedTensorType):
        dims = tuple(
            None if ir.ShapedType.is_dynamic_size(size) else size
            for size in value_type.shape
        )
        return Param(str(value_type), str(value_type.element_type), dims)
    if isinstance(value_type, (ir.IndexType, ir.IntegerType, ir.FloatType)):
        return Param(str(value_type), str(value_type), None)
    if isinstance(value_type, (ir.MemRefType, ir.UnrankedMemRefType)):
        raise exc.InvalidAPIUsage(
            f"inline_mlir: {what} of @{entry} is {value_type}; the entry takes and "
            "returns tensors (use bufferization.to_buffer inside it for memrefs)"
        )
    raise exc.InvalidAPIUsage(
        f"inline_mlir: {what} of @{entry} is {value_type}; expected a ranked "
        "tensor, index, integer or float"
    )


def _check(entry: str, what: str, param: Param, value: object) -> None:
    if isinstance(value, torch.Tensor):
        if (
            param.is_tensor
            and mlir_dtype_to_torch(param.element) == value.dtype
            and len(param.dims) == value.dim()
        ):
            return
        passed = f"a {value.dim()}-d {value.dtype} tensor"
    elif not isinstance(value, _SCALARS):
        passed = type(value).__name__
    elif param.is_tensor:
        passed = f"the scalar {value!r}"
    elif isinstance(value, (float, torch.SymFloat)) and not param.is_float:
        passed = f"the float {value!r}"
    else:
        return
    raise exc.InvalidAPIUsage(
        f"inline_mlir {what}: @{entry} declares {param.type}, the kernel passes {passed}"
    )


def _layout_hints(function: ir.Operation) -> list[str]:
    hints: list[str] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name != "bufferization.to_buffer":
            return ir.WalkResult.ADVANCE
        source, buffer_type = op.operands[0], op.results[0].type
        if (
            isinstance(source, ir.BlockArgument)
            and source.owner.owner.operation.name == "func.func"
            and not _fully_dynamic_layout(buffer_type)
        ):
            hints.append(
                f"inline_mlir: bufferization.to_buffer of an argument of "
                f"@{_symbol(function)} gives {buffer_type} ({op.location}). Tiles are "
                "strided views whose strides bufferization may not know statically, "
                "so a fixed layout makes it copy the tile on every call (or fail on "
                "a partially static one). Take strided<[?, ?], offset: ?> and "
                "memref.cast it to the layout the code needs. Use "
                "ignore_warnings=[helion_mlir_backend.InlineMLIRHint] to silence."
            )
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    function.walk(visit)
    return hints


def _fully_dynamic_layout(buffer_type: ir.Type) -> bool:
    layout = getattr(buffer_type, "layout", None)
    if not isinstance(layout, ir.StridedLayoutAttr):
        return False
    dynamic = ir.ShapedType.get_dynamic_stride_or_offset()
    return all(value == dynamic for value in [*layout.strides, layout.offset])


def _symbol(function: ir.Operation) -> str:
    return ir.StringAttr(function.attributes["sym_name"]).value


def _is_declaration(function: ir.Operation) -> bool:
    return len(function.regions[0].blocks) == 0


def _function_type(function: ir.Operation | ir.OpView) -> ir.FunctionType:
    return ir.FunctionType(ir.TypeAttr(function.attributes["function_type"]).value)
