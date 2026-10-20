"""``inline_mlir``: a ``func.call`` to the user's function, cloned into the module.

Operands and results are converted to the function's declared types (``tensor.cast``
between compatible tensor types, scalar casts, constants for Python numbers); the
call is inlined with the phase functions before bufferization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion import exc
from mlir.dialects import func as func_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from .. import snippets
from ..support import ValueNotFoundError
from . import emit
from .registry import lowers
from helion_mlir_backend.language import _inline_mlir

if TYPE_CHECKING:
    from ..build_context import BuildContext


@lowers(_inline_mlir)
def lower_inline_mlir(
    ctx: BuildContext, node: torch.fx.Node
) -> ir.Value | emit.Results:
    source, args, output_like = node.args
    text = snippets.source_text(source)
    entry = snippets.signature(text).entry
    callee, function_type = snippets.define(ctx.mlir_module, text)
    operands = [
        _operand(ctx, arg, wanted, f"argument {index}", entry)
        for index, (arg, wanted) in enumerate(
            zip(args, function_type.inputs, strict=True)
        )
    ]
    likes = output_like if isinstance(output_like, (tuple, list)) else [output_like]
    call = func_d.CallOp(list(function_type.results), callee, operands)
    results = [
        _convert(result, _value(ctx, like).type, f"result {index}", entry)
        for index, (result, like) in enumerate(zip(call.results, likes, strict=True))
    ]
    return results[0] if len(results) == 1 else emit.Results(results)


def _operand(
    ctx: BuildContext, arg: object, wanted: ir.Type, what: str, entry: str
) -> ir.Value:
    if isinstance(arg, torch.fx.Node):
        return _convert(_value(ctx, arg), wanted, what, entry)
    return emit.constant(wanted, arg)


def _value(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    value = ctx.get_value(node)
    if value is None:
        raise ValueNotFoundError(node, "inline_mlir operand")
    return value


def _convert(value: ir.Value, wanted: ir.Type, what: str, entry: str) -> ir.Value:
    """``value`` as ``wanted``: a cast between compatible tensors, or a scalar cast."""
    if value.type == wanted:
        return value
    if isinstance(wanted, ir.RankedTensorType):
        if _cast_compatible(value.type, wanted):
            return tensor_d.cast(wanted, value)
    elif (converted := emit.cast_scalar(value, wanted)) is not None:
        return converted
    raise exc.InvalidAPIUsage(
        f"inline_mlir {what}: the kernel has {value.type}, @{entry} declares "
        f"{wanted}. Static dims must equal the kernel's (a tile of x[tile] is "
        "min(block size, dim) wide); declare ? for dims that vary with the config"
    )


def _cast_compatible(source: ir.Type, target: ir.RankedTensorType) -> bool:
    if not isinstance(source, ir.RankedTensorType):
        return False
    if source.element_type != target.element_type or source.rank != target.rank:
        return False
    return all(
        a == b or ir.ShapedType.is_dynamic_size(a) or ir.ShapedType.is_dynamic_size(b)
        for a, b in zip(source.shape, target.shape, strict=True)
    )
