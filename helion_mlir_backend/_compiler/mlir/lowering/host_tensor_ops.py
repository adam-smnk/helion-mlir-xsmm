"""Host tensors: function arguments, and reshapes of the parameters they alias."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import sympy

from ..support import UnsupportedOperationError
from .registry import lowers

if TYPE_CHECKING:
    import torch

    from ..build_context import BuildContext


@lowers(tracing_ops._host_tensor)
def lower_host_tensor(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_host_tensor(name)``: the function argument of ``name``, or the reshape of
    the parameter a host view aliases (``KernelSignature.aliases``)."""
    name = node.args[0]
    if name in ctx.param_to_value:
        return ctx.param_to_value[name]
    return _reshaped_alias(ctx, ctx.signature.aliases[name], node)


def _reshaped_alias(ctx: BuildContext, base: str, alias: torch.fx.Node) -> ir.Value:
    """Parameter ``base`` in the shape of its host reshape ``alias`` (collapsed to
    1-D, then expanded). Runtime sizes come from ``ctx.sizes``."""
    base_value = ctx.param_to_value[base]
    base_sizes = ctx.sizes.ref(base)
    alias_sizes = [ctx.sizes.expr(size) for size in alias.meta["val"].shape]
    if alias_sizes == base_sizes:
        return base_value
    base_numel = sympy.prod(base_sizes)
    if sympy.simplify(sympy.prod(alias_sizes) - base_numel) != 0:
        raise UnsupportedOperationError(
            f"host view '{alias.args[0]}'",
            reason=f"cannot show that {alias_sizes} has {base_numel} elements",
        )
    dynamic = ir.ShapedType.get_dynamic_size()
    element_type = ir.RankedTensorType(base_value.type).element_type
    base_rank = len(base_sizes)
    alias_shape = [dynamic if size.free_symbols else int(size) for size in alias_sizes]
    result_type = ir.RankedTensorType.get(alias_shape, element_type)
    if len(alias_shape) == 1:
        return tensor_d.CollapseShapeOp(
            result_type, base_value, [list(range(base_rank))]
        ).result
    flat_value = base_value
    if base_rank != 1:
        flat_dim = dynamic if base_numel.free_symbols else int(base_numel)
        flat_value = tensor_d.CollapseShapeOp(
            ir.RankedTensorType.get([flat_dim], element_type),
            base_value,
            [list(range(base_rank))],
        ).result
    output = [ctx.sizes.value(size) for size in alias_sizes]
    return tensor_d.ExpandShapeOp(
        result_type,
        flat_value,
        [list(range(len(alias_shape)))],
        [size for size in output if isinstance(size, ir.Value)],
        alias_shape,
    ).result
