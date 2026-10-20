"""Direct elementwise lowerings that torch-mlir helpers cannot express."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._gelu_tanh_approx as gelu_ops
from mlir.dialects import linalg as linalg_d
import mlir.ir as ir
import torch

from ..support import torch_dtype_to_mlir
from . import emit
from .registry import NOT_APPLICABLE
from .registry import lowers

if TYPE_CHECKING:
    from ..build_context import BuildContext

aten = torch.ops.aten

_BINARY_KINDS = {
    aten.add.Tensor: "add",
    aten.sub.Tensor: "sub",
    aten.mul.Tensor: "mul",
    aten.div.Tensor: "div",
}


@lowers(*_BINARY_KINDS)
def lower_scalar_binary(ctx: BuildContext, node: torch.fx.Node) -> object:
    """``tensor <op> scalar`` where the scalar is an MLIR scalar value.

    Covers tile positions (``tile.begin`` and friends) and ``hl.grid`` indices,
    which reach codegen as ``index`` scalars; torch-mlir cannot import an index
    scalar as the second operand of a ``.Tensor`` overload.
    """
    args = list(node.args)
    alpha = args[2] if len(args) > 2 else node.kwargs.get("alpha", 1)
    if len(args) < 2 or alpha != 1:
        return NOT_APPLICABLE
    values = [
        ctx.get_value(arg) if isinstance(arg, torch.fx.Node) else None
        for arg in args[:2]
    ]
    tensor_index = next(
        (i for i in (0, 1) if values[i] is not None and _is_tensor(values[i])), None
    )
    scalar_index = next(
        (i for i in (0, 1) if values[i] is not None and not _is_tensor(values[i])),
        None,
    )
    if tensor_index is None or scalar_index is None:
        return NOT_APPLICABLE

    tensor_value = values[tensor_index]
    tensor_type = ir.RankedTensorType(tensor_value.type)
    element_type = tensor_type.element_type
    if torch_dtype_to_mlir(node.meta["val"].dtype) != element_type:
        return NOT_APPLICABLE  # type promotion: left to the helper
    scalar = emit.cast_scalar(values[scalar_index], element_type)
    if scalar is None:
        return NOT_APPLICABLE

    shape = emit.sizes(tensor_value)
    splat = linalg_d.fill(scalar, outs=[emit.empty(shape, element_type)])
    lhs, rhs = (tensor_value, splat) if tensor_index == 0 else (splat, tensor_value)
    kind = getattr(linalg_d.ElementwiseKind, _BINARY_KINDS[node.target])
    return linalg_d.elementwise(
        lhs, rhs, outs=[emit.empty(shape, element_type)], kind=kind
    )


@lowers(
    aten.alias.default,
    aten.detach.default,
    aten.clone.default,
    aten.lift_fresh_copy.default,
)
def lower_passthrough(ctx: BuildContext, node: torch.fx.Node) -> object:
    """Value-semantics aliases: tensors are immutable SSA values."""
    source = node.args[0] if node.args else None
    value = ctx.get_value(source) if isinstance(source, torch.fx.Node) else None
    return NOT_APPLICABLE if value is None else value


@lowers(gelu_ops._gelu_erf, gelu_ops._gelu_tanh_approx)
def lower_gelu(ctx: BuildContext, node: torch.fx.Node) -> object:
    """Helion's single-node GELU ops, as the ``aten.gelu`` helper they stand for."""
    from ..aten_bridge import call_helper

    approximate = "tanh" if node.target is gelu_ops._gelu_tanh_approx else "none"
    return call_helper(
        ctx, node, aten.gelu.default, (node.args[0],), {"approximate": approximate}
    )


def _is_tensor(value: ir.Value) -> bool:
    return isinstance(value.type, ir.RankedTensorType)
