"""Lower tensor creation operations to Linalg-on-Tensors IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.creation_ops as creation_ops
import torch

from ..support import UnsupportedOperationError
from ..support import torch_dtype_to_mlir
from . import emit
from .registry import lowers

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


@lowers(creation_ops.full)
def lower_full(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``full(shape, value, dtype)`` (also ``hl.zeros``) to ``linalg.fill``."""
    shape_nodes, fill_value = node.args[:2]
    if not isinstance(fill_value, (int, float, bool)):
        raise UnsupportedOperationError(
            "full", reason=f"fill value must be a literal, got {fill_value!r}"
        )
    dtype = node.args[2] if len(node.args) > 2 else torch.float32
    shape = ctx.shape_from_nodes(shape_nodes, "full")
    return emit.filled(shape, torch_dtype_to_mlir(dtype), fill_value)


@lowers(tracing_ops._constant_tensor)
def lower_constant_tensor(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``torch.tensor(value)`` in device code: a filled 0-d tensor."""
    value, dtype = node.args
    return emit.filled([], torch_dtype_to_mlir(dtype), value)
