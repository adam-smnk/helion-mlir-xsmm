"""Lowering of ``permute`` (Helion traces every transpose as one)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir.dialects import linalg as linalg_d
import mlir.ir as ir
import torch

from ..analysis.contractions import transpose_permutation
from ..support import UnsupportedOperationError
from . import emit
from .registry import lowers

if TYPE_CHECKING:
    from ..build_context import BuildContext

aten = torch.ops.aten


@lowers(aten.permute.default)
def lower_transpose(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """A permute to ``linalg.transpose`` (none if folded into a contraction)."""
    if node in ctx.contractions.absorbed:
        return None
    source = ctx.get_value(node.args[0])
    permutation = transpose_permutation(node)
    if source is None or permutation is None:
        raise UnsupportedOperationError(
            str(node.target), reason="the permutation or its input is unknown"
        )
    source_type = ir.RankedTensorType(source.type)
    shape = list(source_type.shape)
    if len(permutation) != len(shape):
        raise UnsupportedOperationError(
            str(node.target),
            reason=f"permutation {permutation} does not match rank {len(shape)}",
        )
    sizes = emit.sizes(source)
    init = emit.empty([sizes[dim] for dim in permutation], source_type.element_type)
    return linalg_d.transpose(source, outs=[init], permutation=permutation).results[0]
