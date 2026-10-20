"""``partition_linalg``: linalg ops grouped by how ``vectorize_linalg`` vectorizes
them. Lighthouse vectorizes without vector sizes, which fails for ops of
runtime shape: tiling bounds each runtime extent (an ``affine.min`` with a
constant), so such ops are vectorized with masks, the bounds as vector sizes."""

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import has_static_shape
from .utils import linalg_loop_bounds
from .utils import linalg_tile_sizes

# Loop counts of the ops vectorized with masks.
MAX_LOOPS = 8


@transform_op(modifies_payload=False)
class PartitionLinalgOp(HelionTransformDialect.Operation, name="partition_linalg"):
    """Group the target's linalg ops with loops: those needing tiling first
    (``linalg_tile_sizes``); the rest of static shape; and those of runtime shape
    with evident loop bounds, by loop count (1 to ``MAX_LOOPS``). Upstream
    vectorizes a loop-free op reading with ``tensor.extract`` into invalid IR."""

    target: ext.Operand[transform.AnyOpType]
    groups: Sequence[ext.Result[transform.AnyOpType]]

    @staticmethod
    def run(
        op: "PartitionLinalgOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        groups: list[list[ir.Operation]] = [[] for _ in op.groups]

        def collect(visited: ir.Operation) -> ir.WalkResult:
            bounds = linalg_loop_bounds(visited.opview)
            if not bounds:
                return ir.WalkResult.ADVANCE
            if any(linalg_tile_sizes(visited.opview)):
                groups[-1].append(visited)
            elif has_static_shape(visited.opview):
                groups[0].append(visited)
            elif 0 < len(bounds) < len(groups) - 1:
                groups[len(bounds)].append(visited)
            return ir.WalkResult.ADVANCE

        for target in state.get_payload_ops(op.target):
            target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
        for handle, ops in zip(op.groups, groups, strict=True):
            results.set_ops(handle, ops)
        return DiagnosedSilenceableFailure.Success


def partition_linalg(target: ir.Value) -> list[ir.Value]:
    """snake_case wrapper to create a PartitionLinalgOp; its groups: ops of static
    shape, then of runtime shape by loop count (1 to ``MAX_LOOPS``), then ops
    needing tiling."""
    groups = [transform.AnyOpType.get()] * (MAX_LOOPS + 2)
    return list(PartitionLinalgOp(target=target, groups=groups).groups)
