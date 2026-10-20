"""Generic ATen nodes as calls to torch-mlir helper functions, typed at the call site.

At each call site the operands' MLIR types (tensors, and runtime scalars as
``i64``/``f64``/``i1``) and the node's literal arguments form a
:class:`~.helper_cache.HelperRequest`; running the op on samples of those types
(:mod:`.samples`) gives the result types, so nothing is guessed from Helion's
symbolic metadata. The call names a helper derived from the request. After every
function is built, :meth:`AtenHelperTable.materialize` lowers the requests not yet
in the process-wide cache (:mod:`.helper_cache`) and clones the helpers into the
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir.dialects import func as func_d
import mlir.ir as ir
import torch
from torch._ops import OpOverload
import torch.fx

from ..support import UnsupportedOperationError
from . import helper_cache
from .helper_cache import HelperRequest
from .helper_cache import located
from .original_args import original_args
from .samples import Sampler
from .samples import as_operand
from .samples import as_tuple
from .samples import bind
from .samples import evaluate
from .samples import tensor_type

if TYPE_CHECKING:
    from ..build_context import BuildContext


def is_aten_op(node: torch.fx.Node) -> bool:
    """An ATen ``OpOverload`` call with tensor result(s)."""
    if node.op != "call_function" or not isinstance(node.target, OpOverload):
        return False
    value = node.meta.get("val")
    if isinstance(value, (list, tuple)):
        return bool(value) and all(isinstance(item, torch.Tensor) for item in value)
    return isinstance(value, torch.Tensor)


class AtenHelperTable:
    """The helper requests of one module, and the sampler of their operands."""

    def __init__(self) -> None:
        self._requests: dict[str, tuple[HelperRequest, torch.fx.Node]] = {}
        self.sampler = Sampler()

    def add(self, request: HelperRequest, node: torch.fx.Node) -> None:
        self._requests.setdefault(request.name, (request, node))

    def materialize(self, module: ir.Module) -> None:
        """Define every requested helper in ``module`` (one torch-mlir run at most)."""
        cache = helper_cache.CACHE
        missing = [
            (request, node)
            for request, node in self._requests.values()
            if request.name not in cache
        ]
        if missing:
            cache.lower(missing)
        with ir.InsertionPoint.at_block_begin(module.body):
            for request, node in self._requests.values():
                helper = cache[request.name]
                actual = ir.TypeAttr(helper.attributes["function_type"]).value
                if str(actual) != str(request.function_type):
                    with located(node):
                        raise UnsupportedOperationError(
                            str(request.target),
                            reason=(
                                f"torch-mlir lowered it to {actual}, but the call "
                                f"site needs {request.function_type}"
                            ),
                        )
                helper.clone().attributes["sym_visibility"] = ir.StringAttr.get(
                    "private"
                )


def lower_via_aten_helper(ctx: BuildContext, node: torch.fx.Node) -> object:
    """``func.call`` to the helper for this node's operand types."""
    return call_helper(ctx, node, node.target, *original_args(node))


def call_helper(
    ctx: BuildContext,
    node: torch.fx.Node,
    target: OpOverload,
    args: tuple,
    kwargs: dict,
) -> object:
    """``func.call`` to the helper for ``target(*args, **kwargs)``.

    Arguments may be FX nodes, MLIR values or literals; ``node`` locates errors.
    """
    bound = bind(ctx, node, args, kwargs, target)
    result = evaluate(target, bound, ctx.aten_helpers.sampler)
    result_types = [tensor_type(item) for item in as_tuple(result)]
    operands = [as_operand(value) for value in bound.values]
    request = HelperRequest(
        target,
        bound.args,
        tuple(sorted(bound.kwargs.items())),
        tuple(str(operand.type) for operand in operands),
        tuple(str(result_type) for result_type in result_types),
        tuple(bound.samples),
        result,
    )
    ctx.aten_helpers.add(request, node)
    call = func_d.CallOp(result_types, request.name, operands)
    return call.results[0] if len(call.results) == 1 else call


def infer_results(ctx: BuildContext, node: torch.fx.Node) -> tuple[torch.Tensor, ...]:
    """The node's results as meta or fake tensors, from its operands' MLIR types."""
    bound = bind(ctx, node, *original_args(node), node.target)
    return as_tuple(evaluate(node.target, bound, ctx.aten_helpers.sampler))
