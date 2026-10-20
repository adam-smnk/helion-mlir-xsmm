"""Scalar (``SymInt``/``SymBool``) arithmetic and comparisons.

These act on tile positions, ``hl.grid`` indices and runtime integer arguments,
which are ``index``/``iN`` values; Python's floor division and modulo semantics
are kept. Booleans are ``i1``.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import arith as arith_d
import mlir.ir as ir

from ..support import UnsupportedOperationError
from .registry import lowers

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from ..build_context import BuildContext

_PREDICATES = {
    operator.eq: arith_d.CmpIPredicate.eq,
    operator.ne: arith_d.CmpIPredicate.ne,
    operator.lt: arith_d.CmpIPredicate.slt,
    operator.le: arith_d.CmpIPredicate.sle,
    operator.gt: arith_d.CmpIPredicate.sgt,
    operator.ge: arith_d.CmpIPredicate.sge,
}


def _floordiv(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    return arith_d.FloorDivSIOp(lhs, rhs).result


def _mod(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    """Python's ``%``: the remainder has the sign of the divisor."""
    product = arith_d.MulIOp(_floordiv(lhs, rhs), rhs).result
    return arith_d.SubIOp(lhs, product).result


_ARITHMETIC: dict[Callable, Callable[[ir.Value, ir.Value], ir.Value]] = {
    operator.add: lambda lhs, rhs: arith_d.AddIOp(lhs, rhs).result,
    operator.sub: lambda lhs, rhs: arith_d.SubIOp(lhs, rhs).result,
    operator.mul: lambda lhs, rhs: arith_d.MulIOp(lhs, rhs).result,
    operator.floordiv: _floordiv,
    operator.mod: _mod,
}


def _index(ctx: BuildContext, operand: object) -> ir.Value:
    value = operand if isinstance(operand, int) else ctx.get_value(operand)
    if value is None or not (
        isinstance(value, int) or isinstance(value.type, (ir.IndexType, ir.IntegerType))
    ):
        raise UnsupportedOperationError(
            "scalar arithmetic", reason=f"operand {operand!r} is not an integer"
        )
    return ctx.as_index(value)


@lowers(*_PREDICATES)
def lower_comparison(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    lhs, rhs = (_index(ctx, operand) for operand in node.args)
    return arith_d.CmpIOp(_PREDICATES[node.target], lhs, rhs).result


@lowers(*_ARITHMETIC)
def lower_arithmetic(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    lhs, rhs = (_index(ctx, operand) for operand in node.args)
    return _ARITHMETIC[node.target](lhs, rhs)


@lowers(tracing_ops._and, tracing_ops._or)
def lower_logical(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    lhs, rhs = (condition(ctx, operand) for operand in node.args)
    op = arith_d.AndIOp if node.target is tracing_ops._and else arith_d.OrIOp
    return op(lhs, rhs).result


@lowers(tracing_ops._not)
def lower_not(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    i1 = ir.IntegerType.get_signless(1)
    true = arith_d.ConstantOp(i1, ir.IntegerAttr.get(i1, 1)).result
    return arith_d.XOrIOp(condition(ctx, node.args[0]), true).result


def condition(ctx: BuildContext, operand: object) -> ir.Value:
    """``operand`` as an ``i1``: nonzero is true; a one-element tensor is read."""
    from mlir.dialects import tensor as tensor_d

    from . import emit

    value = operand if isinstance(operand, ir.Value) else ctx.get_value(operand)
    if value is None:
        raise UnsupportedOperationError(
            "condition", reason=f"{operand!r} has no lowered value"
        )
    if isinstance(value.type, ir.RankedTensorType):
        zero = ctx.index_const(0)
        value = tensor_d.ExtractOp(value, [zero] * value.type.rank).result
    if isinstance(value.type, ir.IndexType):
        value = emit.cast_scalar(value, ir.IntegerType.get_signless(64))
    return emit.cast_scalar(value, ir.IntegerType.get_signless(1))
