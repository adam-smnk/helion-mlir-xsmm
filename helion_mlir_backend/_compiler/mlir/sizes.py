"""Sizes of the function being built, as static ints or ``index`` values.

A size is a sympy expression over Helion's size symbols. Block-size symbols
take the config's block sizes (the span, for a block covering a whole runtime
dim). A remaining size symbol is ``tensor.dim`` of the first host tensor
argument with that size, else the runtime scalar that carries it; compound
expressions are ``index`` arithmetic. Values are emitted at the start of the
function, so they dominate every use, and shared by equal expressions, so equal
sizes are the same SSA value.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import sympy
import torch
from torch.utils._sympy import functions as sympy_functions

from .support import DynamicShapeError

if TYPE_CHECKING:
    from .build_context import BuildContext


class Sizes:
    """Size values of a :class:`BuildContext`'s current function."""

    def __init__(self, ctx: BuildContext) -> None:
        self._ctx = ctx
        self._values: dict[sympy.Expr, ir.Value] = {}
        self._block: ir.Block | None = None
        self._setup = 0
        self._anchor: ir.Operation | None = None
        self._block_symbols: dict[sympy.Expr, sympy.Expr] | None = None
        self._refs: dict[str, list[sympy.Expr]] = {}

    def begin_function(self, block: ir.Block) -> None:
        """Size values of the function with entry ``block`` go after its current ops."""
        self._values.clear()
        self._block = block
        self._setup = len(block.operations)
        self._anchor = None

    def expr(self, value: object) -> sympy.Expr:
        """``value`` (int, ``SymInt`` or expression) over size symbols only: the
        shape env's replacements applied and block sizes substituted."""
        if isinstance(value, torch.SymInt):
            value = value.node.expr
        expr = self._ctx.env.shape_env.replace(sympy.sympify(value))
        return expr.xreplace(self._block_size_symbols())

    def value(self, value: object) -> int | ir.Value:
        """A size as a static int or an ``index`` value, never its example value."""
        expr = self.expr(value)
        if not expr.free_symbols:
            return int(expr)
        return self._value(expr)

    def ref(self, name: str) -> list[sympy.Expr]:
        """The size expressions of host tensor ``name``'s dims."""
        if name not in self._refs:
            fake = self._ctx.signature.refs[name].fake
            self._refs[name] = [self.expr(size) for size in fake.shape]
        return self._refs[name]

    def extent(
        self, value: ir.Value, dim: int, name: str | None = None
    ) -> int | ir.Value:
        """Size of ``value``'s dimension ``dim``. For the full extent of host tensor
        ``name`` it is :meth:`value` of that size, so equal sizes are the same value."""
        size = ir.RankedTensorType(value.type).shape[dim]
        if not ir.ShapedType.is_dynamic_size(size):
            return size
        if name is not None and name in self._ctx.signature.refs:
            return self.value(self.ref(name)[dim])
        return tensor_d.DimOp(value, self._ctx.index_const(dim)).result

    def tile(self, block_id: int) -> int | ir.Value:
        """The static tile extent of ``block_id``, or the runtime span of a whole tile."""
        block = self._ctx.geometry.block(block_id)
        return self.value(block.span_expr) if block.whole else block.tile_extent

    def _block_size_symbols(self) -> dict[sympy.Expr, sympy.Expr]:
        if self._block_symbols is None:
            env, geometry = self._ctx.env, self._ctx.geometry
            self._block_symbols = {
                info.var.node.expr: env.shape_env.replace(block.span_expr)
                if block.whole
                else sympy.Integer(block.block_size)
                for info in env.block_sizes
                if (block := geometry.blocks.get(info.block_id)) is not None
                and isinstance(info.var, torch.SymInt)
            }
        return self._block_symbols

    def _value(self, expr: sympy.Expr) -> ir.Value:
        value = self._values.get(expr)
        if value is None:
            with self._insertion_point():
                value = self._emit(expr)
            self._values[expr] = value
            if self._anchor is None:
                self._setup = len(self._block.operations)
        return value

    def _insertion_point(self) -> ir.InsertionPoint:
        if self._anchor is None:
            operations = list(self._block.operations)
            if len(operations) == self._setup:
                return ir.InsertionPoint(self._block)
            self._anchor = operations[self._setup]
        return ir.InsertionPoint(self._anchor)

    def _emit(self, expr: sympy.Expr) -> ir.Value:
        source = self._source(expr)
        if source is not None:
            return source
        operands = [
            self._ctx.index_const(int(arg)) if arg.is_Integer else self._value(arg)
            for arg in expr.args
        ]
        combine = None
        if isinstance(expr, sympy.Add):
            combine = arith_d.addi
        elif isinstance(expr, sympy.Mul):
            combine = arith_d.muli
        elif isinstance(expr, (sympy.Max, sympy_functions.Max)):
            combine = arith_d.maxsi
        elif isinstance(expr, (sympy.Min, sympy_functions.Min)):
            combine = arith_d.minsi
        elif isinstance(expr, sympy_functions.FloorDiv):
            combine = arith_d.floordivsi
        elif isinstance(expr, (sympy_functions.Mod, sympy_functions.PythonMod)):
            combine = arith_d.remsi
        if combine is None or not operands:
            raise DynamicShapeError(expr, symbol_name=self._describe(expr))
        return reduce(combine, operands)

    def _source(self, expr: sympy.Expr) -> ir.Value | None:
        ctx = self._ctx
        for name in ctx.signature.refs:
            value = ctx.param_to_value.get(name)
            if value is None:
                continue
            for dim, size in enumerate(self.ref(name)):
                if size == expr:
                    return tensor_d.DimOp(value, ctx.index_const(dim)).result
        for key, scalar in ctx.signature.scalars.items():
            if (
                key in ctx.scalars
                and scalar.expr is not None
                and self.expr(scalar.expr) == expr
            ):
                return ctx.cast_to_index(ctx.scalars[key])
        return None

    def _describe(self, expr: sympy.Expr) -> str:
        origins = {
            str(symbol): origin.origin
            for symbol in expr.free_symbols
            if (origin := self._ctx.host_function.expr_to_origin.get(symbol))
            is not None
        }
        return (
            f"size {expr} (origins {origins}), which no host tensor argument or "
            "runtime scalar of this function provides"
        )
