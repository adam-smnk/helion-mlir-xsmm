"""Example values standing in for the MLIR operands of an ATen call.

The node's arguments are bound to literals and :class:`Operand` markers; each
operand gets a sample of its MLIR type (a meta tensor or a Python scalar), and
running the op on the samples gives its result types. A ``?`` dim, or a runtime
scalar where the op takes a size, is a fresh size symbol of the module's own
fake tensor mode instead, so the sizes it determines stay dynamic.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NamedTuple

from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ..support import UnsupportedOperationError
from ..support import mlir_dtype_to_torch
from ..support import static_dim
from ..support import torch_dtype_to_mlir

if TYPE_CHECKING:
    from torch._ops import OpOverload

    from ..build_context import BuildContext


@dataclass(frozen=True)
class Operand:
    """The ``index``-th operand of a helper call, in place of an argument."""

    index: int

    def __repr__(self) -> str:
        return f"%{self.index}"


class Bound(NamedTuple):
    args: tuple
    kwargs: dict
    values: list[ir.Value]
    """The operand values, by :class:`Operand` index."""
    samples: list[object]
    """A sample per operand value."""


class Sampler:
    """Samples of one module; symbolic ones share a fake tensor mode and shape env."""

    def __init__(self) -> None:
        self._mode: FakeTensorMode | None = None

    @property
    def mode(self) -> FakeTensorMode:
        if self._mode is None:
            self._mode = FakeTensorMode(shape_env=ShapeEnv())
        return self._mode

    def scope(self, samples: list[object]) -> contextlib.AbstractContextManager:
        """The mode to run an op on ``samples`` in: fake if any is symbolic."""
        return self.mode if _symbolic(samples) else contextlib.nullcontext()

    def size_symbol(self) -> torch.SymInt:
        size = self.mode.shape_env.create_unbacked_symint()
        torch._check(size >= 0)
        return size

    def sample(self, value_type: ir.Type, *, fake: bool = False) -> object:
        """A meta tensor or Python scalar standing in for an operand of ``value_type``;
        a fake tensor with a size symbol per ``?`` dim if it has any (or ``fake``)."""
        if isinstance(value_type, ir.RankedTensorType):
            element = value_type.element_type
            dtype = (
                torch.int64
                if isinstance(element, ir.IndexType)
                else mlir_dtype_to_torch(str(element))
            )
            if dtype is None:
                raise UnsupportedOperationError(
                    "ATen helper operand", reason=f"unsupported element type {element}"
                )
            shape = list(value_type.shape)
            dynamic = [ir.ShapedType.is_dynamic_size(dim) for dim in shape]
            if not fake and not any(dynamic):
                return torch.empty(shape, dtype=dtype, device="meta")
            sizes = [
                self.size_symbol() if is_dynamic else dim
                for dim, is_dynamic in zip(shape, dynamic, strict=True)
            ]
            with self.mode:
                return torch.empty(sizes, dtype=dtype)
        if isinstance(value_type, ir.IntegerType) and value_type.width == 1:
            return True
        if isinstance(value_type, (ir.IndexType, ir.IntegerType)):
            return 1
        if isinstance(value_type, ir.FloatType):
            return 1.0
        raise UnsupportedOperationError(
            "ATen helper operand", reason=f"unsupported scalar type {value_type}"
        )


def bind(
    ctx: BuildContext,
    node: torch.fx.Node,
    args: tuple,
    kwargs: dict,
    target: OpOverload,
) -> Bound:
    """Replace inputs by literals or :class:`Operand` markers; sample the operands.

    ``node`` names the call in errors; ``target``'s schema decides how runtime
    scalars are sampled (see :func:`_apply_schema`).
    """
    sampler = ctx.aten_helpers.sampler
    values: list[ir.Value] = []
    samples: list[object] = []

    def convert(arg: object) -> object:
        if isinstance(arg, (list, tuple)):
            return type(arg)(convert(item) for item in arg)
        if isinstance(arg, torch.fx.Node):
            value = ctx.get_value(arg)
            if value is None:
                raise UnsupportedOperationError(
                    str(node.target), reason=f"input {arg.name} has no lowered value"
                )
        elif isinstance(arg, ir.Value):
            value = arg
        else:
            return arg
        if not isinstance(value.type, ir.RankedTensorType):
            constant = _constant(value)
            if constant is not None:
                return constant
        values.append(value)
        samples.append(sampler.sample(value.type))
        return Operand(len(values) - 1)

    bound = Bound(
        convert(args),
        {key: convert(value) for key, value in kwargs.items()},
        values,
        samples,
    )
    _apply_schema(target, bound, sampler)
    if _symbolic(samples):
        # One mode for every tensor operand of the op.
        for index, value in enumerate(values):
            if isinstance(value.type, ir.RankedTensorType):
                samples[index] = sampler.sample(value.type, fake=True)
    return bound


def _apply_schema(target: OpOverload, bound: Bound, sampler: Sampler) -> None:
    """A runtime scalar where ``target``'s schema takes a tensor becomes a 0-d tensor;
    one where it takes a ``SymInt`` (a size) is sampled as a fresh size symbol."""
    schema = target._schema
    by_name = {argument.name: argument for argument in schema.arguments}
    pairs = [
        *zip(schema.arguments, bound.args, strict=False),
        *((by_name[key], arg) for key, arg in bound.kwargs.items() if key in by_name),
    ]
    values, samples = bound.values, bound.samples
    for argument, arg in pairs:
        scalars = [
            operand
            for operand in _operands(arg)
            if not isinstance(values[operand.index].type, ir.RankedTensorType)
        ]
        if isinstance(argument.type, torch.TensorType) and isinstance(arg, Operand):
            for operand in scalars:
                scalar = as_operand(values[operand.index])
                values[operand.index] = tensor_d.FromElementsOp(
                    ir.RankedTensorType.get([], scalar.type), [scalar]
                ).result
                samples[operand.index] = sampler.sample(values[operand.index].type)
        elif _takes_sym_int(argument.real_type):
            for operand in scalars:
                samples[operand.index] = sampler.size_symbol()


def evaluate(target: OpOverload, bound: Bound, sampler: Sampler) -> object:
    """``target``'s result (a tensor or a tuple of them) on the samples."""
    args = substitute(bound.args, bound.samples)
    kwargs = substitute(bound.kwargs, bound.samples)
    try:
        with torch.no_grad(), sampler.scope(bound.samples):
            return target(*args, **kwargs)
    except Exception as error:
        raise UnsupportedOperationError(
            str(target),
            reason=f"cannot infer its result for these operand types: {error}",
        ) from error


def as_tuple(result: object) -> tuple:
    return tuple(result) if isinstance(result, (list, tuple)) else (result,)


def substitute(structure: object, replacements: list[object]) -> object:
    """``structure`` with each :class:`Operand` replaced by ``replacements[index]``."""
    if isinstance(structure, Operand):
        return replacements[structure.index]
    if isinstance(structure, dict):
        return {
            key: substitute(value, replacements) for key, value in structure.items()
        }
    if isinstance(structure, (list, tuple)):
        return type(structure)(substitute(item, replacements) for item in structure)
    return structure


def as_operand(value: ir.Value) -> ir.Value:
    """Index tensors as ``i64``; runtime scalars as ``i1``, ``i64`` or ``f64``."""
    value_type = value.type
    i64 = ir.IntegerType.get_signless(64)
    if isinstance(value_type, ir.RankedTensorType):
        if isinstance(value_type.element_type, ir.IndexType):
            return arith_d.IndexCastOp(
                ir.RankedTensorType.get(list(value_type.shape), i64), value
            ).result
        return value
    if isinstance(value_type, ir.IndexType):
        return arith_d.IndexCastOp(i64, value).result
    if isinstance(value_type, ir.IntegerType):
        if 1 < value_type.width < 64:
            return arith_d.ExtSIOp(i64, value).result
        return value
    f64 = ir.F64Type.get()
    return value if value_type == f64 else arith_d.ExtFOp(f64, value).result


def tensor_type(result: object) -> ir.RankedTensorType:
    """The MLIR type of a sampled result (``?`` for a symbolic dim)."""
    if not isinstance(result, torch.Tensor):
        raise UnsupportedOperationError(
            "ATen helper result", reason=f"non-tensor result {type(result).__name__}"
        )
    return ir.RankedTensorType.get(
        [static_dim(dim) for dim in result.shape], torch_dtype_to_mlir(result.dtype)
    )


def _symbolic(samples: list[object]) -> bool:
    return any(isinstance(sample, (FakeTensor, torch.SymInt)) for sample in samples)


def _takes_sym_int(schema_type: torch.Type) -> bool:
    """A ``SymInt``, ``SymInt?`` or ``SymInt[]`` schema argument."""
    if isinstance(schema_type, torch.OptionalType):
        schema_type = schema_type.getElementType()
    if isinstance(schema_type, torch.ListType):
        schema_type = schema_type.getElementType()
    return isinstance(schema_type, torch.SymIntType)


def _operands(structure: object) -> list[Operand]:
    if isinstance(structure, Operand):
        return [structure]
    if isinstance(structure, (list, tuple)):
        return [operand for item in structure for operand in _operands(item)]
    return []


def _constant(value: ir.Value) -> bool | int | float | None:
    owner = value.owner
    if not isinstance(owner, ir.OpView) or owner.name != "arith.constant":
        return None
    attribute = owner.attributes["value"]
    if isinstance(value.type, ir.FloatType):
        return ir.FloatAttr(attribute).value
    integer = ir.IntegerAttr(attribute).value
    if isinstance(value.type, ir.IntegerType) and value.type.width == 1:
        return bool(integer)
    return integer
