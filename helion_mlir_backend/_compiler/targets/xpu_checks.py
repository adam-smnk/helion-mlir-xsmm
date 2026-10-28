"""What the XPU (XeGPU) pipeline compiles, checked on the inlined tensor IR before
lowering: kernels outside it fail with an :class:`UnsupportedOperationError`
naming the gap, not a transform failure deep in lighthouse.

The gates are conservative, by IR property (not by kernel):

- static shapes, and full static tiles in every grid loop (no masking: block
  sizes divide the sizes they tile);
- every grid loop (top-level ``scf.forall``, one GPU kernel each) is a kernel
  class lighthouse handles: one matmul-like contraction (GEMM, plus epilogue) or
  a reduction; elementwise-only loops are not supported yet;
- a contraction runs in a K loop (an inner ``hl.tile`` with a block size below
  K), on f16/bf16 operands accumulating in f32 (XeGPU DPAS).
"""

from __future__ import annotations

from mlir import ir
from mlir.dialects import linalg

from ..mlir.support.errors import UnsupportedOperationError

_DPAS_OPERANDS = frozenset({"f16", "bf16"})
_DPAS_ACCUMULATOR = "f32"
_CPU = "target='cpu' (CPU tensors)"


def check_xegpu_capabilities(module: ir.Module, entry: str) -> None:
    """Raise :class:`UnsupportedOperationError` if ``entry`` of the inlined
    ``module`` is outside what the XeGPU pipeline compiles."""
    func = _function(module, entry)
    _check_static_arguments(func)
    for loop in _grid_loops(func):
        _check_full_tiles(loop)
        _check_kernel_class(loop)


def _function(module: ir.Module, name: str) -> ir.Operation:
    for op in module.body.operations:
        if (
            op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == name
        ):
            return op.operation
    raise ValueError(f"no function named '{name}' in the module")


def _walk(op: ir.Operation) -> list[ir.Operation]:
    """The ops nested in ``op``, in pre-order (``op`` excluded)."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested != op:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    op.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _grid_loops(func: ir.Operation) -> list[ir.Operation]:
    """The top-level ``scf.forall`` loops of ``func``: one kernel launch each."""
    loops = [op for op in _walk(func) if op.name == "scf.forall"]
    return [loop for loop in loops if _parent_named(loop, "scf.forall") is None]


def _parent_named(op: ir.Operation, name: str) -> ir.Operation | None:
    parent = op.parent
    while parent is not None and parent.name != "func.func":
        if parent.name == name:
            return parent
        parent = parent.parent
    return None


def _check_static_arguments(func: ir.Operation) -> None:
    arg_types = ir.FunctionType(ir.TypeAttr(func.attributes["function_type"]).value)
    if any(
        isinstance(arg_type, ir.ShapedType) and not arg_type.has_static_shape
        for arg_type in arg_types.inputs
    ):
        raise UnsupportedOperationError(
            "dynamic shapes on XPU",
            reason="the XeGPU pipeline compiles static shapes only",
            alternatives=["@helion.kernel(static_shapes=True)", _CPU],
        )


def _check_full_tiles(loop: ir.Operation) -> None:
    for op in _walk(loop):
        values = [*op.operands, *op.results]
        if op.name == "tensor.pad" or any(
            isinstance(value.type, ir.ShapedType)
            and not ir.ShapedType(value.type).has_static_shape
            for value in values
        ):
            raise UnsupportedOperationError(
                "partial tiles on XPU",
                reason=(
                    f"'{op.name}' works on a partial or runtime-sized tile; the "
                    "XeGPU pipeline has no masked loads and stores, so every tile "
                    "must be full and static"
                ),
                alternatives=["block sizes that divide the sizes they tile", _CPU],
            )


def _is_reduction(op: ir.OpView) -> bool:
    """A linalg op with a loop dim its results do not index."""
    maps = [ir.AffineMapAttr(m).value for m in linalg.get_indexing_maps(op)]
    inits = maps[len(maps) - len(op.results) :] if op.results else maps[-1:]
    indexed = {
        ir.AffineDimExpr(expr).position
        for affine_map in inits
        for expr in affine_map.results
        if isinstance(expr, ir.AffineDimExpr)
    }
    return bool(maps) and len(indexed) < maps[0].n_dims


def _element_type(value: ir.Value) -> str:
    value_type = value.type
    if isinstance(value_type, ir.ShapedType):
        value_type = ir.ShapedType(value_type).element_type
    return str(value_type)


def _check_kernel_class(loop: ir.Operation) -> None:
    contractions: list[ir.OpView] = []
    reductions = 0
    for op in _walk(loop):
        view = op.opview
        if linalg.get_indexing_maps(view) is None:
            continue
        if linalg.isa_contraction_op(view):
            contractions.append(view)
        elif _is_reduction(view):
            reductions += 1
    if not contractions and not reductions:
        raise UnsupportedOperationError(
            "elementwise-only grid loop on XPU",
            reason=(
                "the XeGPU pipeline compiles grid loops with a matmul or a "
                "reduction; elementwise-only kernels are not supported on XPU yet"
            ),
            alternatives=[_CPU],
        )
    if len(contractions) > 1:
        raise UnsupportedOperationError(
            "several matmuls in one grid loop on XPU",
            reason=(
                f"the grid loop has {len(contractions)} contractions; the XeGPU "
                "layout annotation handles one matmul per kernel"
            ),
            alternatives=["one matmul per top-level hl.tile loop", _CPU],
        )
    for contraction in contractions:
        _check_contraction(contraction, loop)


def _check_contraction(op: ir.OpView, loop: ir.Operation) -> None:
    operation = op.operation
    k_loop = _parent_named(operation, "scf.for")
    if k_loop is None or _parent_named(k_loop, "scf.forall") != loop:
        raise UnsupportedOperationError(
            f"'{operation.name}' without a K loop on XPU",
            reason=(
                "the XeGPU GEMM schedule pipelines loads over a K loop; tile the "
                "reduction dim with an inner hl.tile whose block size is below K"
            ),
            alternatives=["a K block size below K", _CPU],
        )
    operands = list(operation.operands)
    inputs = [_element_type(value) for value in operands[: -len(operation.results)]]
    accumulator = _element_type(operands[-1])
    if not set(inputs) <= _DPAS_OPERANDS or accumulator != _DPAS_ACCUMULATOR:
        raise UnsupportedOperationError(
            f"matmul of {' x '.join(inputs)} into {accumulator} on XPU",
            reason="XeGPU DPAS takes f16 or bf16 operands and accumulates in f32",
            alternatives=[
                "torch.float16 or torch.bfloat16 operands, a torch.float32 accumulator",
                _CPU,
            ],
        )
