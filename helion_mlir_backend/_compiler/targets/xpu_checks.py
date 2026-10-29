"""What the XPU (XeGPU) pipeline compiles, checked on the inlined tensor IR before
lowering: kernels outside it fail with an :class:`UnsupportedOperationError`
naming the gap, not a transform failure deep in lighthouse.

The gates are conservative, by IR property (not by kernel):

- static shapes, and full static tiles in every grid loop (no masking: block
  sizes divide the sizes they tile);
- 1D or 2D tiles: the checks run after ``prepare`` folded unit tile dims away,
  so a rank-3+ tile with block size 1 on its leading (e.g. batch) dims is 2D
  here, and only tiles with more than two non-unit dims are rejected;
- every grid loop (top-level ``scf.forall``, one GPU kernel each) is a kernel
  class the pipeline handles: one matmul-like contraction (GEMM, plus epilogue),
  two in flash-attention form (``xpu_params.attention_shape``: ``Q @ K^T``, row
  reductions, ``P @ V`` accumulated in a loop over K/V tiles; a head dim of
  64, as lighthouse's attention layout requires), a reduction over the
  innermost dim (lighthouse's classes: rows of 2D tiles), or elementwise
  (only parallel ops: tiles loaded and stored whole, that an XeGPU layout
  fits);
- a contraction runs in a K loop (an inner ``hl.tile`` with a block size below
  K), on f16/bf16 operands (also f32 values the contraction truncates to them,
  e.g. attention's ``P``) accumulating in f32 (XeGPU DPAS).
"""

from __future__ import annotations

from mlir import ir
from mlir.dialects import linalg

from ..mlir.support.errors import UnsupportedOperationError
from .xpu_params import ATTENTION_D_HEAD
from .xpu_params import ElementwiseShape
from .xpu_params import attention_shape
from .xpu_params import kernel_shape

_DPAS_OPERANDS = frozenset({"f16", "bf16"})
_DPAS_ACCUMULATOR = "f32"
_CPU = "target='cpu' (CPU tensors)"
_UNIT_LEADING_BLOCKS = (
    "block size 1 on the leading (e.g. batch or head) dims, tiling only the last two"
)


def check_xegpu_capabilities(module: ir.Module, entry: str) -> list[ir.Operation]:
    """Raise :class:`UnsupportedOperationError` if ``entry`` of the inlined and
    prepared (unit tile dims folded) ``module`` is outside what the XeGPU
    pipeline compiles; returns its grid loops (top-level ``scf.forall``), one
    GPU kernel each, in program order."""
    func = _function(module, entry)
    _check_static_arguments(func)
    loops = _grid_loops(func)
    for loop in loops:
        _check_full_tiles(loop)
        _check_tile_ranks(loop)
        _check_kernel_class(loop)
    return loops


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


def _check_tile_ranks(loop: ir.Operation) -> None:
    """Every tile the grid loop slices or stores has at most two dims (after
    unit-dim folding: at most two block sizes above 1)."""
    for op in _walk(loop):
        if op.name == "tensor.extract_slice":
            tile = op.results[0]
        elif op.name == "tensor.parallel_insert_slice":
            tile = op.operands[0]
        else:
            continue
        shape = ir.ShapedType(tile.type).shape
        if len(shape) > 2:
            raise UnsupportedOperationError(
                f"{len(shape)}D tiles on XPU",
                reason=(
                    f"the grid loop works on {'x'.join(map(str, shape))} tiles; the "
                    "XeGPU kernel classes take 1D or 2D tiles (unit tile dims "
                    "are folded away, so only the last two tiled dims may have "
                    "block sizes above 1)"
                ),
                alternatives=[_UNIT_LEADING_BLOCKS, _CPU],
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


def _check_reduction(op: ir.OpView) -> None:
    """The reduction ``op`` reduces its inputs over their innermost dim only
    (lighthouse's reductions: the rows of a 2D tile)."""
    maps = [ir.AffineMapAttr(m).value for m in linalg.get_indexing_maps(op)]
    count = len(op.operation.operands) - len(op.results)
    inits = maps[count:] if op.results else maps[-1:]
    indexed = {
        ir.AffineDimExpr(expr).position
        for affine_map in inits
        for expr in affine_map.results
        if isinstance(expr, ir.AffineDimExpr)
    }
    for value, affine_map in zip(op.operation.operands, maps[:count], strict=False):
        dims = [
            ir.AffineDimExpr(expr).position
            if isinstance(expr, ir.AffineDimExpr)
            else None
            for expr in affine_map.results
        ]
        inner = [
            position
            for position, dim in enumerate(dims[:-1])
            if dim is not None and dim not in indexed
        ]
        if inner:
            shape = "x".join(map(str, ir.ShapedType(value.type).shape))
            raise UnsupportedOperationError(
                "reduction over a non-innermost dim on XPU",
                reason=(
                    f"'{op.operation.name}' reduces dim {inner[0]} of a {shape} "
                    "tile; the XeGPU reduction class reduces the rows of 2D tiles "
                    "over their last (contiguous) dim"
                ),
                alternatives=[
                    "a reduction over the last dim (e.g. of a transposed copy)",
                    _CPU,
                ],
            )


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
            _check_reduction(view)
            reductions += 1
    if not contractions and not reductions:
        _check_elementwise(loop)
        return
    if len(contractions) == 2:
        _check_attention(contractions, loop)
    elif len(contractions) > 2:
        raise UnsupportedOperationError(
            "several matmuls in one grid loop on XPU",
            reason=(
                f"the grid loop has {len(contractions)} contractions; the XeGPU "
                "layout annotation handles one matmul per kernel, or two in "
                "flash-attention form"
            ),
            alternatives=["one matmul per top-level hl.tile loop", _CPU],
        )
    for contraction in contractions:
        _check_contraction(contraction, loop)


def _check_attention(contractions: list[ir.OpView], loop: ir.Operation) -> None:
    """Two contractions in a grid loop are a flash-attention kernel
    (``attention_shape``) whose head dims lighthouse's attention layout takes."""
    try:
        shape = attention_shape(loop, [op.operation for op in contractions])
    except ValueError as exc:
        raise UnsupportedOperationError(
            "two matmuls outside the attention form on XPU",
            reason=(
                f"{exc}; the XeGPU layout annotation handles one matmul per "
                "kernel, or two in flash-attention form: S = Q @ K^T, row "
                "reductions of S (online softmax), O = O * alpha + P @ V with P "
                "from S, in one inner hl.tile loop over K/V tiles"
            ),
            alternatives=[
                "one matmul per top-level hl.tile loop",
                "the flash-attention form",
                _CPU,
            ],
        ) from None
    if shape.d_head != ATTENTION_D_HEAD or shape.d_value != ATTENTION_D_HEAD:
        raise UnsupportedOperationError(
            f"attention with head dim {shape.d_head} (V: {shape.d_value}) on XPU",
            reason=(
                "lighthouse's attention layout (annotate_attention) is written "
                f"for a head dim of {ATTENTION_D_HEAD}: it hard-codes the Q, K, "
                "V and output tile widths, and infer_xegpu_attention_params "
                "asserts them"
            ),
            alternatives=[f"Q, K and V with a head dim of {ATTENTION_D_HEAD}", _CPU],
        )


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
    inputs = _operand_types(operation)
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


def _operand_types(op: ir.Operation) -> list[str]:
    """The element types a contraction multiplies: its inputs', or the type its
    body truncates an input to first (e.g. attention's f32 ``P`` rounded to
    bf16 for ``P @ V``: vectorized, the truncation feeds the DPAS)."""
    count = len(op.operands) - len(op.results)
    inputs = list(op.operands)[:count]
    if not op.regions or not list(op.regions[0].blocks):
        return [_element_type(value) for value in inputs]
    arguments = list(op.regions[0].blocks[0].arguments)[:count]
    types = []
    for value, argument in zip(inputs, arguments, strict=True):
        users = [use.owner for use in argument.uses]
        if len(users) == 1 and users[0].name == "arith.truncf":
            types.append(str(users[0].results[0].type))
        else:
            types.append(_element_type(value))
    return types


def _check_elementwise(loop: ir.Operation) -> None:
    """An elementwise grid loop stores tiles of one shape that an XeGPU layout
    fits (the inferred one; forced parameters are checked with the config)."""
    shape = kernel_shape(loop)
    if not isinstance(shape, ElementwiseShape):
        raise UnsupportedOperationError(
            "grid loop without stores on XPU",
            reason="an elementwise grid loop must store its results",
            alternatives=[_CPU],
        )
    try:
        shape.layout()
    except ValueError as exc:
        raise UnsupportedOperationError(
            f"{shape} on XPU",
            reason=f"no XeGPU layout fits its tiles: {exc}",
            alternatives=[
                (
                    "2D tiles of 16- or 32-bit elements whose columns are a "
                    "multiple of 16 (block sizes)"
                ),
                _CPU,
            ],
        ) from None
