"""What the XPU (XeGPU) pipeline compiles, checked on the inlined tensor IR before
lowering: kernels outside it fail with an :class:`UnsupportedOperationError`
naming the gap, not a transform failure deep in lighthouse.

The gates are conservative, by IR property (not by kernel):

- static shapes, and computation on static tiles in every grid loop: a partial
  tile is a full tile with out-of-bounds accesses (``partial_tiles``), on rows
  of a multiple of 16 bytes if it is 2D;
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
  e.g. attention's ``P``) accumulating in f32 (XeGPU DPAS);
- no math function outside the device-verified ones in a form IGC is known to
  miscompile (:func:`check_math`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir import ir
from mlir.dialects import linalg

from ..mlir.support.errors import UnsupportedOperationError
from .xpu_params import ATTENTION_D_HEAD
from .xpu_params import ElementwiseShape
from .xpu_params import GemmShape
from .xpu_params import attention_shape
from .xpu_params import is_sibling_gemms
from .xpu_params import kernel_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

_PITCH_ALIGNMENT = 16
_DPAS_OPERANDS = frozenset({"f16", "bf16"})
_DPAS_ACCUMULATOR = "f32"
_CPU = "target='cpu' (CPU tensors)"
_UNIT_LEADING_BLOCKS = (
    "block size 1 on the leading (e.g. batch or head) dims, tiling only the last two"
)


def check_xegpu_capabilities(
    module: ir.Module,
    entry: str,
    ir_layouts: Sequence[bool] = (),
    ir_layouts_default: bool = False,
) -> list[ir.Operation]:
    """Raise :class:`UnsupportedOperationError` if ``entry`` of the inlined and
    prepared (unit tile dims folded) ``module`` is outside what the XeGPU
    pipeline compiles; returns its grid loops (top-level ``scf.forall``), one
    GPU kernel each, in program order. ``ir_layouts[i]``: kernel ``i``'s layouts
    are derived from its IR, which has no kernel classes: what they cannot lay
    out is refused there, by IR property."""
    func = _function(module, entry)
    _check_static_arguments(func)
    loops = _grid_loops(func)
    for index, loop in enumerate(loops):
        _check_full_tiles(loop)
        _check_tile_ranks(loop)
        _check_kernel_class(
            loop, ir_layouts[index] if index < len(ir_layouts) else ir_layouts_default
        )
        check_math(loop)
    return loops


_HALF = frozenset({"f16", "bf16"})
DEVICE_VERIFIED = frozenset(
    {
        "math.absf",
        "math.acos",
        "math.asin",
        "math.atan",
        "math.atanh",
        "math.ceil",
        "math.cos",
        "math.cosh",
        "math.erf",
        "math.exp",
        "math.exp2",
        "math.expm1",
        "math.floor",
        "math.fpowi",
        "math.log",
        "math.log1p",
        "math.log2",
        "math.powf",
        "math.roundeven",
        "math.rsqrt",
        "math.sin",
        "math.sinh",
        "math.sqrt",
        "math.tan",
        "math.tanh",
    }
)
"""Math functions that compute right on the device on bf16 values and with their
result rounded straight to a 16-bit float, in 1D and 2D tiles, for every input
and output dtype, in f32 and in the input's own dtype (Arc B50, IGC of
/swtools/intel-gpu/latest, llvm-project 79ab4054;
``tests/test_xpu_runtime.py``: ``test_math_every_pattern``, every bf16 and f16
pattern; ``test_math_functions``, compound forms). Any other
function in those forms is rejected: these forms hit IGC miscompiles that the
pipeline works around (``xpu_transforms.IGC_OPTIONS``, ``xevm_bf16_extensions``,
``xevm_fabs``), and a function nobody ran on the device has no such evidence.
f32 math on f32 values is not affected."""


def check_math(loop: ir.Operation) -> None:
    """Raise :class:`UnsupportedOperationError` for a math function in ``loop``,
    not in :data:`DEVICE_VERIFIED`, in a form that IGC is known to miscompile:
    its result rounded straight to a 16-bit float, or on a bf16 value."""
    for op in _walk(loop):
        if (
            not op.name.startswith("math.")
            or len(op.results) != 1
            or op.name in DEVICE_VERIFIED
        ):
            continue
        rounded = _rounded_to(op)
        if rounded is not None:
            raise UnsupportedOperationError(
                f"{op.name} rounded to {rounded} on XPU",
                reason=(
                    f"{op.name}'s result rounded straight to {rounded} was not "
                    "verified on the device (verified right: f32 results, or "
                    "more math on the result before the rounding)"
                ),
                alternatives=["an f32 output", _CPU],
            )
        if any(_from_bf16(operand, {}) for operand in op.operands):
            raise UnsupportedOperationError(
                f"{op.name} of a bf16 value on XPU",
                reason=(
                    f"{op.name} of a bf16 value (extended to f32, also after "
                    "arithmetic on it) was not verified on the device"
                ),
                alternatives=["f16 or f32 inputs", _CPU],
            )


def _from_bf16(value: ir.Value, seen: dict[ir.Value, bool]) -> bool:
    """``value`` is a bf16 value, or computed from one by ``arith.extf`` and
    elementwise ``arith`` ops (``negf``, ``mulf``, ``select``, ...)."""
    if value in seen:
        return seen[value]
    seen[value] = False
    if _element_type(value) == "bf16":
        seen[value] = True
        return True
    owner = value.owner
    if isinstance(owner, ir.Block) or not owner.name.startswith("arith."):
        return False
    result = any(_from_bf16(operand, seen) for operand in owner.operands)
    seen[value] = result
    return result


def _rounded_to(op: ir.Operation) -> str | None:
    """The 16-bit float type ``op``'s result is rounded to directly: a user's
    ``arith.truncf``; for 16-bit math (extended to f32 and rounded back), its
    own type unless only more math uses it (``tanh(x) * exp(x)`` in bf16 is
    right; ``abs(x)`` stored or converted is not)."""
    users = [use.owner for use in op.results[0].uses]
    result = _element_type(op.results[0])
    if result in _HALF:
        computed = all(
            user.name.startswith(("arith.", "math."))
            and user.name not in ("arith.extf", "arith.truncf")
            for user in users
        )
        return None if users and computed else result
    for user in users:
        if user.name == "arith.truncf":
            rounded = _element_type(user.results[0])
            if rounded in _HALF:
                return rounded
    return None


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


_RUNTIME_SIZED_REGION_OPS = frozenset(
    {
        "tensor.extract_slice",
        "tensor.parallel_insert_slice",
        "tensor.cast",
        "tensor.dim",
        "vector.transfer_read",
        "vector.transfer_write",
        "scf.for",
        "scf.yield",
    }
)
"""Ops that may hold runtime-sized tensors in a grid loop: a partial tile's region
of an output, written and read by out-of-bounds transfers
(:mod:`~..mlir_transforms.partial_tiles`). All computation is on static tiles."""


def _check_full_tiles(loop: ir.Operation) -> None:
    """Every computation in a grid loop is on static tiles. A partial tile needs
    its accesses as out-of-bounds transfers (``partial_tiles.to_transfers``):
    other pads, runtime-sized math or a runtime-sized tile are refused."""
    for op in _walk(loop):
        if op.name == "tensor.pad" or (
            op.name not in _RUNTIME_SIZED_REGION_OPS
            and any(
                isinstance(value.type, ir.ShapedType)
                and not ir.ShapedType(value.type).has_static_shape
                for value in [*op.operands, *op.results]
            )
        ):
            raise UnsupportedOperationError(
                "partial tiles on XPU",
                reason=(
                    f"'{op.name}' works on a partial or runtime-sized tile that "
                    "the XeGPU pipeline cannot express as out-of-bounds block "
                    "accesses of a full tile"
                ),
                alternatives=["block sizes that divide the sizes they tile", _CPU],
            )
    _check_partial_pitches(loop)


def check_partial_gemms(loops: Sequence[ir.Operation], derived: Sequence[bool]) -> None:
    """A GEMM with partial tiles needs layouts derived from the IR: lighthouse's
    parameter selection, used by its annotation, takes tiles that divide the
    matrix."""
    for loop, from_ir in zip(loops, derived, strict=True):
        if from_ir or not isinstance(kernel_shape(loop), GemmShape):
            continue
        for op in _walk(loop):
            in_bounds = (
                op.attributes.get("in_bounds")
                if op.name in ("vector.transfer_read", "vector.transfer_write")
                else None
            )
            if in_bounds and not all(ir.BoolAttr(flag).value for flag in in_bounds):
                raise UnsupportedOperationError(
                    "a GEMM with partial tiles on lighthouse's parameter selection",
                    reason=(
                        "lighthouse's selection takes workgroup and K tiles that "
                        "divide the matrix"
                    ),
                    alternatives=[
                        'xegpu_layouts="ir" (the default)',
                        "block sizes that divide the sizes they tile",
                    ],
                )


def _surface_columns(value: ir.Value) -> int | None:
    """The row size of the surface ``value`` is a region of: through slices and the
    values loops carry, to the first value that is neither. ``None``: runtime-sized
    there, refused elsewhere."""
    while True:
        owner = value.owner
        if isinstance(owner, ir.Block):
            loop = owner.owner
            carried = ir.BlockArgument(value).arg_number - 1
            if loop.name != "scf.for" or carried < 0:
                break
            value = loop.operands[3 + carried]
        elif owner.name == "tensor.extract_slice":
            value = owner.operands[0]
        else:
            break
    columns = ir.ShapedType(value.type).shape[-1]
    return None if columns == ir.ShapedType.get_dynamic_size() else columns


def _check_partial_pitches(loop: ir.Operation) -> None:
    """A partial tile in 2D is a block access running off its surface, and a
    surface's pitch must be a multiple of 16 bytes: the B50 rounds any other pitch
    up, silently, so rows come out wrong (upstream's check also refuses it, deep
    in the lowering). One-row tiles use gathers and scatters, on any row size."""
    for op in _walk(loop):
        if op.name not in ("vector.transfer_read", "vector.transfer_write"):
            continue
        read = op.name == "vector.transfer_read"
        vector_type = ir.ShapedType(op.results[0].type if read else op.operands[0].type)
        flags = [ir.BoolAttr(flag).value for flag in op.attributes["in_bounds"]]
        if all(flags) or vector_type.rank < 2:
            continue
        columns = _surface_columns(op.operands[0 if read else 1])
        if columns is None:
            continue
        # Narrower than a byte (i1) is stored in a byte each.
        bytes_per_row = columns * ((vector_type.element_type.width + 7) // 8)
        if bytes_per_row % _PITCH_ALIGNMENT != 0:
            raise UnsupportedOperationError(
                f"partial tiles on rows of {columns} elements on XPU",
                reason=(
                    f"its rows of {bytes_per_row} bytes are no multiple of "
                    f"{_PITCH_ALIGNMENT}, the pitch 2D block accesses need; "
                    "the hardware rounds another pitch up, with wrong rows"
                ),
                alternatives=[
                    f"row sizes that are a multiple of {_PITCH_ALIGNMENT} bytes",
                    "one-row tiles of an elementwise kernel (block size 1 on rows)",
                    _CPU,
                ],
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
        # A partial tile's region of an output: its leading unit dims are not in
        # the transfer's vector.
        if not ir.ShapedType(tile.type).has_static_shape:
            while len(shape) > 2 and shape[0] == 1:
                shape = shape[1:]
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


def _check_kernel_class(loop: ir.Operation, ir_layouts: bool = False) -> None:
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
    if ir_layouts:
        for contraction in contractions:
            _check_contraction(contraction, loop)
        return
    if not contractions and not reductions:
        _check_elementwise(loop)
        return
    if len(contractions) == 1 and reductions:
        _check_matmul_and_reduction(loop)
    if len(contractions) == 2 and is_sibling_gemms(contractions):
        if reductions:
            raise UnsupportedOperationError(
                "two GEMMs and a row reduction in one grid loop on XPU",
                reason="no layout form has both",
                alternatives=[
                    "the reduction in its own top-level hl.tile loop, after hl.barrier()",
                    _CPU,
                ],
            )
    elif len(contractions) >= 2 and len(contractions) % 2 == 0:
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


def _check_matmul_and_reduction(loop: ir.Operation) -> None:
    """A matmul and a row reduction in one grid loop: lighthouse's annotation tries
    its attention layout on them, which asserts a contracted tile of 64, and with
    such a tile falls to the GEMM annotation with a reduction it does not lay out:
    180 of 256 rows wrong on the B50, silently."""
    shape = kernel_shape(loop)
    if isinstance(shape, GemmShape):
        raise UnsupportedOperationError(
            "a matmul and a row reduction in one grid loop on XPU",
            reason=(
                "lighthouse's layout annotation has no form for them: its attention "
                f"layout asserts a contracted tile of {ATTENTION_D_HEAD}, and its "
                "GEMM layout leaves most rows of the reduction wrong"
            ),
            alternatives=[
                "the reduction in its own top-level hl.tile loop, after hl.barrier()",
                _CPU,
            ],
        )


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
            alternatives=[
                "a K block size below K",
                "K padded with zeros beyond its block size, in the model",
                _CPU,
            ],
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
