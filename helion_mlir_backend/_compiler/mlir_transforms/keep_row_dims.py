"""``keep_row_dims``: a reduction kernel's row results stored as a ``[rows]``
vector into a rank-1 buffer are stored as ``[rows, 1]`` into the buffer
expanded to ``[n, 1]`` (the ``keepdim`` form), in GPU kernels before the XeGPU
conversion.

Works around wrong results on hardware (Arc B50, llvm-project 79ab4054): the
``xegpu.store`` scatter of a ``[rows]`` row vector, anchored with the slice of
the reduction's layout (``#xegpu.slice<..., dims = [1]>``, 8 rows per subgroup
of 16 lanes), writes pairs of equal, wrong values (``out[2i] == out[2i+1]``),
while the same rows stored as a ``[rows, 1]`` column (lighthouse's layout of
``keepdim`` results) are right. The two stores write the same elements, so the
rewrite keeps the semantics.

Matched exactly: a ``vector.transfer_write`` without a mask of a rank-1
vector produced by a ``vector.multi_reduction`` (of the innermost dim of a 2D
vector) into a rank-1, static, identity-layout memref, in bounds, with the identity
permutation map, in a ``gpu.func``. Everything else is left alone.
"""

from __future__ import annotations

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import vector


def _row_result(value: ir.Value) -> bool:
    """``value`` is the ``[rows]`` result of a row reduction of a 2D vector, or
    of elementwise ``arith``/``math`` ops on such results (e.g. a mean, or the
    cast to the output's dtype)."""
    owner = value.owner
    if isinstance(owner, ir.Block):
        return False
    if owner.name == "vector.multi_reduction":
        source = ir.VectorType(owner.operands[0].type)
        dims = [
            int(d) for d in ir.DenseI64ArrayAttr(owner.attributes["reduction_dims"])
        ]
        return source.rank == 2 and dims == [1]
    if owner.name.split(".")[0] not in ("arith", "math") or owner.regions:
        return False
    vectors = [v for v in owner.operands if isinstance(v.type, ir.VectorType)]
    shape = ir.VectorType(value.type).shape
    return all(ir.VectorType(v.type).shape == shape for v in vectors) and any(
        _row_result(v) for v in vectors
    )


def matches(op: ir.Operation) -> bool:
    """Whether ``op`` is a row-result store :func:`keep_row_dims` rewrites."""
    if op.name != "vector.transfer_write" or len(op.operands) != 3:
        return False  # a mask is a fourth operand
    value, base, _index = op.operands
    if not isinstance(value.type, ir.VectorType) or ir.VectorType(value.type).rank != 1:
        return False
    if not isinstance(base.type, ir.MemRefType):
        return False
    buffer = ir.MemRefType(base.type)
    identity = ir.AffineMap.get_identity(1)
    if buffer.rank != 1 or not buffer.has_static_shape:
        return False
    if not isinstance(buffer.layout, ir.AffineMapAttr):
        return False
    if ir.AffineMapAttr(buffer.layout).value != identity:
        return False
    in_bounds = op.attributes.get("in_bounds")
    if in_bounds is None or [ir.BoolAttr(b).value for b in in_bounds] != [True]:
        return False
    permutation = ir.AffineMapAttr(op.attributes["permutation_map"]).value
    return permutation == identity and _row_result(value)


def _rewrite(op: ir.Operation) -> None:
    value, base, index = op.operands
    rows = ir.VectorType(value.type)
    buffer = ir.MemRefType(base.type)
    column = ir.VectorType.get([rows.shape[0], 1], rows.element_type)
    expanded = ir.MemRefType.get(
        [buffer.shape[0], 1], buffer.element_type, memory_space=buffer.memory_space
    )
    with ir.InsertionPoint(op), op.location:
        zero = arith.ConstantOp(ir.IndexType.get(), 0).result
        target = memref.ExpandShapeOp(
            expanded, base, [[0, 1]], [], [buffer.shape[0], 1]
        ).result
        cast = vector.ShapeCastOp(column, value).result
        vector.TransferWriteOp(
            None,
            cast,
            target,
            [index, zero],
            ir.AffineMap.get_identity(2),
            [True, True],
        )
    op.erase()


def keep_row_dims(module: ir.Module) -> int:
    """Rewrites the matching stores in ``module``'s ``gpu.func``s in place;
    returns how many."""
    found: list[ir.Operation] = []

    def in_kernel(op: ir.Operation) -> ir.WalkResult:
        if matches(op):
            found.append(op)
        return ir.WalkResult.ADVANCE

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.func":
            op.walk(in_kernel)
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    for op in found:
        _rewrite(op)
    return len(found)
