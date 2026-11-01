"""``keep_row_dims``: a reduction kernel's row results stored as a ``[rows]``
vector into a rank-1 buffer are stored as ``[rows, 1]`` into the buffer
expanded to ``[n, 1]`` (the ``keepdim`` form), in GPU kernels before the XeGPU
conversion.

Works around wrong results on hardware (Arc B50, llvm-project 79ab4054). With
subgroups of 8 rows, the ``xegpu.store`` scatter of a ``[rows]`` vector gets
``lane_layout = [8]``; ``xegpu-propagate-layout`` derives the source of the
reduction from it as ``lane_layout = [8, 2]`` (a lane holds a row, pairs of
lanes hold its columns). The load's own lanes cannot follow it (upstream warns
"Failed to identify lane layouts" and drops the load's layout, and the warning
does not reach a Helion lowering), so the lane pass hands it that layout, whose
lanes hold columns; ``xegpu-sg-to-lane-distribute`` only reinterprets them with
a ``vector.shape_cast``. Each lane then reduces a column over the rows, one
shuffle joins pairs of lanes, and lanes ``k`` and ``k + 8`` store to row
``k % 8``: pairs of equal, wrong values (``out[2i] == out[2i+1]``). The same rows
stored as a ``[rows, 1]`` column get ``lane_layout = [1, 16]`` for the source,
which is what the load gives. The two stores write the same elements, so the
rewrite keeps the semantics. ``xevm_lane_reductions`` refuses the reductions
over part of a subgroup that a store this stage does not match leads to.

Matched exactly: a ``vector.transfer_write`` without a mask of a rank-1 vector
derived from a ``vector.multi_reduction`` of the innermost dim of a 2D vector
(through elementwise ``arith``/``math`` ops and the results of ``scf.for`` loops
that carry it), into a rank-1, static, contiguous memref (identity layout, or
strided with stride 1: a rank-3+ kernel's row of a collapsed unit tile), in
bounds, with the identity permutation map, in a ``gpu.func``. Everything else
is left alone.
"""

from __future__ import annotations

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import vector


def _row_result(value: ir.Value, seen: dict[ir.Value, bool] | None = None) -> bool:
    """``value`` is the ``[rows]`` result of a row reduction of a 2D vector, of
    elementwise ``arith``/``math`` ops on such results (e.g. a mean, or the cast
    to the output's dtype), or of an ``scf.for`` that yields one (a running
    maximum or sum over column tiles). ``seen`` memoizes: a value can be reached
    along many paths."""
    seen = {} if seen is None else seen
    if value not in seen:
        seen[value] = _compute_row_result(value, seen)
    return seen[value]


def _compute_row_result(value: ir.Value, seen: dict[ir.Value, bool]) -> bool:
    owner = value.owner
    if isinstance(owner, ir.Block):
        return False
    if owner.name == "scf.for":
        terminator = owner.regions[0].blocks[0].operations[-1]
        yielded = terminator.operands[ir.OpResult(value).result_number]
        return _row_result(yielded, seen)
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
        _row_result(v, seen) for v in vectors
    )


def _contiguous(buffer: ir.MemRefType) -> bool:
    """A rank-1 ``buffer``'s elements are adjacent: an identity layout, or a
    strided one with stride 1 and any offset."""
    layout = buffer.layout
    if isinstance(layout, ir.StridedLayoutAttr):
        return list(layout.strides) == [1]
    return isinstance(
        layout, ir.AffineMapAttr
    ) and layout.value == ir.AffineMap.get_identity(1)


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
    if not _contiguous(buffer):
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
    layout = None
    if isinstance(buffer.layout, ir.StridedLayoutAttr):
        layout = ir.StridedLayoutAttr.get(buffer.layout.offset, [1, 1])
    expanded = ir.MemRefType.get(
        [buffer.shape[0], 1],
        buffer.element_type,
        layout=layout,
        memory_space=buffer.memory_space,
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
