"""Partial tiles as full tiles on the true surface, for the XPU's block accesses.

Helion generates a partial last tile as a padded copy of a shorter slice, stored
back through a shorter slice, and versions the tile on being full. On the XPU a
2D block access does this by itself: reads outside the surface return zero and
writes outside it are dropped, in width and height, for plain, VNNI and
transposed block ops (measured on the Arc B50). So a partial tile is the full
tile's kernel with accesses that may run out of bounds:

- :func:`to_transfers`, on the inlined tensor IR, before ``prepare``:

  * the version ``scf.if (extent == block)`` keeps its general branch;
  * ``tensor.pad(tensor.extract_slice(x, ...))`` padding zeros becomes a
    ``vector.transfer_read`` of ``x`` with ``in_bounds = false`` on the partial
    dims, written into an empty tile;
  * ``tensor.insert_slice(tensor.extract_slice(tile, 0, sizes), dest, ...)``
    becomes a ``vector.transfer_write`` of the whole tile into ``dest``, with
    ``in_bounds = false`` on the partial dims: its elements past ``dest`` are
    not stored.

  Both are defined IR, with exactly the padded slices' semantics. Leading unit
  dims of a tile are left out of the vectors (a permutation map drops them), as
  the unit-dim folding of ``prepare`` leaves full tiles.
- :func:`fold_tile_views`, after bufferization: such a transfer moves off the
  tile's subview onto the buffer, when the subview ends where the buffer does,
  so bounds checks are against the true surface (and ``VectorToXeGPU`` emits
  ``boundary_check`` block ops, or masked gathers and scatters for 1D tiles).

Conditions of tile masks on rows go too (:func:`_drop_row_masks`). A kernel
without partial tiles has no tile extents, and this is a no-op for it.

Partial M on a rank-3 tensor reaches plane bases that are no multiple of 64 bytes
(bf16 ``(3, 253, 248)``: 48 off), which ``MLIRTarget.buffer_alignment`` calls
undefined. Right on the B50 (``scale_rank3_planes``); unchecked on the B70.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import tensor
from mlir.dialects import vector

if TYPE_CHECKING:
    from collections.abc import Sequence


def _walk(op: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested != op:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    op.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _owner(value: ir.Value) -> ir.Operation | None:
    return None if isinstance(value.owner, ir.Block) else value.owner


_EXTENT_EXPR = re.compile(r"^(?:-d0|d0 \* -(\d+)) \+ (\d+)$")


def _extent_parts(op: ir.Operation | None) -> tuple[int, int, int] | None:
    """``(block, step, n)`` of a tile extent ``affine.min (d0) -> (block, n - step *
    d0)``, in either order; ``d0`` is the tile's offset or, with a ``step``, its
    number."""
    if op is None or op.name != "affine.min":
        return None
    results = ir.AffineMapAttr(op.attributes["map"]).value.results
    if len(results) != 2:
        return None
    constants = [e for e in results if isinstance(e, ir.AffineConstantExpr)]
    others = [e for e in results if not isinstance(e, ir.AffineConstantExpr)]
    if len(constants) != 1 or len(others) != 1:
        return None
    match = _EXTENT_EXPR.match(str(others[0]))
    if match is None:
        return None
    return constants[0].value, int(match[1] or 1), int(match[2])


def _inline_general_versions(module: ir.Module) -> None:
    """Each ``scf.if (extent == block)`` becomes its else region: with partial
    tiles it is right for a full tile too."""
    while True:
        version = next(
            (
                op
                for op in _walk(module.operation)
                if op.name == "scf.if"
                and list(op.regions[1].blocks)
                and (cond := _owner(op.operands[0])) is not None
                and cond.name == "arith.cmpi"
                and _extent_parts(_owner(cond.operands[0])) is not None
            ),
            None,
        )
        if version is None:
            return
        ops = list(version.regions[1].blocks[0].operations)
        for nested in ops[:-1]:
            nested.operation.move_before(version)
        for result, value in zip(version.results, ops[-1].operands, strict=True):
            result.replace_all_uses_with(value)
        version.erase()


def _indices(offsets: Sequence[ir.Value], statics: Sequence[int]) -> list[ir.Value]:
    index = ir.IndexType.get()
    dynamic = ir.ShapedType.get_dynamic_size()
    values = iter(offsets)
    return [
        next(values) if s == dynamic else arith.ConstantOp(index, s).result
        for s in statics
    ]


def _kept(shape: Sequence[int]) -> int:
    """How many leading unit dims a tile's vector leaves out: all of them, but at
    least a 1D tile is left."""
    leading = 0
    while leading < len(shape) - 1 and shape[leading] == 1:
        leading += 1
    return leading


def _reassociation(rank: int, leading: int) -> list[list[int]]:
    return [list(range(leading + 1)), *[[d] for d in range(leading + 1, rank)]]


def _zero(element: ir.Type) -> ir.Value:
    integer = isinstance(element, ir.IntegerType)
    return arith.ConstantOp(element, 0 if integer else 0.0).result


def _zero_pad(pad: tensor.PadOp) -> bool:
    block = pad.region.blocks[0]
    yielded = _owner(list(block.operations)[-1].operands[0])
    if yielded is None or yielded.name != "arith.constant":
        return False
    value = yielded.attributes["value"]
    return isinstance(value, (ir.FloatAttr, ir.IntegerAttr)) and value.value == 0


def _partial_dims_end_surface(
    view: ir.OpView, surface: ir.Value, leading: int, partial: Sequence[bool]
) -> bool:
    """Every partial dim of ``view`` ends where ``surface`` does: only there a block
    access of the full tile reads zeros and drops writes. Elsewhere it would touch
    real elements past the tile. A runtime-sized ``surface`` is a slice of an outer
    tile, itself checked when it is written back. An offset clamped to the surface
    is a tile that may lie wholly past it, which nothing proves in range."""
    for dim, short in enumerate(partial, leading):
        if not short or ir.ShapedType.is_dynamic_size(
            ir.ShapedType(surface.type).shape[dim]
        ):
            continue
        offset = _at(view.static_offsets, view.offsets, dim)
        owner = None if isinstance(offset, int) else _owner(offset)
        if (owner is not None and owner.name in ("arith.minsi", "affine.min")) or (
            not _ends_with_source(view, dim, surface)
        ):
            return False
    return True


def _pad_to_read(pad: tensor.PadOp) -> bool:
    """See the module's docstring; leaves other pads, which the gate refuses."""
    source = _owner(pad.source)
    if (
        source is None
        or source.name != "tensor.extract_slice"
        or any(pad.static_low)
        or not _zero_pad(pad)
    ):
        return False
    slice_op = source.opview
    if any(s != 1 for s in slice_op.static_strides):
        return False
    result = ir.RankedTensorType(pad.result.type)
    base = ir.RankedTensorType(slice_op.source.type)
    if not result.has_static_shape or base.rank != result.rank:
        return False
    leading = _kept(result.shape)
    shape = list(result.shape[leading:])
    partial = [
        size != full
        for size, full in zip(slice_op.static_sizes, result.shape, strict=True)
    ][leading:]
    if not _partial_dims_end_surface(slice_op, slice_op.source, leading, partial):
        return False
    with ir.InsertionPoint(pad), pad.location:
        zero = _zero(result.element_type)
        read = vector.TransferReadOp(
            ir.VectorType.get(shape, result.element_type),
            slice_op.source,
            _indices(slice_op.offsets, slice_op.static_offsets),
            ir.AffineMap.get_minor_identity(base.rank, len(shape)),
            zero,
            [not p for p in partial],
        ).result
        empty = tensor.EmptyOp(shape, result.element_type).result
        index = ir.IndexType.get()
        written = vector.TransferWriteOp(
            empty.type,
            read,
            empty,
            [arith.ConstantOp(index, 0).result] * len(shape),
            ir.AffineMap.get_minor_identity(len(shape), len(shape)),
            [True] * len(shape),
        ).result
        if leading:
            written = tensor.ExpandShapeOp(
                pad.result.type,
                written,
                _reassociation(result.rank, leading),
                [],
                list(result.shape),
            ).result
    pad.result.replace_all_uses_with(written)
    pad.operation.erase()
    return True


def _insert_to_write(op: tensor.InsertSliceOp) -> bool:
    """See the module's docstring; leaves other inserts, which the gate refuses
    if they are runtime-sized."""
    part = _owner(op.source)
    if part is None or part.name != "tensor.extract_slice":
        return False
    extract = part.opview
    tile_type = ir.RankedTensorType(extract.source.type)
    dest_type = ir.RankedTensorType(op.dest.type)
    if (
        not tile_type.has_static_shape
        or tile_type.rank != dest_type.rank
        or any(extract.offsets)
        or any(extract.static_offsets)
        or any(s != 1 for s in [*extract.static_strides, *op.static_strides])
    ):
        return False
    leading = _kept(tile_type.shape)
    shape = list(tile_type.shape[leading:])
    partial = [
        size != full
        for size, full in zip(op.static_sizes, tile_type.shape, strict=True)
    ][leading:]
    if not any(partial) or not _partial_dims_end_surface(op, op.dest, leading, partial):
        return False
    with ir.InsertionPoint(op), op.location:
        element = tile_type.element_type
        zero = _zero(element)
        tile = extract.source
        if leading:
            tile = tensor.CollapseShapeOp(
                ir.RankedTensorType.get(shape, element),
                tile,
                _reassociation(tile_type.rank, leading),
            ).result
        index = ir.IndexType.get()
        loaded = vector.TransferReadOp(
            ir.VectorType.get(shape, element),
            tile,
            [arith.ConstantOp(index, 0).result] * len(shape),
            ir.AffineMap.get_minor_identity(len(shape), len(shape)),
            zero,
            [True] * len(shape),
        ).result
        written = vector.TransferWriteOp(
            op.dest.type,
            loaded,
            op.dest,
            _indices(op.offsets, op.static_offsets),
            ir.AffineMap.get_minor_identity(dest_type.rank, len(shape)),
            [not p for p in partial],
        ).result
    op.result.replace_all_uses_with(written)
    op.operation.erase()
    return True


def _in_linalg_body(op: ir.Operation) -> ir.Operation | None:
    parent = op.parent
    while parent is not None and parent.name != "scf.forall":
        if parent.name.startswith("linalg."):
            return parent
        parent = parent.parent
    return None


def _reduces_only_innermost(loop: ir.Operation) -> bool:
    """Every linalg op in ``loop`` that reduces, reduces the last dim of its
    operands: kernels with other reductions keep their row masks."""
    for op in _walk(loop):
        if not op.name.startswith("linalg.") or "iterator_types" not in op.attributes:
            continue
        kinds = [str(t) for t in ir.ArrayAttr(op.attributes["iterator_types"])]
        reduced = [i for i, kind in enumerate(kinds) if "reduction" in kind]
        if reduced and reduced != [len(kinds) - 1]:
            return False
    return True


def _selects_a_constant(condition: ir.Operation) -> bool:
    """Everything ``condition`` decides is a select between a value and a constant
    when false, directly or through ``arith.andi``: a mask, nothing else."""
    stack = [condition]
    while stack:
        current = stack.pop().results[0]
        for use in current.uses:
            user = use.owner
            if user.name == "arith.andi":
                stack.append(user)
                continue
            if user.name != "arith.select" or use.operand_number != 0:
                return False
            false_value = _owner(user.operands[2])
            if false_value is None or false_value.name != "arith.constant":
                return False
    return True


def _drop_row_masks(module: ir.Module) -> None:
    """A tile mask's condition on a dim other than the last goes, in grid loops
    that only reduce the last dim: padded rows reach no reduction over rows, and
    their results are not stored. Such conditions cost selects, and their index
    vectors assert in upstream's ``WgToSgConvertLayoutOp`` when a dim's index
    vector takes two layouts (LLVM 79ab4054). Only a mask's: a select that keeps
    an old value (a block argument) past the tile says what a running value
    holds there."""
    i1 = ir.IntegerType.get_signless(1)
    for loop in _walk(module.operation):
        if loop.name != "scf.forall" or not _reduces_only_innermost(loop):
            continue
        for op in _walk(loop):
            if op.name != "arith.cmpi":
                continue
            generic = _in_linalg_body(op)
            lhs = _owner(op.operands[0])
            if generic is None or lhs is None or lhs.name != "linalg.index":
                continue
            rank = len(ir.ArrayAttr(generic.attributes["iterator_types"]))
            if ir.IntegerAttr(lhs.attributes["dim"]).value == rank - 1:
                continue
            if not _selects_a_constant(op):
                continue
            with ir.InsertionPoint(op):
                true = arith.ConstantOp(i1, 1).result
            op.results[0].replace_all_uses_with(true)
            op.erase()


def to_transfers(module: ir.Module) -> bool:
    """See the module's docstring. Whether ``module`` has partial tiles."""
    _inline_general_versions(module)
    found = False
    for op in _walk(module.operation):
        if op.name == "tensor.pad":
            found = _pad_to_read(op.opview) or found
    for op in _walk(module.operation):
        if op.name == "tensor.insert_slice":
            found = _insert_to_write(op.opview) or found
    if found:
        _drop_row_masks(module)
    return found


def _at(statics: Sequence[int], values: Sequence[ir.Value], dim: int) -> ir.Value | int:
    """Entry ``dim`` of a slice's mixed offsets or sizes."""
    dynamic = ir.ShapedType.get_dynamic_size()
    remaining = iter(values)
    entry: ir.Value | int = 0
    for position, static in enumerate(statics):
        current = next(remaining) if static == dynamic else static
        if position == dim:
            entry = current
    return entry


def _ends_with_source(
    view: ir.OpView, dim: int, surface: ir.Value | None = None
) -> bool:
    """The slice ``view``'s dim ``dim`` ends where ``surface`` (default: its source)
    does: its size is ``source - offset``, statically or as a tile extent of its
    offset."""
    surface = view.source if surface is None else surface
    source = ir.ShapedType(surface.type).shape[dim]
    offset = _at(view.static_offsets, view.offsets, dim)
    size = _at(view.static_sizes, view.sizes, dim)
    if isinstance(size, int):
        return isinstance(offset, int) and offset + size == source
    extent = _owner(size)
    parts = _extent_parts(extent)
    if parts is None or parts[2] != source:
        return False
    number = extent.operands[0]
    if parts[1] == 1:
        return number == offset
    apply = None if isinstance(offset, int) else _owner(offset)
    return (
        apply is not None
        and apply.name == "affine.apply"
        and apply.operands[0] == number
        and str(ir.AffineMapAttr(apply.attributes["map"]).value.results[0])
        == f"d0 * {parts[1]}"
    )


def fold_tile_views(module: ir.Module) -> None:
    """See the module's docstring. In place, for kernels' out-of-bounds transfers
    on subviews whose partial dims end with the buffer's."""
    for op in _walk(module.operation):
        if op.name not in ("vector.transfer_read", "vector.transfer_write"):
            continue
        in_bounds = op.attributes.get("in_bounds")
        flags = [ir.BoolAttr(flag).value for flag in in_bounds] if in_bounds else []
        if all(flags):
            continue
        at = 0 if op.name == "vector.transfer_read" else 1
        while True:
            view = _owner(op.operands[at])
            if view is None or view.name != "memref.subview":
                break
            sv = view.opview
            memory = ir.ShapedType(sv.source.type)
            vector_rank = len(flags)
            first = memory.rank - vector_rank
            if (
                memory.rank != ir.ShapedType(sv.result.type).rank
                or any(s != 1 for s in sv.static_strides)
                or not all(
                    _ends_with_source(sv, first + i)
                    for i, flag in enumerate(flags)
                    if not flag
                )
            ):
                break
            indices = list(op.operands)[at + 1 : at + 1 + memory.rank]
            with ir.InsertionPoint(op), op.location:
                offsets = _indices(sv.offsets, sv.static_offsets)
                moved = [
                    arith.AddIOp(a, b).result
                    for a, b in zip(offsets, indices, strict=True)
                ]
            for i, value in enumerate(moved):
                op.operands[at + 1 + i] = value
            op.operands[at] = sv.source
