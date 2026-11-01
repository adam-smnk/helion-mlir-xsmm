"""``check_block_surfaces``: refuse 2D block accesses whose surface exceeds the
hardware's 2D block limits, which compute wrong values instead of failing.

Upstream's XeGPU-to-XeVM lowering gives each ``xegpu.create_nd_tdesc`` of a
memref the whole buffer as its 2D block surface: width ``size[R-1]``, pitch
``stride[R-2]`` and height ``size[R-2] + sum_d (size[d] - 1) * stride[d] /
stride[R-2]`` over the leading dims (rows of all its planes;
``XeGPUToXeVM.cpp`` names this limitation). The block load and store payloads
hold the height (and width and pitch, in bytes) minus 1 in 24-bit fields, so a
taller surface wraps: on the Arc B50 (llvm-project 79ab4054) a reduction over
``25165824 x 64`` f32 rows read rows from 2^23 on as zeros (16777216 rows of
wrong results, no error), while ``16777216 x 64`` was right.

Checked on the kernels after the layout annotation (``xegpu-wg``): every
``xegpu.create_nd_tdesc`` of a statically shaped, strided memref of rank 2 or
more. A surface with more than 2^24 rows, or rows or a pitch of more than 2^24
bytes (the limits of the ``cl_intel_subgroup_2d_block_io`` extension; only the
height was seen to fail), raises :class:`UnsupportedOperationError`. Other
descriptors (pointers with runtime shapes, dynamic memrefs) are left alone.
"""

from __future__ import annotations

from mlir import ir

from ..mlir.support.errors import UnsupportedOperationError

SURFACE_LIMIT = 1 << 24
"""The most rows, and bytes per row and pitch, of a 2D block surface."""


def surface(memref_type: ir.MemRefType) -> tuple[int, int, int] | None:
    """``(height in rows, width in bytes, pitch in bytes)`` of the 2D block
    surface upstream gives a descriptor of the whole ``memref_type``, or
    ``None`` if it is dynamic, has no strided layout or fewer than 2 dims."""
    if memref_type.rank < 2 or not memref_type.has_static_shape:
        return None
    try:
        strides, _ = memref_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        return None
    dynamic = ir.ShapedType.get_dynamic_stride_or_offset()
    if dynamic in strides:
        return None
    shape = memref_type.shape
    row_stride = strides[-2]
    height = shape[-2] + sum(
        (size - 1) * (stride // row_stride if row_stride else 0)
        for size, stride in zip(shape[:-2], strides[:-2], strict=True)
    )
    element_bytes = max(1, memref_type.element_type.width // 8)
    return height, shape[-1] * element_bytes, row_stride * element_bytes


def check_block_surfaces(module: ir.Module) -> None:
    """Raise :class:`UnsupportedOperationError` if a 2D block descriptor in
    ``module`` covers a surface beyond the hardware limits (module docstring)."""
    found: list[str] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name != "xegpu.create_nd_tdesc":
            return ir.WalkResult.ADVANCE
        source = op.operands[0].type
        if not isinstance(source, ir.MemRefType):
            return ir.WalkResult.ADVANCE
        dims = surface(ir.MemRefType(source))
        if dims is None:
            return ir.WalkResult.ADVANCE
        height, width, pitch = dims
        if max(height, width, pitch) > SURFACE_LIMIT:
            found.append(
                f"{source}: {height} rows, {width}-byte rows, {pitch}-byte pitch"
            )
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    if found:
        raise UnsupportedOperationError(
            "2D block surface beyond the hardware limits",
            reason=(
                f"a 2D block access of {found[0]}: at most {SURFACE_LIMIT} rows "
                f"and {SURFACE_LIMIT}-byte rows and pitch (taller surfaces "
                "read and write wrong rows)"
            ),
            alternatives=[
                "split the tensor into views of at most 2^24 rows (or pass the "
                "rows a kernel tiles as separate tensors)",
                "target='cpu' (CPU tensors)",
            ],
        )
