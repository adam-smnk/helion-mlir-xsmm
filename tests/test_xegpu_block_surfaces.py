"""``xegpu_block_surfaces``: the 2D block surface upstream gives a descriptor
of a whole memref, and the hardware limits it is checked against (the kernel
level tests: ``test_xpu_kernelbench.py::test_block_surface_rows_limit``)."""

from __future__ import annotations

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms.xegpu_block_surfaces import (
    SURFACE_LIMIT,
)
from helion_mlir_backend._compiler.mlir_transforms.xegpu_block_surfaces import (
    check_block_surfaces,
)
from helion_mlir_backend._compiler.mlir_transforms.xegpu_block_surfaces import surface


def _surface(type_text: str) -> tuple[int, int, int] | None:
    with ir.Context():
        return surface(ir.MemRefType(ir.Type.parse(type_text)))


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("memref<4096x393216xf32>", (4096, 393216 * 4, 393216 * 4)),
        # Leading dims add their planes' rows (upstream's flattened plane).
        ("memref<8x64x32xbf16>", (8 * 64, 64, 64)),
        # A strided view: the pitch is the row stride, planes 2 rows apart.
        (
            "memref<4x16x8xf32, strided<[256, 32, 1], offset: ?>>",
            (16 + 3 * 8, 32, 128),
        ),
        ("memref<25165824x64xf32>", (25165824, 256, 256)),
    ],
)
def test_surface(type_text: str, expected: tuple[int, int, int]) -> None:
    assert _surface(type_text) == expected


@pytest.mark.parametrize(
    "type_text",
    [
        "memref<?x64xf32>",
        "memref<64xf32>",
        "memref<8x64xf32, strided<[?, 1]>>",
        "memref<8x64xf32, affine_map<(d0, d1) -> (d1, d0)>>",
    ],
    ids=["dynamic", "1d", "dynamic_stride", "affine_layout"],
)
def test_surface_none(type_text: str) -> None:
    """Descriptors whose surface is not static are not checked."""
    assert _surface(type_text) is None


def test_surface_limit() -> None:
    assert SURFACE_LIMIT == 1 << 24
    height, width, pitch = _surface("memref<16777216x64xf32>")
    assert height == SURFACE_LIMIT
    _, width, pitch = _surface("memref<2x4194305xf32>")
    assert width > SURFACE_LIMIT and pitch > SURFACE_LIMIT


def _check(arg_type: str, descriptor: str, source: str | None = None) -> None:
    """``check_block_surfaces`` on a function creating one descriptor of its
    argument (``source``: the op's operand and type text, default a memref)."""
    operand = source or f"%src : {arg_type}"
    text = f"""
func.func @f(%src: {arg_type}, %h: index, %w: index) {{
  %c1 = arith.constant 1 : index
  %t = xegpu.create_nd_tdesc {operand} -> !xegpu.tensor_desc<{descriptor}>
  return
}}
"""
    with ir.Context(), ir.Location.unknown():
        check_block_surfaces(ir.Module.parse(text))


@pytest.mark.parametrize(
    "arg_type",
    [
        "memref<16777217x16xf32>",  # 2^24 + 1 rows
        "memref<8x4194305xf32>",  # rows of 2^24 + 4 bytes
        "memref<8x16xf32, strided<[4194305, 1]>>",  # a 2^24 + 4-byte pitch
        "memref<4097x4096x16xf32>",  # planes adding up to 2^24 + 4096 rows
    ],
    ids=["height", "width", "pitch", "planes"],
)
def test_check_block_surfaces_rejects(arg_type: str) -> None:
    with pytest.raises(UnsupportedOperationError, match="2D block surface"):
        _check(arg_type, "8x16xf32")


@pytest.mark.parametrize(
    ("arg_type", "descriptor", "source"),
    [
        ("memref<16777216x16xf32>", "8x16xf32", None),  # exactly 2^24 rows
        ("memref<4194304xf32>", "16xf32", None),  # rank 1: no 2D surface
        # A pointer with runtime shape and strides: nothing static to check.
        (
            "ui64",
            "8x16xf32",
            "%src, shape : [%h, %w], strides : [%w, %c1] : ui64",
        ),
    ],
    ids=["2^24_rows", "rank1", "integer_source"],
)
def test_check_block_surfaces_leaves(
    arg_type: str, descriptor: str, source: str | None
) -> None:
    _check(arg_type, descriptor, source)
