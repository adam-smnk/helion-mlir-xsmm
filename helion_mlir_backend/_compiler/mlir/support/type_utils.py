"""Torch dtype ↔ MLIR type utilities."""

from __future__ import annotations

import mlir.ir as ir
import torch

_DTYPE_TO_MLIR: dict[torch.dtype, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
}

_MLIR_TO_DTYPE: dict[str, torch.dtype] = {
    mlir_name: dtype for dtype, mlir_name in _DTYPE_TO_MLIR.items()
}


def mlir_dtype_to_torch(name: str) -> torch.dtype | None:
    """The PyTorch dtype of an MLIR scalar type name, if it has one."""
    return _MLIR_TO_DTYPE.get(name)


def torch_dtype_to_mlir(dtype: torch.dtype) -> ir.Type:
    """Convert a :mod:`torch` dtype to an MLIR scalar type.

    Must be called while an ``mlir.ir.Context`` is active.
    """

    from .errors import UnsupportedOperationError

    name = _DTYPE_TO_MLIR.get(dtype)
    if name is None:
        reason = (
            "arith/linalg integers are signless and this backend lowers them with "
            "signed semantics"
            if dtype == torch.uint8
            else "no MLIR element type mapping"
        )
        raise UnsupportedOperationError(f"dtype {dtype}", reason=reason)
    return ir.Type.parse(name)


def static_dim(size: int | torch.SymInt) -> int:
    """A tensor dim as an MLIR dim: the dynamic size sentinel if it is symbolic."""
    if isinstance(size, torch.SymInt):
        expr = size.node.expr
        return ir.ShapedType.get_dynamic_size() if expr.free_symbols else int(expr)
    return int(size)
