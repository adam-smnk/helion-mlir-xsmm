"""A ``TorchFunctionMode`` around Helion's device-IR tracing.

It records a ``torch.einsum`` expressible as ``linalg.contract`` as one custom op:
PyTorch decomposes ``aten::einsum`` (a ``CompositeImplicitAutograd`` op) into
``permute``/``view``/``bmm`` chains before Helion's ``make_fx`` tracer sees it, and
torch-mlir's own einsum lowering is an equally indirect decomposition. Other
equations keep PyTorch's decomposition.

It also routes a tensor method that mirrors a ``torch`` function with a Helion
device replacement (``x.cumsum(d)``) to that replacement.
"""

from __future__ import annotations

import inspect
import threading
from typing import Callable

import torch
from torch.overrides import TorchFunctionMode

from .support.einsum_spec import build_contract_spec
from .support.einsum_spec import is_contractible

_state = threading.local()

_library = torch.library.Library("helion_mlir", "FRAGMENT")
_library.define("einsum(str equation, Tensor[] operands) -> Tensor")


def _einsum_impl(equation: str, operands: list[torch.Tensor]) -> torch.Tensor:
    _state.reentrant = True
    try:
        return torch.functional.einsum(equation, *operands)
    finally:
        _state.reentrant = False


def _einsum_meta(equation: str, operands: list[torch.Tensor]) -> torch.Tensor:
    # Shape-only: the meta kernel runs on tensors that cannot be handed back to
    # ``_VF.einsum``, and tile extents here are symbolic.
    spec = build_contract_spec(equation, [list(operand.shape) for operand in operands])
    dtype = operands[0].dtype
    for operand in operands[1:]:
        dtype = torch.promote_types(dtype, operand.dtype)
    return operands[0].new_empty(spec.out_shape, dtype=dtype)


_library.impl("einsum", _einsum_impl, "CompositeExplicitAutograd")
_library.impl("einsum", _einsum_meta, "Meta")


def einsum_op_target() -> torch._ops.OpOverload:
    return torch.ops.helion_mlir.einsum.default


def is_einsum_node(node: torch.fx.Node) -> bool:
    """Return whether *node* is a captured ``helion_mlir::einsum`` call."""
    return node.op == "call_function" and node.target is einsum_op_target()


class TraceMode(TorchFunctionMode):
    """Rewrite contractible ``torch.einsum`` calls into ``helion_mlir::einsum``, and
    tensor methods with a Helion device replacement into that replacement."""

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if (replacement := _method_replacement(func, args, kwargs)) is not None:
            return replacement(*args, **kwargs)
        if func is not torch.functional.einsum or kwargs:
            return func(*args, **kwargs)
        if getattr(_state, "reentrant", False):
            return func(*args, **kwargs)

        equation, operands = _normalize_einsum_args(args)
        if equation is not None and _should_capture(equation, operands):
            return torch.ops.helion_mlir.einsum(equation, operands)
        return func(*args, **kwargs)


def _method_replacement(
    func: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]
) -> Callable[..., object] | None:
    """Helion's device replacement of the ``torch`` function a tensor method mirrors
    (``x.cumsum(d)`` for ``torch.cumsum``), if it accepts these arguments: Helion
    replaces only the function form, and the method reaches Inductor's lowering."""
    from helion.language._decorators import get_device_func_replacement

    name = getattr(func, "__name__", None)
    if name is None or getattr(torch.Tensor, name, None) is not func:
        return None
    replacement = get_device_func_replacement(getattr(torch, name, None))
    if replacement is None:
        return None
    try:
        inspect.signature(replacement).bind(*args, **kwargs)
    except TypeError:
        return None
    return replacement


def _normalize_einsum_args(
    args: tuple[object, ...],
) -> tuple[str | None, list[torch.Tensor]]:
    """Return ``(equation, operands)`` for both einsum calling conventions."""
    if not args or not isinstance(args[0], str):
        return None, []
    equation = args[0]
    rest = list(args[1:])
    if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
        rest = list(rest[0])
    if not rest or not all(isinstance(operand, torch.Tensor) for operand in rest):
        return None, []
    return equation, rest


def _should_capture(equation: str, operands: list[torch.Tensor]) -> bool:
    # Symbolic tile extents are compared by symbol, never evaluated, so this
    # installs no shape guards.
    return is_contractible(
        equation,
        [list(operand.shape) for operand in operands],
        require_reduction=True,
    )


def install_trace_mode() -> None:
    """Patch Helion's device-IR lowering to trace under :class:`TraceMode`."""
    from helion._compiler.aten_lowering import aten_lowering_dispatch
    from helion._compiler.aten_lowering import register_lowering
    from helion._compiler.kernel_compiler import KernelCompiler

    if getattr(KernelCompiler, "_helion_mlir_trace_mode_patched", False):
        return

    # The MLIR backend consumes device IR directly and never runs this
    # lowering's codegen; registering it only keeps Helion's shared
    # `prepare_graph_lowerings` pass from rejecting the custom op.
    if einsum_op_target() not in aten_lowering_dispatch:
        register_lowering(einsum_op_target())

    original_lower = KernelCompiler.lower

    def _lower(self: KernelCompiler, hf: object) -> None:
        from .backend import MLIRBackend

        if not isinstance(self.env.backend, MLIRBackend):
            return original_lower(self, hf)
        with TraceMode():
            return original_lower(self, hf)

    KernelCompiler.lower = _lower
    KernelCompiler._helion_mlir_trace_mode_patched = True
