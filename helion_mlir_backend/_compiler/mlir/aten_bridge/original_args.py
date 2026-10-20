"""ATen node arguments as they were before Helion's ``strip_unused_inputs``."""

from __future__ import annotations

import torch.fx

ORIGINAL_ARGS = "helion_mlir_original_args"
"""Node meta key: ``(args, kwargs)`` before Helion's ``strip_unused_inputs`` (see inject)."""


def original_args(node: torch.fx.Node) -> tuple[tuple, dict]:
    """The node's arguments with the inputs Helion masked as ``None`` restored.

    Helion's ``strip_unused_inputs`` replaces repeated inputs (``x * x`` becomes
    ``mul(x, None)``) in place; :func:`install_original_args_capture` records the
    arguments just before, and only those ``None`` positions are filled back, so
    later rewrites of the node (e.g. inserted ``_mask_to``) are kept.
    """
    before_args, before_kwargs = node.meta.get(ORIGINAL_ARGS, ((), {}))
    args = tuple(
        _restore(arg, before_args[index] if index < len(before_args) else None)
        for index, arg in enumerate(node.args)
    )
    kwargs = {
        key: _restore(value, before_kwargs.get(key))
        for key, value in node.kwargs.items()
        if not key.startswith("_extra")
    }
    return args, kwargs


def _restore(current: object, before: object) -> object:
    if current is None and isinstance(before, torch.fx.Node):
        return before
    if (
        isinstance(current, (list, tuple))
        and isinstance(before, (list, tuple))
        and len(current) == len(before)
    ):
        return type(current)(map(_restore, current, before))
    return current


def install_original_args_capture() -> None:
    """Record node arguments before Helion's ``strip_unused_inputs`` masks them."""
    from helion._compiler.compile_environment import CompileEnvironment
    import helion._compiler.inductor_lowering as inductor_lowering

    from ..backend import MLIRBackend

    strip = inductor_lowering.strip_unused_inputs
    if getattr(strip, "_helion_mlir_capture", False):
        return

    def strip_unused_inputs(node: torch.fx.Node, *args: object) -> object:
        if isinstance(CompileEnvironment.current().backend, MLIRBackend):
            node.meta.setdefault(ORIGINAL_ARGS, (node.args, node.kwargs))
        return strip(node, *args)

    strip_unused_inputs._helion_mlir_capture = True
    inductor_lowering.strip_unused_inputs = strip_unused_inputs
