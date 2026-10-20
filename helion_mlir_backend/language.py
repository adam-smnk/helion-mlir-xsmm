"""Device-code API of the MLIR backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion.language import _decorators
from helion.runtime.ref_mode import is_in_ref_mode_context
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import mlir.ir as ir


class InlineMLIRHint(exc.BaseWarning):
    """Advice on an ``inline_mlir`` snippet, reported once per snippet."""

    message = "{0}"


def inline_mlir(
    source: str | ir.Module,
    args: Sequence[object],
    output_like: torch.Tensor | Sequence[torch.Tensor],
    *,
    reference: Callable[..., object] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Call a user MLIR function inside a device loop.

    The function is cloned into the kernel's module and inlined into it before
    bufferization, so the lighthouse pipeline lowers it with the rest of the kernel.

    Args:
        source: MLIR text, or an ``mlir.ir.Module``, with ``func.func`` ops only;
            the first one is called. It must be a module-level global (Helion
            kernels cannot capture locals). Its parameters and results are ranked
            tensors or ``index``/integer/float scalars, never memrefs.
        args: The arguments, in order: tiles and other device tensors, tile
            attributes and sizes, runtime scalars, Python numbers.
        output_like: A tensor, or a tuple of tensors, giving the shape and dtype
            of each result.
        reference: A Python function of ``args`` computing the same results, run
            instead of the snippet in Helion's eager ref mode and by other backends.

    Returns:
        The result tensor, or a tuple of them when ``output_like`` is a sequence.
    """
    if not CompileEnvironment.has_current():
        raise exc.NotInsideKernel
    if is_in_ref_mode_context() or CompileEnvironment.current().backend_name != "mlir":
        if reference is None:
            raise exc.InvalidAPIUsage(
                "inline_mlir needs reference= outside the MLIR backend "
                "(other backends and Helion's ref mode)"
            )
        return reference(*args)
    return _inline_mlir(source, list(args), output_like)


@_decorators.api(is_device_only=True)
def _inline_mlir(source: str, args: list[object], output_like: object) -> object:
    raise exc.NotInsideKernel


@_decorators.register_fake(_inline_mlir)
def _(source: str, args: list[object], output_like: object) -> object:
    from ._compiler.mlir import snippets

    outputs = (
        list(output_like) if isinstance(output_like, (tuple, list)) else [output_like]
    )
    signature = snippets.validate(snippets.source_text(source))
    snippets.check_call(signature, args, outputs)
    results = tuple(torch.empty_like(like) for like in outputs)
    return results if isinstance(output_like, (tuple, list)) else results[0]
