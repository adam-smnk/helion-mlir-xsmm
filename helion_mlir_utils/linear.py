"""Linear helpers with optional deployment-style weight prepacking."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple

import torch

from .matmul import identity_epilogue
from .matmul import matmul
from .matmul import matmul_prepacked_b
from .matmul import pack_b_blocked_t
from .matmul import pack_b_vnni_t
from .matmul import vnni_panel

if TYPE_CHECKING:
    from torch import Tensor
    from torch import nn

CACHE_PREPACKED_WEIGHTS_ENV = "HELION_MLIR_CACHE_PREPACKED_WEIGHTS"


def cache_prepacked_weights() -> bool:
    """Whether constant weights are packed once and reused (the default);
    ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0`` packs them on every call."""
    value = os.environ.get(CACHE_PREPACKED_WEIGHTS_ENV, "").strip().lower()
    return value not in ("0", "false", "no")


class LinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    out_features: int


class AffineLinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    post_scale: Tensor
    post_bias: Tensor | None
    out_features: int


def _parameter_key(parameter: Tensor | None) -> tuple | None:
    if parameter is None:
        return None
    return (
        parameter.data_ptr(),
        parameter._version,
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


def _combine_biases(biases: tuple[Tensor, ...], x: Tensor) -> Tensor | None:
    if not biases:
        return None
    bias = biases[0].to(dtype=x.dtype, device=x.device)
    for extra_bias in biases[1:]:
        bias = bias + extra_bias.to(dtype=x.dtype, device=x.device)
    return bias


def _pack_weight(weight: Tensor, rows: int) -> Tensor:
    """``[N, K]`` weight packed for :func:`matmul_prepacked_b` of ``rows`` rows:
    AMX's VNNI panels for bf16 with an even K, else 32x32 blocks."""
    if weight.dtype == torch.bfloat16 and not weight.shape[1] % 2:
        return pack_b_vnni_t(weight, vnni_panel(rows, int(weight.shape[0])))
    return pack_b_blocked_t(weight)


def linear(
    x: Tensor,
    layer: nn.Linear,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: LinearCache | None = None,
    biases: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, LinearCache | None]:
    """Run a linear layer, by default with its constant weight packed once and
    reused until the source parameters, input dtype, or device change;
    ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0`` packs it on every call.
    """
    source_biases = (
        (() if layer.bias is None else (layer.bias,)) if biases is None else biases
    )
    if not cache_prepacked_weights():
        weight = layer.weight.to(dtype=x.dtype, device=x.device)
        return (
            matmul(
                x,
                weight,
                trans_b=True,
                bias=_combine_biases(source_biases, x),
                epilogue=epilogue,
            ),
            None,
        )

    key = (
        _parameter_key(layer.weight),
        tuple(_parameter_key(bias) for bias in source_biases),
        x.dtype,
        x.device,
        x.shape[0],
    )
    if cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = _combine_biases(source_biases, x)
        cache = LinearCache(
            key,
            _pack_weight(weight, int(x.shape[0])),
            bias,
            int(layer.out_features),
        )

    return (
        matmul_prepacked_b(
            x,
            cache.packed_weight,
            n=cache.out_features,
            bias=cache.bias,
            epilogue=epilogue,
        ),
        cache,
    )


def linear_affine(
    x: Tensor,
    layer: nn.Linear,
    post_scale: Tensor,
    post_bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: AffineLinearCache | None = None,
) -> tuple[Tensor, AffineLinearCache | None]:
    """Run ``epilogue((linear(x)) * post_scale + post_bias)``; ``post_scale`` and
    ``post_bias`` are ``[1]`` or ``[out_features]``. Cached like :func:`linear`."""
    use_cache = cache_prepacked_weights()
    if x.dtype == torch.bfloat16 and not use_cache:
        bias = None if layer.bias is None else layer.bias.to(dtype=x.dtype)
        result = matmul(
            x,
            layer.weight.to(dtype=x.dtype, device=x.device),
            trans_b=True,
            bias=bias,
            epilogue=epilogue,
            scale=post_scale,
            shift=post_bias,
        )
        return result, None
    key = (
        _parameter_key(layer.weight),
        _parameter_key(layer.bias),
        _parameter_key(post_scale),
        _parameter_key(post_bias),
        x.dtype,
        x.device,
        x.shape[0],
    )
    if not use_cache or cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = (
            None
            if layer.bias is None
            else layer.bias.detach().to(dtype=x.dtype, device=x.device)
        )
        cache = AffineLinearCache(
            key,
            _pack_weight(weight, int(x.shape[0])),
            bias,
            post_scale.detach().to(dtype=x.dtype, device=x.device),
            None
            if post_bias is None
            else post_bias.detach().to(dtype=x.dtype, device=x.device),
            int(layer.out_features),
        )

    result = matmul_prepacked_b(
        x,
        cache.packed_weight,
        cache.out_features,
        cache.bias,
        epilogue,
        cache.post_scale,
        cache.post_bias,
    )
    return result, cache if use_cache else None
