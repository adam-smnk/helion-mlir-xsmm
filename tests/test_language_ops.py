"""Helion language ops lowered directly: ``torch.tensor`` constants, ``hl.reduce``,
``hl.associative_scan``/``torch.cumsum`` (also as a tensor method) and
``hl.split``/``hl.join``."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


def max_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def keep_first(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Non-commutative: exposes the combine order of a reverse scan."""
    return a * 0.0 + a + b * 0.5


def add_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return b + a


def min_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def or_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(a, b)


def mul_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a * b


@_kernel(8)
def constant_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * torch.tensor(2.0)
    return out


@_kernel(4)
def reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(max_combine, x[tile, :], dim=1)
    return out


@_kernel(4)
def sum_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(add_combine, x[tile, :], dim=1)
    return out


@_kernel(4)
def min_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(min_combine, x[tile, :], dim=1)
    return out


@_kernel(4)
def any_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(or_combine, x[tile, :], dim=1)
    return out


@_kernel(4)
def prod_keep_dims_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = hl.reduce(mul_combine, x[tile, :], dim=1, keep_dims=True)
    return out


@_kernel(4)
def fold_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(keep_first, x[tile, :], dim=1)
    return out


@_kernel(4)
def cumsum_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = torch.cumsum(x[tile, :], dim=1)
    return out


@_kernel(4)
def scan_methods_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :].cumsum(-1) + x[tile, :].cumprod(dim=1)
    return out


@_kernel(4)
def reverse_scan_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = hl.associative_scan(keep_first, x[tile, :], dim=1, reverse=True)
    return out


@_kernel(4)
def swap_pairs_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a, b = hl.split(x[tile, :].reshape([tile, 4, 2]))
        out[tile, :] = hl.join(b, a).reshape([tile, 8])
    return out


@_kernel(4)
def atomic_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [0], x[tile].sum())
    return out


def _reverse_scan_reference(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    acc = x[:, -1]
    out[:, -1] = acc
    for i in range(x.size(1) - 2, -1, -1):
        acc = keep_first(x[:, i], acc)
        out[:, i] = acc
    return out


def test_constant_tensor() -> None:
    torch.manual_seed(0)
    check_kernel(constant_kernel, lambda x: x * 2.0, [torch.randn(16)])


@pytest.mark.parametrize(
    ("kernel", "reference", "x"),
    [
        (reduce_kernel, lambda x: x.amax(dim=1), torch.randn(8, 16)),
        (sum_reduce_kernel, lambda x: x.sum(dim=1), torch.randn(8, 16)),
        (
            min_reduce_kernel,
            lambda x: x.amin(dim=1),
            torch.randint(-1000, 1000, (8, 16), dtype=torch.int32),
        ),
        (any_reduce_kernel, lambda x: x.any(dim=1), torch.rand(8, 16) > 0.9),
        (
            prod_keep_dims_kernel,
            lambda x: x.prod(dim=1, keepdim=True),
            torch.rand(8, 16) + 0.5,
        ),
    ],
    ids=["maximum", "add", "minimum_int", "logical_or", "mul_keep_dims"],
)
def test_reduce_with_known_combiner_is_linalg_reduce(
    kernel: object, reference: object, x: torch.Tensor
) -> None:
    text = str(generate_mlir(kernel, [x]))
    assert "linalg.reduce" in text
    assert "scf.for " not in text
    check_kernel(kernel, reference, [x])


def test_reduce_with_other_combiner_is_a_loop() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    assert "scf.for " in str(generate_mlir(fold_reduce_kernel, [x]))
    check_kernel(fold_reduce_kernel, lambda x: x[:, 0] + 0.5 * x[:, 1:].sum(dim=1), [x])


def test_cumsum() -> None:
    torch.manual_seed(0)
    check_kernel(
        cumsum_kernel,
        lambda x: torch.cumsum(x, dim=1),
        [torch.randn(8, 16)],
        atol=1e-5,
        rtol=1e-5,
    )


def test_scan_methods() -> None:
    torch.manual_seed(0)
    check_kernel(
        scan_methods_kernel,
        lambda x: x.cumsum(-1) + x.cumprod(dim=1),
        [torch.rand(8, 16) + 0.5],
        atol=1e-4,
        rtol=1e-4,
    )


def test_reverse_scan() -> None:
    torch.manual_seed(0)
    check_kernel(reverse_scan_kernel, _reverse_scan_reference, [torch.randn(8, 6)])


def test_split_and_join() -> None:
    torch.manual_seed(0)
    check_kernel(
        swap_pairs_kernel,
        lambda x: x.view(8, 4, 2).flip(-1).reshape(8, 8),
        [torch.randn(8, 8)],
    )


def test_atomics_are_rejected_with_a_reason() -> None:
    with pytest.raises(UnsupportedOperationError, match="atomics need memory"):
        generate_mlir(atomic_kernel, [torch.randn(16)])
