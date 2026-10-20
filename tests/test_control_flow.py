"""Device control flow: ``if`` (``scf.if``) and ``while`` (``scf.while``), and the
scalar arithmetic and comparisons their conditions use."""

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


@_kernel(8)
def if_value_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        if tile.begin > 0 and tile.id < 3:
            v = v * 2.0
        out[tile] = v
    return out


@_kernel(8)
def if_else_value_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        if tile.id % 2 == 0:
            w = v + 1.0
        else:
            w = v - 1.0
            v = v * 3.0
        out[tile] = v + w
    return out


@_kernel(8, 8)
def if_in_reduction_loop_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            if tn.begin != 8:
                acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@_kernel(8)
def scalar_tensor_condition_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        if v.sum() > 0:
            out[tile] = v
    return out


@_kernel(8)
def tensor_condition_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        if v > 0:
            out[tile] = v
    return out


@_kernel(8)
def while_doubling_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        total = v.sum()
        while total < 100.0:
            v = v * 2.0
            total = v.sum()
            out[tile] = v
    return out


@_kernel(8)
def while_invariant_step_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        step = y[tile]
        total = v.sum()
        while total < 100.0:
            v = v + step
            total = v.sum()
        out[tile] = v
    return out


def _if_value_reference(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[8:24] *= 2.0
    return out


def _if_else_reference(x: torch.Tensor) -> torch.Tensor:
    chunks = []
    for index, chunk in enumerate(x.split(8)):
        if index % 2 == 0:
            chunks.append(chunk + chunk + 1.0)
        else:
            chunks.append(chunk * 3.0 + chunk - 1.0)
    return torch.cat(chunks)


def _skip_second_block_reference(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1) - x[:, 8:16].sum(-1)


def test_if_updates_value() -> None:
    torch.manual_seed(0)
    check_kernel(if_value_kernel, _if_value_reference, [torch.randn(40)])


def test_if_else_with_new_and_updated_values() -> None:
    torch.manual_seed(0)
    check_kernel(if_else_value_kernel, _if_else_reference, [torch.randn(32)])


def test_if_carried_through_loop() -> None:
    torch.manual_seed(0)
    check_kernel(
        if_in_reduction_loop_kernel,
        _skip_second_block_reference,
        [torch.randn(16, 24)],
        atol=1e-4,
        rtol=1e-4,
    )


def test_one_element_tensor_condition() -> None:
    torch.manual_seed(0)

    def reference(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [c if c.sum() > 0 else torch.zeros_like(c) for c in x.split(8)]
        )

    check_kernel(scalar_tensor_condition_kernel, reference, [torch.randn(64)])


def test_tensor_condition_is_rejected() -> None:
    with pytest.raises(UnsupportedOperationError, match="tensor of more than one"):
        generate_mlir(tensor_condition_kernel, [torch.randn(16)])


def test_data_dependent_while() -> None:
    def reference(x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for chunk in x.split(8):
            while chunk.sum() < 100.0:
                chunk = chunk * 2.0
            chunks.append(chunk)
        return torch.cat(chunks)

    torch.manual_seed(0)
    check_kernel(while_doubling_kernel, reference, [torch.rand(24) + 0.1])


def test_while_with_invariant_input() -> None:
    def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        chunks = []
        for v, step in zip(x.split(8), y.split(8), strict=True):
            while v.sum() < 100.0:
                v = v + step
            chunks.append(v)
        return torch.cat(chunks)

    torch.manual_seed(0)
    check_kernel(
        while_invariant_step_kernel,
        reference,
        [torch.rand(24) + 0.1, torch.rand(24) + 0.5],
    )
