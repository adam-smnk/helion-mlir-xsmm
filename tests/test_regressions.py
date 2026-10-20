"""Regression probes for kernels the backend once got wrong or rejected.

A new gap gets a probe marked
``[pytest.mark.xfail(strict=True, reason="..."), pytest.mark.isolated]``
(isolated because gaps can crash natively).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import run_direct

if TYPE_CHECKING:
    from collections.abc import Callable


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16))
def ragged_elementwise_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def ragged_k_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 16))
def ragged_row_max_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n):
            acc = torch.maximum(acc, x[tm, tn].amax(dim=-1))
        out[tm] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def row_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        exp = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        out[tm, :] = exp / exp.sum(dim=-1, keepdim=True)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def where_relu_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        value = x[tile]
        out[tile] = torch.where(value > 0, value, torch.zeros_like(value))
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def acc_plus_addmm_kernel(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = acc + torch.addmm(bias[tm, tn], x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def store_then_load_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
        out[tile] = out[tile] * 2.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def nested_and_outer_store_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn] = x[tm, tn] + 1.0
        out[tm, 0] = x[tm, 0] * 0.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_if_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        if tile.begin == 0:
            out[tile] = x[tile] + 1.0
        else:
            out[tile] = x[tile]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def scale_kernel(x: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * alpha
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def partial_write_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, 7.0)
    for tile in hl.tile(x.size(0) // 2):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def inplace_kernel(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1.0
    return x


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def out_param_kernel(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_one_2d_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def dot_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16))
def broadcast_store_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), y.size(0)], dtype=y.dtype, device=x.device)
    for tm, tn in hl.tile(out.size()):
        out[tm, tn] = x[tm, None]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def block_sums_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.zeros((m, n // block_n), dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            out[tm, tn.id] = x[tm, tn].sum(-1)
    return out


def _nested_and_outer_store_reference(x: torch.Tensor) -> torch.Tensor:
    out = x + 1.0
    out[:, 0] = 0.0
    return out


def _tile_if_reference(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:8] += 1.0
    return out


def _partial_write_reference(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, 7.0)
    half = x.shape[0] // 2
    out[:half] = x[:half]
    return out


_CASES: dict[
    str, tuple[Callable[..., object], Callable[..., object], Callable[[], list]]
] = {
    "ragged_elementwise": (
        ragged_elementwise_kernel,
        lambda x: x * 2.0,
        lambda: [torch.randn(100)],
    ),
    "ragged_k_matmul": (
        ragged_k_matmul_kernel,
        torch.matmul,
        lambda: [torch.randn(32, 20), torch.randn(20, 32)],
    ),
    "ragged_row_max": (
        ragged_row_max_kernel,
        lambda x: x.amax(dim=-1),
        lambda: [torch.randn(8, 20)],
    ),
    "row_softmax": (
        row_softmax_kernel,
        lambda x: torch.softmax(x, dim=-1),
        lambda: [torch.randn(16, 32)],
    ),
    "where_relu": (where_relu_kernel, torch.relu, lambda: [torch.randn(16)]),
    "acc_plus_addmm": (
        acc_plus_addmm_kernel,
        lambda x, y, b: b + x @ y,
        lambda: [torch.randn(16, 8), torch.randn(8, 16), torch.randn(16, 16)],
    ),
    "store_then_load": (
        store_then_load_kernel,
        lambda x: (x + 1.0) * 2.0,
        lambda: [torch.randn(16)],
    ),
    "nested_and_outer_store": (
        nested_and_outer_store_kernel,
        _nested_and_outer_store_reference,
        lambda: [torch.randn(16, 16)],
    ),
    "tile_if": (tile_if_kernel, _tile_if_reference, lambda: [torch.randn(16)]),
    "partial_write": (
        partial_write_kernel,
        _partial_write_reference,
        lambda: [torch.randn(16)],
    ),
    "non_contiguous_input": (
        add_one_2d_kernel,
        lambda x: x + 1.0,
        lambda: [torch.randn(16, 16).t()],
    ),
    "hl_dot": (
        dot_matmul_kernel,
        torch.matmul,
        lambda: [torch.randn(32, 16), torch.randn(16, 32)],
    ),
    "broadcast_store": (
        broadcast_store_kernel,
        lambda x, y: x[:, None].expand(x.size(0), y.size(0)).to(y.dtype),
        lambda: [torch.randn(32), torch.randn(40, dtype=torch.bfloat16)],
    ),
    "block_size_host_shape": (
        block_sums_kernel,
        lambda x: x.view(x.size(0), -1, 8).sum(-1),
        lambda: [torch.randn(8, 32)],
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        "ragged_elementwise",
        "ragged_k_matmul",
        "ragged_row_max",
        "row_softmax",
        "where_relu",
        "acc_plus_addmm",
        "store_then_load",
        "nested_and_outer_store",
        "tile_if",
        "partial_write",
        "non_contiguous_input",
        "hl_dot",
        "broadcast_store",
        "block_size_host_shape",
    ],
)
def test_review_probe(case: str) -> None:
    kernel, reference, make_args = _CASES[case]
    torch.manual_seed(0)
    check_kernel(kernel, reference, make_args(), atol=1e-4, rtol=1e-4)


def test_scalar_argument_changes_between_calls() -> None:
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(scale_kernel, [x, 2.0]), x * 2.0)
    torch.testing.assert_close(run_direct(scale_kernel, [x, 3.0]), x * 3.0)


def test_inplace_update_of_input() -> None:
    x = torch.randn(16)
    expected = x + 1.0
    result = run_direct(inplace_kernel, [x])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x, expected)


def test_out_parameter_is_written() -> None:
    x = torch.randn(16)
    out = torch.zeros(16)
    run_direct(out_param_kernel, [x, out])
    torch.testing.assert_close(out, x + 1.0)
