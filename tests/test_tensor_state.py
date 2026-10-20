"""Host tensors threaded as SSA state through forall/for (stores and loads)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def two_stores_last_wins(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
        out[tile] = x[tile] * 3.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8, 8))
def nested_store_then_load(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn] = x[tm, tn] + 1.0
        for tn in hl.tile(n):
            out[tm, tn] = out[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8, 8))
def multi_output_nested(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    shifted = torch.empty_like(x)
    row_sum = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            shifted[tm, tn] = x[tm, tn] + 1.0
            acc = acc + x[tm, tn].sum(-1)
        row_sum[tm] = acc
    return shifted, row_sum


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def partial_inplace(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    for tm in hl.tile(m):
        for tn in hl.tile(4, n):
            x[tm, tn] = x[tm, tn] + 1.0
    return x


def _partial_inplace_reference(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 4:] += 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def row_broadcast_per_column_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm, _tn in hl.tile([m, n]):
        out[tm, :] = x[tm, :] * 2.0
    return out


@pytest.mark.parametrize(
    ("kernel", "reference", "make_args"),
    [
        pytest.param(
            two_stores_last_wins,
            lambda x: x * 3.0,
            lambda: [torch.randn(32)],
            id="last_wins",
        ),
        pytest.param(
            nested_store_then_load,
            lambda x: (x + 1.0) * 2.0,
            lambda: [torch.randn(16, 16)],
            id="nested_store_then_load",
        ),
        pytest.param(
            multi_output_nested,
            lambda x: [x + 1.0, x.sum(-1)],
            lambda: [torch.randn(16, 16)],
            id="multi_output_nested",
        ),
        pytest.param(
            partial_inplace,
            _partial_inplace_reference,
            lambda: [torch.randn(16, 12)],
            id="partial_write_keeps_input",
        ),
        pytest.param(
            row_broadcast_per_column_tile,
            lambda x: x * 2.0,
            lambda: [torch.randn(16, 16)],
            id="sequential_fallback",
        ),
    ],
)
def test_tensor_state(kernel, reference, make_args) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        make_args(),
        paths=("direct", "generated"),
        atol=1e-5,
        rtol=1e-5,
    )


def test_unpartitioned_write_runs_the_grid_sequentially() -> None:
    text = str(generate_mlir(row_broadcast_per_column_tile, [torch.randn(16, 16)]))
    assert "scf.forall" not in text
    assert text.count("scf.for ") == 2


def test_full_tile_store_needs_no_insert_after_canonicalize() -> None:
    from helion_mlir_backend._compiler.execution import inline_module

    module = inline_module(generate_mlir(two_stores_last_wins, [torch.randn(32)]))
    text = str(module)
    assert "scf.forall" in text
    assert "tensor.insert_slice" not in text
