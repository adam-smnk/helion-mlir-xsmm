"""Sibling GEMMs (two matmuls of one A tile in one K loop, e.g. SwiGLU's gate and up
projections) in one grid loop on XPU: each DPAS gets the layout of a lone GEMM."""

from __future__ import annotations

from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import lower_xpu
from tests.harness import xpu_tensor

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler import xpu_transforms
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

BF16 = torch.bfloat16
M, N, K = 64, 512, 256
CONFIG = {"block_sizes": [64, 256, 32], "xegpu_sg_tiles": [[8, 64]]}


def _kernel(fn: object, **config: object) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(**config)
    )


def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            g = torch.addmm(g, a, gate[tn, tk].T)
            u = torch.addmm(u, a, up[tn, tk].T)
        out[tm, tn] = (g / (1.0 + torch.exp(-g)) * u).to(x.dtype)
    return out


def other_a_tiles(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            g = torch.addmm(g, x[tm, tk], gate[tn, tk].T)
            u = torch.addmm(u, y[tm, tk], up[tn, tk].T)
        out[tm, tn] = (g + u).to(x.dtype)
    return out


def separate_k_loops(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            g = torch.addmm(g, x[tm, tk], gate[tn, tk].T)
        for tk in hl.tile(k):
            u = torch.addmm(u, x[tm, tk], up[tn, tk].T)
        out[tm, tn] = (g + u).to(x.dtype)
    return out


def three_gemms(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = w1.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        a1 = hl.zeros([tm, tn], dtype=torch.float32)
        a2 = hl.zeros([tm, tn], dtype=torch.float32)
        a3 = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            a1 = torch.addmm(a1, a, w1[tn, tk].T)
            a2 = torch.addmm(a2, a, w2[tn, tk].T)
            a3 = torch.addmm(a3, a, w3[tn, tk].T)
        out[tm, tn] = (a1 + a2 + a3).to(x.dtype)
    return out


def computed_a(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = (x[tm, tk].to(torch.float32) * 2.0).to(x.dtype)
            g = torch.addmm(g, a, gate[tn, tk].T)
            u = torch.addmm(u, a, up[tn, tk].T)
        out[tm, tn] = (g + u).to(x.dtype)
    return out


def one_accumulator(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            g = torch.addmm(g, a, gate[tn, tk].T)
            g = torch.addmm(g, a, up[tn, tk].T)
        out[tm, tn] = g.to(x.dtype)
    return out


def crossed_accumulators(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            h = torch.addmm(g, a, gate[tn, tk].T)
            g = torch.addmm(h, a, up[tn, tk].T)
            u = u + h
        out[tm, tn] = (g + u).to(x.dtype)
    return out


def shared_b(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            b = w[tn, tk].T
            g = torch.addmm(g, a, b)
            u = torch.addmm(u, a, b)
        out[tm, tn] = (g + u).to(x.dtype)
    return out


def with_row_reduction(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = gate.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=torch.float32)
        u = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            a = x[tm, tk]
            g = torch.addmm(g, a, gate[tn, tk].T)
            u = torch.addmm(u, a, up[tn, tk].T)
        out[tm, tn] = (g * u.sum(-1, keepdim=True)).to(x.dtype)
    return out


def _bf(*shape: int) -> torch.Tensor:
    return xpu_tensor(*shape, dtype=BF16)


@pytest.mark.isolated
def test_both_dpas_anchored() -> None:
    ir = lower_xpu(
        _kernel(swiglu, **CONFIG), [_bf(M, K), _bf(N, K), _bf(N, K)], stop_at="xegpu-wg"
    )
    lines = [line for line in ir.splitlines() if "xegpu.dpas" in line]
    assert len(lines) == 2
    assert all("sg_layout = [8, 4]" in line and "sg_data" in line for line in lines)


@pytest.mark.isolated
def test_lowers_to_xevm_without_xegpu() -> None:
    ir = lower_xpu(
        _kernel(swiglu, **CONFIG), [_bf(M, K), _bf(N, K), _bf(N, K)], stop_at="xevm"
    )
    assert "xegpu." not in ir


@pytest.mark.isolated
def test_other_a_tiles_are_refused() -> None:
    args = [_bf(M, K), _bf(M, K), _bf(N, K), _bf(N, K)]
    with pytest.raises(UnsupportedOperationError, match="two matmuls outside"):
        lower_xpu(_kernel(other_a_tiles, **CONFIG), args)


@pytest.mark.isolated
def test_separate_k_loops_are_refused() -> None:
    config = {**CONFIG, "block_sizes": [64, 256, 32, 32]}
    with pytest.raises(UnsupportedOperationError, match="two matmuls outside"):
        lower_xpu(
            _kernel(separate_k_loops, **config), [_bf(M, K), _bf(N, K), _bf(N, K)]
        )


@pytest.mark.isolated
def test_three_are_refused() -> None:
    with pytest.raises(UnsupportedOperationError, match="several matmuls"):
        lower_xpu(_kernel(three_gemms, **CONFIG), [_bf(M, K), *[_bf(N, K)] * 3])


@pytest.mark.isolated
@pytest.mark.parametrize("fn", [computed_a, one_accumulator, crossed_accumulators])
def test_not_two_gemms_of_one_tile_are_refused(fn: object) -> None:
    """Exact pattern: a computed A crashed (``visitDpasOp``), a result as the other's
    accumulator gave a generic error or lowered, against the docstring."""
    with pytest.raises(UnsupportedOperationError, match="two matmuls outside"):
        lower_xpu(_kernel(fn, **CONFIG), [_bf(M, K), _bf(N, K), _bf(N, K)])


@pytest.mark.isolated
def test_shared_b_is_refused() -> None:
    """The same GEMM twice: one is enough, and siblings differ in B."""
    with pytest.raises(UnsupportedOperationError, match="two matmuls outside"):
        lower_xpu(_kernel(shared_b, **CONFIG), [_bf(M, K), _bf(N, K)])


@pytest.mark.isolated
def test_row_reduction_next_to_siblings_is_refused() -> None:
    with pytest.raises(UnsupportedOperationError, match="and a row reduction"):
        lower_xpu(
            _kernel(with_row_reduction, **CONFIG), [_bf(M, K), _bf(N, K), _bf(N, K)]
        )


@pytest.mark.isolated
def test_unanchored_dpas_is_refused() -> None:
    """A sibling annotation that leaves a DPAS out must not lower silently: the
    second call's depth 0 leaves it unanchored, and the kernel lowered with no error."""
    original = xpu_transforms.annotate_gemm
    calls = []

    def second_fails(scope: object, dpas: object, **options: object) -> object:
        calls.append(dpas)
        if len(calls) == 2:
            options["force_prefetch_a_nb"] = 0
        return original(scope, dpas, **options)

    with (
        mock.patch.object(xpu_transforms, "annotate_gemm", second_fails),
        pytest.raises(UnsupportedOperationError, match="without a layout"),
    ):
        lower_xpu(_kernel(swiglu, **CONFIG), [_bf(M, K), _bf(N, K), _bf(N, K)])
