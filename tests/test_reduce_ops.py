"""Numerical tests for reduction & broadcast ATen ops routed through the
generic torch-mlir helper path."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from tests.harness import execute_module as _execute

from helion_mlir_backend import generate_mlir


def test_sum_reduction():
    @helion.kernel(static_shapes=True)
    def sum_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].sum(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(sum_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="sum_kernel")
    torch.testing.assert_close(actual, x.sum(dim=1))


def test_min_reduction():
    @helion.kernel(static_shapes=True)
    def min_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].amin(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(min_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="min_kernel")
    torch.testing.assert_close(actual, x.amin(dim=1))


def test_max_reduction():
    @helion.kernel(static_shapes=True)
    def max_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].amax(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(max_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="max_kernel")
    torch.testing.assert_close(actual, x.amax(dim=1))


def test_argmax_reduction():
    @helion.kernel(static_shapes=True)
    def argmax_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=torch.int64, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].argmax(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(argmax_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="argmax_kernel")
    torch.testing.assert_close(actual, x.argmax(dim=1))


def test_rank_mismatch_broadcast_add():
    """A 1-D bias added to a 2-D accumulator (rank-expanding broadcast).

    Regression test for a real bug found while auditing examples/: the ATen
    helper pre-build path used to normalize all operands of a broadcasting op
    to a common *rank-expanded* shape purely for helper-signature purposes,
    without ever materializing that expansion in the actual call-site MLIR
    value -- causing every rank-mismatched broadcast (e.g. matmul + 1-D bias)
    to fail with a helper signature mismatch. Fixed by restricting that
    same-shape normalization to same-rank operands only.
    """

    @helion.kernel(static_shapes=True)
    def matmul_add(
        x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc + bias[tile_n]
        return out

    x = torch.randn(64, 128)
    y = torch.randn(128, 96)
    bias = torch.randn(96)
    module = generate_mlir(matmul_add, [x, y, bias])
    actual = _execute(module, x, y, bias, kernel_name="matmul_add")
    torch.testing.assert_close(actual, x @ y + bias, atol=1e-3, rtol=1e-3)


def test_multi_output_kernel_within_one_phase():
    """A kernel returning two tensors, both written by the same hl.tile() loop.

    Every tensor a loop writes is threaded through it as an SSA value and
    returned from the phase function (`lowering/loops.py`).
    """

    @helion.kernel(static_shapes=True)
    def two_outputs(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = x[tm, tn] + y[tm, tn]
            out2[tm, tn] = x[tm, tn] - y[tm, tn]
        return out1, out2

    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    module = generate_mlir(two_outputs, [x, y])
    actual = _execute(module, x, y, kernel_name="two_outputs")
    assert isinstance(actual, list) and len(actual) == 2
    torch.testing.assert_close(actual[0], x + y)
    torch.testing.assert_close(actual[1], x - y)


def test_multi_output_kernel_sharing_a_reduction_accumulator():
    """Two outputs derived from the same nested-reduction accumulator.

    Each store (`out1[tm, tn] = acc`, `out2[tm, tn] = acc * 2.0`) uses the
    already-resolved final accumulator value, so both go through the normal
    terminal-store path (not the synthetic-accumulator flush path) and are
    routed to their own `shared_outs` entry independently.
    """

    @helion.kernel(static_shapes=True)
    def two_outputs_with_reduction(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = x.shape
        k2, n = y.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out1[tm, tn] = acc
            out2[tm, tn] = acc * 2.0
        return out1, out2

    x = torch.randn(16, 32)
    y = torch.randn(32, 16)
    module = generate_mlir(two_outputs_with_reduction, [x, y])
    actual = _execute(module, x, y, kernel_name="two_outputs_with_reduction")
    expected = x @ y
    torch.testing.assert_close(actual[0], expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual[1], expected * 2.0, atol=1e-3, rtol=1e-3)


def test_independent_root_loops_with_different_grids():
    """Two independent top-level loops (no hl.barrier) with different shapes.

    Each root loop is its own ``scf.forall``, emitted in sequence.
    """

    @helion.kernel(static_shapes=True)
    def mismatched_outputs(x: torch.Tensor):
        m, n = x.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m,), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = x[tm, tn]
        for tm in hl.tile(m):
            out2[tm] = x[tm, :].sum(-1)
        return out1, out2

    x = torch.randn(16, 16)
    module = generate_mlir(mismatched_outputs, [x])
    assert str(module).count("scf.forall ") == 2
    out1, out2 = _execute(module, x, kernel_name="mismatched_outputs")
    torch.testing.assert_close(out1, x)
    torch.testing.assert_close(out2, x.sum(-1), atol=1e-4, rtol=1e-4)
