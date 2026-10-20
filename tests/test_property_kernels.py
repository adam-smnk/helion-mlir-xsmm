"""Property-based fuzz coverage for nested-tile/grid indexing geometry.

Targets the exact failure class that has twice caused real segfaults in this
backend (block-id/offset resolution across combined tiles, nested reductions,
and grid+tile reordering) by fuzzing shapes and block sizes across the known
structurally-distinct kernel patterns, instead of relying only on a fixed set
of hand-picked shapes.

A combined-2D-tile + nested-reduction-accumulator fuzz test (varying m/n/k
and block sizes for a tiled matmul) found and fixed two real bugs (block-id
collision in codegen.py's upper-bound pre-pass; a missing upper-bound clamp
in build_context.shape_from_nodes) and exposed ragged (non-evenly-divisible)
boundary tiles, which are now padded on load and partially stored.
"""

from __future__ import annotations

import helion
import helion.language as hl
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import pytest
import torch

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

# Derandomized + bounded: deterministic across CI runs, small enough to keep
# per-run compile cost (each example does a full MLIR compile) reasonable.
_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)

_PHASE_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_BLOCK_SIZES = [4, 8, 16, 32]


@pytest.mark.isolated
@given(
    m=st.sampled_from([8, 20, 32, 48]),
    n=st.sampled_from([8, 16, 36, 48]),
    k=st.sampled_from([8, 20, 32]),
    bm=st.sampled_from(_BLOCK_SIZES),
    bn=st.sampled_from(_BLOCK_SIZES),
    bk=st.sampled_from(_BLOCK_SIZES),
)
@_SETTINGS
def test_combined_tile_matmul_random_shapes(
    m: int, n: int, k: int, bm: int, bn: int, bk: int
) -> None:
    """Combined 2D tile + nested reduction across randomized shapes/blocks,
    including ragged ones."""

    @helion.kernel(static_shapes=True)
    def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m_, k_ = x.shape
        k2_, n_ = y.shape
        out = torch.zeros((m_, n_), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m_, n_]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k_):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
        return out

    torch.manual_seed(0)
    x = torch.randn(m, k)
    y = torch.randn(k, n)
    config = helion.Config(block_sizes=[bm, bn, bk])
    module = generate_mlir(mm, [x, y], config=config)
    actual = MLIRBackend().execute_mlir(module, x, y, kernel_name="mm")
    torch.testing.assert_close(actual, x @ y, atol=1e-3, rtol=1e-3)


@given(
    n_panels=st.sampled_from([2, 3, 4]),
    m=st.sampled_from([8, 16, 24]),
    bn=st.sampled_from([4, 8, 16]),
)
@_SETTINGS
def test_reordered_store_random_shapes(n_panels: int, m: int, bn: int) -> None:
    """Grid+tile reordered store across randomized shapes/block sizes.

    This exact pattern (``out[tm, panel, :] = ...``) previously segfaulted
    from an exact-set-vs-subset block-id matching bug; fuzz it broadly.
    """

    @helion.kernel(static_shapes=True)
    def unpack_panels(src: torch.Tensor) -> torch.Tensor:
        n_panels_, m_, bnn = src.shape
        out = torch.empty((m_, n_panels_, bnn), dtype=src.dtype, device=src.device)
        for panel in hl.grid(n_panels_):
            for tm in hl.tile(m_):
                out[tm, panel, :] = src[panel, tm, :]
        return out

    torch.manual_seed(1)
    src = torch.randn(n_panels, m, 8)
    config = helion.Config(block_sizes=[1, bn])
    module = generate_mlir(unpack_panels, [src], config=config)
    actual = MLIRBackend().execute_mlir(module, src, kernel_name="unpack_panels")
    torch.testing.assert_close(actual, src.permute(1, 0, 2).contiguous())


@given(
    m=st.sampled_from([8, 16, 32]),
    n=st.sampled_from([8, 16, 32]),
    block_size=st.sampled_from(_BLOCK_SIZES),
)
@_SETTINGS
def test_multi_phase_multi_output_with_host_interop_random_shapes(
    m: int, n: int, block_size: int
) -> None:
    """Fuzz the direct-call phase driver across all V1 feature boundaries.

    The host-computed scalar is consumed in phase 0; phase 1 reads phase 0's
    output and returns two tensors. Each phase has two tile dimensions, so
    the config deliberately supplies four block-size slots.
    """

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[block_size] * 4),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def multi_phase_multi_output(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, columns = x.shape
        scale = x.mean() * 2.0
        mid = torch.zeros((rows, columns), dtype=torch.float32, device=x.device)
        out = torch.zeros((rows, columns), dtype=torch.float32, device=x.device)
        residual = torch.zeros((rows, columns), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([rows, columns]):
            s = hl.load(scale, [])
            mid[tm, tn] = (x[tm, tn] + y[tm, tn]) * s
        hl.barrier()
        for tm, tn in hl.tile([rows, columns]):
            out[tm, tn] = mid[tm, tn] * 2.0
            residual[tm, tn] = mid[tm, tn] - x[tm, tn]
        return out, residual

    torch.manual_seed(11)
    x = torch.randn(m, n)
    y = torch.randn(m, n)
    actual = multi_phase_multi_output(x, y)
    scale = x.mean() * 2.0
    mid = (x + y) * scale

    assert isinstance(actual, tuple) and len(actual) == 2
    torch.testing.assert_close(actual[0], mid * 2.0)
    torch.testing.assert_close(actual[1], mid - x)


@given(
    m_blocks=st.sampled_from([2, 3]),
    panels=st.sampled_from([2, 3, 4]),
    block_m=st.sampled_from([8, 16]),
    block_n=st.sampled_from([4, 8]),
    k=st.sampled_from([8, 16, 32]),
    tile_m=st.sampled_from([4, 8, 16]),
    tile_k=st.sampled_from([4, 8, 16, 32]),
)
@_PHASE_SETTINGS
def test_multiphase_packed_blocked_matmul_random_shapes(
    m_blocks: int,
    panels: int,
    block_m: int,
    block_n: int,
    k: int,
    tile_m: int,
    tile_k: int,
) -> None:
    """Fuzz phase-local grid IDs plus a nested tiled reduction.

    Phase 0 reorders B into [panel, k, block_n]. Phase 1 uses a distinct
    2-D grid plus nested tm/tk tiles to produce [m_block, panel, row, col].
    Helion can declare the nested tm loop with a stale phase-0 raw block ID;
    the body-derived ID must therefore control synthetic accumulator shape.
    """
    assume(block_m % tile_m == 0)
    assume(k % tile_k == 0)

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[tile_k, tile_m, tile_k]),
    )
    def packed_blocked_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m_blocks_, block_m_, k_ = a.shape
        k2_, panels_, block_n_ = b.shape
        packed_b = torch.empty((panels_, k_, block_n_), dtype=b.dtype, device=b.device)
        out = torch.empty(
            (m_blocks_, panels_, block_m_, block_n_),
            dtype=torch.float32,
            device=a.device,
        )
        for panel in hl.grid(panels_):
            for tile_k_ in hl.tile(k_):
                packed_b[panel, tile_k_, :] = b[tile_k_, panel, :]
        hl.barrier()
        for m_block, panel in hl.grid([m_blocks_, panels_]):
            for tile_m_ in hl.tile(block_m_):
                acc = hl.zeros([tile_m_, block_n_], dtype=torch.float32)
                for tile_k_ in hl.tile(k_):
                    acc = torch.addmm(
                        acc,
                        a[m_block, tile_m_, tile_k_],
                        packed_b[panel, tile_k_, :],
                    )
                out[m_block, panel, tile_m_, :] = acc
        return out

    torch.manual_seed(29)
    a = torch.randn(m_blocks, block_m, k, dtype=torch.bfloat16)
    b = torch.randn(k, panels, block_n, dtype=torch.bfloat16)
    actual = packed_blocked_matmul(a, b)
    expected = torch.einsum("mik,jkn->mjin", a.float(), b.permute(1, 0, 2).float())
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1.0)


@given(
    m=st.sampled_from([8, 16, 32]),
    n=st.sampled_from([8, 16, 32]),
    phase0_m=st.sampled_from([4, 8, 16, 32]),
    phase0_n=st.sampled_from([4, 8, 16, 32]),
    phase1_m=st.sampled_from([4, 8, 16, 32]),
    phase1_n=st.sampled_from([4, 8, 16, 32]),
)
@_PHASE_SETTINGS
def test_multiphase_combined_tiles_random_shapes(
    m: int,
    n: int,
    phase0_m: int,
    phase0_n: int,
    phase1_m: int,
    phase1_n: int,
) -> None:
    """Fuzz combined-tile ID recovery independently in two phases.

    Each phase's `hl.tile([m, n])` can carry a duplicated/reused raw block-ID
    list. The phase-local body metadata must resolve both dimensions without
    accidentally using a same-sized or stale candidate from the other phase.
    """
    assume(m % phase0_m == 0)
    assume(n % phase0_n == 0)
    assume(m % phase1_m == 0)
    assume(n % phase1_n == 0)

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[phase0_m, phase0_n, phase1_m, phase1_n]),
    )
    def two_phase_combined_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        mid = torch.empty((rows, columns), dtype=torch.float32, device=x.device)
        out = torch.empty((rows, columns), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([rows, columns]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([rows, columns]):
            out[tm, tn] = mid[tm, tn] * 3.0 - x[tm, tn]
        return out

    torch.manual_seed(37)
    x = torch.randn(m, n)
    y = torch.randn(m, n)
    actual = two_phase_combined_tiles(x, y)
    torch.testing.assert_close(actual, (x + y) * 3.0 - x)
