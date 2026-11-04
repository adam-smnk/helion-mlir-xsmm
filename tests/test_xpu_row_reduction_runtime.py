"""One-row reductions split across subgroups, on the device: per row against float64,
outputs NaN-poisoned, with NaN, inf and zero rows, fresh inputs, and after other kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import pytest
import torch

from tests import xpu_row_reduction_kernels as rows

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated]


def _l1(x: torch.Tensor) -> torch.Tensor:
    return x / x.abs().mean(1, keepdim=True)


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.square().sum(1, keepdim=True).sqrt()


def _standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(1, keepdim=True)
    return (x - mean) / (x - mean).square().mean(1, keepdim=True).sqrt()


def _minus_maximum(x: torch.Tensor) -> torch.Tensor:
    return x - x.amax(1, keepdim=True)


def _minus_minimum(x: torch.Tensor) -> torch.Tensor:
    return x - x.amin(1, keepdim=True)


def _over_sum(x: torch.Tensor) -> torch.Tensor:
    return x / x.abs().sum(1, keepdim=True)


CASES: dict[str, tuple[Callable[..., object], Callable[..., object]]] = {
    "l1": (rows.l1_two_passes, _l1),
    "l2": (rows.l2_two_passes, _l2),
    "standardize": (rows.standardize, _standardize),
    "maximum": (rows.minus_row_maximum, _minus_maximum),
    "minimum": (rows.minus_row_minimum, _minus_minimum),
}
"""What the kernels compute. Columns and block sizes come from ``SPLITS``."""
SPLITS = {
    "s32": (2048, 1024),
    "s16": (2048, 512),
    "s16_narrow": (2048, 256),
    "s1": (1024, 128),
}
_TOL = 1e-5
"""Sums over 2048 f32 values round to about 1e-7 of the row's largest output; the output
divides by them."""


@pytest.fixture(autouse=True)
def _full_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


def _kernel(fn: Callable[..., object], block_n: int | None) -> helion.Kernel:
    blocks = [1] if block_n is None else [block_n, 1]
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )


_OVERFLOWS = 7
"""The row of :func:`_inputs` whose sums overflow."""


def _inputs(m: int, n: int, seed: int) -> torch.Tensor:
    """Random rows, and rows of what a reduction across subgroups could lose: each special
    value in the first subgroup's columns and in the last's, whole rows of them, and rows
    whose sums overflow."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=generator, dtype=torch.float32)
    x[1, 0] = float("nan")
    x[2, n - 1] = float("nan")
    x[3, 0] = float("inf")
    x[4, n - 1] = float("-inf")
    x[5, :] = 0.0
    x[6, :] = float("nan")
    x[_OVERFLOWS, :] = 3e38
    x[8, n // 2] = float("inf")
    x[8, n // 2 + 1] = float("-inf")
    return x


def _check_rows(
    out: torch.Tensor,
    x: torch.Tensor,
    ref: Callable[..., object],
    overflows: bool = True,
) -> None:
    expected = ref(x.double()).float().double()
    if overflows:
        # Its sums don't fit f32, in float64 they do.
        expected[_OVERFLOWS] = ref(x[_OVERFLOWS : _OVERFLOWS + 1]).double()[0]
    got = out.cpu().double()
    assert torch.equal(got.isnan(), expected.isnan()), (
        got.isnan().sum(1).nonzero().flatten().tolist(),
        expected.isnan().sum(1).nonzero().flatten().tolist(),
    )
    assert torch.equal(got.isinf(), expected.isinf())
    finite = got.isfinite()
    largest = expected.where(finite, 0.0).abs().amax(1, keepdim=True).clamp_min(1e-30)
    errors = ((got - expected).where(finite, 0.0).abs() / largest).amax(1)
    assert errors.max() <= _TOL, (errors.argmax().item(), errors.max().item())


def _poison(shape: torch.Size) -> None:
    block = torch.full(tuple(shape), float("nan"), dtype=torch.float32, device="xpu")
    torch.xpu.synchronize()
    del block


@pytest.mark.parametrize("split", list(SPLITS))
@pytest.mark.parametrize("name", list(CASES))
def test_rows_against_float64(name: str, split: str) -> None:
    fn, ref = CASES[name]
    n, block_n = SPLITS[split]
    kernel = _kernel(fn, block_n)
    for seed in range(3):
        x = _inputs(64, n, seed)
        _poison(x.shape)
        _check_rows(kernel(x.to("xpu")), x, ref)


def test_a_reduction_of_a_load() -> None:
    x = _inputs(64, 1024, 1)
    _poison(x.shape)
    _check_rows(_kernel(rows.direct_sum_of_a_load, None)(x.to("xpu")), x, _over_sum)


def test_bf16_tiles_with_f32_sums() -> None:
    x = torch.randn(64, 2048, generator=torch.Generator().manual_seed(3)).bfloat16()
    out = _kernel(rows.bf16_with_f32_sums, 1024)(x.to("xpu"))
    expected = _l1(x.double()).bfloat16().double()
    # Two roundings to bf16 of nearly the same value, doubled: as test_xpu_runtime.
    errors = (out.cpu().double() - expected).abs().amax(1) / expected.abs().amax(1)
    assert errors.max() <= 2**-7


def test_a_subnormal_row_mean() -> None:
    """``1e-40 * randn``: the row sums are normal, their means subnormal, and the output
    divides by them. Exact regression: gathers of 2 elements per lane in the output pass
    gave inf on the B50."""
    x = (
        1e-40
        * torch.randn(
            64, 2048, generator=torch.Generator().manual_seed(6), dtype=torch.float64
        )
    ).float()
    out = _kernel(rows.l1_two_passes, 1024)(x.to("xpu")).cpu()
    expected = _l1(x.double())
    assert out.isfinite().all()
    # The subnormal mean holds fewer bits: as on torch's own f32 (9e-6).
    errors = (out.double() - expected).abs().amax(1) / expected.abs().amax(1)
    assert errors.max() <= 2e-5


def test_a_non_contiguous_input() -> None:
    x = _inputs(64, 4096, 7)
    view = x[:, ::2].contiguous()
    _check_rows(_kernel(rows.l1_two_passes, 1024)(x.to("xpu")[:, ::2]), view, _l1)


def test_every_row_is_its_own() -> None:
    """Rows of different scales: a partial result of another workgroup's row would show."""
    x = torch.randn(
        256, 2048, generator=torch.Generator().manual_seed(4), dtype=torch.float32
    )
    x *= (2.0 ** (torch.arange(256, dtype=torch.float32) % 32 - 16)).unsqueeze(1)
    _check_rows(_kernel(rows.l1_two_passes, 1024)(x.to("xpu")), x, _l1, overflows=False)


def test_after_other_kernels() -> None:
    """Stale registers and SLM of another kernel: the same bits as the first call."""
    x = _inputs(64, 2048, 5).to("xpu")
    kernel = _kernel(rows.l1_two_passes, 1024)
    first = kernel(x).cpu()
    other = _kernel(rows.minus_row_maximum, 1024)(x * 2.0)
    torch.xpu.synchronize()
    del other
    second = kernel(x).cpu()
    same = first.view(-1).to(torch.float32).nan_to_num(7.0) == second.view(
        -1
    ).nan_to_num(7.0)
    assert same.all()
