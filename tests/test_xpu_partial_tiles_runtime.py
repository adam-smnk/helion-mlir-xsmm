"""Partial tiles on the device, against float64 row by row. Inputs and outputs sit
in buffers with guards of their own sentinels (NaN past inputs, a number past
outputs: a NaN read past an input and stored past an output must show), outputs
are NaN-poisoned, and each kernel also runs after an unrelated one (registers)."""

from __future__ import annotations

import math

import helion
import pytest
import torch

from tests import xpu_one_tile_kernels as ok
from tests import xpu_partial_kernels as pk

pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated]

GUARD = 4096
"""Elements each side; a multiple of 64 bytes for 16- and 32-bit elements."""
OUT_GUARD = 1024.0  # Exact in bf16 and f32.
BOUND = {torch.bfloat16: 2**-7, torch.float32: 1e-5}
# A row sum of bf16 values rounds in bf16, stored in f32.
DEVICE = "xpu"


def _input(shape: tuple[int, ...], kind: str) -> torch.Tensor:
    x = torch.randn(*shape)
    if kind == "neg":
        return -x.abs() - 1.0
    if kind == "mask":
        return (torch.rand(*shape) > 0.5).float() * 2 - 1
    return x


def _buffer(
    shape: tuple[int, ...], dtype: torch.dtype, guard: float
) -> tuple[torch.Tensor, torch.Tensor]:
    n = math.prod(shape)
    base = torch.full((GUARD + n + GUARD,), guard, dtype=dtype, device=DEVICE)
    return base, base[GUARD : GUARD + n].view(*shape)


def _touched(base: torch.Tensor, n: int, guard: float) -> int:
    b = base.detach().to("cpu").float()
    rest = torch.cat([b[:GUARD], b[GUARD + n :]])
    changed = ~torch.isnan(rest) if math.isnan(guard) else rest != guard
    return int(changed.sum())


def _unrelated() -> None:
    x = torch.full((256, 512), 3.25, device=DEVICE)
    o = torch.empty(256, 512, device=DEVICE)
    helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 128]),
    )(pk.scale)(x, o)
    torch.xpu.synchronize()


def check_case(case: pk.Case, **config: object) -> None:
    torch.manual_seed(0)
    inputs, guards = [], []
    for shape, dtype, kind in case.inputs:
        base, view = _buffer(shape, dtype, float("nan"))
        view.copy_(_input(shape, kind).to(dtype).to(DEVICE))
        inputs.append(view)
        guards.append((base, math.prod(shape), float("nan")))
    copies = [x.detach().to("cpu").clone() for x in inputs]
    shape, dtype = case.output
    base, out = _buffer(shape, dtype, OUT_GUARD)
    out.fill_(float("nan"))
    guards.append((base, math.prod(shape), OUT_GUARD))
    kernel = case.kernel(**config)
    kernel(*inputs, out)
    torch.xpu.synchronize()
    for repeat in (False, True):
        if repeat:
            _unrelated()
            out.fill_(float("nan"))
            kernel(*inputs, out)
            torch.xpu.synchronize()
        for buffer, n, guard in guards:
            assert _touched(buffer, n, guard) == 0, "a guard changed"
        got = out.detach().to("cpu").to(torch.float64)
        ref = case.reference(*copies).to(torch.float64)
        assert not torch.isnan(got).any()
        bound = case.bound or BOUND[copies[0].dtype]
        if got.dim() == 1:
            errors = (got - ref).abs() / ref.abs().clamp_min(1e-30)
        else:
            rows = (got - ref).abs().flatten(0, -2).amax(-1)
            errors = rows / ref.abs().flatten(0, -2).amax(-1).clamp_min(1e-30)
        assert errors.max() <= bound, f"max row error {errors.max():.3g}"
    for x, copy in zip(inputs, copies, strict=True):
        assert torch.equal(x.to("cpu"), copy)


@pytest.mark.parametrize("vectorizer", ["cmp", "off"])
@pytest.mark.parametrize("name", list(pk.CASES))
def test_partial_tile(name: str, vectorizer: str) -> None:
    check_case(pk.CASES[name], igc_vectorizer=[vectorizer])


@pytest.mark.parametrize("vectorizer", ["cmp", "off"])
@pytest.mark.parametrize("name", list(ok.CASES))
def test_one_tile(name: str, vectorizer: str) -> None:
    check_case(ok.CASES[name], igc_vectorizer=[vectorizer])
