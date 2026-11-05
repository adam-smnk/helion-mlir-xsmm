"""Masks of padded lanes in running values, on the CPU target against float64: right
whatever the mask passes, since a running value ignores a partial tile's padding."""

from __future__ import annotations

import pytest
import torch

from tests import xpu_partial_kernels as pk

import helion_mlir_backend  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        "softmax_f32_cols",
        "softmax_f32_both",
        "softmax_running_f32_cols",
        "running_sum_cols",
        "running_mask_on_rows",
        "running_mask_passing_padding",
    ],
)
def test_mask_forms(name: str) -> None:
    torch.manual_seed(0)
    case = {**pk.CASES, **pk.CPU_ONLY}[name]
    inputs = []
    for shape, dtype, kind in case.inputs:
        x = torch.randn(*shape)
        if kind == "neg":
            x = -x.abs() - 1.0
        elif kind == "mask":
            x = (torch.rand(*shape) > 0.5).float() * 2 - 1
        inputs.append(x.to(dtype))
    out = torch.full(case.output[0], float("nan"), dtype=case.output[1])
    got = case.kernel()(*inputs, out).to(torch.float64)
    ref = case.reference(*inputs).to(torch.float64)
    if got.dim() == 1:
        errors = (got - ref).abs() / ref.abs().clamp_min(1e-30)
    else:
        errors = (got - ref).abs().amax(-1) / ref.abs().amax(-1)
    assert errors.max() < 1e-5
