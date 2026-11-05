"""The forms of ``xpu_tile_form_kernels`` on the device, at every shape the rule emits: per row
against float64, with NaN-poisoned outputs. A wrong layout gives wrong rows with no diagnostic.
A sample by default; ``HELION_TEST_FULL_SWEEP=1`` for every shape."""

from __future__ import annotations

import helion
import pytest
import torch

from tests import xpu_tile_form_kernels as kernels

# A hang guard, far over the slowest case: cases take seconds, and one ran for over 24
# minutes once, cause unknown.
pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated(timeout=300)]

_BOUND = {torch.bfloat16: 2**-7, torch.float16: 2**-9, torch.float32: 2**-18}
"""Two roundings of nearly the same value, doubled: as ``test_xpu_runtime``."""
_CASES = kernels.default_cases()


@pytest.fixture(autouse=True)
def _full_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_TEST_MARK_STAGES", "1")
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", list(_CASES))
def test_form_rows(name: str) -> None:
    case = _CASES[name]
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    inputs = [
        (torch.randn(*shape, generator=generator) * (1.0 if not i else 0.05)).to(
            case.dtype
        )
        for i, shape in enumerate(case.shapes)
    ]
    ref = case.ref(*(t.double() for t in inputs))
    kernel = helion.kernel(
        case.fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(
            block_sizes=case.blocks, xegpu_layouts=["ir"], **case.config
        ),
    )
    device_inputs = [t.to("xpu") for t in inputs]
    block = torch.full(tuple(ref.shape), float("nan"), dtype=case.dtype, device="xpu")
    torch.xpu.synchronize()
    del block
    print("calling the kernel", flush=True)
    out = kernel(*device_inputs)
    torch.xpu.synchronize()
    print("the kernel returned", flush=True)
    assert not out.isnan().any()
    errors = (out.double().cpu() - ref).abs()
    rows = errors.reshape(ref.shape[0], -1).max(-1).values / ref.abs().max()
    assert rows.max() < _BOUND[case.dtype], f"worst row {rows.argmax()}: {rows.max()}"
