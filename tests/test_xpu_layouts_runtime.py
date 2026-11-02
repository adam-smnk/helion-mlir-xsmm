"""The forms of ``xpu_layout_forms`` on the device: per row against float64 with
NaN-poisoned outputs, again after other kernels ran (stale registers), and with
the output a view inside a NaN-filled buffer (writes astray)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests import xpu_layout_forms as forms

pytestmark = [pytest.mark.xpu_runtime, pytest.mark.isolated]

_BOUND = 2**-7
"""Two roundings to bf16 of nearly the same value, doubled: as ``test_xpu_runtime``."""


@pytest.fixture(autouse=True)
def _full_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


def _kernel(form: forms.Form) -> helion.Kernel:
    return helion.kernel(
        form.fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=form.blocks, xegpu_layouts=["ir"]),
    )


def _inputs(form: forms.Form, name: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    scale = 1.0 if form.fn is forms.attention else 0.05
    return [
        (
            torch.randn(*shape, generator=generator) * (1.0 if not i else scale)
        ).bfloat16()
        for i, shape in enumerate(form.shapes)
    ]


def _row_errors(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    errors = (out.double().cpu() - ref).abs()
    return errors.reshape(ref.shape[0], -1).max(-1).values / ref.abs().max()


def _poison(shape: torch.Size) -> None:
    """A NaN block of the output's size, freed: the output may reuse it."""
    block = torch.full(tuple(shape), float("nan"), dtype=torch.bfloat16, device="xpu")
    torch.xpu.synchronize()
    del block


def _noise() -> None:
    """Other kernels' registers: a different GEMM and torch's elementwise ops."""
    form = forms.FORMS["gemm_m32"]
    _kernel(form)(*(t.to("xpu") for t in _inputs(form, "noise")))
    x = torch.randn(1024, 1024, device="xpu")
    (x * 2.0).exp().sum()
    torch.xpu.synchronize()


@pytest.mark.parametrize("name", list(forms.FORMS))
def test_form_rows_after_other_kernels(name: str) -> None:
    form = forms.FORMS[name]
    inputs = _inputs(form, name)
    ref = form.ref(*(t.double() for t in inputs))
    kernel = _kernel(form)
    device_inputs = [t.to("xpu") for t in inputs]
    results = []
    for _ in range(2):
        _poison(ref.shape)
        out = kernel(*device_inputs)
        torch.xpu.synchronize()
        assert not out.isnan().any()
        errors = _row_errors(out, ref)
        assert errors.max() < _BOUND, f"worst row {errors.argmax()}: {errors.max()}"
        results.append(out.cpu())
        _noise()
    assert torch.equal(results[0].float(), results[1].float())


def _swiglu_into(x, gate, up, out):
    m, k = x.size()
    n = gate.size(0)
    for tm, tn in hl.tile([m, n]):
        g = hl.zeros([tm, tn], dtype=forms.F32)
        u = hl.zeros([tm, tn], dtype=forms.F32)
        for tk in hl.tile(k):
            g = torch.addmm(g, x[tm, tk], gate[tn, tk].T)
            u = torch.addmm(u, x[tm, tk], up[tn, tk].T)
        out[tm, tn] = (g / (1.0 + torch.exp(-g)) * u).to(x.dtype)


def _row_norm_into(x, w, out):
    m, k = x.size()
    n = w.size(0)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=forms.F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        out[tm, :] = (acc / torch.abs(acc).sum(-1, keepdim=True)).to(x.dtype)


def _linear_into(x, w, out):
    m, k = x.size()
    n = w.size(0)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=forms.F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        out[tm, tn] = acc.to(x.dtype)


_INTO = {
    "sibling_gemms_m2048": _swiglu_into,
    "row_norm_n256": _row_norm_into,
    "gemm_m8": _linear_into,
}


@pytest.mark.parametrize("name", list(_INTO))
def test_form_writes_only_its_output(name: str) -> None:
    form = forms.FORMS[name]
    inputs = _inputs(form, name)
    ref = form.ref(*(t.double() for t in inputs))
    m, n = ref.shape
    buffer = torch.full(
        (m + 64, n + 128), float("nan"), dtype=torch.bfloat16, device="xpu"
    )
    kernel = helion.kernel(
        _INTO[name],
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=form.blocks, xegpu_layouts=["ir"]),
    )
    kernel(*(t.to("xpu") for t in inputs), buffer[32 : 32 + m, 64 : 64 + n])
    torch.xpu.synchronize()
    host = buffer.cpu().float()
    outside = torch.ones_like(host, dtype=torch.bool)
    outside[32 : 32 + m, 64 : 64 + n] = False
    assert host[outside].isnan().all()
    inside = host[32 : 32 + m, 64 : 64 + n]
    assert not inside.isnan().any()
    assert _row_errors(inside, ref).max() < _BOUND
