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


def _check_rows(
    kernel: helion.Kernel, inputs: list[torch.Tensor], expected: torch.Tensor
) -> None:
    """Per row against ``expected``, NaN-poisoned, and again after other kernels: the
    same bits both times."""
    device_inputs = [t.to("xpu") for t in inputs]
    results = []
    for _ in range(2):
        _poison(expected.shape)
        out = kernel(*device_inputs)
        torch.xpu.synchronize()
        assert not out.isnan().any()
        errors = _row_errors(out, expected)
        assert errors.max() < _BOUND, f"worst row {errors.argmax()}: {errors.max()}"
        results.append(out.cpu())
        _noise()
    assert torch.equal(results[0].float(), results[1].float())


def _sibling_forms() -> dict[str, tuple[object, object, int]]:
    """Forms the sibling GEMM check refuses on lighthouse's path, that derived layouts
    take: ``(kernel, float64 reference, operand count)``."""
    from tests import test_xpu_sibling_gemms as sib

    return {
        "other_a_tiles": (sib.other_a_tiles, lambda x, y, g, u: x @ g.T + y @ u.T, 4),
        "separate_k_loops": (
            sib.separate_k_loops,
            lambda x, g, u: x @ g.T + x @ u.T,
            3,
        ),
        "three_gemms": (
            sib.three_gemms,
            lambda x, a, b, c: x @ a.T + x @ b.T + x @ c.T,
            4,
        ),
        "computed_a": (sib.computed_a, lambda x, g, u: 2 * x @ g.T + 2 * x @ u.T, 3),
        "one_accumulator": (sib.one_accumulator, lambda x, g, u: x @ g.T + x @ u.T, 3),
        "shared_b": (sib.shared_b, lambda x, w: 2 * x @ w.T, 2),
    }


@pytest.mark.parametrize(
    "name",
    [
        "other_a_tiles",
        "separate_k_loops",
        "three_gemms",
        "computed_a",
        "one_accumulator",
        "shared_b",
    ],
)
def test_forms_lighthouse_refuses_are_right_under_the_default(name: str) -> None:
    """They lower under derived layouts, so they must be right, per row."""
    fn, ref, operands = _sibling_forms()[name]
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    shapes = {2: ((64, 256), (512, 256))}.get(operands)
    if shapes is None:
        count = operands - 1
        first = 2 if name == "other_a_tiles" else 1
        shapes = ((64, 256),) * first + ((512, 256),) * (operands - first)
        assert count
    inputs = [
        (
            torch.randn(*shape, generator=generator) * (1.0 if i == 0 else 0.05)
        ).bfloat16()
        for i, shape in enumerate(shapes)
    ]
    expected = ref(*(t.double() for t in inputs))
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(
            block_sizes=[64, 256, 32, 32]
            if name == "separate_k_loops"
            else [64, 256, 32],
            xegpu_layouts=["ir"],
        ),
    )
    _check_rows(kernel, inputs, expected)


def _gemm_inputs(name: str, *shapes: tuple[int, int]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    return [
        (torch.randn(*shape, generator=generator) * (1.0 if not i else 0.05)).bfloat16()
        for i, shape in enumerate(shapes)
    ]


def test_crossed_accumulators_are_right_under_the_default() -> None:
    """``h = g + a @ gate^T; g = h + a @ up^T; u += h``, K step by K step: the second
    DPAS accumulates into the first's result, and ``u`` sums an intermediate."""
    from tests import test_xpu_sibling_gemms as sib

    inputs = _gemm_inputs("crossed", (64, 256), (512, 256), (512, 256))
    x, gate, up = (t.double() for t in inputs)
    g = torch.zeros(64, 512, dtype=torch.float64)
    u = torch.zeros(64, 512, dtype=torch.float64)
    for start in range(0, 256, 32):
        a = x[:, start : start + 32]
        h = g + a @ gate[:, start : start + 32].T
        g = h + a @ up[:, start : start + 32].T
        u = u + h
    kernel = helion.kernel(
        sib.crossed_accumulators,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 256, 32], xegpu_layouts=["ir"]),
    )
    _check_rows(kernel, inputs, g + u)


def test_siblings_with_a_row_reduction_are_right_under_the_default() -> None:
    """``g * u.sum(-1, keepdim)``, the sum over each workgroup's columns."""
    from tests import test_xpu_sibling_gemms as sib

    inputs = _gemm_inputs("with_row_reduction", (64, 256), (512, 256), (512, 256))
    x, gate, up = (t.double() for t in inputs)
    g, u = x @ gate.T, x @ up.T
    expected = torch.cat(
        [
            g[:, start : start + 256] * u[:, start : start + 256].sum(-1, keepdim=True)
            for start in (0, 256)
        ],
        -1,
    )
    kernel = helion.kernel(
        sib.with_row_reduction,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 256, 32], xegpu_layouts=["ir"]),
    )
    _check_rows(kernel, inputs, expected)


def _dropped_hints() -> dict:
    from tests.test_xpu_layouts import DROPPED_HINTS

    return DROPPED_HINTS


@pytest.mark.parametrize(
    "name",
    [
        "load_a_24_rows",
        "prefetch_not_whole_blocks",
        "load_a_1_row",
        "load_a_2_rows",
        "load_a_4_rows",
        "load_a_1_row_16_cols",
        "load_a_8_cols",
        "load_b_1_row",
        "load_b_2_rows",
        "load_b_4_rows",
        "load_b_8_rows",
        "load_b_8_cols",
        "sg_not_dividing",
        "sg_too_many_subgroups",
        "sg_too_many_dpas",
        "sg_zero_rows",
        "sg_splits_reduced_columns",
        "sg_attention_rows",
        "sg_elementwise",
        "sg_reduction",
        "load_not_dividing",
        "load_too_big",
        "load_two_entries_on_gemm",
        "load_transposed_32_rows",
        "load_four_on_elementwise",
        "load_four_on_reduction",
        "load_reduction_rows",
        "prefetch_not_dividing",
        "prefetch_too_small",
        "prefetch_tile_on_reduction",
        "prefetch_depth_on_elementwise",
    ],
)
def test_dropped_hint_runs_right(name: str) -> None:
    form, key, value, _ = _dropped_hints()[name]
    inputs = _inputs(form, name)
    if form.fn in (forms.scaled, forms.row_sums):
        inputs = [inputs[0] if form.fn is forms.scaled else inputs[0] * 0.05]
    expected = form.ref(*(t.double() for t in inputs))
    kernel = helion.kernel(
        form.fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(
            block_sizes=form.blocks, xegpu_layouts=["ir"], **{key: [value]}
        ),
    )
    _check_rows(kernel, inputs, expected)


_LIGHTHOUSE_SUBSET = [
    "sibling_gemms_m2048",
    "attention_head64",
    "plain_gemm",
    "elementwise",
    "reduction",
]


@pytest.mark.parametrize("name", _LIGHTHOUSE_SUBSET)
def test_lighthouses_path_on_the_device(name: str) -> None:
    """Lighthouse's annotation stays a supported option: one kernel of each class."""
    form = {
        "sibling_gemms_m2048": forms.FORMS["sibling_gemms_m2048"],
        "plain_gemm": forms.GEMM_FORM,
        "elementwise": forms.ELEMENTWISE_FORM,
        "reduction": forms.REDUCTION_FORM,
        "attention_head64": forms.Form(
            forms.attention,
            forms.attention_ref,
            [1, 512, 64],
            ((32, 512, 64), (32, 512, 64), (32, 512, 64)),
            "",
            (32, 1),
        ),
    }[name]
    inputs = _inputs(form, name)
    if form.fn is forms.row_sums:
        inputs = [inputs[0] * 0.05]
    kernel = helion.kernel(
        form.fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=form.blocks, xegpu_layouts=["lighthouse"]),
    )
    _check_rows(kernel, inputs, form.ref(*(t.double() for t in inputs)))
