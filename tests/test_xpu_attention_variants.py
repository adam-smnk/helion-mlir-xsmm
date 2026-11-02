"""Attention variants of ``examples/xpu/attention_variants.py``: they lower for the XPU
(no device needed), and on a device they match float64 references."""

from __future__ import annotations

import helion.language as hl
import pytest
import torch

from tests.harness import load_example
from tests.harness import lower_xpu
from tests.harness import xpu_tensor

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

av = load_example("xpu/attention_variants.py")

BF16 = torch.bfloat16
_SHAPE = (2, 4, 512, 64)
_MASKS = [
    "causal",
    "sliding_window",
    "dilated_sw",
    "block_diffusion",
    "document",
    "gemma2",
]
_NAMES = list(av.VARIANTS)


def _fake_args(name: str, shape: tuple[int, int, int, int] = _SHAPE) -> list[object]:
    batch, heads, seq, dim = shape
    kv_heads = av.VARIANTS[name].kv_heads(heads)
    extra = av.call_args(name, batch, heads, seq, dim, "cpu")
    if name == "document":
        extra[1] = xpu_tensor(batch * heads, seq, dtype=torch.int32)
    return [
        xpu_tensor(batch * heads, seq, dim, dtype=BF16),
        xpu_tensor(batch * kv_heads, seq, dim, dtype=BF16),
        xpu_tensor(batch * kv_heads, seq, dim, dtype=BF16),
        *extra,
    ]


@pytest.mark.isolated
@pytest.mark.parametrize("name", _NAMES)
def test_variant_lowers_as_a_fused_attention_kernel(name: str) -> None:
    text = lower_xpu(av.make_kernel(name), _fake_args(name))
    assert text.count("xegpu.dpas") == 2
    assert text.count("scf.for") == 1
    assert "xegpu.prefetch_nd" in text


@pytest.mark.isolated
@pytest.mark.parametrize("name", _MASKS)
def test_masks_skip_key_tiles(name: str) -> None:
    """The K/V loop's bound is a value, not the sequence: tiles are skipped, and K/V
    loads are not masked (no masked loads on XPU)."""
    text = lower_xpu(av.make_kernel(name), _fake_args(name))
    loop = next(line for line in text.splitlines() if "scf.for" in line)
    assert " to %c512 " not in loop
    assert "xegpu.load_nd" in text
    assert "mask" not in text.replace("masked", "")


@pytest.mark.isolated
def test_head_dim_128_is_a_clear_error() -> None:
    """Lighthouse's attention layout is written for a head dim of 64 (the blocker of
    Gemma 2's D = 128)."""
    args = _fake_args("gemma2", (2, 4, 512, 128))
    with pytest.raises(UnsupportedOperationError, match="head dim 128"):
        lower_xpu(av.make_kernel("gemma2"), args)


# --- The reference itself, on the CPU --------------------------------------------------


def _small(name: str) -> list[torch.Tensor]:
    return av.inputs_of(name, 2, 4, 128, 64)


@pytest.mark.parametrize("name", ["plain", "causal", "sliding_window", "document"])
def test_reference_matches_sdpa_with_the_dense_mask(name: str) -> None:
    q, k, v = (x.double() for x in _small(name))
    fn = av.VARIANTS[name].mask
    full = torch.ones(128, 128, dtype=torch.bool)
    masks = [
        full if fn is None else fn(av.RowContext(b, 2, 0, 4, 128)) for b in range(2)
    ]
    mask = torch.stack(masks)[:, None]
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(
        av.reference(name, q, k, v), sdpa, rtol=1e-12, atol=1e-12
    )


def test_reference_alibi_matches_sdpa_with_a_float_mask() -> None:
    q, k, v = (x.double() for x in _small("alibi"))
    i, j = torch.arange(128)[:, None], torch.arange(128)[None, :]
    slopes = torch.tensor([av._slope(h, 4) for h in range(4)], dtype=torch.float64)
    bias = (j - i).double()[None] * slopes[:, None, None]
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias[None]
    )
    torch.testing.assert_close(
        av.reference("alibi", q, k, v), sdpa, rtol=1e-12, atol=1e-12
    )


def test_documents_have_rows_with_a_single_key() -> None:
    counts = av.visible_keys("document", 2, 512)
    assert counts.min() == 1


# --- On a device ------------------------------------------------------------------------


def _ulp_bound(result: dict[str, object]) -> None:
    """A row's largest error stays that of bf16 rounding of probabilities and outputs:
    plain attention on these inputs gives 5.5e-3, torch's SDPA on the device the same."""
    assert result["row_max"] < 2**-6, result
    assert result["extra_nan"] == 0, result
    assert result["lost_nan"] == 0, result
    assert result["second_call_equal"], result
    assert result["differs_after"] == [], result


@pytest.mark.xpu_runtime
@pytest.mark.isolated
@pytest.mark.parametrize("vectorizer", ["off", "cmp"])
@pytest.mark.parametrize("name", _NAMES)
def test_variant_against_float64(name: str, vectorizer: str) -> None:
    _ulp_bound(av.check(name, vectorizer=vectorizer))


@pytest.mark.xpu_runtime
@pytest.mark.isolated
@pytest.mark.parametrize("name", ["softcap", "gemma2"])
def test_softcap_where_it_caps(name: str) -> None:
    """Inputs 8 times larger: scores up to the cap and beyond."""
    _ulp_bound(av.check(name, gain=8.0))


@pytest.mark.xpu_runtime
@pytest.mark.isolated
def test_row_without_visible_keys_is_nan() -> None:
    """As torch's SDPA and softmax: row 0 sees nothing under a strict causal mask."""
    result = av.check("causal_strict")
    _ulp_bound(result)
    assert result["visible_min"] == 0


@pytest.mark.xpu_runtime
@pytest.mark.isolated
@pytest.mark.parametrize("name", ["plain", "alibi", "causal", "sliding_window"])
def test_same_source_through_triton(name: str) -> None:
    """One source: the kernel runs through Helion's Triton backend as it is, with its
    own config, and is as close to the reference."""
    q, k, v = av.inputs_of(name, 2, 4, 512, 64)
    kernel = av.make_kernel(
        name, backend="triton", block_sizes=[1, 32, 16], num_warps=4, num_stages=2
    )
    args = av.call_args(name, 2, 4, 512, 64)
    if name == "sliding_window":
        args[-1] = hl.constexpr(16)  # The K/V tile the kernel aligns its loop to.
    out = kernel(*(av.as_streams(x).to("xpu") for x in (q, k, v)), *args).view(q.shape)
    errors = av.errors(out, av.reference(name, q, k, v))
    assert errors["row_max"] < 2**-6, errors
    assert errors["extra_nan"] == errors["lost_nan"] == 0, errors
