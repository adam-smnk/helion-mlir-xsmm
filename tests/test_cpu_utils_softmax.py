"""``helion_mlir_utils.softmax`` over rows whose width is not a whole number
of 128-column chunks.

The kernel kept lane-wise ``[rows, 128]`` maxima and sums across chunks and
reduced them after the loop, so the zero padding of a partial last chunk
entered both: each padded lane added ``exp(-max)`` to the row sum.
AI-bench's L2/66 at 130x1001x515 (softmax of a [130, 515] GEMM output) was
off by 0.18 x max|ref|, by 1.0 for rows far below zero."""

from __future__ import annotations

from helion_mlir_utils import softmax
import pytest
import torch


@pytest.mark.parametrize(
    ("m", "n", "dtype", "shift"),
    [
        (130, 515, torch.float32, 0.0),  # L2/66's GEMM output, 130x1001x515
        (4, 129, torch.float32, -50.0),  # one padded chunk of 127 lanes
        (2, 200, torch.bfloat16, -50.0),
        (4, 256, torch.float32, -50.0),  # whole chunks only
        (3, 47, torch.float32, 0.0),  # one chunk narrower than 128
    ],
)
def test_softmax_partial_last_chunk(m, n, dtype, shift) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, n) + shift).to(dtype)
    out = softmax(x)
    ref = torch.softmax(x.double(), -1)
    error = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert error < (1e-6 if dtype is torch.float32 else 1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_one_column(dtype) -> None:
    """One column (AI-bench L2/66, 84, 99 with OUT_FEATURES = 1): the chunked
    kernel raised ControlFlowTensorMismatch. Softmax is 1, NaN for NaN and
    infinite inputs, as torch's."""
    x = torch.tensor([[0.5], [-3e4], [float("nan")], [float("inf")], [-float("inf")]])
    x = torch.cat([x, torch.randn(60, 1)]).to(dtype)
    out = softmax(x)
    ref = torch.softmax(x, -1)
    assert torch.equal(out.isnan(), ref.isnan())
    assert torch.equal(out[~ref.isnan()], ref[~ref.isnan()])
