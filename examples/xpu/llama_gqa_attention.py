"""Llama-3.2-1B attention on Intel Arc as a fused flash-attention Helion kernel.

Scaled dot-product attention with grouped-query attention (GQA): 32 query heads
share 8 key/value heads, Q (4, 32, 512, 64), K and V (4, 8, 512, 64), bf16, no
mask, ``F.scaled_dot_product_attention(q, k, v, enable_gqa=True)``. RoPE and the
causal mask are not part of the kernel.

Run on a node with an Arc GPU, with an LLVM and a PyTorch as in the main README's
XPU section:

    python examples/xpu/llama_gqa_attention.py
    python examples/xpu/llama_gqa_attention.py --device-time

The program checks the kernel and torch's SDPA against a float64 reference, then
times both on the same inputs. ``--device-time`` reruns the timing under
``unitrace`` for the device time of the kernel alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

import helion_mlir_backend  # noqa: F401

BATCH, HEADS, KV_HEADS, SEQ, HEAD_DIM = 4, 32, 8, 512, 64
"""The Llama-3.2-1B attention shape."""

BLOCK_SIZES = [1, 1, 256, 64]
"""``[batch, K/V head, query rows, K/V rows]``, picked for an Arc Pro B50 (16
Xe-cores, 32 large-register threads per core, 2048 bf16 DPAS flop/clk per core),
with IGC's vectorizer on, the default:

- Batch and K/V head 1: a workgroup works on one K/V head; those dims become
  grid dims (grid 4 x 8 x 8 = 256 workgroups, 16 waves of 16 Xe-cores).
- 256 query rows: 16 subgroups of 16 rows, so 2 workgroups on an Xe-core. Every
  subgroup reads the whole K and V tile: more subgroups on one tile share it
  through the L1.
- Subgroup rows are left at lighthouse's attention layout, 16: two 8x16 DPAS
  blocks.
- K/V tile 64: per subgroup and step, 32 DPAS calls per matmul.
- The fastest of the query and K/V row counts tried; K/V tiles of 32 also round worse.
"""

OTHER_SHAPES_BLOCK_SIZES = [1, 1, 128, 64]
"""Shapes :data:`BLOCK_SIZES` leaves one workgroup or does not divide."""


def gqa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: hl.constexpr
) -> torch.Tensor:
    """Flash attention of packed queries ``q: [B, Hkv, G * S, D]`` over K and V
    ``[B, Hkv, S, D]``: each workgroup takes a tile of query rows and runs over
    the K/V sequence in tiles with an online softmax (running row maxima and
    sums, the accumulator rescaled when the maximum grows).

    Packing gives the mapping ``K/V head = query head // group`` with no index
    computation and no repeated K/V: query head ``h = kv * G + g``, row ``s`` is
    row ``g * S + s`` of K/V head ``kv``.

    Matmuls take bf16 operands and accumulate in f32 (DPAS); the softmax is in
    f32, the probabilities are rounded to bf16 for ``P @ V``, the output once
    to bf16. A NaN score makes its row NaN through ``exp`` and the sum, as in
    torch, whichever form the maximum takes, so the row maxima may ignore NaN
    (``xegpu_nan_ignoring_maxima``). ``row_max`` starts at -FLT_MAX, not -inf: a
    first K/V tile of -inf scores adds ``exp(-inf) = 0``."""
    batch, kv_heads, rows, dim = q.shape
    seq = k.size(2)
    out = torch.empty_like(q)
    for tb, th, tm in hl.tile([batch, kv_heads, rows]):
        q_tile = q[tb, th, tm, :]  # load order Q, K, V: the attention layout's
        row_max = hl.full([tb, th, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, th, tm], dtype=torch.float32)
        acc = hl.zeros([tb, th, tm, dim], dtype=torch.float32)
        for tn in hl.tile(seq):
            scores = hl.zeros([tb, th, tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("bhmd,bhnd->bhmn", q_tile, k[tb, th, tn, :])
            scores = scores * scale
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, :, None] + torch.einsum(
                "bhmn,bhnd->bhmd", p.to(v.dtype), v[tb, th, tn, :]
            )
            row_max = new_max
        out[tb, th, tm, :] = (acc / row_sum[:, :, :, None]).to(out.dtype)
    return out


def _make_kernel(block_sizes: list[int]) -> object:
    return helion.kernel(
        gqa_attention,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(
            block_sizes=block_sizes,
            xegpu_nan_ignoring_maxima=[True],
        ),
    )


_KERNEL = _make_kernel(BLOCK_SIZES)
_OTHER_SHAPES_KERNEL = _make_kernel(OTHER_SHAPES_BLOCK_SIZES)


def llama_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel: object | None = None
) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, enable_gqa=True)`` for ``q: [B, H, S,
    D]`` (contiguous), ``k, v: [B, Hkv, S, D]``, ``Hkv`` dividing ``H``, ``D = 64``, ``S``
    a multiple of 64 above 64. The view costs nothing. With :data:`BLOCK_SIZES` if 256
    divides the ``R = S * H / Hkv`` query rows and that makes several workgroups (``B *
    Hkv * R > 256``), else with :data:`OTHER_SHAPES_BLOCK_SIZES`, for which the same
    must hold of 128: other shapes are refused. ``kernel`` (takes the packed queries,
    K, V and the scale) replaces that choice."""
    batch, heads, seq, dim = q.shape
    kv_heads = k.size(1)
    rows = heads // kv_heads * seq
    if kernel is None:
        fits = rows % 256 == 0 and batch * kv_heads * rows > 256
        kernel = _KERNEL if fits else _OTHER_SHAPES_KERNEL
    packed = q.view(batch, kv_heads, rows, dim)
    return kernel(packed, k, v, hl.constexpr(1.0 / math.sqrt(dim))).view(q.shape)


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


def _inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(*shape, generator=generator).to(torch.bfloat16).to(device)
        for shape in (
            (BATCH, HEADS, SEQ, HEAD_DIM),
            (BATCH, KV_HEADS, SEQ, HEAD_DIM),
            (BATCH, KV_HEADS, SEQ, HEAD_DIM),
        )
    )
    return q, k, v


def _poisoned_call(fn: object, *args: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """``fn(*args)`` with the block its output will likely reuse full of NaN."""
    poison = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="xpu")
    del poison
    torch.xpu.synchronize()
    out = fn(*args)
    torch.xpu.synchronize()
    return out


def _errors(out: torch.Tensor, ref: torch.Tensor) -> str:
    error = (out.cpu().double() - ref).abs()
    scale = ref.abs().max()
    nans = int(out.isnan().sum())
    return f"max {error.max() / scale:.2e}, mean {error.mean() / scale:.2e}, NaN {nans}"


def check() -> None:
    """Normalized errors against a float64 reference, on outputs NaN-poisoned."""
    q, k, v = _inputs("xpu")
    torch_out = _poisoned_call(torch_attention, q, k, v, shape=q.shape)
    helion_out = _poisoned_call(llama_attention, q, k, v, shape=q.shape)
    ref = torch_attention(q.cpu().double(), k.cpu().double(), v.cpu().double())
    print(f"helion error vs float64: {_errors(helion_out, ref)}")
    print(f"torch  error vs float64: {_errors(torch_out, ref)}")
    if not torch.equal(llama_attention(q, k, v), helion_out):
        raise SystemExit("repeated call differs")
    if helion_out.isnan().any():
        raise SystemExit("NaN in the output")


def _timed(fn: object, inputs: tuple, flush: torch.Tensor, calls: int) -> list[float]:
    """Wall time per synchronized call, us. Filling a 256 MiB buffer first takes
    the inputs out of the GPU's L2, as AI-bench does."""
    times = []
    for _ in range(calls):
        flush.zero_()
        torch.xpu.synchronize()
        start = time.perf_counter()
        fn(*inputs)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    return times


def bench(warmup: int = 100, calls: int = 200, rounds: int = 3) -> None:
    """Interleaved, on the same inputs: a B50 is power-capped, its clock drifts."""
    inputs = _inputs("xpu")
    flush = torch.empty(256 * 2**20, dtype=torch.int8, device="xpu")
    backends = {"helion": llama_attention, "torch": torch_attention}
    with torch.no_grad():
        for fn in backends.values():
            _timed(fn, inputs, flush, warmup)
        times: dict[str, list[float]] = {name: [] for name in backends}
        for _ in range(rounds):
            for name, fn in backends.items():
                times[name] += _timed(fn, inputs, flush, calls)
    for name, values in times.items():
        print(f"{name:7s} wall time per call: {statistics.median(values):7.1f} us")


def _device_times(trace_dir: Path) -> dict[str, float]:
    """Median device time per kernel, from a unitrace chrome trace."""
    events: dict[str, list[float]] = {}
    for path in trace_dir.glob("*.json"):
        trace = json.loads(path.read_text())
        for event in trace["traceEvents"] if isinstance(trace, dict) else trace:
            if event.get("ph") == "X" and event.get("cat") == "gpu_op":
                events.setdefault(event["name"].split("[")[0], []).append(event["dur"])
    return {name: statistics.median(durs) for name, durs in events.items()}


def device_time() -> None:
    """Reruns the timing under unitrace and prints the attention kernel's time."""
    if shutil.which("unitrace") is None:
        print("no unitrace on PATH: wall times only")
        return
    with tempfile.TemporaryDirectory() as trace_dir:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{sys.prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
        command = [
            "unitrace",
            # Following children would preload the tool into ocloc, which then
            # drops -igc_opts: another binary (xpu-perf-guide in the wiki).
            "--follow-child-process",
            "0",
            "--chrome-device-logging",
            "--output-dir-path",
            trace_dir,
        ]
        command += [sys.executable, __file__, "--bench-only"]
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
        medians = _device_times(Path(trace_dir))
    for name, duration in medians.items():
        if "gqa_attention" in name:
            print(f"device time per call: {duration:7.1f} us  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device-time", action="store_true")
    parser.add_argument("--bench-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.bench_only:
        check()
    bench()
    if args.device_time:
        device_time()


if __name__ == "__main__":
    main()
