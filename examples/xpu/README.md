# XPU examples

Kernels for Intel Arc through Helion → helion-mlir-backend → lighthouse XeGPU →
XeVM → SPIR-V → Level Zero.

## `llama_gqa_attention.py`

Llama-3.2-1B attention (grouped-query SDPA, 32 query heads on 8 K/V heads, Q
`(4, 32, 512, 64)`, K/V `(4, 8, 512, 64)`, bf16, no mask) as one fused
flash-attention kernel:

- The query heads are viewed as rows of their K/V head, `q.view(B, Hkv, G * S, D)`:
  query head `h` uses K/V head `h // G`, with nothing repeated or copied.
- Hand-picked config, with its rationale next to `BLOCK_SIZES`: workgroups of 256
  query rows (16 subgroups of 16 rows) over K/V tiles of 64. Shapes that does not fit
  take 128 query rows (`llama_attention`'s docstring).
- `xegpu_nan_ignoring_maxima=[True]`: the row maxima ignore NaN, with the same
  output given IGC's options (below).
- The scale is an `hl.constexpr` (a constant in the IR); a runtime float also works.

Run it on a node with an Arc GPU (only the B50 was measured), with an LLVM and a
PyTorch as in the main README's XPU section:

```bash
python examples/xpu/llama_gqa_attention.py --device-time
```

It prints the normalized errors of the kernel and of torch's SDPA against a float64
reference, on NaN-poisoned outputs, then the times of both on the same inputs.

### Measured

Arc Pro B50, IGC 2.41, median per call over 3 processes, randn inputs, L2 flushed
before each call, device times from `unitrace --follow-child-process 0`. Wall time
includes the host work of a call. µs.

| | Llama GQA device | wall | 97 (32x32x512x64) device | wall |
|---|---|---|---|---|
| this kernel: `[1, 1, 256, 64]`, 97 `[1, 1, 512, 64]`, vectorizer on (the default), NaN-ignoring maxima | 217 | 319 | 2129 | 2405 |
| `igc_vectorizer=["off"]`, NaN-ignoring, `[1, 1, 512, 128]` (best found off) | 298 | 410 | 2798 | 3028 |
| vectorizer on, NaN-propagating, same blocks | 349 | 479 | 3423 | 3660 |
| `igc_vectorizer=["off"]`, NaN-propagating, `[1, 1, 128, 64]` | 399 | 527 | 4434 | 4652 |
| torch SDPA | 283 | 327 | 2813 | 2939 |
| lighthouse SDPA fusion through AI-bench's pipeline, this backend's tail (default options) | 445 | 579 | 4660 | 4878 |

Rows beside the first are one process each. Errors of this kernel and of torch against
float64 at the Llama shape: max 2.26e-3, mean 1.40e-4 and 1.41e-4.

Lighthouse's own `kernel_bench.py -i llama-bench_1B_attention.py` (`torch.rand` inputs,
verification passed), not comparable with the table: 416 device, 443 by its timer, which
has no L2 flush, while the table's rows flush before each call. It also needs more than
10 warm-up calls: 20-call runs after 10 warm-ups averaged 784 and 753 us, after 20
warm-ups 439.

Device times come from `unitrace --follow-child-process 0`. Its default of following
child processes preloads the tool into `ocloc`, which starts OpenCL before the compile
and then does not apply `-igc_opts` (they reach IGC's options file, not its command
line), so the trace would time another binary; `--device-time` passes the flag.

### NaN and infinities

An online softmax's maximum `m` only shifts: `exp(s - m) / sum(exp(s - m))`. A NaN
score gives a NaN `exp`, the running sum stays NaN, and so does the row, whatever `m`
was. Without a NaN, `maxnum` and `maximum` differ at most in the sign of a zero. So
a row maximum used only that way may ignore NaN, in attention kernels: the key is
refused elsewhere, and a reduction used otherwise in the kernel (stored, compared)
keeps propagating NaN.

**This holds only with IGC's options:** `maxnum` row maxima are the form IGC's
vectorizer miscompiles without `VectorizerAllowCMP=0` or `EnableVectorizer=0`. With a
default `unitrace`, which loses them, up to half of the rows came out NaN. Nothing
refuses that at compile time, since IGC versions differ; the backend's tests scan
these binaries.

Checked on the device with the vectorizer on against torch's SDPA on the device and in f64, in
the tests 188 cases plus 4 of the row below (47 edits, small GQA and 97 shapes, bf16 and
f16), and once by hand 284 with the spec shapes in bf16 (NaN, inf and -inf in Q, K and V, single elements, rows,
tiles and heads): the two forms give the same bits in all of them, with torch's NaN
and inf elements in all but those below. f32 operands do not lower on XPU (DPAS takes
16-bit floats), so f32 was checked in eager torch only: five forms of the maximum
give the same NaN pattern.

- **A row of only -inf scores** gives NaN here, as `F.softmax(s, -1) @ V` does, and
  finite outputs in torch's SDPA.
- **Scores that overflow f32** (K near the type's maximum) give NaN here and in
  torch's SDPA on the device; f64 does not overflow.
- Torch's SDPA on the CPU in bf16 gives no NaN for an inf input where the device and
  f64 do; the comparison above does not use it.

### IGC's vectorizer

The kernel runs with IGC's vectorizer on, as every kernel does by default
(`xpu_transforms.IGC_OPTIONS`); `igc_vectorizer=["off"]` turns it off for a kernel.
On an Arc B50 with IGC 2.41, every kernel benchmarked is correct. The two known bugs
are avoided by a flag (`VectorizerAllowCMP=0`) and an IR rewrite
(`xevm_lane_conversions`), on that one IGC build. Other kernels and other builds
are not proven; `agent-tools/igc_binary_check.py` in helion-mlir-dev checks a kernel's
binaries.

### Not covered

- The causal mask (`is_causal=True`), not attempted here. Skipping K/V tiles above
  the diagonal needs full tiles proven, which Helion's `affine.min` + `tensor.pad` of
  a triangular loop does not give (masking is shelved on XPU).
- A head dim other than 64: lighthouse's attention layout is written for it.

## `attention_variants.py`

Forward attention variants as fused flash-attention kernels: each is the plain kernel with
a few lines changed, in full, so a diff of two kernels shows what a variant takes.
`python examples/xpu/attention_variants.py check causal` compares one to a float64 reference.
Kernels take `[B * H, S, D]` tiles of one stream and query rows; `as_streams` makes the view.
Shapes: `S` a multiple of 256, `D = 64`; no masked loads and no partial tiles. Config and its
rationale: `BLOCK_SIZES`, `MASK_BLOCK_SIZES`. `check --vectorizer off` runs without IGC's vectorizer.

The same file also runs through Helion's Triton backend, with its own config; that shows the source is
portable, not what the Triton-XPU stack can do. Rank 3 with `torch.exp` is about 2460 µs at `(4, 16, 1024, 64)`
there: `torch.exp` lowers to `__imf_expf`. Rank 2 with `indexing="tensor_descriptor"` and `tl.math.exp2` is 673 µs on the
same stack. `hl.dot` is used as `torch.einsum` of bf16 operands rounds its result to bf16 in that backend.

Definitions used (`i` query row, `j` key row, `h` query head; scores after the scale `1 / sqrt(D)`):

| Kernel | Score or mask | Source |
|---|---|---|
| `alibi` | `s + (j - i) * 2 ** (-8 * (h + 1) / H)` | Helion 1.4 `add_alibi_bias_t2r`; Triton-XPU's `alibi_functional` |
| `alibi_abs` | `s - abs(i - j) * 2 ** (-8 * (h + 1) / H)` | Press et al. 2021, ALiBi |
| `softcap` | `50 * tanh(s / 50)` | Helion 1.4 `softcap_t2r`; Triton-XPU's `tanh_softcap_functional`; Gemma 2 (the value) |
| `sigmoid_act` | `sigmoid(s)`, then softmax | **inferred** from the name; no source found |
| `rel_bias` | `s + (i - j) * 0.01` | Helion 1.4 `add_relative_bias_t2r`; the slope is **inferred** |
| `causal` | `j <= i` | SDPA's `is_causal`; Triton-XPU's causal benchmark |
| `causal_split` | as `causal`, in two K/V loops | form of `causal` |
| `causal_strict` | `j < i` (row 0 sees nothing) | edge case of the mask, not a slide's variant |
| `sliding_window` | `0 <= i - j <= 256` | Helion 1.4 `sliding_window_mask_t2r`; the window is **inferred** |
| `dilated_sw` | the same and `(i - j) % 2 == 0` | **inferred**; no source found |
| `block_diffusion` | `j // 128 <= i // 128` | **inferred** from Block Diffusion (Arriola et al., 2025); block size chosen |
| `block_diffusion_aligned` | as `block_diffusion`, query tile = block, no mask | form of it |
| `document` | `j <= i` and `documents[b, i] == documents[b, j]` | Helion 1.4 `document_mask_t2r` (causal and same document) |
| `gemma2` | `50 * tanh(s / 50)`, `j <= i`, 16 query heads on 8 K/V heads | **inferred** as the combination; `D = 64` here, see below |

Masks skip the K/V tiles a query tile cannot see: the K/V loop ends at the query tile's end and,
in the windows, starts behind it. Inside the tiles it reads, a compare and a select apply the
mask. `hl.tile`'s bounds must then be provably multiples of the K/V tile, and within K's
extent, or the kernel is refused as having partial tiles.

### Measured

Arc Pro B50, IGC 2.41, `(4, 16, S, 64)`, bf16, randn inputs, L2 flushed before each call,
`agent-tools/xpu_kernel_time.py`: 200 calls in 2 processes, last 50 of each, one node and session. Device time / wall
in µs; wall minus device is the host path and differs by backend. Largest error per row against float64: at most
6.6e-3. The vectorizer is on, its default. FlexAttention and the flash kernel of Intel's benchmarks
(`128,32`, `128,64` with mask) go through Triton-XPU 3.7.2.

| Kernel | S = 1024 | Flex | Native | S = 2048 | Flex | Native | S = 4096 | Flex | Native |
|---|---|---|---|---|---|---|---|---|---|
| plain | 403 / 520 | 476 / 585 | 487 / 565 | 1677 / 1943 | 2045 / 2259 | 2162 / 2325 | 7450 / 7767 | 8682 / 8900 | 8853 / 9099 |
| causal | 307 / 441 | 299 / 365 | 308 / 367 | 1226 / 1463 | 1098 / 1242 | 1152 / 1270 | 5292 / 5536 | 4471 / 4689 | 4621 / 4796 |
| causal_split | 285 / 409 | | | 1106 / 1348 | | | 4826 / 5102 | | |
| sliding_window | 236 / 355 | 259 / 314 | | 483 / 640 | 483 / 556 | | 1045 / 1255 | 1024 / 1135 | |
| dilated_sw | 266 / 398 | 248 / 301 | | 554 / 721 | 487 / 569 | | | | |
| block_diffusion | 301 / 434 | 287 / 358 | | 1209 / 1440 | 1112 / 1258 | | | | |
| block_diffusion_aligned | 253 / 374 | | | 933 / 1161 | | | | | |
| document | 345 / 501 | 272 / 348 | | 1356 / 1567 | 557 / 657 | | 5452 / 5721 | 1181 / 1296 | |

FlexAttention skips K/V blocks by a mask computed on the host, so `document` also skips by data there.

What the numbers rest on, each measured alone:

- **Query tiles first in `hl.tile`:** the dispatch runs them fastest, so consecutive workgroups
  of a thread read one stream's K/V. `causal_split` 1617 → 1084 µs at S = 2048.
- **K/V tile 32 under masks or selects:** IGC's vectorizer only stubs a per-element `select`; at 64
  K/V rows that takes the softmax out of vector form, at 32 it stays. Causal asm 4395 → 1784 lines,
  softcap 2832 → 952 µs wall. Plain itself stays at 64.
- **Not masking fully visible tiles** (`causal_split`, about 12 more lines): 7 %. Where the
  query tile is the block, nothing needs a mask (`block_diffusion_aligned`).

Not covered:

- **`document` skips only the causal part:** a loop bound read from a tensor would skip by data. It needs a
  proof of full tiles from a tensor read, which no source form gave.
- `D = 128` (Gemma 2's): lighthouse's attention layout hard-codes a head dim of 64.
- A bias from a tensor, `bias[tm, tn]`: a third 2D load in an attention kernel has no layout (f32:
  over the block load's limit in XeVM; bf16: a 64-element vector, LLVM's SPIR-V backend aborts).
- A per-head runtime scalar (`slopes[th]`): a 0-D read that XeGPU's layout propagation cannot convert. The
  slope is computed from `th.index`.
- `masked_fill` and `torch.where` with a Python float: torch-mlir does not lower the scalar
  tensor; the kernels use `hl.full`.

Edge cases, all measured against a dense float64 reference on the device: a row with no visible key is
NaN, as with torch's softmax. A NaN in `V` at a key a query cannot see in a skipped K/V tile
does not reach the row in the kernel, whereas it does in a dense `P @ V` (`0 * NaN`). FlexAttention
gives 0, by its template, for a row with no key.
