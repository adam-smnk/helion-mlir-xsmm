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
- `igc_vectorizer=["cmp"]`: IGC's vectorizer on but for vector compares, the kernel's
  own choice against the default `"off"` (`xpu_transforms.IGC_OPTIONS`).
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
| this kernel: `[1, 1, 256, 64]`, 97 `[1, 1, 512, 64]`, `"cmp"`, NaN-ignoring maxima | 217 | 319 | 2129 | 2405 |
| vectorizer off, NaN-ignoring, `[1, 1, 512, 128]` (best found off) | 298 | 410 | 2798 | 3028 |
| `"cmp"`, NaN-propagating, same blocks | 349 | 479 | 3423 | 3660 |
| all defaults: off, NaN-propagating, `[1, 1, 128, 64]` | 399 | 527 | 4434 | 4652 |
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

Checked on the device with `"cmp"` against torch's SDPA on the device and in f64, in
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

### Not covered

- The causal mask (`is_causal=True`), not attempted here. Skipping K/V tiles above
  the diagonal needs full tiles proven, which Helion's `affine.min` + `tensor.pad` of
  a triangular loop does not give (masking is shelved on XPU).
- A head dim other than 64: lighthouse's attention layout is written for it.
