# XPU examples

Helion kernels for Intel Arc GPUs on the XPU target. Requirements: the main README's XPU section.

## `llama_gqa_attention.py`

Llama-3.2-1B attention (grouped-query SDPA, 32 query heads on 8 K/V heads, bf16, no mask) as one
fused flash-attention kernel. The query heads are viewed as rows of their K/V head,
`q.view(B, Hkv, G * S, D)`, so nothing is repeated or copied. The config and its rationale are
next to `BLOCK_SIZES`.

```bash
python examples/xpu/llama_gqa_attention.py --device-time
```

It prints the kernel's and torch's errors against a float64 reference, then the times of both.

`xegpu_nan_ignoring_maxima=[True]` lets the row maxima ignore NaN: an online softmax's maximum
only shifts its exponents, so a NaN score still makes its row NaN. This holds only with IGC's
options as the backend passes them (`VectorizerAllowCMP=0` or `EnableVectorizer=0`).

## `attention_variants.py`

Forward attention variants as fused flash-attention kernels: each is the plain kernel with a few
lines changed, in full, so a diff of two kernels shows what a variant takes. Kernels take
`[B * H, S, D]` tiles; `S` a multiple of 256, `D = 64`. The configs and their rationale:
`BLOCK_SIZES`, `MASK_BLOCK_SIZES`.

```bash
python examples/xpu/attention_variants.py check causal
python examples/xpu/attention_variants.py bench causal --seq 1024
```

`check` compares a kernel to a float64 reference; `bench` prints the time per call.

| Kernel | Score or mask (`i` query row, `j` key row, `h` query head) |
|---|---|
| `alibi` | `s + (j - i) * 2 ** (-8 * (h + 1) / H)` |
| `alibi_abs` | `s - abs(i - j) * 2 ** (-8 * (h + 1) / H)` |
| `softcap` | `50 * tanh(s / 50)` |
| `sigmoid_act` | `sigmoid(s)`, then softmax |
| `rel_bias` | `s + (i - j) * 0.01` |
| `causal` | `j <= i` |
| `causal_split` | as `causal`, in two K/V loops |
| `causal_strict` | `j < i` (row 0 sees nothing) |
| `sliding_window` | `0 <= i - j <= 256` |
| `dilated_sw` | the same and `(i - j) % 2 == 0` |
| `block_diffusion` | `j // 128 <= i // 128` |
| `block_diffusion_aligned` | as `block_diffusion`, query tile = block, no mask |
| `document` | `j <= i` and `documents[b, i] == documents[b, j]` |
| `gemma2` | `50 * tanh(s / 50)`, `j <= i`, 16 query heads on 8 K/V heads |
