# helion-mlir-backend

External MLIR backend package for Helion.

Currently, only CPU execution is supported. Intel XPU support is experimental and
IR-only (see [XPU (experimental)](#xpu-experimental)).

## Usage

When `helion_mlir_backend` is imported, it registers the `mlir` backend in Helion.

Usable by standard Helion kernels with `mlir` backend selected:

```python
import helion
import helion_mlir_backend  # noqa: F401

@helion.kernel(backend="mlir")
def kernel():
  ...
```

Following Helion configurations (`config=helion.Config()`) are supported:
* **block\_sizes** (`list[int]`):
Controls tile sizes corresponding to each dimension passed `hl.tile` or call
to `hl.register_block_size` in the kernel.

Additional MLIR backend kernel (`helion.kernel(...)`) options:
* **mlir\_pipeline** (`"opt"`, `"scalar"`, `"xegpu"`):
Selects compilation pipeline:
  - `opt`: Optimizing pipeline (default). Vectorization, parallelization etc.
  - `scalar`: Scalar lowering. A simple serial fallback; mostly useful for debugging.
  - `xegpu`: The XPU target's pipeline (its only one; CPU pipelines are rejected on the XPU).
* **xegpu\_sg\_tiles**, **xegpu\_load\_tiles**, **xegpu\_prefetch\_tiles**,
**xegpu\_prefetch\_depths**: XeGPU parameters per GPU kernel on the XPU, ignored on
the CPU (see [XeGPU parameters](#xegpu-parameters)).

## XPU (experimental)

Kernels on `xpu` tensors (or with `target="xpu"` / `HELION_MLIR_TARGET=xpu`) use the
XPU target: lighthouse's XeGPU pipeline (`xegpu`, the only XPU pipeline), with
Helion's grid loops as the workgroup tiles. Each top-level `hl.tile` loop becomes one
kernel launch; the launches of a kernel (`hl.barrier()` phases included) run in
program order on one stream. Runtime scalars are passed by value and converted to the
kernel's types on the host, so a kernel never takes an `f64` (a kernel that would, e.g.
one doing `f64` math on a scalar, is rejected).

Status: IR only. Without an XPU, kernels lower to annotated XeGPU (`xegpu-wg`) and to
LLVM/XeVM without device binaries (`xevm`):

```python
import helion_mlir_backend

ir = helion_mlir_backend.lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")
```

`args` may be fake XPU tensors (`with FakeTensorMode(): torch.empty(..., device="xpu")`),
meta tensors or CPU tensors. `lower_mlir` returns the lowered module without JIT
compilation, for either target (on the CPU: the LLVM dialect).

Supported for now: static shapes with block sizes dividing the sizes they tile, and
grid loops of one of these kernel classes:

- one matmul: f16/bf16 inputs, f32 accumulation, a K loop; epilogues, casts of the
  result and transposed operands (A, B or both) included. E.g. `nn.Linear` with its
  activations fused: `x @ weight.T` (the weight stored `[out, in]`), a bias row
  broadcast over the tile's rows and any elementwise epilogue (scales, residual adds
  of the result, ReLU/LeakyReLU, sigmoid/swish, tanh, clamps, Mish) on the f32
  accumulator, rounded once at the store;
- row reductions (lighthouse's 2D class): softmax, log-softmax, L1/L2 norms and other
  multi-pass row kernels; row results may also be stored as such (`out[tile_m] =
  acc.sum(-1)`, or a `keepdim` column);
- elementwise: any number of inputs and outputs, fused unary/binary math, casts,
  broadcast rows (e.g. a bias), over 2D tiles of 16- or 32-bit elements (bf16, f16,
  f32) whose columns are a multiple of 16, or 1D tiles (gathers and scatters). bf16
  math runs in f32;
- attention, written as flash attention (lighthouse's attention class): in an inner
  `hl.tile` loop over K/V tiles, `S = Q @ K^T`, row reductions of `S` (the online
  softmax's maxima and sums), and `O = O * alpha + P @ V` accumulated across the
  loop, with `P` computed from `S`; both matmuls on f16/bf16 operands with f32
  accumulation (DPAS). Recognized by that structure, not by name. Lighthouse's
  attention layout takes K and V stored `[seq, head dim]`, a head dim of 64, query
  rows per workgroup in 16-row subgroups (at most 32) and K/V tiles of a multiple of
  32 rows. Lighthouse fuses unfused SDPA into this form in its workgroup tiling,
  which Helion's grid replaces, so unfused SDPA is not supported. KernelBench 97
  (`tests/kernelbench_xpu.py`, `attention`) uses block sizes `[1, 1, 128, 64]`
  (batch, head, query rows, K/V rows) on `[B, H, S, 64]` tiles.

Rank 3 and higher work with block size 1 on the leading dims (batch, channel, head):
unit tile dims are folded away before the checks and lighthouse's stages, so the grid
loop is its 2D kernel class and the leading dims become workgroup grid dims. E.g.
batched matmuls (`hl.dot(a[tb, tm, tk], b[tb, tk, tn], acc=acc)` with blocks
`[1, 256, 256, 32]`), a 3D tensor times a matrix (an `einsum("bmk,kn->bmn", ...)`),
NCHW activations (`[1, 1, 128, 256]`) and softmax over the last dim of `[B, H, S, D]`.
A launch has three grid dims: a fourth leading dim with more than one tile becomes a
loop in each workgroup.

Kernels outside them raise `UnsupportedOperationError` before lowering (e.g. dynamic
shapes, partial tiles, f32 matmuls, two matmuls in one grid loop, elementwise tiles of
8 or 64-bit elements or with columns that are not a multiple of 16, rank-3 tiles with a
leading block size above 1, reductions over a dim other than the last).

Elementwise kernels are not one of lighthouse's classes: the backend recognizes them
(stores of one workgroup tile, no contraction or reduction) and anchors one
workgroup layout on all their tile loads and stores. Its parameters are inferred from
the workgroup tile (the block sizes) and the element types:

- subgroup tile: dividing the workgroup tile in at most 32 subgroups, with a multiple
  of 16 columns and at most 2048 elements (8 KB of f32 registers); the one closest to
  32x32, then the squarest (a 128x256 tile: 32 subgroups of 32x32, 512 threads);
- load tile (`inst_data` of the loads): the largest 2D block load dividing it, up to
  32 rows and 32 (16-bit) or 16 (32-bit) columns;
- store tile: the largest 2D block store dividing it, `[rows <= 8, 16]`.

Row reductions take lighthouse's layout (subgroups of 8 rows of the reduced tile,
`[rows, 0]` forced). Lighthouse sets no per-instruction tiles, so the backend also
anchors the tiles the reduction loads (and its 2D tile stores) with that layout and,
by the rules above for the subgroup tile `[rows, block_n]`, the largest 2D block load
and store tiles: `[8, 16]` for f32 (8-row block reads rather than one row each),
`[8, 32]` for bf16. Stores of the row results themselves (scatter stores) get no
per-instruction tile.

`AI-bench`'s KernelBench level 1 problems lower to `xevm` at the `mlir-gpu-bench`
shapes (`tests/test_xpu_kernelbench.py`, kernels and block sizes in
`tests/kernelbench_xpu.py`): the matmuls 1, 2, 6, 7, 9, 13, 16, 17 and 18 (bf16), the
elementwise 19-22, 25 and 27-32 (bf16), and the reductions 23, 24, 38 and 39 (f32).
So do the level 2 problems with an `mlir-gpu-bench` variant, `nn.Linear` with its
epilogue in one kernel (bf16, `LEVEL2_PROBLEMS`): 9, 12, 29, 40, 59, 63, 70 and 81
(block sizes `[256, 256, 32]`, `[128, 256, 32]` for 59's 128 rows, A loaded in one
block read per subgroup tile); their sources also run on the CPU target against the
KernelBench models.

### XeGPU parameters

The workgroup tile and the K tile are the kernel's `block_sizes` (the grid tiles and
the inner K tile). The other XeGPU parameters are inferred by lighthouse (its tuned
database or cost model), or given in the config, per GPU kernel: each key is a list
with one entry per top-level `hl.tile` loop, in program order (across `hl.barrier()`
phases). An entry that is `None`, and every kernel past the end of a shorter list,
keeps lighthouse's inference.

| Key | Entry | Applies to |
|-----|-------|------------|
| `xegpu_sg_tiles` | `[sg_m, sg_n]` subgroup tile (`threads` = 16 x subgroups); `[rows, 0]` for a row reduction or attention (query rows, a multiple of 16) | GEMM, attention, reduction, 2D elementwise kernels |
| `xegpu_load_tiles` | `[a_m, a_k, b_k, b_n]`: A and B tiles loaded per instruction; `[m, n]`: the tile an elementwise kernel, or a row reduction (of the tiles it reduces), loads per instruction | GEMM, reduction, 2D elementwise kernels |
| `xegpu_prefetch_tiles` | `[a_m, a_k, b_k, b_n]`: A and B prefetch tiles, in the operand's stored orientation (`[n, k]` for a transposed B) | GEMM kernels |
| `xegpu_prefetch_depths` | `[a_nb, b_nb]`: K iterations A and B are prefetched ahead | GEMM kernels |

```python
config = helion.Config(
    block_sizes=[256, 256, 32, 128, 256, 32],  # two GEMM kernels (hl.barrier)
    xegpu_sg_tiles=[[64, 32], None],           # kernel 1: inferred
    xegpu_load_tiles=[None, [16, 16, 32, 16]],
    xegpu_prefetch_depths=[[2, 2]],            # kernel 1: inferred
)
```

Before lowering, forced parameters are checked against each kernel's shapes with
lighthouse's XeGPU constraints (`matmul_constraints`, the cost model's register and
DPAS limits; for elementwise and reduction kernels the limits of the inference
above, e.g. a reduction's load tile rows divide its subgroup rows and f32 loads take
16 columns; the prefetch keys are rejected: GEMM only; attention kernels take only
`xegpu_sg_tiles`); invalid ones, malformed entries and lists longer than the number of
kernels raise `XeGPUConfigError` (an `MLIRBackendError`, so the autotuner skips the
config). The CPU target ignores the `xegpu_*` keys, so one config can serve both
devices.

### Autotuning on the XPU

Given configs are used as they are; autotuning is an extra. The XPU search space
(`targets/xpu_search.py`) classifies each GPU kernel from Helion's facts on the
kernel (GEMM, attention, row reduction, elementwise) and narrows the block sizes Helion samples
to what XeGPU compiles: powers of two dividing the sizes they tile (no masking), at
least 16 for the blocks of a tile's contiguous dim; matmul M/N tiles of 32 to 1024
and K tiles of 16 to 64, below K; reduction rows of 8 to 256 (whole 8-row subgroups,
at most 32) and 16 to 256 columns; attention query rows of 16 to 512 and K/V tiles of
32 or 64 rows, below the sequence; 2D elementwise tiles of at most 2048 per side. The
default config infers every XeGPU parameter, with a matmul's block sizes from
lighthouse's parameter selection for its sizes and an attention kernel's from
lighthouse's attention defaults (128 query rows, 64 K/V rows), and lowers. The `xegpu_*` keys are
searched as per-kernel fragments, `None` (inferred) the default: `xegpu_sg_tiles`
(GEMM 32x32, 32x64, 64x32; reduction 16 or 32 rows; attention 32 rows; elementwise
16x32, 32x32, 32x64),
`xegpu_load_tiles` (reduction and elementwise: 8 or 32 rows at the widest block load)
and `xegpu_prefetch_depths` (GEMM: 2 and 2); `xegpu_prefetch_tiles` is not searched.
Sampled configs that cannot lower (e.g. a subgroup tile not dividing the sampled
workgroup tile, block sizes lighthouse's matmul cost model rejects, an elementwise
tile needing more than 32 subgroups) fail before any lighthouse stage with an
`MLIRBackendError` the autotuner skips. Timing the search needs an XPU: benchmarking
with XPU synchronization is not implemented yet.

Device binaries and execution need an LLVM built with the SPIR-V target and the Level
Zero runtime (`-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV -DMLIR_ENABLE_LEVELZERO_RUNNER=1`
and Python bindings, at the commit of the pinned `mlir-python-bindings`), used through
`PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core`, and a PyTorch with XPU
support. The stock bindings cannot emit SPIR-V.

## Install

The project is using [uv](https://docs.astral.sh/uv/) package manager.

`uv` can be [installed](https://docs.astral.sh/uv/getting-started/installation/) locally using:

```bash
pip install uv
```

The project can be installed using:

```bash
uv sync
```

## Requirements

The project extended Helion which targets Linux systems and requires a recent Python, PyTorch, and MLIR environment:
- Linux-based OS
- Python 3.11 or later
- [PyTorch](https://github.com/pytorch/pytorch) 2.12 or later
- [Helion](https://github.com/pytorch/helion) 1.4 or later
- [llvm/lighthouse](https://github.com/llvm/lighthouse)

For development, this repo is developed and tested against dependencies pulled by the `dev` dependency-group.
External users must supply compatible versions themselves.

## Testing

Run tests with pytest:

```bash
pytest -v
```

A whole-suite run is parallel (`pytest-xdist`, `-n auto`: one worker per
physical core) and takes about 2 minutes on 8 cores (about 8 minutes with
`-n 0`). Runs that name files or node ids, or select with `-k`, stay in one
process; pass `-n <count>` to force workers. More workers than physical cores
(`-n logical`, `-n 12`) were slower. Tests marked `isolated` run in a child of a
fork server, so a native crash fails only that test; `-m "not slow"` skips the
longest groups.

Most of the time is compilation: each kernel shape is a new compile (Helion
tracing, MLIR lowering, LLVM JIT). XPU IR tests that check both stages take
them from one lowering (`tests/harness.py`, `lower_xpu_stages`: the `xegpu-wg`
IR is the module after that stage on the way to `xevm`).

## Linting

This repo uses `pre-commit` with linter rules aligned with Helion style.

Initial config:

```bash
pip install pre-commit
pre-commit install
```

All checks can be run using:

```bash
pre-commit run -a
```

## Config variables

| Variable | Description |
|---|---|
| `HELION_MLIR_COMPILE_TIMEOUT` | Bounds MLIR lowering of each kernel, in seconds (default `30`; `0` disables the limit) |
| `HELION_MLIR_DUMP_IR` | Raw generated IR before inlining (shows `@_aten_*` helpers and tile loops) |
| `HELION_MLIR_DUMP_PRE_LOWERING` | After inlining / all pre-passes, immediately before the lighthouse lowering schedule |
| `HELION_MLIR_DUMP_LOWERED` | After lighthouse lowering (LLVM dialect, expanded memref descriptors; on the XPU, the IR at the stop point) |
| `HELION_MLIR_DUMP_STAGES` | XPU: after each stage of the XeGPU pipeline |
| `HELION_MLIR_PIPELINE=opt\|scalar` | Selects compilation pipeline (default: `opt`) |
| `HELION_MLIR_TARGET=auto\|cpu\|xpu` | Selects the device target (default: `auto`, from the kernel's tensors; an explicit `target=` of `generate_mlir`/`compile_mlir`/`lower_mlir` wins). `meta` tensors need a named target |
| `HELION_MLIR_XPU_STOP_AT=xegpu-wg\|xevm\|binary` | XPU: where lowerings stop by default (default: `binary`, the full lowering). Kernel calls then fail after lowering (nothing to run); `lower_mlir` returns the IR |

## License

MIT License - see [LICENSE](LICENSE) for details.
