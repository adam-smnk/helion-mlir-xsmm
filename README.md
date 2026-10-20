# helion-mlir-backend

External MLIR backend package for Helion.

Currently, only CPU execution is supported.

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
* **mlir\_pipeline** (`"opt"`, `"scalar"`):
Selects compilation pipeline:
  - `opt`: Optimizing pipeline (default). Vectorization, parallelization etc.
  - `scalar`: Scalar lowering. A simple serial fallback; mostly useful for debugging.

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

## Linting

This repo uses `pre-commit` with linter rules aligned with Helion style.

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
| `HELION_MLIR_DUMP_LOWERED` | After lighthouse lowering (LLVM dialect, expanded memref descriptors) |
| `HELION_MLIR_PIPELINE=opt\|scalar` | Selects compilation pipeline (default: `opt`) |

## License

MIT License - see [LICENSE](LICENSE) for details.
