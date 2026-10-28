"""The XPU autotuning search space at the IR level (no XPU, no benchmarking):
the config spec the XPU target narrows (``targets/xpu_search.py``) for
representative KernelBench problems.

The default config and seeded random configs of the spec either lower to
annotated XeGPU (``xegpu-wg``) or are rejected before any lighthouse stage with
an error the autotuner skips; the ``xegpu_*`` keys are searched on the XPU only.
"""

from __future__ import annotations

import os
import random
from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import xpu_tensor
from tests.kernelbench_xpu import PROBLEMS
from tests.kernelbench_xpu import RANK3_PROBLEMS
from tests.kernelbench_xpu import Problem

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.support import MLIRBackendError
from helion_mlir_backend._compiler.targets import get_target
from helion_mlir_backend._compiler.targets import xpu
from helion_mlir_backend._compiler.targets.xpu_params import XEGPU_CONFIG_KEYS
from helion_mlir_backend._compiler.targets.xpu_search import KernelChoicesFragment
from helion_mlir_backend.api import _compile

_PROBLEMS = {problem.name: problem for problem in (*PROBLEMS, *RANK3_PROBLEMS)}
GEMM = "1_Square_matrix_multiplication_"
ELEMENTWISE = "19_ReLU"
REDUCTION = "23_Softmax"
SAMPLES = 6


def _args(problem: Problem) -> list[object]:
    return [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]


def _xpu_spec(problem: Problem) -> object:
    hf, _, env = _compile(problem.kernel, _args(problem), None)
    spec = env.config_spec
    get_target("xpu").adjust_config_spec(spec, hf.device_ir)
    return spec


def _samples(spec: object, count: int) -> list[helion.Config]:
    generation = spec.create_config_generation()
    random.seed(0)
    return [generation.unflatten(generation.random_flat()) for _ in range(count)]


@pytest.fixture(autouse=True)
def xpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.isolated
@pytest.mark.parametrize("name", [GEMM, ELEMENTWISE, REDUCTION])
def test_sampled_configs_lower_or_are_skipped(name: str) -> None:
    problem = _PROBLEMS[name]
    spec = _xpu_spec(problem)
    applied: list[str] = []
    original = xpu.XpuLowering.apply

    def apply(self: xpu.XpuLowering, module: object) -> object:
        applied.append("stages")
        return original(self, module)

    args = _args(problem)
    default = spec.default_config()
    assert all(
        all(entry is None for entry in default.config[key])
        for key in XEGPU_CONFIG_KEYS & default.config.keys()
    )
    outcomes = []
    configs = [default, *_samples(spec, SAMPLES)]
    # Lower in this process, so the stages run are seen here.
    with (
        mock.patch.dict(os.environ, {"HELION_MLIR_COMPILE_TIMEOUT": "0"}),
        mock.patch.object(xpu.XpuLowering, "apply", apply),
    ):
        for index, config in enumerate(configs):
            outcomes.append(_lower_or_skip(problem, args, config, index, applied))
    assert outcomes.count("lowered") >= 2, outcomes


def _lower_or_skip(
    problem: Problem,
    args: list[object],
    config: helion.Config,
    index: int,
    applied: list[str],
) -> str:
    """``lowered``, or ``skipped`` for a config rejected before lowering."""
    applied.clear()
    try:
        text = str(
            lower_mlir(
                problem.kernel, args, config=config, target="xpu", stop_at="xegpu-wg"
            )
        )
    except MLIRBackendError as error:
        assert index > 0, f"the default config {config} does not lower: {error}"
        # Rejected before any lighthouse stage, as a config the autotuner skips.
        assert not applied, error
        assert MLIRBackend().classify_autotune_exception(error) is not None
        return "skipped"
    assert applied
    assert "gpu.launch_func" in text and "#xegpu.layout" in text
    return "lowered"


def axpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2.0 + y[tile]
    return out


def test_xegpu_fragments_on_the_xpu_only() -> None:
    gemm = _xpu_spec(_PROBLEMS[GEMM])
    assert {"xegpu_sg_tiles", "xegpu_prefetch_depths"} <= set(gemm._flat_fields())
    reduction = _xpu_spec(_PROBLEMS[REDUCTION])
    fields = reduction._flat_fields()
    assert {"xegpu_sg_tiles", "xegpu_load_tiles"} <= set(fields)
    # GEMM-only keys are not searched for a reduction.
    assert "xegpu_prefetch_depths" not in fields
    assert fields["xegpu_sg_tiles"].choices == ((None, [16, 0], [32, 0]),)

    x, y = torch.randn(64, 64), torch.randn(64, 64)
    hf, _, env = _compile(helion.kernel(backend="mlir")(axpy), [x, y], None)
    cpu = env.config_spec
    before = [(b.min_size, b.max_size) for b in cpu.block_sizes]
    get_target("cpu").adjust_config_spec(cpu, hf.device_ir)
    assert not XEGPU_CONFIG_KEYS & set(cpu._flat_fields())
    assert not XEGPU_CONFIG_KEYS & cpu.default_config().config.keys()
    assert [b.max_size for b in cpu.block_sizes] == [high for _, high in before]


@pytest.mark.parametrize("name", [ELEMENTWISE, REDUCTION, GEMM])
def test_block_sizes_divide_and_fill_block_rows(name: str) -> None:
    """No masking (every block size divides its size); elementwise columns are
    a multiple of 16 and reduction rows of the 8-row subgroups."""
    problem = _PROBLEMS[name]
    spec = _xpu_spec(problem)
    sizes = [block.size_hint for block in spec.block_sizes]
    for config in [spec.default_config(), *_samples(spec, 200)]:
        blocks = config.config["block_sizes"]
        assert all(size % block == 0 for size, block in zip(sizes, blocks, strict=True))
        if name == ELEMENTWISE:
            rows, cols = blocks
            assert cols % 16 == 0 and cols <= 2048 and rows <= 2048
        elif name == REDUCTION:
            cols, rows = blocks  # register_block_size(n) comes first
            assert rows % 8 == 0 and rows <= 256 and cols % 16 == 0
        else:
            m, n, k = blocks
            assert m >= 32 and n >= 32 and 16 <= k <= 64


def test_gemm_default_is_lighthouses_selection() -> None:
    spec = _xpu_spec(_PROBLEMS[GEMM])
    assert spec.default_config().config["block_sizes"] == [256, 256, 32]


def test_kernel_choices_fragment() -> None:
    fragment = KernelChoicesFragment(((None, [8, 0]), (None, [32, 32], [16, 32])))
    assert fragment.default() == [None, None]
    assert fragment.dim() == 5
    # Configs read back from JSON hold lists; tuples are equal entries.
    assert fragment.encode([(8, 0), [16, 32]]) == [0.0, 1.0, 0.0, 0.0, 1.0]
    neighbors = fragment.pattern_neighbors([None, [32, 32]])
    assert neighbors == [[[8, 0], [32, 32]], [None, None], [None, [16, 32]]]
    random.seed(0)
    for _ in range(20):
        value = fragment.random()
        assert value[0] in (None, [8, 0]) and value[1] in fragment.choices[1]
        mutated = fragment.differential_mutation(value, [None, None], [None, None])
        assert mutated == value
    assert fragment.fingerprint() != KernelChoicesFragment(((None,),)).fingerprint()


@pytest.mark.isolated
@pytest.mark.parametrize("name", [problem.name for problem in RANK3_PROBLEMS])
def test_leading_blocks_are_unit(name: str) -> None:
    """Rank-3+ kernels search block size 1 on their leading dims (where the
    hand-picked configs have it), and their class's bounds on the others."""
    problem = _PROBLEMS[name]
    spec = _xpu_spec(problem)
    unit = [size == 1 for size in problem.block_sizes]
    assert [block.max_size == 1 for block in spec.block_sizes] == unit
    default = spec.default_config()
    assert [size == 1 for size in default.config["block_sizes"]] == unit
    if name.startswith("10_"):
        # An einsum has no matmul fact: no GEMM bounds or default (generic
        # bounds; such configs are rejected before lowering).
        return
    text = str(
        lower_mlir(
            problem.kernel,
            _args(problem),
            config=default,
            target="xpu",
            stop_at="xegpu-wg",
        )
    )
    assert "#xegpu.layout" in text
