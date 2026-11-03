"""The workarounds of IGC miscompiles, checked where they act: on the binaries
(IGC's own dumps, no device) and on the device. Causes and reproducers are in
helion-mlir-dev's ``agent-wiki/upstream-repro/xpu-nan/README.md``.

- ``IGC_OPTIONS``'s ``VectorizerAllowCMP=0``: ``IGCVectorizer`` rebuilds a
  compare and select as a vector one, and IGC emits that as a size-1 min/max
  into a vector: the other elements read stale registers. The kernels here are
  deterministic: they are wrong on any run with it (the row value stays in the
  result), unlike 97, whose output hides the wrong maximum unless the stale
  value is huge (``test_attention_row_maxima_after_other_kernels``).
- ``xevm_unit_clamps``: IGC takes a clamp to ``[0, 1]`` for ``(sat)``: NaN gives
  0, -0 gives +0 and a bf16 ``clamp_max(clamp_min(x, 0.0), 1.0)`` does nothing.
"""

from __future__ import annotations

import math
import pathlib
import re
import shutil

import helion
import helion.language as hl
import pytest
import torch
import torch.nn.functional as F

from tests import kernelbench_xpu as kb
from tests import xpu_math_full_kernels as full_kernels
from tests import scalar_compare_kernels as compares
from tests import xpu_partial_kernels as pk
from tests.harness import xpu_tensor
from tests.igc_dumps import elements_read_never_written
from tests.igc_dumps import igc_dump_problems
from tests.igc_dumps import vector_intrinsics
from tests.igc_dumps import narrow_minmax

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler import xpu_transforms
from helion_mlir_backend._compiler.targets import xpu as xpu_target
from helion_mlir_backend._compiler.targets.xpu import bindings_emit_spirv

pytestmark = pytest.mark.isolated
BINARIES = pytest.mark.skipif(
    not bindings_emit_spirv() or shutil.which("ocloc") is None,
    reason="needs the SPIR-V target and ocloc",
)
DEVICE = pytest.mark.xpu_runtime
BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32

# --- Kernels: a row value compared with another one ----------------------------


def running_maximum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """97's flash attention (``kernelbench_xpu.attention``) storing its running
    row maximum, which the real output cancels out (NaN rows stay visible)."""
    batch, heads, m, d = q.shape
    n = k.size(2)
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q, dtype=torch.float32)
    for tb, th, tm in hl.tile([batch, heads, m]):
        q_tile = q[tb, th, tm, :]
        row_max = hl.full([tb, th, tm], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tb, th, tm], dtype=torch.float32)
        acc = hl.zeros([tb, th, tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
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
        r = acc / row_sum[:, :, :, None]
        out[tb, th, tm, :] = torch.where(
            r == r,
            row_max[:, :, :, None] + torch.zeros_like(r),
            row_max[:, :, :, None] - 1000.0,
        )
    return out


def l1_where(x: torch.Tensor) -> torch.Tensor:
    """Two row sums through ``torch.where``."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        top = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
            top = top + x[tile_m, tile_n] * x[tile_m, tile_n]
        a = acc.sum(-1, keepdim=True)
        b = top.sum(-1, keepdim=True)
        mean = torch.where(a > b, a, b)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / mean
    return out


def l1_clamp(x: torch.Tensor) -> torch.Tensor:
    """A row sum through a clamp."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        mean = torch.clamp(acc.sum(-1, keepdim=True), 1.0, 1e6)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / mean
    return out


def _l1_where_reference(x: torch.Tensor) -> torch.Tensor:
    a, b = x.abs().sum(-1, keepdim=True), (x * x).sum(-1, keepdim=True)
    return x / torch.where(a > b, a, b)


def _l1_clamp_reference(x: torch.Tensor) -> torch.Tensor:
    return x / x.abs().sum(-1, keepdim=True).clamp(1.0, 1e6)


L1_KERNELS = {
    "where": (l1_where, _l1_where_reference),
    "clamp": (l1_clamp, _l1_clamp_reference),
}
"""Shape (256, 256) and [64, 64] blocks, 38_L1Norm's: the regressions' config."""
L1_SHAPE = (256, 256)
L1_BLOCKS = [64, 64]


def _attention_problem() -> kb.Problem:
    return next(p for p in kb.RANK3_PROBLEMS if p.name.startswith("97"))


def _kernel(fn: object, blocks: list[int], **keys: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, **keys),
    )


# --- Kernels: clamps to [0, 1] -------------------------------------------------


def clamp_ints(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(x[tile], 0, 1)
    return out


def clamp(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(x[tile], 0.0, 1.0)
    return out


def method(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile].clamp(0.0, 1.0)
    return out


def clip(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = x[tile].clip(0.0, 1.0)
    return out


def hardtanh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = F.hardtanh(x[tile], 0.0, 1.0)
    return out


def min_then_max(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp_max(torch.clamp_min(x[tile], 0.0), 1.0)
    return out


def max_then_min(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp_min(torch.clamp_max(x[tile], 1.0), 0.0)
    return out


def scaled(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(x[tile] * 2.0, 0.0, 1.0)
    return out


def hardtanh_of_relu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """IGC folds hardtanh's select with -1: a pair only after its optimizer."""
    for tile in hl.tile(x.shape):
        out[tile] = F.hardtanh(F.relu(x[tile]))
    return out


def clamp_of_clamp(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A guarded maximum between the selects, which IGC folds."""
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(
            torch.clamp(x[tile], min=torch.zeros_like(x[tile])), max=1.0
        )
    return out


def other_range(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(x[tile], -1.0, 2.0)
    return out


UNIT_CLAMPS = {
    "clamp_ints": (clamp_ints, lambda x: torch.clamp(x, 0, 1)),
    "clamp": (clamp, lambda x: torch.clamp(x, 0.0, 1.0)),
    "method": (method, lambda x: x.clamp(0.0, 1.0)),
    "clip": (clip, lambda x: x.clip(0.0, 1.0)),
    "hardtanh": (hardtanh, lambda x: F.hardtanh(x, 0.0, 1.0)),
    "min_then_max": (
        min_then_max,
        lambda x: torch.clamp_max(torch.clamp_min(x, 0.0), 1.0),
    ),
    "max_then_min": (
        max_then_min,
        lambda x: torch.clamp_min(torch.clamp_max(x, 1.0), 0.0),
    ),
    "scaled": (scaled, lambda x: torch.clamp(x * 2.0, 0.0, 1.0)),
    "hardtanh_of_relu": (hardtanh_of_relu, lambda x: F.hardtanh(F.relu(x))),
    "clamp_of_clamp": (
        clamp_of_clamp,
        lambda x: torch.clamp(torch.clamp(x, min=torch.zeros_like(x)), max=1.0),
    ),
}
"""Each kernel and its torch reference."""
CLAMP_SHAPE = (64, 512)
CLAMP_BLOCKS = [32, 256]
"""[32, 256] tiles of 64x512: the regressions' config."""
DTYPES = {"bf16": BF16, "f16": F16, "f32": F32}


def _unit_clamp(name: str) -> helion.Kernel:
    return _kernel(UNIT_CLAMPS[name][0], CLAMP_BLOCKS)


def _clamp_inputs(dtype: torch.dtype) -> torch.Tensor:
    """The special values, both signs of the bounds, the numbers next to them."""
    info = torch.finfo(dtype)
    values = [
        math.nan,
        math.inf,
        -math.inf,
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        1.0 + info.eps,
        1.0 - info.eps / 2,
        1.5,
        -1.5,
        -3.375,
        3.0,
        info.tiny,
        -info.tiny,
        info.max,
        -info.max,
    ]
    table = torch.tensor(values, dtype=torch.float64).to(dtype)
    size = CLAMP_SHAPE[0] * CLAMP_SHAPE[1]
    return table.repeat(size // len(values) + 1)[:size].reshape(CLAMP_SHAPE)


def _assert_same_values(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The same NaN pattern, the other elements bit for bit."""
    assert out.dtype == ref.dtype and out.shape == ref.shape
    nan = torch.isnan(ref.float())
    wrong_nan = torch.isnan(out.float()) != nan
    as_int = torch.int16 if out.element_size() == 2 else torch.int32
    wrong = (out.view(as_int) != ref.view(as_int)) & ~nan | wrong_nan
    assert not wrong.any(), f"{int(wrong.sum())} of {wrong.numel()} wrong"


# --- The binaries (IGC's dumps) ------------------------------------------------


def _igc_dump(
    monkeypatch: pytest.MonkeyPatch,
    directory: pathlib.Path,
    kernel: helion.Kernel,
    args: list[object],
    config: helion.Config | None = None,
) -> pathlib.Path:
    """``kernel``'s binary lowering with IGC's dumps in ``directory``."""
    monkeypatch.delenv("HELION_MLIR_XPU_STOP_AT", raising=False)
    monkeypatch.setenv("IGC_ShaderDumpEnable", "1")
    monkeypatch.setenv("IGC_DumpToCustomDir", str(directory))
    assert "gpu.binary" in str(
        lower_mlir(kernel, args, config=config, target="xpu", stop_at="binary")
    )
    assert list(directory.glob("*.visaasm")), "IGC wrote no vISA"
    return directory


def _problem(prefix: str) -> tuple[helion.Kernel, list[object], helion.Config]:
    problems = (*kb.PROBLEMS, *kb.RANK3_PROBLEMS, *kb.LEVEL2_PROBLEMS)
    problem = next(p for p in problems if p.name.startswith(prefix))
    tensors = [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]
    config = helion.Config(block_sizes=list(problem.block_sizes), **problem.xegpu)
    return problem.kernel, problem.args(*tensors), config


SCANNED_PROBLEMS = [
    problem.name for problem in (*kb.PROBLEMS, *kb.RANK3_PROBLEMS, *kb.LEVEL2_PROBLEMS)
]
"""Every benchmarked XPU problem, as each compiles in its own way."""


@BINARIES
@pytest.mark.parametrize("prefix", SCANNED_PROBLEMS)
def test_problems_have_no_narrow_minmax_and_use_all_registers(
    prefix: str, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Every binary keeps the 256 registers (a second option in ``opts``'s one
    pair of quotes silently loses them, no error) and reaches IGC with the
    workaround; nothing that shows a known miscompile: no size-1 min/max into a
    vector, no vector compare, no use before its def.

    Only the min/max is checked, not all uninitialized reads: seven level 2
    kernels read uninitialized rows of the vectors they then insert into, each
    overwritten before anything reads it."""
    kernel, args, config = _problem(prefix)
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args, config)
    for visaasm in dump.glob("*.visaasm"):
        assert "-TotalGRFNum 256" in visaasm.read_text(errors="replace"), visaasm.name
    assert _dump_problems(dump) == []
    _assert_options_in_force(dump, off=False)


def _assert_options_in_force(dump: pathlib.Path, *, off: bool) -> None:
    """IGC's command line has the option: one that reaches only ``_options.txt`` is
    lost (a tool preloaded into ``ocloc``, a libocloc build of LLVM, ...); no vector
    compare, and no vectorized code at all in an ``"off"`` binary."""
    options = [path.read_text() for path in dump.glob("*_options.txt")]
    assert any("-ze-opt-large-register-file" in text for text in options)
    wanted = "EnableVectorizer=0" if off else "VectorizerAllowCMP=0"
    other = "VectorizerAllowCMP=0" if off else "EnableVectorizer=0"
    commands = [path.read_text(errors="replace") for path in dump.glob("*_cmd.txt")]
    assert any(wanted in text for text in commands)
    assert not any(other in text for text in commands)
    assert all("DisableRecompilation=1" in text for text in commands)
    optimized = list(dump.glob("*_optimized.ll"))
    assert optimized
    for path in optimized:
        unwanted = "vectorized_" if off else "vectorized_cmp"
        assert unwanted not in path.read_text(errors="replace"), path.name


@BINARIES
@pytest.mark.parametrize("vectorizer", ["off", "cmp", None])
def test_vectorizer_choice_reaches_igc(
    vectorizer: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """``igc_vectorizer``: the kernel's choice on IGC's command line, ``"cmp"`` the
    default, on a kernel with no other reason to be checked."""
    problem = next(p for p in kb.PROBLEMS if p.name.startswith("1_Square"))
    config = helion.Config(
        block_sizes=problem.block_sizes,
        **problem.xegpu,
        **({"igc_vectorizer": [vectorizer]} if vectorizer else {}),
    )
    args = [xpu_tensor(*shape, dtype=problem.dtype) for shape in problem.shapes]
    dump = _igc_dump(monkeypatch, tmp_path, problem.kernel, args, config)
    _assert_options_in_force(dump, off=vectorizer == "off")


@BINARIES
@pytest.mark.parametrize("vectorizer", ["off", None], ids=["off", "cmp"])
def test_a_spilling_kernel_ships_no_retry(
    vectorizer: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """IGC's retry after a spill ships a second binary (``_1`` in the dumps), larger
    and slower: sibling GEMMs at partial tiles retry under both settings without
    ``DisableRecompilation=1``, 70 % and 12 % slower."""
    case = pk.CASES["sibling_gemms_k"]
    config = helion.Config(
        block_sizes=case.blocks,
        **case.config,
        **({"igc_vectorizer": [vectorizer]} if vectorizer else {}),
    )
    args = [xpu_tensor(*shape, dtype=dtype) for shape, dtype, _ in case.inputs]
    args.append(xpu_tensor(*case.output[0], dtype=case.output[1]))
    kernel = helion.kernel(case.fn, backend="mlir", static_shapes=True)
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args, config)
    assert list(dump.glob("*.visaasm"))
    assert not [path.name for path in dump.glob("*_1_simd*.visaasm")]


def _two_elementwise(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(x.shape):
        a[tile] = torch.exp(x[tile].to(F32)) * 3.0
    hl.barrier()
    for tile in hl.tile(x.shape):
        b[tile] = torch.exp(x[tile].to(F32)) * 5.0


_THREE = re.compile(r"3\.000000e\+00|0x4008000000000000")
_FIVE = re.compile(r"5\.000000e\+00|0x4014000000000000")


@BINARIES
@pytest.mark.parametrize(
    "choices", [["off", None], [None, "off"]], ids=["first", "second"]
)
def test_vectorizer_choice_is_per_kernel(
    choices: list[str | None], monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Each GPU module compiles with its own options: the module multiplying by 3 or
    by 5, told by its constant in IGC's IR, has its entry's key on IGC's command line."""
    config = helion.Config(block_sizes=[64, 64, 64, 64], igc_vectorizer=choices)
    kernel = helion.kernel(_two_elementwise, backend="mlir", static_shapes=True)
    args = [xpu_tensor(256, 256, dtype=F32) for _ in range(3)]
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args, config)
    seen = {}
    for command in dump.glob("*_cmd.txt"):
        codegen = pathlib.Path(str(command)[: -len("_cmd.txt")] + "_codegen.ll")
        if not codegen.exists():
            continue
        text = codegen.read_text(errors="replace")
        module = 0 if _THREE.search(text) else 1 if _FIVE.search(text) else None
        if module is None:
            continue
        keys = command.read_text(errors="replace")
        seen[module] = (
            "off" if "EnableVectorizer" in keys else None,
            "VectorizerAllowCMP" in keys,
        )
    assert set(seen) == {0, 1}
    for module, choice in enumerate(choices):
        assert seen[module] == (choice, choice is None), module


def _lost_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """As a tool that drops ``-igc_opts`` does."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    for name in ("IGC_OPTIONS", "IGC_OPTIONS_OFF"):
        monkeypatch.setattr(xpu_transforms, name, "-ze-opt-large-register-file")


def _attention_kernel(**config: object) -> tuple[helion.Kernel, list[object]]:
    problem = _attention_problem()
    kernel = helion.kernel(
        kb.attention,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=problem.block_sizes, **config),
    )
    return kernel, [xpu_tensor(2, 2, 512, 64, dtype=BF16) for _ in range(3)]


def _dump_problems(dump: pathlib.Path) -> list[str]:
    return igc_dump_problems(sorted(dump.iterdir()))


@BINARIES
def test_lost_igc_options_show_in_the_binary_with_nan_ignoring_maxima(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Without the options these kernels gave up to half of their rows NaN (a default
    ``unitrace`` drops ``-igc_opts``). The scan finds it, so the tests below say
    something."""
    _lost_options(monkeypatch)
    kernel, args = _attention_kernel(xegpu_nan_ignoring_maxima=[True])
    assert _dump_problems(_igc_dump(monkeypatch, tmp_path, kernel, args)) != []


@BINARIES
def test_lost_igc_options_do_not_matter_without_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The negative: NaN-propagating maxima do not take that path."""
    _lost_options(monkeypatch)
    kernel, args = _attention_kernel()
    assert _dump_problems(_igc_dump(monkeypatch, tmp_path, kernel, args)) == []


def _rounded_sine(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A scalar builtin's result straight into a conversion to 16 bits."""
    for tile in hl.tile(x.shape):
        out[tile] = torch.sin(x[tile].to(F32)).to(out.dtype)
    return out


@BINARIES
def test_the_vectorizer_needs_the_lane_conversions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The default, ``"cmp"``, on the form of ``VectorizerAllowFPTRUNC``'s bug: a use
    before its def in IGC's IR, stale registers in the output, unless
    ``xevm_lane_conversions`` runs. If IGC no longer shows it without the rewrite, the
    rewrite's removal condition may hold: see that module's docstring. The negative:
    ``"off"`` without the rewrite."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    kernel = helion.kernel(_rounded_sine, backend="mlir", static_shapes=True)
    args = [xpu_tensor(64, 512, dtype=BF16), xpu_tensor(64, 512, dtype=BF16)]
    rewrite = xpu_target.convert_lanes_where_computed
    for choice, rewritten, wrong in (
        ("cmp", False, True),
        ("cmp", True, False),
        ("off", False, False),
    ):
        directory = tmp_path / f"{choice}-{rewritten}"
        directory.mkdir()
        monkeypatch.setattr(
            xpu_target,
            "convert_lanes_where_computed",
            rewrite if rewritten else lambda module: 0,
        )
        config = helion.Config(block_sizes=[32, 256], igc_vectorizer=[choice])
        dump = _igc_dump(monkeypatch, directory, kernel, args, config)
        assert bool(_dump_problems(dump)) == wrong, (choice, rewritten)


_ROUNDED = ["sin", "log", "powf", "f64_sin"]
"""Kernels of ``fptrunc/``'s scope: a builtin with branches rounded straight to 16 bits,
and ``sin`` in f64 narrowed to f32 and then to 16 bits."""
_ROUNDED_DTYPES = [(BF16, BF16), (F32, BF16), (F16, F16)]
_ROUNDED_TILES = [((16, 8192), [1, 8192]), ((256, 512), [128, 256])]
"""1D and 2D tiles: the conversion of a slice in the latter."""


@BINARIES
@pytest.mark.parametrize("dtypes", _ROUNDED_DTYPES, ids=str)
@pytest.mark.parametrize(("shape", "blocks"), _ROUNDED_TILES, ids=["1d", "2d"])
@pytest.mark.parametrize("name", _ROUNDED)
def test_builtins_rounded_to_16_bits_have_no_use_before_def(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    name: str,
    shape: tuple[int, int],
    blocks: list[int],
    dtypes: tuple[torch.dtype, torch.dtype],
) -> None:
    """The exact kernels of ``xevm_lane_conversions``'s regressions."""
    kernel = helion.kernel(
        getattr(full_kernels, f"cast_{name}"),
        backend="mlir",
        static_shapes=True,
    )
    args = [xpu_tensor(*shape, dtype=dtypes[0]), xpu_tensor(*shape, dtype=dtypes[1])]
    config = helion.Config(block_sizes=blocks)
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args, config)
    # Narrowing f64: IGCVectorizer's fptrunc of f64 gives wrong rows (_narrows_f64).
    _assert_options_in_force(dump, off=name == "f64_sin")
    assert _dump_problems(dump) == []


def test_igc_dump_problems(tmp_path: pathlib.Path) -> None:
    """The check itself: a vector compare in IGC's final IR, a use before its def in
    its optimized IR (fptrunc before the inserts), and no vISA at all."""
    assert igc_dump_problems([]) == ["IGC wrote no vISA to check"]
    visa = tmp_path / "k.visaasm"
    visa.write_text(".decl V1 v_type=G type=f num_elts=1 align=dword\n")
    assert igc_dump_problems([visa]) == []
    codegen = tmp_path / "k_codegen.ll"
    codegen.write_text("%c = fcmp ogt <8 x float> %a, %b\n")
    assert igc_dump_problems([visa, codegen]) == ["k_codegen.ll: fcmp ogt <8 x float>"]
    codegen.write_text("%c = fcmp ogt float %a, %b\n")
    assert igc_dump_problems([visa, codegen]) == []
    optimized = tmp_path / "k_optimized.ll"
    optimized.write_text(
        "define void @k() {\n"
        "entry:\n"
        "  %c = fptrunc <8 x float> %v to <8 x bfloat>\n"
        "  %v = insertelement <8 x float> %u, float %f, i32 7\n"
        "  br label %next\n"
        "next:\n"
        "  %p = phi float [ %q, %next ]\n"
        "  %q = fadd float %p, %p\n"
        "  ret void\n"
        "}\n"
    )
    assert igc_dump_problems([visa, optimized]) == [
        "k_optimized.ll: %c = fptrunc <8 x float> %v to <8 x bfloat>"
    ]


@BINARIES
def test_the_scan_finds_a_narrow_minmax(tmp_path: pathlib.Path) -> None:
    """The check itself, on the instructions of the wrong binaries."""
    visaasm = tmp_path / "k.visaasm"
    visaasm.write_text(
        ".decl V0081 v_type=G type=f num_elts=8 align=GRF\n"
        ".decl V0082 v_type=G type=f num_elts=1 align=dword\n"
        ".decl V0083 v_type=G type=f num_elts=1 alias=<V0081, 0>\n"
        "max (M1_NM, 1) V0081(0,0)<1> V0075(0,0)<0;1,0> V0079(0,0)<0;1,0> /// $1\n"
        "max (M1_NM, 1) V0082(0,0)<1> V0075(0,0)<0;1,0> V0079(0,0)<0;1,0>\n"
        "min (M1_NM, 1) V0083(0,0)<1> V0075(0,0)<0;1,0> V0079(0,0)<0;1,0>\n"
        "max (M1, 16) V0084(0,0)<1> V0075(0,0)<1;1,0> V0079(0,0)<1;1,0>\n"
    )
    assert len(narrow_minmax(visaasm)) == 2


_LANE_KERNELS: dict[str, object] = {
    **{name: fn for name, (fn, _) in compares.COMBINED.items()},
    **compares.INT_ABS_KERNELS,
}


@BINARIES
@pytest.mark.parametrize("name", list(_LANE_KERNELS))
def test_compare_kernels_leave_igc_nothing_to_fold(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The check of the whole class: no vector ``llvm.*`` call but those IGC writes whole, and
    no element read that was never written (``llvm.fabs.v16f32``, ``llvm.abs.v16i32``: one
    row of 16 written). ``_dump_problems``, on every problem's binary too, also has no vector
    compare left: the static form of ``xevm_lane_compares``'s promise."""
    dtype = torch.int32 if name in compares.INT_ABS_KERNELS else torch.float32
    kernel = _kernel(_LANE_KERNELS[name], [32, 32])
    dump = _igc_dump(monkeypatch, tmp_path, kernel, [xpu_tensor(128, 256, dtype=dtype)])
    assert _dump_problems(dump) == []
    for visaasm in dump.glob("*.visaasm"):
        assert elements_read_never_written(visaasm) == [], visaasm.name


def test_the_scan_finds_a_vector_fabs_and_a_row_never_written(
    tmp_path: pathlib.Path,
) -> None:
    """On the vISA and final IR the reproducer's ``idiom`` kernel has without the stage."""
    codegen = tmp_path / "k_codegen.ll"
    codegen.write_text(
        "  %1 = call <16 x float> @llvm.fabs.v16f32(<16 x float> %0)\n"
        "  %2 = call <16 x i32> @llvm.umin.v16i32(<16 x i32> %a, <16 x i32> %b)\n"
        "  %3 = call <16 x i32> @llvm.genx.GenISA.x.v16i32(i32 0)\n"
        "  %4 = call float @llvm.fabs.f32(float %x)\n"
    )
    assert vector_intrinsics(codegen) == ["fabs.v16f32"]
    visaasm = tmp_path / "k.visaasm"
    visaasm.write_text(
        ".decl V0089 v_type=G type=f num_elts=256 align=wordx32\n"
        ".decl V0090 v_type=G type=f num_elts=256 align=wordx32\n"
        ".decl V0092 v_type=G type=f num_elts=16 align=wordx32\n"
        "    raw_send (M1, 16) V0089(0,0)<1>\n"
        "    mov (M1, 16) V0090(0,0)<1> (abs)V0089(0,0)<1;1,0>\n"
        "    add (M1, 16) V0092(0,0)<1> V0090(0,0)<1;1,0> V0090(1,0)<1;1,0>\n"
    )
    found = elements_read_never_written(visaasm)
    assert len(found) == 1
    assert found[0].startswith("V0090")


@BINARIES
@pytest.mark.parametrize("name", list(L1_KERNELS))
def test_row_values_compared_have_no_narrow_minmax(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Regression (exact kernels and config): ``max (M1_NM, 1)`` into 8 elements."""
    kernel = _kernel(L1_KERNELS[name][0], L1_BLOCKS)
    args = [xpu_tensor(*L1_SHAPE)]
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args)
    visas = list(dump.glob("*.visaasm"))
    assert all(narrow_minmax(visaasm) == [] for visaasm in visas)
    selects = [p.read_text() for p in dump.glob("*_optimized.ll")]
    assert not any("vectorized_select" in text for text in selects)


@BINARIES
def test_running_maximum_has_no_narrow_minmax(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    problem = _attention_problem()
    kernel = _kernel(running_maximum, problem.block_sizes)
    args = [xpu_tensor(2, 2, 512, 64, dtype=BF16) for _ in range(3)]
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args)
    assert all(narrow_minmax(visaasm) == [] for visaasm in dump.glob("*.visaasm"))


@BINARIES
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("name", list(UNIT_CLAMPS))
def test_unit_clamps_are_not_saturations(
    name: str, dtype: str, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Regression: ``(sat)``, on bf16 also ``mov (sat) :uw :uw``, which does
    nothing."""
    args = [xpu_tensor(*CLAMP_SHAPE, dtype=DTYPES[dtype]) for _ in range(2)]
    dump = _igc_dump(monkeypatch, tmp_path, _unit_clamp(name), args)
    for asm in dump.glob("*_simd*_entry_*.asm"):
        assert "(sat)" not in asm.read_text(errors="replace"), asm.name


@BINARIES
def test_other_clamps_keep_their_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The workaround is for compares with 1: other bounds keep theirs."""
    kernel = _kernel(other_range, CLAMP_BLOCKS)
    args = [xpu_tensor(*CLAMP_SHAPE) for _ in range(2)]
    dump = _igc_dump(monkeypatch, tmp_path, kernel, args)
    asm = "".join(
        path.read_text(errors="replace") for path in dump.glob("*_simd*_entry_*.asm")
    )
    assert "0x40000000" in asm  # compares with 2 itself
    assert "0x40000001" not in asm


# --- The device ----------------------------------------------------------------


@DEVICE
def test_running_maximum_is_right() -> None:
    """Regression: wrong in 1792 of 2048 rows, always rows 1-7 and 9-15 of 16
    (with no kernel run before it: their stale registers held 0.0)."""
    problem = _attention_problem()
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 512, 64, dtype=BF16, device="xpu") for _ in range(3))
    reference = (q.float().cpu() @ k.float().cpu().transpose(-1, -2) / 8).amax(-1)
    out = _kernel(running_maximum, problem.block_sizes)(q, k, v).float().cpu()
    rows = out[..., 0]
    # Not "> tol": a stale NaN row, the original failure, must count.
    wrong = ~((rows - reference).abs() <= 1e-5 * reference.abs().amax())
    assert not wrong.any(), f"{int(wrong.sum())} of {wrong.numel()} rows wrong"


@DEVICE
@pytest.mark.parametrize("name", list(L1_KERNELS))
def test_row_values_compared_are_right(name: str) -> None:
    """Regression (exact kernels and config): 224 of 256 rows wrong by ``l1_clamp``
    (the rows at positions 1-7 and 9-15 of 16)."""
    fn, reference = L1_KERNELS[name]
    torch.manual_seed(0)
    x = torch.randn(*L1_SHAPE, dtype=F32)
    ref = reference(x)
    out = _kernel(fn, L1_BLOCKS)(x.to("xpu")).cpu()
    assert torch.isfinite(out).all()
    wrong = ((out - ref).abs() > 1e-5 * ref.abs().amax()).any(-1)
    assert not wrong.any(), f"{int(wrong.sum())} of {wrong.numel()} rows wrong"


@DEVICE
@pytest.mark.parametrize("dtype", [BF16, F16, F32], ids=list(DTYPES))
def test_clamp_chain_clamps(dtype: torch.dtype) -> None:
    """Regression (exact kernel and config, bf16): ``clamp_max(clamp_min(x, 0.0),
    1.0)`` returned its input: 28528 of 32768 wrong, e.g. -3.375."""
    torch.manual_seed(0)
    x = (torch.randn(*CLAMP_SHAPE) * 3).to(dtype)
    kernel = _unit_clamp("min_then_max")
    out = kernel(x.to("xpu"), torch.empty_like(x, device="xpu")).cpu()
    _assert_same_values(out, torch.clamp_max(torch.clamp_min(x, 0.0), 1.0))


@DEVICE
@pytest.mark.parametrize("dtype", [BF16, F16, F32], ids=list(DTYPES))
@pytest.mark.parametrize("name", list(UNIT_CLAMPS))
def test_unit_clamps_match_torch(name: str, dtype: torch.dtype) -> None:
    """Regression (bf16 and f32): NaN gave 0 and -0 gave +0, against torch on the
    CPU, which keeps both. ``hardtanh_of_relu`` and ``clamp_of_clamp`` (f32 and
    f16): NaN gave 0 in 1368 elements. For those two the zero's sign is not
    compared: separate open items, also on bounds other than 1 (our relu gives +0
    for -0, and a clamp with a tensor bound keeps -0 where torch gives +0)."""
    x = _clamp_inputs(dtype)
    kernel = _unit_clamp(name)
    out = kernel(x.to("xpu"), torch.empty_like(x, device="xpu")).cpu()
    ref = UNIT_CLAMPS[name][1](x)
    if name in ("hardtanh_of_relu", "clamp_of_clamp"):
        out, ref = out + 0.0, ref + 0.0
    _assert_same_values(out, ref)


def _paint(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Leaves ``x``'s value in the registers a later kernel reads if it is wrong."""
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * 1.0 + x[tile] * 1.0 - x[tile]
    return out


@DEVICE
@pytest.mark.parametrize("dtypes", _ROUNDED_DTYPES, ids=str)
@pytest.mark.parametrize(("shape", "blocks"), _ROUNDED_TILES, ids=["1d", "2d"])
def test_f64_narrowed_to_16_bits_matches_fp64(
    shape: tuple[int, int],
    blocks: list[int],
    dtypes: tuple[torch.dtype, torch.dtype],
) -> None:
    """Regression: ``fptrunc <8 x double>`` before its operand, so the output was what
    earlier kernels left, up to 130037 of 131072 elements off by over 1 %."""
    source, target = dtypes
    kernel = _kernel(full_kernels.cast_f64_sin, blocks)
    paint = _kernel(_paint, [32, 256])
    generator = torch.Generator().manual_seed(0)
    x = (torch.rand(shape, generator=generator) * 0.8 + 0.1).to(source)
    ref = torch.sin(x.double()).to(torch.float32).double()
    on_device = x.to("xpu")
    first = None
    for value in (1234.5, math.nan, 3e38):
        filled = torch.full((64, 512), value, device="xpu")
        paint(filled, torch.empty(64, 512, device="xpu"))
        torch.xpu.synchronize()
        out = kernel(on_device, torch.empty(shape, dtype=target, device="xpu")).cpu()
        assert not out.isnan().any()
        error = (out.double() - ref).abs().max() / ref.abs().max()
        assert error <= 2**-6, (value, float(error))
        first = out if first is None else first
        assert torch.equal(out, first), value
