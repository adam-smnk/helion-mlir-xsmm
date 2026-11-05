"""``mlir_transforms.xpu_bool_casts``: the rewrite on small IR, what it leaves alone, and the
exact kernels that aborted LLVM's SPIR-V backend, to a binary and on the device."""

from __future__ import annotations

import helion
from mlir import ir
import pytest
import torch

from tests import xpu_bool_cast_kernels as kernels
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir_transforms.xpu_bool_casts import (
    select_bool_casts,
)
from helion_mlir_backend._compiler.targets.xpu import bindings_emit_spirv


def _module(body: str, types: str = "vector<16xi1>") -> ir.Module:
    return ir.Module.parse(f"""
module {{
  func.func @host(%h: {types}) {{
    %r = arith.uitofp %h : {types} to vector<16xf32>
    return
  }}
  gpu.module @m {{
    gpu.func @k(%x: {types}) kernel {{
      {body}
      gpu.return
    }}
  }}
}}
""")


@pytest.fixture
def _context():
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        yield context


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    ("cast", "one"),
    [("uitofp", "1.000000e+00"), ("sitofp", "-1.000000e+00")],
)
def test_vector_bool_cast_becomes_a_select(cast: str, one: str) -> None:
    module = _module(f"""
      %r = arith.{cast} %x : vector<16xi1> to vector<16xf32>
      "test.use"(%r) : (vector<16xf32>) -> ()""")
    assert select_bool_casts(module) == 1
    module.operation.verify()
    text = str(module)
    kernel = text[text.index("gpu.module") :]
    assert f"arith.{cast}" not in kernel
    assert f"dense<{one}> : vector<16xf32>" in kernel
    assert "dense<0.000000e+00> : vector<16xf32>" in kernel
    assert "arith.select %arg0" in kernel or "arith.select %x" in kernel
    # Host code isn't lowered by this backend.
    assert "arith.uitofp" in text[: text.index("gpu.module")]


@pytest.mark.usefixtures("_context")
def test_bf16_result_keeps_its_type() -> None:
    module = _module("""
      %r = arith.uitofp %x : vector<16xi1> to vector<16xbf16>
      "test.use"(%r) : (vector<16xbf16>) -> ()""")
    assert select_bool_casts(module) == 1
    module.operation.verify()
    assert "dense<1.000000e+00> : vector<16xbf16>" in str(module)


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize(
    "cast",
    [
        "%r = arith.uitofp %s : i1 to f32",
        "%r = arith.uitofp %i : vector<16xi32> to vector<16xf32>",
        "%r = arith.sitofp %i : vector<16xi32> to vector<16xf32>",
        "%r = arith.extui %x : vector<16xi1> to vector<16xi32>",
    ],
    ids=["scalar", "i32-unsigned", "i32-signed", "zext"],
)
def test_other_casts_are_left_alone(cast: str) -> None:
    module = _module(f"""
      %s = "test.bool"() : () -> i1
      %i = "test.ints"() : () -> vector<16xi32>
      {cast}""")
    before = str(module)
    assert select_bool_casts(module) == 0
    assert str(module) == before


def _kernel(name: str) -> helion.Kernel:
    fn, _, blocks, _ = kernels.CASES[name]
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )


@pytest.mark.isolated
@pytest.mark.skipif(not bindings_emit_spirv(), reason="needs the SPIR-V target")
@pytest.mark.parametrize("name", list(kernels.CASES))
def test_kernel_compiles_to_a_binary(name: str) -> None:
    """Exact regression: ``VReg's low-level type and register class have different sizes``
    aborted the process in llc, at full tiles of 8 and 16 lanes. Isolated for that."""
    _, _, _, dtype = kernels.CASES[name]
    binary = lower_mlir(
        _kernel(name),
        [xpu_tensor(*kernels.SHAPES["full"], dtype=dtype)],
        target="xpu",
        stop_at="binary",
    )
    assert "gpu.binary" in str(binary)


@pytest.mark.isolated
@pytest.mark.parametrize("path", ["lighthouse", "ir"])
def test_both_layout_paths_take_the_rewrite(path: str) -> None:
    fn = kernels.a
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64], xegpu_layouts=[path]),
    )
    text = str(
        lower_mlir(kernel, [xpu_tensor(128, 256)], target="xpu", stop_at="xegpu-wg")
    )
    assert "arith.uitofp" not in text
    assert "arith.select" in text


@pytest.mark.xpu_runtime
@pytest.mark.isolated
@pytest.mark.parametrize("shape", list(kernels.SHAPES))
@pytest.mark.parametrize("name", list(kernels.CASES))
def test_kernel_on_the_device(
    name: str, shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("HELION_MLIR_TARGET", "HELION_MLIR_XPU_STOP_AT"):
        monkeypatch.delenv(variable, raising=False)
    _, ref, _, dtype = kernels.CASES[name]
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(*kernels.SHAPES[shape], generator=generator)
    x = (x * 100).to(dtype) if dtype == torch.int32 else x.to(dtype)
    expected = ref(x.double())
    # Half the elements take each side, or a wrong constant could go unseen.
    assert 0 < (x <= 0).double().mean() < 1
    out = _kernel(name)(x.to("xpu"))
    torch.xpu.synchronize()
    assert not out.isnan().any()
    error = (out.double().cpu() - expected).abs().max() / expected.abs().max()
    assert error < 2**-7
