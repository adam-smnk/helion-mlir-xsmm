"""KernelSignature: refs, roles, per-phase ins/inouts and runtime scalars."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend._compiler.mlir.analysis.signature import KernelSignature
from helion_mlir_backend._compiler.mlir.analysis.tensor_effects import TensorEffects
from helion_mlir_backend.api import _compile


def _signature(kernel: helion.Kernel, args: list[object]) -> KernelSignature:
    hf, _, env = _compile(kernel, args, None)
    with env:
        return KernelSignature.from_host_function(
            hf, TensorEffects.from_host_function(hf)
        )


def _roles(signature: KernelSignature) -> dict[str, str]:
    return {
        name: "inout" if ref.written else "in" for name, ref in signature.refs.items()
    }


def test_single_phase_refs_are_params_then_host_tensors():
    @helion.kernel(static_shapes=True)
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + y[tm, tn]
        return out

    signature = _signature(add, [torch.randn(8, 8), torch.randn(8, 8)])
    assert _roles(signature) == {"x": "in", "y": "in", "out": "inout"}
    assert [ref.tensor_param for ref in signature.refs.values()] == [0, 1, None]
    (phase,) = signature.phases
    assert phase.ins == ("x", "y")
    assert phase.inouts == ("out",)


def test_two_phase_kernel_threads_the_cross_phase_tensor():
    @helion.kernel(static_shapes=True)
    def two_phase(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        _, n = y.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            mid[tm, tn] = acc
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    signature = _signature(two_phase, [torch.randn(8, 16), torch.randn(16, 8)])
    assert _roles(signature) == {"x": "in", "y": "in", "mid": "inout", "out": "inout"}
    first, second = signature.phases
    assert (first.ins, first.inouts) == (("x", "y"), ("mid",))
    assert (second.ins, second.inouts) == (("mid",), ("out",))


def test_in_place_param_and_host_input_and_scalar():
    @helion.kernel(static_shapes=True)
    def update(x: torch.Tensor, alpha: float) -> torch.Tensor:
        scale = x.mean()
        for tile in hl.tile(x.size(0)):
            x[tile] = x[tile] * alpha + hl.load(scale, [])
        return x

    signature = _signature(update, [torch.randn(16), 2.0])
    assert _roles(signature) == {"x": "inout", "scale": "in"}
    (scalar,) = signature.scalars.values()
    assert (scalar.host_expr, scalar.dtype) == ("alpha", torch.float64)
    assert signature.phases[0].scalars == (scalar.key,)


def test_read_only_view_of_a_param_is_an_alias():
    @helion.kernel(static_shapes=True)
    def flat_copy(x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1)
        out = torch.empty_like(flat)
        for tile in hl.tile(flat.size(0)):
            out[tile] = flat[tile]
        return out

    signature = _signature(flat_copy, [torch.randn(4, 4)])
    assert signature.aliases == {"flat": "x"}
    assert _roles(signature) == {"x": "in", "out": "inout"}
    assert signature.phases[0].ins == ("x",)
