"""What the full key of an mlir kernel tells apart beyond stock Helion: ints in
closure cells by value and ``hl.constexpr`` by type and bits. Each of these ran an
earlier compile silently, on stock Helion too
(agent-wiki/upstream-repro/helion-function-key). Marked ``helion_internals``.
"""

from __future__ import annotations

import dataclasses
import math

import helion
import helion.language as hl
import numpy as np
import pytest
import torch

import helion_mlir_backend
from helion_mlir_backend._compiler import native

pytestmark = pytest.mark.helion_internals


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[8])
    )


def _apply(x: torch.Tensor, fn: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile])
    return out


def _value(out: torch.Tensor) -> float:
    return out.flatten()[0].item()


@dataclasses.dataclass
class _Field:
    n: int


def _divide(x: torch.Tensor, s: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] / s
    return out


def _divide_annotated(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] / s
    return out


def _divide_by(s: object) -> object:
    return lambda t: t / s


@pytest.mark.parametrize("zeros", [(0.0, -0.0), (-0.0, 0.0)], ids=["pos", "neg"])
@pytest.mark.parametrize("form", ["plain", "wrapped", "cell"])
def test_signed_zero_constexpr_compiles_apart(
    form: str, zeros: tuple[float, float]
) -> None:
    x = torch.ones(32)
    if form == "plain":
        kernel = _kernel(_divide_annotated)
        call = lambda s: kernel(x, s)  # noqa: E731
    elif form == "wrapped":
        kernel = _kernel(_divide)
        call = lambda s: kernel(x, hl.constexpr(s))  # noqa: E731
    else:
        kernel = _kernel(_apply)
        call = lambda s: kernel(x, _divide_by(hl.constexpr(s)))  # noqa: E731
    for s in (*zeros, *zeros):
        for _ in range(2):  # the second call takes the dispatch cache
            assert _value(call(s)) == math.copysign(math.inf, s)
    assert len(kernel._bound_kernels) == 2


def test_constexpr_type_is_part_of_the_key() -> None:
    kernel = _kernel(_divide)
    x = torch.ones(8)
    key = kernel._base_specialization_key
    wrapped = [hl.constexpr(v) for v in (1, 1.0, True)]
    assert len({key((x, v)) for v in wrapped}) == 3
    assert key((x, hl.constexpr(2.0))) == key((x, hl.constexpr(2.0)))
    annotated = _kernel(_divide_annotated)
    assert annotated._base_specialization_key((x, 0.0)) != (
        annotated._base_specialization_key((x, -0.0))
    )
    assert annotated._base_specialization_key((x, 1)) != (
        annotated._base_specialization_key((x, 1.0))
    )


@pytest.mark.parametrize("kind", [np.float64, np.float32], ids=["f64", "f32"])
def test_numpy_float_in_a_constexpr_cell_keeps_its_sign_of_zero(kind: type) -> None:
    key = _kernel(_apply)._base_specialization_key
    x = torch.ones(8)

    def cell(value: float) -> object:
        return _divide_by(hl.constexpr(kind(value)))

    assert key((x, cell(0.0))) != key((x, cell(-0.0)))
    assert key((x, cell(0.0))) == key((x, cell(0.0)))


def test_numpy_float64_constexpr_compiles_apart() -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    for value in (0.0, -0.0):
        out = kernel(x, _divide_by(hl.constexpr(np.float64(value))))
        assert _value(out) == math.copysign(math.inf, value)


def test_constexpr_tuple_keys_its_elements() -> None:
    key = _kernel(_divide)._base_specialization_key
    x = torch.ones(8)
    assert key((x, hl.constexpr((0.0,)))) != key((x, hl.constexpr((-0.0,))))
    assert key((x, hl.constexpr((1,)))) != key((x, hl.constexpr((1.0,))))


def test_other_backends_keep_helions_constexpr_key() -> None:
    kernel = helion.kernel(_divide, backend="triton")
    x = torch.ones(8)
    assert kernel._specialization_key(hl.constexpr(0.0)) == 0.0
    base = kernel._base_specialization_key
    assert base((x, hl.constexpr(0.0))) == base((x, hl.constexpr(-0.0)))


def test_mutated_list_default_compiles_apart() -> None:
    held = [1]

    def fn(t, h=held):
        if h[0] > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    held[0] = 3
    assert _value(kernel(x, fn)) == 10.0


def test_mutated_dataclass_default_compiles_apart() -> None:
    held = _Field(1)

    def fn(t, h=held):
        if h.n > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    held.n = 3
    assert _value(kernel(x, fn)) == 10.0


def _scale_ints(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * v
    return out


def _ir(value: object) -> str:
    x = torch.ones(8, dtype=torch.int32)
    return str(helion_mlir_backend.generate_mlir(_kernel(_scale_ints), [x, value]))


def test_constexpr_type_changes_the_ir() -> None:
    """Why the type belongs in the key: on an int tensor, ``3`` multiplies as ints
    and ``3.0`` promotes to floats; ``1`` and ``1.0`` fold differently; ``True``
    has its own helper."""
    assert "arith.muli" in _ir(hl.constexpr(3))
    assert "arith.mulf" in _ir(hl.constexpr(3.0))
    one, one_float, true = (
        _ir(hl.constexpr(1)),
        _ir(hl.constexpr(1.0)),
        _ir(hl.constexpr(True)),
    )
    assert "arith.sitofp" not in one
    assert "arith.sitofp" in one_float
    assert one != true


@pytest.mark.parametrize(
    "order", [(1, 1.0), (1.0, 1)], ids=["int-first", "float-first"]
)
def test_results_follow_the_constexpr_type(order: tuple[object, object]) -> None:
    """Through f32, 2**24 + 1 rounds: multiplying by ``1.0`` differs from ``1``, so
    sharing a compile gave a wrong number."""
    kernel = _kernel(_scale_ints)
    x = torch.full((8,), 2**24 + 1, dtype=torch.int32)
    for value in (*order, *order):
        want = (x * value).to(torch.int32)
        assert kernel(x, hl.constexpr(value)).equal(want), value
    assert len(kernel._bound_kernels) == 2
    assert kernel.bind((x, hl.constexpr(1))) is not kernel.bind((x, hl.constexpr(1.0)))


@pytest.mark.skipif(
    native.module() is None, reason="the native extension could not be built"
)
def test_negative_zero_is_no_native_hit_of_a_plan_on_zero() -> None:
    kernel = _kernel(_divide)
    x = torch.ones(32)
    positive = hl.constexpr(0.0)
    kernel(x, positive)
    kernel(x, positive)
    bound = kernel.bind((x, positive))
    run = bound._run
    assert run.planned and run.native_calls == 1
    # Through the kernel: its own bound kernel.
    assert _value(kernel(x, hl.constexpr(-0.0))) == -math.inf
    assert run.native_calls == 1
    # A bound kernel called directly: the plan's own check refuses it.
    bound(x, hl.constexpr(-0.0))
    assert run.native_calls == 1
    bound(x, hl.constexpr(0.0))
    assert run.native_calls == 2
