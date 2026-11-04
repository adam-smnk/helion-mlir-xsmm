"""A config key that looks like a typo of a valid one warns once; nothing else warns or raises."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import helion
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl
import pytest
import torch

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir import driver

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _nothing_warned_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(driver, "_warned_spellings", set(), raising=False)


def scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * 2
    return out


def other_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * 3
    return out


def tunable_scale(x: torch.Tensor) -> torch.Tensor:
    factor = hl.register_tunable("xegpu_sg_tiles_x", PowerOfTwoFragment(1, 8, 2))
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * factor
    return out


def _kernel(fn: object = scale, **keys: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[32, 32], **keys),
    )


def _x() -> torch.Tensor:
    return torch.ones(64, 64)


class _Seen:
    """Messages of the config key warnings so far."""

    def __init__(self, caught: list[warnings.WarningMessage]) -> None:
        self.caught = caught

    def __iter__(self) -> Iterator[str]:
        return (str(w.message) for w in self.caught if "config key" in str(w.message))

    def __len__(self) -> int:
        return sum(1 for _ in self)


@pytest.fixture
def seen() -> Iterator[_Seen]:
    """An explicit filter, since ``pyproject.toml``'s ``ignore::UserWarning`` would hide
    the warnings a test says are absent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield _Seen(caught)


@pytest.mark.parametrize(
    ("key", "valid"),
    [
        ("xegpu_sg_tile", "xegpu_sg_tiles"),
        ("xepgu_sg_tiles", "xegpu_sg_tiles"),
        ("xegpu_prefetch_depth", "xegpu_prefetch_depths"),
        ("mlir_pipline", "mlir_pipeline"),
        ("igc_vectoriser", "igc_vectorizer"),
        ("block_size", "block_sizes"),
    ],
)
@pytest.mark.parametrize("target", ["cpu", "xpu"])
def test_misspelt_key_warns_and_names_the_valid_key(
    key: str, valid: str, target: str
) -> None:
    stop = {"xpu": "xegpu-wg"}.get(target)
    with pytest.warns(UserWarning, match=rf"{key}'.*being ignored.*'{valid}'"):
        lower_mlir(_kernel(**{key: [[32, 64]]}), [_x()], target=target, stop_at=stop)


def test_unknown_key_is_silent(seen: _Seen) -> None:
    """Next to a misspelt key, so that the check sees warnings at all."""
    lower_mlir(_kernel(not_a_key=[1], xegpu_sg_tile=[[32, 64]]), [_x()], target="cpu")
    assert [m for m in seen if "not_a_key" in m] == []
    assert len([m for m in seen if "xegpu_sg_tile" in m]) == 1


def test_valid_keys_are_silent(seen: _Seen) -> None:
    """The backend's own keys and those of Helion's other backends (``VALID_KEYS``), next to
    a misspelt key, so that the check sees warnings at all."""
    keys = {
        "xegpu_sg_tiles": [[32, 64]],
        "xegpu_load_tiles": [[32, 32]],
        "xegpu_prefetch_tiles": [[32, 32]],
        "xegpu_prefetch_depths": [[1, 1]],
        "xegpu_nan_ignoring_maxima": [True],
        "xegpu_layouts": ["ir"],
        "igc_vectorizer": ["off"],
        "mlir_pipeline": "scalar",
        "num_warps": 4,
        "num_stages": 3,
        "maxnreg": 256,
        "pid_type": "linear",
        "loop_orders": [[0, 1]],
    }
    lower_mlir(_kernel(xegpu_load_tile=[[32, 32]], **keys), [_x()], target="cpu")
    assert [m for m in seen if "xegpu_load_tile'" not in m] == []
    assert len(seen) == 1


def test_a_tunable_of_the_kernel_is_silent(seen: _Seen) -> None:
    """A kernel's tunables are config keys too; this name is close to ``xegpu_sg_tiles``."""
    keys = {"xegpu_sg_tiles_x": 2, "xegpu_load_tile": [[32, 32]]}
    lower_mlir(_kernel(tunable_scale, **keys), [_x()], target="cpu")
    assert [m for m in seen if "xegpu_load_tile'" in m] != []
    assert [m for m in seen if "xegpu_sg_tiles_x'" in m] == []


def test_warns_once_per_kernel_and_key(seen: _Seen) -> None:
    for _ in range(3):
        lower_mlir(_kernel(xegpu_sg_tile=[[32, 64]]), [_x()], target="cpu")
    assert len(seen) == 1
    lower_mlir(_kernel(xegpu_load_tile=[[32, 32]]), [_x()], target="cpu")
    assert len(seen) == 2
    lower_mlir(_kernel(other_scale, xegpu_sg_tile=[[32, 64]]), [_x()], target="cpu")
    assert len(seen) == 3


def test_a_call_warns_once_and_a_typo_changes_nothing(seen: _Seen) -> None:
    """The call path: one warning however many calls and shapes, and the same bits."""
    plain = _kernel()
    typo = _kernel(xegpu_sg_tile=[[32, 64]])
    x = torch.randn(64, 64, generator=torch.Generator().manual_seed(0))
    for _ in range(3):
        assert torch.equal(typo(x), plain(x))
    assert typo(x[:32]).shape == (32, 64)
    assert len(seen) == 1
    assert "xegpu_sg_tile" in next(iter(seen))


@pytest.mark.parametrize(
    ("target", "stop"), [("cpu", None), ("xpu", "xegpu-wg"), ("xpu", "xevm")]
)
def test_a_typo_leaves_the_ir_alone(target: str, stop: str | None) -> None:
    def lowered(**keys: object) -> str:
        return str(lower_mlir(_kernel(**keys), [_x()], target=target, stop_at=stop))

    assert lowered(xegpu_sg_tile=[[32, 64]], not_a_key=[1]) == lowered()
