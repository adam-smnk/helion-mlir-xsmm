"""Identity-keyed lowering registry."""

from __future__ import annotations

import helion.language.memory_ops as memory_ops
import pytest
import torch

from helion_mlir_backend._compiler.mlir.lowering import registry
from helion_mlir_backend._compiler.mlir.lowering.contraction_ops import (
    lower_accumulating_add,
)
from helion_mlir_backend._compiler.mlir.lowering.contraction_ops import (
    lower_contraction_node,
)
from helion_mlir_backend._compiler.mlir.lowering.elementwise_ops import (
    lower_scalar_binary,
)
from helion_mlir_backend._compiler.mlir.lowering.load_slice_ops import lower_load

aten = torch.ops.aten


def test_overload_filter_excludes_other_overloads(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_ENTRIES", {})

    @registry.lowers(aten.div, overloads=("Tensor",))
    def handler(ctx, node):
        return None

    assert registry.handlers_for(aten.div.Tensor) == [handler]
    assert registry.handlers_for(aten.div.Tensor_mode) == []


def test_overload_filter_requires_a_packet() -> None:
    with pytest.raises(TypeError):
        registry.lowers(aten.div.Tensor, overloads=("Tensor",))


def test_substring_names_do_not_match() -> None:
    assert lower_contraction_node in registry.handlers_for(aten.mm.default)
    assert lower_contraction_node in registry.handlers_for(aten.addmm.default)
    assert registry.handlers_for(aten._int_mm.default) == []
    assert lower_scalar_binary not in registry.handlers_for(aten.div.Tensor_mode)
    assert registry.handlers_for(aten.add.Tensor) == [
        lower_accumulating_add,
        lower_scalar_binary,
    ]


def test_helion_functions_are_matched_by_identity() -> None:
    assert registry.handlers_for(memory_ops.load) == [lower_load]
    assert registry.handlers_for(lambda: None) == []
