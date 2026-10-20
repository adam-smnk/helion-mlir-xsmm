"""Kernel calling convention: the host tensors and runtime scalars device code uses.

A tensor ref is named by its ``_host_tensor`` name, which is the host expression
that produces it (``x``, ``out``, ``self.weight``). Refs are ordered declared tensor
parameters first (declaration order), then other host tensors in first-use order.
A ref written anywhere is ``inout``; every other ref is ``in``. A read-only host
view of a declared parameter holding the same elements in the same order (a
reshape) is not a ref: it lowers to a reshape of that parameter. Any other view
(a slice, a transpose) is a ref of its own, which the host code computes.

Runtime scalars are symbolic ints/floats with a host origin (non-constexpr scalar
parameters); Helion keys them by type only, so their values are call arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..support.block_ids import block_id_from_key
from ..support.block_ids import symbol_origin_info
from .tensor_effects import accessed_tensor
from .tensor_effects import host_tensor_name
from .tensor_effects import subgraph_ids

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.host_function import HostFunction

    from .tensor_effects import TensorEffects


@dataclass(frozen=True)
class TensorRef:
    name: str
    fake: torch.Tensor
    written: bool
    tensor_param: int | None
    """Position among the kernel's tensor parameters, if the ref is one."""


@dataclass(frozen=True)
class ScalarArg:
    key: str
    """The ``_get_symnode`` key device code uses."""
    host_expr: str
    dtype: torch.dtype
    expr: object = None
    """The ``sympy`` expression an int scalar carries (a size source)."""


@dataclass(frozen=True)
class PhaseSignature:
    root_positions: tuple[int, ...]
    ins: tuple[str, ...]
    inouts: tuple[str, ...]
    scalars: tuple[str, ...]


@dataclass(frozen=True)
class KernelSignature:
    refs: dict[str, TensorRef]
    scalars: dict[str, ScalarArg]
    aliases: dict[str, str]
    """Read-only host reshapes of a declared parameter -> that parameter."""
    phases: tuple[PhaseSignature, ...]

    @property
    def inouts(self) -> list[TensorRef]:
        return [ref for ref in self.refs.values() if ref.written]

    @property
    def ins(self) -> list[TensorRef]:
        return [ref for ref in self.refs.values() if not ref.written]

    @classmethod
    def from_host_function(
        cls, hf: HostFunction, effects: TensorEffects
    ) -> KernelSignature:
        tensor_params = [
            name
            for name, value in hf.params.arguments.items()
            if isinstance(value, torch.Tensor)
        ]
        device_ir = hf.device_ir
        phase_nodes = [
            [
                node
                for position in phase.roots
                for node in _program_order(hf, device_ir.root_ids[position])
            ]
            for phase in device_ir.phases
        ]
        all_nodes = [node for nodes in phase_nodes for node in nodes]
        used = dict.fromkeys(
            name for node in all_nodes if (name := host_tensor_name(node)) is not None
        )
        written = _stored(all_nodes)
        scalars: dict[str, ScalarArg] = {}
        for node in all_nodes:
            if (scalar := _runtime_scalar(hf, node)) is not None:
                scalars.setdefault(scalar.key, scalar)

        aliases = {
            name: base
            for name in used
            if name not in tensor_params
            and name not in written
            and (base := _declared_base(hf, effects.fakes[name], tensor_params))
            and _same_elements(effects.fakes[name], hf.params.arguments[base])
        }
        ordered = [
            name for name in tensor_params if name in used or name in aliases.values()
        ]
        ordered += [
            name for name in used if name not in tensor_params and name not in aliases
        ]
        refs = {
            name: TensorRef(
                name,
                effects.fakes.get(name, hf.params.arguments.get(name)),
                name in written,
                tensor_params.index(name) if name in tensor_params else None,
            )
            for name in ordered
        }

        phases = []
        for phase, nodes in zip(device_ir.phases, phase_nodes, strict=True):
            touched = {
                aliases.get(name, name)
                for node in nodes
                if (name := host_tensor_name(node)) is not None
            }
            stored = _stored(nodes)
            keys = {
                node.args[0] for node in nodes if _runtime_scalar(hf, node) is not None
            }
            phases.append(
                PhaseSignature(
                    tuple(phase.roots),
                    tuple(n for n in refs if n in touched and n not in stored),
                    tuple(n for n in refs if n in stored),
                    tuple(key for key in scalars if key in keys),
                )
            )
        return cls(refs, scalars, aliases, tuple(phases))


def _program_order(hf: HostFunction, graph_id: int) -> Iterator[torch.fx.Node]:
    """The graph's nodes, with each subgraph's nodes at its loop, if or while node."""
    for node in hf.device_ir.graphs[graph_id].graph.nodes:
        yield node
        for subgraph_id in subgraph_ids(node):
            yield from _program_order(hf, subgraph_id)


def _stored(nodes: list[torch.fx.Node]) -> set[str]:
    return {
        name
        for node in nodes
        if node.op == "call_function"
        and node.target is memory_ops.store
        and (name := accessed_tensor(node)) is not None
    }


def _runtime_scalar(hf: HostFunction, node: torch.fx.Node) -> ScalarArg | None:
    if node.op != "call_function" or node.target is not tracing_ops._get_symnode:
        return None
    key = node.args[0]
    value = node.meta.get("val")
    if (
        not isinstance(value, (torch.SymInt, torch.SymFloat))
        or not value.node.expr.free_symbols
        or block_id_from_key(key) is not None
        or symbol_origin_info(hf, value) is not None
    ):
        return None
    symbol_origin = hf.expr_to_origin.get(value.node.expr)
    if symbol_origin is None or not symbol_origin.origin.is_host():
        return None
    if isinstance(value, torch.SymFloat):
        return ScalarArg(key, symbol_origin.origin.host_str(), torch.float64)
    return ScalarArg(key, symbol_origin.origin.host_str(), torch.int64, value.node.expr)


def _declared_base(
    hf: HostFunction, tensor: torch.Tensor, tensor_params: list[str]
) -> str | None:
    """The declared parameter a host tensor is a view of, via origins and ``._base``."""
    seen: set[int] = set()
    current: torch.Tensor | None = tensor
    while isinstance(current, torch.Tensor) and id(current) not in seen:
        seen.add(id(current))
        origin = hf.tensor_to_origin.get(current)
        if origin is not None and origin.host_str() in tensor_params:
            return origin.host_str()
        current = getattr(current, "_base", None)
    return None


def _same_elements(view: torch.Tensor, base: torch.Tensor) -> bool:
    """Whether ``view`` provably holds all of ``base``'s elements in row-major order."""
    return (
        statically_known_true(view.storage_offset() == 0)
        and statically_known_true(view.numel() == base.numel())
        and _row_major(view)
        and _row_major(base)
    )


def _row_major(tensor: torch.Tensor) -> bool:
    expected: int | torch.SymInt = 1
    for size, stride in reversed(list(zip(tensor.shape, tensor.stride(), strict=True))):
        if not statically_known_true(size == 1) and not statically_known_true(
            stride == expected
        ):
            return False
        expected = expected * size
    return True
