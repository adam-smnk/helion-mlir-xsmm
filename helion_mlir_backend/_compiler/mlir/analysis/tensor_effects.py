"""Host tensors loaded and stored by each device graph (read-only analysis).

A host tensor is identified by its ``_host_tensor(name)`` name. Effects are
transitive through nested loop, ``if`` and ``while`` bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
import torch
import torch.fx

from .geometry import is_loop_node

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction


def host_tensor_name(node: object) -> str | None:
    """The host tensor a ``_host_tensor(name)`` node refers to, else ``None``."""
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is tracing_ops._host_tensor
        and isinstance(node.args[0], str)
    ):
        return node.args[0]
    return None


def accessed_tensor(node: torch.fx.Node) -> str | None:
    """The host tensor a ``load``/``store`` node reads or writes, else ``None``."""
    if node.op == "call_function" and node.target in (
        memory_ops.load,
        memory_ops.store,
    ):
        return host_tensor_name(node.args[0])
    return None


def subgraph_ids(node: torch.fx.Node) -> list[int]:
    """The device graphs a loop, ``_if`` or ``_while_loop`` node runs."""
    if node.op != "call_function":
        return []
    if is_loop_node(node):
        return [node.args[0]]
    if node.target is tracing_ops._if:
        return [node.args[1], node.args[2]]
    if node.target is tracing_ops._while_loop:
        cond_id, body_id, _, *orelse = node.args
        return [
            cond_id,
            body_id,
            *(graph_id for graph_id in orelse if graph_id is not None),
        ]
    return []


@dataclass
class TensorEffects:
    hf: HostFunction
    fakes: dict[str, torch.Tensor] = field(default_factory=dict)
    _writes: dict[int, tuple[str, ...]] = field(default_factory=dict)
    _accesses: dict[int, list[torch.fx.Node]] = field(default_factory=dict)

    @classmethod
    def from_host_function(cls, hf: HostFunction) -> TensorEffects:
        effects = cls(hf)
        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                name = host_tensor_name(node)
                value = node.meta.get("val")
                if name is not None and isinstance(value, torch.Tensor):
                    effects.fakes.setdefault(name, value)
        return effects

    def accesses(self, graph_id: int) -> list[torch.fx.Node]:
        """Every ``load``/``store`` of a host tensor in the graph and its subgraphs."""
        if graph_id not in self._accesses:
            found: list[torch.fx.Node] = []
            for node in self.hf.device_ir.graphs[graph_id].graph.nodes:
                if accessed_tensor(node) is not None:
                    found.append(node)
                for subgraph_id in subgraph_ids(node):
                    found.extend(self.accesses(subgraph_id))
            self._accesses[graph_id] = found
        return self._accesses[graph_id]

    def writes(self, graph_id: int) -> tuple[str, ...]:
        """Host tensors stored in the graph and its loop bodies, in first-store order."""
        if graph_id not in self._writes:
            names = (
                accessed_tensor(node)
                for node in self.accesses(graph_id)
                if node.target is memory_ops.store
            )
            self._writes[graph_id] = tuple(dict.fromkeys(names))
        return self._writes[graph_id]
