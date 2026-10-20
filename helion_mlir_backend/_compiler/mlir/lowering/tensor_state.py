"""SSA state of the host tensors a device function writes.

Each written host tensor has one current SSA value. Inside a parallel
``scf.forall`` iteration that value is the iteration's *owned region*: the
tile/point it covers along every dimension owned by a grid block id, and the
full extent elsewhere. Loads and stores index that local value relative to the
region's origin; the forall terminator inserts it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from ..support.index_meta import resolve_index_descriptor

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch

    from ..build_context import BuildContext


@dataclass(frozen=True)
class OwnedDim:
    block_id: int
    point: bool
    """A scalar position (size 1) rather than a tile."""
    kind: str | None = None
    """The position's symbol kind: ``grid``, ``tile_begin`` or ``tile_id``."""


@dataclass
class _RefState:
    value: ir.Value
    owned: dict[int, OwnedDim]


@dataclass
class TensorState:
    _refs: dict[str, _RefState] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        return name in self._refs

    def value(self, name: str) -> ir.Value:
        return self._refs[name].value

    def owned(self, name: str) -> dict[int, OwnedDim]:
        return self._refs[name].owned

    def bind(
        self, name: str, value: ir.Value, owned: dict[int, OwnedDim] | None = None
    ) -> None:
        self._refs[name] = _RefState(value, dict(owned or {}))

    def rebind(self, name: str, value: ir.Value) -> None:
        """A new value for the same region (e.g. after a store or a loop)."""
        self._refs[name].value = value

    def clear(self) -> None:
        self._refs.clear()


def owned_dims(
    ctx: BuildContext,
    accesses: list[torch.fx.Node],
    rank: int,
    grid_block_ids: list[int],
) -> dict[int, OwnedDim]:
    """Dims every access indexes with the same grid block id, in the same form."""
    candidates: dict[int, OwnedDim | None] = {}
    for node in accesses:
        # ``None`` adds an axis to the loaded value, not a dim of the tensor.
        index_nodes = [item for item in node.args[1] if item is not None]
        for dim in range(rank):
            index = index_nodes[dim] if dim < len(index_nodes) else slice(None)
            candidate = _owner(ctx, index, grid_block_ids)
            if dim not in candidates:
                candidates[dim] = candidate
            elif candidates[dim] != candidate:
                candidates[dim] = None
    return {dim: owner for dim, owner in candidates.items() if owner is not None}


def _owner(
    ctx: BuildContext, index: object, grid_block_ids: list[int]
) -> OwnedDim | None:
    if isinstance(index, slice):
        return None
    descriptor = resolve_index_descriptor(ctx, index)
    if descriptor.block_id is None or descriptor.bias:
        return None
    if not descriptor.is_scalar:
        # A tile of a loop over one grid tile's range stays inside that tile.
        owner = ctx.geometry.owning_block(descriptor.block_id, grid_block_ids)
        return None if owner is None else OwnedDim(owner, point=False)
    if descriptor.block_id not in grid_block_ids:
        return None
    # Positions distinct in every iteration of the block's loop.
    if descriptor.is_offset or descriptor.kind == "tile_id":
        return OwnedDim(descriptor.block_id, point=True, kind=descriptor.kind)
    return None
