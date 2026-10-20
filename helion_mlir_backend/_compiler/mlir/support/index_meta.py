"""Authoritative (block_id, bias, is_scalar) resolution for index nodes.

Single source of truth for mapping a Helion device-IR index expression to the
block id / bias it represents, built entirely from Helion's own device-IR
metadata (``tile_with_offset``, symbol origins) instead of name-based or
id()-keyed heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops

from .block_ids import OFFSET_SYMBOL_KINDS
from .block_ids import SCALAR_SYMBOL_KINDS
from .block_ids import block_id_from_key
from .errors import UnsupportedOperationError

if TYPE_CHECKING:
    from ..build_context import BuildContext


@dataclass(frozen=True)
class IndexDescriptor:
    """Resolved identity of a single subscript index position."""

    block_id: int | None
    bias: int
    is_scalar: bool
    kind: str | None = None
    """The symbol kind (``grid``, ``tile_id``, ...) of a scalar position."""

    @property
    def is_offset(self) -> bool:
        """A scalar position equal to its loop's current offset."""
        return self.kind in OFFSET_SYMBOL_KINDS


_UNRESOLVED = IndexDescriptor(block_id=None, bias=0, is_scalar=False)


def resolve_index_descriptor(ctx: BuildContext, index_node: object) -> IndexDescriptor:
    """Resolve an index expression to its block id, bias, and scalar-ness.

    Resolution order (all authoritative, no name/heuristic matching):
    1. Literal int -> scalar constant offset.
    2. ``meta['tile_with_offset']`` (``tile.index + k``) -> block id + constant
       offset, set for every backend by Helion's ``add_tile_with_offset_metadata``.
    3. Symbol origin (``HostFunction.expr_to_origin`` via
       ``BuildContext.symbol_info``) -> block id, plus whether it denotes a
       scalar grid/tile position or a tile extent.
    4. ``_get_symnode('block_size_N')`` key -> block id directly; a constant
       key names the active tile loop whose block size Helion specialized to
       it (``hl.register_block_size`` of a size-1 dim).
    """
    import torch
    import torch.fx

    if isinstance(index_node, int):
        return IndexDescriptor(block_id=None, bias=index_node, is_scalar=True)
    if not isinstance(index_node, torch.fx.Node):
        return _UNRESOLVED

    tile_meta = index_node.meta.get("tile_with_offset")
    if tile_meta is not None:
        offset = tile_meta.get("offset", 0)
        if isinstance(offset, torch.SymInt) and not offset.node.expr.free_symbols:
            offset = int(offset.node.expr)
        if not isinstance(offset, int):
            raise UnsupportedOperationError(
                "tile index", reason=f"tile.index plus the runtime offset {offset}"
            )
        return IndexDescriptor(
            block_id=tile_meta.get("block_id"), bias=offset, is_scalar=False
        )

    symbol_info = ctx.symbol_info(index_node.meta.get("val"))
    if symbol_info is not None:
        block_id, kind = symbol_info
        return IndexDescriptor(
            block_id=block_id,
            bias=0,
            is_scalar=kind in SCALAR_SYMBOL_KINDS,
            kind=kind,
        )

    target = index_node.target
    if target is tracing_ops._get_symnode and index_node.args:
        key = index_node.args[0]
        block_id = block_id_from_key(key)
        if block_id is None and key.isdigit():
            block_id = _specialized_block(ctx, key)
        if block_id is not None:
            return IndexDescriptor(block_id=block_id, bias=0, is_scalar=False)

    return _UNRESOLVED


def _specialized_block(ctx: BuildContext, key: str) -> int | None:
    """The one active tile loop whose block size symbol is the constant ``key``."""
    import torch

    matches = [
        info.block_id
        for info in ctx.env.block_sizes
        if isinstance(info.var, torch.SymInt)
        and str(info.var.node.expr) == key
        and info.block_id in ctx.block_id_to_bounds
    ]
    return matches[0] if len(matches) == 1 else None
