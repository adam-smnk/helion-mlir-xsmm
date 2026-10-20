"""Helpers for resolving Helion block-id naming conventions."""

from __future__ import annotations


def block_id_from_key(value: object) -> int | None:
    """Return a block id from a ``block_size_N`` key."""
    if not isinstance(value, str) or not value.startswith("block_size_"):
        return None
    suffix = value.removeprefix("block_size_")
    return int(suffix) if suffix.isdigit() else None


#: Symbol kinds that denote a scalar position rather than a tile extent.
SCALAR_SYMBOL_KINDS = frozenset(
    {"grid", "tile_begin", "tile_end", "tile_id", "tile_count"}
)
#: Scalar positions whose value is their loop's current offset.
OFFSET_SYMBOL_KINDS = frozenset({"grid", "tile_begin"})


def _symbol_expr(value: object) -> object | None:
    """Return the sympy expression backing a SymInt, if any."""
    import sympy
    import torch

    if isinstance(value, sympy.Basic):
        return value
    if not isinstance(value, torch.SymInt):
        return None
    expr = getattr(getattr(value, "node", None), "expr", None)
    return expr if isinstance(expr, sympy.Basic) else None


def symbol_origin_info(host_function: object, value: object) -> tuple[int, str] | None:
    """Resolve a SymInt to its ``(block_id, kind)`` via Helion symbol origins.

    ``kind`` is ``block_size`` for tile extents, or one of
    :data:`SCALAR_SYMBOL_KINDS` for scalar grid/tile positions.
    """
    from helion._compiler.variable_origin import BlockSizeOrigin
    from helion._compiler.variable_origin import GridOrigin
    from helion._compiler.variable_origin import TileBeginOrigin
    from helion._compiler.variable_origin import TileCountOrigin
    from helion._compiler.variable_origin import TileEndOrigin
    from helion._compiler.variable_origin import TileIdOrigin

    expr = _symbol_expr(value)
    if expr is None:
        return None
    origin_map = getattr(host_function, "expr_to_origin", None)
    if not origin_map:
        return None
    symbol_origin = origin_map.get(expr)
    if symbol_origin is None:
        return None
    origin = getattr(symbol_origin, "origin", None)

    # Tile* origins subclass GridOrigin, so they must be matched first.
    if isinstance(origin, TileBeginOrigin):
        return origin.block_id, "tile_begin"
    if isinstance(origin, TileEndOrigin):
        return origin.block_id, "tile_end"
    if isinstance(origin, TileIdOrigin):
        return origin.block_id, "tile_id"
    if isinstance(origin, TileCountOrigin):
        return origin.block_id, "tile_count"
    if isinstance(origin, GridOrigin):
        return origin.block_id, "grid"
    if isinstance(origin, BlockSizeOrigin):
        return origin.block_id, "block_size"
    return None
