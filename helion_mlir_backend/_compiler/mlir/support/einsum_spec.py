"""Einsum equation analysis and mapping onto ``linalg.contract`` semantics.

``linalg.contract`` computes ``D[H] = (SUM_{(I u J) \\ H} A[I] * B[J]) + C[H]``
with *projected permutation* indexing maps.  That is strictly less expressive
than einsum, so only a subset of equations can be mapped directly; the rest
must fall back to PyTorch's own decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

_ELLIPSIS = "..."


class EinsumNotContractible(ValueError):
    """Raised when an equation cannot be expressed as ``linalg.contract``."""


@dataclass(frozen=True)
class ContractSpec:
    """A contraction expressed over a shared iteration space.

    ``iteration_dims`` is the ordered list of subscript letters forming the
    iteration space; ``lhs``/``rhs``/``out`` hold, for each operand, the
    positions into ``iteration_dims`` selected by its indexing map.
    """

    iteration_dims: tuple[str, ...]
    lhs: tuple[int, ...]
    rhs: tuple[int, ...]
    out: tuple[int, ...]
    dim_sizes: tuple[int, ...]

    @property
    def out_shape(self) -> list[int]:
        return [self.dim_sizes[position] for position in self.out]

    @property
    def reduction_dims(self) -> tuple[str, ...]:
        out_positions = set(self.out)
        return tuple(
            self.iteration_dims[position]
            for position in range(len(self.iteration_dims))
            if position not in out_positions
        )


def parse_equation(equation: str, operand_ranks: list[int]) -> tuple[list[str], str]:
    """Split an einsum equation into per-operand subscripts and the output.

    Implicit-output equations (no ``->``) get the standard einsum output:
    subscripts occurring exactly once, sorted alphabetically.
    """
    equation = equation.replace(" ", "")
    if "->" in equation:
        inputs_text, output = equation.split("->", 1)
    else:
        inputs_text, output = equation, None

    inputs = inputs_text.split(",")
    if len(inputs) != len(operand_ranks):
        raise EinsumNotContractible(
            f"equation '{equation}' has {len(inputs)} operand(s) but "
            f"{len(operand_ranks)} were passed"
        )
    if any(_ELLIPSIS in subscript for subscript in inputs) or (
        output is not None and _ELLIPSIS in output
    ):
        raise EinsumNotContractible("ellipsis subscripts are not supported")
    for subscript, rank in zip(inputs, operand_ranks, strict=True):
        if not all(character.isalpha() for character in subscript):
            raise EinsumNotContractible(f"invalid subscript '{subscript}'")
        if len(subscript) != rank:
            raise EinsumNotContractible(
                f"subscript '{subscript}' does not match operand rank {rank}"
            )

    if output is None:
        counts: dict[str, int] = {}
        for subscript in inputs:
            for character in subscript:
                counts[character] = counts.get(character, 0) + 1
        output = "".join(sorted(c for c, n in counts.items() if n == 1))
    elif not all(character.isalpha() for character in output):
        raise EinsumNotContractible(f"invalid output subscript '{output}'")

    return inputs, output


def build_contract_spec(
    equation: str, operand_shapes: list[list[object]]
) -> ContractSpec:
    """Return a :class:`ContractSpec` or raise :class:`EinsumNotContractible`.

    Rejects everything ``linalg.contract`` cannot represent: a number of
    operands other than two, repeated subscripts inside one operand
    (diagonals), output subscripts absent from both inputs (broadcasts),
    duplicated output subscripts, and any dimension reduced over a single
    operand only (``linalg.contract`` has no iterator type for that).
    """
    inputs, output = parse_equation(equation, [len(s) for s in operand_shapes])

    if len(inputs) != 2:
        raise EinsumNotContractible(
            f"linalg.contract takes exactly 2 inputs, got {len(inputs)}"
        )
    for subscript in (*inputs, output):
        if len(set(subscript)) != len(subscript):
            raise EinsumNotContractible(
                f"repeated subscript in '{subscript}' is not a projected permutation"
            )

    lhs_subscript, rhs_subscript = inputs
    lhs_set, rhs_set, out_set = set(lhs_subscript), set(rhs_subscript), set(output)

    if not out_set <= (lhs_set | rhs_set):
        missing = sorted(out_set - (lhs_set | rhs_set))
        raise EinsumNotContractible(
            f"output subscript(s) {missing} appear in no input operand"
        )

    # Iteration space: output dims first (parallel), then contracted dims.
    contracted = [
        character
        for character in dict.fromkeys(lhs_subscript + rhs_subscript)
        if character not in out_set
    ]
    for character in contracted:
        if character not in lhs_set or character not in rhs_set:
            raise EinsumNotContractible(
                f"subscript '{character}' is reduced over a single operand"
            )
    iteration_dims = tuple(output) + tuple(contracted)
    position_of = {character: i for i, character in enumerate(iteration_dims)}

    sizes: dict[str, object] = {}
    observed: dict[str, list[object | None]] = {}
    for subscript, shape in zip(inputs, operand_shapes, strict=True):
        for character, extent in zip(subscript, shape, strict=True):
            observed.setdefault(character, []).append(_extent_key(extent))
            if character not in sizes or not isinstance(sizes[character], int):
                sizes[character] = extent
    for character, keys in observed.items():
        _check_extent_agreement(character, keys)

    return ContractSpec(
        iteration_dims=iteration_dims,
        lhs=tuple(position_of[c] for c in lhs_subscript),
        rhs=tuple(position_of[c] for c in rhs_subscript),
        out=tuple(position_of[c] for c in output),
        dim_sizes=tuple(sizes[c] for c in iteration_dims),
    )


def _extent_key(extent: object) -> object | None:
    """Comparable identity for a dimension extent, or ``None`` if unknown.

    A symbolic extent is keyed by its symbol, so two operands can be *proven*
    to share a dimension without evaluating it (which would install a shape
    guard on an unbacked SymInt).
    """
    if isinstance(extent, int) and not isinstance(extent, bool):
        return extent
    expression = getattr(getattr(extent, "node", None), "expr", None)
    return None if expression is None else f"sym:{expression}"


def _check_extent_agreement(character: str, keys: list[object | None]) -> None:
    """Reject extents a contraction cannot share across both operands."""
    known = [key for key in keys if key is not None]
    static = {key for key in known if isinstance(key, int)}
    if len(static) > 1:
        first, second = sorted(static)[:2]
        raise EinsumNotContractible(
            f"subscript '{character}' bound to sizes {first} and {second}"
        )
    if len(known) == len(keys) and len(set(known)) <= 1:
        return
    # Not provably one extent: a size-1 partner is the case einsum silently
    # broadcasts and linalg.contract cannot.
    if 1 in static:
        raise EinsumNotContractible(
            f"subscript '{character}' has a size-1 extent that einsum would broadcast"
        )


def is_contractible(
    equation: str,
    operand_shapes: list[list[object]],
    require_reduction: bool = False,
) -> bool:
    """Return whether ``equation`` maps directly onto ``linalg.contract``.

    With *require_reduction*, equations without a contracted dimension (pure
    elementwise products and outer products) are excluded so they keep their
    natural elementwise lowering.
    """
    try:
        spec = build_contract_spec(equation, operand_shapes)
    except EinsumNotContractible:
        return False
    return bool(spec.reduction_dims) or not require_reduction
