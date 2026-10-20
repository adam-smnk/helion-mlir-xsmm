"""Structural matching of contractions in Helion device IR (read-only).

Every recognized contraction becomes one :class:`Contraction`, lowered by a single
emitter (``lowering/contraction_ops.py``). Matching is purely structural (targets,
literal scale factors, single use); type conditions such as accumulator shape and
dtype are checked at emission, which falls back when they do not hold.

Recognized, with anything outside these conditions left to the generic path:
- ``mm``/``bmm``/``matmul(a, b)`` of two rank-2 or two rank-3 operands;
- ``addmm``/``baddbmm(acc, a, b)`` with ``beta == alpha == 1``;
- ``hl.dot(a, b, acc=?, out_dtype=?)`` with same-rank operands;
- a captured ``helion_mlir::einsum`` over two operands;
- ``add.Tensor(acc, X)`` / ``(X, acc)`` with ``alpha == 1`` where ``X`` is one of the
  above without an accumulator and has no other user (``X`` is absorbed);
- an operand that is a transpose of its input's last two dims is folded into the
  operand's subscripts (and absorbed when that was its only use).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import helion.language.matmul_ops as matmul_ops
import torch
import torch.fx

from ..aten_bridge import original_args
from ..trace_mode import is_einsum_node

aten = torch.ops.aten

_MM = "mk,kn->mn"
_BMM = "bmk,bkn->bmn"
_RANK_EQUATIONS = {2: _MM, 3: _BMM}


@dataclass(frozen=True)
class Contraction:
    root: torch.fx.Node
    """The node whose value the contraction produces."""
    equation: str
    """Einsum-style equation over ``(lhs, rhs)``, transposes already folded."""
    lhs: torch.fx.Node
    rhs: torch.fx.Node
    acc: torch.fx.Node | None
    named: bool
    """Identity maps may use ``linalg.matmul``/``batch_matmul`` (not for einsum)."""
    absorbed: tuple[torch.fx.Node, ...] = ()
    """Nodes whose values this contraction replaces; they are not lowered."""
    unfused: Contraction | None = None
    """For an ``add`` root: the absorbed contraction, used when fusion is invalid."""


@dataclass
class ContractionPlan:
    roots: dict[torch.fx.Node, Contraction] = field(default_factory=dict)
    absorbed: set[torch.fx.Node] = field(default_factory=set)

    @classmethod
    def from_graphs(cls, graphs: list[torch.fx.Graph]) -> ContractionPlan:
        plan = cls()
        for graph in graphs:
            for node in graph.nodes:
                contraction = match_contraction(node)
                if contraction is not None:
                    plan._add(contraction)
                fused = _match_accumulate(node, plan)
                if fused is not None:
                    plan._add(fused)
        return plan

    def _add(self, contraction: Contraction) -> None:
        for node in contraction.absorbed:
            self.roots.pop(node, None)
            self.absorbed.add(node)
        self.roots[contraction.root] = contraction


def match_contraction(node: torch.fx.Node) -> Contraction | None:
    """The contraction ``node`` computes on its own (without add fusion)."""
    if node.op != "call_function":
        return None
    args, kwargs = original_args(node)
    args = list(args)
    if is_einsum_node(node):
        if (
            len(args) != 2
            or not isinstance(args[0], str)
            or not isinstance(args[1], (list, tuple))
            or len(args[1]) != 2
        ):
            return None
        return _with_operands(node, args[0], *args[1], acc=None, named=False)

    target = node.target
    if target in (aten.mm.default, aten.bmm.default, aten.matmul.default):
        return _ranked(node, args[0], args[1], acc=None)
    if target in (aten.addmm.default, aten.baddbmm.default):
        if _scalar_arg(node, 3, "beta") != 1 or _scalar_arg(node, 4, "alpha") != 1:
            return None
        return _ranked(node, args[1], args[2], acc=args[0])
    if target is matmul_ops.dot:
        acc = args[2] if len(args) > 2 else kwargs.get("acc")
        return _ranked(node, args[0], args[1], acc=acc)
    return None


def _match_accumulate(node: torch.fx.Node, plan: ContractionPlan) -> Contraction | None:
    if node.target is not aten.add.Tensor or _scalar_arg(node, 2, "alpha") != 1:
        return None
    first, second = original_args(node)[0][:2]
    for acc, operand in ((first, second), (second, first)):
        inner = plan.roots.get(operand) if isinstance(operand, torch.fx.Node) else None
        if (
            inner is None
            or inner.acc is not None
            or not isinstance(acc, torch.fx.Node)
            or acc is operand
            or len(operand.users) != 1
        ):
            continue
        return replace(
            inner,
            root=node,
            acc=acc,
            absorbed=(*inner.absorbed, operand),
            unfused=inner,
        )
    return None


def _ranked(
    node: torch.fx.Node, lhs: object, rhs: object, acc: object
) -> Contraction | None:
    ranks = {_rank(lhs), _rank(rhs)}
    if len(ranks) != 1:
        return None
    equation = _RANK_EQUATIONS.get(ranks.pop())
    if equation is None or (acc is not None and not isinstance(acc, torch.fx.Node)):
        return None
    return _with_operands(node, equation, lhs, rhs, acc=acc, named=True)


def _with_operands(
    node: torch.fx.Node,
    equation: str,
    lhs: object,
    rhs: object,
    *,
    acc: torch.fx.Node | None,
    named: bool,
) -> Contraction | None:
    if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
        return None
    inputs, arrow, output = equation.replace(" ", "").partition("->")
    subscripts = inputs.split(",")
    operands: list[torch.fx.Node] = []
    absorbed: list[torch.fx.Node] = []
    for position, operand in enumerate((lhs, rhs)):
        base = _transposed_input(operand)
        subscript = subscripts[position]
        if (
            base is not None
            and len(subscript) >= 2
            and not _is_vnni_operand(operand, subscript, output)
        ):
            subscripts[position] = subscript[:-2] + subscript[-1] + subscript[-2]
            if len(operand.users) == 1:
                absorbed.append(operand)
            operand = base
        operands.append(operand)
    return Contraction(
        root=node,
        equation=",".join(subscripts) + arrow + output,
        lhs=operands[0],
        rhs=operands[1],
        acc=acc,
        named=named,
        absorbed=tuple(absorbed),
    )


def _is_vnni_operand(operand: torch.fx.Node, subscript: str, output: str) -> bool:
    """Whether ``operand``'s innermost dim is a reduction over its dtype's VNNI
    factor: the layout AMX dot-products read, so a transpose producing it is a
    pack, not a view to fold."""
    value = operand.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        return False
    factor = _VNNI_FACTORS.get(value.dtype)
    return (
        factor is not None and value.shape[-1] == factor and subscript[-1] not in output
    )


_VNNI_FACTORS = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int8: 4,
    torch.uint8: 4,
}


def _transposed_input(node: torch.fx.Node) -> torch.fx.Node | None:
    """The input of a transpose that only swaps the last two dims, else ``None``."""
    permutation = transpose_permutation(node)
    if permutation is None or len(permutation) < 2:
        return None
    rank = len(permutation)
    if permutation != [*range(rank - 2), rank - 1, rank - 2]:
        return None
    base = node.args[0]
    return base if isinstance(base, torch.fx.Node) else None


def transpose_permutation(node: object) -> list[int] | None:
    """The permutation of a ``permute`` node (Helion traces ``t``/``transpose``/``.T``
    as ``permute``), if statically known."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is aten.permute.default
    ):
        return None
    dims = node.args[1]
    return [int(dim) for dim in dims] if isinstance(dims, (list, tuple)) else None


def _rank(node: object) -> int | None:
    value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return value.ndim if isinstance(value, torch.Tensor) else None


def _scalar_arg(node: torch.fx.Node, position: int, name: str) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name, 1)
