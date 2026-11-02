"""``copy_attention_layouts``: the layouts of a flash-attention kernel's first K/V loop
on its later ones.

An attention kernel may run the online softmax over K/V tiles in several loops, e.g.
tiles fully visible to a query tile, with no mask, and tiles that need one. Lighthouse's
``annotate_attention`` anchors the Q load, the K and V loads and the two ``xegpu.dpas`` of
one loop (it takes them by position). Every loop has the same operands and tiles, so the
later loops take the same anchors.

Matched exactly: a ``gpu.func`` with ``1 + 2 * n`` ``xegpu.load_nd``, ``2 * n``
``xegpu.dpas``, ``n >= 2``, whose first K, V and two ``xegpu.dpas`` are anchored and where
every K, V and pair of ``xegpu.dpas`` sit directly in one ``scf.for`` of their own; any
other function stays untouched.
"""

from __future__ import annotations

from mlir import ir

_LOAD_LAYOUT = "layout"
_DPAS_LAYOUTS = ("layout_a", "layout_b", "layout_cd")


def _of(function: ir.Operation, name: str) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == name:
            found.append(op)
        return ir.WalkResult.ADVANCE

    function.walk(visit)
    return found


def _rewrite(function: ir.Operation) -> bool:
    loads, dpas = _of(function, "xegpu.load_nd"), _of(function, "xegpu.dpas")
    loops = len(dpas) // 2
    if loops < 2 or len(dpas) % 2 or len(loads) != 1 + 2 * loops:
        return False
    groups = [
        [loads[1 + 2 * i], loads[2 + 2 * i], dpas[2 * i], dpas[2 * i + 1]]
        for i in range(loops)
    ]
    parents = [group[0].parent for group in groups]
    if any(
        parent is None
        or parent.name != "scf.for"
        or any(op.parent != parent for op in group)
        for parent, group in zip(parents, groups, strict=True)
    ):
        return False
    key, value, scores, update = groups[0]
    anchors = {
        0: (key, (_LOAD_LAYOUT,)),
        1: (value, (_LOAD_LAYOUT,)),
        2: (scores, _DPAS_LAYOUTS),
        3: (update, _DPAS_LAYOUTS),
    }
    if any(n not in op.attributes for op, names in anchors.values() for n in names):
        return False
    for group in groups[1:]:
        for position, (source, names) in anchors.items():
            for name in names:
                group[position].attributes[name] = source.attributes[name]
    return True


def copy_attention_layouts(module: ir.Module) -> int:
    """Anchors the later K/V loops of ``module``'s attention kernels; returns how many
    functions it changed."""
    functions: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "gpu.func":
            functions.append(op)
            return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return sum(_rewrite(function) for function in functions)
