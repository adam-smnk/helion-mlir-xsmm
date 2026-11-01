"""``check_lane_reductions`` refuses a function with a ``sub_group_shuffle_xor``
chain that stops before mask 8, by def-use; nothing else. IR only."""

from __future__ import annotations

from itertools import count

from mlir import ir
import pytest

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms.xevm_lane_reductions import (
    check_lane_reductions,
)

_SHUFFLE = "_Z21sub_group_shuffle_xorfj"
_ids = count()


def _shuffle(value: str, mask: int) -> tuple[str, str]:
    """The ops of a shuffle of ``value`` and its result."""
    i = next(_ids)
    return (
        f"%m{i} = llvm.mlir.constant({mask} : i32) : i32\n"
        f"%s{i} = llvm.call spir_funccc @{_SHUFFLE}({value}, %m{i}) "
        ": (f32, i32) -> f32\n",
        f"%s{i}",
    )


def _combine(a: str, b: str, op: str = "fadd") -> tuple[str, str]:
    i = next(_ids)
    return f"%c{i} = llvm.{op} {a}, {b} : f32\n", f"%c{i}"


def _chain(value: str, masks: tuple[int, ...], op: str = "fadd") -> tuple[str, str]:
    """A reduction of ``value``: each mask shuffles the last combination."""
    text = ""
    for mask in masks:
        ops, shuffled = _shuffle(value, mask)
        combined, value = _combine(value, shuffled, op)
        text += ops + combined
    return text, value


def _gpu_module(body: str) -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @m {{
    llvm.func spir_funccc @{_SHUFFLE}(f32, i32) -> f32
    llvm.func spir_kernelcc @k(%v: f32, %w: f32, %n: i32) {{
      {body}
      llvm.return
    }}
  }}
}}"""


def _check(text: str) -> None:
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = False
        check_lane_reductions(ir.Module.parse(text))


def _merged_prefix() -> str:
    """Two reductions of ``%v``, whose first shuffles CSE merged: one mask-1
    shuffle, two of each other mask."""
    first, shuffled = _shuffle("%v", 1)
    add, a = _combine("%v", shuffled, "fadd")
    mul, b = _combine("%v", shuffled, "fmul")
    return first + add + mul + _chain(a, (2, 4, 8))[0] + _chain(b, (2, 4, 8))[0]


@pytest.mark.parametrize(
    "body",
    [
        lambda: "",
        lambda: _chain("%v", (1, 2, 4, 8))[0],
        lambda: _chain("%v", (1, 2, 4, 8))[0] + _chain("%w", (1, 2, 4, 8))[0],
        _merged_prefix,
    ],
    ids=["none", "full", "two-full", "merged-prefix"],
)
def test_full_chains_pass(body) -> None:
    _check(_gpu_module(body()))


@pytest.mark.parametrize(
    ("masks", "end"),
    [((1,), 1), ((1, 2), 2), ((1, 2, 4), 4)],
    ids=["pairs", "fours", "eights"],
)
def test_partial_chains_are_refused(masks: tuple[int, ...], end: int) -> None:
    with pytest.raises(
        UnsupportedOperationError, match=f"ends at shuffle mask {end} instead of 8"
    ):
        _check(_gpu_module(_chain("%v", masks)[0]))


def test_partial_chain_next_to_a_full_one_is_refused() -> None:
    with pytest.raises(UnsupportedOperationError, match="ends at shuffle mask 1"):
        _check(_gpu_module(_chain("%v", (1, 2, 4, 8))[0] + _chain("%w", (1,))[0]))


def test_balanced_counts_do_not_hide_a_partial_chain() -> None:
    """Counts of {1: 2, 2: 2, 4: 2, 8: 2} (a merged prefix and a wrong chain, as
    in a kernel with one wrong row store next to two right ones of one source)."""
    with pytest.raises(UnsupportedOperationError, match="ends at shuffle mask 1"):
        _check(_gpu_module(_merged_prefix() + _chain("%w", (1,))[0]))


def test_partial_chain_flowing_into_a_later_reduction_is_refused() -> None:
    """A later reduction of the wrong value starts over at mask 1: its mask 2 does
    not continue the first chain."""
    short, value = _chain("%v", (1,))
    with pytest.raises(UnsupportedOperationError, match="ends at shuffle mask 1"):
        _check(_gpu_module(short + _chain(value, (1, 2, 4, 8))[0]))


def test_shuffle_in_another_block_does_not_continue_a_chain() -> None:
    short, value = _chain("%v", (1,))
    later, _ = _shuffle(value, 2)
    with pytest.raises(UnsupportedOperationError, match="ends at shuffle mask 1"):
        _check(_gpu_module(f"{short}llvm.br ^next\n^next:\n{later}"))


def test_chain_through_conversions_passes() -> None:
    """Ops between the steps of a chain (e.g. a bf16 round trip)."""
    first, shuffled = _shuffle("%v", 1)
    negated = f"%neg = llvm.fneg {shuffled} : f32\n"
    combine, value = _combine("%v", "%neg")
    _check(_gpu_module(first + negated + combine + _chain(value, (2, 4, 8))[0]))


def test_other_shuffles_are_ignored() -> None:
    """A mask that is no constant and masks outside the chain mean nothing here;
    the host side is not looked at."""
    runtime = f"%x = llvm.call spir_funccc @{_SHUFFLE}(%v, %n) : (f32, i32) -> f32"
    _check(_gpu_module(runtime))
    _check(_gpu_module(_shuffle("%v", 3)[0]))
    _check(f"""
module {{
  llvm.func spir_funccc @{_SHUFFLE}(f32, i32) -> f32
  llvm.func @host(%v: f32) {{
    {_shuffle("%v", 1)[0]}
    llvm.return
  }}
}}""")
