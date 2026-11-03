"""Kernels of the forms ``mlir_transforms.xegpu_tile_forms`` rewrites, each with its float64
reference: computed operands then transposed, transposed elementwise tiles, ``sum(a * a)``
over a row, one tile as both DPAS operands, and a B tile with other users."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import itertools
import os
from typing import TYPE_CHECKING

import helion.language as hl
import torch

from tests import xpu_layout_forms as forms

if TYPE_CHECKING:
    from collections.abc import Callable

F32 = torch.float32
BF16 = torch.bfloat16


def _out(like: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return torch.empty([rows, cols], dtype=like.dtype, device=like.device)


def _round(x: torch.Tensor) -> torch.Tensor:
    return x.float().bfloat16().double()


def computed_a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a + 1)^T @ b``, with ``a``'s rounding to its own type."""
    out = _out(a, a.size(1), b.size(1))
    for tm, tn in hl.tile([a.size(1), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            acc = torch.addmm(acc, (a[tk, tm].to(F32) + 1.0).to(a.dtype).T, b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def computed_a_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a + 1.0).T @ b


def computed_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ (2 b)^T``."""
    out = _out(a, a.size(0), b.size(0))
    for tm, tn in hl.tile([a.size(0), b.size(0)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk], (b[tn, tk].to(F32) * 2.0).to(a.dtype).T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def computed_b_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ (b * 2.0).T


def computed_mixed(a: torch.Tensor, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a^T + z) @ b``: A's math has a transposed read and a plain one."""
    out = _out(a, a.size(1), b.size(1))
    for tm, tn in hl.tile([a.size(1), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            acc = torch.addmm(
                acc, (a[tk, tm].T.to(F32) + z[tm, tk].to(F32)).to(BF16), b[tk, tn]
            )
        out[tm, tn] = acc.to(a.dtype)
    return out


def computed_mixed_ref(
    a: torch.Tensor, z: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return _round(a.T + z) @ b


def computed_two_inputs(
    a: torch.Tensor, c: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``(a + c)^T @ b``: refused, see ``test_two_input_computed_operand_...``."""
    out = _out(a, a.size(1), b.size(1))
    for tm, tn in hl.tile([a.size(1), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            acc = torch.addmm(
                acc, (a[tk, tm].to(F32) + c[tk, tm].to(F32)).to(BF16).T, b[tk, tn]
            )
        out[tm, tn] = acc.to(a.dtype)
    return out


def elementwise_transposed(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        out[tm, tn] = (a[tm, tn].to(F32) + z[tn, tm].T.to(F32)).to(a.dtype)
    return out


def elementwise_transposed_ref(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return a + z.T


def only_transposed(z: torch.Tensor) -> torch.Tensor:
    out = _out(z, z.size(1), z.size(0))
    for tm, tn in hl.tile([z.size(1), z.size(0)]):
        out[tm, tn] = (z[tn, tm].T.to(F32) * 2.0).to(z.dtype)
    return out


def only_transposed_ref(z: torch.Tensor) -> torch.Tensor:
    return z.T * 2.0


def math_then_transposed(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        out[tm, tn] = (torch.exp(z[tn, tm].to(F32) * 0.5).T + a[tm, tn].to(F32)).to(
            a.dtype
        )
    return out


def math_then_transposed_ref(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.exp(z * 0.5).T + a


def tile_plus_transpose(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        t = a[tm, tn].to(F32)
        out[tm, tn] = (t + t.T).to(a.dtype)
    return out


def row_l2(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A GEMM, each row divided by its L2 norm."""
    m, k = x.size()
    n = w.size(0)
    out = _out(x, m, n)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[:, tk].T)
        out[tm, :] = (acc / torch.sqrt((acc * acc).sum(-1, keepdim=True))).to(x.dtype)
    return out


def row_l2_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    acc = x @ w.T
    return acc / torch.sqrt((acc * acc).sum(-1, keepdim=True))


def row_l2_of_input(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = _out(x, m, n)
    for tm in hl.tile(m):
        t = x[tm, :].to(F32)
        out[tm, :] = (t / torch.sqrt((t * t).sum(-1, keepdim=True))).to(x.dtype)
    return out


def row_l2_of_input_ref(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True))


def row_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = _out(x, m, n)
    for tm in hl.tile(m):
        t = x[tm, :].to(F32)
        out[tm, :] = (t * (t * y[tm, :].to(F32)).sum(-1, keepdim=True)).to(x.dtype)
    return out


def row_dot_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * (x * y).sum(-1, keepdim=True)


def syrk(a: torch.Tensor) -> torch.Tensor:
    """``a @ a^T``: one buffer, reads of two tiles."""
    m = a.size(0)
    out = _out(a, m, m)
    for tm, tn in hl.tile([m, m]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk], a[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def syrk_ref(a: torch.Tensor) -> torch.Tensor:
    return a @ a.T


def gram(a: torch.Tensor) -> torch.Tensor:
    """``a^T @ a``."""
    m = a.size(1)
    out = _out(a, m, m)
    for tm, tn in hl.tile([m, m]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            acc = torch.addmm(acc, a[tk, tm].T, a[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gram_ref(a: torch.Tensor) -> torch.Tensor:
    return a.T @ a


def tile_times_itself(a: torch.Tensor) -> torch.Tensor:
    """``t @ t``: one tile as A and as B, of ``M = K = N`` tiles."""
    n = a.size(0)
    out = _out(a, n, n)
    for tm, tn in hl.tile([n, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(n):
            t = a[tm, tk]
            acc = torch.addmm(acc, t, t)
        out[tm, tn] = acc.to(a.dtype)
    return out


def tile_times_transpose(a: torch.Tensor) -> torch.Tensor:
    """``t @ t^T``: one tile as A and, transposed, as B."""
    m = a.size(0)
    out = _out(a, m, m)
    for tm, tn in hl.tile([m, m]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            t = a[tm, tk]
            acc = torch.addmm(acc, t, t.T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def transpose_times_tile(a: torch.Tensor) -> torch.Tensor:
    """``t^T @ t``: one tile, transposed, as A and as B."""
    m = a.size(1)
    out = _out(a, m, m)
    for tm, tn in hl.tile([m, m]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            t = a[tk, tm]
            acc = torch.addmm(acc, t.T, t)
        out[tm, tn] = acc.to(a.dtype)
    return out


def b_with_other_user(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum(a[tm, tk] @ t + t)`` of ``t = b[tk, tn]``, ``M = K = N`` tiles."""
    out = _out(a, a.size(0), b.size(1))
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            t = b[tk, tn]
            acc = torch.addmm(acc, a[tm, tk], t)
            acc = acc + t.to(F32)
        out[tm, tn] = acc.to(a.dtype)
    return out


def a_with_other_user(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum(t @ b[tk, tn] + t)`` of ``t = a[tm, tk]``: one read serves both."""
    out = _out(a, a.size(0), b.size(1))
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            t = a[tm, tk]
            acc = torch.addmm(acc, t, b[tk, tn])
            acc = acc + t.to(F32)
        out[tm, tn] = acc.to(a.dtype)
    return out


def transposed_a_with_other_user(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum(t^T @ b[tk, tn] + t^T)`` of ``t = a[tk, tm]``."""
    out = _out(a, a.size(1), b.size(1))
    for tm, tn in hl.tile([a.size(1), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(0)):
            t = a[tk, tm].T
            acc = torch.addmm(acc, t, b[tk, tn])
            acc = acc + t.to(F32)
        out[tm, tn] = acc.to(a.dtype)
    return out


def transposed_b_with_other_user(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum(a[tm, tk] @ t^T + t^T)`` of ``t = b[tn, tk]``."""
    out = _out(a, a.size(0), b.size(0))
    for tm, tn in hl.tile([a.size(0), b.size(0)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            t = b[tn, tk].T
            acc = torch.addmm(acc, a[tm, tk], t)
            acc = acc + t.to(F32)
        out[tm, tn] = acc.to(a.dtype)
    return out


def in_place_transposed(a: torch.Tensor) -> torch.Tensor:
    """Refused: see ``test_in_place_transposed_read_is_refused``."""
    for tm, tn in hl.tile(list(a.size())):
        a[tm, tn] = a[tm, tn].T
    return a


def in_place_transposed_minus_one(a: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(list(a.size())):
        a[tm, tn] = (a[tm, tn].T.to(F32) - 1.0).to(a.dtype)
    return a


def in_place_math_transposed(a: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(list(a.size())):
        a[tm, tn] = (1.0 - a[tm, tn].to(F32)).T.to(a.dtype)
    return a


def in_place_transposed_plus_z(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(list(a.size())):
        a[tm, tn] = (a[tm, tn].T.to(F32) + z[tm, tn].to(F32)).to(a.dtype)
    return a


def in_place_transposed_after_a_gemm(
    x: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = w.size(0)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        z[tm, tn] = (acc + z[tm, tn].T.to(F32)).to(x.dtype)
    return z


def narrowed_operands(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """f32 tiles narrowed for a DPAS: refused."""
    out = _out(a, a.size(0), b.size(1))
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk].to(BF16), b[tk, tn].to(BF16))
        out[tm, tn] = acc.to(a.dtype)
    return out


def row_dot_with_transpose(a: torch.Tensor) -> torch.Tensor:
    """``sum(t * t^T, -1)``: refused, as a contraction that is no matmul."""
    out = torch.empty_like(a)
    for tm, tn in hl.tile(list(a.size())):
        t = a[tm, tn].to(F32)
        out[tm, tn] = (t * (t * t.T).sum(-1, keepdim=True)).to(a.dtype)
    return out


def one_row_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of ``a: [1, K]``: refused, see ``test_one_row_gemm_is_refused_clearly``."""
    m, k = a.size()
    n = b.size(1)
    out = _out(a, m, n)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=F32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _tiles(size: int, block: int) -> list[slice]:
    return [slice(i, i + block) for i in range(0, size, block)]


def tile_plus_transpose_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for tm in _tiles(a.size(0), blocks[0]):
            for tn in _tiles(a.size(1), blocks[1]):
                out[tm, tn] = a[tm, tn] + a[tm, tn].T
        return out

    return ref


def _per_tile(
    rows: int, cols: int, k: int, blocks: list[int], body: Callable[..., torch.Tensor]
) -> torch.Tensor:
    out = torch.zeros(rows, cols, dtype=torch.float64)
    for tm in _tiles(rows, blocks[0]):
        for tn in _tiles(cols, blocks[1]):
            for tk in _tiles(k, blocks[2]):
                out[tm, tn] += body(tm, tn, tk)
    return out


def a_with_other_user_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _per_tile(
            a.size(0),
            b.size(1),
            a.size(1),
            blocks,
            lambda m, n, k: a[m, k] @ b[k, n] + a[m, k],
        )

    return ref


def transposed_a_with_other_user_ref(
    blocks: list[int],
) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _per_tile(
            a.size(1),
            b.size(1),
            a.size(0),
            blocks,
            lambda m, n, k: a[k, m].T @ b[k, n] + a[k, m].T,
        )

    return ref


def transposed_b_with_other_user_ref(
    blocks: list[int],
) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _per_tile(
            a.size(0),
            b.size(0),
            a.size(1),
            blocks,
            lambda m, n, k: a[m, k] @ b[n, k].T + b[n, k].T,
        )

    return ref


def tile_times_itself_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor) -> torch.Tensor:
        n = a.size(0)
        return _per_tile(n, n, n, blocks, lambda m, _, k: a[m, k] @ a[m, k])

    return ref


def tile_times_transpose_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor) -> torch.Tensor:
        m = a.size(0)
        return _per_tile(m, m, a.size(1), blocks, lambda r, _, k: a[r, k] @ a[r, k].T)

    return ref


def transpose_times_tile_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor) -> torch.Tensor:
        m = a.size(1)
        return _per_tile(m, m, a.size(0), blocks, lambda r, _, k: a[k, r].T @ a[k, r])

    return ref


def b_with_other_user_ref(blocks: list[int]) -> Callable[..., torch.Tensor]:
    def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _per_tile(
            a.size(0),
            b.size(1),
            a.size(1),
            blocks,
            lambda m, n, k: a[m, k] @ b[k, n] + b[k, n],
        )

    return ref


@dataclass(frozen=True)
class Case:
    fn: Callable[..., torch.Tensor]
    ref: Callable[..., torch.Tensor]
    blocks: list[int]
    shapes: tuple[tuple[int, ...], ...]
    config: dict[str, object] = field(default_factory=dict)
    dtype: torch.dtype = BF16


FULL_SWEEP = bool(os.environ.get("HELION_TEST_FULL_SWEEP"))
_PER_KIND = 3
"""Cases per form and kind (no hint, each hint, ragged) by default: the first, the middle and the
last of the sweep, smallest to largest tile. ``HELION_TEST_FULL_SWEEP=1`` runs them all, which is
the proof for a change to the rule or the rewrites (13 minutes on one card)."""


def default_cases() -> dict[str, Case]:
    """What the device test runs by default: see :data:`_PER_KIND`."""
    every = cases()
    if FULL_SWEEP:
        return every
    kinds: dict[tuple[str, str, str], list[str]] = {}
    for name, case in every.items():
        ragged = any(size % 32 for shape in case.shapes for size in shape)
        hint = next(iter(case.config), "ragged" if ragged else "")
        kinds.setdefault((name.split(":")[0], str(case.dtype), hint), []).append(name)
    chosen = {
        names[index]
        for names in kinds.values()
        for index in {0, len(names) // 2, len(names) - 1}
    }
    return {name: every[name] for name in every if name in chosen}


def cases() -> dict[str, Case]:
    """Every shape the rule emits for these forms, with block sizes, rows and columns of each
    tile the grids can split, ragged shapes and what the hints can force: in this stack, a path
    no device ran is undefined. Names say ``form:blocks:shapes[:hints]``."""
    found: dict[str, Case] = {}

    def case(name, fn, ref, blocks, shapes, dtype=BF16, **config) -> None:
        tag = "" if dtype == BF16 else f"{str(dtype).split('.')[-1]}-"
        tag += f"{name}:{'x'.join(map(str, blocks))}:"
        tag += "_".join("x".join(map(str, shape)) for shape in shapes)
        if config:
            tag += ":" + "".join(
                f"{key.replace('xegpu_', '')}={value}" for key, value in config.items()
            ).replace(" ", "")
        found[tag] = Case(fn, ref, blocks, tuple(shapes), config, dtype)

    T = (32, 64, 128, 256)
    for bm, bn, bk in itertools.product(T, T, (32, 64)):
        M, N, K = 2 * bm, 2 * bn, 4 * bk
        case("computed_a", computed_a, computed_a_ref, [bm, bn, bk], [(K, M), (K, N)])
        case("computed_b", computed_b, computed_b_ref, [bm, bn, bk], [(M, K), (N, K)])
        case(
            "computed_mixed",
            computed_mixed,
            computed_mixed_ref,
            [bm, bn, bk],
            [(K, M), (M, K), (K, N)],
        )
        if bm == bn:
            case("syrk", syrk, syrk_ref, [bm, bn, bk], [(M, K)])
            case("gram", gram, gram_ref, [bm, bn, bk], [(K, M)])
            case(
                "tile_times_transpose",
                tile_times_transpose,
                tile_times_transpose_ref([bm, bn, bk]),
                [bm, bn, bk],
                [(M, K)],
            )
            case(
                "transpose_times_tile",
                transpose_times_tile,
                transpose_times_tile_ref([bm, bn, bk]),
                [bm, bn, bk],
                [(K, M)],
            )
    for b in T:
        case(
            "tile_times_itself",
            tile_times_itself,
            tile_times_itself_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b)],
        )
        case(
            "b_with_other_user",
            b_with_other_user,
            b_with_other_user_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b), (4 * b, 4 * b)],
        )
        case(
            "a_with_other_user",
            a_with_other_user,
            a_with_other_user_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b), (4 * b, 4 * b)],
        )
        case(
            "transposed_a_with_other_user",
            transposed_a_with_other_user,
            transposed_a_with_other_user_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b), (4 * b, 4 * b)],
        )
        case(
            "transposed_b_with_other_user",
            transposed_b_with_other_user,
            transposed_b_with_other_user_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b), (4 * b, 4 * b)],
        )
    for bm, bn in itertools.product((16, 32, 64, 128, 256, 512), repeat=2):
        M, N = 2 * bm, 2 * bn
        case(
            "elementwise_transposed",
            elementwise_transposed,
            elementwise_transposed_ref,
            [bm, bn],
            [(M, N), (N, M)],
        )
        case(
            "only_transposed", only_transposed, only_transposed_ref, [bm, bn], [(N, M)]
        )
        case(
            "math_then_transposed",
            math_then_transposed,
            math_then_transposed_ref,
            [bm, bn],
            [(N, M), (M, N)],
        )
        if bm == bn:
            case(
                "tile_plus_transpose",
                tile_plus_transpose,
                tile_plus_transpose_ref([bm, bn]),
                [bm, bn],
                [(M, N)],
            )
    for bm in (8, 16, 32, 64, 128, 256):
        for n in (32, 64, 128, 256, 512):
            case(
                "row_l2_of_input",
                row_l2_of_input,
                row_l2_of_input_ref,
                [bm],
                [(2 * bm, n)],
            )
            case("row_dot", row_dot, row_dot_ref, [bm], [(2 * bm, n), (2 * bm, n)])
            for bk in (32, 64):
                case(
                    "row_l2",
                    row_l2,
                    row_l2_ref,
                    [bm, bk],
                    [(2 * bm, 4 * bk), (n, 4 * bk)],
                )
    # Ragged: partial tiles of every form that has them.
    for M, N, K in (
        (248, 248, 96),
        (120, 248, 96),
        (248, 120, 160),
        (248, 248, 160),
        (120, 120, 120),
        (128, 128, 120),
        (248, 256, 96),
        (120, 256, 160),
    ):
        case("computed_a", computed_a, computed_a_ref, [64, 64, 32], [(K, M), (K, N)])
        case("computed_b", computed_b, computed_b_ref, [64, 64, 32], [(M, K), (N, K)])
        if N % 16 == 0:
            case("row_l2", row_l2, row_l2_ref, [64, 32], [(M, K), (N, K)])
        case(
            "elementwise_transposed",
            elementwise_transposed,
            elementwise_transposed_ref,
            [64, 64],
            [(M, N), (N, M)],
        )
        case(
            "only_transposed", only_transposed, only_transposed_ref, [64, 64], [(N, M)]
        )
        if N % 16 == 0:
            case(
                "row_l2_of_input", row_l2_of_input, row_l2_of_input_ref, [64], [(M, N)]
            )
        if N % 16 == 0:
            case("row_dot", row_dot, row_dot_ref, [64], [(M, N), (M, N)])
        case("syrk", syrk, syrk_ref, [64, 64, 32], [(M, K)])
        case("gram", gram, gram_ref, [64, 64, 32], [(K, M)])
        if M % 64 == 0:
            case(
                "tile_times_transpose",
                tile_times_transpose,
                tile_times_transpose_ref([64, 64, 32]),
                [64, 64, 32],
                [(M, K)],
            )
            case(
                "transpose_times_tile",
                transpose_times_tile,
                transpose_times_tile_ref([64, 64, 32]),
                [64, 64, 32],
                [(K, M)],
            )
    # Hints: what a config can force.
    for sg in (
        [8, 16],
        [16, 16],
        [32, 16],
        [32, 32],
        [16, 32],
        [16, 64],
        [8, 64],
        [8, 32],
    ):
        case(
            "computed_a",
            computed_a,
            computed_a_ref,
            [128, 128, 32],
            [(128, 256), (128, 256)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "computed_b",
            computed_b,
            computed_b_ref,
            [128, 128, 32],
            [(256, 128), (256, 128)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "elementwise_transposed",
            elementwise_transposed,
            elementwise_transposed_ref,
            [128, 128],
            [(256, 256), (256, 256)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "tile_plus_transpose",
            tile_plus_transpose,
            tile_plus_transpose_ref([128, 128]),
            [128, 128],
            [(256, 256)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "tile_times_transpose",
            tile_times_transpose,
            tile_times_transpose_ref([128, 128, 32]),
            [128, 128, 32],
            [(256, 128)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "b_with_other_user",
            b_with_other_user,
            b_with_other_user_ref([128, 128, 128]),
            [128, 128, 128],
            [(256, 256), (256, 256)],
            xegpu_sg_tiles=[sg],
        )
    for lt in ([1, 16], [2, 16], [4, 16], [8, 16], [16, 16], [32, 16]):
        case(
            "elementwise_transposed",
            elementwise_transposed,
            elementwise_transposed_ref,
            [128, 128],
            [(256, 256), (256, 256)],
            xegpu_load_tiles=[lt],
        )
        case(
            "only_transposed",
            only_transposed,
            only_transposed_ref,
            [128, 128],
            [(256, 256)],
            xegpu_load_tiles=[lt],
        )
        case(
            "tile_plus_transpose",
            tile_plus_transpose,
            tile_plus_transpose_ref([128, 128]),
            [128, 128],
            [(256, 256)],
            xegpu_load_tiles=[lt],
        )
        case(
            "math_then_transposed",
            math_then_transposed,
            math_then_transposed_ref,
            [128, 128],
            [(256, 256), (256, 256)],
            xegpu_load_tiles=[lt],
        )
        case(
            "computed_mixed",
            computed_mixed,
            computed_mixed_ref,
            [64, 64, 32],
            [(128, 128), (128, 128), (128, 128)],
            xegpu_load_tiles=[lt],
        )
    for lt in (
        [8, 16, 16, 16],
        [16, 32, 16, 32],
        [16, 16, 32, 32],
        [16, 32, 32, 32],
        [8, 32, 16, 16],
    ):
        case(
            "computed_a",
            computed_a,
            computed_a_ref,
            [64, 64, 32],
            [(128, 128), (128, 128)],
            xegpu_load_tiles=[lt],
        )
        case(
            "computed_b",
            computed_b,
            computed_b_ref,
            [64, 64, 32],
            [(128, 128), (128, 128)],
            xegpu_load_tiles=[lt],
        )
        case(
            "tile_times_itself",
            tile_times_itself,
            tile_times_itself_ref([64, 64, 64]),
            [64, 64, 64],
            [(128, 128)],
            xegpu_load_tiles=[lt],
        )
        case(
            "b_with_other_user",
            b_with_other_user,
            b_with_other_user_ref([64, 64, 64]),
            [64, 64, 64],
            [(128, 128), (128, 128)],
            xegpu_load_tiles=[lt],
        )
        case(
            "transpose_times_tile",
            transpose_times_tile,
            transpose_times_tile_ref([64, 64, 32]),
            [64, 64, 32],
            [(128, 128)],
            xegpu_load_tiles=[lt],
        )
    for lt in ([1, 16], [2, 16], [4, 16], [8, 16]):
        # A form with a transposed read before these rewrites: such a hint aborted.
        case(
            "gemm_plus_transposed",
            forms.gemm_plus_transposed,
            forms.gemm_plus_transposed_ref,
            [128, 128, 32],
            [(256, 256), (256, 256), (256, 256)],
            xegpu_load_tiles=[lt],
        )
    for depths in ([0, 1], [1, 0], [2, 2]):
        case(
            "computed_a",
            computed_a,
            computed_a_ref,
            [64, 64, 32],
            [(128, 128), (128, 128)],
            xegpu_prefetch_depths=[depths],
        )
        case(
            "computed_b",
            computed_b,
            computed_b_ref,
            [64, 64, 32],
            [(128, 128), (128, 128)],
            xegpu_prefetch_depths=[depths],
        )
        case(
            "tile_times_transpose",
            tile_times_transpose,
            tile_times_transpose_ref([64, 64, 32]),
            [64, 64, 32],
            [(128, 128)],
            xegpu_prefetch_depths=[depths],
        )
        case(
            "b_with_other_user",
            b_with_other_user,
            b_with_other_user_ref([64, 64, 64]),
            [64, 64, 64],
            [(128, 128), (128, 128)],
            xegpu_prefetch_depths=[depths],
        )
    for tiles in ([16, 16, 16, 16], [16, 32, 16, 32], [8, 16, 16, 32]):
        case(
            "computed_a",
            computed_a,
            computed_a_ref,
            [64, 64, 32],
            [(128, 128), (128, 128)],
            xegpu_prefetch_tiles=[tiles],
        )
        case(
            "tile_times_itself",
            tile_times_itself,
            tile_times_itself_ref([64, 64, 64]),
            [64, 64, 64],
            [(128, 128)],
            xegpu_prefetch_tiles=[tiles],
        )
        case(
            "transpose_times_tile",
            transpose_times_tile,
            transpose_times_tile_ref([64, 64, 32]),
            [64, 64, 32],
            [(128, 128)],
            xegpu_prefetch_tiles=[tiles],
        )
    # Hints on the forms with a row reduction or one buffer: what they can force.
    for sg in ([8, 0], [16, 0], [32, 0], [4, 0]):
        case(
            "row_l2",
            row_l2,
            row_l2_ref,
            [128, 32],
            [(256, 128), (64, 128)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "row_l2_of_input",
            row_l2_of_input,
            row_l2_of_input_ref,
            [128],
            [(256, 128)],
            xegpu_sg_tiles=[sg],
        )
        case(
            "row_dot",
            row_dot,
            row_dot_ref,
            [128],
            [(256, 128), (256, 128)],
            xegpu_sg_tiles=[sg],
        )
    for lt in ([2, 16], [4, 16], [8, 16], [16, 16]):
        case(
            "row_l2_of_input",
            row_l2_of_input,
            row_l2_of_input_ref,
            [128],
            [(256, 128)],
            xegpu_load_tiles=[lt],
        )
        case(
            "row_dot",
            row_dot,
            row_dot_ref,
            [128],
            [(256, 128), (256, 128)],
            xegpu_load_tiles=[lt],
        )
    for sg in ([8, 16], [16, 16], [32, 16], [32, 32], [16, 32], [16, 64], [8, 64]):
        case("syrk", syrk, syrk_ref, [128, 128, 32], [(256, 128)], xegpu_sg_tiles=[sg])
        case("gram", gram, gram_ref, [128, 128, 32], [(128, 256)], xegpu_sg_tiles=[sg])
    for lt in ([8, 16, 16, 16], [16, 32, 16, 32], [16, 16, 32, 32], [8, 32, 16, 16]):
        case("syrk", syrk, syrk_ref, [64, 64, 32], [(128, 128)], xegpu_load_tiles=[lt])
        case("gram", gram, gram_ref, [64, 64, 32], [(128, 128)], xegpu_load_tiles=[lt])
        case(
            "row_l2",
            row_l2,
            row_l2_ref,
            [128, 32],
            [(256, 128), (64, 128)],
            xegpu_load_tiles=[lt],
        )
    # Other element types, where a form has them: f32 has no transposed read and no DPAS.
    for block, rows, cols in ((64, 128, 256), (32, 64, 64), (128, 256, 512)):
        for dtype in (torch.float32, torch.float16):
            case(
                "row_l2_of_input",
                row_l2_of_input,
                row_l2_of_input_ref,
                [block],
                [(rows, cols)],
                dtype,
            )
            case(
                "row_dot",
                row_dot,
                row_dot_ref,
                [block],
                [(rows, cols), (rows, cols)],
                dtype,
            )
    f16 = torch.float16
    for b in (32, 64, 128):
        case(
            "computed_a",
            computed_a,
            computed_a_ref,
            [b, b, 32],
            [(128, 2 * b), (128, 2 * b)],
            f16,
        )
        case(
            "computed_b",
            computed_b,
            computed_b_ref,
            [b, b, 32],
            [(2 * b, 128), (2 * b, 128)],
            f16,
        )
        case(
            "elementwise_transposed",
            elementwise_transposed,
            elementwise_transposed_ref,
            [b, b],
            [(2 * b, 2 * b), (2 * b, 2 * b)],
            f16,
        )
        case(
            "only_transposed",
            only_transposed,
            only_transposed_ref,
            [b, b],
            [(2 * b, 4 * b)],
            f16,
        )
        case(
            "math_then_transposed",
            math_then_transposed,
            math_then_transposed_ref,
            [b, b],
            [(2 * b, 2 * b), (2 * b, 2 * b)],
            f16,
        )
        case(
            "tile_plus_transpose",
            tile_plus_transpose,
            tile_plus_transpose_ref([b, b]),
            [b, b],
            [(2 * b, 2 * b)],
            f16,
        )
        case(
            "tile_times_transpose",
            tile_times_transpose,
            tile_times_transpose_ref([b, b, 32]),
            [b, b, 32],
            [(2 * b, 128)],
            f16,
        )
        case(
            "b_with_other_user",
            b_with_other_user,
            b_with_other_user_ref([b, b, b]),
            [b, b, b],
            [(4 * b, 4 * b), (4 * b, 4 * b)],
            f16,
        )
    return found
