"""Matmul on Helion's MLIR backend: AMX-friendly kernels with fused epilogues.

bf16 runs :func:`_matmul_fused_pack`: each output tile packs the B chunk it
needs per K step (``[K, N]`` or, with ``trans_b``, ``[N, K]``) into AMX's VNNI
layout in a private buffer; A is read in place or, with ``trans_a`` (``[K, M]``),
packed per tile too. Partial edge tiles are padded by the compiler; full tiles
run unpadded.
:func:`matmul_prepacked_b` with a weight packed once by :func:`pack_b_vnni_t`
skips the per-call pack: AMX reads both operands in place.

f32 GEMMs of whole tiles run :func:`_matmul_f32_fused_pack`: each output tile
packs the A and B chunks it needs per K step into 32x32 blocks in private
buffers and accumulates in blocks. Other f32 shapes (and f32 with a bias,
scale or shift) pack both operands into the blocked (mmt4d-style) layout by
separate kernels and run :func:`_matmul_blocked_kernel`:

    A = [M/BM, K/BK, BM, BK]
    B = [N/BN, K/BK, BK, BN]
    C = [M/BM, BM, N/BN, BN]   <- a free view of row-major [M, N]

Each packing kernel copies one 32x32 block per loop iteration and pads in the
same pass: a load past the end of the operand reads zeros. Storing C as
``[M/BM, BM, N/BN, BN]`` removes a separate unpack pass.

Packing is *not* hoisted out of :func:`matmul`: the work matches what an eager
``torch.matmul`` does with plain row-major inputs. An optional bias and epilogue
are fused into the accumulator before the store, so a fused linear+activation
costs one kernel instead of a matmul kernel followed by an elementwise one.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import TYPE_CHECKING
from typing import Callable

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.runtime.kernel import BoundKernel

try:
    from helion_mlir_backend import prepared_call as _prepared_call
except ImportError:  # a backend without native calls
    _prepared_call = None

# AMX bf16 register tile. All three extents must divide by this to use the
# blocked path.
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def identity_epilogue(x: Tensor) -> Tensor:
    return x


# Block sizes are literals in the packing loops: Helion cannot resolve a block
# size from a global name.
_PACK_CONFIG = helion.Config(block_sizes=[])


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pad_kernel(x: Tensor, rows: hl.constexpr, cols: hl.constexpr) -> Tensor:
    """``x`` zero-padded to ``[rows, cols]``: tiles past its end read zeros."""
    out = torch.empty((int(rows), int(cols)), dtype=x.dtype, device=x.device)
    for tr, tc in hl.tile([int(rows), int(cols)], block_size=[32, 32]):
        out[tr, tc] = x[tr, tc]
    return out


def pad_2d(x: Tensor, rows: int, cols: int) -> Tensor:
    """Contiguous ``x`` zero-padded to ``[rows, cols]``."""
    return _pad_kernel(x, hl.constexpr(rows), hl.constexpr(cols))


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_a_kernel(a: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    mb, kb = int(m_pad) // 32, int(k_pad) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a.dtype, device=a.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a[tm, tk]
    return out


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_a_kernel_t(a_t: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[K, M]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    mb, kb = int(m_pad) // 32, int(k_pad) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a_t.dtype, device=a_t.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a_t[tk, tm].permute(1, 0)
    return out


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_b_kernel(b: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N_pad/BN, K_pad/BK, BK, BN]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b[tk, tn]
    return out


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_b_kernel_t(b_t: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK, BN]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b_t[tn, tk].permute(1, 0)
    return out


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_b_vnni_kernel(b: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N_pad/BN, K_pad/BK, BK/2, BN, 2]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b[tk, tn].reshape(tk.block_size // 2, 2, tn.block_size)
        out[tn.id, tk.id, :, :, :] = pairs.permute(0, 2, 1)
    return out


@helion.kernel(backend="mlir", config=_PACK_CONFIG)
def _pack_b_vnni_kernel_t(
    b_t: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr
) -> Tensor:
    """Pack transposed ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK/2, BN, 2]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b_t[tn, tk].reshape(tn.block_size, tk.block_size // 2, 2)
        out[tn.id, tk.id, :, :, :] = pairs.permute(1, 0, 2)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1, 1]))
def _matmul_blocked_kernel(
    a4: Tensor, b4: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """``[MB, KB, BM, BK] x [NB, KB, BK, BN] -> [MB, BM, NB, BN]``, epilogue fused in."""
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"
    # Static under dynamic shapes: the accumulator's block dims and K.
    block_m = hl.specialize(block_m)
    block_n = hl.specialize(block_n)
    hl.specialize(blocks_k)
    hl.specialize(block_k)

    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        y = epilogue(acc)
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1, 1]))
def _matmul_blocked_kernel_bias(
    a4: Tensor, b4: Tensor, bias3: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """Like ``_matmul_blocked_kernel``, plus a ``[NB, 1, BN]`` bias fused before the epilogue.

    ``bias3`` must stay an explicit kernel argument, not folded into a
    closure captured by ``epilogue``: a closure-captured tensor traces
    incorrectly on this backend (produced garbage, not just stale values, in
    testing) since it never becomes a real kernel input/memref.
    """
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"
    # Static under dynamic shapes: the accumulator's block dims and K.
    block_m = hl.specialize(block_m)
    block_n = hl.specialize(block_n)
    hl.specialize(blocks_k)
    hl.specialize(block_k)
    hl.specialize(bias3.size(1))

    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        y = epilogue(acc + bias3[tile_blocks_n, :, :])
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


def _chunk_dot(
    a: Tensor,
    b: Tensor,
    tile_m: object,
    tile_n: object,
    tile_k: object,
    trans_a: bool,
    trans_b: bool,
) -> Tensor:
    """``a @ b`` of bf16 ``[M, K]`` and ``[K, N]`` (``a`` ``[K, M]`` with
    ``trans_a``, ``b`` ``[N, K]`` with ``trans_b``) on one tile and K chunk,
    each operand packed into the VNNI layout of AMX tiles: ``[M, K/2, 2]``
    and ``[K/2, N, 2]``."""
    if trans_a:
        a_tile = a[tile_k, tile_m]
        a3 = a_tile.reshape(a_tile.size(0) // 2, 2, a_tile.size(1)).permute(2, 0, 1)
    else:
        a_tile = a[tile_m, tile_k]
        a3 = a_tile.reshape(a_tile.size(0), a_tile.size(1) // 2, 2)
    if trans_b:
        b_tile = b[tile_n, tile_k]
        b3 = b_tile.reshape(b_tile.size(0), b_tile.size(1) // 2, 2).permute(1, 0, 2)
    else:
        b_tile = b[tile_k, tile_n]
        b3 = b_tile.reshape(b_tile.size(0) // 2, 2, b_tile.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _matmul_fused_pack(
    a: Tensor,
    b: Tensor,
    bias: Tensor | None,
    scale: Tensor | None,
    shift: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    k_chunked: hl.constexpr,
    k_even: hl.constexpr,
    trans_a: hl.constexpr,
    trans_b: hl.constexpr,
    out: Tensor | None = None,
) -> Tensor:
    """``epilogue((a @ b + bias) * scale + shift)`` of bf16 row-major ``[M, K]``
    and ``[K, N]`` (``[K, M]`` with ``trans_a``, ``[N, K]`` with ``trans_b``);
    ``bias``, ``scale`` and ``shift`` are ``[N]`` or ``None``; written into
    ``out`` if given.

    Each output tile packs the chunks it needs per K step into AMX's layouts,
    in private buffers: B's chunk always (VNNI), reused for all its rows; A's
    with ``trans_a``, reused for all its columns. Packed operands never leave
    the core (a separate pack kernel's output is read by many cores, and
    rewriting those lines on the next call costs more than the GEMM saves).

    Tiles past the end of M or N are partial (zero-padded loads, masked
    stores); the compiler runs full tiles without padding. K runs in chunks up
    to ``k_chunked``, one chunk of the rest up to ``k_even`` (a multiple of 64:
    whole AMX steps), then a partial chunk of 64 for the remaining columns.
    """
    if trans_a:
        k, m = a.shape
    else:
        m, k = a.shape
    if trans_b:
        n, k2 = b.shape
    else:
        k2, n = b.shape
    assert k == k2, "K mismatch"
    # With dynamic shapes, a static K keeps the K chunks unmasked.
    k = hl.specialize(k)
    hl.specialize(k2)

    if out is None:
        result = torch.empty((m, n), dtype=a.dtype, device=a.device)
    else:
        result = out
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k_chunked):
            acc = acc + _chunk_dot(a, b, tile_m, tile_n, tile_k, trans_a, trans_b)
        if k_even > k_chunked:
            for tile_rest in hl.tile(k_chunked, k_even):
                acc = acc + _chunk_dot(
                    a, b, tile_m, tile_n, tile_rest, trans_a, trans_b
                )
        if k > k_even:
            # A tile as wide as the loop; past K, both operands read zeros.
            for tile_tail in hl.tile(k_even, k_even + 64, block_size=64):
                acc = acc + _chunk_dot(
                    a, b, tile_m, tile_n, tile_tail, trans_a, trans_b
                )
        if bias is not None:
            acc = acc + bias[tile_n]
        if scale is not None:
            acc = acc * scale[tile_n]
        if shift is not None:
            acc = acc + shift[tile_n]
        result[tile_m, tile_n] = epilogue(acc).to(a.dtype)
    return result


_FUSED_PACK_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}
# Helion flags the host-side `out is None` check as an unfused tensor operation.
_OUT_CHECK_WARNINGS = [helion.exc.TensorOperationInWrapper]
# LRU: epilogues created per call must not pile up (each entry pins its epilogue).
_FUSED_PACK_BOUND: OrderedDict[
    tuple[object, ...], tuple[BoundKernel, tuple[hl.constexpr, ...]]
] = OrderedDict()
_FUSED_PACK_BOUND_SIZE = 64
# K elements per chunk of a 128-column tile: its packed B chunk stays within 512 KiB.
_MAX_K_CHUNK = 2048
# K elements of a 128-column tile run as one chunk (768 KiB of packed B).
_MAX_K_SINGLE = 3072
# bf16 elements of a cache line.
_LINE_ELEMENTS = 32
# Rows of a tile below which packing its B chunks costs more than sharing them.
_MIN_SHARED_ROWS = 256
# Columns of a VNNI panel of :func:`pack_b_vnni_t` (literal in its pack kernel).
_VNNI_PANEL = 64
# K pairs per tile of :func:`pack_b_vnni_t` (literal in its pack kernel).
_PANEL_PAIRS = 16


def _fused_pack_tiles(
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    tiles: Sequence[int] | None = None,
) -> tuple[list[int], int, int]:
    """Block sizes of :func:`_matmul_fused_pack` (rows, columns, K chunk, rest
    chunk), and its ``k_chunked`` and ``k_even``; ``tiles`` (rows, columns, K
    chunk) overrides the choice below.

    Tiles are columns of up to 128 (fewest B chunk packs), as tall as one tile
    per thread allows (512x128 best at 2K, 2048x128 at 4K); narrower columns
    when that leaves tiles of fewer than ``_MIN_SHARED_ROWS`` rows. With
    ``trans_a``, A chunks are packed per tile too: columns widen until tiles
    are about square, which balances the two packs. Rows of tiles are evened
    out: a ragged M leaves no near-empty last row of tiles, which would run
    whole register tiles of padding. The f32 accumulator stays within 1 MiB,
    the stack promotion limit of ``pipeline.yaml``: heap buffers per tile cost
    malloc and page faults. K runs in the fewest chunks whose packed operands
    stay within 512 KiB each.
    """
    k_even = k // (2 * BLOCK_K) * 2 * BLOCK_K
    if tiles is not None:
        tile_m, tile_n, tile_k = tiles
    else:
        tile_m, tile_n, tile_k = _fused_pack_heuristic(m, n, k, k_even, trans_a)
    k_chunked = k_even // tile_k * tile_k
    rest = max(k_even - k_chunked, 2 * BLOCK_K)
    return [tile_m, tile_n, tile_k, rest], k_chunked, k_even


def _fused_pack_heuristic(
    m: int, n: int, k: int, k_even: int, trans_a: bool
) -> tuple[int, int, int]:
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))

    def rows(tile_n: int) -> tuple[int, int]:
        tiles_m = max(1, threads // -(-n // tile_n))
        tile_m = min(
            _round_up(-(-m // tiles_m), BLOCK_M),
            m // BLOCK_M * BLOCK_M,
            (1 << 18) // tile_n,
        )
        return tiles_m, tile_m

    tile_n = min(4 * BLOCK_N, n // BLOCK_N * BLOCK_N)
    tiles_m, tile_m = rows(tile_n)
    # Few rows: narrower columns, each B chunk packed by one tile, not several.
    while tiles_m > 1 and tile_m < _MIN_SHARED_ROWS and tile_n > BLOCK_N:
        tile_n = max(tile_n // 2 // BLOCK_N * BLOCK_N, BLOCK_N)
        tiles_m, tile_m = rows(tile_n)
    # All rows in one tile: no B chunk is shared, more columns of tiles balance
    # the threads (128x32768x32768: 15% faster at 64 columns than 128); K
    # chunks stay those of the wider tile.
    chunk_cols = tile_n
    if not trans_a and tiles_m == 1 and tile_m < _MIN_SHARED_ROWS:
        tile_n = min(tile_n, 2 * BLOCK_N)
        tiles_m, tile_m = rows(tile_n)
    while trans_a and tile_n < tile_m and 2 * tile_n <= n:
        tile_n *= 2
        tiles_m, tile_m = rows(tile_n)
        chunk_cols = tile_n
    tile_m = _balanced_rows(m, -(-n // tile_n), tile_m, threads)
    # A single-chunk K loop is what the AMX rewrite of a one-register-tile
    # accumulator handles.
    packed = max(chunk_cols, tile_m if trans_a else 0)
    max_chunk = _MAX_K_CHUNK * 4 * BLOCK_N // packed
    # A read in place from aligned rows: one chunk while its packed B fits
    # 768 KiB (8192x5888x2944: 16% faster than two; 4096^3: 16% slower).
    if (
        not trans_a
        and not k % _LINE_ELEMENTS
        and k_even <= _MAX_K_SINGLE * 4 * BLOCK_N // chunk_cols
    ):
        max_chunk = max(max_chunk, k_even)
    chunks = -(-k_even // max_chunk)
    tile_k = _round_up(-(-k_even // chunks), 2 * BLOCK_K)
    return tile_m, tile_n, tile_k


def _balanced_rows(m: int, cols: int, tile_m: int, threads: int) -> int:
    """Rows of tiles at most ``tile_m`` (and, if it is, at least
    ``_MIN_SHARED_ROWS``) whose busiest thread computes the fewest rows: tiles
    in waves of ``threads``, the last wave as long as the others. Ties go to
    the fewest tiles. E.g. 8205 rows by 47 columns of tiles on 64 threads:
    8 rows of tiles of 1056 (6 waves), not 5 of 1664 (4 waves, 8% slower)."""
    fewest = -(-m // tile_m)
    smallest = min(_MIN_SHARED_ROWS, tile_m)

    def rows(count: int) -> int:
        return _round_up(-(-m // count), BLOCK_M)

    counts = [fewest] + [
        count for count in range(fewest + 1, 4 * fewest + 1) if rows(count) >= smallest
    ]
    best = min(
        counts, key=lambda count: (-(-count * cols // threads) * rows(count), count)
    )
    return rows(best)


def _matmul_fused_pack_bound(
    a: Tensor,
    b: Tensor,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    trans_a: bool,
    trans_b: bool,
    out: Tensor | None = None,
    tiles: Sequence[int] | None = None,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> tuple[BoundKernel, tuple[hl.constexpr, ...]]:
    """The fused-pack kernel bound to contiguous ``a``, ``b``, ``bias``,
    ``scale``, ``shift``, ``epilogue`` and ``out``, with its constexpr
    arguments; ``tiles`` as in :func:`_fused_pack_tiles`. Call it as
    ``bound(a, b, bias, scale, shift, epilogue, *consts, out)``.

    Cached by problem size and epilogue: the tile choice and Helion's
    specialization lookup (no fast path for callable arguments) cost tens of
    microseconds per call.
    """
    m, k = a.shape[::-1] if trans_a else a.shape
    n = b.shape[0] if trans_b else b.shape[1]
    key = (
        m,
        n,
        k,
        trans_a,
        trans_b,
        bias is None,
        scale is None,
        shift is None,
        out is None,
        None if tiles is None else tuple(tiles),
        os.environ.get("OMP_NUM_THREADS"),
        epilogue,
    )
    if (cached := _FUSED_PACK_BOUND.get(key)) is not None:
        _FUSED_PACK_BOUND.move_to_end(key)
        return cached
    block_sizes, k_chunked, k_even = _fused_pack_tiles(m, n, k, trans_a, tiles)
    if (config := tuple(block_sizes)) not in _FUSED_PACK_KERNELS:
        _FUSED_PACK_KERNELS[config] = helion.kernel(
            _matmul_fused_pack,
            backend="mlir",
            config=helion.Config(block_sizes=block_sizes),
            ignore_warnings=_OUT_CHECK_WARNINGS,
        )
    consts = (
        hl.constexpr(k_chunked),
        hl.constexpr(k_even),
        hl.constexpr(trans_a),
        hl.constexpr(trans_b),
    )
    bound = _FUSED_PACK_KERNELS[config].bind(
        (a, b, bias, scale, shift, epilogue, *consts, out)
    )
    cached = _FUSED_PACK_BOUND[key] = (bound, consts)
    if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
        _FUSED_PACK_BOUND.popitem(last=False)
    return cached


def _panel_dot(
    a3: Tensor, b4: Tensor, tile_m: object, tile_p: object, tile_kp: object
) -> Tensor:
    return torch.einsum(
        "mcv,bcnv->mbn", a3[tile_m, tile_kp, :], b4[tile_p, tile_kp, :, :]
    )


def _matmul_prepacked_vnni(
    a3: Tensor,
    b4: Tensor,
    bias2: Tensor | None,
    scale2: Tensor | None,
    shift2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    pairs_chunked: hl.constexpr,
    pairs_even: hl.constexpr,
    out: Tensor | None = None,
) -> Tensor:
    """``epilogue((a @ b + bias) * scale + shift)`` of ``a3``, row-major
    ``[M, K]`` viewed as K pairs ``[M, K/2, 2]``, and ``b4``, B in contiguous
    column panels of AMX's VNNI layout ``[N/P, K/2, P, 2]``
    (:func:`pack_b_vnni_t`); ``bias2``, ``scale2`` and ``shift2`` are
    ``[N/P, P]`` or ``None``. Tiles are one panel wide; written into ``out``
    (``[M, N/P, P]``) if given.

    AMX loads both operands' tiles in place: nothing is packed per call. K
    pairs run in chunks up to ``pairs_chunked``, one chunk of the rest up to
    ``pairs_even`` (whole pairs of AMX steps), then a padded chunk of 32 for
    the remaining pairs: a ragged chunk would leave masked AMX steps, which
    lower to per-element code.
    """
    m, pairs, _ = a3.shape
    panels, _, panel, _ = b4.shape
    # With dynamic shapes, static K pairs keep the K chunks unmasked.
    pairs = hl.specialize(pairs)
    panel = hl.specialize(panel)
    hl.specialize(b4.size(1))
    hl.specialize(a3.size(2))
    hl.specialize(b4.size(3))

    if out is None:
        result = torch.empty((m, panels, panel), dtype=a3.dtype, device=a3.device)
    else:
        result = out
    for tile_m, tile_p in hl.tile([m, panels]):
        acc = hl.zeros([tile_m, tile_p, panel], dtype=torch.float32)
        if pairs_chunked > 0:
            for tile_kp in hl.tile(pairs_chunked):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_kp)
        if pairs_even > pairs_chunked:
            for tile_rest in hl.tile(pairs_chunked, pairs_even):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_rest)
        if pairs > pairs_even:
            for tile_tail in hl.tile(pairs_even, pairs_even + 32, block_size=32):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_tail)
        if bias2 is not None:
            acc = acc + bias2[tile_p, :]
        if scale2 is not None:
            acc = acc * scale2[tile_p, :]
        if shift2 is not None:
            acc = acc + shift2[tile_p, :]
        result[tile_m, tile_p, :] = epilogue(acc).to(a3.dtype)
    return result


def _matmul_prepacked_rows(
    a: Tensor,
    b3: Tensor,
    bias2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    out: Tensor | None = None,
) -> Tensor:
    """``epilogue(a @ b + bias)`` of row-major ``a`` ``[M, K]``, read in place,
    and ``b3``, B in contiguous column panels ``[N/BN, K, BN]`` (the layout of
    :func:`pack_b_blocked_t`, viewed); ``bias2`` is ``[N/BN, BN]`` or ``None``.
    Written into ``out`` (``[M, N/BN, BN]``) if given.

    K runs in chunks (the last block size) small enough that a register tile's
    panel chunk stays in L1 across the rows of the tile.
    """
    m, k = a.shape
    panels, _, panel = b3.shape
    k = hl.specialize(k)
    panel = hl.specialize(panel)
    hl.specialize(b3.size(1))
    if out is None:
        result = torch.empty((m, panels, panel), dtype=a.dtype, device=a.device)
    else:
        result = out
    for tile_m, tile_p in hl.tile([m, panels]):
        acc = hl.zeros([tile_m, tile_p, panel], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.einsum(
                "mk,bkn->mbn", a[tile_m, tile_k], b3[tile_p, tile_k, :]
            )
        if bias2 is not None:
            acc = acc + bias2[tile_p, :]
        result[tile_m, tile_p, :] = epilogue(acc).to(a.dtype)
    return result


# K elements per chunk of :func:`_matmul_prepacked_rows`: a 256 x 32 f32 panel
# chunk is 32 KiB, within the 48 KiB L1D, so it serves every 8-row register tile
# of a 32-row tile from L1 (128x4096x4096 on EMR: 605 us; 512: 1140 us, the
# chunk no longer fits; 128: 595 us).
_ROWS_K_CHUNK = 256
_ROWS_KERNELS: dict[int, helion.Kernel] = {}


def _rows_bound(
    a: Tensor,
    b3: Tensor,
    bias2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    out: Tensor | None = None,
) -> BoundKernel:
    """:func:`_matmul_prepacked_rows` of 32-row, one-panel tiles bound to the
    operands (and ``out`` if given); cached like the fused kernel."""
    m, k = map(int, a.shape)
    panels, _, panel = map(int, b3.shape)
    key = (
        "rows",
        m,
        k,
        panels * panel,
        a.dtype,
        a.device,
        out is None,
        bias2 is None,
        epilogue,
    )
    if (cached := _FUSED_PACK_BOUND.get(key)) is None:
        # The largest multiple of BK up to _ROWS_K_CHUNK dividing K: a ragged
        # chunk would be masked.
        chunk = next(c for c in range(_ROWS_K_CHUNK, 0, -BLOCK_K) if k % c == 0)
        if chunk not in _ROWS_KERNELS:
            _ROWS_KERNELS[chunk] = helion.kernel(
                _matmul_prepacked_rows,
                backend="mlir",
                config=helion.Config(block_sizes=[BLOCK_M, 1, chunk]),
                ignore_warnings=_OUT_CHECK_WARNINGS,
            )
        cached = _FUSED_PACK_BOUND[key] = (
            _ROWS_KERNELS[chunk].bind((a, b3, bias2, epilogue, out)),
            (),
        )
        if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
            _FUSED_PACK_BOUND.popitem(last=False)
    else:
        _FUSED_PACK_BOUND.move_to_end(key)
    return cached[0]


def _rows_operands(
    b4: Tensor, n: int, bias: Tensor | None
) -> tuple[Tensor, Tensor | None]:
    """``b4`` (:func:`pack_b_blocked_t`) as panels and ``bias`` as ``[N/BN, BN]``."""
    blocks_n, blocks_k, _, _ = map(int, b4.shape)
    n_pad = blocks_n * BLOCK_N
    b3 = b4.view(blocks_n, blocks_k * BLOCK_K, BLOCK_N)
    if bias is None:
        return b3, None
    row = bias.reshape(1, n)
    return b3, (row if n == n_pad else pad_2d(row, 1, n_pad)).view(blocks_n, BLOCK_N)


def _matmul_prepacked_rows_call(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """:func:`matmul_prepacked_b` of an f32 RHS packed by :func:`pack_b_blocked_t`
    and an activation whose K is a whole number of blocks.

    Rows past the last whole 32-row tile run as a second call, zero-padded to
    a whole tile: a partial row tile would be masked.
    """
    m, k = map(int, a.shape)
    b3, bias2 = _rows_operands(b4, n, bias)
    n_pad = int(b3.shape[0]) * BLOCK_N
    a = a.contiguous()
    bulk = m // BLOCK_M * BLOCK_M
    if bulk == m:
        out = _rows_bound(a, b3, bias2, epilogue)(a, b3, bias2, epilogue)
        out = out.view(m, n_pad)
        return out if n == n_pad else out[:, :n]
    out3 = torch.empty(
        (bulk + BLOCK_M, int(b3.shape[0]), BLOCK_N), dtype=a.dtype, device=a.device
    )
    for rows, out_rows in (
        (a[:bulk], out3[:bulk]),
        (pad_2d(a[bulk:].contiguous(), BLOCK_M, k), out3[bulk:]),
    ):
        if rows.shape[0]:
            bound = _rows_bound(rows, b3, bias2, epilogue, out_rows)
            bound(rows, b3, bias2, epilogue, out_rows)
    out = out3[:m].view(m, n_pad)
    return out if n == n_pad else out[:, :n]


_PREPACKED_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}


def _matmul_prepacked_vnni_call(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """:func:`matmul_prepacked_b` of a RHS packed by :func:`pack_b_vnni_t`.

    Rows past the last whole row tile run as a second call, zero-padded to
    whole register tiles: a partial row tile lowers to per-element code (about
    35 ms for one tile, 12x the whole GEMM at 131104x256x1536).
    """
    panels, pairs_b, panel, vnni = map(int, b4.shape)
    m, k = map(int, a.shape)
    n_pad = panels * panel
    if (
        a.dtype != torch.bfloat16
        or b4.dtype != a.dtype
        or vnni != 2
        or panel % BLOCK_N
        or k % 2
        or 2 * pairs_b != _round_up(k, 2 * _PANEL_PAIRS)
        or _round_up(n, panel) != n_pad
    ):
        raise ValueError(
            f"VNNI-packed RHS {tuple(b4.shape)} {b4.dtype} is incompatible with "
            f"a.shape={tuple(a.shape)} {a.dtype} and n={n}"
        )
    a3 = a.contiguous().view(m, k // 2, 2)

    def panels2(vector: Tensor | None) -> Tensor | None:
        if vector is None:
            return None
        row = vector.reshape(1, n)
        return (row if n == n_pad else pad_2d(row, 1, n_pad)).view(panels, panel)

    bias2, scale2, shift2 = panels2(bias), panels2(scale), panels2(shift)
    operands = (b4, bias2, scale2, shift2, epilogue)
    tile_m = _prepacked_tile_m(m, panels, panel)
    bulk = m // tile_m * tile_m
    if bulk == m:
        bound, consts = _prepacked_bound(a3, *operands, tile_m)
        out = bound(a3, *operands, *consts).view(m, n_pad)
        return out if n == n_pad else out[:, :n]

    rest = _round_up(m - bulk, BLOCK_M)
    out3 = torch.empty((bulk + rest, panels, panel), dtype=a.dtype, device=a.device)
    if bulk:
        bound, consts = _prepacked_bound(a3[:bulk], *operands, tile_m, out3[:bulk])
        bound(a3[:bulk], *operands, *consts, out3[:bulk])
    a_rest = a[bulk:]
    if rest != m - bulk:
        a_rest = pad_2d(a_rest.contiguous(), rest, k)
    a3_rest = a_rest.contiguous().view(rest, k // 2, 2)
    bound, consts = _prepacked_bound(a3_rest, *operands, rest, out3[bulk:])
    bound(a3_rest, *operands, *consts, out3[bulk:])
    out = out3[:m].view(m, n_pad)
    return out if n == n_pad else out[:, :n]


def _prepacked_tile_m(m: int, panels: int, panel: int) -> int:
    """Rows of a :func:`_matmul_prepacked_vnni` tile: every thread a tile, the
    f32 accumulator within 1 MiB, whole register tiles (a ragged row tile loses
    AMX: 768-column panels gave 341 rows)."""
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    tiles_m = max(1, threads // panels)
    return min(
        _round_up(-(-m // tiles_m), BLOCK_M),
        max(m // BLOCK_M * BLOCK_M, BLOCK_M),
        max((1 << 18) // panel // BLOCK_M * BLOCK_M, BLOCK_M),
    )


def _prepacked_bound(
    a3: Tensor,
    b4: Tensor,
    bias2: Tensor | None,
    scale2: Tensor | None,
    shift2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    tile_m: int,
    out: Tensor | None = None,
) -> tuple[BoundKernel, tuple[hl.constexpr, ...]]:
    """:func:`_matmul_prepacked_vnni` of ``tile_m``-row tiles bound to the
    operands (and ``out`` if given), with its constexpr arguments; cached like
    the fused kernel."""
    m, pairs, _ = map(int, a3.shape)
    panels, _, panel, _ = map(int, b4.shape)
    key = (
        "prepacked",
        m,
        panels * panel,
        panel,
        pairs,
        tile_m,
        out is None,
        bias2 is None,
        scale2 is None,
        shift2 is None,
        os.environ.get("OMP_NUM_THREADS"),
        epilogue,
    )
    if (cached := _FUSED_PACK_BOUND.get(key)) is None:
        pairs_even = pairs // BLOCK_K * BLOCK_K
        # Packed B per K chunk: 512 KiB, 256 KiB for tiles of few rows, which
        # stream the weight (128x32768x32768 at 256 columns: 8.7 ms, 9.2 at 512 KiB).
        chunk_bytes = (256 if tile_m < _MIN_SHARED_ROWS else 512) << 10
        max_pairs = max(chunk_bytes // (panel * 2 * b4.element_size()), BLOCK_K)
        chunks = max(1, -(-pairs_even // max_pairs))
        chunk = max(_round_up(-(-pairs_even // chunks), BLOCK_K), BLOCK_K)
        pairs_chunked = pairs_even // chunk * chunk
        tiles = (tile_m, 1, chunk, max(pairs_even - pairs_chunked, BLOCK_K))
        if tiles not in _PREPACKED_KERNELS:
            _PREPACKED_KERNELS[tiles] = helion.kernel(
                _matmul_prepacked_vnni,
                backend="mlir",
                config=helion.Config(block_sizes=list(tiles)),
                ignore_warnings=_OUT_CHECK_WARNINGS,
            )
        consts = (hl.constexpr(pairs_chunked), hl.constexpr(pairs_even))
        bound = _PREPACKED_KERNELS[tiles].bind(
            (a3, b4, bias2, scale2, shift2, epilogue, *consts, out)
        )
        cached = _FUSED_PACK_BOUND[key] = (bound, consts)
        if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
            _FUSED_PACK_BOUND.popitem(last=False)
    else:
        _FUSED_PACK_BOUND.move_to_end(key)
    return cached


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1, 1]))
def _matmul_blocked_kernel_affine(
    a4: Tensor,
    b4: Tensor,
    bias3: Tensor,
    scale3: Tensor,
    post_bias3: Tensor,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """Blocked matmul with ``(acc + bias) * scale + post_bias`` epilogue."""
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"
    # Static under dynamic shapes: the accumulator's block dims and K.
    block_m = hl.specialize(block_m)
    block_n = hl.specialize(block_n)
    hl.specialize(blocks_k)
    hl.specialize(block_k)
    hl.specialize(bias3.size(1))
    hl.specialize(scale3.size(1))
    hl.specialize(post_bias3.size(1))

    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        y = epilogue(
            (acc + bias3[tile_blocks_n, :, :]) * scale3[tile_blocks_n, :, :]
            + post_bias3[tile_blocks_n, :, :]
        )
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


def pack_a_blocked(
    a: Tensor, m_pad: int | None = None, k_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M/BM, K/BK, BM, BK]`` with optional padding."""
    m, k = int(a.shape[0]), int(a.shape[1])
    m_target = _round_up(m, BLOCK_M) if m_pad is None else m_pad
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    return _pack_a_kernel(a, hl.constexpr(m_target), hl.constexpr(k_target))


def pack_a_blocked_t(
    a_t: Tensor, m_pad: int | None = None, k_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[K, M]`` into ``[M/BM, K/BK, BM, BK]`` with optional padding."""
    k, m = int(a_t.shape[0]), int(a_t.shape[1])
    m_target = _round_up(m, BLOCK_M) if m_pad is None else m_pad
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    return _pack_a_kernel_t(a_t, hl.constexpr(m_target), hl.constexpr(k_target))


def pack_b_blocked(
    b: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N/BN, K/BK, BK, BN]`` with optional padding."""
    k, n = int(b.shape[0]), int(b.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_kernel(b, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_t(
    b_t: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK, BN]`` with optional padding."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_kernel_t(b_t, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_vnni(
    b: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[K, N]`` into VNNI blocks ``[N/BN, K/BK, BK/2, BN, 2]``.

    Block ``(j, i)`` is the ``[16, 64]`` bf16 AMX tile of
    ``B[32i:32i+32, 32j:32j+32]``: ``out[j, i, r, n, p] = B[32i + 2r + p, 32j + n]``.
    """
    k, n = int(b.shape[0]), int(b.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_vnni_kernel(b, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_vnni_t(
    b_t: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[N, K]`` (e.g. ``nn.Linear`` weights) into the
    layout of :func:`pack_b_blocked_vnni` of ``b_t.T``."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_vnni_kernel_t(b_t, hl.constexpr(k_target), hl.constexpr(n_target))


def _pack_b_vnni_panels_kernel_t(b3_t: Tensor, panel: hl.constexpr) -> Tensor:
    """``[N, K/2, 2]`` K pairs of transposed B (N a multiple of ``panel``) into
    VNNI panels ``[N/panel, K/2, panel, 2]``."""
    n, pairs, vnni = b3_t.shape
    panel = int(panel)
    out = torch.empty(
        (n // panel, pairs, panel, vnni), dtype=b3_t.dtype, device=b3_t.device
    )
    for tp, tn in hl.tile([pairs, n]):
        out[tn.id, tp, :, :] = b3_t[tn, tp, :].permute(1, 0, 2)
    return out


_PANEL_PACK_KERNELS: dict[int, helion.Kernel] = {}


def vnni_panel(m: int, n: int) -> int:
    """Columns of :func:`pack_b_vnni_t` panels for ``m`` rows: the widest (up to
    256) that still gives every thread a tile of at least ``_MIN_SHARED_ROWS``
    rows, or one of all rows. Wider panels reuse each A row over more columns
    (1024x8192x8192: 64 columns 3118 us, 256: 2458)."""
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    if m < _MIN_SHARED_ROWS:
        # One row tile does not give every thread a tile, so the threads split
        # the rows too. Each tile reads all K of its rows of A and of its panel,
        # so its operand bytes grow with rows + columns; rows count twice, as A
        # is read by every tile of its rows and is usually the previous layer's
        # output, still in other cores' caches. 128x4096x4096 (MLP layers):
        # 64-row x 128-column tiles 133 us per layer end to end, 128 x 64 (the
        # rule below) 145 us, 32 x 256 156 us.
        def cost(panel: int) -> int:
            return 2 * _prepacked_tile_m(m, -(-n // panel), panel) + panel

        return min((_VNNI_PANEL, 128, 256), key=cost)
    row_tiles = max(1, m // _MIN_SHARED_ROWS)
    for panel in (256, 128):
        if -(-n // panel) * row_tiles >= threads:
            return panel
    return _VNNI_PANEL


def pack_b_vnni_t(b_t: Tensor, panel: int = _VNNI_PANEL) -> Tensor:
    """Pack transposed-layout ``[N, K]`` (e.g. ``nn.Linear`` weights, K even) into
    contiguous column panels of AMX's bf16 VNNI layout of ``b_t.T``,
    ``[N_pad/P, K_pad/2, P, 2]`` with ``P = panel`` (a multiple of 32), for
    :func:`matmul_prepacked_b`: each output tile streams one panel. N and K are
    zero-padded to whole panels first: a pack tile ragged in both would compile
    to per-element code."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    if k % 2 or panel % BLOCK_N:
        raise ValueError(
            f"pack_b_vnni_t() needs an even K and a panel of a multiple of "
            f"{BLOCK_N} columns, got b_t.shape={(n, k)}, panel={panel}"
        )
    n_pad, k_pad = _round_up(n, panel), _round_up(k, 2 * _PANEL_PAIRS)
    b_t = b_t.contiguous()
    if (n_pad, k_pad) != (n, k):
        b_t = pad_2d(b_t, n_pad, k_pad)
    if panel not in _PANEL_PACK_KERNELS:
        _PANEL_PACK_KERNELS[panel] = helion.kernel(
            _pack_b_vnni_panels_kernel_t,
            backend="mlir",
            config=helion.Config(block_sizes=[_PANEL_PAIRS, panel]),
        )
    return _PANEL_PACK_KERNELS[panel](
        b_t.view(n_pad, k_pad // 2, 2), hl.constexpr(panel)
    )


def supports(
    a: Tensor, b: Tensor, trans_a: bool = False, trans_b: bool = False
) -> bool:
    """Whether :func:`matmul` can handle this pair of operands.

    Any 2-D cpu tensor pair of matching dtype/contracted-dim is supported --
    shapes that aren't multiples of the AMX block size are zero-padded up
    to the next multiple internally (lighthouse's own non-Helion matmul
    pipeline handles arbitrary shapes the same way; this mirrors that rather
    than rejecting them).
    """
    if a.dtype != b.dtype or a.dtype not in _SUPPORTED_DTYPES:
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2:
        return False
    k_a = a.shape[0] if trans_a else a.shape[1]
    k_b = b.shape[1] if trans_b else b.shape[0]
    return k_a == k_b


def _round_up(value: int, block: int) -> int:
    return (value + block - 1) // block * block


def _pad_to(t: Tensor, shape: tuple[int, int]) -> Tensor:
    """Zero-pad a 2-D tensor up to ``shape`` (a no-op if already that shape)."""
    if tuple(t.shape) == shape:
        return t
    padded = torch.zeros(shape, dtype=t.dtype, device=t.device)
    padded[: t.shape[0], : t.shape[1]] = t
    return padded


def _split_k_partials(a3: Tensor, b3: Tensor) -> Tensor:
    """f32 partial products ``[S, M, N]`` of ``a3`` ``[M, S, C]`` and ``b3``
    ``[S, C, N]``: row-major ``A`` ``[M, K]`` and ``B`` ``[K, N]`` viewed with
    ``K = S * C``, read in place (no packing)."""
    m, splits, chunk = a3.shape
    n = b3.size(2)
    chunk = hl.specialize(chunk)
    hl.specialize(b3.size(1))
    out = torch.empty((splits, m, n), dtype=torch.float32, device=a3.device)
    for tile_s, tile_m, tile_n in hl.tile([splits, m, n]):
        acc = hl.zeros([tile_s, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(chunk):
            acc = acc + torch.einsum(
                "msc,scn->smn", a3[tile_m, tile_s, tile_k], b3[tile_s, tile_k, tile_n]
            )
        out[tile_s, tile_m, tile_n] = acc
    return out


def _sum_partials(partials: Tensor, dtype: torch.dtype) -> Tensor:
    """``partials`` ``[S, M, N]`` summed over S, as ``dtype``."""
    splits, m, n = partials.shape
    hl.specialize(splits)
    out = torch.empty((m, n), dtype=dtype, device=partials.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = partials[:, tile_m, tile_n].sum(0).to(dtype)
    return out


# Split-K tiles (rows, columns, K step): 64x128 f32 accumulators reuse each
# loaded A and B element 128 and 64 times (32x32 tiles: 32), so a K split of a
# few output tiles beats more, smaller tiles (256x256x8192 on 64 threads:
# 0.20 ms with S=8, 64x128; one K loop of 64 32x32 tiles: 0.26 ms).
_SPLIT_K_TILE_M = 64
_SPLIT_K_TILE_N = 128
_SPLIT_K_STEP = 256
# K elements per split at least: shorter chunks make the partial sums (S x M x
# N f32, written and read again) cost more than the parallelism gains.
_SPLIT_K_MIN_CHUNK = 512
_SPLIT_K_PARTIALS = helion.kernel(
    _split_k_partials,
    backend="mlir",
    static_shapes=True,
    config=helion.Config(
        block_sizes=[1, _SPLIT_K_TILE_M, _SPLIT_K_TILE_N, _SPLIT_K_STEP]
    ),
)
_SPLIT_K_SUM = helion.kernel(
    _sum_partials,
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[32, 256]),
)


def _omp_threads() -> int:
    """The thread count OpenMP uses at the outermost level: the first entry of
    ``OMP_NUM_THREADS`` (a list such as ``64,8`` sets nested levels), else
    the CPU count."""
    value = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    return int(value) if value.isdigit() and int(value) > 0 else os.cpu_count() or 1


def _split_k(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    bias: Tensor | None,
    scale: Tensor | None,
    shift: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
) -> int:
    """K splits of an f32 GEMM with fewer 64x128 output tiles than threads,
    whole tiles and a K of whole chunks (0: run it unsplit). The splits fill
    the threads: e.g. 256x256x8192 on 64 threads, 8 tiles x 8 splits."""
    if trans_a or trans_b or bias is not None or scale is not None or shift is not None:
        return 0
    if epilogue is not identity_epilogue:
        return 0
    if m % _SPLIT_K_TILE_M or n % _SPLIT_K_TILE_N:
        return 0
    threads = _omp_threads()
    tiles = (m // _SPLIT_K_TILE_M) * (n // _SPLIT_K_TILE_N)
    splits = 1
    while (
        tiles * splits * 2 <= threads
        and k % (splits * 2 * _SPLIT_K_STEP) == 0
        and k // (splits * 2) >= _SPLIT_K_MIN_CHUNK
    ):
        splits *= 2
    return splits if splits > 1 else 0


def _matmul_split_k(a: Tensor, b: Tensor, splits: int) -> Tensor:
    """``a @ b`` (f32 row-major) as ``splits`` K chunks read in place, then a
    sum of the partial products: both are Helion kernels, so no other
    runtime's threads run between them."""
    m, k = map(int, a.shape)
    n = int(b.shape[1])
    a, b = a.contiguous(), b.contiguous()
    partials = _SPLIT_K_PARTIALS(
        a.view(m, splits, k // splits), b.view(splits, k // splits, n)
    )
    return _SPLIT_K_SUM(partials, a.dtype)


def _f32_chunk_dot(
    a: Tensor,
    b: Tensor,
    tile_m: object,
    tile_n: object,
    tile_k: object,
    trans_a: bool,
    trans_b: bool,
) -> Tensor:
    """``a @ b`` of f32 ``[M, K]`` and ``[K, N]`` (``a`` ``[K, M]`` with
    ``trans_a``, ``b`` ``[N, K]`` with ``trans_b``) on one tile and K chunk,
    each operand packed into 32x32 blocks: ``[TM/32, TK/32, 32, 32]`` and
    ``[TN/32, TK/32, 32, 32]``, the blocked layout of :func:`pack_a_blocked`
    and :func:`pack_b_blocked`. The packs land in private buffers."""
    bm, bn, bk = (
        tile_m.block_size // 32,
        tile_n.block_size // 32,
        tile_k.block_size // 32,
    )
    if trans_a:
        a4 = a[tile_k, tile_m].reshape(bk, 32, bm, 32).permute(2, 0, 3, 1)
    else:
        a4 = a[tile_m, tile_k].reshape(bm, 32, bk, 32).permute(0, 2, 1, 3)
    if trans_b:
        b4 = b[tile_n, tile_k].reshape(bn, 32, bk, 32).permute(0, 2, 3, 1)
    else:
        b4 = b[tile_k, tile_n].reshape(bk, 32, bn, 32).permute(2, 0, 1, 3)
    return torch.einsum("akmc,bkcn->abmn", a4, b4)


def _matmul_f32_fused_pack(
    a: Tensor,
    b: Tensor,
    epilogue: Callable[[Tensor], Tensor],
    trans_a: hl.constexpr,
    trans_b: hl.constexpr,
) -> Tensor:
    """``epilogue(a @ b)`` of f32 row-major ``[M, K]`` and ``[K, N]`` (``[K, M]``
    with ``trans_a``, ``[N, K]`` with ``trans_b``), all extents whole tiles and
    K chunks (:func:`_f32_fused_tiles`).

    Each tile packs the A and B chunks it needs per K step into 32x32 blocks in
    private buffers (:func:`_f32_chunk_dot`) and accumulates in blocks
    ``[TM/32, TN/32, 32, 32]``, whose last two dims get the 8x32 FMA register
    tile; the accumulator is stored straight into row-major C. Separate pack
    kernels cost more: their output blocks are read by many cores, so
    rewriting those lines on the next call invalidates copies in other cores'
    caches (4096^3 on EMR: 2.2 ms per 64 MiB operand after a GEMM, 0.1 ms
    without one).
    """
    if trans_a:
        k, m = a.shape
    else:
        m, k = a.shape
    n = b.shape[0] if trans_b else b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros(
            [tile_m.block_size // 32, tile_n.block_size // 32, 32, 32],
            dtype=torch.float32,
        )
        for tile_k in hl.tile(k):
            acc = acc + _f32_chunk_dot(a, b, tile_m, tile_n, tile_k, trans_a, trans_b)
        y = epilogue(acc).permute(0, 2, 1, 3)
        out[tile_m, tile_n] = y.reshape(tile_m.block_size, tile_n.block_size).to(
            a.dtype
        )
    return out


# Tiles of :func:`_matmul_f32_fused_pack` (EMR, 64 threads, 4096^3 / 2048x4096x8192,
# kernel ms): 256x256 tiles reuse each packed A and B element 256 times per
# 8x32 register tile pass (128x128: 19.8 / -, 256x256: 18.2 / 18.0, 512x256:
# 18.7); a 128-deep K chunk keeps a block column of the packed B chunk
# (128 x 32 x 4 B = 16 KiB) in L1 across the tile's row blocks (64: 19.2,
# 256: 19.5, 512: 25.7 ms at 4096^3).
_F32_TILE = 256
_F32_MIN_TILE = 64
_F32_K_CHUNK = 128
_F32_FUSED_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}


def _f32_fused_tiles(m: int, n: int, k: int) -> tuple[int, int, int] | None:
    """Tiles (rows, columns, K chunk) of :func:`_matmul_f32_fused_pack`, or
    ``None`` unless the extents are whole tiles and chunks (the kernel has no
    partial tiles; other shapes take the blocked path). Starts from
    ``_F32_TILE`` squares and halves the longer side, down to
    ``_F32_MIN_TILE``, while there are fewer tiles than threads."""
    tile_k = next((c for c in (_F32_K_CHUNK, 64, 32) if k % c == 0), None)
    if tile_k is None:
        return None

    def side(extent: int) -> int | None:
        return next(
            (t for t in (_F32_TILE, 128, _F32_MIN_TILE) if extent % t == 0), None
        )

    tile_m, tile_n = side(m), side(n)
    if tile_m is None or tile_n is None:
        return None
    threads = _omp_threads()
    while (m // tile_m) * (n // tile_n) < threads:
        if tile_m >= tile_n and tile_m > _F32_MIN_TILE:
            tile_m //= 2
        elif tile_n > _F32_MIN_TILE:
            tile_n //= 2
        else:
            break
    return tile_m, tile_n, tile_k


def _matmul_f32_fused(
    a: Tensor,
    b: Tensor,
    trans_a: bool,
    trans_b: bool,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor | None:
    """:func:`_matmul_f32_fused_pack` of ``a`` and ``b``, or ``None`` if their
    shape does not fit it. Bound kernels are cached like the bf16 ones."""
    m = int(a.shape[1]) if trans_a else int(a.shape[0])
    n = int(b.shape[0]) if trans_b else int(b.shape[1])
    k = int(a.shape[0]) if trans_a else int(a.shape[1])
    key = ("f32", m, n, k, trans_a, trans_b, _omp_threads(), epilogue)
    if (cached := _FUSED_PACK_BOUND.get(key)) is None:
        tiles = _f32_fused_tiles(m, n, k)
        if tiles is None:
            return None
        if tiles not in _F32_FUSED_KERNELS:
            _F32_FUSED_KERNELS[tiles] = helion.kernel(
                _matmul_f32_fused_pack,
                backend="mlir",
                static_shapes=True,
                config=helion.Config(block_sizes=list(tiles)),
            )
        a, b = a.contiguous(), b.contiguous()
        consts = (hl.constexpr(trans_a), hl.constexpr(trans_b))
        bound = _F32_FUSED_KERNELS[tiles].bind((a, b, epilogue, *consts))
        cached = _FUSED_PACK_BOUND[key] = (bound, consts)
        if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
            _FUSED_PACK_BOUND.popitem(last=False)
    else:
        _FUSED_PACK_BOUND.move_to_end(key)
    bound, consts = cached
    return bound(a.contiguous(), b.contiguous(), epilogue, *consts)


def matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    block_sizes: Sequence[int] | None = None,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """``epilogue((op(A) @ op(B) + bias) * scale + shift)``, ``op`` transposing
    with ``trans_a``/``trans_b``.

    No pre-packing. bf16 runs the fused kernel (:func:`_matmul_fused_pack`): B
    is packed per output tile, A read in place (with ``trans_a``, packed per
    tile too), partial edge tiles handle any shape, and operands smaller than
    one tile are zero-padded to one; ``block_sizes`` (tile rows, columns, K
    chunk) overrides its tile choice. f32 without bias, scale or shift runs
    :func:`_matmul_f32_fused_pack` when M, N and K are whole tiles and K
    chunks (:func:`_f32_fused_tiles`; ``block_sizes`` is ignored), else both
    operands are packed into 32x32 blocks by separate kernels, zero-padded to
    block multiples (:func:`_matmul_blocked_kernel`). ``bias``, ``scale`` and
    ``shift`` are ``[N]`` (``scale``/``shift`` also ``[1]``), broadcast over
    rows. ``epilogue`` is fused into the same kernel as the contraction (see
    module docstring), not a separate pass.

    Raises ``ValueError`` if the operands aren't otherwise compatible (mismatched
    dtype/device/rank/contracted-dim -- see :func:`supports`) instead of
    silently falling back to eager PyTorch -- a silent fallback would benchmark
    eager while looking like it benchmarked the Helion kernel. Callers that
    need a fallback must check :func:`supports` themselves and choose one
    explicitly.
    """
    if not supports(a, b, trans_a=trans_a, trans_b=trans_b):
        raise ValueError(
            f"matmul() does not support these operands (trans_a={trans_a}, "
            f"trans_b={trans_b}): a.shape={tuple(a.shape)} a.dtype={a.dtype} "
            f"a.device={a.device}, b.shape={tuple(b.shape)} b.dtype={b.dtype} "
            f"b.device={b.device}. dtype must be one of {_SUPPORTED_DTYPES}, "
            "both tensors must be 2-D on cpu with a matching contracted dim. "
            "Check supports() before calling, or use torch.matmul directly "
            "for unsupported cases."
        )

    m = int(a.shape[1]) if trans_a else int(a.shape[0])
    n = int(b.shape[0]) if trans_b else int(b.shape[1])
    k = int(a.shape[0]) if trans_a else int(a.shape[1])

    m_pad = _round_up(m, BLOCK_M)
    n_pad = _round_up(n, BLOCK_N)
    k_pad = _round_up(k, BLOCK_K)

    if a.dtype == torch.bfloat16:
        scale = _affine_vector(scale, n, n, a)
        shift = _affine_vector(shift, n, n, a)
        if min(m, n) >= BLOCK_M and k >= 2 * BLOCK_K:
            bound, consts = _matmul_fused_pack_bound(
                a,
                b,
                bias,
                epilogue,
                trans_a,
                trans_b,
                tiles=block_sizes,
                scale=scale,
                shift=shift,
            )
            return bound(a, b, bias, scale, shift, epilogue, *consts)
        # Smaller than one tile: padded to one (K to pairs of AMX steps of 32).
        k_fused = _round_up(k, 2 * BLOCK_K)
        a_p = pad_2d(a, k_fused, m_pad) if trans_a else pad_2d(a, m_pad, k_fused)
        b_p = pad_2d(b, n_pad, k_fused) if trans_b else pad_2d(b, k_fused, n_pad)
        bias_p, scale_p, shift_p = (
            None if v is None else pad_2d(v.reshape(1, n), 1, n_pad)[0]
            for v in (bias, scale, shift)
        )
        bound, consts = _matmul_fused_pack_bound(
            a_p,
            b_p,
            bias_p,
            epilogue,
            trans_a,
            trans_b,
            tiles=block_sizes,
            scale=scale_p,
            shift=shift_p,
        )
        return bound(a_p, b_p, bias_p, scale_p, shift_p, epilogue, *consts)[:m, :n]

    splits = _split_k(m, n, k, trans_a, trans_b, bias, scale, shift, epilogue)
    if splits:
        return _matmul_split_k(a, b, splits)
    if bias is None and scale is None and shift is None:
        fused = _matmul_f32_fused(a, b, trans_a, trans_b, epilogue)
        if fused is not None:
            return fused
    a4 = (
        pack_a_blocked_t(a, m_pad=m_pad, k_pad=k_pad)
        if trans_a
        else pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    )
    b4 = (
        pack_b_blocked_t(b, k_pad=k_pad, n_pad=n_pad)
        if trans_b
        else pack_b_blocked(b, k_pad=k_pad, n_pad=n_pad)
    )
    if scale is not None or shift is not None:
        blocks = (n_pad // BLOCK_N, 1, BLOCK_N)
        out4 = _matmul_blocked_kernel_affine(
            a4,
            b4,
            _affine_vector(bias, n, n_pad, a, 0.0).reshape(blocks),
            _affine_vector(scale, n, n_pad, a, 1.0).reshape(blocks),
            _affine_vector(shift, n, n_pad, a, 0.0).reshape(blocks),
            epilogue,
        )
    elif bias is None:
        out4 = _matmul_blocked_kernel(a4, b4, epilogue)
    else:
        bias_padded = _pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(-1)
        bias3 = bias_padded.reshape(n_pad // BLOCK_N, 1, BLOCK_N)
        out4 = _matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    out = out4.reshape(m_pad, n_pad)
    return out[:m, :n]


def _affine_vector(
    vector: Tensor | None, n: int, n_pad: int, like: Tensor, fill: float | None = None
) -> Tensor | None:
    """``vector`` (``[1]`` or ``[n]``) as a contiguous ``[n_pad]`` in ``like``'s
    dtype; ``None`` stays ``None`` unless ``fill`` is given."""
    if vector is None:
        if fill is None:
            return None
        return torch.full((n_pad,), fill, dtype=like.dtype, device=like.device)
    if vector.numel() not in (1, n) or vector.dim() > 1:
        raise ValueError(
            f"affine vector must be scalar or have shape (1,) or ({n},), "
            f"got {tuple(vector.shape)}"
        )
    vector = vector.to(dtype=like.dtype, device=like.device).reshape(-1)
    if vector.numel() == 1:
        return vector.expand(n_pad).contiguous()
    return _pad_to(vector.reshape(1, -1), (1, n_pad)).reshape(-1)


def matmul_prepacked_b(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """Multiply row-major ``a`` by a RHS produced by ``pack_b_blocked_t`` or, for
    bf16, :func:`pack_b_vnni_t`; ``scale``/``shift`` as in :func:`matmul`.

    Only the runtime activation is packed on each call (not even that for a
    VNNI-packed RHS). ``n`` is the original output width before padding;
    callers own the lifetime and invalidation of ``b4`` and must repack it when
    the source weight changes.
    """
    if a.dim() == 2 and b4.dim() == 4 and b4.shape[-1] == 2:
        return _matmul_prepacked_vnni_call(
            a,
            b4,
            n,
            bias,
            epilogue,
            _affine_vector(scale, n, n, a),
            _affine_vector(shift, n, n, a),
        )
    if scale is not None or shift is not None:
        return matmul_prepacked_b_affine(
            a, b4, n, bias, _affine_vector(scale, n, n, a, 1.0), shift, epilogue
        )
    if a.dim() != 2 or b4.dim() != 4:
        raise ValueError(
            f"matmul_prepacked_b() expects rank-2 a and rank-4 b4, got "
            f"a.shape={tuple(a.shape)} and b4.shape={tuple(b4.shape)}"
        )
    if a.dtype != b4.dtype or a.device != b4.device:
        raise ValueError(
            "matmul_prepacked_b() requires matching dtype and device: "
            f"a=({a.dtype}, {a.device}), b4=({b4.dtype}, {b4.device})"
        )

    blocks_n, blocks_k, block_k, block_n = map(int, b4.shape)
    if block_k != BLOCK_K or block_n != BLOCK_N:
        raise ValueError(
            f"invalid packed RHS block shape {tuple(b4.shape)}; "
            f"expected trailing dimensions ({BLOCK_K}, {BLOCK_N})"
        )

    m, k = map(int, a.shape)
    m_pad = _round_up(m, BLOCK_M)
    k_pad = _round_up(k, BLOCK_K)
    n_pad = blocks_n * BLOCK_N
    if (
        blocks_k * BLOCK_K != k_pad
        or not 0 < n <= n_pad
        or _round_up(n, BLOCK_N) != n_pad
    ):
        raise ValueError(
            f"packed RHS shape {tuple(b4.shape)} is incompatible with "
            f"a.shape={tuple(a.shape)} and n={n}"
        )

    if a.dtype == torch.float32 and k == k_pad:
        if bias is not None and (bias.dim() != 1 or int(bias.shape[0]) != n):
            raise ValueError(f"bias must have shape ({n},), got {tuple(bias.shape)}")
        # A read in place: packing it per call costs a kernel (128x4096 f32:
        # 8 us alone, 125 us right after the GEMM that wrote its buffer).
        return _matmul_prepacked_rows_call(a, b4, n, bias, epilogue)
    a4 = pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    if bias is None:
        out4 = _matmul_blocked_kernel(a4, b4, epilogue)
    else:
        if bias.dim() != 1 or int(bias.shape[0]) != n:
            raise ValueError(f"bias must have shape ({n},), got {tuple(bias.shape)}")
        bias_padded = _pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(-1)
        bias3 = bias_padded.reshape(blocks_n, 1, BLOCK_N)
        out4 = _matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    return out4.reshape(m_pad, n_pad)[:m, :n]


def prepacked_b_call(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Callable[[Tensor], Tensor]:
    """``lambda x: matmul_prepacked_b(x, b4, n, bias, epilogue, scale, shift)``
    with the host work of a call (operand views, tile choice, kernel lookup)
    done once, for activations like ``a``: the same shape and dtype,
    contiguous; others, and cases with partial row tiles, take
    :func:`matmul_prepacked_b`. ``bias``, ``scale`` and ``shift`` are
    constants of the prepared call: the caller rebuilds it when they change
    (:func:`~.linear.linear_affine` keys them by data pointer and
    ``_version``, so ``load_state_dict`` and in-place ops are seen; an
    in-place change through ``.data``, which leaves ``_version`` unchanged,
    is not seen in every layout). Valid while ``OMP_NUM_THREADS`` stays the
    same."""

    def general(x: Tensor) -> Tensor:
        return matmul_prepacked_b(x, b4, n, bias, epilogue, scale, shift)

    # Operands matmul_prepacked_b rejects (or handles specially) take it.
    if (
        a.dim() != 2
        or b4.dim() != 4
        or a.dtype != b4.dtype
        or a.device != b4.device
        or (
            bias is not None
            and (bias.dim() != 1 or int(bias.shape[0]) != n or bias.device != a.device)
        )
    ):
        return general
    m, k = map(int, a.shape)
    shape, dtype = a.shape, a.dtype
    a = a.contiguous()
    if b4.shape[-1] == 2:
        panels, pairs_b, panel, _ = map(int, b4.shape)
        n_pad = panels * panel
        tile_m = _prepacked_tile_m(m, panels, panel)
        if (
            dtype != torch.bfloat16
            or k % 2
            or panel % BLOCK_N
            or m % tile_m
            or 2 * pairs_b != _round_up(k, 2 * _PANEL_PAIRS)
            or _round_up(n, panel) != n_pad
        ):
            return general

        def panels2(vector: Tensor | None) -> Tensor | None:
            # As _matmul_prepacked_vnni_call lays them out.
            if vector is None:
                return None
            row = vector.reshape(1, n)
            return (row if n == n_pad else pad_2d(row, 1, n_pad)).view(panels, panel)

        bias2 = panels2(bias)
        scale2 = panels2(_affine_vector(scale, n, n, a))
        shift2 = panels2(_affine_vector(shift, n, n, a))
        pairs = (m, k // 2, 2)
        bound, consts = _prepacked_bound(
            a.view(pairs), b4, bias2, scale2, shift2, epilogue, tile_m
        )
        bound_args = (a.view(pairs), b4, bias2, scale2, shift2, epilogue, *consts)

        def run(x: Tensor) -> Tensor:
            return bound(x.view(pairs), b4, bias2, scale2, shift2, epilogue, *consts)

    elif scale is not None or shift is not None:
        # The affine blocked path keeps its per-call staging.
        return general
    else:
        blocks_n, blocks_k, block_k, block_n = map(int, b4.shape)
        n_pad = blocks_n * BLOCK_N
        if (
            dtype != torch.float32
            or (block_k, block_n) != (BLOCK_K, BLOCK_N)
            or blocks_k * BLOCK_K != k
            or m % BLOCK_M
            or _round_up(n, BLOCK_N) != n_pad
        ):
            return general
        b3, bias2 = _rows_operands(b4, n, bias)
        bound = _rows_bound(a, b3, bias2, epilogue)
        bound_args = (a, b3, bias2, epilogue)

        def run(x: Tensor) -> Tensor:
            return bound(x, b3, bias2, epilogue)

    def call(x: Tensor) -> Tensor:
        if x.shape != shape or x.dtype is not dtype or not x.is_contiguous():
            return general(x)
        out = run(x).view(m, n_pad)
        return out if n == n_pad else out[:, :n]

    if n != n_pad or _prepared_call is None:
        return call
    # The same call natively: x (contiguous, this shape and dtype) as the
    # kernel's operand view, the result as [M, N]; anything else takes call.
    prepared = _prepared_call(
        bound, bound_args, [0], call, views=[tuple(shape)], result_shape=(m, n_pad)
    )
    return call if prepared is None else prepared


def matmul_prepacked_b_affine(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None,
    post_scale: Tensor,
    post_bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
) -> Tensor:
    """Prepacked RHS matmul with a per-output affine post-op."""
    blocks_n, blocks_k, block_k, block_n = map(int, b4.shape)
    m, k = map(int, a.shape)
    m_pad = _round_up(m, BLOCK_M)
    k_pad = _round_up(k, BLOCK_K)
    n_pad = blocks_n * BLOCK_N
    if (
        a.dim() != 2
        or b4.dim() != 4
        or a.dtype != b4.dtype
        or a.device != b4.device
        or block_k != BLOCK_K
        or block_n != BLOCK_N
        or blocks_k * BLOCK_K != k_pad
        or not 0 < n <= n_pad
        or _round_up(n, BLOCK_N) != n_pad
    ):
        raise ValueError(
            f"packed RHS shape {tuple(b4.shape)} is incompatible with "
            f"a.shape={tuple(a.shape)} and n={n}"
        )

    def prepare(vector: Tensor | None, fill: float) -> Tensor:
        if vector is None:
            return torch.full((n_pad,), fill, dtype=a.dtype, device=a.device)
        if vector.dim() > 1 or (
            vector.dim() == 1 and int(vector.shape[0]) not in (1, n)
        ):
            raise ValueError(
                f"affine vector must be scalar or have shape (1,) or ({n},), "
                f"got {tuple(vector.shape)}"
            )
        vector = vector.to(dtype=a.dtype, device=a.device)
        if vector.numel() == 1:
            return vector.expand(n_pad).contiguous()
        return _pad_to(vector.reshape(1, -1), (1, n_pad)).reshape(-1)

    a4 = pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    bias3 = prepare(bias, 0.0).reshape(blocks_n, 1, BLOCK_N)
    scale3 = prepare(post_scale, 1.0).reshape(blocks_n, 1, BLOCK_N)
    post_bias3 = prepare(post_bias, 0.0).reshape(blocks_n, 1, BLOCK_N)
    out4 = _matmul_blocked_kernel_affine(a4, b4, bias3, scale3, post_bias3, epilogue)
    return out4.reshape(m_pad, n_pad)[:m, :n]


def _bmm_direct(a: Tensor, b: Tensor) -> Tensor:
    """Batched f32 ``a @ b``, the batch a grid dim: one kernel, A and B read in
    place (no packing), output tiles of all batches shared by the threads."""
    batch, m, k = a.shape
    n = b.size(2)
    k = hl.specialize(k)
    hl.specialize(b.size(1))
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.bmm(a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc.to(a.dtype)
    return out


# Tiles of _bmm_direct (batch, rows, columns, K step): 128x256 f32 tiles with
# 128-deep K steps (a 128 KiB B chunk, in L2) were the fastest of 8 configs at
# 3x2048^3 on 64 threads (7.8 ms; 128x128: 8.1, 64x128x256: 13.0).
_BMM_TILES = (1, 128, 256, 128)
_BMM_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}


def _divisor_tile(extent: int, largest: int) -> int | None:
    """The largest of ``largest``, its halves down to 32, that divides ``extent``."""
    tile = largest
    while tile >= BLOCK_M:
        if extent % tile == 0:
            return tile
        tile //= 2
    return None


def _bmm_f32(a: Tensor, b: Tensor) -> Tensor | None:
    """:func:`_bmm_direct` on tiles dividing the problem (``None`` if none do):
    partial tiles along up to four dims are versioned per dim, and the kernel
    takes minutes to compile (2x65x129x129: 25 min)."""
    _, m, k = map(int, a.shape)
    n = int(b.shape[2])
    _, tile_m, tile_n, tile_k = _BMM_TILES
    tiles = (
        1,
        _divisor_tile(m, tile_m),
        _divisor_tile(n, tile_n),
        _divisor_tile(k, tile_k),
    )
    if None in tiles:
        return None
    if tiles not in _BMM_KERNELS:
        _BMM_KERNELS[tiles] = helion.kernel(
            _bmm_direct,
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=list(tiles)),
        )
    return _BMM_KERNELS[tiles](a.contiguous(), b.contiguous())


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """Batched ``A @ B``.

    bf16: each batch slice is contracted by its own fully parallel kernel call,
    written in place into the batched result (a single kernel with the batch
    threaded through the pack + contract kernels was ~30% slower at
    3x2048^3). f32: one kernel with the batch as a grid dim, operands read in
    place (:func:`_bmm_direct`). 3x2048^3 on 64 threads, one OpenMP runtime
    (``env.sh cpu-bench``): 8.00 ms; per-slice blocked kernels, packing both
    operands per slice, stacked by ``torch.stack``: 11.49 ms; eager 7.36 ms.
    Tiles that do not divide the problem fall back to the per-slice path.
    """
    batch, m, k = a.shape
    if b.dim() != 3 or b.shape[0] != batch or b.shape[1] != k:
        raise RuntimeError(
            "Expected size for first two dimensions of batch2 tensor to be: "
            f"[{batch}, {k}] but got: {list(b.shape[:2])}."
        )
    n = b.shape[2]
    if a.dtype == torch.bfloat16 and min(m, n) >= BLOCK_M and k >= 2 * BLOCK_K:
        a, b = a.contiguous(), b.contiguous()
        out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
        for i in range(batch):
            bound, consts = _matmul_fused_pack_bound(
                a[i], b[i], None, identity_epilogue, False, False, out[i]
            )
            bound(a[i], b[i], None, None, None, identity_epilogue, *consts, out[i])
        return out
    if a.dtype == torch.float32:
        # One kernel, no per-slice packing or stack (3x2048^3: 8.00 ms, not
        # 11.49). Also no torch op between kernels: with torch on libgomp and
        # the kernels on libomp, torch.stack's threads compete with libomp's
        # still-spinning workers (68 ms without the cpu-bench libomp preload).
        out = _bmm_f32(a, b)
        if out is not None:
            return out
    return torch.stack([matmul(a[i], b[i]) for i in range(a.shape[0])])
