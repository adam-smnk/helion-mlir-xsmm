"""Kernels of one tile: one-workgroup shapes and the 64^3 GEMM."""

from __future__ import annotations

from tests import xpu_partial_kernels as pk

BF16 = pk.BF16
F32 = pk.F32


def _gemm(size: int) -> pk.Case:
    ops = ((size, size), BF16, "n"), ((size, size), BF16, "n")
    return pk.Case(pk.gemm, ops, ((size, size), BF16), [size, size, 32], pk._gemm_ref)


CASES = {
    "gemm_128": _gemm(128),
    "gemm_64": _gemm(64),
    "scale_128_256": pk.Case(
        pk.scale,
        (((128, 256), F32, "n"),),
        ((128, 256), F32),
        [128, 256],
        pk._scale_ref,
    ),
    "scale_16": pk.Case(
        pk.scale, (((16, 16), F32, "n"),), ((16, 16), F32), [16, 16], pk._scale_ref
    ),
    "row_max": pk.Case(
        pk.row_max,
        (((64, 256), F32, "neg"),),
        ((64,), F32),
        [64, 64],
        pk._max_ref,
    ),
    "softmax": pk.Case(
        pk.softmax_masked,
        (((64, 256), F32, "neg"),),
        ((64, 256), F32),
        [64, 64],
        pk._softmax_ref,
    ),
    "attention": pk.Case(
        pk.attention,
        (((1, 1, 128, 64), BF16, "n"),) * 3,
        ((1, 1, 128, 64), BF16),
        [1, 1, 128, 64],
        pk._attention_ref,
    ),
}
