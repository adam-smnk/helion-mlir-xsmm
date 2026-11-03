"""Bool to float casts of whole tiles: ``xpu_bool_casts``. ``a``, ``b``, ``c`` and ``int_gt``
are the kernels that aborted (full tiles, LLVM's SPIR-V backend)."""

from __future__ import annotations

import helion.language as hl
import torch

F32 = torch.float32


def a(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn] <= 0.0).to(F32)
    return out


def a_ref(x: torch.Tensor) -> torch.Tensor:
    return (x <= 0.0).double()


def b(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn] <= 0.0).to(F32) * x[tm, tn]
    return out


def b_ref(x: torch.Tensor) -> torch.Tensor:
    return (x <= 0.0).double() * x


def c(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.size()
    block = hl.register_block_size(cols)
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tm in hl.tile(rows):
        total = hl.zeros([tm], dtype=F32)
        for tn in hl.tile(cols, block_size=block):
            total = total + (x[tm, tn] <= 0.0).to(F32).sum(-1)
        out[tm] = total
    return out


def c_ref(x: torch.Tensor) -> torch.Tensor:
    return (x <= 0.0).double().sum(-1)


def int_gt(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn] >= 0).to(F32)
    return out


def int_gt_ref(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).double()


def to_own_type(x: torch.Tensor) -> torch.Tensor:
    """To bf16, as well as to f32."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn] <= 0.0).to(x.dtype)
    return out


def int_to_float(x: torch.Tensor) -> torch.Tensor:
    """An integer tile itself: no bool in it, and it never aborted."""
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn].to(F32) * 2.0
    return out


def int_to_float_ref(x: torch.Tensor) -> torch.Tensor:
    return x.double() * 2.0


CASES = {
    # name: kernel, reference, block sizes, input type
    "a": (a, a_ref, [64, 64], torch.float32),
    "b": (b, b_ref, [64, 64], torch.float32),
    "c": (c, c_ref, [64, 64], torch.float32),
    "int_gt": (int_gt, int_gt_ref, [64, 64], torch.int32),
    "a_bf16": (a, a_ref, [64, 64], torch.bfloat16),
    "b_bf16": (b, b_ref, [64, 64], torch.bfloat16),
    "own_bf16": (to_own_type, a_ref, [64, 64], torch.bfloat16),
    "a_tile_32": (a, a_ref, [32, 32], torch.float32),
    "a_tile_16": (a, a_ref, [16, 16], torch.float32),
    "a_tile_8": (a, a_ref, [8, 16], torch.float32),
    "int_to_float": (int_to_float, int_to_float_ref, [64, 64], torch.int32),
}
SHAPES = {"full": (128, 256), "ragged": (120, 248)}
