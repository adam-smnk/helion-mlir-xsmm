"""Elementwise kernels of one math function each, ``out = f(x.float())`` rounded
to ``out``'s dtype, for the device's math numerics (``test_xpu_runtime.
test_math_functions``) and the XPU math check (``xpu_checks.check_math``). The
functions and arguments keep them in range: ``log`` of ``|v| + 0.5``, ``log1p``
of ``0.4 v`` for ``v`` in [-2, 2]."""

from __future__ import annotations

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

EXPRS: dict[str, str] = {
    "abs_p1": "torch.abs(v) + 1",
    "abs_x2": "torch.abs(v) * 2",
    "neg": "-v",
    "softsign": "v / (1 + torch.abs(v))",
    "absmax": "torch.maximum(torch.abs(v), v * 0.5)",
    "abs_raw": "torch.abs(x[tile]).to(torch.float32)",
    "exp": "torch.exp(v)",
    "exp2": "torch.exp2(v)",
    "log": "torch.log(torch.abs(v) + 0.5)",
    "log2": "torch.log2(torch.abs(v) + 0.5)",
    "log1p": "torch.log1p(v * 0.4)",
    "expm1": "torch.expm1(v)",
    "sqrt": "torch.sqrt(torch.abs(v) + 0.5)",
    "rsqrt": "torch.rsqrt(torch.abs(v) + 0.5)",
    "sin": "torch.sin(v)",
    "cos": "torch.cos(v)",
    "tanh": "torch.tanh(v)",
    "erf": "torch.erf(v)",
    "sigmoid": "torch.sigmoid(v)",
    "atan": "torch.atan(v)",
    "abs": "torch.abs(v)",
    "recip": "1.0 / (torch.abs(v) + 0.5)",
    "softplus": "F.softplus(v)",
    "gelu": "F.gelu(v)",
    "gelu_tanh": "F.gelu(v, approximate='tanh')",
    "mish": "F.mish(v)",
    "silu": "F.silu(v)",
    "tanh_x2": "torch.tanh(v) * 2",
    "pow": "torch.pow(torch.abs(v) + 0.5, 2.5)",
    "elu": "F.elu(v)",
    "selu": "F.selu(v)",
    "tan": "torch.tan(v * 0.5)",
    "asin": "torch.asin(v * 0.4)",
    "acos": "torch.acos(v * 0.4)",
    "atanh": "torch.atanh(v * 0.4)",
    "sinh": "torch.sinh(v)",
    "cosh": "torch.cosh(v)",
    "floor": "torch.floor(v * 3)",
    "ceil": "torch.ceil(v * 3)",
    "round": "torch.round(v * 3)",
    "powi": "torch.pow(v, 3)",
    "abs_neg": "torch.abs(-v)",
    "abs_mul1": "torch.abs(v * 1.0)",
    "atan_mul1": "torch.atan(v * 1.0)",
    "abs_shared": "torch.abs(v) * 2 + v * 0.0",
    "atan_shared": "torch.atan(v) + v * 0.0",
}
"""Each kernel's function of ``v = x[tile].to(torch.float32)`` (``x`` for the reference)."""


def _kernel(fn):
    return helion.kernel(
        backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[128, 256])
    )(fn)


@_kernel
def k_abs_p1(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(v) + 1).to(out.dtype)
    return out


@_kernel
def k_abs_x2(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(v) * 2).to(out.dtype)
    return out


@_kernel
def k_neg(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (-v).to(out.dtype)
    return out


@_kernel
def k_softsign(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v / (1 + torch.abs(v))).to(out.dtype)
    return out


@_kernel
def k_absmax(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.maximum(torch.abs(v), v * 0.5)).to(out.dtype)
    return out


@_kernel
def k_abs_raw(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``abs`` of the tile in its own dtype."""
    for tile in hl.tile(x.shape):
        out[tile] = (torch.abs(x[tile]).to(torch.float32)).to(out.dtype)
    return out


@_kernel
def k_exp(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.exp(v)).to(out.dtype)
    return out


@_kernel
def k_exp2(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.exp2(v)).to(out.dtype)
    return out


@_kernel
def k_log(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.log(torch.abs(v) + 0.5)).to(out.dtype)
    return out


@_kernel
def k_log2(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.log2(torch.abs(v) + 0.5)).to(out.dtype)
    return out


@_kernel
def k_log1p(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.log1p(v * 0.4)).to(out.dtype)
    return out


@_kernel
def k_expm1(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.expm1(v)).to(out.dtype)
    return out


@_kernel
def k_sqrt(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sqrt(torch.abs(v) + 0.5)).to(out.dtype)
    return out


@_kernel
def k_rsqrt(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.rsqrt(torch.abs(v) + 0.5)).to(out.dtype)
    return out


@_kernel
def k_sin(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sin(v)).to(out.dtype)
    return out


@_kernel
def k_cos(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.cos(v)).to(out.dtype)
    return out


@_kernel
def k_tanh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.tanh(v)).to(out.dtype)
    return out


@_kernel
def k_erf(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.erf(v)).to(out.dtype)
    return out


@_kernel
def k_sigmoid(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sigmoid(v)).to(out.dtype)
    return out


@_kernel
def k_atan(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.atan(v)).to(out.dtype)
    return out


@_kernel
def k_abs(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(v)).to(out.dtype)
    return out


@_kernel
def k_recip(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (1.0 / (torch.abs(v) + 0.5)).to(out.dtype)
    return out


@_kernel
def k_softplus(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.softplus(v)).to(out.dtype)
    return out


@_kernel
def k_gelu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.gelu(v)).to(out.dtype)
    return out


@_kernel
def k_gelu_tanh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.gelu(v, approximate="tanh")).to(out.dtype)
    return out


@_kernel
def k_mish(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.mish(v)).to(out.dtype)
    return out


@_kernel
def k_silu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.silu(v)).to(out.dtype)
    return out


@_kernel
def k_tanh_x2(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.tanh(v) * 2).to(out.dtype)
    return out


@_kernel
def k_pow(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.pow(torch.abs(v) + 0.5, 2.5)).to(out.dtype)
    return out


@_kernel
def k_elu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.elu(v)).to(out.dtype)
    return out


@_kernel
def k_selu(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.selu(v)).to(out.dtype)
    return out


@_kernel
def k_tan(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.tan(v * 0.5)).to(out.dtype)
    return out


@_kernel
def k_asin(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.asin(v * 0.4)).to(out.dtype)
    return out


@_kernel
def k_acos(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.acos(v * 0.4)).to(out.dtype)
    return out


@_kernel
def k_atanh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.atanh(v * 0.4)).to(out.dtype)
    return out


@_kernel
def k_sinh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sinh(v)).to(out.dtype)
    return out


@_kernel
def k_cosh(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.cosh(v)).to(out.dtype)
    return out


@_kernel
def k_floor(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.floor(v * 3)).to(out.dtype)
    return out


@_kernel
def k_ceil(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.ceil(v * 3)).to(out.dtype)
    return out


@_kernel
def k_round(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.round(v * 3)).to(out.dtype)
    return out


@_kernel
def k_powi(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.pow(v, 3)).to(out.dtype)
    return out


@_kernel
def k_abs_neg(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(-v)).to(out.dtype)
    return out


@_kernel
def k_abs_mul1(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(v * 1.0)).to(out.dtype)
    return out


@_kernel
def k_atan_mul1(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.atan(v * 1.0)).to(out.dtype)
    return out


@_kernel
def k_abs_shared(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.abs(v) * 2 + v * 0.0).to(out.dtype)
    return out


@_kernel
def k_atan_shared(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.atan(v) + v * 0.0).to(out.dtype)
    return out


# GEMM epilogues and row reductions with tanh (the check's reviewer cases).
_GEMM = helion.Config(block_sizes=[256, 256, 32])


def _config_kernel(config: helion.Config):
    return lambda fn: helion.kernel(backend="mlir", static_shapes=True, config=config)(
        fn
    )


@_config_kernel(_GEMM)
def mm_tanh(a, b):
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = torch.tanh(acc * 0.05).to(a.dtype)
    return out


@_config_kernel(_GEMM)
def mm_tanh_where(a, b):
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        t = torch.tanh(acc * 0.05)
        out[tm, tn] = torch.where(acc > 1e9, acc, t).to(a.dtype)
    return out


@_config_kernel(helion.Config(block_sizes=[64, 64]))
def row_tanh_max(x):
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, torch.tanh(x[tile_m, tile_n].to(torch.float32)))
        out[tile_m, :] = torch.amax(top, dim=-1, keepdim=True).to(x.dtype)
    return out


@_config_kernel(helion.Config(block_sizes=[64, 64]))
def row_max_tanh(x):
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        out[tile_m, :] = torch.tanh(torch.amax(top, dim=-1, keepdim=True)).to(x.dtype)
    return out
