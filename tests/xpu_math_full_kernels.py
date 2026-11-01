"""Math kernels over the full range of their inputs (no range-limiting wrappers),
for ``test_xpu_runtime.test_math_every_pattern``: ``cast_<fn>`` computes in f32
between conversions, ``nat_<fn>`` in the input's own dtype. Explicit functions:
Helion kernels are parsed from their source and cannot close over a function."""

from __future__ import annotations

import helion.language as hl
import torch
import torch.nn.functional as F


def cast_absf(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.abs(v)).to(out.dtype)
    return out


def nat_absf(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.abs(x[t])
    return out


def cast_acos(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.acos(v)).to(out.dtype)
    return out


def nat_acos(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.acos(x[t])
    return out


def cast_asin(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.asin(v)).to(out.dtype)
    return out


def nat_asin(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.asin(x[t])
    return out


def cast_atan(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.atan(v)).to(out.dtype)
    return out


def nat_atan(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.atan(x[t])
    return out


def cast_atanh(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.atanh(v)).to(out.dtype)
    return out


def nat_atanh(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.atanh(x[t])
    return out


def cast_ceil(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.ceil(v)).to(out.dtype)
    return out


def nat_ceil(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.ceil(x[t])
    return out


def cast_cos(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.cos(v)).to(out.dtype)
    return out


def nat_cos(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.cos(x[t])
    return out


def cast_cosh(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.cosh(v)).to(out.dtype)
    return out


def nat_cosh(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.cosh(x[t])
    return out


def cast_erf(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.erf(v)).to(out.dtype)
    return out


def nat_erf(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.erf(x[t])
    return out


def cast_exp(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.exp(v)).to(out.dtype)
    return out


def nat_exp(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.exp(x[t])
    return out


def cast_exp2(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.exp2(v)).to(out.dtype)
    return out


def nat_exp2(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.exp2(x[t])
    return out


def cast_expm1(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.expm1(v)).to(out.dtype)
    return out


def nat_expm1(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.expm1(x[t])
    return out


def cast_floor(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.floor(v)).to(out.dtype)
    return out


def nat_floor(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.floor(x[t])
    return out


def cast_fpowi(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.pow(v, 3)).to(out.dtype)
    return out


def nat_fpowi(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.pow(x[t], 3)
    return out


def cast_log(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.log(v)).to(out.dtype)
    return out


def nat_log(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.log(x[t])
    return out


def cast_log1p(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.log1p(v)).to(out.dtype)
    return out


def nat_log1p(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.log1p(x[t])
    return out


def cast_log2(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.log2(v)).to(out.dtype)
    return out


def nat_log2(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.log2(x[t])
    return out


def cast_powf(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.pow(v, 2.5)).to(out.dtype)
    return out


def nat_powf(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.pow(x[t], 2.5)
    return out


def cast_roundeven(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.round(v)).to(out.dtype)
    return out


def nat_roundeven(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.round(x[t])
    return out


def cast_rsqrt(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.rsqrt(v)).to(out.dtype)
    return out


def nat_rsqrt(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.rsqrt(x[t])
    return out


def cast_sin(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.sin(v)).to(out.dtype)
    return out


def nat_sin(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.sin(x[t])
    return out


def cast_sinh(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.sinh(v)).to(out.dtype)
    return out


def nat_sinh(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.sinh(x[t])
    return out


def cast_sqrt(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.sqrt(v)).to(out.dtype)
    return out


def nat_sqrt(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.sqrt(x[t])
    return out


def cast_tan(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.tan(v)).to(out.dtype)
    return out


def nat_tan(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.tan(x[t])
    return out


def cast_tanh(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.tanh(v)).to(out.dtype)
    return out


def nat_tanh(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.tanh(x[t])
    return out


def cast_powvv(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (torch.pow(v, v)).to(out.dtype)
    return out


def nat_powvv(x, out):
    for t in hl.tile(x.shape):
        out[t] = torch.pow(x[t], x[t])
    return out


def cast_softsign(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (v / (1 + torch.abs(v))).to(out.dtype)
    return out


def nat_softsign(x, out):
    for t in hl.tile(x.shape):
        out[t] = x[t] / (1 + torch.abs(x[t]))
    return out


def cast_softplus(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (F.softplus(v)).to(out.dtype)
    return out


def nat_softplus(x, out):
    for t in hl.tile(x.shape):
        out[t] = F.softplus(x[t])
    return out


def cast_copy(x, out):
    for t in hl.tile(x.shape):
        v = x[t].to(torch.float32)
        out[t] = (v).to(out.dtype)
    return out


def nat_copy(x, out):
    for t in hl.tile(x.shape):
        out[t] = x[t]
    return out


FN = {
    "absf": "torch.abs({v})",
    "acos": "torch.acos({v})",
    "asin": "torch.asin({v})",
    "atan": "torch.atan({v})",
    "atanh": "torch.atanh({v})",
    "ceil": "torch.ceil({v})",
    "cos": "torch.cos({v})",
    "cosh": "torch.cosh({v})",
    "erf": "torch.erf({v})",
    "exp": "torch.exp({v})",
    "exp2": "torch.exp2({v})",
    "expm1": "torch.expm1({v})",
    "floor": "torch.floor({v})",
    "fpowi": "torch.pow({v}, 3)",
    "log": "torch.log({v})",
    "log1p": "torch.log1p({v})",
    "log2": "torch.log2({v})",
    "powf": "torch.pow({v}, 2.5)",
    "roundeven": "torch.round({v})",
    "rsqrt": "torch.rsqrt({v})",
    "sin": "torch.sin({v})",
    "sinh": "torch.sinh({v})",
    "sqrt": "torch.sqrt({v})",
    "tan": "torch.tan({v})",
    "tanh": "torch.tanh({v})",
    "powvv": "torch.pow({v}, {v})",
    "softsign": "{v} / (1 + torch.abs({v}))",
    "softplus": "F.softplus({v})",
    "copy": "{v}",
}
