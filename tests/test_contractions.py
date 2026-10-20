"""One contraction lowering: ATen matmul family, ``hl.dot``, fused accumulators."""

from __future__ import annotations

import helion
import helion.language as hl
import mlir.ir as ir
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def dot_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16))
def dot_no_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = hl.dot(x[tm, :], y[:, tn])
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def dot_out_dtype_plus_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + hl.dot(x[tm, tk], y[tk, tn], out_dtype=torch.float32)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def dot_transposed_rhs(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = yt.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], yt[tn, tk].t(), acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(1, 16, 16, 8))
def dot_batched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tb, tm, tk], y[tb, tk, tn], acc=acc)
        out[tb, tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16))
def mm_no_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.mm(x[tm, :], y[:, tn])
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def addmm_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def acc_plus_addmm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = acc + torch.addmm(bias[tm, tn], x[tm, :], y[:, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def floor_div(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] // 3
    return out


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _f32_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.float() @ y.float()


@pytest.mark.parametrize(
    ("kernel", "reference", "make_args", "tolerance"),
    [
        pytest.param(
            dot_acc, _f32_mm, lambda: [_bf16(32, 16), _bf16(16, 32)], 1e-2, id="dot_acc"
        ),
        pytest.param(
            dot_no_acc,
            _f32_mm,
            lambda: [_bf16(32, 16), _bf16(16, 32)],
            1e-2,
            id="dot_no_acc",
        ),
        pytest.param(
            dot_out_dtype_plus_acc,
            _f32_mm,
            lambda: [_bf16(32, 16), _bf16(16, 32)],
            1e-2,
            id="dot_out_dtype_plus_acc",
        ),
        pytest.param(
            dot_transposed_rhs,
            lambda x, yt: _f32_mm(x, yt.t()),
            lambda: [_bf16(32, 16), _bf16(32, 16)],
            1e-2,
            id="dot_transposed_rhs",
        ),
        pytest.param(
            dot_batched,
            lambda x, y: x.float() @ y.float(),
            lambda: [torch.randn(2, 16, 16), torch.randn(2, 16, 32)],
            1e-4,
            id="dot_batched",
        ),
        pytest.param(
            mm_no_acc,
            torch.mm,
            lambda: [_bf16(32, 16), _bf16(16, 32)],
            2e-2,
            id="mm_bf16",
        ),
        pytest.param(
            addmm_acc,
            _f32_mm,
            lambda: [torch.randn(32, 16), torch.randn(16, 32)],
            1e-4,
            id="addmm_acc",
        ),
        pytest.param(
            acc_plus_addmm,
            lambda x, y, b: b + x @ y,
            lambda: [torch.randn(16, 8), torch.randn(8, 16), torch.randn(16, 16)],
            1e-4,
            id="acc_plus_addmm",
        ),
        pytest.param(
            floor_div,
            lambda x: x // 3,
            lambda: [torch.randn(16) * 10],
            1e-5,
            id="floor_div",
        ),
    ],
)
def test_contraction_numerics(kernel, reference, make_args, tolerance) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        make_args(),
        paths=("direct", "generated"),
        atol=tolerance,
        rtol=tolerance,
    )


_CONTRACTIONS = ("linalg.matmul", "linalg.batch_matmul", "linalg.contract")


def _ops(module: ir.Module) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def walk(operation: ir.Operation) -> None:
        found.append(operation)
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    walk(module.operation)
    return found


def _contractions(module: ir.Module) -> list[ir.Operation]:
    return [op for op in _ops(module) if op.name in _CONTRACTIONS]


def _is_for_iter_arg(value: ir.Value) -> bool:
    if not isinstance(value, ir.BlockArgument):
        return False
    return value.owner.owner.name == "scf.for"


@pytest.mark.parametrize(
    ("kernel", "make_args", "op_name"),
    [
        pytest.param(
            dot_acc, lambda: [_bf16(32, 16), _bf16(16, 32)], "linalg.matmul", id="dot"
        ),
        pytest.param(
            dot_out_dtype_plus_acc,
            lambda: [_bf16(32, 16), _bf16(16, 32)],
            "linalg.matmul",
            id="acc_plus_dot",
        ),
        pytest.param(
            dot_transposed_rhs,
            lambda: [_bf16(32, 16), _bf16(32, 16)],
            "linalg.contract",
            id="transposed",
        ),
        pytest.param(
            dot_batched,
            lambda: [torch.randn(2, 16, 16), torch.randn(2, 16, 32)],
            "linalg.batch_matmul",
            id="batched",
        ),
        pytest.param(
            addmm_acc,
            lambda: [_bf16(32, 16), _bf16(16, 32)],
            "linalg.matmul",
            id="addmm",
        ),
    ],
)
def test_accumulating_contraction_is_one_op_into_the_iter_arg(
    kernel, make_args, op_name
) -> None:
    module = generate_mlir(kernel, make_args())
    contractions = _contractions(module)
    assert [op.name for op in contractions] == [op_name]
    (contraction,) = contractions
    assert _is_for_iter_arg(contraction.operands[2])
    assert not any(op.name == "linalg.transpose" for op in _ops(module))


def test_transposed_operand_is_folded_into_the_map() -> None:
    module = generate_mlir(dot_transposed_rhs, [_bf16(32, 16), _bf16(32, 16)])
    (contraction,) = _contractions(module)
    maps = [str(m) for m in contraction.attributes["indexing_maps"]]
    assert maps == [
        "affine_map<(d0, d1, d2) -> (d0, d2)>",
        "affine_map<(d0, d1, d2) -> (d1, d2)>",
        "affine_map<(d0, d1, d2) -> (d0, d1)>",
    ]


def test_bf16_contraction_without_accumulator_accumulates_in_f32() -> None:
    module = generate_mlir(mm_no_acc, [_bf16(32, 16), _bf16(16, 32)])
    (contraction,) = _contractions(module)
    assert str(contraction.results[0].type) == "tensor<16x16xf32>"
    assert "arith.truncf" in str(module)


def test_add_of_addmm_fuses_only_the_addmm() -> None:
    module = generate_mlir(
        acc_plus_addmm, [torch.randn(16, 8), torch.randn(8, 16), torch.randn(16, 16)]
    )
    assert [op.name for op in _contractions(module)] == ["linalg.matmul"]


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(16, 16, 8))
def dot_mismatched_out_dtype(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc, out_dtype=torch.float16)
        out[tm, tn] = acc
    return out


def test_lowering_errors_point_at_the_kernel_source_line() -> None:
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

    with pytest.raises(UnsupportedOperationError) as error:
        generate_mlir(dot_mismatched_out_dtype, [_bf16(32, 16), _bf16(16, 32)])
    message = str(error.value)
    assert "out_dtype" in message
    assert "test_contractions.py" in message
    assert "acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc, out_dtype=torch.float16)" in (
        message
    )
