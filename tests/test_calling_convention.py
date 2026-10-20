"""Calling convention and host semantics.

Every call runs the kernel's host code, passes host tensors in place (inouts are
written through) and returns the kernel's own ``return`` value.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import run_direct
from tests.harness import scalar_pipeline

from helion_mlir_backend import compile_mlir
from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


# Helion rejects closures (ClosuresNotSupported); module globals are host tensors.
_GLOBAL_BIAS = torch.randn(16)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def add_global_bias(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + _GLOBAL_BIAS[tile]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def inplace_no_return(x: torch.Tensor) -> None:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def add_int(x: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + k
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 4))
def add_one_view(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out.view(-1)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 4))
def add_one_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def copy_plus_one(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        b[tile] = a[tile] + 1.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def write_both(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        a[tile] = a[tile] + 1.0
        b[tile] = b[tile] * 2.0


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=_cfg(8),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def add_host_scale(x: torch.Tensor) -> torch.Tensor:
    scale = x.mean()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + hl.load(scale, [])
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def row_sums_blocked(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    block_n = hl.register_block_size(n)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, block_n], dtype=x.dtype)
        for tn in hl.tile(n, block_size=block_n):
            acc += x[tm, tn]
        out[tm] = acc.sum(-1)
    return out, n // block_n


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 8))
def row_block_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros((m, n // 8), dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn.id] = x[tm, tn].sum(-1)
    return out


def test_inplace_update_without_return() -> None:
    x = torch.randn(16)
    expected = x + 1.0
    assert run_direct(inplace_no_return, [x]) is None
    torch.testing.assert_close(x, expected)


def test_int_scalar_changes_between_calls() -> None:
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(add_int, [x, 2]), x + 2)
    torch.testing.assert_close(run_direct(add_int, [x, 5]), x + 5)


def test_return_view_expression() -> None:
    x = torch.randn(8, 8)
    torch.testing.assert_close(run_direct(add_one_view, [x]), (x + 1.0).view(-1))


def test_non_contiguous_out_parameter_is_written_back() -> None:
    x = torch.randn(8, 8)
    storage = torch.zeros(8, 8)
    out = storage.t()
    result = run_direct(add_one_into, [x, out])
    assert result is out
    torch.testing.assert_close(out, x + 1.0)
    torch.testing.assert_close(storage, (x + 1.0).t())


def test_input_aliasing_a_written_argument_reads_its_prior_value() -> None:
    x = torch.randn(16)
    expected = x + 1.0
    run_direct(copy_plus_one, [x, x])
    torch.testing.assert_close(x, expected)


def test_two_written_arguments_sharing_memory_are_rejected() -> None:
    x = torch.randn(16)
    with pytest.raises(UnsupportedOperationError, match="share memory"):
        run_direct(write_both, [x, x])


def test_host_block_size_is_the_config_value() -> None:
    x = torch.randn(8, 32)
    out, blocks = run_direct(row_sums_blocked, [x])
    assert blocks == 4
    torch.testing.assert_close(out, x.sum(-1), atol=1e-5, rtol=1e-5)


def test_tile_id_as_store_index() -> None:
    x = torch.randn(8, 32)
    expected = x.view(8, 4, 8).sum(-1)
    torch.testing.assert_close(
        run_direct(row_block_sums, [x]), expected, atol=1e-5, rtol=1e-5
    )


def test_module_global_tensor_is_read_at_call_time() -> None:
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(add_global_bias, [x]), x + _GLOBAL_BIAS)


def test_compile_mlir_has_kernel_call_semantics() -> None:
    x = torch.randn(16)
    with scalar_pipeline():
        fn = compile_mlir(add_int, [x, 1])
        torch.testing.assert_close(fn(x, 3), x + 3)
        torch.testing.assert_close(fn(x, 4), x + 4)


def test_generate_mlir_leaves_kernel_settings_unchanged() -> None:
    @helion.kernel(static_shapes=True, config=_cfg(8))
    def add_one(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + 1.0
        return out

    before = add_one.settings.backend
    generate_mlir(add_one, [torch.randn(16)])
    assert add_one.settings.backend == before


def test_entry_function_states_argument_roles() -> None:
    text = str(generate_mlir(copy_plus_one, [torch.randn(16), torch.randn(16)]))
    assert "func.func private @copy_plus_one__phase0" in text
    assert "bufferization.to_tensor %arg0 restrict writable" in text
    assert "bufferization.to_tensor %arg1 restrict :" in text
    assert "materialize_in_destination" in text
    assert 'helion.name = "b", helion.param = 1 : i64, helion.role = "inout"' in text


def test_execute_mlir_rejects_host_computed_inputs() -> None:
    x = torch.randn(16)
    module = generate_mlir(add_host_scale, [x])
    with pytest.raises(UnsupportedOperationError, match="host value 'scale'"):
        MLIRBackend().execute_mlir(module, x, kernel_name="add_host_scale")


def test_execute_mlir_writes_declared_parameters_in_place() -> None:
    a, b = torch.randn(16), torch.randn(16)
    expected = a + 1.0
    module = generate_mlir(copy_plus_one, [a, b])
    with scalar_pipeline():
        result = MLIRBackend().execute_mlir(module, a, b, kernel_name="copy_plus_one")
    assert result is b
    torch.testing.assert_close(b, expected)


def offset_columns(x: torch.Tensor) -> torch.Tensor:
    y = x[:, 2:]
    out = torch.empty(y.size(), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] * 2.0
    return out


def even_rows(x: torch.Tensor) -> torch.Tensor:
    y = x[::2]
    out = torch.empty(y.size(), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] + 1.0
    return out


def transposed_view(x: torch.Tensor) -> torch.Tensor:
    y = x.t()
    out = torch.empty(y.size(), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] + 1.0
    return out


def leading_columns(x: torch.Tensor) -> torch.Tensor:
    y = x[:, :3]
    out = torch.empty(y.size(), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] - 1.0
    return out


@pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("fn", "reference"),
    [
        (offset_columns, lambda x: x[:, 2:] * 2.0),
        (even_rows, lambda x: x[::2] + 1.0),
        (transposed_view, lambda x: x.t() + 1.0),
        (leading_columns, lambda x: x[:, :3] - 1.0),
    ],
    ids=["offset_columns", "even_rows", "transposed", "leading_columns"],
)
def test_host_views_that_are_not_reshapes(fn, reference, static: bool) -> None:
    """Only a reshape aliases its parameter; other views are inputs of their own."""
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=static,
        config=_cfg(4, 4),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    x = torch.randn(6, 7)
    torch.testing.assert_close(run_direct(kernel, [x]), reference(x))
