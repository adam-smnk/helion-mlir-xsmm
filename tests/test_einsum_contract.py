"""Direct ``torch.einsum`` -> ``linalg.contract`` lowering.

Covers the equation analysis (positive and negative cases), the fact that
non-contractible equations silently fall back to PyTorch's decomposition, and
numerical correctness of the generated kernels.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import execute_module as _execute

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.support.einsum_spec import EinsumNotContractible
from helion_mlir_backend._compiler.mlir.support.einsum_spec import build_contract_spec
from helion_mlir_backend._compiler.mlir.support.einsum_spec import is_contractible

# ---------------------------------------------------------------------------
# Equation analysis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("equation", "shapes"),
    [
        ("mk,kn->mn", [[4, 8], [8, 16]]),
        ("mk,kn", [[4, 8], [8, 16]]),  # implicit output
        ("mk,nk->mn", [[4, 8], [16, 8]]),  # transposed rhs
        ("km,kn->mn", [[8, 4], [8, 16]]),  # transposed lhs
        ("bmk,bkn->bmn", [[2, 4, 8], [2, 8, 16]]),  # batch
        ("bmk,kn->bmn", [[2, 4, 8], [8, 16]]),  # broadcast rhs over batch
        ("mkl,kln->mn", [[4, 8, 2], [8, 2, 16]]),  # two reduction dims
        ("mklp,klpn->mn", [[4, 8, 2, 3], [8, 2, 3, 16]]),  # three reduction dims
        ("mk,kn->nm", [[4, 8], [8, 16]]),  # permuted output
        ("m,n->mn", [[4], [8]]),  # outer product, no reduction
        ("ij,ij->ij", [[4, 8], [4, 8]]),  # elementwise, no reduction
        ("mn,mn->", [[4, 8], [4, 8]]),  # full reduction to a scalar
    ],
)
def test_contractible_equations(equation, shapes):
    assert is_contractible(equation, shapes)


@pytest.mark.parametrize(
    ("equation", "shapes"),
    [
        ("m,n->mn", [[4], [8]]),
        ("ij,ij->ij", [[4, 8], [4, 8]]),
    ],
)
def test_reduction_free_equations_are_left_elementwise(equation, shapes):
    """Legal as a contraction, but nothing is contracted: keep the plain
    elementwise lowering instead."""
    assert is_contractible(equation, shapes)
    assert not is_contractible(equation, shapes, require_reduction=True)


@pytest.mark.parametrize(
    ("equation", "shapes", "message"),
    [
        ("mk->km", [[4, 8]], "exactly 2 inputs"),
        ("mk,kn,np->mp", [[4, 8], [8, 16], [16, 2]], "exactly 2 inputs"),
        ("ii,jj->ij", [[4, 4], [8, 8]], "repeated subscript"),
        ("mk,kn->mm", [[4, 8], [8, 16]], "repeated subscript"),
        ("mk,kn->mp", [[4, 8], [8, 16]], "appear in no input"),
        ("mkl,kn->mn", [[4, 8, 2], [8, 16]], "reduced over a single operand"),
        ("...k,kn->...n", [[4, 8], [8, 16]], "ellipsis"),
        ("mk,kn->mn", [[4, 8], [4, 16]], "bound to sizes"),
        ("mk,kn->mn", [[4, 1], [8, 16]], "bound to sizes"),  # size-1 broadcast
        ("mk,kn->mn", [[4, 8, 2], [8, 16]], "does not match operand rank"),
        ("mk,kn->mn", [[4, 8]], "operand"),
    ],
)
def test_non_contractible_equations(equation, shapes, message):
    assert not is_contractible(equation, shapes)
    with pytest.raises(EinsumNotContractible, match=message):
        build_contract_spec(equation, shapes)


def test_contract_spec_iteration_space():
    spec = build_contract_spec("mkl,kln->mn", [[4, 8, 2], [8, 2, 16]])
    assert spec.iteration_dims == ("m", "n", "k", "l")
    assert spec.lhs == (0, 2, 3)
    assert spec.rhs == (2, 3, 1)
    assert spec.out == (0, 1)
    assert spec.out_shape == [4, 16]
    assert spec.reduction_dims == ("k", "l")


# ---------------------------------------------------------------------------
# Symbolic extents
# ---------------------------------------------------------------------------


def _unbacked_symints(count):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    return shape_env, [shape_env.create_unbacked_symint() for _ in range(count)]


def test_extent_key_identifies_symbols_without_evaluating_them():
    from helion_mlir_backend._compiler.mlir.support.einsum_spec import _extent_key

    shape_env, (first, second) = _unbacked_symints(2)
    before = len(shape_env.guards)

    assert _extent_key(8) == 8
    assert _extent_key(None) is None
    assert _extent_key(first) == _extent_key(first)
    assert _extent_key(first) != _extent_key(second)
    assert len(shape_env.guards) == before


def test_shared_symbol_proves_a_single_extent():
    shape_env, (k,) = _unbacked_symints(1)
    before = len(shape_env.guards)

    assert is_contractible("mk,kn->mn", [[4, k], [k, 16]])
    # The contracted dim was matched by symbol, not by evaluating it.
    assert len(shape_env.guards) == before


def test_distinct_symbols_are_not_assumed_to_differ():
    """Two symbols may still resolve to the same extent, so this stays a
    contraction rather than being rejected on an unprovable mismatch."""
    _, (first, second) = _unbacked_symints(2)

    assert is_contractible("mk,kn->mn", [[4, first], [second, 16]])


def test_symbolic_extent_paired_with_static_size_one_is_rejected():
    """einsum would broadcast the size-1 side; equality cannot be proven here,
    so the equation must not be captured."""
    _, (k,) = _unbacked_symints(1)

    assert is_contractible("mk,kn->mn", [[4, k], [8, 16]])
    assert not is_contractible("mk,kn->mn", [[4, k], [1, 16]])
    with pytest.raises(EinsumNotContractible, match="size-1 extent"):
        build_contract_spec("mk,kn->mn", [[4, k], [1, 16]])


def test_matching_static_size_one_extents_are_fine():
    """Both sides are 1: nothing is broadcast, so this is a real contraction."""
    assert is_contractible("mk,kn->mn", [[4, 1], [1, 16]])


# ---------------------------------------------------------------------------
# Indexing maps
# ---------------------------------------------------------------------------

# Batch `b`, an M-like `m`, an N-like `n`, and two contracted dims `k`/`l`,
# with `n` before the contracted dims on the rhs so the maps are not a plain
# left-to-right reading of the operand shapes.
_HIGH_DIM_EQUATION = "bmkl,bnkl->bmn"
_HIGH_DIM_SHAPES = [[2, 4, 8, 3], [2, 6, 8, 3]]
_HIGH_DIM_MAPS = [
    "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>",
    "affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>",
    "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>",
]


def _linalg_contract_ops(module):
    """Return every ``linalg.contract`` operation in *module*, at any depth."""
    found = []

    def visit(operation):
        if operation.name == "linalg.contract":
            found.append(operation)
        for region in operation.regions:
            for block in region:
                for child in block.operations:
                    visit(child.operation)

    visit(module.operation)
    return found


def _indexing_map_strings(operation):
    import mlir.ir as ir

    return [str(entry) for entry in ir.ArrayAttr(operation.attributes["indexing_maps"])]


def test_contract_spec_high_dim_multi_reduction():
    spec = build_contract_spec(_HIGH_DIM_EQUATION, _HIGH_DIM_SHAPES)
    assert spec.iteration_dims == ("b", "m", "n", "k", "l")
    assert spec.lhs == (0, 1, 3, 4)
    assert spec.rhs == (0, 2, 3, 4)
    assert spec.out == (0, 1, 2)
    assert spec.reduction_dims == ("k", "l")
    assert spec.out_shape == [2, 4, 6]


def test_indexing_maps_for_high_dim_multi_reduction():
    import mlir.ir as ir

    from helion_mlir_backend._compiler.mlir.codegen import _get_shared_mlir_context
    from helion_mlir_backend._compiler.mlir.lowering.contraction_ops import (
        indexing_maps,
    )

    spec = build_contract_spec(_HIGH_DIM_EQUATION, _HIGH_DIM_SHAPES)
    with _get_shared_mlir_context(), ir.Location.unknown():
        maps = [str(ir.AffineMapAttr.get(m)) for m in indexing_maps(spec)]
    assert maps == _HIGH_DIM_MAPS


def test_generated_contract_carries_expected_indexing_maps():
    @helion.kernel(static_shapes=True)
    def einsum_high_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, m, k, ell = x.shape
        b2, n, k2, l2 = y.shape
        out = torch.zeros((b, m, n), dtype=x.dtype, device=x.device)
        for tb in hl.tile(b):
            for tm, tn in hl.tile([m, n]):
                out[tb, tm, tn] = torch.einsum(
                    "bmkl,bnkl->bmn", x[tb, tm, :, :], y[tb, tn, :, :]
                )
        return out

    x = torch.randn(2, 8, 4, 3)
    y = torch.randn(2, 12, 4, 3)
    module = generate_mlir(
        einsum_high_dim, [x, y], config=helion.Config(block_sizes=[1, 4, 6])
    )

    contracts = _linalg_contract_ops(module)
    assert contracts
    for operation in contracts:
        assert _indexing_map_strings(operation) == _HIGH_DIM_MAPS

    actual = _execute(module, x, y, kernel_name="einsum_high_dim")
    expected = torch.einsum("bmkl,bnkl->bmn", x, y)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_generated_contract_maps_encode_a_transposed_operand():
    """A transposed rhs shows up as permuted map results, not a data movement."""

    @helion.kernel(static_shapes=True)
    def einsum_nt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n, k2 = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mk,nk->mn", x[tm, :], y[tn, :])
        return out

    module = generate_mlir(
        einsum_nt,
        [torch.randn(16, 8), torch.randn(12, 8)],
        config=helion.Config(block_sizes=[8, 4]),
    )

    contracts = _linalg_contract_ops(module)
    assert contracts
    for operation in contracts:
        assert _indexing_map_strings(operation) == [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d1, d2)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ]
    assert "linalg.transpose" not in str(module)


# ---------------------------------------------------------------------------
# Code generation + numerics
# ---------------------------------------------------------------------------


def test_einsum_matmul_reduction_loop():
    @helion.kernel(static_shapes=True)
    def einsum_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=x.dtype)
            for tk in hl.tile(k):
                acc = acc + torch.einsum("mk,kn->mn", x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
        return out

    x = torch.randn(32, 48)
    y = torch.randn(48, 16)
    module = generate_mlir(
        einsum_matmul, [x, y], config=helion.Config(block_sizes=[16, 8, 16])
    )
    assert "linalg.contract" in str(module)
    actual = _execute(module, x, y, kernel_name="einsum_matmul")
    torch.testing.assert_close(actual, x @ y, atol=1e-4, rtol=1e-4)


def test_einsum_transposed_rhs():
    @helion.kernel(static_shapes=True)
    def einsum_nt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n, k2 = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mk,nk->mn", x[tm, :], y[tn, :])
        return out

    x = torch.randn(32, 24)
    y = torch.randn(16, 24)
    module = generate_mlir(einsum_nt, [x, y], config=helion.Config(block_sizes=[16, 8]))
    assert "linalg.contract" in str(module)
    actual = _execute(module, x, y, kernel_name="einsum_nt")
    torch.testing.assert_close(actual, x @ y.T, atol=1e-4, rtol=1e-4)


def test_einsum_multiple_reduction_dims():
    @helion.kernel(static_shapes=True)
    def einsum_two_reductions(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k, ell = x.shape
        k2, l2, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mkl,kln->mn", x[tm, :, :], y[:, :, tn])
        return out

    x = torch.randn(16, 8, 4)
    y = torch.randn(8, 4, 12)
    module = generate_mlir(
        einsum_two_reductions, [x, y], config=helion.Config(block_sizes=[8, 4])
    )
    # Both k and l stay reduction dims of one contraction: a 4-dim iteration
    # space (m, n, k, l) over 3-D operands, not a decomposed bmm chain.
    contracts = _linalg_contract_ops(module)
    assert contracts
    for operation in contracts:
        assert _indexing_map_strings(operation) == [
            "affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
        ]
    actual = _execute(module, x, y, kernel_name="einsum_two_reductions")
    expected = torch.einsum("mkl,kln->mn", x, y)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_einsum_broadcast_operand_over_batch():
    """An operand that omits the batch subscript is broadcast by its map."""

    @helion.kernel(static_shapes=True)
    def einsum_broadcast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((b, m, n), dtype=x.dtype, device=x.device)
        for tb in hl.tile(b):
            for tm, tn in hl.tile([m, n]):
                out[tb, tm, tn] = torch.einsum("bmk,kn->bmn", x[tb, tm, :], y[:, tn])
        return out

    x = torch.randn(3, 16, 8)
    y = torch.randn(8, 12)
    module = generate_mlir(
        einsum_broadcast, [x, y], config=helion.Config(block_sizes=[1, 8, 4])
    )

    contracts = _linalg_contract_ops(module)
    assert contracts
    for operation in contracts:
        assert _indexing_map_strings(operation) == [
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d3, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>",
        ]

    actual = _execute(module, x, y, kernel_name="einsum_broadcast")
    expected = torch.einsum("bmk,kn->bmn", x, y)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_size_one_broadcast_is_not_captured():
    """einsum broadcasts a size-1 shared dim; linalg.contract cannot, so the
    equation must be left to PyTorch even though its structure qualifies."""
    from helion_mlir_backend._compiler.mlir.trace_mode import _should_capture

    assert _should_capture("mk,kn->mn", [torch.randn(4, 8), torch.randn(8, 6)])
    assert not _should_capture("mk,kn->mn", [torch.randn(4, 1), torch.randn(8, 6)])


def test_einsum_batch_matmul():
    @helion.kernel(static_shapes=True)
    def einsum_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, m, k = x.shape
        b2, k2, n = y.shape
        out = torch.zeros((b, m, n), dtype=x.dtype, device=x.device)
        for tb in hl.tile(b):
            for tm, tn in hl.tile([m, n]):
                out[tb, tm, tn] = torch.einsum(
                    "bmk,bkn->bmn", x[tb, tm, :], y[tb, :, tn]
                )
        return out

    x = torch.randn(3, 16, 8)
    y = torch.randn(3, 8, 12)
    module = generate_mlir(
        einsum_bmm, [x, y], config=helion.Config(block_sizes=[1, 8, 4])
    )
    assert "linalg.contract" in str(module)
    actual = _execute(module, x, y, kernel_name="einsum_bmm")
    torch.testing.assert_close(actual, torch.bmm(x, y), atol=1e-4, rtol=1e-4)


def test_einsum_implicit_output():
    @helion.kernel(static_shapes=True)
    def einsum_implicit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mk,kn", x[tm, :], y[:, tn])
        return out

    x = torch.randn(24, 16)
    y = torch.randn(16, 8)
    module = generate_mlir(
        einsum_implicit, [x, y], config=helion.Config(block_sizes=[8, 4])
    )
    assert "linalg.contract" in str(module)
    actual = _execute(module, x, y, kernel_name="einsum_implicit")
    torch.testing.assert_close(actual, x @ y, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_einsum_dtypes(dtype):
    @helion.kernel(static_shapes=True)
    def einsum_typed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mk,kn->mn", x[tm, :], y[:, tn])
        return out

    x = torch.randn(16, 8, dtype=dtype)
    y = torch.randn(8, 16, dtype=dtype)
    module = generate_mlir(
        einsum_typed, [x, y], config=helion.Config(block_sizes=[8, 8])
    )
    assert "linalg.contract" in str(module)
    actual = _execute(module, x, y, kernel_name="einsum_typed")
    torch.testing.assert_close(actual, x @ y, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Negative / fallback paths
# ---------------------------------------------------------------------------


def _device_ir_nodes(kernel, args, *, backend="mlir"):
    """Compile *kernel* to device IR and return every FX node in it."""
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings

    settings = Settings()
    settings.backend = backend
    env = CompileEnvironment(torch.device("cpu"), settings)
    names = list(kernel.signature.parameters)
    with env:
        fake = [
            env.to_fake(arg, ArgumentOrigin(name))
            for name, arg in zip(names, args, strict=True)
        ]
        host_function = KernelCompiler(env).compile(kernel.fn, fake, {})
    return [
        node
        for graph_info in host_function.device_ir.graphs
        for node in graph_info.graph.nodes
    ]


def _assert_not_captured(kernel, args):
    from helion_mlir_backend._compiler.mlir.trace_mode import is_einsum_node

    nodes = _device_ir_nodes(kernel, args)
    assert nodes
    assert not any(is_einsum_node(node) for node in nodes)


@pytest.mark.parametrize(
    ("equation", "shapes", "captured"),
    [
        ("mk,kn->mn", [(4, 8), (8, 16)], True),
        ("mk,nk->mn", [(4, 8), (16, 8)], True),
        ("mkl,kln->mn", [(4, 8, 2), (8, 2, 16)], True),
        ("bmk,bkn->bmn", [(2, 4, 8), (2, 8, 16)], True),
        ("bmk,kn->bmn", [(2, 4, 8), (8, 16)], True),  # broadcast by omitted dim
        ("mk,kn->mn", [(4, 1), (8, 16)], False),  # size-1 broadcast
        ("m,n->mn", [(4,), (8,)], False),  # no contracted dim
        ("ij,ij->ij", [(4, 8), (4, 8)], False),  # no contracted dim
        ("mn->nm", [(4, 8)], False),  # single operand
        ("mk,kn,np->mp", [(4, 8), (8, 16), (16, 2)], False),  # three operands
        ("mkl,kn->mn", [(4, 8, 2), (8, 16)], False),  # reduced over one operand
        ("kk,kn->kn", [(8, 8), (8, 16)], False),  # diagonal
        ("...k,kn->...n", [(4, 8), (8, 16)], False),  # ellipsis
    ],
)
def test_capture_mode_only_rewrites_contractible_equations(equation, shapes, captured):
    from torch.fx.experimental.proxy_tensor import make_fx

    from helion_mlir_backend._compiler.mlir.trace_mode import TraceMode
    from helion_mlir_backend._compiler.mlir.trace_mode import einsum_op_target

    operands = [torch.randn(shape) for shape in shapes]
    with TraceMode():
        traced = make_fx(lambda *tensors: torch.einsum(equation, *tensors))(*operands)

    targets = [node.target for node in traced.graph.nodes]
    assert (einsum_op_target() in targets) is captured
    torch.testing.assert_close(traced(*operands), torch.einsum(equation, *operands))


def test_three_operand_einsum_is_rejected():
    @helion.kernel(static_shapes=True)
    def einsum_chain(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        p = z.shape[1]
        out = torch.zeros((m, p), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm, :] = torch.einsum("mk,kn,np->mp", x[tm, :], y[:, :], z[:, :])
        return out

    _assert_not_captured(
        einsum_chain, [torch.randn(16, 8), torch.randn(8, 12), torch.randn(12, 4)]
    )


def test_single_operand_einsum_is_rejected():
    @helion.kernel(static_shapes=True)
    def einsum_permute(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((n, m), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tn, tm] = torch.einsum("mn->nm", x[tm, tn])
        return out

    _assert_not_captured(einsum_permute, [torch.randn(16, 8)])


def test_partial_single_operand_reduction_is_rejected():
    """``l`` is summed out of one operand only: no ``linalg.contract`` iterator."""

    @helion.kernel(static_shapes=True)
    def einsum_partial(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k, ell = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mkl,kn->mn", x[tm, :, :], y[:, tn])
        return out

    _assert_not_captured(einsum_partial, [torch.randn(16, 8, 4), torch.randn(8, 12)])


def test_repeated_subscript_einsum_is_rejected():
    """A diagonal access is not a projected permutation.

    Helion's own shared lowering pass cannot handle the resulting
    ``aten.diagonal``, so the rejection is observable as a compile error
    rather than as a working fallback kernel.
    """
    import helion.exc

    @helion.kernel(static_shapes=True)
    def einsum_diagonal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k, k2 = x.shape
        k3, n = y.shape
        out = torch.zeros((k, n), dtype=x.dtype, device=x.device)
        for tn in hl.tile(n):
            out[:, tn] = torch.einsum("kk,kn->kn", x[:, :], y[:, tn])
        return out

    with pytest.raises(helion.exc.InductorLoweringError, match="aten.diagonal"):
        _device_ir_nodes(einsum_diagonal, [torch.randn(8, 8), torch.randn(8, 12)])


def test_single_operand_einsum_falls_back_to_decomposition():
    @helion.kernel(static_shapes=True)
    def einsum_permute(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((n, m), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tn, tm] = torch.einsum("mn->nm", x[tm, tn])
        return out

    x = torch.randn(16, 8)
    module = generate_mlir(
        einsum_permute, [x], config=helion.Config(block_sizes=[8, 4])
    )
    assert "linalg.contract" not in str(module)
    actual = _execute(module, x, kernel_name="einsum_permute")
    torch.testing.assert_close(actual, x.T, atol=1e-4, rtol=1e-4)


def test_codegen_rejects_inconsistent_operand_shapes():
    """The tracing-time check runs on symbolic extents, so the MLIR-level
    analysis must reject a shape mismatch that only concrete tiles reveal."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from helion_mlir_backend._compiler.mlir.analysis.contractions import (
        match_contraction,
    )
    from helion_mlir_backend._compiler.mlir.codegen import _get_shared_mlir_context
    from helion_mlir_backend._compiler.mlir.lowering.contraction_ops import (
        emit_contraction,
    )
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

    graph = torch.fx.Graph()
    lhs_node = graph.placeholder("lhs")
    rhs_node = graph.placeholder("rhs")
    einsum_node = graph.call_function(
        torch.ops.helion_mlir.einsum.default, ("mk,kn->mn", [lhs_node, rhs_node])
    )

    context = _get_shared_mlir_context()
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            lhs = tensor_d.EmptyOp([4, 8], f32).result
            rhs = tensor_d.EmptyOp([16, 32], f32).result
            ctx = _StubBuildContext({lhs_node: lhs, rhs_node: rhs})
            with pytest.raises(UnsupportedOperationError, match="bound to sizes"):
                emit_contraction(ctx, match_contraction(einsum_node))


class _StubBuildContext:
    """Minimal ``BuildContext`` surface used by ``emit_contraction``."""

    def __init__(self, values):
        self._values = values

    def get_value(self, node):
        return self._values.get(node)


def test_genuinely_mismatched_extents_still_fail_in_type_propagation():
    """Capture wraps device-IR lowering only, so einsum's own shape validation
    still runs first and reports the error at the call site."""
    import helion.exc

    @helion.kernel(static_shapes=True)
    def einsum_mismatched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        j, n = y.shape
        out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = torch.einsum("mk,kn->mn", x[tm, :], y[:, tn])
        return out

    with pytest.raises(helion.exc.TorchOpTracingError, match="does not broadcast"):
        generate_mlir(
            einsum_mismatched,
            [torch.randn(16, 8), torch.randn(12, 6)],
            config=helion.Config(block_sizes=[8, 3]),
        )


def test_capture_is_scoped_to_the_mlir_backend():
    """A non-MLIR backend must keep PyTorch's own einsum decomposition."""
    from helion_mlir_backend._compiler.mlir.trace_mode import is_einsum_node

    nodes = _device_ir_nodes(
        _triton_einsum_kernel,
        [torch.randn(16, 16), torch.randn(16, 16)],
        backend="triton",
    )
    assert nodes
    assert not any(is_einsum_node(node) for node in nodes)


@helion.kernel(static_shapes=True)
def _triton_einsum_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    k2, n = y.shape
    out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.einsum("mk,kn->mn", x[tm, :], y[:, tn])
    return out
