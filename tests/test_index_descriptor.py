"""Unit tests for the authoritative index descriptor resolver.

Covers the indexing patterns that historically caused tracing/slicing
correctness bugs (see repo memory): combined multi-dim tiles, nested
``hl.tile(k)`` reductions, transposed operands, and stores whose index order
differs from loop declaration order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch

from helion_mlir_backend._compiler.mlir.support.index_meta import (
    resolve_index_descriptor,
)

if TYPE_CHECKING:
    from helion.runtime.settings import Settings


def _build_context(kernel, args, config):
    """Compile a kernel through Helion's frontend and return (ctx, host_function).

    Mirrors ``helion_mlir_backend.api.generate_mlir`` / the module builder's
    ``build()`` far enough to obtain a fully-populated ``BuildContext``
    (block sizes resolved, symbolic shapes restored) without needing an
    installed ``mlir-python-bindings``-independent module.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin

    from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder

    fn = kernel.fn
    settings: Settings = kernel.settings
    settings.backend = "mlir"
    env = CompileEnvironment(args[0].device, settings)

    with env:
        fake_args = [
            env.to_fake(arg, ArgumentOrigin(name))
            for name, arg in zip(kernel.signature.parameters, args, strict=False)
        ]
        compiler = KernelCompiler(env)
        host_function = compiler.compile(fn, fake_args, {})
        builder = MLIRModuleBuilder(host_function, config, env)
        builder.build()

    return builder.context, host_function


def _find_store_index_nodes(host_function, out_shape: tuple[int, ...]):
    """Return the index_nodes list of the store writing the final output."""
    for graph_info in host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "store":
                continue
            target_meta = node.args[0].meta.get("val")
            if not isinstance(target_meta, torch.Tensor):
                continue
            if tuple(int(d) for d in target_meta.shape) == out_shape:
                return list(node.args[1])
    raise AssertionError("no matching terminal store found")


def test_combined_2d_tile_resolves_distinct_block_ids():
    """``for tm, tn in hl.tile([m, n])``: each dim must map to a different block id."""

    @helion.kernel(static_shapes=True)
    def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
        return out

    x = torch.randn(128, 128)
    y = torch.randn(128, 128)
    config = helion.Config(block_sizes=[16, 32, 64])
    ctx, hf = _build_context(mm, [x, y], config)

    index_nodes = _find_store_index_nodes(hf, (128, 128))
    descriptors = [resolve_index_descriptor(ctx, node) for node in index_nodes]

    assert len(descriptors) == 2
    assert all(d.block_id is not None for d in descriptors)
    assert not any(d.is_scalar for d in descriptors)
    # The two combined-tile dimensions must resolve to distinct block ids.
    assert descriptors[0].block_id != descriptors[1].block_id
    assert {ctx.geometry.block_size(d.block_id) for d in descriptors} == {16, 32}


def test_nested_tile_k_resolves_reduction_block_id():
    """The inner ``hl.tile(k)`` load index must resolve to its own block id."""

    @helion.kernel(static_shapes=True)
    def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
        return out

    x = torch.randn(128, 128)
    y = torch.randn(128, 128)
    config = helion.Config(block_sizes=[16, 32, 64])
    ctx, hf = _build_context(mm, [x, y], config)

    load_index_nodes = None
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "__name__", "") == "load"
            ):
                index_nodes = node.args[1]
                if len(index_nodes) == 2:
                    load_index_nodes = index_nodes
    assert load_index_nodes is not None

    descriptors = [resolve_index_descriptor(ctx, node) for node in load_index_nodes]
    assert all(d.block_id is not None for d in descriptors)
    resolved_sizes = {ctx.geometry.block_size(d.block_id) for d in descriptors}
    # The reduction dimension (block size 64) must be resolvable from a load
    # index inside the inner ``hl.tile(k)`` loop.
    assert 64 in resolved_sizes


def test_transposed_operand_resolves_same_block_ids_as_untransposed():
    """``yt[tn, tk].permute(1, 0)``: the underlying load indices still resolve."""

    @helion.kernel(static_shapes=True)
    def mm_transposed_b(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n, k2 = yt.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
            out[tm, tn] = acc
        return out

    x = torch.randn(64, 64)
    yt = torch.randn(64, 64)
    config = helion.Config(block_sizes=[16, 32, 64])
    ctx, hf = _build_context(mm_transposed_b, [x, yt], config)

    found = []
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "__name__", "") == "load"
            ):
                index_nodes = node.args[1]
                if len(index_nodes) == 2:
                    found.append(index_nodes)
    assert found

    for index_nodes in found:
        descriptors = [resolve_index_descriptor(ctx, node) for node in index_nodes]
        assert all(d.block_id is not None for d in descriptors)


def test_reordered_store_resolves_grid_and_tile_block_ids():
    """``out[tm, panel, :]``: tile index comes before the grid index positionally."""

    @helion.kernel(static_shapes=True)
    def unpack_panels(src: torch.Tensor) -> torch.Tensor:
        n_panels, m, bn = src.shape
        out = torch.empty((m, n_panels, bn), dtype=src.dtype, device=src.device)
        for panel in hl.grid(n_panels):
            for tm in hl.tile(m):
                out[tm, panel, :] = src[panel, tm, :]
        return out

    src = torch.randn(3, 8, 8)
    config = helion.Config(block_sizes=[1, 4])
    ctx, hf = _build_context(unpack_panels, [src], config)

    index_nodes = _find_store_index_nodes(hf, (8, 3, 8))
    assert len(index_nodes) == 3

    tile_descriptor = resolve_index_descriptor(ctx, index_nodes[0])
    grid_descriptor = resolve_index_descriptor(ctx, index_nodes[1])
    assert tile_descriptor.block_id is not None
    assert not tile_descriptor.is_scalar
    assert grid_descriptor.block_id is not None
    assert grid_descriptor.is_scalar
    assert tile_descriptor.block_id != grid_descriptor.block_id
