"""Unit tests for the host-code generator (host_code.py), in isolation."""

from __future__ import annotations

import ast

import helion
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
import torch

from helion_mlir_backend._compiler.mlir.host_code import build_host_function


def _no_block_sizes(block_id: int) -> int:
    raise AssertionError(f"unexpected block size lookup for block {block_id}")


def _finish(host) -> object:
    try:
        next(host)
    except StopIteration as done:
        return done.value
    raise AssertionError("the host function yields once")


def _compile_host_function(fn, args):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings

    settings = Settings()
    settings.backend = "mlir"
    env = CompileEnvironment(args[0].device, settings)
    with env:
        fake_args = [
            env.to_fake(arg, ArgumentOrigin(f"a{i}")) for i, arg in enumerate(args)
        ]
        compiler = KernelCompiler(env)
        return compiler.compile(fn, fake_args, {})


def test_extra_host_tensor_is_recomputed_for_real():
    @helion.kernel(static_shapes=True)
    def host_side_scale(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 100.0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = x[tm, tn] + s
        return out

    x = torch.randn(8, 8)
    hf = _compile_host_function(host_side_scale.fn, [x])
    local_vars = next(build_host_function(hf, _no_block_sizes)(x))

    assert "scale" in local_vars
    torch.testing.assert_close(local_vars["scale"], x.mean() * 100.0)


def test_host_for_loop_is_preserved_and_reexecuted():
    @helion.kernel(static_shapes=True)
    def with_host_loop(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        total = 0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for i in range(3):
            total = total + i
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn]
        return out

    x = torch.randn(4, 4)
    hf = _compile_host_function(with_host_loop.fn, [x])
    local_vars = next(build_host_function(hf, _no_block_sizes)(x))

    assert local_vars["total"] == 3


def test_statements_after_the_device_loops_run_on_resume():
    @helion.kernel(static_shapes=True)
    def simple_copy(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + 1.0
        return out.sum()

    x = torch.randn(4, 4)
    hf = _compile_host_function(simple_copy.fn, [x])
    body_before = [ast.dump(stmt) for stmt in hf.body]
    host = build_host_function(hf, _no_block_sizes)(x)

    local_vars = next(host)
    torch.testing.assert_close(local_vars["out"], torch.zeros(4, 4))
    local_vars["out"].fill_(1.0)  # stands in for the compiled kernel's writes
    assert _finish(host) == 16.0
    # Helion's AST is untouched, so the host code can be rebuilt for another config.
    assert [ast.dump(stmt) for stmt in hf.body] == body_before


def test_host_block_size_resolves_to_config_value():
    @helion.kernel(static_shapes=True)
    def blocked_rows(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        block_n = hl.register_block_size(n)
        out = torch.zeros((m, n // block_n), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            for tn in hl.tile(n, block_size=block_n):
                out[tm, tn.id] = x[tm, tn].sum(-1)
        return out

    x = torch.randn(4, 16)
    hf = _compile_host_function(blocked_rows.fn, [x])
    host = build_host_function(hf, lambda block_id: 8)(x)

    assert next(host)["block_n"] == 8
    assert _finish(host).shape == (4, 2)


def test_host_specialize_and_tunable_become_constants():
    @helion.kernel(static_shapes=True)
    def specialized(x: torch.Tensor) -> torch.Tensor:
        n = hl.specialize(x.size(1))
        chunks = hl.register_tunable("chunks", PowerOfTwoFragment(1, 8, 2))
        out = torch.zeros((x.size(0), n * chunks), dtype=x.dtype, device=x.device)
        for tm in hl.tile(x.size(0)):
            out[tm, :n] = x[tm, :]
        return out

    x = torch.randn(4, 6)
    hf = _compile_host_function(specialized.fn, [x])
    config = helion.Config(chunks=4)
    local_vars = next(build_host_function(hf, _no_block_sizes, config)(x))

    assert (local_vars["n"], local_vars["chunks"]) == (6, 4)


def test_kernel_without_return_returns_none():
    @helion.kernel(static_shapes=True)
    def write_only(x: torch.Tensor, out: torch.Tensor) -> None:
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile]

    x = torch.randn(4)
    hf = _compile_host_function(write_only.fn, [x, torch.empty(4)])
    host = build_host_function(hf, _no_block_sizes)(x, torch.empty(4))
    next(host)
    assert _finish(host) is None


def test_early_return_skips_the_device_loops():
    @helion.kernel(static_shapes=True)
    def early_return(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        if m == 4:
            return out
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn]
        return out

    x = torch.randn(4, 4)
    hf = _compile_host_function(early_return.fn, [x])
    result = _finish(build_host_function(hf, _no_block_sizes)(x))
    torch.testing.assert_close(result, torch.zeros(4, 4))
