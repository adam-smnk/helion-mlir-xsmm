"""Integration tests for MLIR backend with downstream compilation targets.

Tests that the generated MLIR is valid and suitable for downstream
compilation pipelines (e.g., Triton, LLVM).
"""

from __future__ import annotations

import re

import helion
import helion.language as hl
import pytest
import torch

from helion_mlir_backend import generate_mlir


class TestDownstreamIntegration:
    """Test integration with downstream compilers."""

    def test_matmul_mlir_structure(self):
        """Test that matmul MLIR has expected structure for Triton."""

        @helion.kernel(static_shapes=True)
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        device = torch.device("cpu")
        x = torch.randn(128, 256, dtype=torch.float32, device=device)
        y = torch.randn(256, 192, dtype=torch.float32, device=device)

        module = generate_mlir(matmul_kernel, [x, y])
        ir_str = str(module)

        # Verify key MLIR constructs for downstream
        assert "func.func" in ir_str, "Should have func dialect"
        assert "scf.forall" in ir_str, "Should have parallel loops"
        assert "scf.for" in ir_str, "Should have sequential loops"
        assert "linalg.matmul" in ir_str, "Should have linalg operations"
        assert "tensor." in ir_str, "Should use tensor dialect"
        assert "f32" in ir_str, "Should have float32 type"

    def test_mlir_module_validity(self):
        """Test that generated MLIR is a valid module."""

        @helion.kernel(static_shapes=True)
        def simple_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(simple_kernel, [x])

        # Module should be properly formed
        assert module is not None
        assert hasattr(module, "body"), "Module should have body"
        ir_str = str(module)
        assert "module" in ir_str.lower(), "Should contain module declaration"

    def test_elementwise_ops_linalg_generic(self):
        """Test that elementwise ops use linalg.generic."""

        @helion.kernel(static_shapes=True)
        def elementwise_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(elementwise_kernel, [x, y])
        ir_str = str(module)

        # Elementwise operations should use linalg.generic
        assert "linalg.generic" in ir_str or "arith." in ir_str, (
            "Should use linalg.generic or arith ops for elementwise"
        )

    def test_different_dtypes_in_module(self):
        """Test that multiple dtypes are properly handled."""

        @helion.kernel(static_shapes=True)
        def mixed_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            # Work with the input dtype
            out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")

        # Test with float64
        x = torch.randn(32, 64, dtype=torch.float64, device=device)
        module = generate_mlir(mixed_dtype_kernel, [x])
        ir_str = str(module)
        assert "f64" in ir_str, "Should preserve float64 dtype"

    def test_non_square_matrices(self):
        """Test MLIR generation with non-square matrix dimensions."""

        @helion.kernel(static_shapes=True)
        def tall_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape  # Tall matrix
            k2, n = y.shape  # Wide matrix
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        device = torch.device("cpu")
        x = torch.randn(256, 128, dtype=torch.float32, device=device)  # Tall
        y = torch.randn(128, 64, dtype=torch.float32, device=device)  # Wide

        module = generate_mlir(tall_matrix, [x, y])
        ir_str = str(module)

        # Should handle various dimensions
        assert "256" in ir_str, "Should reflect tall dimension"
        assert "scf.forall" in ir_str, "Should have parallel iteration"

    def test_fused_operations(self):
        """Test MLIR generation for fused operations."""

        @helion.kernel(static_shapes=True)
        def fused_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                # Apply relu to the result
                out[tile_m, tile_n] = torch.relu(acc)
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(128, 96, dtype=torch.float32, device=device)

        module = generate_mlir(fused_kernel, [x, y])
        ir_str = str(module)

        # Should have both matmul and relu operations
        assert "linalg.matmul" in ir_str or "linalg.generic" in ir_str
        # Verify function structure is preserved
        assert "func.return" in ir_str or "return" in ir_str

    def test_multiple_tiles(self):
        """Test MLIR with multiple tiling levels."""

        @helion.kernel(static_shapes=True)
        def multi_tile_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                # Could further subdivide within tiles
                result = torch.relu(x[tile_m, tile_n])
                out[tile_m, tile_n] = result
            return out

        device = torch.device("cpu")
        x = torch.randn(256, 256, dtype=torch.float32, device=device)

        module = generate_mlir(multi_tile_kernel, [x])
        ir_str = str(module)

        # Should have structured tiling
        assert "scf.forall" in ir_str
        assert "256" in ir_str

    def test_mlir_ir_printable(self):
        """Test that MLIR can be printed without errors."""

        @helion.kernel(static_shapes=True)
        def printable_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)
        y = torch.randn(32, 64, dtype=torch.float32, device=device)

        module = generate_mlir(printable_kernel, [x, y])
        ir_str = str(module)

        # Should be printable and non-empty
        assert len(ir_str) > 0
        assert "func.func" in ir_str

    def test_module_has_proper_function_signature(self):
        """Test that generated module has proper function signature."""

        @helion.kernel(static_shapes=True)
        def signature_test(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(signature_test, [x, y])
        ir_str = str(module)

        # Function should have proper signature
        assert "func.func" in ir_str
        assert "tensor<" in ir_str, "Should have tensor types"
        assert "x32" in ir_str or "f32" in ir_str, "Should have float type"
        assert "func.return" in ir_str or "return" in ir_str, (
            "Should have return statement"
        )


class TestScalarBlockIndices:
    """`hl.grid` indices and tile position ops (authoring gaps 1 and 2)."""

    def test_grid_batched_matmul_rank_reduces(self):
        """`hl.grid` index selects a 2-D view and lowers to linalg.matmul."""

        @helion.kernel(static_shapes=True)
        def grid_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            out = torch.zeros((nb, m, b.shape[2]), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                out[i, :, :] = torch.matmul(a[i, :, :], b[i, :, :])
            return out

        a = torch.randn(4, 32, 32, dtype=torch.float32)
        b = torch.randn(4, 32, 32, dtype=torch.float32)

        module = generate_mlir(grid_bmm, [a, b])
        module.operation.verify()
        ir_str = str(module)

        assert "scf.forall" in ir_str
        # The scalar grid dimension is dropped from the loaded tile via an
        # unambiguous rank-preserving extract followed by an explicit collapse
        # (avoids MLIR's ambiguous same-size rank-reduction inference).
        assert "tensor<4x32x32xf32> to tensor<1x32x32xf32>" in ir_str
        assert "tensor<1x32x32xf32> into tensor<32x32xf32>" in ir_str
        assert "linalg.matmul" in ir_str
        # The store writes the iteration's owned [1, 32, 32] region back.
        assert "tensor<1x32x32xf32> into tensor<4x32x32xf32>" in ir_str

    def test_scalar_grid_tile_slice_preserves_rank_metadata(self):
        """Scalar grid indices with a tiled slice must keep explicit size-1 offsets."""

        @helion.kernel(static_shapes=True)
        def copy_slices(a: torch.Tensor) -> torch.Tensor:
            nb, m, kk = a.shape
            out = torch.empty_like(a)
            for i in hl.grid(nb):
                for tk in hl.tile(kk):
                    out[i, :, tk] = a[i, :, tk]
            return out

        a = torch.randn(2, 64, 128, dtype=torch.float32)
        module = generate_mlir(copy_slices, [a], config=helion.Config(block_sizes=[32]))
        module.operation.verify()
        ir_str = str(module)

        assert "tensor.extract_slice" in ir_str
        assert "tensor.parallel_insert_slice" in ir_str
        assert "1x" in ir_str or "tensor<1" in ir_str
        assert "tensor<64x32xf32>" in ir_str

    def test_nested_grid_indices_lower_each_dimension(self):
        """Nested grid indices select the corresponding two outer dimensions."""

        @helion.kernel(static_shapes=True)
        def nested_grid_copy(a: torch.Tensor) -> torch.Tensor:
            nb, nc, m = a.shape
            out = torch.zeros_like(a)
            for i in hl.grid(nb):
                for j in hl.grid(nc):
                    out[i, j, :] = a[i, j, :]
            return out

        module = generate_mlir(
            nested_grid_copy,
            [torch.randn(2, 3, 8)],
            config=helion.Config(block_sizes=[4, 4, 8]),
        )
        module.operation.verify()
        ir_str = str(module)

        assert "scf.forall" in ir_str
        assert "tensor.extract_slice" in ir_str
        assert "tensor.parallel_insert_slice" in ir_str

    def test_grid_index_with_tiled_reduction_lowers(self):
        """A scalar grid index remains valid through a tiled K reduction."""

        @helion.kernel(static_shapes=True)
        def grid_tiled_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            out = torch.zeros((nb, m, b.shape[2]), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                acc = hl.zeros([m, b.shape[2]], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[i, :, tk], b[i, tk, :])
                out[i, :, :] = acc
            return out

        module = generate_mlir(
            grid_tiled_k,
            [torch.randn(2, 8, 16), torch.randn(2, 16, 8)],
            config=helion.Config(block_sizes=[4, 4, 8]),
        )
        module.operation.verify()
        ir_str = str(module)

        assert "scf.forall" in ir_str
        assert "scf.for" in ir_str
        assert "linalg.matmul" in ir_str

    def test_static_view_and_reshape_change_shape(self):
        """Static shape-changing view and reshape lower through tensor.reshape."""

        @helion.kernel(static_shapes=True)
        def reshape_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((16, 8), dtype=x.dtype, device=x.device)
            reshaped = x.view(16, 8)
            for _ in hl.tile(16):
                out[:, :] = reshaped
            return out

        module = generate_mlir(
            reshape_kernel,
            [torch.randn(8, 16)],
            config=helion.Config(block_sizes=[8, 16]),
        )
        module.operation.verify()
        ir_str = str(module)
        assert "tensor<16x8xf32>" in ir_str

    def test_tile_begin_uses_induction_variable(self):
        """`tile.begin` lowers to the enclosing loop induction variable."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.begin
            return out

        module = generate_mlir(kernel, [torch.randn(40, 32)])
        module.operation.verify()
        ir_str = str(module)

        assert "scf.forall" in ir_str
        assert "arith.index_cast" in ir_str
        assert "linalg.elementwise" in ir_str

    def test_tile_end_clamps_only_with_known_bound(self):
        """`tile.end` clamps when the extent is not a multiple of the block."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def ragged(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.end
            return out

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def exact(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.end
            return out

        bs = helion.Config(block_sizes=[16])
        ragged_ir = str(generate_mlir(ragged, [torch.randn(40, 32)], config=bs))
        exact_ir = str(generate_mlir(exact, [torch.randn(64, 32)], config=bs))

        assert "arith.addi" in ragged_ir
        assert "affine.min" in ragged_ir, "ragged extent must clamp the last tile"
        assert "arith.addi" in exact_ir
        assert "affine.min" not in exact_ir, "evenly divided extent needs no clamp"

    def test_tile_id_divides_by_block_size(self):
        """`tile.id` of an outer tile is the normalized forall IV (offset / block_size)."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.id
            return out

        module = generate_mlir(
            kernel, [torch.randn(40, 32)], config=helion.Config(block_sizes=[16])
        )
        module.operation.verify()
        assert "arith.divui" not in str(module)
        assert re.search(r"scf\.forall \(%\w+\) in \(3\)", str(module))

    def test_tile_count_is_constant(self):
        """`tile.count` folds to a compile-time cdiv constant."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.count
            return out

        module = generate_mlir(
            kernel, [torch.randn(40, 32)], config=helion.Config(block_sizes=[16])
        )
        module.operation.verify()
        ir_str = str(module)
        # cdiv(40, 16) == 3
        assert "arith.constant 3 : index" in ir_str

    def test_tile_block_size_is_constant(self):
        """`tile.block_size` lowers without going through an ATen helper."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.block_size
            return out

        module = generate_mlir(kernel, [torch.randn(40, 32)])
        module.operation.verify()
        ir_str = str(module)
        assert "linalg.fill" in ir_str
        assert "linalg.elementwise" in ir_str


class TestContractionExpressiveness:
    """Mixed precision and transposed contractions (authoring gaps 6 and 7)."""

    def test_bf16_operands_accumulate_into_f32(self):
        """bf16 x bf16 -> f32 uses linalg.matmul directly, not an ATen helper."""

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

        x = torch.randn(128, 128, dtype=torch.bfloat16)
        module = generate_mlir(mm, [x, x])
        module.operation.verify()
        ir_str = str(module)

        assert "linalg.matmul" in ir_str
        assert "bf16" in ir_str and "f32" in ir_str
        # Narrow operands accumulate into the wider tile without a helper call.
        assert re.search(
            r"linalg\.matmul ins\(%extracted_slice\w*, %extracted_slice\w* : "
            r"tensor<16x16xbf16>, tensor<16x16xbf16>\) outs\(",
            ir_str,
        )
        assert "-> tensor<16x16xf32>" in ir_str

    def test_f16_operands_accumulate_into_f32(self):
        """f16 x f16 -> f32 also takes the direct contraction path."""

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

        x = torch.randn(128, 128, dtype=torch.float16)
        module = generate_mlir(mm, [x, x])
        module.operation.verify()
        ir_str = str(module)

        assert "linalg.matmul" in ir_str
        assert "f16" in ir_str

    def test_transposed_rhs_emits_contract_with_indexing_maps(self):
        """A transposed operand folds into linalg.contract indexing maps."""

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

        x = torch.randn(128, 128, dtype=torch.float32)
        module = generate_mlir(mm_transposed_b, [x, x])
        module.operation.verify()
        ir_str = str(module)

        assert "linalg.contract" in ir_str
        assert "indexing_maps" in ir_str
        # RHS is indexed (n, k) rather than (k, n).
        assert "affine_map<(d0, d1, d2) -> (d1, d2)>" in ir_str

    def test_standalone_transpose_emits_linalg_transpose(self):
        """A permute that feeds no contraction lowers to linalg.transpose."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((n, m), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tn, tm] = x[tm, tn].permute(1, 0)
            return out

        module = generate_mlir(kernel, [torch.randn(64, 64)])
        module.operation.verify()
        ir_str = str(module)

        assert "linalg.transpose" in ir_str
        assert "permutation = [1, 0]" in ir_str


class TestDtypeConversionEpilogue:
    """Numerical `.to` casts in kernel epilogues (authoring gap 5)."""

    def test_explicit_to_narrowing_cast(self):
        """`acc.to(torch.bfloat16)` emits a truncating cast, not a func.call."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = (x[tm, tn] * 2.0).to(torch.bfloat16)
            return out

        module = generate_mlir(kernel, [torch.randn(64, 64)])
        module.operation.verify()
        ir_str = str(module)

        assert "arith.truncf" in ir_str
        assert "bf16" in ir_str

    def test_explicit_to_widening_cast(self):
        """`.to(torch.float32)` on a narrow tile emits an extending cast."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = x[tm, tn].to(torch.float32) * 2.0
            return out

        module = generate_mlir(kernel, [torch.randn(64, 64, dtype=torch.bfloat16)])
        module.operation.verify()
        ir_str = str(module)

        assert "arith.extf" in ir_str
        assert "f32" in ir_str

    def test_implicit_store_cast(self):
        """Storing an f32 tile into a bf16 output casts on the way out."""

        @helion.kernel(static_shapes=True)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        module = generate_mlir(kernel, [torch.randn(64, 64)])
        module.operation.verify()
        ir_str = str(module)

        assert "arith.truncf" in ir_str
        assert "into tensor<64x64xbf16>" in ir_str


class TestNestedBlockSizeResolution:
    """Nested tile block-id resolution (authoring gap 4)."""

    @pytest.mark.parametrize(
        "block_sizes",
        [[16, 32, 64], [32, 32, 32], [32, 64, 32], [16, 16, 16]],
    )
    def test_nested_tiles_offset_every_dimension(self, block_sizes):
        """Repeated block sizes must not collapse a tile offset to zero."""

        @helion.kernel(
            static_shapes=True, config=helion.Config(block_sizes=block_sizes)
        )
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

        x = torch.randn(128, 128, dtype=torch.float32)
        module = generate_mlir(
            mm, [x, x], config=helion.Config(block_sizes=block_sizes)
        )
        module.operation.verify()
        ir_str = str(module)
        loads = [
            line
            for line in ir_str.splitlines()
            if "tensor.extract_slice" in line and ("%arg0[" in line or "%arg1[" in line)
        ]
        assert loads, "expected tiled loads of both operands"
        # Every tile offset must be an induction variable, never a constant 0.
        offsets = [line.split("[")[1].split("]")[0] for line in loads]
        assert all("%c" not in offset for offset in offsets), (
            f"tile offset collapsed to a constant: {offsets}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
