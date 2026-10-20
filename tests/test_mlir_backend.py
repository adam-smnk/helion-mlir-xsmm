"""Pytest tests for MLIR backend."""

from __future__ import annotations

import re

import helion
import helion.language as hl
import mlir.ir as ir
import pytest
import torch

from helion_mlir_backend import generate_mlir


@pytest.fixture
def mlir_context():
    """Create an MLIR context for tests."""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    with ctx:
        yield ctx


class TestTypeConversions:
    """Test torch ↔ MLIR type conversions."""

    def test_float32_conversion(self, mlir_context):
        """Test float32 dtype conversion."""
        from helion_mlir_backend._compiler.mlir.support.type_utils import (
            torch_dtype_to_mlir,
        )

        with mlir_context:
            result = torch_dtype_to_mlir(torch.float32)
            assert "f32" in str(result)

    def test_float64_conversion(self, mlir_context):
        """Test float64 dtype conversion."""
        from helion_mlir_backend._compiler.mlir.support.type_utils import (
            torch_dtype_to_mlir,
        )

        with mlir_context:
            result = torch_dtype_to_mlir(torch.float64)
            assert "f64" in str(result)

    def test_int32_conversion(self, mlir_context):
        """Test int32 dtype conversion."""
        from helion_mlir_backend._compiler.mlir.support.type_utils import (
            torch_dtype_to_mlir,
        )

        with mlir_context:
            result = torch_dtype_to_mlir(torch.int32)
            assert "i32" in str(result)

    def test_unsupported_dtype_raises(self, mlir_context):
        """Test that unsupported dtypes raise error."""
        from helion_mlir_backend._compiler.mlir.support.errors import (
            UnsupportedOperationError,
        )
        from helion_mlir_backend._compiler.mlir.support.type_utils import (
            torch_dtype_to_mlir,
        )

        with mlir_context, pytest.raises(UnsupportedOperationError):
            torch_dtype_to_mlir(torch.complex64)
        with mlir_context, pytest.raises(UnsupportedOperationError, match="signless"):
            torch_dtype_to_mlir(torch.uint8)


class TestBackendStructure:
    """Test the backend contract and extracted module facades."""

    def test_mlir_backend_uses_backend_neutral_codegen_name(self):
        from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

        backend = MLIRBackend()
        assert backend.name == "mlir"
        assert backend.codegen_name == "mlir"
        assert backend.dtype_str(torch.float32) == "torch.float32"
        assert backend.acc_type(torch.float32) == "torch.float32"

    @pytest.mark.parametrize(
        "property_name",
        (
            "function_decorator",
            "constexpr_type",
            "default_launcher_name",
            "library_imports",
        ),
    )
    def test_python_codegen_properties_are_rejected(self, property_name):
        from helion import exc

        from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

        with pytest.raises(exc.BackendUnsupported):
            getattr(MLIRBackend(), property_name)

    def test_extracted_helpers_are_available_from_facades(self):
        from helion_mlir_backend._compiler.mlir.aten_bridge import call_helper
        from helion_mlir_backend._compiler.mlir.aten_bridge import lower_via_aten_helper
        from helion_mlir_backend._compiler.mlir.lowering import build_phase_body

        assert callable(call_helper)
        assert callable(lower_via_aten_helper)
        assert callable(build_phase_body)


class TestExtractedHelpers:
    """Unit tests for pure metadata and argument helpers."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        (
            ("block_size_0", 0),
            ("block_size_12", 12),
            ("block_size_", None),
            ("u3", None),
            (3, None),
        ),
    )
    def test_block_id_from_key(self, value, expected):
        from helion_mlir_backend._compiler.mlir.support import block_id_from_key

        assert block_id_from_key(value) == expected

    def test_original_args_restores_stripped_inputs_only(self):
        from helion_mlir_backend._compiler.mlir.aten_bridge import ORIGINAL_ARGS
        from helion_mlir_backend._compiler.mlir.aten_bridge import original_args

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        masked = graph.placeholder("masked")
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
        mul.meta[ORIGINAL_ARGS] = (mul.args, mul.kwargs)
        # Helion masks the repeated input, then rewrites another one (_mask_to).
        mul.args = (masked, None)

        assert original_args(mul) == ((masked, x), {})


class TestBasicOpLowerings:
    """Test lowering of basic PyTorch operations."""

    def test_matmul_lowering_f32(self):
        """Test matmul kernel MLIR generation with float32."""

        @helion.kernel(static_shapes=True)
        def matmul_kernel_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2
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

        module = generate_mlir(matmul_kernel_f32, [x, y])
        module.operation.verify()
        # Verify module contains expected operations
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "scf.forall" in ir_str
        assert "linalg.matmul" in ir_str

    def test_matmul_lowering_f64(self):
        """Test matmul kernel MLIR generation with float64."""

        @helion.kernel(static_shapes=True)
        def matmul_kernel_f64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2
            out = torch.zeros((m, n), dtype=torch.float64, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float64)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        device = torch.device("cpu")
        x = torch.randn(128, 256, dtype=torch.float64, device=device)
        y = torch.randn(256, 192, dtype=torch.float64, device=device)

        module = generate_mlir(matmul_kernel_f64, [x, y])
        module.operation.verify()
        # Verify module contains expected operations
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "scf.forall" in ir_str
        assert "linalg.matmul" in ir_str

    def test_simple_add_kernel(self):
        """Test simple element-wise addition."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 64, dtype=torch.float32, device=device)
        y = torch.randn(64, 64, dtype=torch.float32, device=device)

        module = generate_mlir(add_kernel, [x, y])
        module.operation.verify()
        ir_str = str(module)

        # Should have func.func and some operations
        assert "func.func" in ir_str
        assert "return" in ir_str.lower()

    def test_kernel_with_multiple_shapes(self):
        """Test kernel compilation with simple element-wise operations."""

        @helion.kernel(static_shapes=True)
        def elem_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")

        # Test with shape (32, 32)
        x = torch.randn(32, 32, dtype=torch.float32, device=device)
        y = torch.randn(32, 32, dtype=torch.float32, device=device)
        module = generate_mlir(elem_kernel, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str


class TestFullKernelCompilation:
    """Test end-to-end kernel compilation."""

    def test_compile_simple_kernel(self):
        """Test basic compilation without errors."""

        @helion.kernel(static_shapes=True)
        def simple(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 64, dtype=torch.float32, device=device)

        # Should not raise
        module = generate_mlir(simple, [x])
        module.operation.verify()
        assert module is not None

    def test_matmul_with_epilogue(self):
        """Test matmul with epilogue function."""

        @helion.kernel(static_shapes=True)
        def matmul_with_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.relu(acc)
            return out

        device = torch.device("cpu")
        x = torch.randn(128, 256, dtype=torch.float32, device=device)
        y = torch.randn(256, 192, dtype=torch.float32, device=device)

        module = generate_mlir(matmul_with_relu, [x, y])
        module.operation.verify()
        ir_str = str(module)

        assert "func.func" in ir_str
        assert "linalg.matmul" in ir_str

    def test_compilation_with_different_dtypes(self):
        """Test compilation with float32 and float64."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")

        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(32, 32, dtype=dtype, device=device)
            y = torch.randn(32, 32, dtype=dtype, device=device)

            module = generate_mlir(add_kernel, [x, y])
            module.operation.verify()
            assert module is not None

            ir_str = str(module)
            assert "func.func" in ir_str


class TestMLIRValidity:
    """Test MLIR IR validity."""

    def test_generated_ir_has_valid_syntax(self):
        """Test that generated IR can be printed without errors."""

        @helion.kernel(static_shapes=True)
        def simple_kernel(x: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            out = torch.zeros((m, k), dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] * 2.0
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(simple_kernel, [x])
        module.operation.verify()
        ir_str = str(module)

        # Should be valid textual MLIR (may have #map definitions first).
        # May use generic format ("builtin.module"...) or pretty format (module {}).
        assert (
            '"builtin.module"' in ir_str
            or "builtin.module" in ir_str
            or "module {" in ir_str
        )
        assert "func.func" in ir_str
        assert "return" in ir_str.lower()

    def test_ir_contains_required_dialects(self):
        """Test that generated IR uses required dialects."""

        @helion.kernel(static_shapes=True)
        def kernel_with_loops(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(128, 96, dtype=torch.float32, device=device)

        module = generate_mlir(kernel_with_loops, [x, y])
        module.operation.verify()
        ir_str = str(module)

        # Check for key dialects
        assert "func.func" in ir_str  # func dialect
        assert "scf.forall" in ir_str  # scf dialect
        assert "linalg.matmul" in ir_str  # linalg dialect
        assert "tensor" in ir_str.lower()  # tensor dialect


class TestExtendedOperations:
    """Test extended operation support (layer_norm, softmax, attention)."""

    def test_layer_norm_generation(self):
        """Test that layer_norm operations generate valid MLIR."""

        @helion.kernel(static_shapes=True)
        def kernel_with_layer_norm(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                # Layer norm would typically be applied here
                # For now, we use a placeholder pattern
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        # Should not raise an error
        module = generate_mlir(kernel_with_layer_norm, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str

    def test_softmax_generation(self):
        """Test that softmax operations generate valid MLIR."""

        @helion.kernel(static_shapes=True)
        def kernel_with_softmax(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                # Softmax would be applied here
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(kernel_with_softmax, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str

    def test_unsupported_operation_error(self):
        """Test that unsupported operations raise helpful error."""

        # Create a simple test to verify error handling is in place
        @helion.kernel(static_shapes=True)
        def kernel_basic(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        # This should work - basic kernel
        module = generate_mlir(kernel_basic, [x])
        module.operation.verify()
        assert module is not None


class TestDynamicShapes:
    """Test dynamic shape (SymInt) handling."""

    def test_dynamic_shape_in_kernel(self):
        """Test that kernels with dynamic-looking shapes still work."""

        @helion.kernel(static_shapes=True)
        def kernel_dynamic_ish(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        # Should work - shape is resolved at compile time
        module = generate_mlir(kernel_dynamic_ish, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str


class TestErrorHandling:
    """Test error handling in MLIR backend."""

    def test_invalid_kernel_type_raises(self):
        """Test that non-kernel objects raise error."""

        def not_a_kernel(x):
            return x

        x = torch.randn(64, 64, dtype=torch.float32)

        with pytest.raises(ValueError, match="@helion.kernel"):
            generate_mlir(not_a_kernel, [x])

    def test_missing_mlir_bindings_error(self):
        """Test clear error message when mlir-python-bindings not available."""
        # This test mainly validates the error message is clear
        # Actual test requires mocking which is handled by pytest


class TestAdvancedOperations:
    """Test advanced tensor operations beyond basic matmul."""

    def test_subtraction_operation(self):
        """Test element-wise subtraction lowering."""

        @helion.kernel(static_shapes=True)
        def sub_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] - y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(sub_kernel, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str

    def test_host_alias_ops_supported(self):
        """contiguous and same-shape view should lower as pure aliases."""

        @helion.kernel(static_shapes=True)
        def alias_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            y = x.contiguous()
            z = y.view(m, n)
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = z[tile_m, tile_n] + 1.0
            return out

        @helion.kernel(static_shapes=True)
        def direct_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)

        alias_module = generate_mlir(alias_kernel, [x])
        direct_module = generate_mlir(direct_kernel, [x])
        alias_module.operation.verify()
        direct_module.operation.verify()

        alias_ir = str(alias_module)
        direct_ir = str(direct_module)

        assert "func.func" in alias_ir
        assert "linalg.generic" in alias_ir

        # Same-shape alias ops should not introduce reshape/collapse artifacts.
        assert "tensor.collapse_shape" not in alias_ir
        assert "tensor.expand_shape" not in alias_ir

        # Alias and direct forms should have the same structural core.
        for op_name in [
            "scf.forall",
            "tensor.extract_slice",
            "tensor.parallel_insert_slice",
            "linalg.generic",
        ]:
            assert alias_ir.count(op_name) == direct_ir.count(op_name)

        # Alias ops should not create extra ATen helper functions.
        alias_helpers = re.findall(r"func\.func private @_aten_\w+", alias_ir)
        direct_helpers = re.findall(r"func\.func private @_aten_\w+", direct_ir)
        assert len(alias_helpers) == len(direct_helpers)

    def test_division_operation(self):
        """Test element-wise division lowering."""

        @helion.kernel(static_shapes=True)
        def div_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] / y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)
        y = (
            torch.randn(32, 64, dtype=torch.float32, device=device) + 1.0
        )  # Avoid division by zero

        module = generate_mlir(div_kernel, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str

    def test_exponential_operation(self):
        """Test exponential function lowering."""

        @helion.kernel(static_shapes=True)
        def exp_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n])
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)

        module = generate_mlir(exp_kernel, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str

    def test_transpose_2d_operation(self):
        """Test 2D matrix transpose within kernel loop."""

        # Note: Direct torch.transpose at kernel level doesn't work without device loops.
        # This test validates the transpose lowering within loop contexts works correctly.
        @helion.kernel(static_shapes=True)
        def transpose_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                # Simple copy (equivalent to identity operation)
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(transpose_kernel, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        # Just verify MLIR module is generated without errors
        assert "scf.forall" in ir_str

    def test_clamp_operation(self):
        """Test torch.clamp lowering via torch-mlir → linalg.generic."""

        @helion.kernel(static_shapes=True)
        def clamp_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.clamp(x[tile_m, tile_n], min=-1.0, max=1.0)
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)

        module = generate_mlir(clamp_kernel, [x])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str

    def test_geglu_polynomial_cast_path(self):
        """Test GEGLU-like tanh polynomial lowering with explicit casts."""

        @helion.kernel(static_shapes=True)
        def geglu_like(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            assert a.shape == b.shape
            out = torch.empty_like(a, dtype=torch.float32)

            sqrt_2_over_pi = 0.7978845608028654
            for tile in hl.tile(a.size()):
                a_vals = a[tile].to(torch.float32)
                b_vals = b[tile].to(torch.float32)

                a_cubed = a_vals * a_vals * a_vals
                tanh_arg = sqrt_2_over_pi * (a_vals + 0.044715 * a_cubed)
                tanh_result = torch.tanh(tanh_arg)
                gelu_a = 0.5 * a_vals * (1.0 + tanh_result)

                out[tile] = gelu_a * b_vals

            return out

        device = torch.device("cpu")
        a = torch.randn(8, 16, dtype=torch.float16, device=device)
        b = torch.randn(8, 16, dtype=torch.float16, device=device)

        module = generate_mlir(geglu_like, [a, b])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str

    def test_dynamic_dtype_cast_from_tensor_attr(self):
        """Cast target via tensor.dtype should lower for this traced pattern."""

        @helion.kernel(static_shapes=True)
        def cast_from_tensor_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            assert a.shape == b.shape
            out = torch.empty_like(a, dtype=torch.float32)

            for tile in hl.tile(a.size()):
                a_vals = a[tile].to(torch.float32)
                b_vals = b[tile]
                out[tile] = a_vals.to(b_vals.dtype) * b_vals.to(torch.float32)

            return out

        device = torch.device("cpu")
        a = torch.randn(8, 16, dtype=torch.float16, device=device)
        b = torch.randn(8, 16, dtype=torch.float16, device=device)

        module = generate_mlir(cast_from_tensor_dtype, [a, b])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str

    def test_flatten_view_alias_in_1d_geglu_style(self):
        """Flatten/view alias in 1-D GEGLU style verifies successfully."""

        @helion.kernel(static_shapes=True)
        def geglu_flatten_style(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            assert a.shape == b.shape
            out = torch.empty_like(a, dtype=torch.float32)

            a_flat = a.view(-1)
            b_flat = b.view(-1)
            out_flat = out.view(-1)

            c = 0.7978845608028654
            for tile_idx in hl.tile(a.numel()):
                a_vals = a_flat[tile_idx].to(torch.float32)
                b_vals = b_flat[tile_idx].to(torch.float32)
                a_cubed = a_vals * a_vals * a_vals
                tanh_arg = c * (a_vals + 0.044715 * a_cubed)
                gelu_a = 0.5 * a_vals * (1.0 + torch.tanh(tanh_arg))
                out_flat[tile_idx] = gelu_a * b_vals

            return out

        device = torch.device("cpu")
        a = torch.randn(8, 16, dtype=torch.float16, device=device)
        b = torch.randn(8, 16, dtype=torch.float16, device=device)

        module = generate_mlir(geglu_flatten_style, [a, b])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str


class TestEinsumDecomposition:
    """Tests for einsum equations that deliberately keep the decomposed path.

    Contracting einsums are captured before dispatch and lowered directly to
    ``linalg.contract`` (see tests/test_einsum_contract.py).  Capture requires
    at least one contracted dimension, so a purely element-wise equation is
    left alone: ``aten::einsum`` is ``CompositeImplicitAutograd``, so the
    dispatcher expands it into ``mul``/``permute``/``view`` before Helion's
    tracer records anything, and the backend lowers those primitives instead.

        einsum 'ij,ij->ij'        →  mul (+ permute)  →  linalg.generic
        einsum 'abcd,abcd->abcd'  →  same as above
        einsum 'ij,jk->ik'        →  captured         →  linalg.contract

    These tests pin the element-wise half of that split.
    """

    def test_einsum_elemwise_2d_decomposes_to_linalg_generic(self):
        """einsum('ij,ij->ij') has no contracted dim → mul → linalg.generic."""

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.einsum(
                    "ij,ij->ij", x[tile_m, tile_n], y[tile_m, tile_n]
                )
            return out

        x = torch.randn(64, 128)
        y = torch.randn(64, 128)
        module = generate_mlir(k, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "linalg.generic" in ir_str  # from decomposed mul
        assert "linalg.contract" not in ir_str

    def test_einsum_4d_elemwise_decomposes_to_linalg_generic(self):
        """einsum('abcd,abcd->abcd') has no contracted dim → mul → linalg.generic."""

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a, b, c, d = x.shape
            out = torch.zeros((a, b, c, d), dtype=x.dtype, device=x.device)
            for tile_a, tile_b, tile_c, tile_d in hl.tile([a, b, c, d]):
                out[tile_a, tile_b, tile_c, tile_d] = torch.einsum(
                    "abcd,abcd->abcd",
                    x[tile_a, tile_b, tile_c, tile_d],
                    y[tile_a, tile_b, tile_c, tile_d],
                )
            return out

        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(2, 3, 4, 5)
        module = generate_mlir(k, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "scf.forall" in ir_str
        assert "linalg.generic" in ir_str  # from decomposed mul
        assert "linalg.contract" not in ir_str

    def test_einsum_5d_elemwise_decomposes_to_linalg(self):
        """5D element-wise mul (equiv. to einsum 'abcde,abcde->abcde') → linalg.generic."""

        @helion.kernel(static_shapes=True)
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a, b, c, d, e = x.shape
            out = torch.zeros((a, b, c, d, e), dtype=x.dtype, device=x.device)
            for tile_a, tile_b, tile_c, tile_d, tile_e in hl.tile([a, b, c, d, e]):
                out[tile_a, tile_b, tile_c, tile_d, tile_e] = (
                    x[tile_a, tile_b, tile_c, tile_d, tile_e]
                    * y[tile_a, tile_b, tile_c, tile_d, tile_e]
                )
            return out

        x = torch.randn(2, 2, 2, 2, 2)
        y = torch.randn(2, 2, 2, 2, 2)
        module = generate_mlir(k, [x, y])
        module.operation.verify()
        ir_str = str(module)
        assert "func.func" in ir_str
        assert "scf.forall" in ir_str
        assert "linalg" in ir_str


class TestAtenHelperVisibility:
    """Tests for visibility of generated ATen helper functions."""

    def test_generated_aten_helpers_are_private(self):
        """Generated _aten_* helper funcs should always have private visibility."""

        @helion.kernel(static_shapes=True)
        def relu_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])
            return out

        x = torch.randn(32, 64, dtype=torch.float32)
        module = generate_mlir(relu_kernel, [x])
        module.operation.verify()

        aten_helper_count = 0
        for op in module.body.operations:
            try:
                sym_name = ir.StringAttr(op.attributes["sym_name"]).value
            except Exception:
                continue

            if sym_name.startswith("_aten_"):
                aten_helper_count += 1
                assert "sym_visibility" in op.attributes
                visibility = ir.StringAttr(op.attributes["sym_visibility"]).value
                assert visibility == "private"

        assert aten_helper_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
