"""``inline_mlir``: user MLIR functions called from device loops."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import tensor
import mlir.ir as ir
import pytest
import torch

from tests.harness import opt_pipeline
from tests.harness import run_direct

from helion_mlir_backend import InlineMLIRHint
from helion_mlir_backend import inline_mlir
from helion_mlir_backend.language import _inline_mlir


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


_ID2 = "affine_map<(i, j) -> (i, j)>"

SOFTPLUS = f"""
func.func @softplus(%x: tensor<?x?xf32>, %beta: f32) -> tensor<?x?xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %r = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {{
  ^bb0(%a: f32, %o: f32):
    %one = arith.constant 1.0 : f32
    %ba = arith.mulf %a, %beta : f32
    %e = math.exp %ba : f32
    %s = arith.addf %e, %one : f32
    %l = math.log %s : f32
    %v = arith.divf %l, %beta : f32
    linalg.yield %v : f32
  }} -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}}
"""

# (x * scale + global row index, -x): scalars of three kinds and two results.
ROWS = f"""
func.func @rows(%x: tensor<?x?xf32>, %scale: f32, %row0: index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %r = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {{
  ^bb0(%a: f32, %o: f32):
    %i = linalg.index 0 : index
    %g = arith.addi %i, %row0 : index
    %gi = arith.index_cast %g : index to i64
    %gf = arith.sitofp %gi : i64 to f32
    %m = arith.mulf %a, %scale : f32
    %v = arith.addf %m, %gf : f32
    linalg.yield %v : f32
  }} -> tensor<?x?xf32>
  %n = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {{
  ^bb0(%a: f32, %o: f32):
    %v = arith.negf %a : f32
    linalg.yield %v : f32
  }} -> tensor<?x?xf32>
  return %r, %n : tensor<?x?xf32>, tensor<?x?xf32>
}}
"""

# x + the column sums of a host tensor slice (read whole inside the snippet).
ADD_COLUMN_SUMS = """
func.func @add_column_sums(%table: tensor<?x?xf32>, %x: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty(%d1) : tensor<?xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e : tensor<?xf32>) -> tensor<?xf32>
  %sums = linalg.reduce ins(%table : tensor<?x?xf32>) outs(%acc : tensor<?xf32>) dimensions = [0]
    (%a: f32, %b: f32) {
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    }
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %sums : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%v: f32, %s: f32, %o: f32):
    %w = arith.addf %v, %s : f32
    linalg.yield %w : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
"""

_ANY = "memref<?x?xf32, strided<[?, ?], offset: ?>>"

# 3 * x through memrefs: a read-only view of x, results in a new buffer.
SCALE_ALLOC = f"""
func.func @scale(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %three = arith.constant 3.0 : f32
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %xb = bufferization.to_buffer %x read_only : tensor<?x?xf32> to {_ANY}
  %ob = memref.alloc(%d0, %d1) : memref<?x?xf32>
  scf.for %i = %c0 to %d0 step %c1 {{
    scf.for %j = %c0 to %d1 step %c1 {{
      %v = memref.load %xb[%i, %j] : {_ANY}
      %w = arith.mulf %v, %three : f32
      memref.store %w, %ob[%i, %j] : memref<?x?xf32>
    }}
  }}
  %r = bufferization.to_tensor %ob restrict writable : memref<?x?xf32> to tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}}
"""

# 3 * x written into a destination argument, returned as the result.
SCALE_INTO = f"""
func.func @scale_into(%x: tensor<?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %three = arith.constant 3.0 : f32
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %xb = bufferization.to_buffer %x read_only : tensor<?x?xf32> to {_ANY}
  %ob = bufferization.to_buffer %dest : tensor<?x?xf32> to {_ANY}
  scf.for %i = %c0 to %d0 step %c1 {{
    scf.for %j = %c0 to %d1 step %c1 {{
      %v = memref.load %xb[%i, %j] : {_ANY}
      %w = arith.mulf %v, %three : f32
      memref.store %w, %ob[%i, %j] : {_ANY}
    }}
  }}
  %r = bufferization.to_tensor %ob restrict writable : {_ANY} to tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}}
"""

# acc += a @ b in place on the accumulator's buffer.
MATMUL_ACC = f"""
func.func @matmul_acc(%acc: tensor<?x?xf32>, %a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {{
  %accb = bufferization.to_buffer %acc : tensor<?x?xf32> to {_ANY}
  %ab = bufferization.to_buffer %a read_only : tensor<?x?xf32> to {_ANY}
  %bb = bufferization.to_buffer %b read_only : tensor<?x?xf32> to {_ANY}
  linalg.matmul ins(%ab, %bb : {_ANY}, {_ANY}) outs(%accb : {_ANY})
  %r = bufferization.to_tensor %accb restrict writable : {_ANY} to tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}}
"""

STATIC_DOUBLE = f"""
func.func @double(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {{
  %init = tensor.empty() : tensor<8x16xf32>
  %r = linalg.generic {{indexing_maps = [{_ID2}, {_ID2}], iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<8x16xf32>) outs(%init : tensor<8x16xf32>) {{
  ^bb0(%a: f32, %o: f32):
    %v = arith.addf %a, %a : f32
    linalg.yield %v : f32
  }} -> tensor<8x16xf32>
  return %r : tensor<8x16xf32>
}}
"""

# Parsed in a context of its own, like a module a user builds elsewhere.
PARSED_SOFTPLUS = ir.Module.parse(SOFTPLUS, context=ir.Context())


def _build_fma() -> ir.Module:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        dyn = ir.ShapedType.get_dynamic_size()
        tile = ir.RankedTensorType.get([dyn, dyn], f32)
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(tile, tile)
            def fma(x: ir.Value, y: ir.Value) -> ir.Value:
                index = ir.IndexType.get()
                dims = [tensor.dim(x, arith.constant(index, i)) for i in (0, 1)]
                init = tensor.EmptyOp(dims, f32).result
                kinds = linalg.ElementwiseKind
                product = linalg.elementwise(x, y, outs=[init], kind=kinds.mul)
                return linalg.elementwise(product, x, outs=[init], kind=kinds.add)

        return module


# x * y + x, built with the Python bindings.
BUILT_FMA = _build_fma()


def softplus_reference(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.softplus(x, beta)


def _softplus_kernel(**settings: object) -> helion.Kernel:
    @helion.kernel(backend="mlir", config=_cfg(8, 16), **settings)
    def softplus(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(SOFTPLUS, [t, 2.0], t, reference=softplus_reference)
        return out

    return softplus


@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize("shape", [(16, 32), (13, 21)], ids=["full", "partial"])
def test_linalg_snippet(static_shapes: bool, shape: tuple[int, int]) -> None:
    x = torch.randn(shape)
    kernel = _softplus_kernel(static_shapes=static_shapes)
    torch.testing.assert_close(
        run_direct(kernel, [x]), torch.nn.functional.softplus(x, 2.0)
    )


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def rows_kernel(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = torch.empty_like(x)
    negated = torch.empty_like(x)
    for tm, tn in hl.tile(x.shape):
        t = x[tm, tn]
        r, n = inline_mlir(ROWS, [t, scale, tm.begin], (t, t))
        scaled[tm, tn] = r + 1.0
        negated[tm, tn] = n
    return scaled, negated


def test_scalar_arguments_and_results() -> None:
    x = torch.randn(13, 21)
    scaled, negated = run_direct(rows_kernel, [x, 0.5])
    rows = torch.arange(13, dtype=torch.float32)[:, None]
    torch.testing.assert_close(scaled, x * 0.5 + rows + 1.0)
    torch.testing.assert_close(negated, -x)


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def column_sums_kernel(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.shape):
        t = x[tm, tn]
        out[tm, tn] = inline_mlir(ADD_COLUMN_SUMS, [table[:, tn], t], t)
    return out


def test_host_tensor_slice_argument() -> None:
    x, table = torch.randn(13, 21), torch.randn(5, 21)
    torch.testing.assert_close(
        run_direct(column_sums_kernel, [x, table]), x + table.sum(0)
    )


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def parsed_module_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(PARSED_SOFTPLUS, [t, 2.0], t)
    return out


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def built_module_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(BUILT_FMA, [t, y[tile]], t)
    return out


def test_module_sources() -> None:
    x, y = torch.randn(13, 21), torch.randn(13, 21)
    torch.testing.assert_close(
        run_direct(parsed_module_kernel, [x]), torch.nn.functional.softplus(x, 2.0)
    )
    torch.testing.assert_close(run_direct(built_module_kernel, [x, y]), x * y + x)


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def scale_alloc_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(SCALE_ALLOC, [t], t)
    return out


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def scale_into_out_kernel(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tm, tn in hl.tile(x.shape):
        t = x[tm, tn]
        out[tm, tn] = inline_mlir(SCALE_INTO, [t, out[tm, tn]], t)
    return out


@helion.kernel(backend="mlir", config=_cfg(8, 16))
def scale_into_input_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(SCALE_INTO, [t, t], t)
    return out


@helion.kernel(backend="mlir", config=_cfg(8, 16, 8))
def matmul_acc_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = inline_mlir(MATMUL_ACC, [acc, x[tm, tk], y[tk, tn]], acc)
        out[tm, tn] = acc
    return out


@pytest.mark.parametrize("shape", [(16, 32), (13, 21)], ids=["full", "partial"])
def test_memref_bodies(shape: tuple[int, int]) -> None:
    x = torch.randn(shape)
    torch.testing.assert_close(run_direct(scale_alloc_kernel, [x]), 3 * x)
    out = torch.zeros_like(x)
    torch.testing.assert_close(run_direct(scale_into_out_kernel, [x, out]), 3 * x)
    torch.testing.assert_close(out, 3 * x)
    m, k = shape[0], 10
    a, b = torch.randn(m, k), torch.randn(k, shape[1])
    torch.testing.assert_close(
        run_direct(matmul_acc_kernel, [a, b]), a @ b, atol=1e-4, rtol=1e-4
    )


def test_writing_an_input_tile_copies_it() -> None:
    x = torch.randn(16, 32)
    before = x.clone()
    torch.testing.assert_close(run_direct(scale_into_input_kernel, [x]), 3 * before)
    torch.testing.assert_close(x, before)


def _static_double_kernel(block_sizes: list[int], static_shapes: bool) -> helion.Kernel:
    @helion.kernel(
        backend="mlir", config=_cfg(*block_sizes), static_shapes=static_shapes
    )
    def static_double(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(STATIC_DOUBLE, [t], t)
        return out

    return static_double


@pytest.mark.parametrize("static_shapes", [True, False], ids=["static", "dynamic"])
def test_static_snippet_shapes(static_shapes: bool) -> None:
    x = torch.randn(13, 21)
    kernel = _static_double_kernel([8, 16], static_shapes)
    torch.testing.assert_close(run_direct(kernel, [x]), 2 * x)


def test_static_snippet_shape_mismatch() -> None:
    kernel = _static_double_kernel([16, 16], static_shapes=True)
    with pytest.raises(helion.exc.InvalidAPIUsage, match=r"tensor<8x16xf32>"):
        run_direct(kernel, [torch.randn(13, 21)])


_CASE_SOURCE = SOFTPLUS


def _case_kernel() -> helion.Kernel:
    @helion.kernel(backend="mlir", config=_cfg(8, 16))
    def case(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(_CASE_SOURCE, [t, 2.0], t)
        return out

    return case


def _snippet(params: str, results: str = "tensor<?x?xf32>", body: str = "") -> str:
    return f"""
func.func @f({params}) -> {results} {{
  {body or "return %x : tensor<?x?xf32>"}
}}
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _snippet(
                "%x: tensor<?x?xf32>, %s: f32",
                body="%y = linalg.frobnicate %x : tensor<?x?xf32>\n"
                "  return %y : tensor<?x?xf32>",
            ),
            "not valid MLIR",
        ),
        (
            "memref.global @g : memref<4xf32>\n"
            + _snippet("%x: tensor<?x?xf32>, %s: f32"),
            "func.func ops only; found memref.global",
        ),
        (
            "func.func private @g(tensor<?x?xf32>, f32) -> tensor<?x?xf32>\n"
            + _snippet("%x: tensor<?x?xf32>, %s: f32"),
            "has no body",
        ),
        (
            _snippet(
                "%x: memref<?x?xf32>, %s: f32",
                "memref<?x?xf32>",
                "return %x : memref<?x?xf32>",
            ),
            "takes and returns tensors",
        ),
        (
            _snippet("%x: tensor<?x?xf32>, %s: f32", "f32", "return %s : f32"),
            "must return one or more ranked tensors",
        ),
        (_snippet("%x: tensor<?x?xf32>"), "takes 1 arguments"),
        (
            _snippet(
                "%x: tensor<?x?xbf16>, %s: f32",
                "tensor<?x?xbf16>",
                "return %x : tensor<?x?xbf16>",
            ),
            r"declares tensor<\?x\?xbf16>, the kernel passes a 2-d torch.float32",
        ),
        (
            _snippet(
                "%x: tensor<?xf32>, %s: f32",
                "tensor<?xf32>",
                "return %x : tensor<?xf32>",
            ),
            r"declares tensor<\?xf32>",
        ),
        (
            _snippet("%x: tensor<?x?xf32>, %s: index"),
            "declares index, the kernel passes the float 2.0",
        ),
        (
            _snippet("%x: tensor<?x?xf32>, %s: tensor<?x?xf32>"),
            "the kernel passes the scalar 2.0",
        ),
    ],
    ids=[
        "syntax",
        "top_level_op",
        "declaration_entry",
        "memref_signature",
        "scalar_result",
        "arity",
        "dtype",
        "rank",
        "float_for_index",
        "scalar_for_tensor",
    ],
)
def test_invalid_snippets(
    monkeypatch: pytest.MonkeyPatch, source: str, message: str
) -> None:
    monkeypatch.setitem(globals(), "_CASE_SOURCE", source)
    with pytest.raises(helion.exc.InvalidAPIUsage, match=message):
        run_direct(_case_kernel(), [torch.randn(13, 21)])


def _layout_hint_kernel() -> helion.Kernel:
    @helion.kernel(backend="mlir", config=_cfg(8, 16))
    def layout_hint(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(_CASE_SOURCE, [t], t)
        return out

    return layout_hint


def test_identity_layout_hint_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A text of its own: hints are reported once per snippet text per process.
    source = SCALE_ALLOC.replace(_ANY, "memref<?x?xf32>") + "// hint test\n"
    monkeypatch.setitem(globals(), "_CASE_SOURCE", source)
    x = torch.randn(16, 32)
    torch.testing.assert_close(run_direct(_layout_hint_kernel(), [x]), 3 * x)
    first = capsys.readouterr().err
    assert "WARNING[InlineMLIRHint]" in first
    assert "strided<[?, ?], offset: ?>" in first
    run_direct(_layout_hint_kernel(), [x])
    assert "InlineMLIRHint" not in capsys.readouterr().err


def test_hint_can_be_ignored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = SCALE_ALLOC.replace(_ANY, "memref<?x?xf32>") + "// ignored hint\n"
    monkeypatch.setitem(globals(), "_CASE_SOURCE", source)

    @helion.kernel(backend="mlir", config=_cfg(8, 16), ignore_warnings=[InlineMLIRHint])
    def quiet(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(_CASE_SOURCE, [t], t)
        return out

    run_direct(quiet, [torch.randn(16, 32)])
    assert "InlineMLIRHint" not in capsys.readouterr().err


def test_reference_in_ref_mode() -> None:
    x = torch.randn(13, 21)
    kernel = _softplus_kernel(ref_mode=helion.RefMode.EAGER)
    torch.testing.assert_close(kernel(x), torch.nn.functional.softplus(x, 2.0))


def test_ref_mode_needs_a_reference() -> None:
    @helion.kernel(backend="mlir", config=_cfg(8, 16), ref_mode=helion.RefMode.EAGER)
    def no_reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(SOFTPLUS, [t, 2.0], t)
        return out

    with pytest.raises(helion.exc.InvalidAPIUsage, match="needs reference="):
        no_reference(torch.randn(13, 21))


def double_reference(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def test_reference_traced_by_other_backends() -> None:
    @helion.kernel(backend="triton", config=_cfg(8, 16))
    def on_triton(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            t = x[tile]
            out[tile] = inline_mlir(STATIC_DOUBLE, [t], t, reference=double_reference)
        return out

    bound = on_triton.bind((torch.randn(13, 21),))
    targets = {
        node.target
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
    }
    assert _inline_mlir not in targets
    assert torch.ops.aten.mul.Tensor in targets


def test_outside_a_kernel() -> None:
    with pytest.raises(helion.exc.NotInsideKernel):
        inline_mlir(SOFTPLUS, [torch.randn(4, 4), 2.0], torch.randn(4, 4))


PACK_BLOCKS = """
func.func @pack(%b: tensor<?x?xf32>, %dest: tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %packed = linalg.pack %b padding_value(%zero : f32) outer_dims_perm = [1, 0]
      inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest
      : tensor<?x?xf32> -> tensor<?x?x8x8xf32>
  return %packed : tensor<?x?x8x8xf32>
}
"""


@helion.kernel(backend="mlir", config=_cfg())
def linalg_pack_kernel(b: torch.Tensor) -> torch.Tensor:
    k, n = b.shape
    out = torch.empty(
        ((n + 7) // 8, (k + 7) // 8, 8, 8), dtype=b.dtype, device=b.device
    )
    for _ in hl.grid(1):
        whole = out[:, :, :, :]
        out[:, :, :, :] = inline_mlir(PACK_BLOCKS, [b[:, :], whole], whole)
    return out


def _packed_blocks(b: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(b, (0, -b.shape[1] % 8, 0, -b.shape[0] % 8))
    kb, nb = padded.shape[0] // 8, padded.shape[1] // 8
    return padded.reshape(kb, 8, nb, 8).permute(2, 0, 1, 3).contiguous()


def test_linalg_pack_is_lowered() -> None:
    b = torch.randn(21, 30)
    torch.testing.assert_close(run_direct(linalg_pack_kernel, [b]), _packed_blocks(b))


@helion.kernel(backend="mlir", config=_cfg(32, 32))
def opt_softplus_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(SOFTPLUS, [t, 2.0], t)
    return out


@helion.kernel(backend="mlir", config=_cfg(32, 32, 32))
def opt_matmul_acc_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = inline_mlir(MATMUL_ACC, [acc, x[tm, tk], y[tk, tn]], acc)
        out[tm, tn] = acc
    return out


@pytest.mark.isolated
@pytest.mark.slow
def test_opt_pipeline() -> None:
    x, y = torch.randn(64, 64), torch.randn(64, 96)
    b = torch.randn(45, 70)
    with opt_pipeline():
        softplus = opt_softplus_kernel(x)
        product = opt_matmul_acc_kernel(x, y)
        packed = linalg_pack_kernel(b)
    torch.testing.assert_close(softplus, torch.nn.functional.softplus(x, 2.0))
    torch.testing.assert_close(product, x @ y, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(packed, _packed_blocks(b))
