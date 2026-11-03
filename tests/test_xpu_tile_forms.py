"""``mlir_transforms.xegpu_tile_forms``: the rewrites on small IR, what they leave alone,
and what each form's kernel lowers to. On the device: ``test_xpu_tile_forms_runtime``."""

from __future__ import annotations

import helion
from mlir import ir
import pytest
import torch

from tests import xpu_layout_forms as forms
from tests import xpu_tile_form_kernels as kernels
from tests.harness import xpu_tensor

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms import xegpu_tile_forms as tf

_TD = "!xegpu.tensor_desc<32x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>"
_READ = f"xegpu.load_nd %d[%c0, %c0] : {_TD} -> vector<32x32xbf16>"
_WRAP = """
gpu.module @m [#xevm.target<O = 3>] {{
  gpu.func @k(%a: memref<64x64xbf16>, %f: f32) kernel {{
    %c0 = arith.constant 0 : index
    %d = xegpu.create_nd_tdesc %a : memref<64x64xbf16> -> {td}
    {body}
    gpu.return
  }}
}}
"""


@pytest.fixture(autouse=True)
def _context():
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        yield context


def _func(body: str) -> tuple[ir.Module, ir.Operation]:
    module = ir.Module.parse(_WRAP.format(td=_TD, body=body))
    gpu_module = next(iter(module.body.operations)).operation
    return module, next(iter(gpu_module.regions[0].blocks[0].operations)).operation


def _names(func: ir.Operation) -> list[str]:
    return [op.name for op in tf._walk(func)]


def test_hoist_moves_a_transpose_to_the_read() -> None:
    module, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %c = arith.constant dense<1.0> : vector<32x32xf32>
    %s = arith.addf %e, %c : vector<32x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t) : (vector<32x32xf32>) -> ()
    """)
    assert tf.hoist_transposes(func)
    module.operation.verify()
    names = _names(func)
    # The transpose takes the read directly, and nothing is left of the old math.
    assert names.count("vector.transpose") == 1
    assert names.count("arith.extf") == 1
    transpose = next(op for op in tf._walk(func) if op.name == "vector.transpose")
    assert tf._defining(transpose.operands[0]).name == "xegpu.load_nd"


def test_hoist_of_a_non_square_tile_transposes_its_constant() -> None:
    td = "!xegpu.tensor_desc<16x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>"
    module, func = _func(f"""
    %d2 = xegpu.create_nd_tdesc %a : memref<64x64xbf16> -> {td}
    %l = xegpu.load_nd %d2[%c0, %c0] : {td} -> vector<16x32xbf16>
    %e = arith.extf %l : vector<16x32xbf16> to vector<16x32xf32>
    %c = arith.constant dense<2.0> : vector<16x32xf32>
    %s = arith.mulf %e, %c : vector<16x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<16x32xf32> to vector<32x16xf32>
    "test.use"(%t) : (vector<32x16xf32>) -> ()
    """)
    assert tf.hoist_transposes(func)
    module.operation.verify()
    text = str(func)
    assert "dense<2.000000e+00> : vector<32x16xf32>" in text


def test_hoist_keeps_the_math_for_other_users() -> None:
    module, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %t = vector.transpose %e, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t, %e) : (vector<32x32xf32>, vector<32x32xf32>) -> ()
    """)
    assert tf.hoist_transposes(func)
    module.operation.verify()
    assert _names(func).count("arith.extf") == 2


@pytest.mark.parametrize(
    ("source", "result"),
    [
        # A read's and a broadcast's own transposes are the forms that work.
        (f"%s = {_READ}", "vector<32x32xbf16>"),
        ("%s = vector.broadcast %f : f32 to vector<32x32xf32>", "vector<32x32xf32>"),
        # Not reads: an unknown op, a constant with shape.
        ('%s = "test.value"() : () -> vector<32x32xf32>', "vector<32x32xf32>"),
        (
            "%s = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>",
            "vector<2x2xf32>",
        ),
    ],
    ids=["read", "broadcast", "unknown", "non-splat"],
)
def test_hoist_leaves_other_transposes_alone(source: str, result: str) -> None:
    _, func = _func(f"""
    {source}
    %t = vector.transpose %s, [1, 0] : {result} to {result}
    "test.use"(%t) : ({result}) -> ()
    """)
    before = str(func)
    assert not tf.hoist_transposes(func)
    assert str(func) == before


def test_hoist_leaves_math_of_a_broadcast_alone() -> None:
    """No kernel reaches this (``prepare_for_xegpu`` fuses it into its contraction, see
    ``test_two_input_computed_operand_then_transposed_stays_refused``), so it has no device
    run, and it stays as it is."""
    _, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %u = vector.broadcast %f : f32 to vector<32x32xf32>
    %s = arith.mulf %e, %u : vector<32x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t) : (vector<32x32xf32>) -> ()
    """)
    before = str(func)
    assert not tf.hoist_transposes(func)
    assert str(func) == before


def test_hoist_leaves_math_of_an_unknown_value_alone() -> None:
    _, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %u = "test.value"() : () -> vector<32x32xf32>
    %s = arith.addf %e, %u : vector<32x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t) : (vector<32x32xf32>) -> ()
    """)
    before = str(func)
    assert not tf.hoist_transposes(func)
    assert str(func) == before


_DPAS = (
    "xegpu.dpas %A, %B, %acc : vector<32x32xbf16>, vector<32x32xbf16>, "
    "vector<32x32xf32> -> vector<32x32xf32>"
)
_ACC = "%acc = arith.constant dense<0.0> : vector<32x32xf32>"


def _reads(func: ir.Operation) -> int:
    return _names(func).count("xegpu.load_nd")


def _no_refusal(message: str) -> Exception:
    return AssertionError(message)


def test_split_gives_a_and_b_a_read_each() -> None:
    module, func = _func(f"""
    {_ACC}
    %A = {_READ}
    %r = {_DPAS.replace("%B", "%A")}
    "test.use"(%r) : (vector<32x32xf32>) -> ()
    """)
    assert tf.split_reads(func, refuse=_no_refusal)
    module.operation.verify()
    assert _reads(func) == 2
    dpas = next(op for op in tf._walk(func) if op.name == "xegpu.dpas")
    assert dpas.operands[0] != dpas.operands[1]


def test_split_gives_b_with_other_users_a_read_of_its_own() -> None:
    module, func = _func(f"""
    {_ACC}
    %A = "test.value"() : () -> vector<32x32xbf16>
    %B = {_READ}
    %r = {_DPAS}
    "test.use"(%r, %B) : (vector<32x32xf32>, vector<32x32xbf16>) -> ()
    """)
    assert tf.split_reads(func, refuse=_no_refusal)
    module.operation.verify()
    assert _reads(func) == 2
    dpas = next(op for op in tf._walk(func) if op.name == "xegpu.dpas")
    use = next(op for op in tf._walk(func) if op.name == "test.use")
    assert dpas.operands[1] != use.operands[1]


def test_split_of_a_tile_and_its_transpose_leaves_no_dead_transpose() -> None:
    module, func = _func(f"""
    %l = {_READ}
    %t = vector.transpose %l, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    "test.use"(%l, %t) : (vector<32x32xbf16>, vector<32x32xbf16>) -> ()
    """)
    assert tf.split_reads(func, refuse=_no_refusal)
    module.operation.verify()
    assert _reads(func) == 2
    assert _names(func).count("vector.transpose") == 1
    use = next(op for op in tf._walk(func) if op.name == "test.use")
    transposed = tf._defining(use.operands[1])
    assert transposed.name == "vector.transpose"
    assert transposed.operands[0] != use.operands[0]


@pytest.mark.parametrize(
    "body",
    [
        # A's read with other users shares one read, as before.
        f"""{_ACC}
        %A = {_READ}
        %B = "test.value"() : () -> vector<32x32xbf16>
        %r = {_DPAS}
        "test.use"(%r, %A) : (vector<32x32xf32>, vector<32x32xbf16>) -> ()""",
        # One use, on a transpose.
        f"""%l = {_READ}
        %t = vector.transpose %l, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
        "test.use"(%t) : (vector<32x32xbf16>) -> ()""",
        # Two uses of one kind.
        f"""%l = {_READ}
        "test.use"(%l, %l) : (vector<32x32xbf16>, vector<32x32xbf16>) -> ()""",
    ],
    ids=["a-and-other", "only-transposed", "same-kind"],
)
def test_split_leaves_a_read_that_serves_its_uses(body: str) -> None:
    _, func = _func(body)
    before = str(func)
    assert not tf.split_reads(func, refuse=_no_refusal)
    assert str(func) == before


def test_split_refuses_a_buffer_the_kernel_stores_to() -> None:
    """Subgroups would read each other's stored tiles under a second distribution."""
    _, func = _func(f"""
    {_ACC}
    %A = {_READ}
    %r = {_DPAS.replace("%B", "%A")}
    %t = arith.truncf %r : vector<32x32xf32> to vector<32x32xbf16>
    xegpu.store_nd %t, %d[%c0, %c0] : vector<32x32xbf16>, {_TD}
    """)
    before = str(func)
    with pytest.raises(ValueError, match="also stores to"):
        tf.split_reads(func, refuse=ValueError)
    assert str(func) == before


_INT_STORE = """
    %i = memref.extract_aligned_pointer_as_index %a : memref<64x64xbf16> -> index
    %p = arith.index_cast %i : index to i64
    %o = "test.value"() : () -> vector<32xindex>
    %m = "test.value"() : () -> vector<32xi1>
    %v = "test.value"() : () -> vector<32xbf16>
    xegpu.store %v, %p[%o], %m : vector<32xbf16>, i64, vector<32xindex>, vector<32xi1>
"""
_VIEW_STORE = f"""
    %vw = memref.cast %a : memref<64x64xbf16> to memref<64x64xbf16>
    %d2 = xegpu.create_nd_tdesc %vw : memref<64x64xbf16> -> {_TD}
    %t2 = "test.value"() : () -> vector<32x32xbf16>
    xegpu.store_nd %t2, %d2[%c0, %c0] : vector<32x32xbf16>, {_TD}
"""
_ND_STORE = f"""
    %t2 = "test.value"() : () -> vector<32x32xbf16>
    xegpu.store_nd %t2, %d[%c0, %c0] : vector<32x32xbf16>, {_TD}
"""


@pytest.mark.parametrize(
    "store",
    [_ND_STORE, _VIEW_STORE, _INT_STORE],
    ids=["same-memref", "through-a-view", "through-a-pointer"],
)
def test_split_refuses_a_transposed_read_of_a_stored_buffer(store: str) -> None:
    """One use, so nothing to split: the read is still distributed unlike the store."""
    _, func = _func(f"""
    %l = {_READ}
    %t = vector.transpose %l, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    "test.use"(%t) : (vector<32x32xbf16>) -> ()
    {store}
    """)
    with pytest.raises(ValueError, match="transposed read of a buffer"):
        tf.split_reads(func, refuse=ValueError)


def test_split_leaves_a_plain_read_of_a_stored_buffer() -> None:
    """Distributed like the store: ``a[t] = a[t] + 1``."""
    _, func = _func(f"""
    %l = {_READ}
    "test.use"(%l) : (vector<32x32xbf16>) -> ()
    {_ND_STORE}
    """)
    before = str(func)
    assert not tf.split_reads(func, refuse=ValueError)
    assert str(func) == before


def test_split_takes_a_transposed_read_of_another_buffer() -> None:
    _, func = _func(f"""
    %b = memref.alloc() : memref<64x64xbf16>
    %db = xegpu.create_nd_tdesc %b : memref<64x64xbf16> -> {_TD}
    %l = {_READ}
    %t = vector.transpose %l, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    "test.use"(%t, %l) : (vector<32x32xbf16>, vector<32x32xbf16>) -> ()
    %t2 = "test.value"() : () -> vector<32x32xbf16>
    xegpu.store_nd %t2, %db[%c0, %c0] : vector<32x32xbf16>, {_TD}
    """)
    assert tf.split_reads(func, refuse=ValueError)
    assert _reads(func) == 2


@pytest.mark.isolated
def test_hoist_of_a_square_erases_each_op_once() -> None:
    """Exact regression: ``(t * t).T``, one value as both operands, was erased twice, and
    the process died: the bindings have no live-operation map."""
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %s = arith.mulf %e, %e : vector<32x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t) : (vector<32x32xf32>) -> ()
    """)
        assert tf.hoist_transposes(func)
        module.operation.verify()
        assert _names(func).count("arith.mulf") == 1
        assert _names(func).count("arith.extf") == 1


@pytest.mark.isolated
def test_hoist_of_a_clamp_erases_each_op_once() -> None:
    """Exact regression: ``select(cmp(e, c), e, c)``, a shared chain of three users."""
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module, func = _func(f"""
    %l = {_READ}
    %e = arith.extf %l : vector<32x32xbf16> to vector<32x32xf32>
    %c = arith.constant dense<1.0> : vector<32x32xf32>
    %m = arith.cmpf olt, %e, %c : vector<32x32xf32>
    %s = arith.select %m, %e, %c : vector<32x32xi1>, vector<32x32xf32>
    %t = vector.transpose %s, [1, 0] : vector<32x32xf32> to vector<32x32xf32>
    "test.use"(%t) : (vector<32x32xf32>) -> ()
    """)
        assert tf.hoist_transposes(func)
        module.operation.verify()
        names = _names(func)
        assert names.count("arith.select") == 1
        assert names.count("arith.cmpf") == 1
        assert names.count("arith.extf") == 1


_MAP = "affine_map<(d0, d1) -> (d0, d1)>"
_ROW = "affine_map<(d0, d1) -> (d0)>"
_ITER = 'iterator_types = ["parallel", "reduction"], kind = #vector.kind<{kind}>'
_V = "vector<8x16x{e}>"


def _contract(kind: str = "add", element: str = "f32", acc: str = "f32") -> str:
    v = _V.format(e=element)
    return f"""
    %x = "test.value"() : () -> {v}
    %y = "test.value"() : () -> {v}
    %i = "test.value"() : () -> vector<8x{acc}>
    %r = vector.contract {{indexing_maps = [{_MAP}, {_MAP}, {_ROW}],
      {_ITER.format(kind=kind)}}} %x, %y, %i : {v}, {v} into vector<8x{acc}>
    "test.use"(%r) : (vector<8x{acc}>) -> ()
    """


def test_expand_leaves_a_permuted_result_alone() -> None:
    """A reduction keeps its dims in order: this one would need a transpose too."""
    v = "vector<4x8x16xf32>"
    _, func = _func(f"""
    %x = "test.value"() : () -> {v}
    %i = "test.value"() : () -> vector<16x4xf32>
    %r = vector.contract {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2, d0)>],
      iterator_types = ["parallel", "reduction", "parallel"],
      kind = #vector.kind<add>}} %x, %x, %i : {v}, {v} into vector<16x4xf32>
    "test.use"(%r) : (vector<16x4xf32>) -> ()
    """)
    before = str(func)
    assert not tf.expand_elementwise_contractions(func)
    assert str(func) == before


def test_expand_writes_out_a_dot_product() -> None:
    module, func = _func(_contract())
    assert tf.expand_elementwise_contractions(func)
    module.operation.verify()
    names = _names(func)
    assert "vector.contract" not in names
    assert "arith.mulf" in names
    reduction = next(op for op in tf._walk(func) if op.name == "vector.multi_reduction")
    assert list(ir.DenseI64ArrayAttr(reduction.attributes["reduction_dims"])) == [1]
    assert str(reduction.attributes["kind"]) == "#vector.kind<add>"


def test_expand_extends_bf16_operands_into_an_f32_accumulator() -> None:
    module, func = _func(_contract(element="bf16"))
    assert tf.expand_elementwise_contractions(func)
    module.operation.verify()
    assert _names(func).count("arith.extf") == 2
    product = next(op for op in tf._walk(func) if op.name == "arith.mulf")
    assert ir.VectorType(product.results[0].type).element_type == ir.F32Type.get()


@pytest.mark.parametrize(
    "body",
    [
        # A matmul: the operands' maps differ.
        """
    %x = "test.value"() : () -> vector<8x16xf32>
    %y = "test.value"() : () -> vector<16x4xf32>
    %i = "test.value"() : () -> vector<8x4xf32>
    %r = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %x, %y, %i : vector<8x16xf32>, vector<16x4xf32>
      into vector<8x4xf32>
    "test.use"(%r) : (vector<8x4xf32>) -> ()
    """,
        # Another kind.
        _contract(kind="maxnumf"),
        # Integers.
        _contract(element="i32", acc="i32"),
        # A narrower accumulator.
        _contract(element="f32", acc="bf16"),
    ],
    ids=["matmul", "kind", "integer", "narrowing"],
)
def test_expand_leaves_other_contractions_alone(body: str) -> None:
    _, func = _func(body)
    before = str(func)
    assert not tf.expand_elementwise_contractions(func)
    assert str(func) == before


def _layouts(
    fn: object,
    blocks: list[int],
    *shapes: tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, xegpu_layouts=["ir"]),
    )
    text = str(
        lower_mlir(
            kernel,
            [xpu_tensor(*shape, dtype=dtype) for shape in shapes],
            target="xpu",
            stop_at="xegpu-wg",
        )
    )
    return [line for line in text.splitlines() if "xegpu.load_nd " in line]


def test_computed_transposed_a_reads_transposed() -> None:
    (a, b) = _layouts(kernels.computed_a, [64, 64, 32], (256, 256), (256, 256))
    assert "inst_data = [16, 16], order = [0, 1]" in a
    assert "order" not in b


def test_one_tile_as_a_and_transposed_b_reads_twice() -> None:
    lines = _layouts(kernels.tile_times_transpose, [64, 64, 32], (256, 256))
    assert len(lines) == 2
    assert sum("inst_data = [16, 16], order = [0, 1]" in line for line in lines) == 1
    assert sum("order" not in line for line in lines) == 1


def test_a_tile_as_both_operands_is_not_one_dim_class() -> None:
    """The rows and columns of its output stay free: they share a buffer, not a class."""
    lines = _layouts(kernels.tile_times_itself, [64, 64, 64], (256, 256))
    assert len(lines) == 2
    assert all("sg_layout = [1, 1]" not in line for line in lines)


def test_transposed_elementwise_reads_one_tile_each_way() -> None:
    lines = _layouts(kernels.tile_plus_transpose, [64, 64], (256, 256))
    assert len(lines) == 2
    assert sum("order = [0, 1]" in line for line in lines) == 1
    assert sum("order" not in line for line in lines) == 1


def test_two_input_computed_operand_then_transposed_stays_refused() -> None:
    """Fused into a contraction of three inputs in ``prepare_for_xegpu``, ahead of any of
    this. A kernel can't reach the rewrite; the refusal names what it can't do."""
    with pytest.raises(UnsupportedOperationError, match="non-innermost"):
        _layouts(
            kernels.computed_two_inputs,
            [64, 64, 32],
            (256, 256),
            (256, 256),
            (256, 256),
        )


@pytest.mark.parametrize("columns", [120, 248])
@pytest.mark.parametrize(
    ("fn", "blocks", "shapes"),
    [
        (kernels.row_l2, [64, 32], lambda n: ((248, 96), (n, 96))),
        (kernels.row_l2_of_input, [64], lambda n: ((248, n),)),
        (kernels.row_dot, [64], lambda n: ((248, n), (248, n))),
    ],
    ids=["gemm", "input", "dot"],
)
def test_row_forms_of_partial_columns_are_refused(
    fn: object, blocks: list[int], shapes: object, columns: int
) -> None:
    """Their rows stay whole on a subgroup, and no block takes such a row."""
    with pytest.raises(UnsupportedOperationError):
        _layouts(fn, blocks, *shapes(columns))


def _hinted_read(fn: object, rows: int, *shapes: tuple[int, int]) -> tuple:
    blocks = [128, 128, 32] if len(shapes) > 1 else [128, 128]
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(
            block_sizes=blocks,
            xegpu_layouts=["ir"],
            xegpu_load_tiles=[[rows, 16]],
        ),
    )
    return kernel, [xpu_tensor(*shape, dtype=torch.bfloat16) for shape in shapes]


_HINTED = {
    "only_transposed": (kernels.only_transposed, ((256, 256),)),
    "gemm_plus_transposed": (
        forms.gemm_plus_transposed,
        ((256, 256), (256, 256), (256, 256)),
    ),
}


@pytest.mark.isolated
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("name", list(_HINTED))
def test_transposed_read_hint_of_other_rows_lowers(name: str, rows: int) -> None:
    """Exact regression: this hint aborted the process, in upstream's layout code."""
    fn, shapes = _HINTED[name]
    kernel, args = _hinted_read(fn, rows, *shapes)
    lower_mlir(kernel, args, target="xpu", stop_at="xevm")


@pytest.mark.parametrize("forked", [False, True], ids=["in-process", "forked"])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_transposed_read_hint_of_other_rows_is_dropped(
    rows: int,
    forked: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forked is the default: the line comes back on the module."""
    if forked:
        monkeypatch.delenv("HELION_MLIR_COMPILE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    kernel, args = _hinted_read(kernels.only_transposed, rows, (256, 256))
    with caplog.at_level("WARNING"):
        lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg")
    assert "a transposed load takes [16, 16] tiles" in caplog.text


@pytest.mark.parametrize("block", [256, 128])
@pytest.mark.parametrize(
    ("fn", "inputs"),
    [
        (kernels.in_place_transposed, 1),
        (kernels.in_place_transposed_minus_one, 1),
        (kernels.in_place_math_transposed, 1),
        (kernels.in_place_transposed_plus_z, 2),
    ],
    ids=["t", "t-minus-one", "math-t", "t-plus-z"],
)
def test_in_place_transposed_read_is_refused(
    fn: object, inputs: int, block: int
) -> None:
    """Exact regression: wrong rows with no diagnostic, 26 of 30 runs at 256: a subgroup read
    tiles another one had stored."""
    shape = (2 * block, 2 * block)
    with pytest.raises(UnsupportedOperationError, match="transposed read of a buffer"):
        _layouts(fn, [block, block], *[shape] * inputs)


def test_in_place_transposed_read_after_a_gemm_is_refused() -> None:
    """Exact regression, wrong on the base too: ``z[t] = acc + z[t].T``."""
    with pytest.raises(UnsupportedOperationError, match="transposed read of a buffer"):
        _layouts(
            kernels.in_place_transposed_after_a_gemm,
            [256, 256, 64],
            (512, 512),
            (512, 512),
            (512, 512),
        )


@pytest.mark.parametrize(
    ("fn", "blocks", "shapes"),
    [
        (kernels.row_dot, [128], ((512, 256), (512, 256))),
        (kernels.row_l2_of_input, [128], ((512, 256),)),
    ],
    ids=["dot", "l2"],
)
def test_row_form_is_a_reduction_of_rows(
    fn: object, blocks: list[int], shapes: tuple
) -> None:
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks, xegpu_layouts=["ir"]),
    )
    args = [xpu_tensor(*shape, dtype=torch.bfloat16) for shape in shapes]
    text = str(lower_mlir(kernel, args, target="xpu", stop_at="xegpu-wg"))
    assert "vector.contract" not in text
    assert "vector.multi_reduction <add>" in text
    assert "xegpu.dpas" not in text


@pytest.mark.parametrize(
    ("fn", "reads"),
    [
        (kernels.b_with_other_user, 3),
        (kernels.a_with_other_user, 2),
        (kernels.transposed_a_with_other_user, 2),
        (kernels.transposed_b_with_other_user, 3),
    ],
    ids=["b", "a", "transposed-a", "transposed-b"],
)
def test_reads_of_a_tile_with_other_users(fn: object, reads: int) -> None:
    lines = _layouts(fn, [64, 64, 64], (256, 256), (256, 256))
    assert len(lines) == reads


@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (kernels.elementwise_transposed, ((128, 128), (128, 128))),
        (kernels.only_transposed, ((128, 128),)),
        (kernels.tile_plus_transpose, ((128, 128),)),
        (kernels.math_then_transposed, ((128, 128), (128, 128))),
    ],
    ids=["plus", "only", "both-ways", "math"],
)
def test_f32_transposed_read_is_refused_clearly(fn: object, shapes: tuple) -> None:
    """It failed later, with ``tile_width`` or ``pack_register`` from ``xevm.blockload2d``."""
    with pytest.raises(UnsupportedOperationError, match="32-bit elements"):
        _layouts(fn, [64, 64], *shapes, dtype=torch.float32)


def test_f32_dpas_operand_is_refused_clearly() -> None:
    with pytest.raises(UnsupportedOperationError, match="32-bit elements and narrowed"):
        _layouts(
            kernels.narrowed_operands,
            [64, 64, 32],
            (128, 128),
            (128, 128),
            dtype=torch.float32,
        )


def test_dot_of_a_tile_with_its_transpose_is_not_blamed_on_m1() -> None:
    with pytest.raises(UnsupportedOperationError) as refusal:
        _layouts(kernels.row_dot_with_transpose, [64, 64], (128, 128))
    assert "M = 1" not in str(refusal.value)


_T = "!xegpu.tensor_desc<32x32xbf16, #xegpu.block_tdesc_attr<boundary_check = false>>"
_MEM = "memref<64x64xbf16>"
_USE = """%t = vector.transpose %l, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    "test.use"(%t) : (vector<32x32xbf16>) -> ()"""
_SLIPS = {
    "pointer-plus": f"""
    %p = memref.extract_aligned_pointer_as_index %a : {_MEM} -> index
    %q = arith.addi %p, %c0 : index
    %i = arith.index_cast %q : index to i64
    %o = "test.value"() : () -> vector<32xindex>
    %m = "test.value"() : () -> vector<32xi1>
    %v = "test.value"() : () -> vector<32xbf16>
    xegpu.store %v, %i[%o], %m : vector<32xbf16>, i64, vector<32xindex>, vector<32xi1>""",
    "loop-carried": f"""
    %r = scf.for %i = %c0 to %c0 step %c0 iter_args(%m = %a) -> ({_MEM}) {{
      %d2 = xegpu.create_nd_tdesc %m : {_MEM} -> {_T}
      %v = "test.value"() : () -> vector<32x32xbf16>
      xegpu.store_nd %v, %d2[%c0, %c0] : vector<32x32xbf16>, {_T}
      scf.yield %m : {_MEM}
    }}""",
    "if-result": f"""
    %b = "test.cond"() : () -> i1
    %r = scf.if %b -> ({_MEM}) {{
      scf.yield %a : {_MEM}
    }} else {{
      scf.yield %a : {_MEM}
    }}
    %d2 = xegpu.create_nd_tdesc %r : {_MEM} -> {_T}
    %v = "test.value"() : () -> vector<32x32xbf16>
    xegpu.store_nd %v, %d2[%c0, %c0] : vector<32x32xbf16>, {_T}""",
    "select": f"""
    %b = "test.cond"() : () -> i1
    %r = arith.select %b, %a, %a : {_MEM}
    %d2 = xegpu.create_nd_tdesc %r : {_MEM} -> {_T}
    %v = "test.value"() : () -> vector<32x32xbf16>
    xegpu.store_nd %v, %d2[%c0, %c0] : vector<32x32xbf16>, {_T}""",
    "memref-copy": f"""
    %o = memref.alloc() : {_MEM}
    memref.copy %o, %a : {_MEM} to {_MEM}""",
    "memref-store": """
    %z = "test.value"() : () -> bf16
    memref.store %z, %a[%c0, %c0] : memref<64x64xbf16>""",
    "vector-store": f"""
    %v = "test.value"() : () -> vector<32xbf16>
    vector.store %v, %a[%c0, %c0] : {_MEM}, vector<32xbf16>""",
    "transfer-write": f"""
    %v = "test.value"() : () -> vector<32x32xbf16>
    vector.transfer_write %v, %a[%c0, %c0] : vector<32x32xbf16>, {_MEM}""",
    "memref-transpose": f"""
    %w = memref.transpose %a (d0, d1) -> (d1, d0) : {_MEM} to memref<64x64xbf16, affine_map<(d0, d1)[s0] -> (d1 * 64 + d0 + s0)>>""",
    "unknown-op-on-the-buffer": """
    "test.writes"(%a) : (memref<64x64xbf16>) -> ()""",
}


@pytest.mark.parametrize(
    "name", [n for n in _SLIPS if n != "memref-transpose"], ids=str
)
def test_split_refuses_a_transposed_read_if_a_write_may_reach_the_buffer(
    name: str,
) -> None:
    """Fail closed: a store this module can't trace is taken to reach any read."""
    _, func = _func(f"""
    %l = {_READ}
    {_USE}
    {_SLIPS[name]}
    """)
    with pytest.raises(ValueError, match="transposed read of a buffer"):
        tf.split_reads(func, refuse=ValueError)


def test_split_refuses_a_read_of_an_untraced_descriptor() -> None:
    _, func = _func(f"""
    %e = "test.desc"() : () -> {_T}
    %l = xegpu.load_nd %e[%c0, %c0] : {_T} -> vector<32x32xbf16>
    {_USE}
    """)
    with pytest.raises(ValueError, match="transposed read of a buffer"):
        tf.split_reads(func, refuse=ValueError)


def test_split_refuses_a_read_of_a_loop_carried_buffer() -> None:
    _, func = _func(f"""
    %r = scf.for %i = %c0 to %c0 step %c0 iter_args(%m = %a) -> ({_MEM}) {{
      %d2 = xegpu.create_nd_tdesc %m : {_MEM} -> {_T}
      %l = xegpu.load_nd %d2[%c0, %c0] : {_T} -> vector<32x32xbf16>
      {_USE}
      scf.yield %m : {_MEM}
    }}
    """)
    with pytest.raises(ValueError, match="transposed read of a buffer"):
        tf.split_reads(func, refuse=ValueError)
