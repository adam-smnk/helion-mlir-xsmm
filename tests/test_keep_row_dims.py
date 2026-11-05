"""``keep_row_dims``: a GPU kernel's ``[rows]`` row-reduction results stored into a
rank-1 buffer become ``[rows, 1]`` stores into the buffer expanded to
``[n, 1]``; other stores are left alone. IR only."""

from __future__ import annotations

import re

from mlir import ir
import pytest

from tests.harness import visits
from tests.ir_query import ops_named
from tests.ir_query import parse

from helion_mlir_backend._compiler.mlir_transforms import (
    keep_row_dims as keep_row_dims_module,
)
from helion_mlir_backend._compiler.mlir_transforms.keep_row_dims import keep_row_dims


def _gpu_module(body: str, args: str = "%out: memref<128xf32>") -> str:
    return f"""
module attributes {{gpu.container_module}} {{
  gpu.module @m {{
    gpu.func @k(%x: vector<64x64xf32>, %i: index, {args}) kernel {{
      %init = arith.constant dense<0.0> : vector<64xf32>
      {body}
      gpu.return
    }}
  }}
}}"""


def _host_module(body: str, args: str = "%out: memref<128xf32>") -> str:
    return f"""
module {{
  func.func @k(%x: vector<64x64xf32>, %i: index, {args}) {{
    %init = arith.constant dense<0.0> : vector<64xf32>
    {body}
    return
  }}
}}"""


_SUM = "%s = vector.multi_reduction <add>, %x, %init [1] : vector<64x64xf32> to vector<64xf32>"


def _rewrite(text: str) -> tuple[int, str, str]:
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = False
        before = str(ir.Module.parse(text))
        module = ir.Module.parse(text)
        count = keep_row_dims(module)
        assert module.operation.verify()
        return count, before, str(module)


@pytest.mark.parametrize(
    ("body", "args", "element"),
    [
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
            "f32",
        ),
        # The row sums rounded to the output's dtype first.
        (
            f"""{_SUM}
      %t = arith.truncf %s : vector<64xf32> to vector<64xbf16>
      vector.transfer_write %t, %out[%i] {{in_bounds = [true]}} : vector<64xbf16>, memref<128xbf16>""",
            "%out: memref<128xbf16>",
            "bf16",
        ),
    ],
    ids=["sum", "sum-truncated"],
)
def test_row_results_are_stored_as_columns(body: str, args: str, element: str) -> None:
    count, _, text = _rewrite(_gpu_module(body, args))
    assert count == 1
    assert (
        f"memref.expand_shape %arg2 [[0, 1]] output_shape [128, 1] : "
        f"memref<128x{element}> into memref<128x1x{element}>"
    ) in text
    cast = re.search(
        rf"(%\w+) = vector\.shape_cast %\w+ : vector<64x{element}> to "
        rf"vector<64x1x{element}>",
        text,
    )
    assert cast, text
    assert re.search(
        rf"vector\.transfer_write {cast.group(1)}, %\w+\[%arg1, %\w+\] "
        rf"\{{in_bounds = \[true, true\]\}} : vector<64x1x{element}>, "
        rf"memref<128x1x{element}>",
        text,
    ), text
    assert f": vector<64x{element}>, memref<128x{element}>" not in text


_RUNNING = """%r = scf.for %c = %i to %i step %i iter_args(%a = %init) -> (vector<64xf32>) {{
        {reduce}
        %m = arith.maximumf %a, %s : vector<64xf32>
        scf.yield {yielded} : vector<64xf32>
      }}"""


@pytest.mark.parametrize(
    ("body", "args"),
    [
        # A running maximum over column tiles (the store takes the loop's result).
        (
            _RUNNING.format(reduce=_SUM, yielded="%m")
            + """
      vector.transfer_write %r, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
        ),
        # Through an op after the loop.
        (
            _RUNNING.format(reduce=_SUM, yielded="%m")
            + """
      %d = arith.addf %r, %r : vector<64xf32>
      vector.transfer_write %d, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
        ),
        # A loop in a loop.
        (
            """%o = scf.for %e = %i to %i step %i iter_args(%b = %init) -> (vector<64xf32>) {
        """
            + _RUNNING.format(reduce=_SUM, yielded="%m")
            + """
        scf.yield %r : vector<64xf32>
      }
      vector.transfer_write %o, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
        ),
    ],
    ids=["running", "running-then-op", "nested-loops"],
)
def test_loop_carried_row_results_are_stored_as_columns(body: str, args: str) -> None:
    count, _, text = _rewrite(_gpu_module(body, args))
    assert count == 1, text
    module = parse(text)
    writes = ops_named(module, "vector.transfer_write")
    stored = {str(write.operands[1].type) for write in writes}
    assert stored == {"memref<128x1xf32>"}, text


_TWO_ITER_ARGS = """%r:2 = scf.for %c = %i to %i step %i iter_args(%a = %init, %b = %init) -> (vector<64xf32>, vector<64xf32>) {{
        %v = vector.transfer_read %out[%i], %f {{in_bounds = [true]}} : memref<128xf32>, vector<64xf32>
        {reduce}
        scf.yield {yields} : vector<64xf32>, vector<64xf32>
      }}
      vector.transfer_write %r#0, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>
      vector.transfer_write %r#1, %out2[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>"""


@pytest.mark.parametrize(
    ("yields", "rewritten"),
    [("%s, %v", "%arg2"), ("%v, %s", "%arg4")],
    ids=["first-is-row", "second-is-row"],
)
def test_several_iter_args_follow_the_result_number(
    yields: str, rewritten: str
) -> None:
    """Only the loop result whose own yield is a row result is rewritten."""
    args = "%out: memref<128xf32>, %f: f32, %out2: memref<128xf32>"
    count, _, text = _rewrite(
        _gpu_module(_TWO_ITER_ARGS.format(reduce=_SUM, yields=yields), args)
    )
    assert count == 1, text
    expanded = re.findall(r"memref\.expand_shape (%arg\d+)", text)
    assert expanded == [rewritten], text


def test_both_iter_args_swapped_are_rewritten() -> None:
    """Row results that swap iter args are row results either way."""
    body = f"""%r:2 = scf.for %c = %i to %i step %i iter_args(%a = %init, %b = %init) -> (vector<64xf32>, vector<64xf32>) {{{{
        {_SUM}
        %m = arith.maximumf %b, %s : vector<64xf32>
        %n = arith.addf %a, %s : vector<64xf32>
        scf.yield %m, %n : vector<64xf32>, vector<64xf32>
      }}}}
      vector.transfer_write %r#0, %out[%i] {{{{in_bounds = [true]}}}} : vector<64xf32>, memref<128xf32>
      vector.transfer_write %r#1, %out2[%i] {{{{in_bounds = [true]}}}} : vector<64xf32>, memref<128xf32>""".format()
    count, _, text = _rewrite(
        _gpu_module(body, "%out: memref<128xf32>, %out2: memref<128xf32>")
    )
    assert count == 2, text


def test_a_deep_chain_with_no_row_result_is_quick() -> None:
    """A value used twice by each op of a chain: every path is looked at once
    (2^24 paths otherwise, minutes)."""
    body = "%v0 = vector.transfer_read %out[%i], %f {in_bounds = [true]} : memref<128xf32>, vector<64xf32>\n"
    for depth in range(24):
        body += f"%v{depth + 1} = arith.addf %v{depth}, %v{depth} : vector<64xf32>\n"
    body += "vector.transfer_write %v24, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>"
    with visits(keep_row_dims_module, "_compute_row_result", limit=500) as counts:
        count, before, after = _rewrite(
            _gpu_module(body, "%out: memref<128xf32>, %f: f32")
        )
    assert count == 0
    assert after == before
    assert counts["_compute_row_result"] >= 24


def test_contiguous_strided_buffer_is_stored_as_a_column() -> None:
    """A rank-3+ kernel's row buffer: a collapsed unit tile, stride 1 and a
    dynamic offset."""
    args = "%out: memref<128xf32, strided<[1], offset: ?>>"
    body = f"""{_SUM}
      vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32, strided<[1], offset: ?>>"""
    count, _, text = _rewrite(_gpu_module(body, args))
    assert count == 1, text
    assert (
        "memref<128xf32, strided<[1], offset: ?>> into "
        "memref<128x1xf32, strided<[1, 1], offset: ?>>"
    ) in text


@pytest.mark.parametrize(
    ("body", "args"),
    [
        # A loop that yields no row result: its iter arg, unchanged.
        (
            _RUNNING.format(reduce=_SUM, yielded="%a")
            + """
      vector.transfer_write %r, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
        ),
        # A loop that yields a value read from memory.
        (
            """%r = scf.for %c = %i to %i step %i iter_args(%a = %init) -> (vector<64xf32>) {
        %v = vector.transfer_read %out[%i], %f {in_bounds = [true]} : memref<128xf32>, vector<64xf32>
        scf.yield %v : vector<64xf32>
      }
      vector.transfer_write %r, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>, %f: f32",
        ),
        # Not a row result: a value read from memory.
        (
            """%v = vector.transfer_read %out[%i], %c {in_bounds = [true]} : memref<128xf32>, vector<64xf32>
      vector.transfer_write %v, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>, %c: f32",
        ),
        # A reduction over the outer dim (columns), not rows.
        (
            """%s = vector.multi_reduction <add>, %x, %init [0] : vector<64x64xf32> to vector<64xf32>
      vector.transfer_write %s, %out[%i] {in_bounds = [true]} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
        ),
        # Masked.
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i], %m {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>, %m: vector<64xi1>",
        ),
        # A strided buffer.
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32, strided<[2]>>""",
            "%out: memref<128xf32, strided<[2]>>",
        ),
        # A buffer of runtime size (expand_shape would need its size).
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<?xf32>""",
            "%out: memref<?xf32>",
        ),
        # A rank-2 buffer (a row of a matrix).
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i, %i] {{in_bounds = [true]}} : vector<64xf32>, memref<128x128xf32>""",
            "%out: memref<128x128xf32>",
        ),
    ],
    ids=[
        "loop-yields-iter-arg",
        "loop-yields-load",
        "not-reduced",
        "column-reduction",
        "masked",
        "strided",
        "dynamic",
        "rank-2",
    ],
)
def test_other_stores_are_untouched(body: str, args: str) -> None:
    count, before, after = _rewrite(_gpu_module(body, args))
    assert count == 0
    assert after == before


def test_out_of_bounds_stays_out_of_bounds() -> None:
    """A partial tile's row results past the output's end are not stored."""
    body = f"""{_SUM}
    vector.transfer_write %s, %out[%i] {{in_bounds = [false]}} : vector<64xf32>, memref<125xf32>"""
    count, _, after = _rewrite(_gpu_module(body, "%out: memref<125xf32>"))
    assert count == 1
    assert "in_bounds = [false, true]" in after


def test_host_functions_are_untouched() -> None:
    body = f"""{_SUM}
    vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>"""
    count, before, after = _rewrite(_host_module(body))
    assert count == 0
    assert after == before
