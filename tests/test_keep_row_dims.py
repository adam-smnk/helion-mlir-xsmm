"""``keep_row_dims``: a GPU kernel's ``[rows]`` row-reduction results stored into a
rank-1 buffer become ``[rows, 1]`` stores into the buffer expanded to
``[n, 1]``; other stores are left alone. IR only."""

from __future__ import annotations

import re

from mlir import ir
import pytest

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


@pytest.mark.parametrize(
    ("body", "args"),
    [
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
        # Not in bounds.
        (
            f"""{_SUM}
      vector.transfer_write %s, %out[%i] {{in_bounds = [false]}} : vector<64xf32>, memref<128xf32>""",
            "%out: memref<128xf32>",
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
        "not-reduced",
        "column-reduction",
        "masked",
        "out-of-bounds",
        "strided",
        "dynamic",
        "rank-2",
    ],
)
def test_other_stores_are_untouched(body: str, args: str) -> None:
    count, before, after = _rewrite(_gpu_module(body, args))
    assert count == 0
    assert after == before


def test_host_functions_are_untouched() -> None:
    body = f"""{_SUM}
    vector.transfer_write %s, %out[%i] {{in_bounds = [true]}} : vector<64xf32>, memref<128xf32>"""
    count, before, after = _rewrite(_host_module(body))
    assert count == 0
    assert after == before
