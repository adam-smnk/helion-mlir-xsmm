"""Helper requests, their lowering by torch-mlir, and the process-wide helper cache.

A :class:`HelperRequest` is an ATen call typed by its operands' MLIR types. The
cache lowers the requests it lacks in one torch-mlir run (each alone if that
run fails, to name the failing node) and keeps the lowered functions, which live
in the shared MLIR context, for every later module.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field
import hashlib
import operator
from typing import TYPE_CHECKING

import mlir.ir as ir
import torch
import torch.fx
from torch.fx.passes.shape_prop import TensorMetadata

from ..support import UnsupportedOperationError
from .samples import substitute

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch._ops import OpOverload


@dataclass(frozen=True)
class HelperRequest:
    target: OpOverload
    args: tuple
    kwargs: tuple[tuple[str, object], ...]
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]
    samples: tuple = field(compare=False, hash=False, repr=False)
    """Meta or fake tensor, or Python scalar, standing in for each operand."""
    result: object = field(compare=False, hash=False, repr=False)
    """The call's result (a tensor or a tuple of them) on ``samples``."""

    @property
    def name(self) -> str:
        key = repr((str(self.target), self.args, self.kwargs, self.operand_types))
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"_aten_{self.target.__name__.replace('.', '_')}_{digest}"

    @property
    def function_type(self) -> ir.FunctionType:
        return ir.FunctionType.get(
            [ir.Type.parse(text) for text in self.operand_types],
            [ir.Type.parse(text) for text in self.result_types],
        )


class HelperCache:
    """Lowered helper functions by name, and the modules that own them."""

    def __init__(self) -> None:
        self._functions: dict[str, ir.Operation] = {}
        self._modules: list[ir.Module] = []

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def __getitem__(self, name: str) -> ir.Operation:
        return self._functions[name]

    def lower(self, requests: list[tuple[HelperRequest, torch.fx.Node]]) -> None:
        """Lower ``requests`` (with the node each came from) into the cache."""
        try:
            self._add(_run_torch_mlir([request for request, _ in requests]))
            return
        except Exception:
            pass
        for request, node in requests:
            try:
                self._add(_run_torch_mlir([request]))
            except Exception as error:
                message = str(error).strip().splitlines()
                with located(node):
                    raise UnsupportedOperationError(
                        str(request.target),
                        reason=(
                            f"torch-mlir cannot lower it for operand types "
                            f"{list(request.operand_types)}: "
                            f"{message[0] if message else type(error).__name__}"
                        ),
                    ) from error

    def _add(self, helpers: ir.Module) -> None:
        self._modules.append(helpers)
        for operation in helpers.body.operations:
            if "sym_name" in operation.attributes:
                name = ir.StringAttr(operation.attributes["sym_name"]).value
                self._functions[name] = operation


CACHE = HelperCache()


@contextlib.contextmanager
def located(node: torch.fx.Node) -> Iterator[None]:
    """Attribute errors raised inside to ``node``'s kernel source line."""
    location = node.meta.get("location")
    with location if location is not None else contextlib.nullcontext():
        yield


def _run_torch_mlir(requests: list[HelperRequest]) -> ir.Module:
    """Import and lower ``requests`` with torch-mlir; the result lives in this context."""
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.compiler_utils import lower_mlir_module
    from torch_mlir.compiler_utils import run_pipeline_with_repro_report
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter
    import torch_mlir.ir as tm_ir

    context = tm_ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    for request in requests:
        importer.import_stateless_graph(_fx_graph(request), func_name=request.name)
    run_pipeline_with_repro_report(
        importer.module,
        "builtin.module(func.func(torch-match-quantized-custom-ops),"
        " torchdynamo-export-to-torch-backend-pipeline{})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, importer.module)
    return ir.Module.parse(
        importer.module.operation.get_asm(binary=False, enable_debug_info=False)
    )


def _fx_graph(request: HelperRequest) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    placeholders = []
    for index, sample in enumerate(request.samples):
        placeholder = graph.placeholder(f"arg{index}")
        _set_meta(placeholder, sample)
        placeholders.append(placeholder)
    args = substitute(request.args, placeholders)
    kwargs = substitute(dict(request.kwargs), placeholders)
    call = graph.call_function(request.target, args, kwargs)
    if isinstance(request.result, torch.Tensor):
        _set_meta(call, request.result)
        graph.output((call,))
        return graph
    call.meta["val"] = tuple(request.result)
    outputs = []
    for index, result in enumerate(request.result):
        item = graph.call_function(operator.getitem, (call, index))
        _set_meta(item, result)
        outputs.append(item)
    graph.output(tuple(outputs))
    return graph


def _set_meta(node: torch.fx.Node, value: object) -> None:
    node.meta["val"] = value
    if isinstance(value, torch.Tensor):
        node.meta["tensor_meta"] = TensorMetadata(
            value.shape, value.dtype, False, value.stride(), None, False, {}
        )
