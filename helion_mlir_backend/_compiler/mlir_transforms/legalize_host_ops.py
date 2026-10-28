"""``legalize_host_ops``: what a bufferized GPU entry function does outside its
kernels runs on the host, and its buffers are device memory. Copies and fills of
whole buffers become GPU runtime calls; any other host access to device memory,
and kernel arguments of a float type the device lacks (e.g. ``f64``), are
definite failures (``HOST_OPS_ERROR`` messages)."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import gpu
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .hoist_invariant_scalars import defined_outside
from .hoist_invariant_scalars import is_pure

HOST_OPS_ERROR = "legalize_host_ops"
"""Prefix of the failure messages: ``legalize_host_ops: <what>; <reason>``."""
HOST_MEMORY = "host code touches device memory"
KERNEL_ARGUMENT = "kernel argument of an unsupported type"

# Kernels: everything nested in them runs on the device.
_KERNELS = frozenset({"scf.forall", "scf.parallel", "gpu.launch"})
# Views of a buffer: operand 0 is the buffer viewed; no memory is accessed.
_VIEWS = frozenset(
    {
        "memref.cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.memory_space_cast",
        "memref.reinterpret_cast",
        "memref.subview",
        "memref.view",
    }
)
# Buffer metadata, read without accessing the buffer.
_METADATA = frozenset(
    {
        "memref.dim",
        "memref.extract_aligned_pointer_as_index",
        "memref.extract_strided_metadata",
        "memref.rank",
    }
)
# Control flow passing buffers along; nested regions are checked op by op.
_CONTROL_FLOW = frozenset(
    {
        "cf.br",
        "cf.cond_br",
        "func.call",
        "func.return",
        "scf.condition",
        "scf.execute_region",
        "scf.for",
        "scf.if",
        "scf.while",
        "scf.yield",
    }
)


def _is_buffer(value: ir.Value) -> bool:
    return isinstance(value.type, ir.MemRefType)


def _is_contiguous(value: ir.Value) -> bool:
    memref_type = ir.MemRefType(value.type)
    layout = memref_type.layout
    return isinstance(layout, ir.AffineMapAttr) and layout.value == (
        ir.AffineMap.get_identity(memref_type.rank)
    )


def _argument_name(func: ir.Operation, value: ir.Value) -> str:
    """How a failure names a kernel argument: the host expression (``helion.name``)
    of a function argument, else the op computing it."""
    if isinstance(value, ir.BlockArgument) and value.owner.owner == func:
        if "arg_attrs" in func.attributes:
            attrs = ir.DictAttr(
                ir.ArrayAttr(func.attributes["arg_attrs"])[value.arg_number]
            )
            if "helion.name" in attrs:
                return f"'{ir.StringAttr(attrs['helion.name']).value}'"
        return f"argument {value.arg_number}"
    if isinstance(value, ir.OpResult):
        return f"the result of '{value.owner.name}'"
    return "a block argument"


class _Legalizer:
    """Checks and rewrites one function's host code."""

    def __init__(self, func: ir.Operation, max_float_width: int) -> None:
        self.func = func
        self.max_float_width = max_float_width
        self.copies: list[ir.Operation] = []
        self.fills: list[ir.Operation] = []
        self.failure: tuple[ir.Operation, str, str] | None = None

    def fail(self, op: ir.Operation, what: str, reason: str) -> None:
        if self.failure is None:
            self.failure = (op, what, reason)

    def visit(self, block: ir.Block) -> None:
        for op in block.operations:
            op = op.operation
            if op.name in _KERNELS:
                self.check_kernel_arguments(op)
                continue
            self.check(op)
            if op.name in _CONTROL_FLOW:
                for region in op.regions:
                    for nested in region.blocks:
                        self.visit(nested)

    def check(self, op: ir.Operation) -> None:
        name = op.name
        buffers = [value for value in op.operands if _is_buffer(value)]
        created = [value for value in op.results if _is_buffer(value)]
        if not buffers and not created:
            return  # Scalar or vector code: no memory involved.
        if name.startswith("gpu.") or name in _METADATA or name in _CONTROL_FLOW:
            return
        if name in _VIEWS:
            return
        if name == "memref.copy" and _whole_buffers(op.operands):
            self.copies.append(op)
            return
        if (
            name == "linalg.fill"
            and len(buffers) == 1
            and _whole_buffers(buffers)
            and _fill_width(op) in (16, 32)
        ):
            self.fills.append(op)
            return
        if created and not buffers:
            self.fail(
                op,
                HOST_MEMORY,
                f"'{name}' creates a host buffer the kernels cannot access",
            )
            return
        self.fail(
            op,
            HOST_MEMORY,
            f"'{name}' outside the kernels accesses a device buffer from the host",
        )

    def check_kernel_arguments(self, kernel: ir.Operation) -> None:
        """The host values ``kernel`` uses become its arguments: floats wider than
        the device's (``max_float_width``) fail. Constants do not count: outlining
        rematerializes them in the kernel."""
        failure: list[tuple[ir.Operation, ir.Value]] = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            for value in op.operands:
                if (
                    isinstance(value.type, ir.FloatType)
                    and value.type.width > self.max_float_width
                    and defined_outside(value, kernel)
                    and not _is_constant(value)
                ):
                    failure.append((op, value))
                    return ir.WalkResult.INTERRUPT
            return ir.WalkResult.ADVANCE

        kernel.walk(visit, ir.WalkOrder.PRE_ORDER)
        if failure:
            op, value = failure[0]
            self.fail(
                op,
                KERNEL_ARGUMENT,
                f"{_argument_name(self.func, value)} ({value.type}) is used by "
                f"'{op.name}' in a kernel, which would take it as an argument; the "
                f"device has no floats wider than {self.max_float_width} bits",
            )

    def rewrite(self, rewriter: transform.TransformRewriter) -> None:
        for copy in self.copies:
            source, dest = copy.operands
            with ir.InsertionPoint(copy), copy.location:
                gpu.MemcpyOp(None, [], dest, source)
            rewriter.erase_op(copy)
        for fill in self.fills:
            value = fill.operands[0]
            dest = fill.operands[1]
            with ir.InsertionPoint(fill), fill.location:
                gpu.MemsetOp(None, [], dest, value)
            rewriter.erase_op(fill)


def _whole_buffers(values: list[ir.Value]) -> bool:
    return all(_is_buffer(value) and _is_contiguous(value) for value in values)


def _is_constant(value: ir.Value) -> bool:
    """A result of an op with no operands and no effects (e.g. ``arith.constant``)."""
    return (
        isinstance(value, ir.OpResult)
        and not list(value.owner.operands)
        and is_pure(value.owner.operation)
    )


def _fill_width(fill: ir.Operation) -> int | None:
    value_type = fill.operands[0].type
    if isinstance(value_type, (ir.FloatType, ir.IntegerType)):
        return value_type.width
    return None


@transform_op(modifies_payload=True)
class LegalizeHostOpsOp(HelionTransformDialect.Operation, name="legalize_host_ops"):
    """Legalize the host code of each bufferized target function: the code outside
    its kernels (``scf.forall``, ``scf.parallel``, ``gpu.launch``), which runs on
    the host while its buffers are device memory.

    - A ``memref.copy`` between whole (identity-layout) device buffers becomes a
      ``gpu.memcpy``; a ``linalg.fill`` of one with 16- or 32-bit values a
      ``gpu.memset``.
    - Scalar and vector code, views and metadata of buffers, ``gpu`` ops, control
      flow and calls stay. Scalars (e.g. runtime scalars, function arguments by
      value) are computed on the host; outlining passes the values the kernels
      use as kernel arguments.
    - A kernel using a float wider than ``max_float_width`` bits computed on the
      host (it would take it as an argument; constants excepted) is a definite
      failure: hoist conversions to the kernel's types out of it first
      (``hoist_invariant_scalars``).
    - Any other op reading, writing or creating a buffer is a definite failure.
    """

    target: ext.Operand[transform.AnyOpType]
    max_float_width: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 32)
    )

    @staticmethod
    def run(
        op: "LegalizeHostOpsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for func in state.get_payload_ops(op.target):
            func = func.operation
            if not func.regions or not func.regions[0].blocks:
                continue
            legalizer = _Legalizer(func, op.max_float_width.value)
            legalizer.visit(func.regions[0].blocks[0])
            if legalizer.failure is not None:
                failed, what, reason = legalizer.failure
                failed.location.emit_error(f"{HOST_OPS_ERROR}: {what}; {reason}")
                return DiagnosedSilenceableFailure.DefiniteFailure
            legalizer.rewrite(rewriter)
        return DiagnosedSilenceableFailure.Success


def legalize_host_ops(target: ir.Value, max_float_width: int = 32) -> LegalizeHostOpsOp:
    """snake_case wrapper to create a LegalizeHostOpsOp."""
    i64 = ir.IntegerType.get_signless(64)
    return LegalizeHostOpsOp(
        target=target, max_float_width=ir.IntegerAttr.get(i64, max_float_width)
    )
