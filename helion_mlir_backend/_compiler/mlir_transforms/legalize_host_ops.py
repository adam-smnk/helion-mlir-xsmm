"""``legalize_host_ops``: what a bufferized GPU entry function does outside its
kernels runs on the host, and its buffers are device memory. Copies and fills of
whole buffers become GPU runtime calls; any other host access to device memory,
host buffers the kernels would read, and runtime scalars read in a kernel are
definite failures (``HOST_OPS_ERROR`` messages)."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import gpu
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

HOST_OPS_ERROR = "legalize_host_ops"
"""Prefix of the failure messages: ``legalize_host_ops: <what>; <reason>``."""
HOST_MEMORY = "host code touches device memory"
DEVICE_SCALAR = "runtime scalar read on the device"

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


def _root(value: ir.Value) -> ir.Value:
    """The buffer ``value`` is a view of (through ``_VIEWS``)."""
    while isinstance(value, ir.OpResult) and value.owner.name in _VIEWS:
        value = value.owner.operands[0]
    return value


def _in_kernel(op: ir.Operation) -> bool:
    parent = op.parent
    while parent is not None and parent.name != "func.func":
        if parent.name in _KERNELS:
            return True
        parent = parent.parent
    return False


def _is_contiguous(value: ir.Value) -> bool:
    memref_type = ir.MemRefType(value.type)
    layout = memref_type.layout
    return isinstance(layout, ir.AffineMapAttr) and layout.value == (
        ir.AffineMap.get_identity(memref_type.rank)
    )


def _host_args(func: ir.Operation, role_attr: str, role: str) -> dict[ir.Value, str]:
    """The function's arguments whose ``role_attr`` is ``role``: host buffers, by
    the name of the value they pass (``helion.name``, else ``argN``)."""
    if "arg_attrs" not in func.attributes:
        return {}
    args = func.regions[0].blocks[0].arguments
    found = {}
    for index, attrs in enumerate(ir.ArrayAttr(func.attributes["arg_attrs"])):
        attrs = ir.DictAttr(attrs)
        if role_attr in attrs and ir.StringAttr(attrs[role_attr]).value == role:
            name = (
                ir.StringAttr(attrs["helion.name"]).value
                if "helion.name" in attrs
                else f"arg{index}"
            )
            found[args[index]] = name
    return found


class _Legalizer:
    """Checks and rewrites one function's host code."""

    def __init__(self, host_args: dict[ir.Value, str]) -> None:
        self.host_args = host_args
        self.copies: list[ir.Operation] = []
        self.fills: list[ir.Operation] = []
        self.failure: tuple[ir.Operation, str, str] | None = None

    def fail(self, op: ir.Operation, what: str, reason: str) -> None:
        if self.failure is None:
            self.failure = (op, what, reason)

    def is_host(self, value: ir.Value) -> bool:
        return _root(value) in self.host_args

    def visit(self, block: ir.Block) -> None:
        for op in block.operations:
            op = op.operation
            if op.name in _KERNELS:
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
        if name == "memref.load" and self.is_host(op.operands[0]):
            return  # A runtime scalar, read on the host.
        if name == "memref.copy" and self._whole_device_buffers(op.operands):
            self.copies.append(op)
            return
        if (
            name == "linalg.fill"
            and len(buffers) == 1
            and self._whole_device_buffers(buffers)
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

    def _whole_device_buffers(self, values: list[ir.Value]) -> bool:
        return all(
            _is_buffer(value) and not self.is_host(value) and _is_contiguous(value)
            for value in values
        )

    def check_host_args(self) -> None:
        """Host buffers are only read outside the kernels."""
        for arg, name in self.host_args.items():
            pending = [arg]
            while pending:
                value = pending.pop()
                for use in value.uses:
                    owner = use.owner
                    if _in_kernel(owner) or owner.name == "gpu.launch_func":
                        self.fail(
                            owner,
                            DEVICE_SCALAR,
                            f"'{name}' is a host buffer, read by '{owner.name}' in a "
                            "kernel; runtime scalars must be read outside the grid "
                            "loops",
                        )
                        return
                    if owner.name in _VIEWS:
                        pending.append(owner.results[0])

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
      flow and calls stay.
    - Arguments whose ``role_attr`` attribute is ``host_role`` are host buffers
      (runtime scalars): loads of them on the host stay (outlining passes the
      values to the kernels), any use in a kernel is a definite failure.
    - Any other op reading, writing or creating a buffer is a definite failure.
    """

    target: ext.Operand[transform.AnyOpType]
    role_attr: ir.StringAttr = ext.attribute(
        default_factory=lambda: ir.StringAttr.get("helion.role")
    )
    host_role: ir.StringAttr = ext.attribute(
        default_factory=lambda: ir.StringAttr.get("scalar")
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
            host_args = _host_args(func, op.role_attr.value, op.host_role.value)
            legalizer = _Legalizer(host_args)
            legalizer.check_host_args()
            legalizer.visit(func.regions[0].blocks[0])
            if legalizer.failure is not None:
                failed, what, reason = legalizer.failure
                failed.location.emit_error(f"{HOST_OPS_ERROR}: {what}; {reason}")
                return DiagnosedSilenceableFailure.DefiniteFailure
            legalizer.rewrite(rewriter)
        return DiagnosedSilenceableFailure.Success


def legalize_host_ops(
    target: ir.Value, role_attr: str = "helion.role", host_role: str = "scalar"
) -> LegalizeHostOpsOp:
    """snake_case wrapper to create a LegalizeHostOpsOp."""
    return LegalizeHostOpsOp(
        target=target,
        role_attr=ir.StringAttr.get(role_attr),
        host_role=ir.StringAttr.get(host_role),
    )
