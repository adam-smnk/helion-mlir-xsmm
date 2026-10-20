"""The transform dialect extension of the Helion MLIR backend.

A Python-defined dialect only registers the ops defined before it is loaded:
load it after importing the op modules (see the package ``__init__``).
"""

from collections.abc import Callable

from lighthouse.dialects import DialectExtension
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure


class HelionTransformDialect(DialectExtension, name="helion_transform"):
    """Transform ops of the Helion MLIR backend's pipelines."""


def transform_op(*, modifies_payload: bool) -> Callable[[type], type]:
    """Attach the interfaces of a transform op applied by its static ``run``.

    An op modifying the payload produces no handles; others only read it.
    """

    def decorate(cls: type) -> type:
        class Transform(transform.TransformOpInterface):
            @staticmethod
            def apply(
                op: ir.OpView,
                rewriter: transform.TransformRewriter,
                results: transform.TransformResults,
                state: transform.TransformState,
            ) -> DiagnosedSilenceableFailure:
                return cls.run(op, rewriter, results, state)

            @staticmethod
            def allow_repeated_handle_operands(_op: ir.OpView) -> bool:
                return False

        class Effects(ir.MemoryEffectsOpInterface):
            @staticmethod
            def get_effects(op: ir.OpView) -> list:
                effects = transform.only_reads_handle(op.op_operands)
                if modifies_payload:
                    return effects + transform.modifies_payload()
                return (
                    effects
                    + transform.produces_handle(op.results)
                    + transform.only_reads_payload()
                )

        def attach_interface_impls(context: ir.Context | None = None) -> None:
            Transform.attach(cls.OPERATION_NAME, context=context)
            Effects.attach(cls.OPERATION_NAME, context=context)

        cls.attach_interface_impls = staticmethod(attach_interface_impls)
        return cls

    return decorate
