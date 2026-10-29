"""Python-defined transform ops of the Helion MLIR backend, one per module,
with snake_case wrappers to build them in schedules (see ``helion_transforms``).

The op modules are imported before the dialect is loaded: a Python-defined
dialect only registers the ops defined before it is loaded."""

from .align_operand_rows import align_operand_rows
from .carry_loop_inits import carry_loop_inits
from .dialect import HelionTransformDialect
from .fold_loop_unit_dims import fold_loop_unit_dims
from .forward_rows import forward_rows
from .hoist_allocas import hoist_allocas
from .hoist_invariant_scalars import hoist_invariant_scalars
from .legalize_for_llvm import legalize_for_llvm
from .legalize_host_ops import legalize_host_ops
from .loop_bounds import loop_bounds
from .mark_operand_packs import OPERAND_PACK_ATTR_NAME
from .mark_operand_packs import mark_operand_packs
from .match_scalar_contractions import match_scalar_contractions
from .materialize_copies import materialize_copies
from .materialize_operand_pads import materialize_operand_pads
from .move_offsets_to_subview import move_offsets_to_subview
from .outer_transpose_tile import outer_transpose_tile
from .partition_linalg import partition_linalg
from .pin_transposes import pin_transposes
from .rank_reduce_transfers import rank_reduce_transfers
from .schedule_amx_loads import schedule_amx_loads
from .split_bf16_loads import split_bf16_loads
from .split_contraction_operands import split_contraction_operands
from .split_copies import split_copies
from .split_transfers import split_transfers
from .tile_sizes import tile_sizes
from .unmask_contractions import unmask_contractions
from .vectorize_pads import vectorize_pads
from .version_padded_operands import version_padded_operands
from .widen_transposes import widen_transposes
from .xegpu_elementwise import InferXeGPUElementwiseLayoutOp
from .xegpu_elementwise import infer_xegpu_elementwise_layout
from .xegpu_reduction import infer_xegpu_reduction_tiles
from .xegpu_reduction import match_row_result_stores

__all__ = [
    "OPERAND_PACK_ATTR_NAME",
    "HelionTransformDialect",
    "InferXeGPUElementwiseLayoutOp",
    "align_operand_rows",
    "carry_loop_inits",
    "fold_loop_unit_dims",
    "forward_rows",
    "hoist_allocas",
    "hoist_invariant_scalars",
    "infer_xegpu_elementwise_layout",
    "infer_xegpu_reduction_tiles",
    "legalize_for_llvm",
    "legalize_host_ops",
    "loop_bounds",
    "mark_operand_packs",
    "match_row_result_stores",
    "match_scalar_contractions",
    "materialize_copies",
    "materialize_operand_pads",
    "move_offsets_to_subview",
    "outer_transpose_tile",
    "partition_linalg",
    "pin_transposes",
    "rank_reduce_transfers",
    "schedule_amx_loads",
    "split_bf16_loads",
    "split_contraction_operands",
    "split_copies",
    "split_transfers",
    "tile_sizes",
    "unmask_contractions",
    "vectorize_pads",
    "version_padded_operands",
    "widen_transposes",
]
