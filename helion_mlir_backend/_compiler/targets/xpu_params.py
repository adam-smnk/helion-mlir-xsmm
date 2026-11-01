"""XeGPU lowering parameters from the Helion config, per GPU kernel (XPU target).

The keys hold one entry per GPU kernel, i.e. per top-level grid loop of the
Helion kernel in program order (across ``hl.barrier`` phases). An entry, or a
whole key, that is ``None`` or missing -- as are the kernels past the end of a
shorter list -- leaves the parameter to lighthouse's inference:

- ``xegpu_sg_tiles``: the subgroup tile, ``[sg_m, sg_n]`` of a GEMM or 2D
  elementwise kernel; ``[rows, 0]`` of a (2D, row) reduction kernel or of an
  attention kernel (its query rows; the only parameter attention kernels take:
  lighthouse's attention layout fixes their load and prefetch tiles).
- ``xegpu_load_tiles``: ``[a_m, a_k, b_k, b_n]``, the A and B tiles a GEMM
  kernel's subgroups load per instruction (``inst_data`` of the loads);
  ``[m, n]``, the tile a 2D elementwise or (2D, row) reduction kernel's
  subgroups load per instruction (a reduction: of the tiles it reduces; the
  stores take the largest 2D block store dividing the subgroup tile).
- ``xegpu_prefetch_tiles``: ``[a_m, a_k, b_k, b_n]``, the A and B tiles a GEMM
  kernel's subgroups prefetch, in the orientation the operand is stored in (a
  transposed B prefetches ``[n, k]`` tiles).
- ``xegpu_prefetch_depths``: ``[a_nb, b_nb]``, how many K-loop iterations ahead a
  GEMM kernel prefetches A and B.

The workgroup tile and the K tile are not parameters: they are the kernel's
block sizes (the grid tiles and the inner K tile), which lighthouse reads off
the IR. Elementwise kernels (no lighthouse class) infer their parameters from
the workgroup tile and element types
(``mlir_transforms.xegpu_elementwise.elementwise_layout``); 1D ones take no
parameters. Reduction kernels take lighthouse's subgroup tile and infer their
load tile the same way from it
(``mlir_transforms.xegpu_reduction.reduction_tiles``). Attention kernels
(:func:`attention_shape`) take lighthouse's attention layout, whose workgroup
rows and K/V tile are the block sizes.

:func:`xegpu_params` reads the config; :func:`attach_params` records the result
on the generated module (so modules compiled with different parameters differ,
also as JIT cache keys) and :func:`take_params` reads it back for lowering.
:func:`check_params` validates parameters against the kernels' shapes with
lighthouse's ``matmul_constraints`` (elementwise and reduction kernels: the
hardware limits of ``elementwise_layout`` and ``reduction_tiles``) before
lowering (also the GEMM and reduction tiles the block sizes imply when nothing
is forced), so configs that cannot lower fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import TYPE_CHECKING

from lighthouse.schedule.xegpu import matmul_constraints
from lighthouse.schedule.xegpu import matmul_costmodel
from lighthouse.schedule.xegpu.xegpu_parameter_selector import XeGPUParameterSelector
from lighthouse.schedule.xegpu.xegpu_specs import XeGPUSpecs
from mlir import ir
from mlir.dialects import linalg

from ..mlir.support.errors import XeGPUConfigError
from ..mlir_transforms.xegpu_elementwise import BLOCK_COLS
from ..mlir_transforms.xegpu_elementwise import ElementwiseLayout
from ..mlir_transforms.xegpu_elementwise import elementwise_layout
from ..mlir_transforms.xegpu_reduction import reduction_tiles

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

SG_TILES = "xegpu_sg_tiles"
LOAD_TILES = "xegpu_load_tiles"
PREFETCH_TILES = "xegpu_prefetch_tiles"
PREFETCH_DEPTHS = "xegpu_prefetch_depths"
_ARITY = {SG_TILES: 2, LOAD_TILES: 4, PREFETCH_TILES: 4, PREFETCH_DEPTHS: 2}
_ELEMENTWISE_ARITY = {LOAD_TILES: (2, 1)}
"""Other entry lengths: ``[m, n]`` load tiles of 2D elementwise and reduction
kernels, ``[n]`` of 1D elementwise kernels."""
XEGPU_CONFIG_KEYS = frozenset(_ARITY)
"""The config keys of the XPU target's lowering."""
_FIELDS = {
    SG_TILES: "sg_tile",
    LOAD_TILES: "load_tiles",
    PREFETCH_TILES: "prefetch_tiles",
    PREFETCH_DEPTHS: "prefetch_depths",
}
PARAMS_ATTR = "helion.xegpu_params"
"""Module attribute holding the parameters: one dictionary per kernel."""
_GEMM_ONLY = (PREFETCH_TILES, PREFETCH_DEPTHS)


@dataclass(frozen=True)
class KernelParams:
    """The forced XeGPU parameters of one GPU kernel (``None``: inferred)."""

    sg_tile: tuple[int, ...] | None = None
    load_tiles: tuple[int, ...] | None = None
    prefetch_tiles: tuple[int, ...] | None = None
    prefetch_depths: tuple[int, ...] | None = None

    def forced(self) -> dict[str, tuple[int, ...]]:
        """The forced parameters, by config key."""
        return {
            key: value
            for key, name in _FIELDS.items()
            if (value := getattr(self, name)) is not None
        }

    def annotate_options(self) -> dict[str, object]:
        """Keyword arguments of lighthouse's ``analyze_and_annotate_gpu_func``."""
        options: dict[str, object] = {"sg_tile": self.sg_tile}
        if self.load_tiles is not None and len(self.load_tiles) == 4:
            options["load_tile_a"] = self.load_tiles[:2]
            options["load_tile_b"] = self.load_tiles[2:]
        if self.prefetch_tiles is not None:
            options["prefetch_tile_a"] = self.prefetch_tiles[:2]
            options["prefetch_tile_b"] = self.prefetch_tiles[2:]
        if self.prefetch_depths is not None:
            options["prefetch_a_nb"], options["prefetch_b_nb"] = self.prefetch_depths
        return options

    def block_load_tile(self) -> tuple[int, ...] | None:
        """The forced load tile of an elementwise or reduction kernel (a
        2-entry load tile, or a 1-entry one of a 1D elementwise kernel)."""
        if self.load_tiles is not None and len(self.load_tiles) in (1, 2):
            return self.load_tiles
        return None


def xegpu_params(config: Mapping[str, object]) -> tuple[KernelParams, ...]:
    """The per-kernel parameters of ``config``'s ``xegpu_*`` keys, as long as the
    longest list (empty without keys); malformed values raise
    :class:`XeGPUConfigError`."""
    entries: dict[str, list[tuple[int, ...] | None]] = {}
    for key, arity in _ARITY.items():
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise XeGPUConfigError(
                f"{key}={value!r}: expected a list with one entry per GPU kernel "
                f"(top-level grid loop): a list of {arity} ints, or None"
            )
        entries[key] = [_entry(key, index, item) for index, item in enumerate(value)]
    count = max((len(values) for values in entries.values()), default=0)
    return tuple(
        KernelParams(
            **{
                _FIELDS[key]: values[index] if index < len(values) else None
                for key, values in entries.items()
            }
        )
        for index in range(count)
    )


def _entry(key: str, index: int, item: object) -> tuple[int, ...] | None:
    if item is None:
        return None
    arity = _ARITY[key]
    arities = {arity, *_ELEMENTWISE_ARITY.get(key, ())}
    if (
        not isinstance(item, (list, tuple))
        or len(item) not in arities
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in item
        )
    ):
        elementwise = (
            " (2 for an elementwise or reduction kernel, 1 for a 1D elementwise kernel)"
            if key in _ELEMENTWISE_ARITY
            else ""
        )
        raise XeGPUConfigError(
            f"{key}[{index}]={item!r}: expected {arity} non-negative ints"
            f"{elementwise} or None (one entry per GPU kernel)"
        )
    return tuple(item)


def attach_params(module: ir.Module, params: Sequence[KernelParams]) -> None:
    """Record ``params`` on ``module`` (nothing for none)."""
    if not params:
        return
    with module.context:
        kernels = [
            ir.DictAttr.get(
                {
                    _FIELDS[key]: ir.DenseI64ArrayAttr.get(list(value))
                    for key, value in kernel.forced().items()
                }
            )
            for kernel in params
        ]
        module.operation.attributes[PARAMS_ATTR] = ir.ArrayAttr.get(kernels)


def take_params(module: ir.Module) -> tuple[KernelParams, ...]:
    """The parameters :func:`attach_params` recorded on ``module``, removed from it."""
    attributes = module.operation.attributes
    if PARAMS_ATTR not in attributes:
        return ()
    kernels = ir.ArrayAttr(attributes[PARAMS_ATTR])
    del attributes[PARAMS_ATTR]
    names = {field.name for field in fields(KernelParams)}
    params = []
    for kernel in kernels:
        kernel = ir.DictAttr(kernel)
        params.append(
            KernelParams(
                **{
                    name: tuple(ir.DenseI64ArrayAttr(kernel[name]))
                    for name in names
                    if name in kernel
                }
            )
        )
    return tuple(params)


# --- Kernel shapes -------------------------------------------------------------


@dataclass(frozen=True)
class GemmShape:
    """The GEMM of one kernel: the matmul's sizes and its tiles."""

    mnk: tuple[int, int, int] | None
    """Global (M, N, K), if the operands' sources are known."""
    wg_tile: tuple[int, int]
    k_tile: int
    transpose_a: bool
    transpose_b: bool

    def __str__(self) -> str:
        flags = [
            name
            for name, on in (("A", self.transpose_a), ("B", self.transpose_b))
            if on
        ]
        transposed = f", transposed {' and '.join(flags)}" if flags else ""
        sizes = "" if self.mnk is None else "M={}, N={}, K={}, ".format(*self.mnk)
        return (
            f"GEMM ({sizes}WG tile {self.wg_tile[0]}x{self.wg_tile[1]}, "
            f"K tile {self.k_tile}{transposed})"
        )


@dataclass(frozen=True)
class ReductionShape:
    """The (first) reduction of one kernel."""

    rows: int | None
    """The rows a workgroup reduces (2D row reductions), else ``None``."""
    cols: int | None = None
    """The columns of the tiles it reduces (2D row reductions), else ``None``."""
    load_bits: tuple[int, ...] = ()
    """The element sizes of the ``[rows, cols]`` tiles the kernel loads."""

    def __str__(self) -> str:
        rows = "" if self.rows is None else f" of {self.rows} rows per workgroup"
        return f"reduction{rows}"


@dataclass(frozen=True)
class ElementwiseShape:
    """The tiles of an elementwise kernel (no contraction, no reduction)."""

    wg_tile: tuple[int, ...] | None
    """The tile every store writes, ``None`` if the stores disagree."""
    load_bits: tuple[int, ...]
    """The element sizes of the tiles loaded."""
    store_bits: tuple[int, ...]
    """The element sizes of the tiles stored."""

    def layout(
        self,
        sg_tile: Sequence[int] | None = None,
        load_tile: Sequence[int] | None = None,
        contiguous: bool = False,
    ) -> ElementwiseLayout:
        """The kernel's XeGPU layout (``ValueError`` if none fits)."""
        if self.wg_tile is None:
            raise ValueError("the grid loop stores tiles of several shapes")
        return elementwise_layout(
            self.wg_tile,
            self.load_bits,
            self.store_bits,
            sg_tile=sg_tile,
            load_tile=load_tile,
            contiguous=contiguous,
        )

    def __str__(self) -> str:
        if self.wg_tile is None:
            return "elementwise kernel"
        tile = "x".join(map(str, self.wg_tile))
        return f"elementwise kernel (WG tile {tile})"


ATTENTION_SG_ROWS = 16
"""The query rows of lighthouse's attention subgroup tile
(``infer_xegpu_attention_params``: two 8-row DPAS blocks)."""
ATTENTION_D_HEAD = 64
"""The head dim lighthouse's attention annotation lays out: ``annotate_attention``
hard-codes it, ``infer_xegpu_attention_params`` asserts it."""
ATTENTION_KV_ROWS = 32
"""K/V tiles are whole multiples of the 32x32 block tiles lighthouse's attention
annotation loads V in (``inst_data``)."""


@dataclass(frozen=True)
class AttentionShape:
    """A flash-attention kernel: in a loop over K/V tiles, ``S = Q @ K^T`` on the
    workgroup's query rows, row reductions of ``S`` (an online softmax's maxima
    and sums), and ``O = O * alpha + P @ V`` accumulated across the loop, with
    ``P`` computed from ``S``. K and V are stored ``[seq, head dim]`` (``S``
    contracts a transposed K)."""

    rows: int
    """The query rows a workgroup computes."""
    kv_tile: int
    """The K/V rows of one loop step."""
    d_head: int
    """The head dim of Q and K (the contracted dim of ``Q @ K^T``)."""
    d_value: int
    """The head dim of V (the columns of the output)."""

    def __str__(self) -> str:
        return (
            f"flash-attention kernel ({self.rows} query rows per workgroup, K/V tile "
            f"{self.kv_tile}, head dim {self.d_head})"
        )


KernelShape = GemmShape | ReductionShape | ElementwiseShape | AttentionShape


def kernel_shape(loop: ir.Operation) -> KernelShape | None:
    """The GEMM, attention, reduction or elementwise tiles a grid loop computes
    (on tensors, after ``prepare``: unit tile dims are folded, so a kernel with
    block size 1 on leading dims has its 2D shape). Two contractions are an
    attention kernel if they have its structure (:func:`attention_shape`),
    else ``None``."""
    contractions: list[tuple[ir.Operation, list[ir.AffineMap]]] = []
    first_reduction = None
    for op in _walk(loop):
        view = op.opview
        maps = linalg.get_indexing_maps(view)
        if maps is None:
            continue
        affine_maps = [ir.AffineMapAttr(m).value for m in maps]
        if linalg.isa_contraction_op(view):
            contractions.append((op, affine_maps))
        elif first_reduction is None and _reduced_dims(op, affine_maps):
            first_reduction = (op, affine_maps)
    if len(contractions) == 1:
        return _gemm_shape(*contractions[0])
    if contractions:
        try:
            return attention_shape(loop, [op for op, _ in contractions])
        except ValueError:
            return None
    if first_reduction is None:
        return _elementwise_shape(loop)
    return _reduction_shape(loop, *first_reduction)


def attention_shape(
    loop: ir.Operation, contractions: Sequence[ir.Operation]
) -> AttentionShape:
    """The :class:`AttentionShape` of the grid loop ``loop`` whose contractions
    (linalg ops, in program order) are ``contractions``; ``ValueError`` naming
    the part of the structure that is missing.

    By IR property: two 2D contractions in one ``scf.for`` of the grid loop (the
    K/V loop); the first (``Q @ K^T``) contracts its B operand transposed; the
    second (``P @ V``) takes as A operand a value computed from the first's
    result through at least one reduction over its rows, and the loop carries
    its result (the output accumulator)."""
    if len(contractions) != 2:
        raise ValueError(
            f"{len(contractions)} contractions (attention has two: Q @ K^T, P @ V)"
        )
    scores, update = (op.operation for op in contractions)
    kv_loop = scores.parent
    if (
        kv_loop is None
        or kv_loop.name != "scf.for"
        or update.parent.operation != kv_loop.operation
        or _enclosing(kv_loop, "scf.forall") != loop.operation
    ):
        raise ValueError(
            "the two contractions are not in one loop over K/V tiles (an inner "
            "hl.tile of the grid loop)"
        )
    scores_maps, update_maps = (
        [ir.AffineMapAttr(m).value for m in linalg.get_indexing_maps(op.opview)]
        for op in contractions
    )
    scores_form = _matmul_form(scores_maps)
    update_form = _matmul_form(update_maps)
    if scores_form is None or not scores_form[3]:
        raise ValueError(
            "the first contraction is not Q @ K^T of 2D tiles with K stored "
            "[seq, head dim]"
        )
    if update_form is None or update_form[3]:
        raise ValueError(
            "the second contraction is not P @ V of 2D tiles with V stored "
            "[seq, head dim]"
        )
    body = {op.operation for op in kv_loop.regions[0].blocks[0].operations}
    from_scores = _forward_slice(scores.results[0], body)
    into_p = _backward_slice(update.operands[0], body)
    if scores not in into_p:
        raise ValueError("P @ V's left operand is not computed from Q @ K^T")
    if not any(
        not linalg.isa_contraction_op(op.opview)
        and (maps := linalg.get_indexing_maps(op.opview)) is not None
        and _reduced_dims(op, [ir.AffineMapAttr(m).value for m in maps])
        for op in from_scores & into_p
    ):
        raise ValueError(
            "no row reduction (softmax maximum or sum) of the scores between "
            "Q @ K^T and P @ V"
        )
    if not any(use.owner.name == "scf.yield" for use in update.results[0].uses):
        raise ValueError("the loop does not carry P @ V's result (the output)")
    scores_sizes = _loop_sizes(scores, scores_maps)
    update_sizes = _loop_sizes(update, update_maps)
    m, n, k, _ = scores_form
    update_m, update_n, update_k, _ = update_form
    shape = AttentionShape(
        rows=scores_sizes[m],
        kv_tile=scores_sizes[n],
        d_head=scores_sizes[k],
        d_value=update_sizes[update_n],
    )
    if (update_sizes[update_m], update_sizes[update_k]) != (shape.rows, shape.kv_tile):
        raise ValueError("P @ V does not contract the K/V tile of Q @ K^T's rows")
    return shape


def _matmul_form(maps: list[ir.AffineMap]) -> tuple[int, int, int, bool] | None:
    """``(m, n, k, transposed B)`` loop dims of a 2D contraction ``C[m, n] +=
    A[m, k] * B[k, n]`` (``B[n, k]`` if transposed B), else ``None``."""
    a_dims, b_dims, c_dims = (_dims(affine_map) for affine_map in maps[:3])
    if any(len(dims) != 2 or None in dims for dims in (a_dims, b_dims, c_dims)):
        return None
    m, n = c_dims
    if a_dims[0] != m or a_dims[1] in c_dims:
        return None
    k = a_dims[1]
    if b_dims == [k, n]:
        return m, n, k, False
    if b_dims == [n, k]:
        return m, n, k, True
    return None


def _enclosing(op: ir.Operation, name: str) -> ir.Operation | None:
    """The closest ``name`` op enclosing ``op``."""
    parent = op.parent
    while parent is not None and parent.name != "func.func":
        if parent.name == name:
            return parent.operation
        parent = parent.parent
    return None


def _in_body(op: ir.Operation, body: set[ir.Operation]) -> ir.Operation | None:
    """The op of ``body`` that is or encloses ``op``."""
    current = op.operation
    while current not in body:
        parent = current.parent
        if parent is None:
            return None
        current = parent.operation
    return current


def _forward_slice(value: ir.Value, body: set[ir.Operation]) -> set[ir.Operation]:
    """The ops of ``body`` (one block's ops) using ``value``, transitively."""
    found: set[ir.Operation] = set()
    pending = [value]
    while pending:
        for use in pending.pop().uses:
            user = _in_body(use.owner, body)
            if user is not None and user not in found:
                found.add(user)
                pending.extend(user.results)
    return found


def _backward_slice(value: ir.Value, body: set[ir.Operation]) -> set[ir.Operation]:
    """The ops of ``body`` (one block's ops) ``value`` is computed from,
    transitively."""
    found: set[ir.Operation] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if not isinstance(current, ir.OpResult):
            continue
        owner = current.owner.operation
        if owner in body and owner not in found:
            found.add(owner)
            pending.extend(owner.operands)
    return found


def _elementwise_shape(loop: ir.Operation) -> ElementwiseShape | None:
    """The tiles of the elementwise grid loop ``loop``: the tensors its
    ``tensor.parallel_insert_slice`` ops store and the slices of that shape it
    loads (``None`` if it stores nothing)."""
    ops = _walk(loop)
    stored = [op.operands[0] for op in ops if op.name == "tensor.parallel_insert_slice"]
    shapes = {tuple(ir.ShapedType(value.type).shape) for value in stored}
    if not shapes:
        return None
    if len(shapes) > 1:
        return ElementwiseShape(None, (), ())
    (wg_tile,) = shapes
    loaded = [
        op.results[0]
        for op in ops
        if op.name == "tensor.extract_slice"
        and tuple(ir.ShapedType(op.results[0].type).shape) == wg_tile
    ]
    return ElementwiseShape(
        wg_tile,
        tuple(sorted({_bit_width(value) for value in loaded})),
        tuple(sorted({_bit_width(value) for value in stored})),
    )


def _bit_width(value: ir.Value) -> int:
    element = ir.ShapedType(value.type).element_type
    if isinstance(element, (ir.IntegerType, ir.FloatType)):
        return element.width
    return 0


def _walk(op: ir.Operation) -> list[ir.Operation]:
    found: list[ir.Operation] = []
    op.walk(lambda nested: found.append(nested) or ir.WalkResult.ADVANCE)
    return found


def _dims(affine_map: ir.AffineMap) -> list[int | None]:
    return [
        ir.AffineDimExpr(expr).position if isinstance(expr, ir.AffineDimExpr) else None
        for expr in affine_map.results
    ]


def _loop_sizes(op: ir.Operation, maps: list[ir.AffineMap]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for value, affine_map in zip(op.operands, maps, strict=False):
        if not isinstance(value.type, ir.ShapedType):
            continue
        shape = ir.ShapedType(value.type).shape
        for dim, size in zip(_dims(affine_map), shape, strict=False):
            if dim is not None:
                sizes.setdefault(dim, size)
    return sizes


def _source_shape(value: ir.Value) -> list[int] | None:
    """The shape of the tensor ``value`` is a slice of, without the unit dims a
    rank-reducing slice drops (e.g. the batch dim of a tile of one batch)."""
    sliced = False
    kept: list[int] | None = None
    while isinstance(value, ir.OpResult) and value.owner.name == "tensor.extract_slice":
        dims = _kept_dims(value.owner.opview)
        kept = dims if kept is None else [dims[dim] for dim in kept]
        value = value.owner.operands[0]
        sliced = True
    if not sliced or not isinstance(value.type, ir.ShapedType):
        return None
    shape = ir.ShapedType(value.type).shape
    return [shape[dim] for dim in kept] if kept is not None else list(shape)


def _kept_dims(op: ir.OpView) -> list[int]:
    """The source dims of the ``tensor.extract_slice`` ``op`` its result keeps
    (all but the unit dims a rank-reducing slice drops, matched from the
    front as MLIR does)."""
    sizes = list(ir.DenseI64ArrayAttr(op.attributes["static_sizes"]))
    result = ir.ShapedType(op.results[0].type).shape
    kept: list[int] = []
    for dim, size in enumerate(sizes):
        if len(kept) < len(result) and result[len(kept)] == size:
            kept.append(dim)
    return kept


def _gemm_shape(op: ir.Operation, maps: list[ir.AffineMap]) -> GemmShape | None:
    a_dims, b_dims, c_dims = (_dims(affine_map) for affine_map in maps[:3])
    k_dims = [d for d in dict.fromkeys(a_dims + b_dims) if d not in c_dims]
    if len(c_dims) < 2 or len(k_dims) != 1 or None in c_dims[-2:]:
        return None
    m, n = c_dims[-2:]
    (k,) = k_dims
    sizes = _loop_sizes(op, maps)
    mnk = None
    a_source, b_source = (_source_shape(value) for value in op.operands[:2])
    if a_source is not None and b_source is not None:
        global_sizes = {
            dim: size
            for dims, shape in ((a_dims, a_source), (b_dims, b_source))
            if len(shape) == len(dims)
            for dim, size in zip(dims, shape, strict=True)
        }
        if {m, n, k} <= global_sizes.keys():
            mnk = (global_sizes[m], global_sizes[n], global_sizes[k])
    return GemmShape(
        mnk=mnk,
        wg_tile=(sizes[m], sizes[n]),
        k_tile=sizes[k],
        transpose_a=a_dims[-2:] == [k, m],
        transpose_b=b_dims[-2:] == [n, k],
    )


def _reduced_dims(op: ir.Operation, maps: list[ir.AffineMap]) -> list[int]:
    """The loop dims of a linalg op its results do not index."""
    inits = maps[len(maps) - len(op.results) :] if op.results else maps[-1:]
    indexed = {dim for affine_map in inits for dim in _dims(affine_map)}
    return [dim for dim in range(maps[0].n_dims) if dim not in indexed]


def _reduction_shape(
    loop: ir.Operation, op: ir.Operation, maps: list[ir.AffineMap]
) -> ReductionShape:
    """The reduction ``op`` of the grid loop ``loop``: for a 2D row reduction,
    its ``[rows, cols]`` tile and the element sizes of the slices of that shape
    the loop loads (else of the reduced operand)."""
    reduced = _reduced_dims(op, maps)
    parallel = [dim for dim in range(maps[0].n_dims) if dim not in reduced]
    sizes = _loop_sizes(op, maps)
    if len(parallel) != 1 or len(reduced) != 1:
        return ReductionShape(None)
    rows, cols = sizes.get(parallel[0]), sizes.get(reduced[0])
    loaded = [
        nested.results[0]
        for nested in _walk(loop)
        if nested.name == "tensor.extract_slice"
        and tuple(ir.ShapedType(nested.results[0].type).shape) == (rows, cols)
    ] or list(op.operands[:1])
    return ReductionShape(
        rows, cols, tuple(sorted({_bit_width(value) for value in loaded}))
    )


# --- Validation ------------------------------------------------------------------


def check_params(
    params: Sequence[KernelParams],
    shapes: Sequence[KernelShape | None],
) -> None:
    """Raise :class:`XeGPUConfigError` unless every kernel's parameters fit its
    shapes (``shapes``: one per kernel, by :func:`kernel_shape`) under
    lighthouse's XeGPU constraints: the forced ones (``params``; kernels past
    its end infer all), and the inferred ones the block sizes decide (a GEMM's
    workgroup and K tiles must leave lighthouse's parameter selection a choice,
    a reduction's rows and columns must fit its subgroup and block tiles)."""
    if len(params) > len(shapes):
        raise XeGPUConfigError(
            f"xegpu_* lists of {len(params)} entries, but the kernel has "
            f"{len(shapes)} GPU kernel(s) (one per top-level grid loop)"
        )
    specs = XeGPUSpecs.get("B70")
    for index, shape in enumerate(shapes):
        kernel = params[index] if index < len(params) else KernelParams()
        forced = kernel.forced()
        try:
            if isinstance(shape, GemmShape):
                _check_gemm(kernel, shape, specs)
            elif isinstance(shape, ReductionShape):
                _check_reduction(kernel, shape, specs)
            elif isinstance(shape, AttentionShape):
                _check_attention(kernel, shape, specs)
            elif isinstance(shape, ElementwiseShape):
                if forced:
                    _check_elementwise(kernel, shape)
            elif forced:
                raise ValueError(
                    "parameters apply to GEMM (one 2D matmul), attention, "
                    "reduction and elementwise kernels"
                )
        except ValueError as exc:
            kind = f", a {shape}" if shape is not None else ""
            if not forced:
                raise XeGPUConfigError(
                    f"GPU kernel {index}{kind}: its block sizes leave the XeGPU "
                    f"pipeline no parameters: {exc}"
                ) from None
            values = ", ".join(
                f"{key}[{index}]={list(value)}" for key, value in forced.items()
            )
            raise XeGPUConfigError(
                f"{values}: invalid for GPU kernel {index}{kind}: {exc}"
            ) from None


def _check_gemm(kernel: KernelParams, shape: GemmShape, specs: XeGPUSpecs) -> None:
    wg_tile, k_tile = shape.wg_tile, shape.k_tile
    if kernel.load_tiles is not None and len(kernel.load_tiles) != 4:
        raise ValueError("the load tiles of a GEMM are [a_m, a_k, b_k, b_n]")
    sg_tile = kernel.sg_tile
    if sg_tile is not None:
        if 0 in sg_tile:
            raise ValueError("subgroup tiles of a GEMM are positive")
        matmul_constraints.check_sg_tile(
            wg_tile, sg_tile, specs, min_nb_threads=matmul_constraints.MIN_NB_THREADS
        )
        if shape.mnk is not None:
            # The cost model's limits (parallel DPAS ops, registers per subgroup).
            matmul_costmodel.estimate_performance(
                *shape.mnk, wg_tile, sg_tile, k_tile, specs, verbose=False
            )
    if shape.mnk is not None:
        # Lighthouse infers the rest around the forced subgroup tile (fails
        # here if its selection finds nothing for the block sizes), or the
        # subgroup tile forced load tiles must fit.
        sg_tile = _selected_sg_tile(shape, sg_tile)
    if kernel.load_tiles is not None:
        if sg_tile is None:
            raise ValueError(
                "the matmul's sizes are unknown, so load tiles need a forced "
                "subgroup tile"
            )
        a_tile, b_tile = kernel.load_tiles[:2], kernel.load_tiles[2:]
        matmul_constraints.check_load_tile_a(
            a_tile, sg_tile, k_tile, transpose=shape.transpose_a
        )
        matmul_constraints.check_load_tile_b(
            b_tile, sg_tile, k_tile, transpose=shape.transpose_b
        )
    if kernel.prefetch_tiles is not None:
        a_tile, b_tile = kernel.prefetch_tiles[:2], kernel.prefetch_tiles[2:]
        for check, tile, transpose in (
            (matmul_constraints.check_prefetch_tile_a, a_tile, shape.transpose_a),
            (matmul_constraints.check_prefetch_tile_b, b_tile, shape.transpose_b),
        ):
            check(
                tile,
                wg_tile,
                k_tile,
                specs,
                transpose=transpose,
                min_nb_threads=matmul_constraints.MIN_NB_THREADS,
            )
    if kernel.prefetch_depths is not None and 0 in kernel.prefetch_depths:
        raise ValueError("prefetch depths are positive")


def _selected_sg_tile(
    shape: GemmShape, sg_tile: tuple[int, ...] | None
) -> tuple[int, int]:
    """The subgroup tile of lighthouse's parameter selection for ``shape`` (as
    the lowering runs it: for the block sizes and any forced subgroup tile)."""
    assert shape.mnk is not None
    params = XeGPUParameterSelector().get_parameters_dict(
        shape.mnk,
        transpose_a=shape.transpose_a,
        transpose_b=shape.transpose_b,
        wg_tile=shape.wg_tile,
        k_tile=shape.k_tile,
        sg_tile=sg_tile,
    )
    return params["sg_m"], params["sg_n"]


REDUCTION_SG_ROWS = 8
"""The rows of lighthouse's inferred reduction subgroup tile
(``infer_xegpu_reduction_params``)."""


def _check_reduction(
    kernel: KernelParams, shape: ReductionShape, specs: XeGPUSpecs
) -> None:
    gemm_only = [key for key in _GEMM_ONLY if key in kernel.forced()]
    if gemm_only:
        raise ValueError(f"{', '.join(gemm_only)} apply to GEMM kernels only")
    load_tile = kernel.load_tiles
    if load_tile is not None and len(load_tile) != 2:
        raise ValueError("the load tile of a reduction kernel is [m, n]")
    if shape.rows is None:
        if kernel.forced():
            raise ValueError("forced parameters need a 2D row reduction")
        return
    rows = REDUCTION_SG_ROWS
    if kernel.sg_tile is None:
        _check_reduction_tiles(shape, specs)
    else:
        rows, cols = kernel.sg_tile
        if cols != 0 or rows == 0:
            raise ValueError(
                "the subgroup tile of a reduction is [rows, 0] with rows > 0"
            )
        if shape.rows % rows:
            raise ValueError(f"{rows} rows do not divide the workgroup's {shape.rows}")
        if shape.rows // rows > specs.max_nb_threads:
            raise ValueError(
                f"{shape.rows // rows} subgroups exceed the maximum of "
                f"{specs.max_nb_threads}"
            )
    if load_tile is not None:
        if shape.cols is None:
            raise ValueError("the reduced tile's columns are unknown")
        # The subgroup tile lighthouse lays the reduced tile out in: its rows,
        # all columns.
        reduction_tiles((rows, shape.cols), shape.load_bits, load_tile)


def _check_reduction_tiles(shape: ReductionShape, specs: XeGPUSpecs) -> None:
    """The workgroup tile of a reduction with lighthouse's subgroup rows: whole
    subgroups, at most the maximum, and whole block rows of columns."""
    assert shape.rows is not None
    rows = REDUCTION_SG_ROWS
    if shape.rows % rows:
        raise ValueError(
            f"{shape.rows} rows per workgroup are not a multiple of the {rows}-row "
            "subgroups"
        )
    if shape.rows // rows > specs.max_nb_threads:
        raise ValueError(
            f"{shape.rows // rows} subgroups of {rows} rows exceed the maximum of "
            f"{specs.max_nb_threads}"
        )
    if shape.cols is not None and shape.cols % BLOCK_COLS:
        raise ValueError(
            f"{shape.cols} columns are not a multiple of {BLOCK_COLS} (a block load "
            "row)"
        )


def _check_attention(
    kernel: KernelParams, shape: AttentionShape, specs: XeGPUSpecs
) -> None:
    """Lighthouse's attention layout (``annotate_attention``) for the block sizes
    and a forced subgroup tile ``[rows, 0]`` (its only parameter here):
    subgroups of whole 16-row Q load blocks, at most the maximum per workgroup,
    and K/V tiles of whole 32-row V load blocks."""
    other = [key for key in kernel.forced() if key != SG_TILES]
    if other:
        raise ValueError(
            f"{', '.join(other)} do not apply to attention kernels (lighthouse's "
            "attention annotation fixes their load and prefetch tiles)"
        )
    rows = ATTENTION_SG_ROWS
    if kernel.sg_tile is not None:
        rows, cols = kernel.sg_tile
        if cols != 0 or rows == 0 or rows % ATTENTION_SG_ROWS:
            raise ValueError(
                "the subgroup tile of an attention kernel is [rows, 0] with rows a "
                f"multiple of {ATTENTION_SG_ROWS} (its Q load blocks)"
            )
    if shape.rows % rows:
        raise ValueError(
            f"{shape.rows} query rows per workgroup are not a multiple of the "
            f"{rows}-row subgroups"
        )
    if shape.rows // rows > specs.max_nb_threads:
        raise ValueError(
            f"{shape.rows // rows} subgroups of {rows} rows exceed the maximum of "
            f"{specs.max_nb_threads}"
        )
    if shape.kv_tile % ATTENTION_KV_ROWS:
        raise ValueError(
            f"a K/V tile of {shape.kv_tile} rows is not a multiple of "
            f"{ATTENTION_KV_ROWS} (lighthouse's V load blocks)"
        )


def _check_elementwise(kernel: KernelParams, shape: ElementwiseShape) -> None:
    gemm_only = [
        key for key in (PREFETCH_TILES, PREFETCH_DEPTHS) if key in kernel.forced()
    ]
    if gemm_only:
        raise ValueError(
            f"{', '.join(gemm_only)} apply to GEMM kernels only (elementwise kernels "
            "have no K loop to prefetch in)"
        )
    one_d = shape.wg_tile is not None and len(shape.wg_tile) == 1
    if one_d and kernel.sg_tile is not None:
        raise ValueError(
            "1D elementwise kernels take only a load tile [16 x chunk] (the "
            "subgroup tile is inferred)"
        )
    if kernel.load_tiles is not None and len(kernel.load_tiles) != (1 if one_d else 2):
        raise ValueError(
            "the load tile of a 1D elementwise kernel is [n]"
            if one_d
            else "the load tile of an elementwise kernel is [m, n]"
        )
    # Whether a 1D kernel's tiles are contiguous shows only in the XeGPU IR:
    # a chunked tile is checked against that when it is anchored.
    shape.layout(
        sg_tile=kernel.sg_tile, load_tile=kernel.block_load_tile(), contiguous=True
    )
