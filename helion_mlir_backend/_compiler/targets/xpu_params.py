"""XeGPU lowering parameters from the Helion config, per GPU kernel (XPU target).

The keys hold one entry per GPU kernel, i.e. per top-level grid loop of the
Helion kernel in program order (across ``hl.barrier`` phases). An entry, or a
whole key, that is ``None`` or missing -- as are the kernels past the end of a
shorter list -- leaves the parameter to lighthouse's inference:

- ``xegpu_sg_tiles``: the subgroup tile, ``[sg_m, sg_n]`` of a GEMM kernel;
  ``[rows, 0]`` of a (2D, row) reduction kernel.
- ``xegpu_load_tiles``: ``[a_m, a_k, b_k, b_n]``, the A and B tiles a GEMM
  kernel's subgroups load per instruction (``inst_data`` of the loads).
- ``xegpu_prefetch_tiles``: ``[a_m, a_k, b_k, b_n]``, the A and B tiles a GEMM
  kernel's subgroups prefetch, in the orientation the operand is stored in (a
  transposed B prefetches ``[n, k]`` tiles).
- ``xegpu_prefetch_depths``: ``[a_nb, b_nb]``, how many K-loop iterations ahead a
  GEMM kernel prefetches A and B.

The workgroup tile and the K tile are not parameters: they are the kernel's
block sizes (the grid tiles and the inner K tile), which lighthouse reads off
the IR.

:func:`xegpu_params` reads the config; :func:`attach_params` records the result
on the generated module (so modules compiled with different parameters differ,
also as JIT cache keys) and :func:`take_params` reads it back for lowering.
:func:`check_params` validates parameters against the kernels' shapes with
lighthouse's ``matmul_constraints`` before lowering.
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

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

SG_TILES = "xegpu_sg_tiles"
LOAD_TILES = "xegpu_load_tiles"
PREFETCH_TILES = "xegpu_prefetch_tiles"
PREFETCH_DEPTHS = "xegpu_prefetch_depths"
_ARITY = {SG_TILES: 2, LOAD_TILES: 4, PREFETCH_TILES: 4, PREFETCH_DEPTHS: 2}
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
_GEMM_ONLY = (LOAD_TILES, PREFETCH_TILES, PREFETCH_DEPTHS)


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
        if self.load_tiles is not None:
            options["load_tile_a"] = self.load_tiles[:2]
            options["load_tile_b"] = self.load_tiles[2:]
        if self.prefetch_tiles is not None:
            options["prefetch_tile_a"] = self.prefetch_tiles[:2]
            options["prefetch_tile_b"] = self.prefetch_tiles[2:]
        if self.prefetch_depths is not None:
            options["prefetch_a_nb"], options["prefetch_b_nb"] = self.prefetch_depths
        return options


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
    if (
        not isinstance(item, (list, tuple))
        or len(item) != arity
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in item
        )
    ):
        raise XeGPUConfigError(
            f"{key}[{index}]={item!r}: expected {arity} non-negative ints or None "
            "(one entry per GPU kernel)"
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

    def __str__(self) -> str:
        rows = "" if self.rows is None else f" of {self.rows} rows per workgroup"
        return f"reduction{rows}"


def kernel_shape(loop: ir.Operation) -> GemmShape | ReductionShape | None:
    """The GEMM or reduction a grid loop computes (on tensors, before lowering)."""
    first_reduction = None
    for op in _walk(loop):
        view = op.opview
        maps = linalg.get_indexing_maps(view)
        if maps is None:
            continue
        affine_maps = [ir.AffineMapAttr(m).value for m in maps]
        if linalg.isa_contraction_op(view):
            return _gemm_shape(op, affine_maps)
        if first_reduction is None and _reduced_dims(op, affine_maps):
            first_reduction = (op, affine_maps)
    if first_reduction is None:
        return None
    return _reduction_shape(*first_reduction)


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
    """The shape of the tensor ``value`` is a (rank-preserving) slice of."""
    sliced = False
    while isinstance(value, ir.OpResult) and value.owner.name == "tensor.extract_slice":
        value = value.owner.operands[0]
        sliced = True
    if not sliced or not isinstance(value.type, ir.ShapedType):
        return None
    return list(ir.ShapedType(value.type).shape)


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


def _reduction_shape(op: ir.Operation, maps: list[ir.AffineMap]) -> ReductionShape:
    reduced = _reduced_dims(op, maps)
    parallel = [dim for dim in range(maps[0].n_dims) if dim not in reduced]
    sizes = _loop_sizes(op, maps)
    if len(parallel) == 1 and len(reduced) == 1:
        return ReductionShape(sizes.get(parallel[0]))
    return ReductionShape(None)


# --- Validation ------------------------------------------------------------------


def check_params(
    params: Sequence[KernelParams],
    shapes: Sequence[GemmShape | ReductionShape | None],
) -> None:
    """Raise :class:`XeGPUConfigError` unless every kernel's forced parameters
    fit its shapes (``shapes``: one per kernel, by :func:`kernel_shape`) under
    lighthouse's XeGPU constraints."""
    if len(params) > len(shapes):
        raise XeGPUConfigError(
            f"xegpu_* lists of {len(params)} entries, but the kernel has "
            f"{len(shapes)} GPU kernel(s) (one per top-level grid loop)"
        )
    specs = XeGPUSpecs.get("B70")
    for index, (kernel, shape) in enumerate(zip(params, shapes, strict=False)):
        forced = kernel.forced()
        if not forced:
            continue
        try:
            if isinstance(shape, GemmShape):
                _check_gemm(kernel, shape, specs)
            elif isinstance(shape, ReductionShape):
                _check_reduction(kernel, shape, specs)
            else:
                raise ValueError(
                    "parameters apply to GEMM (one 2D matmul) and reduction kernels"
                )
        except ValueError as exc:
            values = ", ".join(
                f"{key}[{index}]={list(value)}" for key, value in forced.items()
            )
            kind = f", a {shape}" if shape is not None else ""
            raise XeGPUConfigError(
                f"{values}: invalid for GPU kernel {index}{kind}: {exc}"
            ) from None


def _check_gemm(kernel: KernelParams, shape: GemmShape, specs: XeGPUSpecs) -> None:
    wg_tile, k_tile = shape.wg_tile, shape.k_tile
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
    if (sg_tile is not None or kernel.load_tiles is not None) and shape.mnk is not None:
        # Lighthouse infers the rest around the forced subgroup tile, or the
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


def _check_reduction(
    kernel: KernelParams, shape: ReductionShape, specs: XeGPUSpecs
) -> None:
    gemm_only = [key for key in _GEMM_ONLY if key in kernel.forced()]
    if gemm_only:
        raise ValueError(f"{', '.join(gemm_only)} apply to GEMM kernels only")
    sg_tile = kernel.sg_tile
    if sg_tile is None:
        return
    if shape.rows is None:
        raise ValueError("forced subgroup tiles need a 2D row reduction")
    rows, cols = sg_tile
    if cols != 0 or rows == 0:
        raise ValueError("the subgroup tile of a reduction is [rows, 0] with rows > 0")
    if shape.rows % rows:
        raise ValueError(f"{rows} rows do not divide the workgroup's {shape.rows}")
    if shape.rows // rows > specs.max_nb_threads:
        raise ValueError(
            f"{shape.rows // rows} subgroups exceed the maximum of "
            f"{specs.max_nb_threads}"
        )
