"""Seeded init → placed arrays, with no host-side full tree.

Each helper computes the declared shardings on an `eqx.filter_eval_shape`'d abstract
value, then runs the seeded init under `jax.jit(init, out_shardings=...)` so each device
generates only its own shard — an eager `device_put` of a host tree onto a multi-process
non-replicated sharding triggers a `process_allgather` (a host allocation of the FULL
unsharded tree per process). A non-dividing declared shard axis is a loud crash at
placement construction / inside `.shardings` (fail-fast), never a silent replicate.

Compile-time doctrine: keep seeded inits FEW-OUTPUTS-under-jit — a jit returning n_sites
(hundreds of) sharded outputs, or n_chunks unrolled RNG bodies, is a multi-minute
SPMD/layout compile. vmap-stack over the same per-site/per-chunk keys (bit-identical
values), then fan out with a trivial slice jit. `init_component_stacks_placed` is the
template; `init_ci_fn_placed` / `init_sources_sharded` follow it.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from param_decomp.core.adversary import (
    ExpertBlockedSource,
    SourceStack,
    SourceStacks,
    init_persistent_sources,
)
from param_decomp.core.axes import MeshAxis
from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIFn,
    CIFn,
    CIFnArch,
    GlobalMLPCIFn,
    LayerwiseMLPCIFn,
    MoEChunkwiseTransformerCIFn,
    build_ci_fn,
    pad_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import (
    ComponentStacks,
    Dense,
    ExpertBlocked,
    SiteSpec,
    init_component_stacks,
    pad_component_stacks,
    site_slots_for,
    vu_groups,
)
from param_decomp.core.configs import (
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    PersistentPGDLossConfig,
    SourceShape,
)
from param_decomp.core.model import (
    DecomposedModel,
    PlacedModel,
    PositionAxis,
    Positioned,
    Positionless,
)
from param_decomp.core.placement import (
    CIFnPlacement,
    PlacementRules,
    batch_axes,
    component_stacks_shardings,
)

type ComponentInitializer[Out, PreparedT] = Callable[
    [DecomposedModel[Out, PreparedT], PRNGKeyArray], ComponentStacks
]
"""A target-aware, unplaced V/U initializer. The placed wrapper below owns sharding."""


def random_component_initializer[Out](
    model: DecomposedModel[Out], key: PRNGKeyArray
) -> ComponentStacks:
    """The domain-neutral random initializer used unless a composition root selects another."""
    return init_component_stacks(model.sites, key)


def _census_stack_pads(rules: PlacementRules) -> dict[str, int]:
    """The rules' resolved persist-stack pads, ready for `pad_component_stacks`."""
    return {
        group: entry.stack_pad
        for group, entry in rules.components.group_census.items()
        if entry.stack_pad
    }


def init_component_stacks_placed(
    sites: tuple[SiteSpec, ...], key: PRNGKeyArray, rules: PlacementRules
) -> ComponentStacks:
    """Seed random V/U directly into the component persistence layout, the census'
    persist-stack pads appended as all-zero slots (real slots draw identically to an
    unpadded init)."""
    pads = _census_stack_pads(rules)
    init = lambda k: pad_component_stacks(init_component_stacks(sites, k), pads)
    abstract = eqx.filter_eval_shape(init, key)
    placement = component_stacks_shardings(abstract, rules)
    return jax.jit(init, out_shardings=placement)(key)


def padded_component_initializer[Out, PreparedT](
    rules: PlacementRules, initializer: ComponentInitializer[Out, PreparedT]
) -> ComponentInitializer[Out, PreparedT]:
    """`initializer` with the census' persist-stack pads appended as all-zero slots — the
    tree the component persistence layout is declared over, so every consumer of the
    initializer's shape (the placed init, the pre-init placement audit) sees the pads."""
    pads = _census_stack_pads(rules)
    return lambda m, k: pad_component_stacks(initializer(m, k), pads)


def init_model_component_stacks_placed[Out, PreparedT](
    model: PlacedModel[Out, PreparedT],
    key: PRNGKeyArray,
    rules: PlacementRules,
    initializer: ComponentInitializer[Out, PreparedT],
) -> ComponentStacks:
    """Run a target-aware initializer directly into the component persistence layout,
    the census' persist-stack pads appended as all-zero slots.

    The frozen model stays a traced argument: an aligned initializer may read target weights.
    Initializers return semantic-group stacks, preserving the no-host-full-tree contract.
    """
    init = padded_component_initializer(rules, initializer)
    abstract = eqx.filter_eval_shape(init, model.model, key)
    placement = component_stacks_shardings(abstract, rules)
    return jax.jit(init, out_shardings=placement)(model.model, key)


def ci_fn_shardings(abstract: CIFn, mesh: Mesh, placement: CIFnPlacement | None) -> CIFn:
    """The CI fn's own declared placement, dispatched by arch: the chunkwise transformers
    from their RESOLVED placement (`resolve_ci_placement` — rows plus the chunk-stack
    census the abstract fn is validated against); the toy MLPs run unplaced and shard
    each weight's output axis. THE single source of truth for the CI fn's persist layout —
    seeded init places onto it, and AOT consumers (the fit check) type their abstract
    state with it."""
    match abstract:
        case ChunkwiseTransformerCIFn() | MoEChunkwiseTransformerCIFn():
            assert placement is not None, f"{type(abstract).__name__} is placed by its rows"
            return abstract.shardings(mesh, placement)
        case LayerwiseMLPCIFn() | GlobalMLPCIFn():
            assert placement is None, f"{type(abstract).__name__} runs unplaced"
            return abstract.shardings(mesh)
        case _:
            raise AssertionError(f"unknown CI fn {type(abstract)}")


def init_ci_fn_placed(
    arch: CIFnArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
) -> CIFn:
    """Seeded CI-fn init (any arch, via `build_ci_fn`) directly into its persist layout,
    the resolved chunk-stack pad appended as all-zero slots (`pad_ci_fn`; real slots draw
    identically to an unpadded init) and placed by `ci_fn_shardings`. Shardings computed
    on the abstract fn, init under jit."""
    placement = resolve_ci_placement(arch, rules)
    init = lambda k: pad_ci_fn(build_ci_fn(arch, sites, k), placement)
    abstract = eqx.filter_eval_shape(init, key)
    return jax.jit(init, out_shardings=ci_fn_shardings(abstract, mesh, placement))(key)


def _source_leading(
    positions: PositionAxis, source_shape: SourceShape, global_batch: int, mesh: Mesh
) -> tuple[tuple[int, ...], tuple[tuple[MeshAxis, ...] | None, ...]]:
    """Each stored (positions x source_shape) leading shape, with its mesh spec: batch-B
    shapes batch-shard over the data axes, batch-1 and position axes replicate."""
    data_axes = batch_axes(mesh)
    match positions, source_shape:
        case Positionless(), "c":
            return (1,), (None,)
        case Positionless(), "bc":
            return (global_batch,), (data_axes,)
        case Positionless(), "sc" | "bsc":
            raise ValueError(
                f"source_shape {source_shape!r} names a position axis; target is positionless"
            )
        case Positioned(), "c":
            return (1, 1), (None, None)
        case Positioned(), "bc":
            return (global_batch, 1), (data_axes, None)
        case Positioned(n_positions=n), "sc":
            return (1, n), (None, None)
        case Positioned(n_positions=n), "bsc":
            return (global_batch, n), (data_axes, None)


def _source_stacks_shardings(
    sites: tuple[SiteSpec, ...],
    leading_spec: tuple[tuple[MeshAxis, ...] | None, ...],
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    stacks: dict[str, SourceStack[NamedSharding]] = {}
    for group, members in vu_groups(sites).items():
        match members.factorization:
            case Dense():
                components: NamedSharding | ExpertBlockedSource = NamedSharding(
                    mesh, P(None, *leading_spec, "tp")
                )
            case ExpertBlocked():
                components = ExpertBlockedSource(
                    values=NamedSharding(mesh, P(None, *leading_spec, "tp", None))  # pyright: ignore[reportArgumentType]
                )
        stacks[group] = SourceStack(
            components=components, delta=NamedSharding(mesh, P(None, *leading_spec))
        )
    return SourceStacks(stacks=stacks, site_slots=site_slots_for(sites))


def persistent_sources_shardings(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    source_shape: SourceShape,
    global_batch: int,
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    """The declared placement of one ordinary persistent adversary's source stacks.

    Batch-B shapes shard over the data axes; batch-1 and position axes replicate. The
    component axis follows the CI's TP placement, and source-delta values replicate
    over TP.
    """
    _, leading_spec = _source_leading(positions, source_shape, global_batch, mesh)
    return _source_stacks_shardings(sites, leading_spec, mesh)


def source_pool_shardings(sites: tuple[SiteSpec, ...], mesh: Mesh) -> SourceStacks[NamedSharding]:
    """Replicate global pool rows over data axes and shard component columns over TP."""
    return _source_stacks_shardings(sites, (None,), mesh)


def init_source_pool_sharded(
    sites: tuple[SiteSpec, ...],
    pool_size: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Initialize ``pool_size`` globally shared cross-site adversarial particles."""
    return jax.jit(
        partial(init_persistent_sources, sites, (pool_size,), source_dtype),
        out_shardings=source_pool_shardings(sites, mesh),
    )(key)


def persistent_sources_shardings_from_config(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    cfg: PersistentPGDLossConfig | MergedStochasticSubsetPooledPPGDReconLossConfig,
    global_batch: int,
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    """The runtime placement declared by one persistent-source config."""
    match cfg:
        case MergedStochasticSubsetPooledPPGDReconLossConfig():
            return source_pool_shardings(sites, mesh)
        case PersistentPGDLossConfig(source_shape=source_shape):
            return persistent_sources_shardings(sites, positions, source_shape, global_batch, mesh)


def init_persistent_sources_from_config(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    cfg: PersistentPGDLossConfig | MergedStochasticSubsetPooledPPGDReconLossConfig,
    global_batch: int,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Initialize one persistent-source config directly into its runtime placement."""
    match cfg:
        case MergedStochasticSubsetPooledPPGDReconLossConfig(pool_size=pool_size):
            return init_source_pool_sharded(sites, pool_size, cfg.source_dtype, key, mesh)
        case PersistentPGDLossConfig(source_shape=source_shape):
            return init_sources_sharded(
                sites,
                positions,
                source_shape,
                global_batch,
                cfg.source_dtype,
                key,
                mesh,
            )


def init_sources_sharded(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    source_shape: SourceShape,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Seeded PPGD-source init placed onto `persistent_sources_shardings` (jit +
    `out_shardings`; same no-host-tree rationale as `init_component_stacks_placed`, and
    the same few-outputs doctrine — one sharded output per semantic group). Every stored
    (positions x source_shape) leading shape is enumerated in `_source_leading`; the rank
    always matches the waist, with size-1 broadcast axes for the letters `source_shape`
    omits (`configs.SourceShape`).

    Batch-1 shapes (`c`, `sc`) share one source across the global batch. Batch-B shapes
    (`bc`, `bsc`) are BATCH-SHARDED over the data-parallel axes (`placement.batch_axes`),
    aligning each batch element's source with that element's `shard_batch`-placed
    residual/CI. The source is independent per element, so the per-element grad is
    already shard-local — NO cross-rank reduction, matching torch's `_skip_all_reduce`.
    (Requires `global_batch % n_dev == 0`, the same divisibility `shard_batch` needs.)"""
    leading_shape, _ = _source_leading(positions, source_shape, global_batch, mesh)
    shardings = persistent_sources_shardings(sites, positions, source_shape, global_batch, mesh)
    return jax.jit(
        partial(init_persistent_sources, sites, leading_shape, source_dtype),
        out_shardings=shardings,
    )(key)
