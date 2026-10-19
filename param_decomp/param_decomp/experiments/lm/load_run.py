"""Restore a finished JAX language-model decomposition for read-only analysis.

`open_run` checks the saved deliverable and checkpoint step, reconstructs the target
on the layout the CALLER names (`ConsumerLayout` — never derived from the run), restores
component weights and the CI function, and returns a `LoadedRun` ready for harvesting
without any training state or optimizer."""

from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, SingleDeviceSharding
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray
from param_decomp_formats.config import BaseConfig

from param_decomp.core import placement
from param_decomp.core.checkpoint import (
    make_read_only_checkpoint_manager,
    restore_components,
    restore_decomposition,
)
from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.configs import (
    HsdpMeshShape,
    MeshShape,
    PlacementSpec,
    ResidentMeshShape,
    SequenceSharding,
)
from param_decomp.core.init_placed import (
    ComponentInitializer,
    random_component_initializer,
    seeded_ci_fn_initializer,
)
from param_decomp.core.model import ComponentActivations, DecomposedModel, PlacedModel
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import mesh_for_shape, place_target
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm.config import hf_model_variant
from param_decomp.experiments.lm.deliverable import (
    ResolvedDeliverable,
    load_deliverable,
    load_dense_target,
)
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    HFSnapshotWeights,
    LlamaSimpleMLPTargetConfig,
    PretrainCacheWeights,
    Qwen36MoeTargetConfig,
    TargetConfig,
    WeightsDtype,
    dense_transformer_anatomy,
    require_unrouted_ci_fn_arch,
    weights_jnp_dtype,
)
from param_decomp.infra import pretrain_cache
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments, LMBatchWithRouting
from param_decomp.targets import llama_simple_mlp, qwen36_moe, transformer
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import QwenPreparedMasking, QwenPreparedWeights
from param_decomp.targets.transformer import (
    TransformerDecomposedModel,
    TransformerPreparedMasking,
    TransformerPreparedWeights,
)

type PlacedTransformer = PlacedModel[
    LMBatchWithDocuments,
    LMOutput,
    TransformerPreparedWeights,
    LMBatchWithDocuments,
    TransformerPreparedMasking,
]
type PlacedQwen = PlacedModel[
    LMBatch, LMOutput, QwenPreparedWeights, LMBatchWithRouting[LMBatch], QwenPreparedMasking
]
type PlacedLM = PlacedTransformer | PlacedQwen


def load_target(
    target: AnyLMTargetConfig, data_root: Path
) -> TransformerDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel:
    """Load the executable target with its weights and constants resident on CPU.

    `build_target` places the result onto a real mesh; the AOT fit check constructs
    shape leaves on compile-only devices instead.
    Pretrain-cache weights (SimpleMLP, the qwen36_moe toys) read under `data_root` (no
    network); HF weights read the HF snapshot. Every loader casts its weights to the
    config's `weights_dtype` on read — this is the ONLY place that dtype is applied, so
    train and consume load the same target."""
    match target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            sites = llama_simple_mlp.site_specs(simple_cfg, target.sites)
            return llama_simple_mlp.load_decomposed_lm_from_pretrain_cache(
                cache_dir,
                simple_cfg,
                sites,
                weights_jnp_dtype(target.weights_dtype),
                target.output_edge,
                target.attention_implementation,
            )
        case TargetConfig():
            variant = hf_model_variant(target.model_name)
            return variant.load(
                target.model_name,
                target.sites,
                weights_jnp_dtype(target.weights_dtype),
                target.output_edge,
                target.attention_implementation,
            )
        case Qwen36MoeTargetConfig():
            moe_sites = qwen36_moe.qwen36_moe_site_specs(target.arch, target.sites)
            match target.weights:
                case HFSnapshotWeights(model_name=model_name):
                    return qwen36_moe.load_decomposed_qwen36_moe_from_hf(
                        model_name,
                        target.arch,
                        moe_sites,
                        weights_jnp_dtype(target.weights_dtype),
                        target.attention_implementation,
                        target.expert_implementation,
                        target.output_edge,
                    )
                case PretrainCacheWeights(pretrain_run_path=run_path):
                    return qwen36_moe.load_decomposed_qwen36_moe_from_pretrain_cache(
                        pretrain_cache.resolved_cache_dir(data_root, run_path),
                        target.arch,
                        moe_sites,
                        weights_jnp_dtype(target.weights_dtype),
                        target.attention_implementation,
                        target.expert_implementation,
                        target.output_edge,
                    )


def component_initializer_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    target: AnyLMTargetConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> ComponentInitializer[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT]:
    """Bind the authored initializer to the model's input and pinned-decision types."""
    match target.component_initialization:
        case "random":
            return random_component_initializer
        case "nonlinearity_aligned":
            assert isinstance(
                model.model, (TransformerDecomposedModel, qwen36_moe.Qwen36MoeDecomposedModel)
            ), "nonlinearity-aligned initialization requires a GLU or Qwen MoE target"

            def initialize(
                placed_target: DecomposedModel[
                    TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT
                ],
                key: PRNGKeyArray,
            ) -> ComponentStacks:
                assert isinstance(
                    placed_target, (TransformerDecomposedModel, qwen36_moe.Qwen36MoeDecomposedModel)
                )
                match placed_target:
                    case TransformerDecomposedModel():
                        return transformer.nonlinearity_aligned_component_initializer(
                            placed_target, key
                        )
                    case qwen36_moe.Qwen36MoeDecomposedModel():
                        return qwen36_moe.nonlinearity_aligned_component_initializer(
                            placed_target, key
                        )

            return initialize


def target_vocab_size[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT] | PlacedLM,
) -> int:
    """The vocabulary a placed LM target's token ids index — its embedding's row count."""
    match placed.model:
        case (
            TransformerDecomposedModel(embed=embed)
            | qwen36_moe.Qwen36MoeDecomposedModel(embed=embed)
        ):
            return embed.shape[0]
        case other:
            raise AssertionError(f"not an LM target: {type(other).__name__}")


@dataclass(frozen=True)
class TargetSite:
    """A decomposed site of a dense run's target, as its config states it: its `name`, the
    target layer it is in, and `C`, its number of components."""

    name: str
    layer: int
    C: int


def open_target_sites(run_dir: Path, *, data_root: Path) -> tuple[TargetSite, ...]:
    """A dense run's decomposed sites in canonical order, from its stored config alone
    (`load_dense_target`, the sections `open_target_and_components` builds the model from),
    with no weights loaded."""
    match load_dense_target(run_dir, data_root):
        case TargetConfig() | LlamaSimpleMLPTargetConfig() as dense:
            parse = dense_transformer_anatomy(dense).family.parse
            return tuple(TargetSite(site.name, parse(site.name)[0], site.C) for site in dense.sites)
        case Qwen36MoeTargetConfig():
            raise NotImplementedError("component substitution supports dense transformers only")


@dataclass(frozen=True)
class DenseTargetDims:
    """A dense run's target's dimensions, as its config states them: `n_layer` layers, each
    with `n_head` query heads `head_dim` wide, writing a residual stream `n_embd` wide."""

    n_layer: int
    n_head: int
    head_dim: int
    n_embd: int


def open_target_dims(run_dir: Path, *, data_root: Path) -> DenseTargetDims:
    """A dense run's target's dimensions, from its stored config and its architecture's
    (a pretrain cache's model config, or a vendored HF variant's), with no weights loaded."""
    match load_dense_target(run_dir, data_root):
        case LlamaSimpleMLPTargetConfig(pretrain_run_path=pretrain_run_path):
            simple = llama_simple_mlp.load_model_config(
                pretrain_cache.resolved_cache_dir(data_root, pretrain_run_path)
            )
            return DenseTargetDims(simple.n_layer, simple.n_head, simple.head_dim, simple.n_embd)
        case TargetConfig(model_name=model_name):
            arch = hf_model_variant(model_name).arch
            return DenseTargetDims(arch.n_layer, arch.n_head, arch.head_dim, arch.n_embd)
        case Qwen36MoeTargetConfig():
            raise NotImplementedError("component substitution supports dense transformers only")


def open_target_anatomy(run_dir: Path, *, data_root: Path) -> transformer.Anatomy:
    """A dense run's target's anatomy (`dense_transformer_anatomy`), from its stored config
    alone (`load_dense_target`, the sections `open_target_and_components` reads), with no
    weights loaded."""
    match load_dense_target(run_dir, data_root):
        case TargetConfig() | LlamaSimpleMLPTargetConfig() as dense:
            return dense_transformer_anatomy(dense)
        case Qwen36MoeTargetConfig():
            raise NotImplementedError("component substitution supports dense transformers only")


def build_target(
    target: AnyLMTargetConfig,
    mesh: jax.sharding.Mesh,
    data_root: Path,
    sharding: PlacementSpec,
    sequence_sharding: SequenceSharding,
) -> PlacedLM:
    """Build and place the frozen target shared by training and every offline consumer.

    The bundle's `.model` (an `eqx.Module`) IS the frozen target — it carries the full
    model weights (embedding included) as fields and embeds its token input internally;
    `.placement` is the resolved rules for `sharding` and `sequence_sharding`."""
    loaded_model = load_target(target, data_root)
    placement_rules = placement.from_config(
        sharding, mesh, loaded_model.sites, sequence_sharding=sequence_sharding
    )
    match loaded_model:
        case TransformerDecomposedModel():
            return place_target(loaded_model, placement_rules)
        case qwen36_moe.Qwen36MoeDecomposedModel():
            return place_target(loaded_model, placement_rules)


@eqx.filter_jit
def _prepare_read_only_consumer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    components: ComponentStacks,
    ci_fn: CIFn[Conditioning],
) -> tuple[PreparedT, CIFn[Conditioning]]:
    return placed.prepare_compute_weights(components), ci_fn.prepare()


@dataclass(frozen=True)
class LoadedRun[TargetIn, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT]:
    """A restored decomposition prepared only for inference and analysis: the frozen
    target with the consumer's rules, the component weights and CI fn prepared
    (the CI fn's conditioning is the target's), and the mesh to run them under
    (`jax.set_mesh(mesh)`)."""

    run_id: str
    step: int
    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT]
    deliverable: ResolvedDeliverable
    prepared_weights: PreparedT
    prepared_ci_fn: CIFn[Conditioning]
    mesh: jax.sharding.Mesh

    @property
    def model(self) -> TransformerDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel:
        model = self.placed.model
        assert isinstance(
            model, TransformerDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel
        ), type(model)
        return model


type LoadedTransformerRun = LoadedRun[
    LMBatchWithDocuments,
    TransformerPreparedWeights,
    LMBatchWithDocuments,
    TransformerPreparedMasking,
]
type LoadedQwenRun = LoadedRun[
    LMBatch, QwenPreparedWeights, LMBatchWithRouting[LMBatch], QwenPreparedMasking
]
type LoadedLMRun = LoadedTransformerRun | LoadedQwenRun
"""The two LM families a consumer can open; match on `placed.model` to recover which."""


def require_transformer_run(run: LoadedLMRun) -> LoadedTransformerRun:
    """Narrow consumers that require the shared transformer's document-aware input contract."""
    match run:
        case LoadedRun(placed=PlacedModel(model=TransformerDecomposedModel())):
            return run
        case LoadedRun(placed=PlacedModel(model=qwen36_moe.Qwen36MoeDecomposedModel())):
            raise TypeError("This consumer requires a document-aware transformer target")
        case other:
            raise AssertionError(f"not an LM run: {type(other.placed.model).__name__}")


class ConsumerLayout(BaseConfig):
    """The layout a read-only consumer restores a run onto — `open_run`'s parameter
    contract. A consumer re-places the frozen target and the restored decomposition on
    ITS OWN topology, never the run's, spelled the way a training run spells
    `runtime.mesh` / `runtime.sharding` and bundled so neither half can be named without
    the other."""

    mesh: MeshShape
    """The consumer's logical mesh; its world size must equal the process's device count."""
    sharding: PlacementSpec
    """The placement the consumer binds on that mesh — a preset name or an explicit table.
    Binding refuses a preset whose rows the run's site set cannot consume (the `-moe`
    presets on a run without routed-expert sites), so a run is opened only under a layout
    that fits it."""


SINGLE_DEVICE_RESIDENT_LAYOUT = ConsumerLayout(
    mesh=ResidentMeshShape(data=1, tp=1), sharding="zero1-replicated-resident"
)
"""The layout a consumer takes when its caller names none: the whole run on *the*
device, weights resident. `open_run` never reads it — it is the value the launching
edges (CLI flags, submission and deploy configs) default to, kept here only because this
module is the one every such edge may import."""

ONE_DEVICE_OWNER_LAYOUT = ConsumerLayout(
    mesh=HsdpMeshShape(replicate=1, fsdp=1, tp=1), sharding="owner"
)
"""The whole run on the process's one device under the owner layout, which a run's global
transformer CI function needs."""


def _consumer_decomposition_abstract[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    ci_fn: CIFnArchitecture[Conditioning],
    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: jax.sharding.Mesh,
) -> Decomposition[Conditioning]:
    rules = placed.placement
    assert rules is not None, "the consumer restore requires the bundle's resolved rules"
    initializer = seeded_ci_fn_initializer(ci_fn, placed.model.sites, rules)
    shape_dtype = jax.eval_shape(
        lambda: init_decomposition(placed, initializer, jax.random.PRNGKey(0))
    )
    component_shardings = placement.component_stacks_shardings(shape_dtype.components, rules)
    ci_fn_shardings = shape_dtype.ci_fn.shardings(mesh)

    # Floating leaves restore at the compute dtype `_prepare_read_only_consumer` casts them
    # to anyway: orbax casts each shard on the host as it reads it, so the device never
    # holds the fp32 decomposition beside its prepared copy (for one large decomposition, an
    # 84 GB peak on one device, past an H100's; 27 GB of bf16 instead of 54 GB of fp32 placed).
    def with_sharding(shape: jax.ShapeDtypeStruct, sharding: NamedSharding):
        dtype = COMPUTE_DT if jnp.issubdtype(shape.dtype, jnp.floating) else shape.dtype
        return jax.ShapeDtypeStruct(shape.shape, dtype, sharding=sharding)

    return Decomposition(
        components=jax.tree.map(with_sharding, shape_dtype.components, component_shardings),
        ci_fn=jax.tree.map(with_sharding, shape_dtype.ci_fn, ci_fn_shardings),
    )


def _restore_decomposition[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    ci_fn: CIFnArchitecture[Conditioning],
    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: jax.sharding.Mesh,
    run_dir: Path,
    step: int | None,
) -> tuple[Decomposition[Conditioning], int]:
    abstract = _consumer_decomposition_abstract(ci_fn, placed, mesh)
    checkpoint_root = run_dir / "ckpts"
    manager = make_read_only_checkpoint_manager(checkpoint_root)
    resolved_step = manager.latest_step() if step is None else step
    assert resolved_step is not None, f"no checkpoints under {checkpoint_root}"
    return restore_decomposition(manager, resolved_step, abstract), resolved_step


def open_run(
    run_dir: Path, step: int | None, *, data_root: Path, layout: ConsumerLayout
) -> LoadedLMRun:
    """Restore one decomposition and prepare its immutable offline compute state. Its
    floating values restore at the compute dtype (bf16), never as the checkpoint's fp32: a
    consumer that needs the raw fp32 values restores them itself.

    Args:
        run_dir: Run directory containing the product description and `ckpts`.
        step: Checkpoint step, or `None` for the latest complete step.
        data_root: Explicit root used to resolve named datasets and target caches.
        layout: The mesh and placement THIS consumer runs under, spanning exactly the
            process's devices. Placement binding refuses a layout the run's site set
            cannot consume.
    """
    assert layout.mesh.world_size == jax.device_count(), (
        f"consumer layout {layout.mesh} spans {layout.mesh.world_size} devices; "
        f"this process has {jax.device_count()}"
    )
    mesh = mesh_for_shape(layout.mesh)
    deliverable = load_deliverable(run_dir, data_root)
    # Consumers re-place with the replicated residual: sequence parallelism is a
    # training-step layout, and consumer masked forwards (eval probes) are not the
    # surface it exists for.
    placed = build_target(
        deliverable.target, mesh, data_root, layout.sharding, sequence_sharding="replicate"
    )

    def loaded[
        TargetIn: LMBatch | LMBatchWithDocuments,
        PreparedT: ComponentActivations,
        Conditioning,
        PreparedMaskingT,
    ](
        placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        ci_fn_arch: CIFnArchitecture[Conditioning],
    ) -> LoadedRun[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
        with jax.set_mesh(mesh):
            decomposition, resolved_step = _restore_decomposition(
                ci_fn_arch, placed, mesh, run_dir, step
            )
            prepared_weights, prepared_ci_fn = _prepare_read_only_consumer(
                placed, decomposition.components, decomposition.ci_fn
            )
            jax.block_until_ready((prepared_weights, prepared_ci_fn))
        return LoadedRun(
            run_id=run_dir.name,
            step=resolved_step,
            placed=placed,
            deliverable=deliverable,
            prepared_weights=prepared_weights,
            prepared_ci_fn=prepared_ci_fn,
            mesh=mesh,
        )

    match placed.model:
        case TransformerDecomposedModel() as inner:
            return loaded(
                PlacedModel(inner, placed.placement), require_unrouted_ci_fn_arch(deliverable.ci_fn)
            )
        case qwen36_moe.Qwen36MoeDecomposedModel() as inner:
            return loaded(PlacedModel(inner, placed.placement), deliverable.ci_fn)
        case other:
            raise AssertionError(f"not an LM target: {type(other).__name__}")


@dataclass(frozen=True)
class TargetAndComponents:
    """A decomposition checkpoint's frozen target and prepared component weights, unplaced
    on the process's one device. It holds no CI fn: its consumers substitute component
    activations and never evaluate CI."""

    run_id: str
    step: int
    model: TransformerDecomposedModel
    prepared_weights: TransformerPreparedWeights


def open_target_and_components(
    run_dir: Path,
    step: int,
    *,
    data_root: Path,
    weights_dtype: WeightsDtype,
    sites: Collection[str] | None = None,
) -> TargetAndComponents:
    """Read exactly a dense run's target (its config's `target` and `decomposition.sites`
    sections, `load_dense_target`) and checkpoint `step`'s component weights
    (`restore_components`), unplaced, on this process's one device, both in `weights_dtype`
    whatever the run trained in. The CI definition and the CI fn are never read: consumers
    that substitute component activations don't use them, and `open_run` is the reader for
    those that do.

    With `sites`, only the components of the decomposed sites of those sites' kinds load
    (`narrowed_to_kinds_of`): every `o_proj`'s for one `o_proj` site, a fraction of the
    memory and reading time of all of them; every other matrix is the target's own. Its
    attention runs as the run configured it where that can run, else as XLA's
    (`consumed_as`).

    Args:
        run_dir: Run directory containing the product description and `ckpts`.
        step: Checkpoint step.
        data_root: Explicit root used to resolve named datasets and target caches.
        weights_dtype: The dtype the target's weights and the components load in.
        sites: Decomposed sites whose kinds' components to load; None for every site's.
    """
    (device,) = jax.devices()
    target = consumed_as(load_dense_target(run_dir, data_root), weights_dtype, device.platform)
    if sites is not None:
        target = narrowed_to_kinds_of(target, sites)
    model = load_target(target, data_root)
    match model:
        case TransformerDecomposedModel():
            model = jax.device_put(model, device)
        case qwen36_moe.Qwen36MoeDecomposedModel():
            raise NotImplementedError("component substitution supports dense transformers only")
    single_device = SingleDeviceSharding(device)
    abstract = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=single_device),
        jax.eval_shape(lambda: init_component_stacks(model.sites, jax.random.PRNGKey(0))),
    )
    components = restore_components(
        make_read_only_checkpoint_manager(run_dir / "ckpts"), step, abstract
    )
    prepared_weights = _prepare_components(model, components, weights_jnp_dtype(weights_dtype))
    return TargetAndComponents(run_dir.name, step, model, jax.block_until_ready(prepared_weights))


def consumed_as(
    target: AnyLMTargetConfig, weights_dtype: WeightsDtype, platform: str
) -> AnyLMTargetConfig:
    """`target` with its weights in `weights_dtype`, on the device of JAX `platform`, its
    attention XLA's where the cuDNN flash attention a run trains with can't run: on the CPU,
    or on fp32 operands."""
    assert platform in ("gpu", "cpu"), f"a checkpoint opens on a GPU or the CPU, not {platform}"
    flash_runs = platform == "gpu" and weights_dtype == "bfloat16"
    return replace(
        target,
        weights_dtype=weights_dtype,
        attention_implementation=target.attention_implementation if flash_runs else "xla",
    )


def narrowed_to_kinds_of(target: AnyLMTargetConfig, sites: Collection[str]) -> AnyLMTargetConfig:
    """`target` decomposing only its sites of the kinds of `sites`, each a site it
    decomposes. A checkpoint keeps each kind's components as one group, so a kind is the
    least that restores."""
    match target:
        case TargetConfig():
            family = transformer.FAMILY
        case LlamaSimpleMLPTargetConfig():
            family = llama_simple_mlp.FAMILY
        case Qwen36MoeTargetConfig():
            raise NotImplementedError("component substitution supports dense transformers only")
    decomposed = {site.name for site in target.sites}
    unknown = sorted(set(sites) - decomposed)
    assert sites and not unknown, f"not decomposed sites of this checkpoint: {unknown or sites}"
    kinds = {family.parse(site)[1] for site in sites}
    return replace(
        target, sites=tuple(site for site in target.sites if family.parse(site.name)[1] in kinds)
    )


def dense_target_tokenizer_dir(run_dir: Path, *, data_root: Path) -> Path:
    """The local HF snapshot holding the tokenizer of a dense run's target, read from the
    same two config sections as `open_target_and_components`."""
    match load_dense_target(run_dir, data_root):
        case TargetConfig(model_name=model_name):
            return transformer.hf_snapshot_dir(model_name)
        case LlamaSimpleMLPTargetConfig() | Qwen36MoeTargetConfig() as other:
            raise NotImplementedError(f"only an HF target names its own tokenizer, not {other}")


@eqx.filter_jit
def _prepare_components(
    model: TransformerDecomposedModel, components: ComponentStacks, dtype: DTypeLike
) -> TransformerPreparedWeights:
    return model.prepare_compute_weights(cast_floating(components, dtype), None)


@dataclass(frozen=True)
class RunMetadata:
    """Target structure available without opening a checkpoint."""

    model_type: str
    n_blocks: int
    vocab_size: int


def run_metadata(run_dir: Path, *, data_root: Path) -> RunMetadata:
    """Read target topology without restoring a checkpoint.

    Args:
        run_dir: Run directory containing a current product description.
        data_root: Explicit root used to resolve target caches.
    """
    target = load_deliverable(run_dir, data_root).target
    match target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            return RunMetadata(
                model_type="LlamaSimpleMLP",
                n_blocks=simple_cfg.n_layer,
                vocab_size=simple_cfg.vocab_size,
            )
        case TargetConfig():
            variant = hf_model_variant(target.model_name)
            arch_cfg = variant.arch
            return RunMetadata(
                model_type=variant.model_type,
                n_blocks=arch_cfg.n_layer,
                vocab_size=arch_cfg.vocab_size,
            )
        case Qwen36MoeTargetConfig(arch=arch):
            return RunMetadata(
                model_type="Qwen3_5Moe",
                n_blocks=arch.n_layer,
                vocab_size=arch.vocab_size,
            )
