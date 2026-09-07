"""Restore a finished JAX language-model decomposition for read-only analysis.

`open_jax_run` checks the saved deliverable and checkpoint step, reconstructs the target
on the layout the CALLER names (`ConsumerLayout` — never derived from the run), restores
component weights and the CI function, and returns a `LoadedJaxRun` ready for harvesting
without any training state or optimizer."""

import dataclasses
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
from jax.sharding import NamedSharding
from jaxtyping import Array

from param_decomp.core import placement
from param_decomp.core.base_config import BaseConfig
from param_decomp.core.checkpoint import make_read_only_checkpoint_manager, restore_decomposition
from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIFn,
    GlobalMLPCIFn,
    MoEChunkwiseTransformerCIFn,
    PlacedCIFn,
    resolve_ci_placement,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import MeshShape, PlacementSpec, ResidentMeshShape, SequenceSharding
from param_decomp.core.init_placed import (
    ComponentInitializer,
    ci_fn_shardings,
    random_component_initializer,
)
from param_decomp.core.model import PlacedModel, prepare_compute_weights
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import mesh_for_shape, place_target
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm.config import LMCIFnArch, hf_model_variant
from param_decomp.experiments.lm.deliverable import ResolvedDeliverable, load_deliverable
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    Qwen36MoeTargetConfig,
    TargetConfig,
    weights_jnp_dtype,
)
from param_decomp.infra import pretrain_cache
from param_decomp.target_ports.llama import AttentionImplementation
from param_decomp.targets import glu_transformer, llama_simple_mlp, qwen36_moe
from param_decomp.targets.glu_transformer import GLUDecomposedModel, glu_site_specs
from param_decomp.targets.lm_output import LMOutput

LMCIFn = ChunkwiseTransformerCIFn | GlobalMLPCIFn | MoEChunkwiseTransformerCIFn


def _with_attention_implementation(
    model: GLUDecomposedModel,
    implementation: AttentionImplementation,
) -> GLUDecomposedModel:
    attn = replace(model.stacked.attn, implementation=implementation)
    return replace(model, stacked=replace(model.stacked, attn=attn))


def load_target(
    target: AnyLMTargetConfig, data_root: Path
) -> GLUDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel:
    """The unplaced frozen target for one target config — the load half of `build_target`,
    for consumers that never place onto a real mesh (the AOT fit check abstracts the
    leaves onto compile-only devices instead).
    SimpleMLP reads its pretrain cache under `data_root` (no network); the HF families
    read the HF snapshot. Both cast their weights to the config's `weights_dtype` on
    read — this is the ONLY place that dtype is applied, so train and consume load the
    same target."""
    match target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            sites = llama_simple_mlp.site_specs(simple_cfg, target.sites)
            loaded_model = llama_simple_mlp.load_decomposed_lm_from_pretrain_cache(
                cache_dir, simple_cfg, sites, weights_jnp_dtype(target.weights_dtype)
            )
        case TargetConfig():
            variant = hf_model_variant(target.model_name)
            arch_cfg = variant.arch_config()
            sites = glu_site_specs(arch_cfg, target.sites)
            loaded_model = variant.load(
                target.model_name,
                arch_cfg,
                sites,
                weights_jnp_dtype(target.weights_dtype),
            )
        case Qwen36MoeTargetConfig():
            moe_cfg = qwen36_moe.qwen36_35b_a3b_config()
            moe_sites = qwen36_moe.qwen36_moe_site_specs(moe_cfg, target.sites)
            # The hybrid mixers have no attention-implementation knob; return directly.
            # The authored experts_execution and output_edge apply HERE, the one
            # resolved boundary — the builders construct the production defaults.
            return dataclasses.replace(
                qwen36_moe.load_decomposed_qwen36_moe_from_hf(
                    target.model_name,
                    moe_cfg,
                    moe_sites,
                    weights_jnp_dtype(target.weights_dtype),
                    target.attention_implementation,
                ),
                experts_execution=target.experts_execution,
                output_edge=target.output_edge,
            )
    return _with_attention_implementation(loaded_model, target.attention_implementation)


def component_initializer_for(target: AnyLMTargetConfig) -> ComponentInitializer[LMOutput, Any]:
    """Resolve the one run-start V/U initializer from the authored target family config."""
    match target.component_initialization:
        case "random":
            return random_component_initializer
        case "neuron_aligned":
            match target:
                case TargetConfig() | LlamaSimpleMLPTargetConfig():
                    return cast(
                        ComponentInitializer[LMOutput, Any],
                        glu_transformer.neuron_aligned_component_initializer,
                    )


def target_vocab_size(placed: PlacedModel[LMOutput]) -> int:
    """The vocabulary a placed LM target's token ids index — its embedding's row count."""
    match placed.model:
        case GLUDecomposedModel(embed=embed) | qwen36_moe.Qwen36MoeDecomposedModel(embed=embed):
            return embed.shape[0]
        case other:
            raise AssertionError(f"not an LM target: {type(other).__name__}")


def build_target(
    target: AnyLMTargetConfig,
    mesh: jax.sharding.Mesh,
    data_root: Path,
    sharding: PlacementSpec,
    sequence_sharding: SequenceSharding,
) -> PlacedModel[LMOutput]:
    """Build and place the frozen target shared by training and every offline consumer.

    The bundle's `.model` (an `eqx.Module`) IS the frozen target — it carries the full
    model weights (embedding included) as fields and embeds its token input internally;
    `.placement` is the resolved rules for `sharding` and `sequence_sharding`."""
    loaded_model = load_target(target, data_root)
    placement_rules = placement.from_config(
        sharding, mesh, loaded_model.sites, sequence_sharding=sequence_sharding
    )
    return place_target(loaded_model, placement_rules)


@eqx.filter_jit
def _prepare_read_only_consumer(
    placed: PlacedModel[LMOutput],
    components: ComponentStacks,
    ci_fn: LMCIFn,
) -> tuple[dict[str, dict[str, Array]], LMCIFn]:
    return prepare_compute_weights(placed, components), cast_floating(ci_fn, COMPUTE_DT)


@dataclass(frozen=True)
class LoadedJaxRun:
    """A restored decomposition prepared only for inference and analysis.

    `placed` pairs the frozen target with the consumer's resolved rules (the pairing is
    structural — a forward can never see one without the other), `prepared_weights` are
    the component weights in the model's compute form, `ci_fn` is the CI function paired
    with its resolved rows, and `mesh` is the layout consumers run under
    (`jax.set_mesh(mesh)`). Training state and optimizer state are intentionally absent.
    """

    run_id: str
    step: int
    placed: PlacedModel[LMOutput]
    deliverable: ResolvedDeliverable
    prepared_weights: dict[str, dict[str, Array]]
    ci_fn: PlacedCIFn
    mesh: jax.sharding.Mesh

    @property
    def model(self) -> GLUDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel:
        model = self.placed.model
        assert isinstance(model, GLUDecomposedModel | qwen36_moe.Qwen36MoeDecomposedModel), type(
            model
        )
        return model


class ConsumerLayout(BaseConfig):
    """The layout a read-only consumer restores a run onto — `open_jax_run`'s parameter
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
device, weights resident. `open_jax_run` never reads it — it is the value the launching
edges (CLI flags, submission and deploy configs) default to, kept here only because this
module is the one every such edge may import."""


def _consumer_decomposition_abstract(
    ci_fn: LMCIFnArch,
    placed: PlacedModel[LMOutput],
    mesh: jax.sharding.Mesh,
) -> Decomposition:
    rules = placed.placement
    assert rules is not None, "the consumer restore requires the bundle's resolved rules"
    shape_dtype = jax.eval_shape(lambda: init_decomposition(placed, ci_fn, jax.random.PRNGKey(0)))
    shardings = Decomposition(
        components=cast(Any, placement.component_stacks_shardings(shape_dtype.components, rules)),
        ci_fn=ci_fn_shardings(shape_dtype.ci_fn, mesh, resolve_ci_placement(ci_fn, rules)),
    )

    def with_sharding(shape: jax.ShapeDtypeStruct, sharding: NamedSharding):
        assert isinstance(sharding, NamedSharding), type(sharding)
        return jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=sharding)

    return jax.tree.map(
        with_sharding,
        shape_dtype,
        shardings,
        is_leaf=lambda value: isinstance(value, NamedSharding),
    )


def _restore_decomposition(
    ci_fn: LMCIFnArch,
    placed: PlacedModel[LMOutput],
    mesh: jax.sharding.Mesh,
    run_dir: Path,
    step: int | None,
) -> tuple[Decomposition, int]:
    abstract = _consumer_decomposition_abstract(ci_fn, placed, mesh)
    checkpoint_root = run_dir / "ckpts"
    manager = make_read_only_checkpoint_manager(checkpoint_root)
    resolved_step = manager.latest_step() if step is None else step
    assert resolved_step is not None, f"no checkpoints under {checkpoint_root}"
    return restore_decomposition(manager, resolved_step, abstract), resolved_step


def open_jax_run(
    run_dir: Path, step: int | None, *, data_root: Path, layout: ConsumerLayout
) -> LoadedJaxRun:
    """Restore one decomposition and prepare its immutable offline compute state.

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
    decomposition, resolved_step = _restore_decomposition(
        deliverable.ci_fn, placed, mesh, run_dir, step
    )
    assert isinstance(decomposition.components, ComponentStacks)
    assert isinstance(decomposition.ci_fn, LMCIFn)
    with jax.set_mesh(mesh):
        prepared_weights, ci_fn = _prepare_read_only_consumer(
            placed, decomposition.components, decomposition.ci_fn
        )
        jax.block_until_ready((prepared_weights, ci_fn))
    del decomposition

    rules = placed.placement
    assert rules is not None, "build_target always resolves the consumer rules"
    return LoadedJaxRun(
        run_id=run_dir.name,
        step=resolved_step,
        placed=placed,
        deliverable=deliverable,
        prepared_weights=prepared_weights,
        ci_fn=PlacedCIFn(fn=ci_fn, placement=resolve_ci_placement(deliverable.ci_fn, rules)),
        mesh=mesh,
    )


@dataclass(frozen=True)
class RunMetadata:
    """Target structure available without opening a checkpoint."""

    model_type: str
    n_blocks: int
    vocab_size: int
    layer_activation_sizes: list[tuple[str, int]]


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
                layer_activation_sizes=[(site.name, site.C) for site in target.sites],
            )
        case TargetConfig():
            variant = hf_model_variant(target.model_name)
            arch_cfg = variant.arch_config()
            return RunMetadata(
                model_type=variant.model_type,
                n_blocks=arch_cfg.n_layer,
                vocab_size=arch_cfg.vocab_size,
                layer_activation_sizes=[(site.name, site.C) for site in target.sites],
            )
        case Qwen36MoeTargetConfig():
            moe_cfg = qwen36_moe.qwen36_35b_a3b_config()
            return RunMetadata(
                model_type="Qwen3_5Moe",
                n_blocks=moe_cfg.n_layer,
                vocab_size=moe_cfg.vocab_size,
                layer_activation_sizes=[(site.name, site.C) for site in target.sites],
            )
