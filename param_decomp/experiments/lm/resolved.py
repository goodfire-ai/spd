"""Runtime objects resolved from an authored LM config."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import jax.numpy as jnp
from jax.typing import DTypeLike

from param_decomp.core.built_run import BuiltRun
from param_decomp.core.components import SiteC
from param_decomp.core.configs import PDConfig, TargetedPDConfig
from param_decomp.target_ports.llama import AttentionImplementation
from param_decomp.targets.qwen36_moe import ExpertsExecution, OutputEdge

WeightsDtype = Literal["float32", "bfloat16"]
ComponentInitialization = Literal["random", "neuron_aligned"]


def weights_jnp_dtype(dtype: WeightsDtype) -> DTypeLike:
    """The authored frozen-target dtype as the array dtype the target loaders cast to."""
    match dtype:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16


@dataclass(frozen=True)
class ResolvedLMData:
    """Pre-tokenized parquet shard directories: `dir` trains; `eval_dir` is the held-out
    split the eval pass reads."""

    dir: Path
    eval_dir: Path


@dataclass(frozen=True)
class TargetConfig:
    """An HF GLU-transformer target (`model_name` must be in `HF_MODEL_VARIANTS` —
    Llama-3.1-8B or a registered Qwen3 checkpoint)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation
    component_initialization: ComponentInitialization

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports. `HFWeights` casts every tensor on
    read, so the family loaders honour whichever of the two the config names. A config
    requesting a dtype outside this set is refused at convert time — no silent downgrade
    (issue #727)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` lab-pretrained target (`param_decomp.targets.llama_simple_mlp`);
    weights from the store entry `pretrain_run_path` resolves to
    (`infra.pretrain_cache.resolved_cache_dir`)."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation
    component_initialization: ComponentInitialization

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports — `_checkpoint_weight_getter` casts
    every safetensor on read. See `TargetConfig.supported_weights_dtypes`."""


@dataclass(frozen=True)
class Qwen36MoeTargetConfig:
    """The Qwen3.6-35B-A3B MoE target (`param_decomp.targets.qwen36_moe`): the one
    registered MoE checkpoint. Expert sites decompose expert-locally, the hybrid
    DeltaNet/attention mixers stay frozen, and only random V/U init exists."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (whole-grid per kind)."""
    weights_dtype: WeightsDtype
    attention_implementation: AttentionImplementation
    """The full-attention SDPA lowering. The maintained 35B configs author `xla` because
    cuDNN rejects this family's head-dimension-256 training graph; the choice remains
    explicit rather than silently falling back."""
    experts_execution: ExpertsExecution
    """Which decomposed-expert execution the masked forwards run
    (`targets.qwen36_moe.ExpertsExecution`) — a seat-authored pricing/arch choice."""
    output_edge: OutputEdge
    """The model-output edge (`targets.qwen36_moe.OutputEdge`): materialized logits, or
    the factored streamed package whose comparisons chunk the 248k vocab axis."""
    component_initialization: Literal["random"]

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """See `TargetConfig.supported_weights_dtypes`; `HFWeights` casts every tensor on read."""


AnyLMTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig | Qwen36MoeTargetConfig
"""The closed set of LM target configs — what every LM `BuiltRun` carries and every LM
consumer (`build_target`, `run_metadata`, the targeted tokenizer route) dispatches on.
Non-LM targets (the toys) satisfy only the core `TargetSites` protocol and never enter
the LM aliases below."""


LMRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, PDConfig]
LMTargetedRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, TargetedPDConfig]
LMAnyRun = LMRun | LMTargetedRun
"""The stored-run consumers' view: the closed union of run shapes — consumers read only
the sections the shapes share."""
