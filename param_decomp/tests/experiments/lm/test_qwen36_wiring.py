"""The qwen36_moe composition wiring: authored config → resolved decomposition.

Resolution-level only — no 35B weights are loaded. The tiny-model forward and the
expert-blocked engine behavior are pinned by the target and core suites."""

from pathlib import Path

import pytest

from param_decomp.core.ci_fn import ChunkwiseTransformerCIArch
from param_decomp.core.components import Dense, ExpertBlocked
from param_decomp.experiments.lm.config import (
    QWEN36_MOE_MODEL_CLASS,
    QWEN36_MOE_MODEL_NAME,
    LMDecompositionConfig,
    LMTargetConfig,
    resolve_decomposition,
    resolve_lm_ci_arch,
)
from param_decomp.experiments.lm.resolved import Qwen36MoeTargetConfig

N_LAYER, N_EXPERTS, D_RESID = 40, 256, 2048


def _target_config(model_class: str = QWEN36_MOE_MODEL_CLASS) -> LMTargetConfig:
    return LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": model_class,
                "model_name": QWEN36_MOE_MODEL_NAME,
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
        }
    )


def _decomposition(
    cs: dict[str, int],
    layers: dict[str, object] | None = None,
    input_tap: str = "first_block_resid",
) -> LMDecompositionConfig:
    return LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "qwen36_moe",
                "layers": layers if layers is not None else {"kind": "all"},
                "cs": cs,
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 4,
                "input_tap": input_tap,
                "d_model": 8,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 2},
                "ffn": {"kind": "gelu", "hidden": 16},
            },
        }
    )


def test_qwen36_resolution_builds_expert_blocked_specs(tmp_path: Path):
    cs = {"experts_gate": 512, "experts_down": 256, "shared_gate": 4}
    resolved = resolve_decomposition(_target_config(), _decomposition(cs), tmp_path)

    target = resolved.target
    assert isinstance(target, Qwen36MoeTargetConfig)
    assert target.model_name == QWEN36_MOE_MODEL_NAME
    # The authored SDPA lowering reaches the resolved target; silently dropping it
    # would leave the placed run on cuDNN's rejected head-dimension-256 training graph.
    assert target.attention_implementation == "xla"
    assert len(target.sites) == N_LAYER * len(cs)
    assert len(resolved.tree.blocks) == N_LAYER

    by_group = {spec.group: spec.factorization for spec in resolved.site_specs}
    assert by_group["experts_gate"] == ExpertBlocked(
        n_experts=N_EXPERTS, d_in=D_RESID, d_out=512, c_per_expert=2
    )
    assert by_group["experts_down"] == ExpertBlocked(
        n_experts=N_EXPERTS, d_in=512, d_out=D_RESID, c_per_expert=1
    )
    assert isinstance(by_group["shared_gate"], Dense)

    ci_arch = resolve_lm_ci_arch(resolved.tree, _decomposition(cs).ci, resolved.grammar)
    assert isinstance(ci_arch, ChunkwiseTransformerCIArch)
    assert len(ci_arch.chunks) == N_LAYER // 4
    assert ci_arch.input_dim == D_RESID


def test_qwen36_resolution_refusals(tmp_path: Path):
    cs = {"experts_gate": 512}
    with pytest.raises(AssertionError, match="whole-grid"):
        resolve_decomposition(
            _target_config(),
            _decomposition(cs, layers={"kind": "range", "start": 0, "end": 8}),
            tmp_path,
        )
    with pytest.raises(AssertionError, match="residual-boundary taps"):
        resolve_decomposition(
            _target_config(), _decomposition(cs, input_tap="all_block_taps"), tmp_path
        )
    with pytest.raises(AssertionError):
        resolve_decomposition(
            _target_config(model_class="transformers.Qwen3ForCausalLM"),
            _decomposition(cs),
            tmp_path,
        )
    with pytest.raises(AssertionError, match="multiple of n_experts"):
        resolve_decomposition(_target_config(), _decomposition({"experts_gate": 300}), tmp_path)


def test_qwen36_output_edge_resolves_and_gates(tmp_path: Path):
    """`target.output_edge` is authored on every config and reaches the resolved target
    as the typed edge; an omitted edge refuses at parse, a non-dividing chunk count and
    a non-qwen36 family both refuse at resolve."""
    from pydantic import ValidationError

    from param_decomp.targets.qwen36_moe import MaterializedOutputEdge, StreamedOutputEdge

    cs = {"experts_gate": 512}
    materialized = resolve_decomposition(_target_config(), _decomposition(cs), tmp_path).target
    assert isinstance(materialized, Qwen36MoeTargetConfig)
    assert materialized.output_edge == MaterializedOutputEdge()

    unauthored = {
        key: value
        for key, value in _target_config().model_dump(mode="json").items()
        if key != "output_edge"
    }
    with pytest.raises(ValidationError, match="output_edge"):
        LMTargetConfig.model_validate(unauthored)

    streamed_config = LMTargetConfig.model_validate(
        {
            **_target_config().model_dump(mode="json"),
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 32},
        }
    )
    streamed = resolve_decomposition(streamed_config, _decomposition(cs), tmp_path).target
    assert isinstance(streamed, Qwen36MoeTargetConfig)
    assert streamed.output_edge == StreamedOutputEdge(n_vocab_chunks=32)

    non_dividing = LMTargetConfig.model_validate(
        {
            **_target_config().model_dump(mode="json"),
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 7},
        }
    )
    with pytest.raises(AssertionError, match="must divide the"):
        resolve_decomposition(non_dividing, _decomposition(cs), tmp_path)

    glu_streamed = LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": "transformers.Qwen3ForCausalLM",
                "model_name": "Qwen/Qwen3-8B-Base",
            },
            "attention_implementation": "auto",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 32},
        }
    )
    from param_decomp.experiments.lm.config import GluTransformerCSpec, LMDecompositionConfig

    glu_decomposition = LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "all"},
                "cs": {"q": 4},
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 4,
                "input_tap": "first_block_resid",
                "d_model": 8,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 2},
                "ffn": {"kind": "gelu", "hidden": 16},
            },
        }
    )
    assert isinstance(glu_decomposition.sites, GluTransformerCSpec)
    with pytest.raises(AssertionError, match="serves the qwen36_moe family only"):
        resolve_decomposition(glu_streamed, glu_decomposition, tmp_path)
