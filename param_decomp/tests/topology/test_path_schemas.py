"""Concrete-path <-> canonical-address round trips per model family."""

import pytest

from param_decomp.topology.canonical import CanonicalWeight
from param_decomp.topology.path_schemas import path_schema_for_model_type

QWEN36_MOE_CASES = {
    "layers.3.mlp.experts.gate_proj": "3.moe.gate",
    "layers.0.mlp.experts.up_proj": "0.moe.up",
    "layers.39.mlp.experts.down_proj": "39.moe.down",
    "layers.11.mlp.shared_expert.up_proj": "11.moe_shared.up",
    "layers.11.mlp.shared_expert.gate_proj": "11.moe_shared.gate",
    "layers.2.self_attn.q_proj": "2.attn.q",
    "embed_tokens": "embed",
    "lm_head": "output",
}


@pytest.mark.parametrize(("path", "canonical"), sorted(QWEN36_MOE_CASES.items()))
def test_qwen36_moe_round_trip(path: str, canonical: str) -> None:
    schema = path_schema_for_model_type("Qwen3_5Moe")
    weight = schema.parse_target_path(path)
    assert weight.canonical_str() == canonical
    assert schema.render_canonical_weight(weight) == path
    assert CanonicalWeight.parse(canonical) == weight


EXISTING_FAMILY_CASES = {
    ("Qwen3", "layers.5.mlp.gate_proj"): "5.glu.gate",
    ("Llama", "layers.5.self_attn.o_proj"): "5.attn.o",
    ("LlamaSimple", "h.2.mlp.down_proj"): "2.glu.down",
    ("LlamaSimpleMLP", "h.2.mlp.c_fc"): "2.mlp.up",
    ("GPT2", "h_torch.1.attn.c_attn"): "1.attn_fused.qkv",
    ("GPT2Simple", "h.0.mlp.down_proj"): "0.mlp.down",
}


@pytest.mark.parametrize(
    ("model_type", "path", "canonical"),
    sorted((model, path, canonical) for (model, path), canonical in EXISTING_FAMILY_CASES.items()),
)
def test_existing_families_round_trip(model_type: str, path: str, canonical: str) -> None:
    schema = path_schema_for_model_type(model_type)
    weight = schema.parse_target_path(path)
    assert weight.canonical_str() == canonical
    assert schema.render_canonical_weight(weight) == path


def test_unknown_block_path_refuses() -> None:
    schema = path_schema_for_model_type("Qwen3_5Moe")
    with pytest.raises(AssertionError):
        schema.parse_target_path("layers.3.mlp.gate_proj")
