"""Tests for TransformerTopology across all supported architectures.

Each test creates a small model of the given architecture, wraps it in
ComponentModel with realistic target module patterns (from the configs),
and verifies that TransformerTopology produces the correct topology.
"""

import pytest
import torch
from torch import nn

from spd.models.component_model import ComponentModel
from spd.topology import TransformerTopology
from spd.utils.module_utils import expand_module_patterns

# Small model hyperparams to keep tests fast
_SMALL_VOCAB = 128
_SMALL_EMBD = 32
_SMALL_N_HEAD = 2
_SMALL_BLOCK_SIZE = 16
_C = 4  # components per module


def _make_component_model(
    target_model: torch.nn.Module,
    target_patterns: list[str],
) -> ComponentModel:
    """Build a ComponentModel from patterns."""
    from spd.configs import ModulePatternInfoConfig

    module_path_info = expand_module_patterns(
        target_model,
        [ModulePatternInfoConfig(module_pattern=p, C=_C) for p in target_patterns],
    )
    from spd.configs import LayerwiseCiConfig

    model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr="idx_0",
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GPT2Simple — standard variant (q_proj, k_proj, v_proj, o_proj)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gpt2_simple():
    from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig

    config = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=_SMALL_BLOCK_SIZE,
        vocab_size=_SMALL_VOCAB,
        n_layer=2,
        n_head=_SMALL_N_HEAD,
        n_embd=_SMALL_EMBD,
        flash_attention=False,
    )
    target = GPT2Simple(config)
    target.eval()
    target.requires_grad_(False)

    patterns = [
        "h.*.mlp.c_fc",
        "h.*.mlp.down_proj",
        "h.*.attn.q_proj",
        "h.*.attn.k_proj",
        "h.*.attn.v_proj",
        "h.*.attn.o_proj",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    assert adapter.embedding_path == "wte"
    assert isinstance(adapter.embedding_module, torch.nn.Embedding)
    # lm_head is always resolved (needed for output attributions)
    assert adapter.unembed_path == "lm_head"
    assert isinstance(adapter.unembed_module, nn.Linear)

    # Cross-seq: k_proj and v_proj are KV, o_proj is O
    assert all("k_proj" in p or "v_proj" in p for p in adapter.kv_paths)
    assert all("o_proj" in p for p in adapter.o_paths)
    # 2 layers * 2 roles = 4 kv paths
    assert len(adapter.kv_paths) == 4
    # 2 layers * 1 role = 2 o paths
    assert len(adapter.o_paths) == 2

    # QKV grouping
    assert "qkv" in adapter.role_groups
    assert adapter.role_groups["qkv"] == ["q_proj", "k_proj", "v_proj"]

    assert adapter.display_names == {"lm_head": "W_U"}

    # Role order should follow pattern order (deduplicated)
    assert adapter.role_order == ["c_fc", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# GPT2Simple — subset of target modules (no q_proj decomposed)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gpt2_simple_partial_targets():
    from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig

    config = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=_SMALL_BLOCK_SIZE,
        vocab_size=_SMALL_VOCAB,
        n_layer=2,
        n_head=_SMALL_N_HEAD,
        n_embd=_SMALL_EMBD,
        flash_attention=False,
    )
    target = GPT2Simple(config)
    target.eval()
    target.requires_grad_(False)

    # Only decomposing a subset of modules (no q_proj)
    patterns = [
        "h.*.mlp.c_fc",
        "h.*.mlp.down_proj",
        "h.*.attn.k_proj",
        "h.*.attn.v_proj",
        "h.*.attn.o_proj",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    assert adapter.embedding_path == "wte"
    assert all("k_proj" in p or "v_proj" in p for p in adapter.kv_paths)
    assert len(adapter.kv_paths) == 4
    assert len(adapter.o_paths) == 2

    # k_proj and v_proj are still present from the qkv_group, so they get grouped
    assert adapter.role_groups == {"qkv": ["k_proj", "v_proj"]}


# ---------------------------------------------------------------------------
# LlamaSimple
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_llama_simple():
    from spd.pretrain.models.llama_simple import LlamaSimple, LlamaSimpleConfig

    config = LlamaSimpleConfig(
        model_type="LlamaSimple",
        block_size=_SMALL_BLOCK_SIZE,
        vocab_size=_SMALL_VOCAB,
        n_layer=2,
        n_head=_SMALL_N_HEAD,
        n_embd=_SMALL_EMBD,
        n_intermediate=_SMALL_EMBD * 2,
        n_key_value_heads=1,
        rotary_dim=_SMALL_EMBD // _SMALL_N_HEAD,
        flash_attention=False,
    )
    target = LlamaSimple(config)
    target.eval()
    target.requires_grad_(False)

    patterns = [
        "h.*.mlp.gate_proj",
        "h.*.mlp.up_proj",
        "h.*.mlp.down_proj",
        "h.*.attn.q_proj",
        "h.*.attn.k_proj",
        "h.*.attn.v_proj",
        "h.*.attn.o_proj",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    assert adapter.embedding_path == "wte"
    assert adapter.unembed_path == "lm_head"

    # Cross-seq detection
    assert len(adapter.kv_paths) == 4  # 2 layers * (k_proj + v_proj)
    assert len(adapter.o_paths) == 2  # 2 layers * o_proj

    # Role grouping: QKV and SwiGLU
    assert adapter.role_groups == {
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "swiglu": ["gate_proj", "up_proj"],
    }

    assert adapter.role_order == [
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


# ---------------------------------------------------------------------------
# LlamaSimpleMLP
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_llama_simple_mlp():
    from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

    config = LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        block_size=_SMALL_BLOCK_SIZE,
        vocab_size=_SMALL_VOCAB,
        n_layer=1,
        n_head=_SMALL_N_HEAD,
        n_embd=_SMALL_EMBD,
        n_intermediate=_SMALL_EMBD * 2,
        n_key_value_heads=1,
        rotary_dim=_SMALL_EMBD // _SMALL_N_HEAD,
        flash_attention=False,
    )
    target = LlamaSimpleMLP(config)
    target.eval()
    target.requires_grad_(False)

    patterns = [
        "h.*.mlp.c_fc",
        "h.*.mlp.down_proj",
        "h.*.attn.q_proj",
        "h.*.attn.k_proj",
        "h.*.attn.v_proj",
        "h.*.attn.o_proj",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    assert adapter.embedding_path == "wte"
    assert len(adapter.kv_paths) == 2  # 1 layer * (k_proj + v_proj)
    assert len(adapter.o_paths) == 1  # 1 layer * o_proj
    assert adapter.role_groups == {"qkv": ["q_proj", "k_proj", "v_proj"]}


# ---------------------------------------------------------------------------
# HuggingFace GPT2LMHeadModel
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hf_gpt2():
    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=_SMALL_VOCAB,
        n_embd=_SMALL_EMBD,
        n_layer=2,
        n_head=_SMALL_N_HEAD,
        n_positions=_SMALL_BLOCK_SIZE,
    )
    target = GPT2LMHeadModel(config)
    target.eval()
    target.requires_grad_(False)

    # Typical HF GPT2 patterns: specific layer index
    patterns = [
        "transformer.h.1.attn.c_attn",
        "transformer.h.1.mlp.c_fc",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    assert adapter.embedding_path == "transformer.wte"
    assert isinstance(adapter.embedding_module, torch.nn.Embedding)
    assert adapter.unembed_path == "lm_head"

    # c_attn is KV role for HF GPT2
    assert adapter.kv_paths == frozenset({"transformer.h.1.attn.c_attn"})
    # No o_paths: c_proj is an o_role but not in our target patterns
    assert adapter.o_paths == frozenset()

    # No QKV grouping (c_attn is fused)
    assert adapter.role_groups == {}

    assert adapter.display_names == {"lm_head": "W_U"}


# ---------------------------------------------------------------------------
# Cross-sequence pair detection
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cross_seq_pair_detection():
    """Verify is_cross_seq_pair works correctly with same/different blocks."""
    from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig

    config = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=_SMALL_BLOCK_SIZE,
        vocab_size=_SMALL_VOCAB,
        n_layer=2,
        n_head=_SMALL_N_HEAD,
        n_embd=_SMALL_EMBD,
        flash_attention=False,
    )
    target = GPT2Simple(config)
    target.eval()
    target.requires_grad_(False)

    patterns = [
        "h.*.attn.q_proj",
        "h.*.attn.k_proj",
        "h.*.attn.v_proj",
        "h.*.attn.o_proj",
        "h.*.mlp.c_fc",
    ]
    model = _make_component_model(target, patterns)
    adapter = TransformerTopology(model)

    # Same block: k_proj -> o_proj in block 0 should be cross-seq
    assert adapter.is_cross_seq_pair("h.0.attn.k_proj", "h.0.attn.o_proj") is True
    assert adapter.is_cross_seq_pair("h.0.attn.v_proj", "h.0.attn.o_proj") is True

    # Different blocks: should NOT be cross-seq
    assert adapter.is_cross_seq_pair("h.0.attn.k_proj", "h.1.attn.o_proj") is False

    # Non-attention roles: should NOT be cross-seq
    assert adapter.is_cross_seq_pair("h.0.mlp.c_fc", "h.0.attn.o_proj") is False
    assert adapter.is_cross_seq_pair("h.0.attn.q_proj", "h.0.attn.o_proj") is False
