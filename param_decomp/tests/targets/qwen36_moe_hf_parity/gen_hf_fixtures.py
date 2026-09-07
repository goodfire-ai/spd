"""Generate the Qwen3.6-MoE HF-parity goldens — a TORCH-ENV script (the repo venv is
torch-free; run it in a throwaway venv). Needs a transformers with `qwen3_5_moe`
(4.57 lineage; install from git main if the release lacks it):

    uv venv /tmp/qwen36-golden --python 3.12
    VIRTUAL_ENV=/tmp/qwen36-golden uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=/tmp/qwen36-golden uv pip install "transformers>=4.57" numpy
    /tmp/qwen36-golden/bin/python param_decomp/targets/tests/qwen36_moe_hf_parity/gen_hf_fixtures.py         # tiny
    /tmp/qwen36-golden/bin/python param_decomp/targets/tests/qwen36_moe_hf_parity/gen_hf_fixtures.py --real  # 35B

Tiny golden (`qwen36_moe_tiny_hf_fixtures.npz`): a seeded random `Qwen3_5MoeForCausalLM`
(the text-only wrapper — no vision tower, no MTP) at an 8-layer / 2-stage toy config,
fp32, eager attention — the exact-architecture check: the gated-DeltaNet recurrence
(conv, decay, l2norm, gated output norm), gated full attention with zero-centered
QK-norm and partial RoPE (mrope degenerating on text position ids), the MoE router's
softmax→topk→renormalize, fused experts, the shared expert and its scalar gate, and the
untied head. Every parameter is re-randomized after construction (sorted-name order) so
the router and the zero-centered norms are non-trivial and top-k has no ties. The full
state dict rides in the npz under `sd::`-prefixed keys.

Real golden (`qwen36_35b_real_logits.npz`): `Qwen/Qwen3.6-35B-A3B` bf16 from the HF
cache, a few fixed prompts, final-position fp32 logits — the weight-loading end-to-end
check (slow test; generate on a machine with the snapshot cached, NOT a laptop)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

HERE = Path(__file__).resolve().parent

TINY_CONFIG = dict(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=8,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    },
    linear_conv_kernel_dim=4,
    linear_key_head_dim=4,
    linear_num_key_heads=2,
    linear_value_head_dim=4,
    linear_num_value_heads=4,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=12,
    num_experts=4,
    num_experts_per_tok=2,
    rms_norm_eps=1e-6,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    attention_bias=False,
    attention_dropout=0.0,
    use_cache=False,
)

REAL_MODEL = "Qwen/Qwen3.6-35B-A3B"
REAL_PROMPTS = (
    "The capital of France is Paris, and the capital of Germany is",
    "In 1859, Charles Darwin published On the Origin of Species, which",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The mitochondria is the powerhouse of the cell, and the nucleus stores the",
)
REAL_SEQ_LEN = 12


def gen_tiny() -> None:
    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(**TINY_CONFIG)
    model = Qwen3_5MoeForCausalLM._from_config(cfg, attn_implementation="eager").eval()
    # Re-randomize EVERY parameter (sorted-name order, one seeded generator): HF init
    # zeroes the router and the zero-centered norm weights, and equal router logits
    # would make top-k tie-break-order-dependent between torch and JAX.
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for _name, param in sorted(model.named_parameters()):
            param.uniform_(-0.3, 0.3, generator=generator)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16), generator=torch.Generator().manual_seed(1))
    residuals = []
    handles = [
        layer.register_forward_pre_hook(lambda _module, args: residuals.append(args[0].detach()))
        for layer in model.model.layers
    ]
    handles.append(
        model.model.norm.register_forward_pre_hook(
            lambda _module, args: residuals.append(args[0].detach())
        )
    )
    with torch.no_grad():
        logits = model(tokens).logits
    for handle in handles:
        handle.remove()
    assert len(residuals) == cfg.num_hidden_layers + 1
    arrays = {f"sd::{k}": v.numpy() for k, v in model.state_dict().items()}
    arrays.update({f"resid::{i}": x.numpy() for i, x in enumerate(residuals)})
    import transformers

    np.savez_compressed(
        HERE / "qwen36_moe_tiny_hf_fixtures.npz",
        **arrays,
        tokens=tokens.numpy(),
        logits=logits.numpy(),
        config_json=np.array(json.dumps(TINY_CONFIG)),
        transformers_version=np.array(transformers.__version__),
    )
    print(f"tiny golden: {len(arrays)} tensors, logits {tuple(logits.shape)}")


def gen_real() -> None:
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    ids = []
    for prompt in REAL_PROMPTS:
        prompt_ids = tokenizer(prompt).input_ids
        assert len(prompt_ids) >= REAL_SEQ_LEN, (prompt, len(prompt_ids))
        ids.append(prompt_ids[:REAL_SEQ_LEN])
    tokens = torch.tensor(ids)
    model = Qwen3_5MoeForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        final_logits = model(tokens).logits[:, -1, :].float()
    import transformers

    np.savez_compressed(
        HERE / "qwen36_35b_real_logits.npz",
        tokens=tokens.numpy(),
        final_logits=final_logits.numpy(),
        transformers_version=np.array(transformers.__version__),
    )
    print(f"real golden: tokens {tuple(tokens.shape)}, final logits {tuple(final_logits.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="generate the Qwen3.6-35B-A3B golden")
    if parser.parse_args().real:
        gen_real()
    else:
        gen_tiny()
