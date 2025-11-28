# %%
"""Compute local attributions for a single prompt."""

import gzip
import json

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.scripts.calc_global_attributions import (
    get_sources_by_target,
    is_qkv_to_o_pair,
    validate_attention_pair_structure,
)
from spd.scripts.model_loading import (
    get_out_dir,
    load_model_from_wandb,
)


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    sampling: SamplingType,
    device: str,
) -> dict[tuple[str, str], Float[Tensor, "s_in C s_out C"]]:
    """Compute local attributions for a single prompt.

    For each valid layer pair (in_layer, out_layer), computes the gradient-based
    attribution of output component activations with respect to input component
    activations, preserving sequence position information.

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        ci_threshold: Threshold for considering a component alive at a position.
        sampling: Sampling type to use for causal importances.
        device: Device to run on.

    Returns:
        Dictionary mapping (in_layer, out_layer) -> attribution tensor.
        For non-attention pairs: shape [seq, C, seq, C] but only diagonal (s_in == s_out) is nonzero.
        For Q/K/V -> O pairs: shape [seq, C, seq, C] with causal structure (s_in <= s_out).
    """
    n_seq = tokens.shape[1]
    C = model.C

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")

    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )

    component_masks = ci.lower_leaky
    mask_infos = make_mask_infos(component_masks=component_masks, routing_masks="all")

    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            tokens, mask_infos=mask_infos, cache_type="component_acts"
        )

    cache = comp_output_with_cache.cache

    # Initialize output attributions
    local_attributions: dict[tuple[str, str], Float[Tensor, "s_in C s_out C"]] = {}

    for out_layer, in_layers in tqdm(sources_by_target.items(), desc="Target layers"):
        out_pre_detach: Float[Tensor, "1 s C"] = cache[f"{out_layer}_pre_detach"]
        ci_out: Float[Tensor, "1 s C"] = ci.lower_leaky[out_layer]

        for in_layer in in_layers:
            in_post_detach: Float[Tensor, "1 s C"] = cache[f"{in_layer}_post_detach"]
            ci_in: Float[Tensor, "1 s C"] = ci.lower_leaky[in_layer]

            attribution: Float[Tensor, "s_in C s_out C"] = torch.zeros(
                n_seq, C, n_seq, C, device=device
            )

            is_attention_pair = is_qkv_to_o_pair(in_layer, out_layer)

            # Determine which (s_out, c_out) pairs are alive
            alive_out_mask: Float[Tensor, "1 s C"] = ci_out >= ci_threshold
            alive_in_mask: Float[Tensor, "1 s C"] = ci_in >= ci_threshold

            for s_out in tqdm(range(n_seq), desc=f"{in_layer} -> {out_layer}", leave=False):
                # Get alive output components at this position
                alive_c_out: list[int] = torch.where(alive_out_mask[0, s_out])[0].tolist()
                if len(alive_c_out) == 0:
                    continue

                for c_out in alive_c_out:
                    grads = torch.autograd.grad(
                        outputs=out_pre_detach[0, s_out, c_out],
                        inputs=in_post_detach,
                        retain_graph=True,
                    )

                    assert len(grads) == 1
                    in_post_detach_grad: Float[Tensor, "1 s C"] = grads[0]
                    assert in_post_detach_grad is not None, f"Gradient is None for {in_layer}"

                    # Weight by input acts and square (we index into the singular batch dimension)
                    weighted: Float[Tensor, "s C"] = (in_post_detach_grad * in_post_detach)[0]

                    # Handle causal attention mask
                    s_in_range = range(s_out + 1) if is_attention_pair else range(s_out, s_out + 1)

                    with torch.no_grad():
                        for s_in in s_in_range:
                            # Only include alive input components
                            alive_c_in: list[int] = torch.where(alive_in_mask[0, s_in])[0].tolist()
                            for c_in in alive_c_in:
                                attribution[s_in, c_in, s_out, c_out] = weighted[s_in, c_in]

            local_attributions[(in_layer, out_layer)] = attribution

    return local_attributions


# %%
# Configuration
# wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L
wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
# wandb_path = "wandb:goodfire/spd/runs/jyo9duz5" # ss_gpt2_simple-1.25M (4L)
n_blocks = 2
batch_size = 1  # Only need 1 for getting sources_by_target
n_ctx = 64
ci_threshold = 1e-6
# prompt = "The quick brown fox"
prompt = "Eagerly, a girl named Kim went"

# Load model
loaded = load_model_from_wandb(wandb_path)
model, config, device = loaded.model, loaded.config, loaded.device

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"
sources_by_target = get_sources_by_target(model, device, config, n_blocks)
validate_attention_pair_structure(sources_by_target)

n_pairs = sum(len(ins) for ins in sources_by_target.values())
print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
for out_layer, in_layers in sources_by_target.items():
    print(f"  {out_layer} <- {in_layers}")

# %%
# Tokenize the prompt
print(f"\nPrompt: {prompt!r}")
tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
assert isinstance(tokens, Tensor), "Expected Tensor"
tokens = tokens.to(device)
print(f"Tokens shape: {tokens.shape}")
print(f"Tokens: {tokens[0].tolist()}")
token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
print(f"Token strings: {token_strings}")

# %%
# Compute local attributions
print("\nComputing local attributions...")
local_attributions = compute_local_attributions(
    model=model,
    tokens=tokens,
    sources_by_target=sources_by_target,
    ci_threshold=ci_threshold,
    sampling=config.sampling,
    device=device,
)

# Print summary statistics
print("\nAttribution summary:")
for pair, attr in local_attributions.items():
    nonzero = (attr > 0).sum().item()
    total = attr.numel()
    print(
        f"  {pair[0]} -> {pair[1]}: "
        f"nonzero={nonzero}/{total} ({100 * nonzero / total:.2f}%), "
        f"max={attr.max():.6f}"
    )

# %%
# Save attributions
out_dir = get_out_dir()

# Save PyTorch format
pt_path = out_dir / f"local_attributions_{loaded.wandb_id}.pt"
save_data = {
    "attributions": local_attributions,
    "tokens": tokens.cpu(),
    "token_strings": token_strings,
    "prompt": prompt,
}
torch.save(save_data, pt_path)
print(f"\nSaved PyTorch format to {pt_path}")

# Convert and save JSON format for web visualization
attributions_json = {}
for (in_layer, out_layer), attr_tensor in local_attributions.items():
    key = f"('{in_layer}', '{out_layer}')"
    attributions_json[key] = attr_tensor.cpu().tolist()

json_data = {
    "n_blocks": n_blocks,
    "attributions": attributions_json,
    "tokens": tokens[0].cpu().tolist(),
    "token_strings": token_strings,
    "prompt": prompt,
}

json_path = out_dir / f"local_attributions_{loaded.wandb_id}.json"
with open(json_path, "w") as f:
    json.dump(json_data, f, separators=(",", ":"), ensure_ascii=False)

gz_path = out_dir / f"local_attributions_{loaded.wandb_id}.json.gz"
with gzip.open(gz_path, "wt", encoding="utf-8") as f:
    json.dump(json_data, f, separators=(",", ":"), ensure_ascii=False)

print(f"Saved JSON format to {json_path}")
print(f"Saved compressed format to {gz_path}")
print(f"  - {len(attributions_json)} layer pairs")
print(f"  - Sequence length: {tokens.shape[1]}")

# %%
