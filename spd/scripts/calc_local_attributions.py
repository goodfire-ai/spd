# %%
"""Compute local attributions for a single prompt."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.scripts.calc_global_attributions import get_sources_by_target, is_kv_to_o_pair
from spd.scripts.model_loading import get_out_dir, load_model_from_wandb


@dataclass
class PairAttribution:
    source: str
    target: str
    attribution: Float[Tensor, "s_in trimmed_c_in s_out trimmed_c_out"]
    trimmed_c_in_idxs: list[int]
    trimmed_c_out_idxs: list[int]
    is_kv_to_o_pair: bool


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    sampling: SamplingType,
    device: str,
) -> list[PairAttribution]:
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
        List of PairAttribution objects.
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

    local_attributions: list[PairAttribution] = []

    for out_layer, in_layers in tqdm(sources_by_target.items(), desc="Target layers"):
        out_pre_detach: Float[Tensor, "1 s C"] = cache[f"{out_layer}_pre_detach"]
        ci_out: Float[Tensor, "1 s C"] = ci.lower_leaky[out_layer]

        in_post_detaches: list[Float[Tensor, "1 s C"]] = [
            cache[f"{in_layer}_post_detach"] for in_layer in in_layers
        ]
        ci_ins: list[Float[Tensor, "1 s C"]] = [ci.lower_leaky[in_layer] for in_layer in in_layers]

        attributions: list[Float[Tensor, "s_in C s_out C"]] = [
            torch.zeros(n_seq, C, n_seq, C, device=device) for _ in in_layers
        ]

        # NOTE: o->q will be treated as an attention pair even though there are no attrs
        # across sequence positions. This is just so we don't have to special case it.
        is_attention_output = any(is_kv_to_o_pair(in_layer, out_layer) for in_layer in in_layers)

        # Determine which (s_out, c_out) pairs are alive
        alive_out_mask: Float[Tensor, "1 s C"] = ci_out >= ci_threshold
        alive_out_c_idxs: list[int] = torch.where(alive_out_mask[0].any(dim=0))[0].tolist()

        alive_in_masks: list[Float[Tensor, "1 s C"]] = [ci_in >= ci_threshold for ci_in in ci_ins]
        alive_in_c_idxs: list[list[int]] = [
            torch.where(alive_in_mask[0].any(dim=0))[0].tolist() for alive_in_mask in alive_in_masks
        ]

        for s_out in tqdm(range(n_seq), desc=f"{out_layer} -> {in_layers}", leave=False):
            # Get alive output components at this position
            s_out_alive_c_idxs: list[int] = torch.where(alive_out_mask[0, s_out])[0].tolist()
            if len(s_out_alive_c_idxs) == 0:
                continue

            for c_out in s_out_alive_c_idxs:
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[0, s_out, c_out],
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                # Handle causal attention mask
                s_in_range = range(s_out + 1) if is_attention_output else range(s_out, s_out + 1)

                with torch.no_grad():
                    for in_post_detach_grad, in_post_detach, alive_in_mask, attribution in zip(
                        in_post_detach_grads,
                        in_post_detaches,
                        alive_in_masks,
                        attributions,
                        strict=True,
                    ):
                        # Weight by input acts and square (we index into the singular batch dimension)
                        weighted: Float[Tensor, "s C"] = (in_post_detach_grad * in_post_detach)[0]
                        for s_in in s_in_range:
                            alive_c_in: list[int] = torch.where(alive_in_mask[0, s_in])[0].tolist()
                            for c_in in alive_c_in:
                                attribution[s_in, c_in, s_out, c_out] = weighted[s_in, c_in]

        for in_layer, attribution, layer_alive_in_c_idxs in zip(
            in_layers, attributions, alive_in_c_idxs, strict=True
        ):
            trimmed_attribution = attribution[:, layer_alive_in_c_idxs][:, :, :, alive_out_c_idxs]
            local_attributions.append(
                PairAttribution(
                    source=in_layer,
                    target=out_layer,
                    attribution=trimmed_attribution,
                    trimmed_c_in_idxs=layer_alive_in_c_idxs,
                    trimmed_c_out_idxs=alive_out_c_idxs,
                    is_kv_to_o_pair=is_kv_to_o_pair(in_layer, out_layer),
                )
            )

    return local_attributions


# %%
# Configuration
# wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L (Old)
wandb_path = "wandb:goodfire/spd/runs/33n6xjjt"  # ss_gpt2_simple-1L (new)
# wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
# wandb_path = "wandb:goodfire/spd/runs/jyo9duz5" # ss_gpt2_simple-1.25M (4L)
n_blocks = 1
ci_threshold = 1e-6
# prompt = "The quick brown fox"
prompt = "Eagerly, a girl named Kim went"

loaded = load_model_from_wandb(wandb_path)
model, config, device = loaded.model, loaded.config, loaded.device

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"
sources_by_target = get_sources_by_target(model, device, config, n_blocks)

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
attr_pairs = compute_local_attributions(
    model=model,
    tokens=tokens,
    sources_by_target=sources_by_target,
    ci_threshold=ci_threshold,
    sampling=config.sampling,
    device=device,
)

# Print summary statistics
print("\nAttribution summary:")
# for pair, attr in local_attributions:
for attr_pair in attr_pairs:
    nonzero = (attr_pair.attribution > 0).sum().item()
    total = attr_pair.attribution.numel()
    print(
        f"  {attr_pair.source} -> {attr_pair.target}: "
        f"nonzero={nonzero}/{total} ({100 * nonzero / total:.2f}%), "
        f"max={attr_pair.attribution.max():.6f}"
    )

# %%
# Get activation contexts

from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.services.run_context_service import TrainRunContext, _build_token_lookup
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.utils.general_utils import runtime_cast

print("\nLoading activation contexts...")

# Build TrainRunContext for activation contexts
task_config = runtime_cast(LMTaskConfig, config.task_config)
train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=task_config.max_seq_len,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=task_config.shuffle_each_epoch,
    seed=None,
)
train_loader, _ = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
)
assert config.tokenizer_name is not None
token_string_lookup = _build_token_lookup(tokenizer, config.tokenizer_name)

run_context = TrainRunContext(
    wandb_id=loaded.wandb_id,
    wandb_path=wandb_path,
    config=config,
    cm=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    token_strings=token_string_lookup,
)

# Get activation contexts (just consume the generator to get final result)
activation_contexts = None
for result in get_activations_data_streaming(
    run_context=run_context,
    importance_threshold=0.1,
    n_batches=50,
    n_tokens_either_side=5,
    batch_size=32,
    topk_examples=10,
):
    if result[0] == "complete":
        activation_contexts = result[1]
        break

assert activation_contexts is not None, "Failed to get activation contexts"
print(f"Got activation contexts for {len(activation_contexts.layers)} layers")

# %%
# Save attributions
import json

out_dir = get_out_dir()

# Serialize to JSON-friendly format for the app
local_attr_data = {
    "tokens": token_strings,
    "pairs": [
        {
            "source": pair.source,
            "target": pair.target,
            "attribution": pair.attribution.tolist(),
            "trimmed_c_in_idxs": pair.trimmed_c_in_idxs,
            "trimmed_c_out_idxs": pair.trimmed_c_out_idxs,
            "is_kv_to_o_pair": pair.is_kv_to_o_pair,
        }
        for pair in attr_pairs
    ],
    # Add activation contexts - convert pydantic models to dicts
    "activation_contexts": {
        layer_name: [subcomp.model_dump() for subcomp in subcomps]
        for layer_name, subcomps in activation_contexts.layers.items()
    },
}

json_path = out_dir / "local_attributions.json"
with open(json_path, "w") as f:
    json.dump(local_attr_data, f)
print(f"\nSaved local attributions to {json_path}")

# %%
