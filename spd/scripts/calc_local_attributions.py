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


def normalize_attributions_by_target(
    attr_pairs: list[PairAttribution],
) -> list[PairAttribution]:
    """Normalize attributions so that for each target node, all incoming edges sum to 1.

    For each (target_layer, s_out, c_out), we collect all incoming attributions and
    divide by the sum of absolute values. Then we scale back up by the number of
    potential sources to avoid penalizing nodes with more upstream connections:
    - For cross-seq (kv->o) pairs: scale by (s_out + 1) * 2 (k and v)
    - For same-seq pairs: scale by number of PairAttributions with this target
    """
    # Count number of same-seq source pairs per target
    same_seq_source_counts: dict[str, int] = {}
    for pair in attr_pairs:
        if not pair.is_kv_to_o_pair:
            same_seq_source_counts[pair.target] = same_seq_source_counts.get(pair.target, 0) + 1

    # First pass: compute sum of |attr| for each target node across all source pairs
    # Key: (target_layer, s_out, c_out) -> sum of |attr|
    target_sums: dict[tuple[str, int, int], float] = {}

    for pair in attr_pairs:
        # attribution shape: [s_in, trimmed_c_in, s_out, trimmed_c_out]
        attr = pair.attribution
        n_s_out = attr.shape[2]
        n_c_out = attr.shape[3]

        for s_out in range(n_s_out):
            for c_out_local in range(n_c_out):
                c_out = pair.trimmed_c_out_idxs[c_out_local]
                key = (pair.target, s_out, c_out)
                # Sum absolute values of all inputs to this target
                abs_sum = attr[:, :, s_out, c_out_local].abs().sum().item()
                target_sums[key] = target_sums.get(key, 0.0) + abs_sum

    # Second pass: normalize each attribution by the target's total, then scale
    normalized_pairs: list[PairAttribution] = []
    for pair in attr_pairs:
        attr = pair.attribution.clone()
        n_s_out = attr.shape[2]
        n_c_out = attr.shape[3]

        for s_out in range(n_s_out):
            for c_out_local in range(n_c_out):
                c_out = pair.trimmed_c_out_idxs[c_out_local]
                key = (pair.target, s_out, c_out)
                total = target_sums.get(key, 1.0)
                if total > 0:
                    attr[:, :, s_out, c_out_local] /= total

                    # Scale by number of potential sources
                    if pair.is_kv_to_o_pair:
                        # Cross-seq (kv->o): can attend to s_out+1 positions, from k and v
                        scale = (s_out + 1) * 2
                    else:
                        # Same-seq: scale by number of source pairs for this target
                        scale = same_seq_source_counts.get(pair.target, 1)

                    attr[:, :, s_out, c_out_local] *= scale

        normalized_pairs.append(
            PairAttribution(
                source=pair.source,
                target=pair.target,
                attribution=attr,
                trimmed_c_in_idxs=pair.trimmed_c_in_idxs,
                trimmed_c_out_idxs=pair.trimmed_c_out_idxs,
                is_kv_to_o_pair=pair.is_kv_to_o_pair,
            )
        )

    return normalized_pairs


# %%
# Configuration
# wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L (Old)
# n_blocks = 1
# wandb_path = "wandb:goodfire/spd/runs/33n6xjjt"  # ss_gpt2_simple-1L (new)
# n_blocks = 1
# wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
# n_blocks = 2

wandb_path = "wandb:goodfire/spd/runs/jyo9duz5"  # ss_gpt2_simple-1.25M (4L)
n_blocks = 4

ci_threshold = 1e-6

print("=" * 60)
print("Loading model from wandb...")
loaded = load_model_from_wandb(wandb_path)
model, config, device = loaded.model, loaded.config, loaded.device
print(f"  Model loaded on {device}")
print(f"  C={model.C} components")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"
print(f"  Tokenizer: {config.tokenizer_name}")

print("\nBuilding source->target layer mapping...")
sources_by_target = get_sources_by_target(model, device, config, n_blocks)

n_pairs = sum(len(ins) for ins in sources_by_target.values())
print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
for out_layer, in_layers in sources_by_target.items():
    print(f"  {out_layer} <- {in_layers}")

# %%
# Sample a random sequence from the dataset
import random

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.utils.general_utils import runtime_cast

print("\n" + "=" * 60)
print("Loading dataset...")
task_config = runtime_cast(LMTaskConfig, config.task_config)
train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=32,
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
print(f"  Dataset: {task_config.dataset_name}")

# Skip a random number of batches and take one sequence
print("\nSampling random sequence...")
n_skip = random.randint(0, 100)

tokens: Float[Tensor, "1 seq"] = None
for i, batch in enumerate(train_loader):
    if i == n_skip:
        tokens = batch["input_ids"].to(device)
        break

assert tokens.shape[0] == 1, "Expected batch size 1"
print(f"  Sampled sequence {n_skip} from dataset")
print(f"  Tokens shape: {tokens.shape}")
token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
print(f"  Sequence: {''.join(token_strings)!r}")

attr_pairs = compute_local_attributions(
    model=model,
    tokens=tokens,
    sources_by_target=sources_by_target,
    ci_threshold=ci_threshold,
    sampling=config.sampling,
    device=device,
)

# Normalize attributions so each target node's incoming edges sum to 1
# NOTE: Disabled for now - conceptually flawed
# print("\nNormalizing attributions by target node...")
# attr_pairs = normalize_attributions_by_target(attr_pairs)

# Print summary statistics
print("\nAttribution summary:")
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

print("\n" + "=" * 60)
print("Computing activation contexts...")

# Build TrainRunContext for activation contexts (reuse train_loader from above)
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

n_batches_act_ctx = 10
print(f"  Processing {n_batches_act_ctx} batches...")

# Get activation contexts (just consume the generator to get final result)
activation_contexts = None
pbar = tqdm(total=1.0, desc="Activation contexts", unit="%", bar_format="{l_bar}{bar}| {n:.0%}")
for result in get_activations_data_streaming(
    run_context=run_context,
    importance_threshold=0.1,
    n_batches=n_batches_act_ctx,
    n_tokens_either_side=5,
    batch_size=8,
    topk_examples=10,
):
    if result[0] == "progress":
        pbar.n = result[1]
        pbar.refresh()
    elif result[0] == "complete":
        activation_contexts = result[1]
        pbar.n = 1.0
        pbar.refresh()
        break
pbar.close()

assert activation_contexts is not None, "Failed to get activation contexts"
print(f"  Got activation contexts for {len(activation_contexts.layers)} layers")

# %%
# Save attributions
import json

print("\n" + "=" * 60)
print("Saving results...")

out_dir = get_out_dir()

# Filtering params to keep file size manageable
# Note: FE does its own filtering (by mean CI, top K edges), so we keep more data here
MAX_SUBCOMPS_PER_LAYER = 100  # Keep top N subcomponents per layer for activation context
MAX_PR_TOKENS = 20  # Keep top N tokens for pr_tokens and predicted_tokens
ATTR_THRESHOLD = 1e-4  # Only keep attributions above this threshold (saves file size)


def filter_subcomp(subcomp_dict: dict) -> dict:
    """Filter a subcomponent dict to reduce size."""
    return {
        **subcomp_dict,
        "pr_tokens": subcomp_dict["pr_tokens"][:MAX_PR_TOKENS],
        "pr_recalls": subcomp_dict["pr_recalls"][:MAX_PR_TOKENS],
        "pr_precisions": subcomp_dict["pr_precisions"][:MAX_PR_TOKENS],
        "predicted_tokens": subcomp_dict["predicted_tokens"][:MAX_PR_TOKENS],
        "predicted_probs": subcomp_dict["predicted_probs"][:MAX_PR_TOKENS],
    }


def sparsify_attribution_cross_seq(attr_tensor: Tensor) -> list[list]:
    """Sparse format for cross-seq pairs: [[s_in, c_in, s_out, c_out, val], ...]"""
    sparse: list[list] = []
    attr_np = attr_tensor.cpu().numpy()
    for s_in in range(attr_np.shape[0]):
        for c_in in range(attr_np.shape[1]):
            for s_out in range(attr_np.shape[2]):
                for c_out in range(attr_np.shape[3]):
                    val = float(attr_np[s_in, c_in, s_out, c_out])
                    if abs(val) >= ATTR_THRESHOLD:
                        sparse.append([s_in, c_in, s_out, c_out, round(val, 6)])
    return sparse


def sparsify_attribution_same_seq(attr_tensor: Tensor) -> list[list]:
    """Sparse format for same-seq pairs: [[s, c_in, c_out, val], ...] where s_in == s_out == s."""
    sparse: list[list] = []
    attr_np = attr_tensor.cpu().numpy()
    n_seq = attr_np.shape[0]
    for s in range(n_seq):
        for c_in in range(attr_np.shape[1]):
            for c_out in range(attr_np.shape[3]):
                val = float(attr_np[s, c_in, s, c_out])
                if abs(val) >= ATTR_THRESHOLD:
                    sparse.append([s, c_in, c_out, round(val, 6)])
    return sparse


def serialize_pair(pair: PairAttribution) -> dict:
    """Serialize a pair with appropriate sparse format based on whether it's cross-seq."""
    if pair.is_kv_to_o_pair:
        return {
            "source": pair.source,
            "target": pair.target,
            "is_cross_seq": True,
            "attribution": sparsify_attribution_cross_seq(pair.attribution),
            "trimmed_c_in_idxs": pair.trimmed_c_in_idxs,
            "trimmed_c_out_idxs": pair.trimmed_c_out_idxs,
        }
    else:
        return {
            "source": pair.source,
            "target": pair.target,
            "is_cross_seq": False,
            "attribution": sparsify_attribution_same_seq(pair.attribution),
            "trimmed_c_in_idxs": pair.trimmed_c_in_idxs,
            "trimmed_c_out_idxs": pair.trimmed_c_out_idxs,
        }


# Serialize to JSON-friendly format for the app
local_attr_data = {
    "tokens": token_strings,
    "pairs": [serialize_pair(pair) for pair in attr_pairs],
    # Add activation contexts - filter and limit subcomponents
    "activation_contexts": {
        layer_name: [
            filter_subcomp(subcomp.model_dump())
            for subcomp in sorted(subcomps, key=lambda x: x.mean_ci, reverse=True)[
                :MAX_SUBCOMPS_PER_LAYER
            ]
        ]
        for layer_name, subcomps in activation_contexts.layers.items()
    },
}

json_path = out_dir / "local_attributions.json"
with open(json_path, "w") as f:
    json.dump(local_attr_data, f)
print(f"  Saved to {json_path}")
print("\n" + "=" * 60)
print("Done!")

# %%
