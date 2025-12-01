"""Compute local attributions for a single prompt."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.scripts.calc_global_attributions import get_sources_by_target, is_kv_to_o_pair
from spd.scripts.model_loading import get_out_dir, load_model_from_wandb
from spd.scripts.plot_local_attributions import PairAttribution, plot_local_graph


@dataclass
class LayerAliveInfo:
    """Info about alive components for a layer."""

    alive_mask: Bool[Tensor, "1 s dim"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]  # Components alive at any position
    c_to_trimmed: dict[int, int]  # original idx -> trimmed idx


def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    output_probs: Float[Tensor, "1 s vocab"] | None,
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, wte, and output layers."""
    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        alive_mask = torch.ones(1, n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
    elif layer_name == "output":
        assert output_probs is not None
        alive_mask = output_probs >= output_prob_threshold
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()
    else:
        ci = ci_lower_leaky[layer_name]
        alive_mask = ci >= ci_threshold
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()

    c_to_trimmed = {c: i for i, c in enumerate(alive_c_idxs)}
    return LayerAliveInfo(alive_mask, alive_c_idxs, c_to_trimmed)


def load_ci_from_json(
    ci_vals_path: str | Path,
    expected_prompt: str,
    device: str,
) -> dict[str, Float[Tensor, "1 seq C"]]:
    """Load precomputed CI values from a JSON file.

    Args:
        ci_vals_path: Path to JSON file from run_optim_cis.py
        expected_prompt: The prompt we're analyzing (must match the JSON)
        device: Device to load tensors to

    Returns:
        Dict mapping layer_name -> CI tensor of shape [1, seq, C]

    Raises:
        ValueError: If the prompt in the JSON doesn't match expected_prompt
    """
    with open(ci_vals_path) as f:
        data = json.load(f)

    json_prompt = data["prompt"]
    if json_prompt != expected_prompt:
        raise ValueError(
            f"Prompt mismatch: JSON has {json_prompt!r}, but expected {expected_prompt!r}"
        )

    ci_lower_leaky: dict[str, Tensor] = {}
    for layer_name, ci_list in data["optimized_ci"].items():
        # ci_list is [seq][C], convert to tensor [1, seq, C]
        ci_tensor = torch.tensor(ci_list, device=device).unsqueeze(0)
        ci_lower_leaky[layer_name] = ci_tensor

    return ci_lower_leaky


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]] | None = None,
) -> tuple[list[PairAttribution], Float[Tensor, "1 seq vocab"]]:
    """Compute local attributions for a single prompt.

    For each valid layer pair (in_layer, out_layer), computes the gradient-based
    attribution of output component activations with respect to input component
    activations, preserving sequence position information.

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        ci_threshold: Threshold for considering a component alive at a position.
        output_prob_threshold: Threshold for considering an output logit alive (on softmax probs).
        sampling: Sampling type to use for causal importances.
        device: Device to run on.
        ci_lower_leaky: Optional precomputed/optimized CI values. If None, will use model CI.
            When provided, we still compute original model CI to track "ghost" nodes.

    Returns:
        List of PairAttribution objects.
    """
    n_seq = tokens.shape[1]

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")

    # Always compute original CI from model (needed for ghost nodes when using optimized CI)
    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )
    ci_original = ci.lower_leaky

    # Use provided CI values if given, otherwise use original
    if ci_lower_leaky is None:
        ci_lower_leaky = ci_original

    # Log the l0 (lower_leaky values > ci_threshold) for each layer
    print("L0 values for final seq position:")
    for layer, ci_vals in ci_lower_leaky.items():
        # We only care about the final position
        l0_vals = (ci_vals[0, -1] > ci_threshold).sum().item()
        print(f"  Layer {layer} has {l0_vals} components alive at {ci_threshold}")

    wte_cache: dict[str, Tensor] = {}

    def wte_hook(_module: nn.Module, _args: Any, _kwargs: Any, output: Tensor) -> Any:
        output.requires_grad_(True)
        # We call it "post_detach" for consistency, we don't bother detaching here as there are
        # no modules before it that we care about
        wte_cache["wte_post_detach"] = output
        return output

    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    mask_infos = make_mask_infos(
        # component_masks={k: torch.ones_like(v) for k, v in ci_lower_leaky.items()},
        component_masks=ci_lower_leaky,
        routing_masks="all",
    )
    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            tokens, mask_infos=mask_infos, cache_type="component_acts"
        )
    wte_handle.remove()

    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
    cache["output_pre_detach"] = comp_output_with_cache.output

    # Compute output probabilities for thresholding
    output_probs = torch.softmax(comp_output_with_cache.output, dim=-1)

    # Compute alive info for all layers upfront
    all_layers: set[str] = set(sources_by_target.keys())
    for sources in sources_by_target.values():
        all_layers.update(sources)

    alive_info: dict[str, LayerAliveInfo] = {}
    original_alive_info: dict[str, LayerAliveInfo] = {}
    for layer in all_layers:
        alive_info[layer] = compute_layer_alive_info(
            layer, ci_lower_leaky, output_probs, ci_threshold, output_prob_threshold, n_seq, device
        )
        # Compute original alive info (from model CI, not optimized CI)
        original_alive_info[layer] = compute_layer_alive_info(
            layer, ci_original, output_probs, ci_threshold, output_prob_threshold, n_seq, device
        )

    local_attributions: list[PairAttribution] = []

    for target, sources in tqdm(sources_by_target.items(), desc="Target layers"):
        target_info = alive_info[target]
        out_pre_detach: Float[Tensor, "1 s dim"] = cache[f"{target}_pre_detach"]

        source_infos = [alive_info[source] for source in sources]
        in_post_detaches: list[Float[Tensor, "1 s dim"]] = [
            cache[f"{source}_post_detach"] for source in sources
        ]

        # Initialize attribution tensors at final trimmed size
        attributions: list[Float[Tensor, "s_in n_c_in s_out n_c_out"]] = [
            torch.zeros(
                n_seq,
                len(source_info.alive_c_idxs),
                n_seq,
                len(target_info.alive_c_idxs),
                device=device,
            )
            for source_info in source_infos
        ]

        # NOTE: o->q will be treated as an attention pair even though there are no attrs
        # across sequence positions. This is just so we don't have to special case it.
        is_attention_output = any(is_kv_to_o_pair(source, target) for source in sources)

        for s_out in tqdm(range(n_seq), desc=f"{target} <- {sources}", leave=False):
            # Get alive output components at this position
            s_out_alive_c: list[int] = [
                c for c in target_info.alive_c_idxs if target_info.alive_mask[0, s_out, c]
            ]
            if not s_out_alive_c:
                continue

            for c_out in s_out_alive_c:
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[0, s_out, c_out],
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                # Handle causal attention mask
                s_in_range = range(s_out + 1) if is_attention_output else range(s_out, s_out + 1)
                trimmed_c_out = target_info.c_to_trimmed[c_out]

                with torch.no_grad():
                    for source, source_info, grad, in_post_detach, attr in zip(
                        sources,
                        source_infos,
                        in_post_detach_grads,
                        in_post_detaches,
                        attributions,
                        strict=True,
                    ):
                        weighted: Float[Tensor, "s dim"] = (grad * in_post_detach)[0]
                        if source == "wte":
                            # Sum over embedding_dim to get single pseudo-component
                            weighted = weighted.sum(dim=1, keepdim=True)

                        for s_in in s_in_range:
                            alive_c_in = [
                                c
                                for c in source_info.alive_c_idxs
                                if source_info.alive_mask[0, s_in, c]
                            ]
                            for c_in in alive_c_in:
                                trimmed_c_in = source_info.c_to_trimmed[c_in]
                                attr[s_in, trimmed_c_in, s_out, trimmed_c_out] = weighted[
                                    s_in, c_in
                                ]

        for source, source_info, attr in zip(sources, source_infos, attributions, strict=True):
            original_source_info = original_alive_info[source]
            original_target_info = original_alive_info[target]
            local_attributions.append(
                PairAttribution(
                    source=source,
                    target=target,
                    attribution=attr,
                    trimmed_c_in_idxs=source_info.alive_c_idxs,
                    trimmed_c_out_idxs=target_info.alive_c_idxs,
                    is_kv_to_o_pair=is_kv_to_o_pair(source, target),
                    # Pass per-position alive masks (squeeze out batch dim)
                    original_alive_mask_in=original_source_info.alive_mask[0],  # [seq, C]
                    original_alive_mask_out=original_target_info.alive_mask[0],  # [seq, C]
                )
            )

    return local_attributions, output_probs


def main(
    wandb_path: str,
    n_blocks: int,
    ci_threshold: float,
    output_prob_threshold: float,
    prompt: str,
    ci_vals_path: str | None,
) -> None:
    loaded = load_model_from_wandb(wandb_path)
    model, config, device = loaded.model, loaded.config, loaded.device

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"
    sources_by_target = get_sources_by_target(model, device, config, n_blocks)

    n_pairs = sum(len(ins) for ins in sources_by_target.values())
    print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
    for out_layer, in_layers in sources_by_target.items():
        print(f"  {out_layer} <- {in_layers}")

    # Tokenize the prompt
    print(f"\nPrompt: {prompt!r}")
    tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    assert isinstance(tokens, Tensor), "Expected Tensor"
    tokens = tokens.to(device)
    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens: {tokens[0].tolist()}")
    token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
    print(f"Token strings: {token_strings}")

    # Load precomputed CI values if path is provided
    ci_lower_leaky: dict[str, Tensor] | None = None
    if ci_vals_path is not None:
        print(f"\nLoading precomputed CI values from {ci_vals_path}")
        ci_lower_leaky = load_ci_from_json(ci_vals_path, prompt, device)
        print(f"Loaded CI values for layers: {list(ci_lower_leaky.keys())}")

    # Compute local attributions
    print("\nComputing local attributions...")
    attr_pairs, output_probs = compute_local_attributions(
        model=model,
        tokens=tokens,
        sources_by_target=sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=config.sampling,
        device=device,
        ci_lower_leaky=ci_lower_leaky,
    )

    # Print summary statistics
    print("\nAttribution summary:")
    for attr_pair in attr_pairs:
        total = attr_pair.attribution.numel()
        if total == 0:
            print(
                f"Ignoring {attr_pair.source} -> {attr_pair.target}: "
                f"shape={list(attr_pair.attribution.shape)}, zero"
            )
            continue
        nonzero = (attr_pair.attribution > 0).sum().item()
        print(
            f"  {attr_pair.source} -> {attr_pair.target}: "
            f"shape={list(attr_pair.attribution.shape)}, "
            f"nonzero={nonzero}/{total} ({100 * nonzero / (total + 1e-12):.2f}%), "
            f"max={attr_pair.attribution.max():.6f}"
        )

    # Save attributions
    out_dir = get_out_dir()
    pt_path = out_dir / f"local_attributions_{loaded.wandb_id}.pt"
    output_path = out_dir / f"local_attribution_graph_{loaded.wandb_id}.png"
    if ci_vals_path is not None:
        pt_path = pt_path.with_stem(pt_path.stem + "_with_ci_optim")
        output_path = output_path.with_stem(output_path.stem + "_with_ci_optim")

    output_token_labels: dict[int, str] = {}
    output_probs_by_pos: dict[tuple[int, int], float] = {}
    for attr_pair in attr_pairs:
        if attr_pair.target == "output":
            for c_idx in attr_pair.trimmed_c_out_idxs:
                if c_idx not in output_token_labels:
                    output_token_labels[c_idx] = tokenizer.decode([c_idx])
                # Store probability for each position
                for s in range(tokens.shape[1]):
                    prob = output_probs[0, s, c_idx].item()
                    if prob >= output_prob_threshold:
                        output_probs_by_pos[(s, c_idx)] = prob
            break

    save_data = {
        "attr_pairs": attr_pairs,
        "token_strings": token_strings,
        "prompt": prompt,
        "ci_threshold": ci_threshold,
        "output_prob_threshold": output_prob_threshold,
        "output_token_labels": output_token_labels,
        "output_probs_by_pos": output_probs_by_pos,
        "wandb_id": loaded.wandb_id,
    }
    torch.save(save_data, pt_path)
    print(f"\nSaved local attributions to {pt_path}")

    fig = plot_local_graph(
        attr_pairs=attr_pairs,
        token_strings=token_strings,
        output_token_labels=output_token_labels,
        output_prob_threshold=output_prob_threshold,
        output_probs_by_pos=output_probs_by_pos,
        min_edge_weight=0.0001,
        node_scale=30.0,
        edge_alpha_scale=0.7,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    # Configuration
    # wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L (Old)
    wandb_path = "wandb:goodfire/spd/runs/33n6xjjt"  # ss_gpt2_simple-1L (new)
    # wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
    # wandb_path = "wandb:goodfire/spd/runs/jyo9duz5"  # ss_gpt2_simple-1.25M (4L)
    # n_blocks = 4
    n_blocks = 1
    ci_threshold = 1e-6
    output_prob_threshold = 1e-1
    # prompt = "The quick brown fox"
    # prompt = "Eagerly, a girl named Kim went"
    prompt = "They walked hand in"
    # Path to precomputed CI values from run_optim_cis.py (None to compute from model)
    # ci_vals_path: str | None = None
    ci_vals_path = "spd/scripts/optim_cis/out/optimized_ci_33n6xjjt.json"
    main(
        wandb_path=wandb_path,
        n_blocks=n_blocks,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        prompt=prompt,
        ci_vals_path=ci_vals_path,
    )
