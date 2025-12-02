"""Compute local attributions for a single prompt."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
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
    """Info about alive components for a layer (single batch item)."""

    alive_mask: Bool[Tensor, "s dim"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]  # Components alive at any position
    c_to_trimmed: dict[int, int]  # original idx -> trimmed idx


@dataclass
class TokensAndCI:
    """Tokenized prompts and their CI values."""

    tokens: Int[Tensor, "N seq"]
    ci_lower_leaky: dict[str, Float[Tensor, "N seq C"]]
    set_names: list[str]
    prompts: list[str]
    all_token_strings: list[list[str]]


def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    output_probs: Float[Tensor, "N s vocab"] | None,
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    n_batch: int,
    device: str,
) -> list[LayerAliveInfo]:
    """Compute alive info for a layer across all batch items.

    Returns list of LayerAliveInfo, one per batch item.
    """
    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        alive_mask = torch.ones(n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
        c_to_trimmed = {0: 0}
        # Same info for all batch items
        return [LayerAliveInfo(alive_mask, alive_c_idxs, c_to_trimmed) for _ in range(n_batch)]

    elif layer_name == "output":
        assert output_probs is not None
        # output_probs: [N, seq, vocab]
        full_alive_mask = output_probs >= output_prob_threshold  # [N, seq, vocab]
        results = []
        for b in range(n_batch):
            batch_mask = full_alive_mask[b]  # [seq, vocab]
            alive_c_idxs = torch.where(batch_mask.any(dim=0))[0].tolist()
            c_to_trimmed = {c: i for i, c in enumerate(alive_c_idxs)}
            results.append(LayerAliveInfo(batch_mask, alive_c_idxs, c_to_trimmed))
        return results

    else:
        ci = ci_lower_leaky[layer_name]  # [N, seq, C]
        full_alive_mask = ci >= ci_threshold  # [N, seq, C]
        results = []
        for b in range(n_batch):
            batch_mask = full_alive_mask[b]  # [seq, C]
            alive_c_idxs = torch.where(batch_mask.any(dim=0))[0].tolist()
            c_to_trimmed = {c: i for i, c in enumerate(alive_c_idxs)}
            results.append(LayerAliveInfo(batch_mask, alive_c_idxs, c_to_trimmed))
        return results


def load_ci_from_json(
    ci_vals_path: str | Path,
    device: str,
) -> tuple[dict[str, Float[Tensor, "N seq C"]], list[str], list[str]]:
    """Load precomputed CI values from a JSON file.

    Expected format:
        {
            "ci_sets": {
                "set_name_1": {
                    "prompt": "...",  # Each set has its own prompt
                    "ci_vals": {"layer1": [[...]], ...},
                    ...
                },
                "set_name_2": {...},
            }
        }

    Args:
        ci_vals_path: Path to JSON file with ci_sets structure
        device: Device to load tensors to

    Returns:
        Tuple of:
            - Dict mapping layer_name -> CI tensor of shape [N, seq, C] where N = number of sets
            - List of set names in batch order
            - List of prompts in batch order (one per set)
    """
    with open(ci_vals_path) as f:
        data = json.load(f)

    ci_sets: dict[str, dict[str, Any]] = data["ci_sets"]
    set_names = list(ci_sets.keys())
    assert len(set_names) > 0, "No CI sets found in JSON"

    # Extract prompt from each set
    prompts: list[str] = [ci_sets[set_name]["prompt"] for set_name in set_names]

    # Get layer names from first set
    first_set = ci_sets[set_names[0]]
    layer_names = list(first_set["ci_vals"].keys())

    # Stack tensors along batch dimension
    ci_lower_leaky: dict[str, Tensor] = {}
    for layer_name in layer_names:
        # Collect [seq, C] tensors from each set
        layer_tensors = [
            torch.tensor(ci_sets[set_name]["ci_vals"][layer_name], device=device)
            for set_name in set_names
        ]
        # Stack to [N, seq, C]
        ci_lower_leaky[layer_name] = torch.stack(layer_tensors, dim=0)

    return ci_lower_leaky, set_names, prompts


def get_tokens_and_ci(
    model: ComponentModel,
    tokenizer: PreTrainedTokenizerFast,
    sampling: SamplingType,
    device: str,
    ci_vals_path: str | Path | None,
    prompts: list[str] | None,
) -> TokensAndCI:
    """Get tokenized prompts and CI values, either from file or computed from model.

    Args:
        model: The ComponentModel to use for computing CI if needed.
        tokenizer: Tokenizer for the prompts.
        sampling: Sampling type for CI computation.
        device: Device to place tensors on.
        ci_vals_path: Path to JSON with precomputed CI values and prompts.
            If provided, prompts are read from the JSON file.
        prompts: List of prompts to use when ci_vals_path is None.

    Returns:
        TokensAndCI containing tokens, CI values, set names, prompts, and token strings.
    """
    if ci_vals_path is not None:
        print(f"\nLoading precomputed CI values from {ci_vals_path}")
        ci_lower_leaky, set_names, prompts = load_ci_from_json(ci_vals_path, device)
        print(f"Loaded CI values for layers: {list(ci_lower_leaky.keys())}")
        print(f"CI sets: {set_names}")
    else:
        assert prompts is not None, "prompts is required when ci_vals_path is None"
        set_names = [f"prompt_{i}" for i in range(len(prompts))]
        ci_lower_leaky = None

    # Tokenize each prompt
    tokens_list: list[Tensor] = []
    all_token_strings: list[list[str]] = []
    for i, p in enumerate(prompts):
        if ci_vals_path is not None:
            print(f"\nPrompt for {set_names[i]}: {p!r}")
        toks = tokenizer.encode(p, return_tensors="pt", add_special_tokens=False)
        assert isinstance(toks, Tensor), "Expected Tensor"
        tokens_list.append(toks)
        first_row = toks[0]
        assert isinstance(first_row, Tensor), "Expected 2D tensor"
        all_token_strings.append([tokenizer.decode([t]) for t in first_row.tolist()])
        if ci_vals_path is not None:
            print(f"  Token strings: {all_token_strings[-1]}")

    # Validate all prompts have the same token length
    token_lengths = [t.shape[1] for t in tokens_list]
    assert all(length == token_lengths[0] for length in token_lengths), (
        f"All prompts must tokenize to the same length, got {token_lengths}"
    )

    tokens = torch.cat(tokens_list, dim=0).to(device)  # [N, seq]
    print(f"\nTokens shape: {tokens.shape}")

    # Compute CI values from model if not provided
    if ci_lower_leaky is None:
        print("\nComputing CI values from model...")
        with torch.no_grad():
            output_with_cache = model(tokens, cache_type="input")
        ci_lower_leaky = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        ).lower_leaky

    return TokensAndCI(
        tokens=tokens,
        ci_lower_leaky=ci_lower_leaky,
        set_names=set_names,
        prompts=prompts,
        all_token_strings=all_token_strings,
    )


def compute_local_attributions(
    model: ComponentModel,
    tokens: Int[Tensor, "N seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    ci_lower_leaky: dict[str, Float[Tensor, "N seq C"]],
    set_names: list[str],
) -> tuple[dict[str, list[PairAttribution]], Float[Tensor, "N seq vocab"]]:
    """Compute local attributions for multiple prompts across multiple CI sets.

    For each valid layer pair (in_layer, out_layer), computes the gradient-based
    attribution of output component activations with respect to input component
    activations, preserving sequence position information.

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompts of shape [N, seq_len] - one per CI set.
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        ci_threshold: Threshold for considering a component alive at a position.
        output_prob_threshold: Threshold for considering an output logit alive (on softmax probs).
        sampling: Sampling type to use for causal importances.
        device: Device to run on.
        ci_lower_leaky: Precomputed/optimized CI values with shape [N, seq, C].
        set_names: Ordered list of set names corresponding to batch dimension.

    Returns:
        Tuple of:
            - Dict mapping set_name -> list of PairAttribution objects
            - Output probabilities of shape [N, seq, vocab]
    """
    n_batch, n_seq = tokens.shape
    # Validate batch size matches CI tensors
    first_ci = next(iter(ci_lower_leaky.values()))
    assert first_ci.shape[0] == n_batch, f"CI batch size {first_ci.shape[0]} != tokens {n_batch}"
    assert len(set_names) == n_batch, f"set_names length {len(set_names)} != n_batch {n_batch}"

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")

    # Always compute original CI from model (needed for ghost nodes when using optimized CI)
    # Note: original CI is computed from single-batch input, then repeated for comparison
    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )
    ci_original = ci.lower_leaky  # [N, seq, C]

    # Log the l0 (lower_leaky values > ci_threshold) for each layer
    print("L0 values for final seq position (first CI set):")
    for layer, ci_vals in ci_lower_leaky.items():
        # We only care about the final position, show first batch item
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

    mask_infos = make_mask_infos(component_masks=ci_lower_leaky, routing_masks="all")
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

    # alive_info[layer] -> list of LayerAliveInfo, one per batch item
    alive_info: dict[str, list[LayerAliveInfo]] = {}
    original_alive_info: dict[str, list[LayerAliveInfo]] = {}
    # alive_c_union[layer] -> union of alive_c_idxs across all batches
    alive_c_union: dict[str, list[int]] = {}

    for layer in all_layers:
        alive_info[layer] = compute_layer_alive_info(
            layer,
            ci_lower_leaky,
            output_probs,
            ci_threshold,
            output_prob_threshold,
            n_seq,
            n_batch,
            device,
        )
        # Compute original alive info (from model CI, not optimized CI)
        original_alive_info[layer] = compute_layer_alive_info(
            layer,
            ci_original,
            output_probs,
            ci_threshold,
            output_prob_threshold,
            n_seq,
            n_batch,
            device,
        )
        # Compute union of alive components across batches
        all_alive: set[int] = set()
        for batch_info in alive_info[layer]:
            all_alive.update(batch_info.alive_c_idxs)
        alive_c_union[layer] = sorted(all_alive)

    # Initialize output dictionary
    local_attributions_by_set: dict[str, list[PairAttribution]] = {name: [] for name in set_names}

    for target, sources in tqdm(sources_by_target.items(), desc="Target layers"):
        target_infos = alive_info[target]  # list of LayerAliveInfo per batch
        out_pre_detach: Float[Tensor, "N s dim"] = cache[f"{target}_pre_detach"]

        all_source_infos = [alive_info[source] for source in sources]  # list of lists
        in_post_detaches: list[Float[Tensor, "N s dim"]] = [
            cache[f"{source}_post_detach"] for source in sources
        ]

        # Initialize per-batch attribution tensors
        # attributions[source_idx][batch_idx] = tensor
        attributions: list[list[Float[Tensor, "s_in n_c_in s_out n_c_out"]]] = [
            [
                torch.zeros(
                    n_seq,
                    len(source_infos[b].alive_c_idxs),
                    n_seq,
                    len(target_infos[b].alive_c_idxs),
                    device=device,
                )
                for b in range(n_batch)
            ]
            for source_infos in all_source_infos
        ]

        # NOTE: o->q will be treated as an attention pair even though there are no attrs
        # across sequence positions. This is just so we don't have to special case it.
        is_attention_output = any(is_kv_to_o_pair(source, target) for source in sources)

        for s_out in tqdm(range(n_seq), desc=f"{target} <- {sources}", leave=False):
            # Union of alive c_out across all batches at this position
            s_out_alive_c_union: list[int] = [
                c
                for c in alive_c_union[target]
                if any(info.alive_mask[s_out, c] for info in target_infos)
            ]
            if not s_out_alive_c_union:
                continue

            for c_out in s_out_alive_c_union:
                # Sum over batch dimension for efficient batched gradient
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[:, s_out, c_out].sum(),
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                # Handle causal attention mask
                s_in_range = range(s_out + 1) if is_attention_output else range(s_out, s_out + 1)

                with torch.no_grad():
                    for source_idx, (source, source_infos, grad, in_post_detach) in enumerate(
                        zip(
                            sources,
                            all_source_infos,
                            in_post_detach_grads,
                            in_post_detaches,
                            strict=True,
                        )
                    ):
                        # grad and in_post_detach: [N, seq, dim]
                        weighted: Float[Tensor, "N s dim"] = grad * in_post_detach
                        if source == "wte":
                            # Sum over embedding_dim to get single pseudo-component
                            weighted = weighted.sum(dim=2, keepdim=True)

                        # Store attributions per-batch
                        for b in range(n_batch):
                            # Only store if c_out is alive in this batch at this position
                            if not target_infos[b].alive_mask[s_out, c_out]:
                                continue
                            if c_out not in target_infos[b].c_to_trimmed:
                                continue
                            trimmed_c_out = target_infos[b].c_to_trimmed[c_out]

                            for s_in in s_in_range:
                                alive_c_in = [
                                    c
                                    for c in source_infos[b].alive_c_idxs
                                    if source_infos[b].alive_mask[s_in, c]
                                ]
                                for c_in in alive_c_in:
                                    trimmed_c_in = source_infos[b].c_to_trimmed[c_in]
                                    attributions[source_idx][b][
                                        s_in, trimmed_c_in, s_out, trimmed_c_out
                                    ] = weighted[b, s_in, c_in]

        # Build output per set
        for source_idx, (source, source_infos) in enumerate(
            zip(sources, all_source_infos, strict=True)
        ):
            original_source_infos = original_alive_info[source]
            original_target_infos = original_alive_info[target]
            for b, set_name in enumerate(set_names):
                local_attributions_by_set[set_name].append(
                    PairAttribution(
                        source=source,
                        target=target,
                        attribution=attributions[source_idx][b],
                        trimmed_c_in_idxs=source_infos[b].alive_c_idxs,
                        trimmed_c_out_idxs=target_infos[b].alive_c_idxs,
                        is_kv_to_o_pair=is_kv_to_o_pair(source, target),
                        original_alive_mask_in=original_source_infos[b].alive_mask,  # [seq, C]
                        original_alive_mask_out=original_target_infos[b].alive_mask,  # [seq, C]
                    )
                )

    return local_attributions_by_set, output_probs


def main(
    wandb_path: str,
    n_blocks: int,
    ci_threshold: float,
    output_prob_threshold: float,
    ci_vals_path: str | Path | None,
    prompts: list[str] | None = None,
) -> None:
    """Compute local attributions for a prompt.

    Args:
        wandb_path: WandB path to load model from.
        n_blocks: Number of transformer blocks to analyze.
        ci_threshold: Threshold for considering a component alive.
        output_prob_threshold: Threshold for considering an output logit alive.
        ci_vals_path: Path to JSON with precomputed CI values and prompts.
            If provided, prompts are read from the JSON file.
        prompts: List of prompts to use when ci_vals_path is None. Required if ci_vals_path is None.
    """
    loaded = load_model_from_wandb(wandb_path)
    model, config, device = loaded.model, loaded.config, loaded.device

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Expected PreTrainedTokenizerFast"
    sources_by_target = get_sources_by_target(model, device, config, n_blocks)

    n_pairs = sum(len(ins) for ins in sources_by_target.values())
    print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
    for out_layer, in_layers in sources_by_target.items():
        print(f"  {out_layer} <- {in_layers}")

    data = get_tokens_and_ci(
        model=model,
        tokenizer=tokenizer,
        sampling=config.sampling,
        device=device,
        ci_vals_path=ci_vals_path,
        prompts=prompts,
    )

    # Compute local attributions
    print("\nComputing local attributions...")
    attr_pairs_by_set, output_probs = compute_local_attributions(
        model=model,
        tokens=data.tokens,
        sources_by_target=sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=config.sampling,
        device=device,
        ci_lower_leaky=data.ci_lower_leaky,
        set_names=data.set_names,
    )

    # Print summary statistics per set
    for set_name, attr_pairs in attr_pairs_by_set.items():
        print(f"\nAttribution summary for {set_name}:")
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

    # Save and plot per set
    out_dir = get_out_dir()

    for b, set_name in enumerate(data.set_names):
        attr_pairs = attr_pairs_by_set[set_name]
        pt_path = out_dir / f"local_attributions_{loaded.wandb_id}_{set_name}.pt"
        output_path = out_dir / f"local_attribution_graph_{loaded.wandb_id}_{set_name}.png"

        output_token_labels: dict[int, str] = {}
        output_probs_by_pos: dict[tuple[int, int], float] = {}
        for attr_pair in attr_pairs:
            if attr_pair.target == "output":
                for c_idx in attr_pair.trimmed_c_out_idxs:
                    if c_idx not in output_token_labels:
                        output_token_labels[c_idx] = tokenizer.decode([c_idx])
                    # Store probability for each position
                    for s in range(data.tokens.shape[1]):
                        prob = output_probs[b, s, c_idx].item()
                        if prob >= output_prob_threshold:
                            output_probs_by_pos[(s, c_idx)] = prob
                break

        save_data = {
            "attr_pairs": attr_pairs,
            "token_strings": data.all_token_strings[b],
            "prompt": data.prompts[b],
            "ci_threshold": ci_threshold,
            "output_prob_threshold": output_prob_threshold,
            "output_token_labels": output_token_labels,
            "output_probs_by_pos": output_probs_by_pos,
            "wandb_id": loaded.wandb_id,
            "set_name": set_name,
        }
        torch.save(save_data, pt_path)
        print(f"\nSaved local attributions for {set_name} to {pt_path}")

        fig = plot_local_graph(
            attr_pairs=attr_pairs,
            token_strings=data.all_token_strings[b],
            output_token_labels=output_token_labels,
            output_prob_threshold=output_prob_threshold,
            output_probs_by_pos=output_probs_by_pos,
            min_edge_weight=0.0001,
            node_scale=30.0,
            edge_alpha_scale=0.7,
        )

        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved figure for {set_name} to {output_path}")


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
    # ci_vals_path = Path("spd/scripts/optim_cis/out/optimized_ci_33n6xjjt.json")
    ci_vals_path = None
    prompts = ["They walked hand in", "She is a happy"]
    main(
        wandb_path=wandb_path,
        n_blocks=n_blocks,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        ci_vals_path=ci_vals_path,
        prompts=prompts,
    )
