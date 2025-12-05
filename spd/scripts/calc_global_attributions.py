# %%

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import Config
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.plotting import plot_mean_component_cis_both_scales
from spd.scripts.model_loading import (
    create_data_loader_from_config,
    get_out_dir,
    load_model_from_wandb,
)
from spd.utils.general_utils import extract_batch_data


def is_kv_to_o_pair(in_layer: str, out_layer: str) -> bool:
    """Check if pair requires per-sequence-position gradient computation.

    For k/v → o_proj within the same attention block, output at s_out
    has gradients w.r.t. inputs at all s_in ≤ s_out (causal attention).
    """
    in_is_kv = any(x in in_layer for x in ["k_proj", "v_proj"])
    out_is_o = "o_proj" in out_layer
    if not (in_is_kv and out_is_o):
        return False

    # Check same attention block: "h.{idx}.attn.{proj}"
    in_block = in_layer.split(".")[1]
    out_block = out_layer.split(".")[1]
    return in_block == out_block


# %%
def compute_mean_ci_per_component(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    max_batches: int | None,
) -> dict[str, Tensor]:
    """Compute mean causal importance per component over the dataset.

    Also computes mean output probability per vocab token.

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        max_batches: Maximum number of batches to process.

    Returns:
        Dictionary mapping module path -> tensor of shape [C] with mean CI per component.
        Also includes "output" -> tensor of shape [vocab_size] with mean output probability.
    """
    # Initialize accumulators
    ci_sums: dict[str, Tensor] = {
        module_name: torch.zeros(model.C, device=device) for module_name in model.components
    }
    examples_seen: dict[str, int] = {module_name: 0 for module_name in model.components}

    # Output prob accumulators (initialized on first batch to get vocab_size)
    output_prob_sum: Tensor | None = None
    output_examples_seen = 0

    if max_batches is not None:
        batch_pbar = tqdm(enumerate(data_loader), desc="Computing mean CI", total=max_batches)
    else:
        batch_pbar = tqdm(enumerate(data_loader), desc="Computing mean CI")

    for batch_idx, batch_raw in batch_pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = extract_batch_data(batch_raw).to(device)

        with torch.no_grad():
            output_with_cache: OutputWithCache = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                sampling=config.sampling,
                detach_inputs=False,
            )

        # Accumulate CI values (using lower_leaky as in CIMeanPerComponent)
        for module_name, ci_vals in ci.lower_leaky.items():
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()
            examples_seen[module_name] += n_examples
            leading_dim_idxs = tuple(range(n_leading_dims))
            ci_sums[module_name] += ci_vals.sum(dim=leading_dim_idxs)

        # Accumulate output probabilities
        output_probs = torch.softmax(output_with_cache.output, dim=-1)  # [b, s, vocab]
        if output_prob_sum is None:
            vocab_size = output_probs.shape[-1]
            output_prob_sum = torch.zeros(vocab_size, device=device)
        output_prob_sum += output_probs.sum(dim=(0, 1))
        output_examples_seen += output_probs.shape[0] * output_probs.shape[1]

    # Compute means
    mean_cis: dict[str, Tensor] = {
        module_name: ci_sums[module_name] / examples_seen[module_name]
        for module_name in model.components
    }
    assert output_prob_sum is not None, "No batches processed"
    mean_cis["output"] = output_prob_sum / output_examples_seen

    return mean_cis


def compute_alive_components(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    max_batches: int | None,
    threshold: float,
    output_mean_prob_threshold: float,
) -> tuple[dict[str, Tensor], dict[str, list[int]], tuple[Image.Image, Image.Image]]:
    """Compute alive components based on mean CI threshold.

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        max_batches: Maximum number of batches to process.
        threshold: Minimum mean CI to consider a component alive.
        output_mean_prob_threshold: Minimum mean output probability to consider a token alive.

    Returns:
        Tuple of:
        - mean_cis: Dictionary mapping module path -> tensor of mean CI per component
          (includes "output" key with mean output probabilities)
        - alive_indices: Dictionary mapping module path -> list of alive component indices
          (includes "output" key)
        - images: Tuple of (linear_scale_image, log_scale_image) for verification
    """
    mean_cis = compute_mean_ci_per_component(model, data_loader, device, config, max_batches)

    alive_indices = {}
    for module_name, mean_val in mean_cis.items():
        # Use output_mean_prob_threshold for output layer, threshold for components
        thresh = output_mean_prob_threshold if module_name == "output" else threshold
        alive_mask = mean_val >= thresh
        alive_indices[module_name] = torch.where(alive_mask)[0].tolist()

    images = plot_mean_component_cis_both_scales(mean_cis)

    return mean_cis, alive_indices, images


def get_sources_by_target(
    model: ComponentModel,
    device: str,
    config: Config,
    n_blocks: int,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
    # Use a small dummy batch - we only need to trace gradient connections
    batch: Float[Tensor, "batch seq"] = torch.zeros(2, 3, dtype=torch.long, device=device)

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(batch, cache_type="input")
        print(f"Output shape: {output_with_cache.output.shape}")

    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )

    # Create masks so we can use all components (without masks)
    mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        routing_masks="all",
    )

    wte_cache: dict[str, Tensor] = {}

    # Add an extra forward hook to the model to cache the output of model.target_model.wte
    def wte_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        # We call it "post_detach" for consistency, we don't bother detaching here as there are
        # no modules before it that we care about
        wte_cache["wte_post_detach"] = output
        return output

    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    with torch.enable_grad():
        comp_output_with_cache_grad: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )
    wte_handle.remove()

    cache = comp_output_with_cache_grad.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
    cache["output_pre_detach"] = comp_output_with_cache_grad.output

    layers = ["wte"]
    component_layers = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    ]
    for i in range(n_blocks):
        layers.extend([f"h.{i}.{layer_name}" for layer_name in component_layers])
    layers.append("output")

    test_pairs = []
    for in_layer in layers[:-1]:  # Don't include "output" in
        for out_layer in layers[1:]:  # Don't include "wte" in out_layers
            if layers.index(in_layer) < layers.index(out_layer):
                test_pairs.append((in_layer, out_layer))

    sources_by_target: dict[str, list[str]] = defaultdict(list)
    for in_layer, out_layer in test_pairs:
        out_pre_detach = cache[f"{out_layer}_pre_detach"]
        in_post_detach = cache[f"{in_layer}_post_detach"]
        out_value = out_pre_detach[0, 0, 0]  # Pick arbitrary value
        grads = torch.autograd.grad(
            outputs=out_value,
            inputs=in_post_detach,
            retain_graph=True,
            allow_unused=True,
        )
        assert len(grads) == 1, "Expected 1 gradient"
        grad = grads[0]
        if grad is not None:  # pyright: ignore[reportUnnecessaryComparison]
            sources_by_target[out_layer].append(in_layer)
    return dict(sources_by_target)


def compute_global_attributions(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    sources_by_target: dict[str, list[str]],
    max_batches: int,
    alive_indices: dict[str, list[int]],
    ci_attribution_threshold: float,
    output_mean_prob_threshold: float,
) -> dict[tuple[str, str], Tensor]:
    """Compute global attributions accumulated over the dataset.

    For each valid layer pair (in_layer, out_layer), computes the mean absolute gradient
    of output component activations with respect to input component activations,
    averaged over batch, sequence positions, and number of batches.

    Optimization: For each target layer, we batch all source layers into a single
    autograd.grad call, sharing backward computation.

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        max_batches: Maximum number of batches to process.
        alive_indices: Dictionary mapping module path -> list of alive component indices.
        ci_attribution_threshold: Threshold for considering a component for the attribution calculation.
        output_mean_prob_threshold: Threshold for considering an output logit alive (on softmax probs).

    Returns:
        Dictionary mapping (in_layer, out_layer) -> attribution tensor of shape [n_alive_in, n_alive_out]
        where attribution[i, j] is the mean absolute gradient from the i-th alive input component
        to the j-th alive output component.
    """
    alive_indices["wte"] = [0]  # Treat wte as single alive component

    # Initialize accumulators for each (in_layer, out_layer) pair
    attribution_sums: dict[tuple[str, str], Float[Tensor, "n_alive_in n_alive_out"]] = {}
    samples_per_pair: dict[tuple[str, str], int] = {}
    for out_layer, in_layers in sources_by_target.items():
        for in_layer in in_layers:
            n_alive_in = len(alive_indices[in_layer])
            n_alive_out = len(alive_indices[out_layer])
            attribution_sums[(in_layer, out_layer)] = torch.zeros(
                n_alive_in, n_alive_out, device=device
            )
            samples_per_pair[(in_layer, out_layer)] = 0

    # Set up wte hook
    wte_handle = None
    wte_cache: dict[str, Tensor] = {}

    def wte_hook(_module: nn.Module, _args: Any, _kwargs: Any, output: Tensor) -> Any:
        output.requires_grad_(True)
        wte_cache["wte_post_detach"] = output
        return output

    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    batch_pbar = tqdm(enumerate(data_loader), desc="Batches", total=max_batches)
    for batch_idx, batch_raw in batch_pbar:
        if batch_idx >= max_batches:
            break

        batch: Float[Tensor, "b s C"] = extract_batch_data(batch_raw).to(device)

        batch_size, n_seq = batch.shape

        # Forward pass to get pre-weight activations
        with torch.no_grad():
            output_with_cache: OutputWithCache = model(batch, cache_type="input")

        # Calculate causal importances for masking
        with torch.no_grad():
            ci = model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                sampling=config.sampling,
                detach_inputs=False,
            )

        mask_infos = make_mask_infos(
            # component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
            component_masks=ci.lower_leaky,
            routing_masks="all",
        )

        with torch.enable_grad():
            comp_output_with_cache: OutputWithCache = model(
                batch,
                mask_infos=mask_infos,
                cache_type="component_acts",
            )

        cache = comp_output_with_cache.cache

        # Add wte to cache and CI
        cache["wte_post_detach"] = wte_cache["wte_post_detach"]
        # Add fake CI for wte: shape (b, s, embedding_dim) with 1.0 at index 0
        ci.lower_leaky["wte"] = torch.zeros_like(wte_cache["wte_post_detach"])
        ci.lower_leaky["wte"][:, :, 0] = 1.0

        # Add output to cache and CI
        cache["output_pre_detach"] = comp_output_with_cache.output
        # Use output probs as fake CI for output layer
        output_probs: Float[Tensor, "b s vocab"] = torch.softmax(
            comp_output_with_cache.output, dim=-1
        )
        # Only consider tokens above threshold as "alive" for this batch
        ci.lower_leaky["output"] = torch.where(
            output_probs >= output_mean_prob_threshold, output_probs, torch.zeros_like(output_probs)
        )

        # Compute attributions grouped by target layer
        for out_layer, in_layers in tqdm(
            sources_by_target.items(), desc="Target layers", leave=False
        ):
            out_pre_detach: Float[Tensor, "b s C"] = cache[f"{out_layer}_pre_detach"]
            alive_out: list[int] = alive_indices[out_layer]
            ci_out = ci.lower_leaky[out_layer]

            # Gather all input tensors for this target layer
            in_tensors = [cache[f"{in_layer}_post_detach"] for in_layer in in_layers]

            # Initialize batch attributions for each input layer
            batch_attributions = {
                in_layer: torch.zeros(len(alive_indices[in_layer]), len(alive_out), device=device)
                for in_layer in in_layers
            }

            # NOTE: o->q will be treated as an attention pair even though there are no attrs
            # across sequence positions. This is just so we don't have to special case it.
            is_attention_output = any(
                is_kv_to_o_pair(in_layer, out_layer) for in_layer in in_layers
            )

            grad_outputs: Float[Tensor, "b s C"] = torch.zeros_like(out_pre_detach)

            for c_enum, c_idx in tqdm(
                enumerate(alive_out), desc="Components", leave=False, total=len(alive_out)
            ):
                if is_attention_output:
                    # Attention target: loop over output sequence positions
                    for s_out in range(n_seq):
                        if ci_out[:, s_out, c_idx].sum() <= ci_attribution_threshold:
                            continue
                        grad_outputs.zero_()
                        grad_outputs[:, s_out, c_idx] = ci_out[:, s_out, c_idx].detach()

                        # Single autograd call for all input layers
                        grads_tuple = torch.autograd.grad(
                            outputs=out_pre_detach,
                            inputs=in_tensors,
                            grad_outputs=grad_outputs,
                            retain_graph=True,
                        )

                        with torch.no_grad():
                            for i, in_layer in enumerate(in_layers):
                                grads = grads_tuple[i]
                                assert grads is not None, f"Gradient is None for {in_layer}"
                                weighted = grads * in_tensors[i]

                                # Special handling for wte: sum over embedding_dim to get single component
                                if in_layer == "wte":
                                    # weighted is (b, s, embedding_dim), sum to (b, s, 1)
                                    weighted = weighted.sum(dim=-1, keepdim=True)
                                    alive_in = [0]
                                else:
                                    alive_in = alive_indices[in_layer]

                                # Only sum contributions from positions s_in <= s_out (causal)
                                weighted_alive = weighted[:, : s_out + 1, alive_in]
                                batch_attributions[in_layer][:, c_enum] += weighted_alive.pow(
                                    2
                                ).sum(dim=(0, 1))
                else:
                    # Non-attention target: vectorize over all (b, s) positions
                    if ci_out[:, :, c_idx].sum() <= ci_attribution_threshold:
                        continue
                    grad_outputs.zero_()
                    grad_outputs[:, :, c_idx] = ci_out[:, :, c_idx].detach()

                    # Single autograd call for all input layers
                    grads_tuple = torch.autograd.grad(
                        outputs=out_pre_detach,
                        inputs=in_tensors,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        allow_unused=True,
                    )

                    with torch.no_grad():
                        for i, in_layer in enumerate(in_layers):
                            grads = grads_tuple[i]
                            assert grads is not None, f"Gradient is None for {in_layer}"
                            weighted = grads * in_tensors[i]

                            # Special handling for wte: sum over embedding_dim to get single component
                            if in_layer == "wte":
                                # weighted is (b, s, embedding_dim), sum to (b, s, 1)
                                weighted = weighted.sum(dim=-1, keepdim=True)
                                alive_in = [0]
                            else:
                                alive_in = alive_indices[in_layer]

                            weighted_alive = weighted[:, :, alive_in]
                            batch_attributions[in_layer][:, c_enum] += weighted_alive.pow(2).sum(
                                dim=(0, 1)
                            )

            # Accumulate batch results and track samples
            for in_layer in in_layers:
                attribution_sums[(in_layer, out_layer)] += batch_attributions[in_layer]
                # Track samples: attention pairs have triangular sum due to causal masking
                if is_attention_output:
                    samples_per_pair[(in_layer, out_layer)] += batch_size * n_seq * (n_seq + 1) // 2
                else:
                    samples_per_pair[(in_layer, out_layer)] += batch_size * n_seq

    wte_handle.remove()

    global_attributions = {}
    for pair, attr_sum in attribution_sums.items():
        attr: Float[Tensor, "n_alive_in n_alive_out"] = attr_sum / samples_per_pair[pair]
        global_attributions[pair] = attr / attr.sum(dim=1, keepdim=True)

    n_pairs = len(attribution_sums)
    total_samples = sum(samples_per_pair.values()) // n_pairs if n_pairs else 0
    print(f"Computed global attributions over ~{total_samples} samples per pair")
    return global_attributions


if __name__ == "__main__":
    # %%
    # Configuration
    # wandb_path = "wandb:goodfire/spd/runs/jyo9duz5" # ss_gpt2_simple-1.25M (4L)
    # wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
    # wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L (Old)
    wandb_path = "wandb:goodfire/spd/runs/33n6xjjt"  # ss_gpt2_simple-1L (New)
    n_blocks = 1
    # batch_size = 1024
    batch_size = 128
    n_ctx = 64
    # n_attribution_batches = 20
    n_attribution_batches = 5
    n_alive_calc_batches = 100
    # n_alive_calc_batches = 200
    ci_mean_alive_threshold = 1e-6
    ci_attribution_threshold = 1e-6
    output_mean_prob_threshold = 1e-8
    dataset_seed = 0

    out_dir = get_out_dir()

    # Load model using shared utility
    loaded = load_model_from_wandb(wandb_path)
    model = loaded.model
    config = loaded.config
    device = loaded.device
    wandb_id = loaded.wandb_id

    # Load the dataset
    data_loader, tokenizer = create_data_loader_from_config(
        config=config,
        batch_size=batch_size,
        n_ctx=n_ctx,
        seed=dataset_seed,
    )

    sources_by_target = get_sources_by_target(model, device, config, n_blocks)
    n_pairs = sum(len(ins) for ins in sources_by_target.values())
    print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
    for out_layer, in_layers in sources_by_target.items():
        print(f"  {out_layer} <- {in_layers}")
    # %%
    # Compute alive components based on mean CI threshold (and output probability threshold)
    print("\nComputing alive components...")
    mean_cis, alive_indices, (img_linear, img_log) = compute_alive_components(
        model=model,
        data_loader=data_loader,
        device=device,
        config=config,
        max_batches=n_alive_calc_batches,
        threshold=ci_mean_alive_threshold,
        output_mean_prob_threshold=output_mean_prob_threshold,
    )

    # Print summary
    print("\nAlive components per layer:")
    for module_name, indices in alive_indices.items():
        n_alive = len(indices)
        print(f"  {module_name}: {n_alive} alive")

    # Save images for verification
    img_linear.save(out_dir / f"ci_mean_per_component_linear_{wandb_id}.png")
    img_log.save(out_dir / f"ci_mean_per_component_log_{wandb_id}.png")
    print(
        f"Saved verification images to {out_dir / f'ci_mean_per_component_linear_{wandb_id}.png'} and {out_dir / f'ci_mean_per_component_log_{wandb_id}.png'}"
    )
    # %%
    # Compute global attributions over the dataset
    print("\nComputing global attributions...")
    global_attributions = compute_global_attributions(
        model=model,
        data_loader=data_loader,
        device=device,
        config=config,
        sources_by_target=sources_by_target,
        max_batches=n_attribution_batches,
        alive_indices=alive_indices,
        ci_attribution_threshold=ci_attribution_threshold,
        output_mean_prob_threshold=output_mean_prob_threshold,
    )

    # Print summary statistics
    for pair, attr in global_attributions.items():
        print(f"{pair[0]} -> {pair[1]}: mean={attr.mean():.6f}, max={attr.max():.6f}")

    # %%
    # Save attributions in both PyTorch and JSON formats
    print("\nSaving attribution data...")

    # Save PyTorch format
    pt_path = out_dir / f"global_attributions_{wandb_id}.pt"
    torch.save(global_attributions, pt_path)
    print(f"Saved PyTorch format to {pt_path}")

    # Convert and save JSON format for web visualization
    attributions_json = {}
    for (in_layer, out_layer), attr_tensor in global_attributions.items():
        key = f"('{in_layer}', '{out_layer}')"
        # Keep full precision - just convert to list
        attributions_json[key] = attr_tensor.cpu().tolist()

    json_data = {
        "n_blocks": n_blocks,
        "attributions": attributions_json,
        "alive_indices": alive_indices,
    }

    json_path = out_dir / f"global_attributions_{wandb_id}.json"

    # Write JSON with compact formatting
    with open(json_path, "w") as f:
        json.dump(json_data, f, separators=(",", ":"), ensure_ascii=False)

    # Also save a compressed version for very large files
    gz_path = out_dir / f"global_attributions_{wandb_id}.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(json_data, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Saved JSON format to {json_path}")
    print(f"Saved compressed format to {gz_path}")
    print(f"  - {len(attributions_json)} layer pairs")
    print(f"  - {sum(len(v) for v in alive_indices.values())} total alive components")
    print(f"\nTo visualize: Open scripts/plot_attributions.html and load {json_path}")

    # %%
