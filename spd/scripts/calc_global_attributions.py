# %%

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.plotting import plot_mean_component_cis_both_scales
from spd.utils.general_utils import extract_batch_data


def is_qkv_to_o_pair(in_layer: str, out_layer: str) -> bool:
    """Check if pair requires per-sequence-position gradient computation.

    For q/k/v → o_proj within the same attention block, output at s_out
    has gradients w.r.t. inputs at all s_in ≤ s_out (causal attention).
    """
    in_is_qkv = any(x in in_layer for x in ["q_proj", "k_proj", "v_proj"])
    out_is_o = "o_proj" in out_layer
    if not (in_is_qkv and out_is_o):
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

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        max_batches: Maximum number of batches to process.

    Returns:
        Dictionary mapping module path -> tensor of shape [C] with mean CI per component.
    """
    # Initialize accumulators
    ci_sums: dict[str, Tensor] = {
        module_name: torch.zeros(model.C, device=device) for module_name in model.components
    }
    examples_seen: dict[str, int] = {module_name: 0 for module_name in model.components}

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

    # Compute means
    mean_cis = {
        module_name: ci_sums[module_name] / examples_seen[module_name]
        for module_name in model.components
    }

    return mean_cis


def compute_alive_components(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    max_batches: int | None,
    threshold: float,
) -> tuple[dict[str, Tensor], dict[str, list[int]], tuple[Image.Image, Image.Image]]:
    """Compute alive components based on mean CI threshold.

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        max_batches: Maximum number of batches to process.
        threshold: Minimum mean CI to consider a component alive.

    Returns:
        Tuple of:
        - mean_cis: Dictionary mapping module path -> tensor of mean CI per component
        - alive_indices: Dictionary mapping module path -> list of alive component indices
        - images: Tuple of (linear_scale_image, log_scale_image) for verification
    """
    mean_cis = compute_mean_ci_per_component(model, data_loader, device, config, max_batches)
    alive_indices = {}
    for module_name, mean_ci in mean_cis.items():
        alive_mask = mean_ci >= threshold
        alive_indices[module_name] = torch.where(alive_mask)[0].tolist()
    images = plot_mean_component_cis_both_scales(mean_cis)

    return mean_cis, alive_indices, images


def get_sources_by_target(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    n_blocks: int,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
    # Get an arbitrary batch
    batch_raw = next(iter(data_loader))
    batch = extract_batch_data(batch_raw).to(device)
    print(f"Batch shape: {batch.shape}")

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(batch, cache_type="input")
        print(f"Output shape: {output_with_cache.output.shape}")
        print(f"Number of cached layers: {len(output_with_cache.cache)}")
        print(f"Cached layer names: {list(output_with_cache.cache.keys())}")

    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )

    # Create masks for component replacement (use all components with causal importance as mask)
    component_masks = ci.lower_leaky
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        routing_masks="all",
    )
    with torch.enable_grad():
        comp_output_with_cache_grad: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )

    cache = comp_output_with_cache_grad.cache
    layers = []
    layer_names = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    ]
    for i in range(n_blocks):
        layers.extend([f"h.{i}.{layer_name}" for layer_name in layer_names])

    test_pairs = []
    for in_layer in layers:
        for out_layer in layers:
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
        if grad is not None and grad.abs().max().item() > 0:  # pyright: ignore[reportUnnecessaryComparison]
            sources_by_target[out_layer].append(in_layer)
    return dict(sources_by_target)


def validate_attention_pair_structure(sources_by_target: dict[str, list[str]]) -> None:
    """Assert that o_proj layers only receive from same-block QKV.

    This structural property allows us to handle attention and non-attention
    cases separately without mixing within a single target layer.
    """
    for out_layer, in_layers in sources_by_target.items():
        if "o_proj" in out_layer:
            out_block = out_layer.split(".")[1]
            for in_layer in in_layers:
                assert any(x in in_layer for x in ["q_proj", "k_proj", "v_proj"]), (
                    f"o_proj output {out_layer} has non-QKV input {in_layer}"
                )
                in_block = in_layer.split(".")[1]
                assert in_block == out_block, (
                    f"o_proj output {out_layer} has input from different block: {in_layer}"
                )
        else:
            for in_layer in in_layers:
                assert not is_qkv_to_o_pair(in_layer, out_layer), (
                    f"Non-o_proj output {out_layer} has attention pair input {in_layer}"
                )


def compute_global_attributions(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    sources_by_target: dict[str, list[str]],
    max_batches: int,
    alive_indices: dict[str, list[int]],
    ci_attribution_threshold: float,
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

    Returns:
        Dictionary mapping (in_layer, out_layer) -> attribution tensor of shape [n_alive_in, n_alive_out]
        where attribution[i, j] is the mean absolute gradient from the i-th alive input component
        to the j-th alive output component.
    """
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

        # Create masks and run forward pass with gradient tracking
        component_masks = ci.lower_leaky
        mask_infos = make_mask_infos(
            component_masks=component_masks,
            routing_masks="all",
        )

        with torch.enable_grad():
            comp_output_with_cache: OutputWithCache = model(
                batch,
                mask_infos=mask_infos,
                cache_type="component_acts",
            )

        cache = comp_output_with_cache.cache

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

            is_attention_output = "o_proj" in out_layer
            tqdm.write(
                f"{'Attention' if is_attention_output else 'Non-attention'} target: "
                f"{out_layer} <- {in_layers}"
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
                            allow_unused=True,
                        )

                        with torch.no_grad():
                            for i, in_layer in enumerate(in_layers):
                                grads = grads_tuple[i]
                                assert grads is not None, f"Gradient is None for {in_layer}"
                                alive_in = alive_indices[in_layer]
                                weighted = grads * in_tensors[i]
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
                            alive_in = alive_indices[in_layer]
                            weighted = grads * in_tensors[i]
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

    global_attributions = {}
    for pair, attr_sum in attribution_sums.items():
        attr: Float[Tensor, "n_alive_in n_alive_out"] = attr_sum / samples_per_pair[pair]
        global_attributions[pair] = attr / attr.sum(dim=1, keepdim=True)

    n_pairs = len(attribution_sums)
    total_samples = sum(samples_per_pair.values()) // n_pairs if n_pairs else 0
    print(f"Computed global attributions over ~{total_samples} samples per pair")
    return global_attributions


# %%
# Configuration
# wandb_path = "wandb:goodfire/spd/runs/jyo9duz5" # ss_gpt2_simple-1.25M (4L)
# wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
wandb_path = "wandb:goodfire/spd/runs/8ynfbr38"  # ss_gpt2_simple-1L
n_blocks = 1
batch_size = 1024
n_ctx = 64
# n_attribution_batches = 20
n_attribution_batches = 5
n_alive_calc_batches = 100
# n_alive_calc_batches = 200
ci_mean_alive_threshold = 1e-6
ci_attribution_threshold = 1e-6
dataset_seed = 0

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
wandb_id = wandb_path.split("/")[-1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading model from {wandb_path}...")

# Load the model
run_info = SPDRunInfo.from_path(wandb_path)
config: Config = run_info.config
model = ComponentModel.from_run_info(run_info)
model = model.to(device)
model.eval()

print("Model loaded successfully!")
print(f"Number of components: {model.C}")
print(f"Target module paths: {model.target_module_paths}")

# Load the dataset
task_config = config.task_config
assert isinstance(task_config, LMTaskConfig), "Expected LM task config"

dataset_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,  # Using train split for now
    n_ctx=n_ctx,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=False,  # No need to shuffle for testing
    seed=dataset_seed,
)

print(f"\nLoading dataset {dataset_config.name}...")
data_loader, tokenizer = create_data_loader(
    dataset_config=dataset_config,
    batch_size=batch_size,
    buffer_size=task_config.buffer_size,
    global_seed=dataset_seed,
    ddp_rank=0,
    ddp_world_size=1,
)

sources_by_target = get_sources_by_target(model, data_loader, device, config, n_blocks)
validate_attention_pair_structure(sources_by_target)

n_pairs = sum(len(ins) for ins in sources_by_target.values())
print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
for out_layer, in_layers in sources_by_target.items():
    print(f"  {out_layer} <- {in_layers}")
# %%
# Compute alive components based on mean CI threshold
print("\nComputing alive components based on mean CI...")
mean_cis, alive_indices, (img_linear, img_log) = compute_alive_components(
    model=model,
    data_loader=data_loader,
    device=device,
    config=config,
    max_batches=n_alive_calc_batches,
    threshold=ci_mean_alive_threshold,
)

# Print summary
print("\nAlive components per layer:")
for module_name, indices in alive_indices.items():
    n_alive = len(indices)
    print(f"  {module_name}: {n_alive}/{model.C} alive")

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
)

# Print summary statistics
for pair, attr in global_attributions.items():
    print(f"{pair[0]} -> {pair[1]}: mean={attr.mean():.6f}, max={attr.max():.6f}")

# %%
# Save attributions in both PyTorch and JSON formats
print("\nSaving attribution data...")
out_dir = Path(__file__).parent / "out"

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
