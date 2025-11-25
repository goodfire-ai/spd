# %%

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


# %%
def compute_mean_ci_per_component(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    max_batches: int | None,
) -> dict[str, torch.Tensor]:
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
    ci_sums: dict[str, torch.Tensor] = {
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
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], tuple[Image.Image, Image.Image]]:
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


def get_valid_pairs(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    n_blocks: int,
) -> list[tuple[str, str]]:
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

    valid_pairs = []
    for in_layer, out_layer in test_pairs:
        out_pre_detach = cache[f"{out_layer}_pre_detach"]
        in_post_detach = cache[f"{in_layer}_post_detach"]
        batch_idx = 0
        seq_idx = 50
        target_component_idx = 10
        out_value = out_pre_detach[batch_idx, seq_idx, target_component_idx]
        try:
            grads = torch.autograd.grad(
                outputs=out_value,
                inputs=in_post_detach,
                retain_graph=True,
                allow_unused=True,
            )
            assert len(grads) == 1, "Expected 1 gradient"
            grad = grads[0]
            # torch.autograd.grad returns None for unused inputs when allow_unused=True
            has_grad = (
                grad.abs().max().item() > 1e-8
                if grad is not None  # pyright: ignore[reportUnnecessaryComparison]
                else False
            )
        except RuntimeError:
            has_grad = False
        if has_grad:
            valid_pairs.append((in_layer, out_layer))
    return valid_pairs


def compute_global_attributions(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    valid_pairs: list[tuple[str, str]],
    max_batches: int,
    alive_indices: dict[str, list[int]],
) -> dict[tuple[str, str], torch.Tensor]:
    """Compute global attributions accumulated over the dataset.

    For each valid layer pair (in_layer, out_layer), computes the mean absolute gradient
    of output component activations with respect to input component activations,
    averaged over batch, sequence positions, and number of batches.

    Args:
        model: The ComponentModel to analyze.
        data_loader: DataLoader providing batches.
        device: Device to run on.
        config: SPD config with sampling settings.
        valid_pairs: List of (in_layer, out_layer) pairs to compute attributions for.
        max_batches: Maximum number of batches to process.
        alive_indices: Dictionary mapping module path -> list of alive component indices.
    Returns:
        Dictionary mapping (in_layer, out_layer) -> attribution tensor of shape [n_alive_in, n_alive_out]
        where attribution[i, j] is the mean absolute gradient from the i-th alive input component to the j-th alive output component.
    """

    # Initialize accumulators for each valid pair
    attribution_sums: dict[tuple[str, str], Float[Tensor, "n_alive_in n_alive_out"]] = {}
    for pair in valid_pairs:
        in_layer, out_layer = pair
        n_alive_in = len(alive_indices[in_layer])
        n_alive_out = len(alive_indices[out_layer])
        attribution_sums[(in_layer, out_layer)] = torch.zeros(
            n_alive_in, n_alive_out, device=device
        )

    total_samples = 0  # Track total (batch * seq) samples processed

    batch_pbar = tqdm(enumerate(data_loader), desc="Batches", total=max_batches)
    for batch_idx, batch_raw in batch_pbar:
        if batch_idx >= max_batches:
            break

        batch = extract_batch_data(batch_raw).to(device)

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

        # Compute attributions for each valid pair
        pair_pbar = tqdm(valid_pairs, desc="Layer pairs", leave=False)
        for in_layer, out_layer in pair_pbar:
            out_pre_detach = cache[f"{out_layer}_pre_detach"]
            in_post_detach = cache[f"{in_layer}_post_detach"]

            # Compute gradients for each output component
            # out_pre_detach shape: [batch, seq, n_components]
            # in_post_detach shape: [batch, seq, n_components]
            batch_attribution = torch.zeros(
                len(alive_indices[in_layer]), len(alive_indices[out_layer]), device=device
            )

            for i, c_out in enumerate(alive_indices[out_layer]):
                # Sum over batch and seq to get a scalar for this output component
                out_sum = out_pre_detach[:, :, c_out].sum()

                grads = torch.autograd.grad(
                    outputs=out_sum, inputs=in_post_detach, retain_graph=True
                )[0]

                assert grads is not None, "Gradient is None"
                # grads shape: [batch, seq, n_components]
                # Only consider the components that are alive
                alive_grads = grads[..., alive_indices[in_layer]]
                # Mean absolute gradient over batch and seq for each input component
                mean_abs_grad = alive_grads.abs().mean(dim=(0, 1))  # [n_alive_components]
                batch_attribution[:, i] = mean_abs_grad

            attribution_sums[(in_layer, out_layer)] += batch_attribution

        total_samples += 1  # Count batches (already averaged over batch/seq within)

    # Average over number of batches
    global_attributions = {
        pair: attr_sum / total_samples for pair, attr_sum in attribution_sums.items()
    }

    print(f"Computed global attributions over {total_samples} batches")
    return global_attributions


# %%
# Configuration
# wandb_path = "wandb:goodfire/spd/runs/jyo9duz5" # ss_gpt2_simple-1.25M (4L)
wandb_path = "wandb:goodfire/spd/runs/c0k3z78g"  # ss_gpt2_simple-2L
n_blocks = 2
batch_size = 512
n_attribution_batches = 10
n_alive_calc_batches = 200
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
    n_ctx=task_config.max_seq_len,
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

valid_pairs = get_valid_pairs(model, data_loader, device, config, n_blocks)
print(f"Valid layer pairs: {valid_pairs}")
# %%
# Compute alive components based on mean CI threshold
print("\nComputing alive components based on mean CI...")
mean_cis, alive_indices, (img_linear, img_log) = compute_alive_components(
    model=model,
    data_loader=data_loader,
    device=device,
    config=config,
    max_batches=n_alive_calc_batches,
    threshold=1e-6,
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
    valid_pairs=valid_pairs,
    max_batches=n_attribution_batches,
    alive_indices=alive_indices,
)

# Print summary statistics
for pair, attr in global_attributions.items():
    print(f"{pair[0]} -> {pair[1]}: mean={attr.mean():.6f}, max={attr.max():.6f}")

torch.save(global_attributions, out_dir / f"global_attributions_{wandb_id}.pt")

# %%
# Plot the attribution graph
print("\nPlotting attribution graph...")
out_dir = Path(__file__).parent / "out"
global_attributions = torch.load(out_dir / f"global_attributions_{wandb_id}.pt")
# graph_img = plot_attribution_graph(
#     global_attributions=global_attributions,
#     alive_indices=alive_indices,
#     n_blocks=n_blocks,
#     output_path=out_dir / f"attribution_graph_{wandb_id}.png",
#     edge_threshold=0.0,
# )
# print(f"Attribution graph has {sum(len(v) for v in alive_indices.values())} nodes")

# %%
