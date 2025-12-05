"""Context length analysis script.

Analyzes how the set of components that activate changes depending on context length.
Measures the per-layer Jaccard similarity between binarized CI vectors at full context
vs each shorter context length.

Usage:
    python spd/scripts/ctx_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, set_seed


def compute_binary_ci(
    model: ComponentModel,
    batch: Float[Tensor, "batch seq"],
    ci_threshold: float,
) -> dict[str, Float[Tensor, "batch n_components"]]:
    """Compute binarized CI values for the final token.

    Args:
        model: The ComponentModel to analyze.
        batch: Input token IDs of shape [batch_size, seq_len].
        ci_threshold: Threshold for binarizing CI values.

    Returns:
        Dictionary mapping layer names to binarized CI tensors of shape [batch_size, n_components].
    """
    output = model(batch, cache_type="input")

    ci = model.calc_causal_importances(
        pre_weight_acts=output.cache,
        detach_inputs=True,
        sampling="continuous",
    )

    binary_ci: dict[str, Float[Tensor, "batch n_components"]] = {}
    for layer_name, ci_tensor in ci.lower_leaky.items():
        # ci_tensor shape: [batch, seq_len, n_components]
        final_token_ci = ci_tensor[:, -1, :]  # [batch, n_components]
        binary_ci[layer_name] = (final_token_ci > ci_threshold).float()

    return binary_ci


class BatchAnalysisResult:
    """Results from analyzing a single batch."""

    similarities: dict[str, Float[Tensor, "n_ctx_minus_1 batch"]]
    active_counts: dict[str, Float[Tensor, " batch"]]  # Number of active components per sample

    def __init__(
        self,
        similarities: dict[str, Float[Tensor, "n_ctx_minus_1 batch"]],
        active_counts: dict[str, Float[Tensor, " batch"]],
    ):
        self.similarities = similarities
        self.active_counts = active_counts


def analyze_batch(
    model: ComponentModel,
    batch: Float[Tensor, "batch max_ctx"],
    ci_threshold: float,
    max_ctx: int,
) -> BatchAnalysisResult:
    """Compute Jaccard similarities at all context lengths vs full context for a single batch.

    Args:
        model: The ComponentModel to analyze.
        batch: Input token IDs of shape [batch_size, max_ctx].
        ci_threshold: Threshold for binarizing CI values.
        max_ctx: Maximum context length.

    Returns:
        BatchAnalysisResult containing similarities and active component counts.
    """
    batch_size = batch.shape[0]
    seq_len = batch.shape[1]

    if seq_len < max_ctx:
        logger.warning(
            f"Batch sequence length ({seq_len}) < max_ctx ({max_ctx}). Using available length."
        )
        effective_max_ctx = seq_len
    else:
        effective_max_ctx = max_ctx

    # Compute reference CI at full context
    full_batch = batch[:, -effective_max_ctx:]
    binary_ci_full = compute_binary_ci(model, full_batch, ci_threshold)

    # Initialize similarities tensor for each layer
    layer_names = list(binary_ci_full.keys())
    similarities: dict[str, Float[Tensor, "n_ctx_minus_1 batch"]] = {
        layer_name: torch.zeros(effective_max_ctx - 1, batch_size, device=batch.device)
        for layer_name in layer_names
    }

    # Count active components at full context for each layer
    active_counts: dict[str, Float[Tensor, " batch"]] = {
        layer_name: binary_ci_full[layer_name].sum(dim=-1)  # [batch]
        for layer_name in layer_names
    }

    # Compute similarities for each context length from 1 to (max_ctx - 1)
    for ctx_idx, ctx_len in enumerate(range(1, effective_max_ctx)):
        # Truncate to last ctx_len tokens
        truncated_batch = batch[:, -ctx_len:]

        # Compute binary CI for truncated context
        binary_ci_ctx = compute_binary_ci(model, truncated_batch, ci_threshold)

        # Compute Jaccard similarity for each layer: |A ∩ B| / |A ∪ B|
        for layer_name in layer_names:
            ci_ctx = binary_ci_ctx[layer_name]
            ci_full = binary_ci_full[layer_name]

            intersection = (ci_ctx * ci_full).sum(dim=-1)  # [batch]
            union = ((ci_ctx + ci_full) > 0).float().sum(dim=-1)  # [batch]

            # Handle case where union is 0 (both vectors all zeros) -> similarity = 1.0
            jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
            similarities[layer_name][ctx_idx] = jaccard

    return BatchAnalysisResult(similarities=similarities, active_counts=active_counts)


def extract_block_number(layer_name: str) -> int | None:
    """Extract transformer block number from layer name.

    Handles patterns like:
    - "h.0.mlp" -> 0
    - "model.layers.0.mlp" -> 0
    """
    import re

    # GPT-2 style: h.0.*, h.1.*, etc.
    match = re.search(r"h\.(\d+)\.", layer_name)
    if match:
        return int(match.group(1))

    # LLaMA style: model.layers.0.*, model.layers.1.*, etc.
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if match:
        return int(match.group(1))

    return None


def plot_results(
    mean_similarities_by_layer: dict[str, Float[Tensor, " n_ctx_minus_1"]],
    mean_active_counts_by_layer: dict[str, float],
    output_path: Path,
    ci_threshold: float,
    max_ctx: int,
    run_str: str,
) -> None:
    """Create and save matplotlib figure showing Jaccard similarity vs context length.

    Layers are grouped by transformer block, with each block in a separate subplot
    arranged in a grid with max 2 columns.

    Args:
        mean_similarities_by_layer: Dictionary mapping layer names to mean Jaccard similarities.
        mean_active_counts_by_layer: Dictionary mapping layer names to mean active component counts.
        output_path: Path to save the figure.
        ci_threshold: CI threshold used (for title).
        max_ctx: Maximum context length (for x-axis).
        run_str: String identifying the results (for title).
    """
    import math
    from collections import defaultdict

    # Group layers by block number
    layers_by_block: dict[int, list[str]] = defaultdict(list)
    for layer_name in mean_similarities_by_layer:
        block_num = extract_block_number(layer_name)
        if block_num is not None:
            layers_by_block[block_num].append(layer_name)
        else:
            # Fallback: treat as block -1 for layers without block numbers
            layers_by_block[-1].append(layer_name)

    n_blocks = len(layers_by_block)
    n_cols = min(2, n_blocks)
    n_rows = math.ceil(n_blocks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    n_ctx_minus_1 = len(next(iter(mean_similarities_by_layer.values())))
    context_lengths = list(range(1, n_ctx_minus_1 + 1))

    for idx, block_num in enumerate(sorted(layers_by_block.keys())):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        layer_names = layers_by_block[block_num]
        for layer_name in sorted(layer_names):
            similarities = mean_similarities_by_layer[layer_name]
            mean_active = mean_active_counts_by_layer[layer_name]
            # Use just the component type (e.g., "mlp", "attn") for cleaner labels
            short_name = layer_name.split(".")[-1]
            label = f"{short_name} (avg active: {mean_active:.1f})"

            ax.plot(
                context_lengths,
                similarities.cpu().numpy(),
                label=label,
                marker=".",
                markersize=2,
                linewidth=0.8,
                alpha=0.8,
            )

        ax.set_xlabel("Context Length")
        ax.set_ylabel("Mean Jaccard Similarity")
        ax.set_ylim(0, 1)
        block_title = f"Block {block_num}" if block_num >= 0 else "Other"
        ax.set_title(block_title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.annotate(
            f"Reference: full context ({max_ctx} tokens)",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=8,
            color="gray",
        )

    # Hide unused subplots
    for idx in range(n_blocks, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"CI Jaccard Similarity vs Context Length\nModel: {run_str}\nCI Threshold: {ci_threshold}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved to {output_path}")


def main(
    model_path: str,
    model_label: str,
    seed: int,
    ci_threshold: float,
    max_ctx: int,
    batch_size: int,
    n_batches: int,
    output_dir: Path,
    device: str | None = None,
) -> None:
    """Run context length analysis.

    Args:
        model_path: WandB path to the SPD model (e.g., "wandb:entity/project/run_id").
        model_label: Label for the model (e.g., "ss_gpt2_simple-1L").
        seed: Random seed for reproducibility.
        ci_threshold: Threshold for binarizing CI values.
        max_ctx: Maximum context length to analyze.
        n_batches: Number of batches to accumulate.
        output_dir: Directory to save outputs.
        device: Device to run on (default: auto-detect).
        batch_size: Batch size for data loading.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")

    # Load model and config
    logger.info(f"Loading model from: {model_path}")
    run_info = SPDRunInfo.from_path(model_path)
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), (
        f"This script only supports LM tasks, got {type(config.task_config).__name__}"
    )
    task_config = config.task_config

    # Create dataloader with our max_ctx
    assert config.tokenizer_name, "tokenizer_name must be set in config"

    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,  # Deterministic for analysis
        seed=seed,
    )

    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=seed,
    )
    data_iter = iter(loader)

    # Accumulate similarities and active counts over batches
    logger.info(f"Analyzing {n_batches} batches with batch size {batch_size}...")
    all_similarities: dict[str, list[Float[Tensor, "n_ctx_minus_1 batch"]]] = {}
    all_active_counts: dict[str, list[Float[Tensor, " batch"]]] = {}
    total_samples = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Analyzing batches"):
            try:
                batch_raw = next(data_iter)
            except StopIteration:
                logger.warning(f"Data exhausted after {batch_idx} batches")
                break

            batch = extract_batch_data(batch_raw).to(device)
            batch_size_actual = batch.shape[0]
            total_samples += batch_size_actual

            result = analyze_batch(
                model=model, batch=batch, ci_threshold=ci_threshold, max_ctx=max_ctx
            )

            # Accumulate similarities
            for layer_name, sim_tensor in result.similarities.items():
                if layer_name not in all_similarities:
                    all_similarities[layer_name] = []
                all_similarities[layer_name].append(sim_tensor)

            # Accumulate active counts
            for layer_name, count_tensor in result.active_counts.items():
                if layer_name not in all_active_counts:
                    all_active_counts[layer_name] = []
                all_active_counts[layer_name].append(count_tensor)

    # Compute mean similarities across all samples
    logger.info(f"Computing mean similarities over {total_samples} samples...")
    mean_similarities_by_layer: dict[str, Float[Tensor, " n_ctx_minus_1"]] = {}

    for layer_name, sim_list in all_similarities.items():
        # Concatenate along batch dimension: [n_ctx_minus_1, total_samples]
        all_sim = torch.cat(sim_list, dim=1)
        # Mean over samples
        mean_similarities_by_layer[layer_name] = all_sim.mean(dim=1)

    # Compute mean active counts across all samples
    mean_active_counts_by_layer: dict[str, float] = {}
    for layer_name, count_list in all_active_counts.items():
        all_counts = torch.cat(count_list, dim=0)
        mean_active_counts_by_layer[layer_name] = all_counts.mean().item()

    # Save raw results
    results = {
        "mean_similarities_by_layer": {
            k: v.cpu().tolist() for k, v in mean_similarities_by_layer.items()
        },
        "mean_active_counts_by_layer": mean_active_counts_by_layer,
        "model_path": model_path,
        "ci_threshold": ci_threshold,
        "max_ctx": max_ctx,
        "n_batches": n_batches,
        "total_samples": total_samples,
    }

    import json

    wandb_id = model_path.split("/")[-1]
    threshold_str = f"{ci_threshold:.2e}".replace(".", "p")
    run_str = f"{model_label}_{wandb_id}_seed{seed}_n-samples{total_samples}_thresh{threshold_str}"
    results_path = output_dir / f"ctx_analysis_{run_str}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    plot_path = output_dir / f"ctx_analysis_{run_str}.png"
    plot_results(
        mean_similarities_by_layer=mean_similarities_by_layer,
        mean_active_counts_by_layer=mean_active_counts_by_layer,
        output_path=plot_path,
        ci_threshold=ci_threshold,
        max_ctx=max_ctx,
        run_str=run_str,
    )


if __name__ == "__main__":
    path_and_labels = [
        ("goodfire/spd/33n6xjjt", "ss_gpt2_simple-1L"),
        ("goodfire/spd/aa38p449", "ss_llama_simple-1L"),
        ("goodfire/spd/jyo9duz5", "ss_gpt2_simple-4L"),
        ("goodfire/spd/lxs77xye", "ss_llama-4L"),
    ]
    for model_path, model_label in path_and_labels:
        main(
            model_path=model_path,
            model_label=model_label,
            seed=0,
            ci_threshold=0.1,
            max_ctx=511,  # We don't actually get training signal on the 512th token with the way we've done the labelling
            batch_size=64,
            n_batches=10,
            output_dir=Path(__file__).parent / "out" / "ctx_analysis_jac",
        )
