# %%
"""Attention pattern visualization for CI decision tree analysis."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.ci_dt.config import CIDTConfig
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# ----------------------- configuration -----------------------

config = CIDTConfig(
    batch_size=16,
    n_batches=4,
    activation_threshold=0.01,
    max_depth=8,
    random_state=42,
)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ----------------------- load model -----------------------

wandb_run_path: str = "wandb:goodfire/spd/runs/lxs77xye"

spd_run: SPDRunInfo = SPDRunInfo.from_path(wandb_run_path)
model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
model.to(device)
cfg: Config = spd_run.config

print(f"Loaded model from {wandb_run_path}")

# %%
# ----------------------- load dataset -----------------------

# Create LM dataset and dataloader
assert isinstance(cfg.task_config, LMTaskConfig)
pretrained_model_name = cfg.pretrained_model_name
assert pretrained_model_name is not None

dataset_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=pretrained_model_name,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    column_name=cfg.task_config.column_name,
    is_tokenized=False,
    streaming=False,
    seed=0,
)
dataloader, _ = create_data_loader(
    dataset_config=dataset_config,
    batch_size=config.batch_size,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
print(f"Created LM dataset with {cfg.task_config.dataset_name}")

# %%
# ----------------------- extract attention patterns -----------------------


def extract_attention_patterns_multibatch(
    model: ComponentModel,
    device: torch.device | str,
    dataloader: DataLoader[Any],
    n_batches: int,
) -> dict[str, Float[Tensor, "total_samples n_heads seq_len seq_len"]]:
    """Extract attention patterns over multiple batches.

    Args:
        model: ComponentModel containing the transformer
        device: Device to run inference on
        dataloader: DataLoader to get batches from
        n_batches: Number of batches to process

    Returns:
        Dictionary mapping layer names to attention patterns (on CPU)
        Format: {layer_name: tensor of shape [total_samples, n_heads, seq_len, seq_len]}
    """
    print(f"Extracting attention patterns for {n_batches} batches...")
    all_attention_patterns: list[dict[str, Tensor]] = []

    for _batch_idx in tqdm(range(n_batches), desc="Batches", total=n_batches):
        batch_data = next(iter(dataloader))
        input_ids: Int[Tensor, "batch seq_len"] = batch_data["input_ids"].to(device)

        # Get attention patterns on GPU
        with torch.no_grad():
            outputs = model.target_model(input_ids, output_attentions=True)

        # Extract attention patterns
        # outputs.attentions is a tuple of tensors, one per layer
        # Each tensor has shape [batch, n_heads, seq_len, seq_len]
        batch_attention: dict[str, Tensor] = {}
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                layer_name = f"layer_{layer_idx}"
                # Move to CPU immediately
                batch_attention[layer_name] = attn_weights.cpu()

        all_attention_patterns.append(batch_attention)

    # Concatenate all batches on CPU
    print("Concatenating batches...")
    layer_names: list[str] = list(all_attention_patterns[0].keys())
    attention_patterns_concat: dict[str, Tensor] = {
        layer_name: torch.cat([batch[layer_name] for batch in all_attention_patterns], dim=0)
        for layer_name in layer_names
    }

    print(f"Extracted attention patterns for {len(layer_names)} layers")
    return attention_patterns_concat


# Extract attention patterns
attention_patterns: dict[str, Float[Tensor, "total_samples n_heads seq_len seq_len"]] = (
    extract_attention_patterns_multibatch(
        model=model,
        device=device,
        dataloader=dataloader,
        n_batches=config.n_batches,
    )
)

# Print shapes
print("\nAttention pattern shapes:")
for layer_name, attn in attention_patterns.items():
    print(f"  {layer_name}: {attn.shape}")

# %%
# ----------------------- compute attention statistics -----------------------


def compute_attention_stats(
    attention_patterns: dict[str, Float[Tensor, "samples n_heads seq_len seq_len"]],
) -> dict[str, dict[str, Float[np.ndarray, "..."]]]:
    """Compute statistics about attention patterns.

    Args:
        attention_patterns: Dictionary of attention patterns per layer

    Returns:
        Dictionary with statistics per layer including:
        - mean_pattern: Average attention pattern [n_heads, seq_len, seq_len]
        - entropy: Entropy of attention distributions [samples, n_heads, seq_len]
        - max_attention: Maximum attention value [samples, n_heads, seq_len]
        - sparsity: Fraction of attention < 0.01 [samples, n_heads]
    """
    stats: dict[str, dict[str, np.ndarray]] = {}

    for layer_name, attn in attention_patterns.items():
        # Convert to numpy for stats
        attn_np: np.ndarray = attn.numpy()

        # Mean pattern across all samples
        mean_pattern: np.ndarray = attn_np.mean(axis=0)  # [n_heads, seq_len, seq_len]

        # Entropy per query position: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        attn_safe = attn_np + epsilon
        entropy: np.ndarray = -(attn_safe * np.log(attn_safe)).sum(
            axis=-1
        )  # [samples, n_heads, seq_len]

        # Max attention per query position
        max_attention: np.ndarray = attn_np.max(axis=-1)  # [samples, n_heads, seq_len]

        # Sparsity: fraction of attention weights < 0.01
        sparsity: np.ndarray = (attn_np < 0.01).mean(axis=(2, 3))  # [samples, n_heads]

        stats[layer_name] = {
            "mean_pattern": mean_pattern,
            "entropy": entropy,
            "max_attention": max_attention,
            "sparsity": sparsity,
        }

    return stats


attention_stats = compute_attention_stats(attention_patterns)
print("Computed attention statistics")

# %%
# ----------------------- plot: average attention patterns per layer -----------------------


def plot_average_attention_per_layer(
    attention_patterns: dict[str, Float[Tensor, "samples n_heads seq_len seq_len"]],
    max_layers: int | None = None,
) -> None:
    """Plot average attention pattern for each layer (averaged over heads and samples).

    Args:
        attention_patterns: Dictionary of attention patterns per layer
        max_layers: Maximum number of layers to plot (default: all)
    """
    layer_names = sorted(attention_patterns.keys())
    if max_layers is not None:
        layer_names = layer_names[:max_layers]

    n_layers = len(layer_names)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, layer_name in enumerate(layer_names):
        attn = attention_patterns[layer_name].numpy()
        # Average over samples and heads
        avg_attn = attn.mean(axis=(0, 1))  # [seq_len, seq_len]

        ax = axes[idx]
        im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
        ax.set_title(f"{layer_name}\n(avg over samples & heads)")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()


plot_average_attention_per_layer(attention_patterns, max_layers=None)
print("Average attention per layer plots generated.")

# %%
# ----------------------- plot: per-head attention for selected layers -----------------------


def plot_per_head_attention(
    attention_patterns: dict[str, Float[Tensor, "samples n_heads seq_len seq_len"]],
    layer_names: list[str] | None = None,
) -> None:
    """Plot attention pattern for each head in selected layers.

    Args:
        attention_patterns: Dictionary of attention patterns per layer
        layer_names: List of layer names to plot (default: first layer)
    """
    if layer_names is None:
        layer_names = [sorted(attention_patterns.keys())[0]]

    for layer_name in layer_names:
        if layer_name not in attention_patterns:
            print(f"Warning: {layer_name} not found in attention patterns")
            continue

        attn = attention_patterns[layer_name].numpy()
        # Average over samples
        avg_attn = attn.mean(axis=0)  # [n_heads, seq_len, seq_len]
        n_heads = avg_attn.shape[0]

        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            im = ax.imshow(avg_attn[head_idx], cmap="viridis", aspect="auto")
            ax.set_title(f"Head {head_idx}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"{layer_name} - Per-Head Attention Patterns", fontsize=14, y=1.00)
        fig.tight_layout()


# Plot first and last layers
all_layer_names = sorted(attention_patterns.keys())
layers_to_plot = [all_layer_names[0], all_layer_names[-1]]
plot_per_head_attention(attention_patterns, layer_names=layers_to_plot)
print(f"Per-head attention plots generated for layers: {layers_to_plot}")

# %%
# ----------------------- plot: attention entropy across layers -----------------------


def plot_attention_entropy(
    attention_stats: dict[str, dict[str, np.ndarray]],
) -> None:
    """Plot attention entropy statistics across layers.

    Args:
        attention_stats: Dictionary of attention statistics per layer
    """
    layer_names = sorted(attention_stats.keys())

    # Collect mean entropy per layer (averaged over samples, heads, and query positions)
    mean_entropies: list[float] = []
    for layer_name in layer_names:
        entropy = attention_stats[layer_name]["entropy"]  # [samples, n_heads, seq_len]
        mean_entropies.append(float(entropy.mean()))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(layer_names)), mean_entropies, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title("Attention Entropy Across Layers\n(Higher = more uniform attention)")
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


plot_attention_entropy(attention_stats)
print("Attention entropy plot generated.")

# %%
# ----------------------- plot: attention sparsity across layers -----------------------


def plot_attention_sparsity(
    attention_stats: dict[str, dict[str, np.ndarray]],
) -> None:
    """Plot attention sparsity across layers.

    Args:
        attention_stats: Dictionary of attention statistics per layer
    """
    layer_names = sorted(attention_stats.keys())

    # Collect mean sparsity per layer (averaged over samples and heads)
    mean_sparsities: list[float] = []
    for layer_name in layer_names:
        sparsity = attention_stats[layer_name]["sparsity"]  # [samples, n_heads]
        mean_sparsities.append(float(sparsity.mean()))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(layer_names)), mean_sparsities, marker="o", color="C1")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Sparsity (fraction < 0.01)")
    ax.set_title("Attention Sparsity Across Layers\n(Higher = more sparse/focused attention)")
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


plot_attention_sparsity(attention_stats)
print("Attention sparsity plot generated.")

# %%
# ----------------------- plot: attention to first/last tokens -----------------------


def plot_attention_to_special_positions(
    attention_patterns: dict[str, Float[Tensor, "samples n_heads seq_len seq_len"]],
) -> None:
    """Plot how much attention each position pays to first and last tokens.

    Args:
        attention_patterns: Dictionary of attention patterns per layer
    """
    layer_names = sorted(attention_patterns.keys())

    # Collect attention to first and last tokens
    attn_to_first: list[float] = []
    attn_to_last: list[float] = []

    for layer_name in layer_names:
        attn = attention_patterns[layer_name].numpy()
        # Average over samples and heads
        avg_attn = attn.mean(axis=(0, 1))  # [seq_len, seq_len]

        # Average attention to first token (across all query positions)
        attn_to_first.append(float(avg_attn[:, 0].mean()))

        # Average attention to last token (across all query positions)
        attn_to_last.append(float(avg_attn[:, -1].mean()))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(layer_names))
    ax.plot(x, attn_to_first, marker="o", label="Attention to first token")
    ax.plot(x, attn_to_last, marker="s", label="Attention to last token")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention to Special Token Positions Across Layers")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


plot_attention_to_special_positions(attention_patterns)
print("Attention to special positions plot generated.")

# %%
