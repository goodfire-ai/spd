"""Visualize embedding component masks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.models.component_model import ComponentModel


def collect_embedding_masks(model: ComponentModel, device: str) -> Float[Tensor, "vocab C"]:
    """Collect masks for each vocab token.

    Args:
        model: The trained ComponentModel
        device: Device to run computation on

    Returns:
        Tensor of shape (vocab_size, C) containing masks for each vocab token
    """
    assert len(model.components) == 1, "Expected exactly one embedding component"
    component_name = model.target_module_paths[0]

    vocab_size = model.target_model.get_parameter("transformer.wte.weight").shape[0]

    all_masks = torch.zeros((vocab_size, model.C), device=device)

    for token_id in tqdm(range(vocab_size), desc="Collecting masks"):
        # Create single token input
        token_tensor = torch.tensor([[token_id]], device=device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            token_tensor, module_names=[component_name]
        )

        masks, _ = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,
        )

        all_masks[token_id] = masks[component_name].squeeze()

    return all_masks


def permute_to_identity(
    mask: Float[Tensor, "vocab C"],
) -> tuple[Float[Tensor, "vocab C"], Float[Tensor, " vocab"]]:
    """Returns (permuted_mask, permutation_indices)"""
    vocab, C = mask.shape
    new_mask = mask.clone()
    effective_rows = min(vocab, C)
    # Store permutation indices for each instance
    perm_indices = torch.zeros((C), dtype=torch.long, device=mask.device)

    mat: Tensor = mask[:, :]
    perm: list[int] = [0] * C
    used: set[int] = set()
    for i in range(effective_rows):
        sorted_indices: list[int] = torch.argsort(mat[i, :], descending=True).tolist()
        chosen: int = next((col for col in sorted_indices if col not in used), sorted_indices[0])
        perm[i] = chosen
        used.add(chosen)
    remaining: list[int] = sorted(list(set(range(C)) - used))
    for idx, col in enumerate(remaining):
        perm[effective_rows + idx] = col
    new_mask[:, :] = mat[:, perm]
    perm_indices = torch.tensor(perm, device=mask.device)

    return new_mask, perm_indices


def plot_embedding_mask_heatmap(masks: Float[Tensor, "vocab C"], out_dir: Path) -> None:
    """Plot heatmap of embedding masks.

    Args:
        masks: Tensor of shape (vocab_size, C) containing masks
        out_dir: Directory to save the plots
    """
    plt.figure(figsize=(20, 10))
    plt.imshow(
        masks.detach().cpu().numpy(),
        aspect="auto",  # Maintain the data aspect ratio
        cmap="Reds",  # white → red
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="Mask value")

    # Set axis ticks
    plt.xticks(range(0, masks.shape[1], 1000))  # Show every 1000th tick on x-axis
    plt.yticks(range(0, masks.shape[0], 1000))  # Show every 1000th tick on y-axis

    plt.xlabel("Component Index (C)")
    plt.ylabel("Vocab Token ID")
    plt.title("Embedding Component Masks per Token")
    plt.tight_layout()
    fname_embed_masks: Path = out_dir / "embedding_masks.png"
    plt.savefig(fname_embed_masks, dpi=300)
    plt.savefig(fname_embed_masks.with_suffix(".svg"))  # vector graphic for zooming
    logger.info(f"Saved embedding masks to {fname_embed_masks} and .svg")
    plt.close()

    # Also plot a histogram of the first token's mask
    threshold = 0.05
    indices = [0, 99, 199, 299]
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs = axs.flatten()  # pyright: ignore[reportAttributeAccessIssue]
    for token_id, ax in zip(indices, axs, strict=False):
        vals = masks[token_id].detach().cpu().numpy()
        vals = vals[vals > threshold]

        # Ensure all sub-plots have the same ticks and visible range
        ax.set_xticks(np.arange(0.0, 1.05 + 1e-6, 0.05))
        ax.set_xlim(0.0, 1.05)
        ax.hist(vals, bins=100)
        ax.set_ylabel(f"Freq for token {token_id}")

    fig.suptitle(f"Mask Values (> {threshold}) for Each Token")
    fname_hist = out_dir / "first_token_histogram.png"
    fig.savefig(fname_hist, dpi=300)
    fig.savefig(fname_hist.with_suffix(".svg"))  # vector graphic for zooming
    logger.info(f"Saved first token histogram to {fname_hist} and .svg")
    plt.close()

    n_alive_components = ((masks > 0.1).any(dim=0)).sum().item()
    logger.info(f"Number of components that have any value > 0.1: {n_alive_components}")
    ...


def main(model_path: str | Path) -> None:
    """Load model and generate embedding mask visualization.

    Args:
        model_path: Path to the model checkpoint
    """
    # Load model
    model, _config, out_dir = ComponentModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Collect masks
    masks = collect_embedding_masks(model, device)
    permuted_masks, _perm_indices = permute_to_identity(masks)
    plot_embedding_mask_heatmap(permuted_masks, out_dir)


if __name__ == "__main__":
    path = "wandb:spd-gf-lm/runs/..."

    main(path)
