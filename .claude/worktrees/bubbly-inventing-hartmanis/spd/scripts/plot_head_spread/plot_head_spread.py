"""Plot head-spread entropy histograms for subcomponents of an SPD run.

For each layer, produces a figure with 4 subplots (q/k/v/o_proj). Each subplot is a bar chart:
  - x-axis: subcomponent index (mean_ci > threshold)
  - y-axis: head-spread entropy H = -sum(p_i * ln(p_i)), where p_i = norm_i / sum(norms)

Higher entropy means the subcomponent's weight is spread across many heads;
lower entropy means it is concentrated in one or few heads.

Usage:
    python -m spd.scripts.plot_head_spread.plot_head_spread wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
MIN_MEAN_CI = 0.01


def _get_alive_indices(
    summary: dict[str, ComponentSummary], module_path: str, min_mean_ci: float
) -> list[int]:
    """Return component indices for a module sorted by CI descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > min_mean_ci
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _head_norms(
    component: LinearComponents,
    alive_indices: list[int],
    proj_name: str,
    head_dim: int,
    n_heads: int,
) -> NDArray[np.floating]:
    """Compute (n_alive, n_heads) array of Frobenius norms per subcomponent per head."""
    norms = np.zeros((len(alive_indices), n_heads), dtype=np.float32)
    for row, c_idx in enumerate(alive_indices):
        w = torch.outer(component.U[c_idx], component.V[:, c_idx]).float()
        for h in range(n_heads):
            if proj_name in ("q_proj", "k_proj", "v_proj"):
                head_slice = w[h * head_dim : (h + 1) * head_dim, :]
            else:
                head_slice = w[:, h * head_dim : (h + 1) * head_dim]
            norms[row, h] = torch.linalg.norm(head_slice).item()
    return norms


def _total_norms(component: LinearComponents, alive_indices: list[int]) -> NDArray[np.floating]:
    """Frobenius norm of each subcomponent's full weight matrix."""
    result = np.zeros(len(alive_indices), dtype=np.float32)
    for row, c_idx in enumerate(alive_indices):
        u_c = component.U[c_idx]
        v_c = component.V[:, c_idx]
        # ||u_c v_c^T||_F = ||u_c|| * ||v_c||
        result[row] = (torch.linalg.norm(u_c) * torch.linalg.norm(v_c)).item()
    return result


def _entropy(norms: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute head-spread entropy for each subcomponent from its per-head norms.

    For each row: p_i = norm_i / sum(norms), H = -sum(p_i * ln(p_i)).
    """
    totals = norms.sum(axis=1, keepdims=True)
    # Avoid division by zero for components with all-zero norms
    totals = np.maximum(totals, 1e-12)
    p = norms / totals
    # Use 0 * log(0) = 0 convention
    log_p = np.where(p > 0, np.log(p), 0.0)
    return -(p * log_p).sum(axis=1)


def plot_head_spread(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    head_dim = blocks[0].attn.head_dim
    n_layers = len(blocks)
    logger.info(f"Model: {n_layers} layers, head_dim={head_dim}")

    cmap = plt.get_cmap("Purples")

    with torch.no_grad():
        for layer_idx in range(n_layers):
            fig, axes = plt.subplots(4, 1, figsize=(14, 16))

            for ax_idx, proj_name in enumerate(PROJ_NAMES):
                ax = axes[ax_idx]
                module_path = f"h.{layer_idx}.attn.{proj_name}"

                component = model.components[module_path]
                assert isinstance(component, LinearComponents)

                alive_indices = _get_alive_indices(summary, module_path, MIN_MEAN_CI)
                logger.info(
                    f"{module_path}: {len(alive_indices)} components with mean_ci > {MIN_MEAN_CI}"
                )

                if len(alive_indices) == 0:
                    ax.set_visible(False)
                    continue

                if proj_name in ("q_proj", "k_proj", "v_proj"):
                    n_heads = component.U.shape[1] // head_dim
                else:
                    n_heads = component.V.shape[0] // head_dim
                max_entropy = np.log(n_heads)

                norms = _head_norms(component, alive_indices, proj_name, head_dim, n_heads)
                entropies = _entropy(norms)
                total = _total_norms(component, alive_indices)

                norm_obj = mpl_colors.Normalize(vmin=0, vmax=total.max())
                colors = cmap(norm_obj(total))

                x = np.arange(len(alive_indices))
                ax.bar(x, entropies, width=1.0, color=colors, edgecolor="none")
                ax.axhline(y=max_entropy, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_ylabel("Entropy (Param norm spread across heads)")
                ax.set_title(proj_name, fontsize=11)
                ax.set_ylim(0, max_entropy * 1.05)

                ax.set_xticks(x)
                ax.set_xticklabels([f"C{idx}" for idx in alive_indices], fontsize=5, rotation=90)

                sm = mpl_cm.ScalarMappable(cmap=cmap, norm=norm_obj)
                fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01, label="Total norm")

            fig.suptitle(
                f"{run_id}  |  Layer {layer_idx} — Head-spread entropy",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            path = out_dir / f"layer{layer_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {path}")

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_head_spread)
