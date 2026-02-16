"""Plot target vs reconstructed attention weight matrices from an SPD run.

For each layer and attention projection (q/k/v/o_proj), produces multiple 4x4 grids.
Each grid shows the target weight, reconstructed (UV^T), and 14 subcomponent weights.
Successive grids page through all alive components (ranked by mean_ci descending).
All grids for a given layer/projection share the same color scale.

Alive components are determined from harvest data (mean_ci > 0).

Usage:
    python -m spd.scripts.plot_attention_weights.plot_attention_weights wandb:goodfire/spd/runs/<run_id>
"""

import math
from datetime import datetime
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from numpy.typing import NDArray
from torch import Tensor

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
GRID_SIZE = 4
COMPONENTS_PER_PAGE = GRID_SIZE * GRID_SIZE - 2  # 14 (2 cells for target + recon)


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices for a module sorted by mean_ci descending, filtered to alive."""
    components = [
        (s.component_idx, s.mean_ci)
        for s in summary.values()
        if s.layer == module_path and s.mean_ci > 0
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _add_head_lines(ax: Axes, proj_name: str, head_dim: int, shape: tuple[int, int]) -> None:
    n_rows, n_cols = shape

    if proj_name in ("q_proj", "k_proj", "v_proj"):
        n_heads = n_rows // head_dim
        for i in range(1, n_heads):
            ax.axhline(
                y=i * head_dim - 0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.6
            )
    else:
        n_heads = n_cols // head_dim
        for i in range(1, n_heads):
            ax.axvline(
                x=i * head_dim - 0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.6
            )


def _style_ax(
    fig: plt.Figure,
    ax: Axes,
    data: NDArray[np.floating],
    vmax: float,
    title: str,
    proj_name: str,
    head_dim: int,
) -> None:
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])
    ax.set_xticks([])
    _add_head_lines(ax, proj_name, head_dim, data.shape)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


def _component_weight(component: LinearComponents, idx: int) -> Tensor:
    """Weight matrix for a single subcomponent: outer(U[c], V[:, c]) -> (d_out, d_in)."""
    return torch.outer(component.U[idx], component.V[:, idx])


def _plot_grids(
    target_weight: Tensor,
    recon_weight: Tensor,
    component: LinearComponents,
    alive_indices: list[int],
    layer_idx: int,
    proj_name: str,
    head_dim: int,
    run_id: str,
    timestamp: str,
    out_dir: Path,
) -> None:
    target_np: NDArray[np.floating] = target_weight.float().cpu().numpy()
    recon_np: NDArray[np.floating] = recon_weight.float().cpu().numpy()

    all_sub_weights = [_component_weight(component, i).float().cpu().numpy() for i in alive_indices]

    vmax = float(max(abs(target_np).max(), abs(recon_np).max()))
    if all_sub_weights:
        vmax = max(vmax, float(max(abs(w).max() for w in all_sub_weights)))

    n_pages = max(1, math.ceil(len(alive_indices) / COMPONENTS_PER_PAGE))

    for page in range(n_pages):
        start = page * COMPONENTS_PER_PAGE
        page_indices = alive_indices[start : start + COMPONENTS_PER_PAGE]
        page_weights = all_sub_weights[start : start + COMPONENTS_PER_PAGE]

        fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(24, 22))

        _style_ax(fig, axes[0, 0], target_np, vmax, "Target weight", proj_name, head_dim)
        _style_ax(fig, axes[0, 1], recon_np, vmax, "Reconstructed (UV\u1d40)", proj_name, head_dim)

        cell = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if r == 0 and c < 2:
                    continue
                ax = axes[r, c]
                if cell < len(page_indices):
                    c_idx = page_indices[cell]
                    _style_ax(fig, ax, page_weights[cell], vmax, f"C{c_idx}", proj_name, head_dim)
                else:
                    ax.set_visible(False)
                cell += 1

        page_label = f"({page + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            f"{run_id}  |  Layer {layer_idx} — {proj_name}  {page_label}\n{timestamp}",
            fontsize=14,
            fontweight="bold",
        )
        fig.subplots_adjust(hspace=0.25, wspace=0.15)

        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        path = out_dir / f"layer{layer_idx}_{proj_name}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_attention_weights(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    head_dim = blocks[0].attn.head_dim
    n_layers = len(blocks)
    logger.info(f"Model: {n_layers} layers, head_dim={head_dim}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with torch.no_grad():
        for layer_idx in range(n_layers):
            attn = blocks[layer_idx].attn
            for proj_name in PROJ_NAMES:
                module_path = f"h.{layer_idx}.attn.{proj_name}"

                target_weight = getattr(attn, proj_name).weight
                component = model.components[module_path]
                assert isinstance(component, LinearComponents)
                recon_weight = component.weight

                alive_indices = _get_alive_indices(summary, module_path)
                logger.info(f"{module_path}: {len(alive_indices)} alive components")

                _plot_grids(
                    target_weight,
                    recon_weight,
                    component,
                    alive_indices,
                    layer_idx,
                    proj_name,
                    head_dim,
                    run_id,
                    timestamp,
                    out_dir,
                )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_attention_weights)
