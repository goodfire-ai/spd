"""Plot k-v component input-direction similarity heatmaps from SPD weight decomposition.

For each layer, produces a heatmap of cosine similarity between the V (input-direction)
vectors of k_proj and v_proj components. High similarity means a k component and v component
respond to the same input directions.

  cos_sim(k_c, v_c') = dot(V_k[:, c], V_v[:, c']) / (||V_k[:, c]|| * ||V_v[:, c']||)

Usage:
    python -m spd.scripts.plot_kv_vt_similarity.plot_kv_vt_similarity \
        wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
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
MIN_MEAN_CI = 0.01


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices for a module sorted by mean_ci descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_ci)
        for s in summary.values()
        if s.layer == module_path and s.mean_ci > MIN_MEAN_CI
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _compute_cosine_similarity(
    k_component: LinearComponents,
    v_component: LinearComponents,
    k_alive: list[int],
    v_alive: list[int],
) -> NDArray[np.floating]:
    """Cosine similarity between V columns of k_proj and v_proj components.

    Returns (n_v_alive, n_k_alive) array.
    """
    V_k = k_component.V[:, k_alive].float()  # (d_in, n_k_alive)
    V_v = v_component.V[:, v_alive].float()  # (d_in, n_v_alive)

    V_k_normed = V_k / torch.linalg.norm(V_k, dim=0, keepdim=True).clamp(min=1e-12)
    V_v_normed = V_v / torch.linalg.norm(V_v, dim=0, keepdim=True).clamp(min=1e-12)

    # (n_v_alive, n_k_alive)
    sim = (V_v_normed.T @ V_k_normed).cpu().numpy()
    return sim


def _plot_heatmap(
    data: NDArray[np.floating],
    k_alive: list[int],
    v_alive: list[int],
    layer_idx: int,
    run_id: str,
    out_dir: Path,
) -> None:
    n_v, n_k = data.shape
    fig, ax = plt.subplots(figsize=(max(8, n_k * 0.25), max(6, n_v * 0.25)))

    abs_max = float(np.abs(data).max()) or 1.0
    im = ax.imshow(data, aspect="auto", cmap="PiYG", vmin=-abs_max, vmax=abs_max)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Cosine similarity")

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=7, rotation=90)
    ax.set_xlabel("k_proj component (sorted by mean_ci)")

    ax.set_yticks(range(n_v))
    ax.set_yticklabels([f"C{idx}" for idx in v_alive], fontsize=7)
    ax.set_ylabel("v_proj component (sorted by mean_ci)")

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} — k/v V-direction cosine similarity  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)

    path = out_dir / f"layer{layer_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_kv_vt_similarity(wandb_path: ModelPath) -> None:
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
    n_layers = len(target_model._h)
    logger.info(f"Model: {n_layers} layers")

    with torch.no_grad():
        for layer_idx in range(n_layers):
            k_path = f"h.{layer_idx}.attn.k_proj"
            v_path = f"h.{layer_idx}.attn.v_proj"

            k_alive = _get_alive_indices(summary, k_path)
            v_alive = _get_alive_indices(summary, v_path)
            logger.info(
                f"Layer {layer_idx}: {len(k_alive)} k components, {len(v_alive)} v components"
            )

            if not k_alive or not v_alive:
                logger.info(f"Layer {layer_idx}: skipping (no alive k or v components)")
                continue

            k_component = model.components[k_path]
            v_component = model.components[v_path]
            assert isinstance(k_component, LinearComponents)
            assert isinstance(v_component, LinearComponents)

            sim = _compute_cosine_similarity(k_component, v_component, k_alive, v_alive)
            _plot_heatmap(sim, k_alive, v_alive, layer_idx, run_id, out_dir)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_kv_vt_similarity)
