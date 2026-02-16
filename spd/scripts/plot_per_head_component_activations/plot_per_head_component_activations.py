"""Plot per-head component activation heatmaps from an SPD run.

For each layer and attention projection (q/k/v_proj), produces a heatmap where:
  - y-axis: alive subcomponents (sorted by mean_ci descending, mean_ci > threshold)
  - x-axis: attention head index
  - color: mean |component activation| * ||u_c[head_h]|| / ||u_c||

Component activations are stored by harvest as (v_c^T @ x) * ||u_c||. Since v_c is
shared across all heads for q/k/v_proj, the per-head contribution is exact:
  |v_c^T @ x| * ||u_c[head_h]|| = |stored_act| * ||u_c[head_h]|| / ||u_c||

o_proj is excluded because its head structure is in V (input), not U (output),
so the stored scalar activation can't be cleanly decomposed per head.

Usage:
    python -m spd.scripts.plot_per_head_component_activations.plot_per_head_component_activations wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData, ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
PROJ_NAMES = ("q_proj", "k_proj", "v_proj")
MIN_MEAN_CI = 0.01


def _get_alive_components(
    summary: dict[str, ComponentSummary], module_path: str, min_mean_ci: float
) -> list[tuple[str, int]]:
    """Return (component_key, component_idx) pairs sorted by mean_ci descending."""
    components = [
        (key, s.component_idx, s.mean_ci)
        for key, s in summary.items()
        if s.layer == module_path and s.mean_ci > min_mean_ci
    ]
    components.sort(key=lambda t: t[2], reverse=True)
    return [(key, idx) for key, idx, _ in components]


def _mean_abs_act_on_ci_data(comp_data: ComponentData, ci_threshold: float) -> float:
    """Compute mean absolute component activation on CI-important positions."""
    acts: list[float] = []
    for example in comp_data.activation_examples:
        for ci, stored_act in zip(example.ci_values, example.component_acts, strict=True):
            if ci > ci_threshold:
                acts.append(abs(stored_act))
    if not acts:
        return 0.0
    return sum(acts) / len(acts)


def _per_head_activations(
    component: LinearComponents,
    alive: list[tuple[str, int]],
    comp_data_map: dict[str, ComponentData],
    ci_threshold: float,
    head_dim: int,
    n_heads: int,
) -> NDArray[np.floating]:
    """Compute (n_alive, n_heads) array: mean |activation| * ||u_c[head_h]|| / ||u_c||."""
    result = np.zeros((len(alive), n_heads), dtype=np.float32)
    for row, (key, c_idx) in enumerate(alive):
        comp_data = comp_data_map.get(key)
        if comp_data is None:
            continue
        mean_act = _mean_abs_act_on_ci_data(comp_data, ci_threshold)
        u_c = component.U[c_idx].float()
        u_norm = torch.linalg.norm(u_c).item()
        if u_norm == 0:
            continue
        for h in range(n_heads):
            u_head = u_c[h * head_dim : (h + 1) * head_dim]
            u_head_norm = torch.linalg.norm(u_head).item()
            result[row, h] = mean_act * u_head_norm / u_norm
    return result


def _plot_heatmap(
    norms: NDArray[np.floating],
    alive_indices: list[int],
    n_heads: int,
    layer_idx: int,
    proj_name: str,
    run_id: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.8), max(6, len(alive_indices) * 0.25)))

    im = ax.imshow(norms, aspect="auto", cmap="Reds", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Per-head component activation")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_xlabel("Head")

    ax.set_yticks(range(len(alive_indices)))
    ax.set_yticklabels([f"C{idx}" for idx in alive_indices], fontsize=7)
    ax.set_ylabel("Component (sorted by mean_ci)")

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} — {proj_name}\nPer-head component activation on CI-important data",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.08)

    path = out_dir / f"layer{layer_idx}_{proj_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_per_head_component_activations(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()
    ci_threshold = repo.get_ci_threshold()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    head_dim = blocks[0].attn.head_dim
    n_layers = len(blocks)
    logger.info(f"Model: {n_layers} layers, head_dim={head_dim}")

    with torch.no_grad():
        for layer_idx in range(n_layers):
            for proj_name in PROJ_NAMES:
                module_path = f"h.{layer_idx}.attn.{proj_name}"

                component = model.components[module_path]
                assert isinstance(component, LinearComponents)

                alive = _get_alive_components(summary, module_path, MIN_MEAN_CI)
                logger.info(f"{module_path}: {len(alive)} components with mean_ci > {MIN_MEAN_CI}")

                if not alive:
                    continue

                keys = [key for key, _ in alive]
                comp_data_map = repo.get_components_bulk(keys)
                indices = [idx for _, idx in alive]

                n_heads = component.U.shape[1] // head_dim
                norms = _per_head_activations(
                    component, alive, comp_data_map, ci_threshold, head_dim, n_heads
                )

                _plot_heatmap(norms, indices, n_heads, layer_idx, proj_name, run_id, out_dir)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_per_head_component_activations)
