"""Plot k-v component co-activation heatmaps from harvest co-occurrence data.

For each layer, produces three heatmaps showing how k_proj and v_proj components
co-activate across the dataset:
  - Raw co-occurrence count (how many tokens where both fired)
  - Phi coefficient (correlation of binary firing indicators)
  - Jaccard similarity (intersection over union of firing sets)

All metrics are derived from the pre-computed CorrelationStorage in the harvest data.

Usage:
    python -m spd.scripts.plot_kv_coactivation.plot_kv_coactivation \
        wandb:goodfire/spd/runs/<run_id>
"""

import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.harvest.storage import CorrelationStorage
from spd.log import logger
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.01


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices for a module sorted by CI descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > MIN_MEAN_CI
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _correlation_indices(
    corr: CorrelationStorage, module_path: str, alive_indices: list[int]
) -> list[int]:
    """Map (module_path, component_idx) pairs to indices in CorrelationStorage."""
    return [corr.key_to_idx[f"{module_path}:{idx}"] for idx in alive_indices]


def _compute_raw_cooccurrence(
    count_ij: torch.Tensor, k_corr_idx: list[int], v_corr_idx: list[int]
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    sub = count_ij[v_idx[:, None], k_idx[None, :]].float()
    return sub.numpy()


def _compute_phi_coefficient(
    count_ij: torch.Tensor,
    count_i: torch.Tensor,
    count_total: int,
    k_corr_idx: list[int],
    v_corr_idx: list[int],
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    a = count_ij[v_idx[:, None], k_idx[None, :]].float()
    n_k = count_i[k_idx].float()  # (n_k_alive,)
    n_v = count_i[v_idx].float()  # (n_v_alive,)
    n = float(count_total)

    numerator = n * a - n_v[:, None] * n_k[None, :]
    denominator = torch.sqrt(n_v[:, None] * (n - n_v[:, None]) * n_k[None, :] * (n - n_k[None, :]))
    phi = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(a))
    return phi.numpy()


def _compute_jaccard(
    count_ij: torch.Tensor,
    count_i: torch.Tensor,
    k_corr_idx: list[int],
    v_corr_idx: list[int],
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    intersection = count_ij[v_idx[:, None], k_idx[None, :]].float()
    union = count_i[v_idx].float()[:, None] + count_i[k_idx].float()[None, :] - intersection
    jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return jaccard.numpy()


def _plot_heatmap(
    data: NDArray[np.floating],
    k_alive: list[int],
    v_alive: list[int],
    layer_idx: int,
    run_id: str,
    metric_name: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    out_dir: Path,
) -> None:
    n_v, n_k = data.shape
    fig, ax = plt.subplots(figsize=(max(8, n_k * 0.25), max(6, n_v * 0.25)))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=metric_name)

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=7, rotation=90)
    ax.set_xlabel("k_proj component (sorted by CI)")

    ax.set_yticks(range(n_v))
    ax.set_yticklabels([f"C{idx}" for idx in v_alive], fontsize=7)
    ax.set_ylabel("v_proj component (sorted by CI)")

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} — k/v {metric_name}  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)

    path = out_dir / f"layer{layer_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _get_n_layers(summary: dict[str, ComponentSummary]) -> int:
    """Infer number of layers from summary keys like 'h.0.attn.k_proj'."""
    layer_indices = {
        int(m.group(1)) for s in summary.values() if (m := re.match(r"h\.(\d+)\.", s.layer))
    }
    assert layer_indices, "No layer indices found in summary"
    return max(layer_indices) + 1


def plot_kv_coactivation(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))

    out_base = SCRIPT_DIR / "out" / run_id
    raw_dir = out_base / "ci_cooccurrence"
    phi_dir = out_base / "phi_coefficient"
    jaccard_dir = out_base / "jaccard"
    for d in (raw_dir, phi_dir, jaccard_dir):
        d.mkdir(parents=True, exist_ok=True)

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    corr = repo.get_correlations()
    assert corr is not None, f"No correlation data found for {run_id}"
    logger.info(
        f"Loaded correlations: {len(corr.component_keys)} components, {corr.count_total} tokens"
    )

    n_layers = _get_n_layers(summary)
    for layer_idx in range(n_layers):
        k_path = f"h.{layer_idx}.attn.k_proj"
        v_path = f"h.{layer_idx}.attn.v_proj"

        k_alive = _get_alive_indices(summary, k_path)
        v_alive = _get_alive_indices(summary, v_path)
        logger.info(f"Layer {layer_idx}: {len(k_alive)} k components, {len(v_alive)} v components")

        if not k_alive or not v_alive:
            logger.info(f"Layer {layer_idx}: skipping (no alive k or v components)")
            continue

        k_corr_idx = _correlation_indices(corr, k_path, k_alive)
        v_corr_idx = _correlation_indices(corr, v_path, v_alive)

        # CI co-occurrence
        raw = _compute_raw_cooccurrence(corr.count_ij, k_corr_idx, v_corr_idx)
        _plot_heatmap(
            raw,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "CI co-occurrence",
            "Purples",
            0,
            None,
            raw_dir,
        )

        # Phi coefficient
        phi = _compute_phi_coefficient(
            corr.count_ij, corr.count_i, corr.count_total, k_corr_idx, v_corr_idx
        )
        phi_abs_max = float(np.abs(phi).max()) or 1.0
        _plot_heatmap(
            phi,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "phi coefficient",
            "RdBu_r",
            -phi_abs_max,
            phi_abs_max,
            phi_dir,
        )

        # Jaccard
        jacc = _compute_jaccard(corr.count_ij, corr.count_i, k_corr_idx, v_corr_idx)
        _plot_heatmap(
            jacc,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "Jaccard similarity",
            "Purples",
            0,
            None,
            jaccard_dir,
        )

    logger.info(f"All plots saved to {out_base}")


if __name__ == "__main__":
    fire.Fire(plot_kv_coactivation)
