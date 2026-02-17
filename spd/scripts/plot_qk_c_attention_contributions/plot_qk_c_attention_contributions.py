"""Plot weight-only attention contribution heatmaps between q and k subcomponents.

For each layer and relative position offset, produces a single grid containing:
  - First cell: summed (all heads) q·k attention contributions
  - Remaining cells: one per query head

Uses V-norm-scaled U dot products with RoPE applied at specified relative position offsets.

Usage:
    python -m spd.scripts.plot_qk_c_attention_contributions.plot_qk_c_attention_contributions \
        wandb:goodfire/spd/runs/<run_id>
"""

import math
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
from spd.scripts.rope_aware_qk import compute_qk_rope_coefficients, evaluate_qk_at_offsets
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.01


def _get_alive_indices(
    summary: dict[str, ComponentSummary], module_path: str, min_mean_ci: float
) -> list[int]:
    """Return component indices for a module sorted by mean_ci descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_ci)
        for s in summary.values()
        if s.layer == module_path and s.mean_ci > min_mean_ci
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _compute_per_head_attention_contributions(
    q_component: LinearComponents,
    k_component: LinearComponents,
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    offsets: tuple[int, ...],
) -> NDArray[np.floating]:
    """Compute (n_offsets, n_q_heads, n_q_alive, n_k_alive) per-head attention contributions.

    Scales U vectors by their corresponding V norms before the dot product, so that the
    result accounts for the unnormalized magnitude split between U and V.
    """
    V_q_norms = torch.linalg.norm(q_component.V[:, q_alive], dim=0).float()  # (n_q_alive,)
    V_k_norms = torch.linalg.norm(k_component.V[:, k_alive], dim=0).float()  # (n_k_alive,)

    U_q = q_component.U[q_alive].float() * V_q_norms[:, None]  # (n_q_alive, n_q_heads * head_dim)
    U_q = U_q.reshape(len(q_alive), n_q_heads, head_dim)

    U_k = k_component.U[k_alive].float() * V_k_norms[:, None]  # (n_k_alive, n_kv_heads * head_dim)
    U_k = U_k.reshape(len(k_alive), n_kv_heads, head_dim)

    g = n_q_heads // n_kv_heads
    U_k_expanded = U_k.repeat_interleave(g, dim=1)  # (n_k_alive, n_q_heads, head_dim)

    head_results = []
    for h in range(n_q_heads):
        A, B = compute_qk_rope_coefficients(U_q[:, h, :], U_k_expanded[:, h, :])
        W_h = evaluate_qk_at_offsets(A, B, rotary_cos, rotary_sin, offsets, head_dim)
        head_results.append(W_h)  # (n_offsets, n_q, n_k)

    # (n_heads, n_offsets, n_q, n_k) -> (n_offsets, n_heads, n_q, n_k)
    return torch.stack(head_results).permute(1, 0, 2, 3).cpu().numpy()


def _plot_heatmaps(
    W_per_head: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offset: int,
    vmax: float,
) -> None:
    n_cells = n_q_heads + 1  # summed + per-head
    n_cols = math.ceil(math.sqrt(n_cells))
    n_rows = math.ceil(n_cells / n_cols)
    n_q, n_k = W_per_head.shape[1], W_per_head.shape[2]

    W_summed = W_per_head.sum(axis=0)

    cell_w = max(4, n_k * 0.15)
    cell_h = max(3, n_q * 0.15)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False
    )

    # All cells: summed first, then per-head
    all_data = [W_summed] + [W_per_head[h] for h in range(n_q_heads)]
    titles = ["Sum"] + [f"H{h}" for h in range(n_q_heads)]

    for cell, (data, title) in enumerate(zip(all_data, titles, strict=True)):
        row, col = divmod(cell, n_cols)
        ax = axes[row, col]
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold" if cell == 0 else "normal")
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f"Q C{idx}" for idx in q_alive], fontsize=5)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f"K C{idx}" for idx in k_alive], fontsize=5, rotation=90)

    # Hide unused cells
    for i in range(n_cells, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 q\u00b7k attention contributions"
        f" (\u0394={offset})  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    path = out_dir / f"layer{layer_idx}_qk_attention_contributions_offset{offset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_diff_heatmaps(
    D_per_head: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_q_heads: int,
    layer_idx: int,
    run_id: str,
    out_dir: Path,
    offset: int,
    vmax: float,
) -> None:
    """Plot W(Δ=offset) - W(Δ=0) heatmaps."""
    n_cells = n_q_heads + 1
    n_cols = math.ceil(math.sqrt(n_cells))
    n_rows = math.ceil(n_cells / n_cols)
    n_q, n_k = D_per_head.shape[1], D_per_head.shape[2]

    D_summed = D_per_head.sum(axis=0)

    cell_w = max(4, n_k * 0.15)
    cell_h = max(3, n_q * 0.15)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False
    )

    all_data = [D_summed] + [D_per_head[h] for h in range(n_q_heads)]
    titles = ["Sum"] + [f"H{h}" for h in range(n_q_heads)]

    for cell, (data, title) in enumerate(zip(all_data, titles, strict=True)):
        row, col = divmod(cell, n_cols)
        ax = axes[row, col]
        im = ax.imshow(data, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold" if cell == 0 else "normal")
        ax.set_yticks(range(n_q))
        ax.set_yticklabels([f"Q C{idx}" for idx in q_alive], fontsize=5)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f"K C{idx}" for idx in k_alive], fontsize=5, rotation=90)

    for i in range(n_cells, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} \u2014 q\u00b7k attention diff"
        f" (\u0394={offset} \u2212 \u0394=0)  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    path = out_dir / f"layer{layer_idx}_qk_attention_diff_offset{offset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_qk_c_attention_contributions(
    wandb_path: ModelPath,
    offsets: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 16, 64),
) -> None:
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
    assert not blocks[0].attn.rotary_adjacent_pairs, "RoPE math assumes non-adjacent pairs layout"
    head_dim = blocks[0].attn.head_dim
    n_q_heads = blocks[0].attn.n_head
    n_kv_heads = blocks[0].attn.n_key_value_heads
    n_layers = len(blocks)
    logger.info(
        f"Model: {n_layers} layers, head_dim={head_dim}, "
        f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}"
    )

    with torch.no_grad():
        for layer_idx in range(n_layers):
            q_path = f"h.{layer_idx}.attn.q_proj"
            k_path = f"h.{layer_idx}.attn.k_proj"

            q_component = model.components[q_path]
            k_component = model.components[k_path]
            assert isinstance(q_component, LinearComponents)
            assert isinstance(k_component, LinearComponents)

            q_alive = _get_alive_indices(summary, q_path, MIN_MEAN_CI)
            k_alive = _get_alive_indices(summary, k_path, MIN_MEAN_CI)
            logger.info(
                f"Layer {layer_idx}: {len(q_alive)} q components, {len(k_alive)} k components"
            )

            if not q_alive or not k_alive:
                logger.info(f"Layer {layer_idx}: skipping (no alive q or k components)")
                continue

            rotary_cos = blocks[layer_idx].attn.rotary_cos
            rotary_sin = blocks[layer_idx].attn.rotary_sin
            assert isinstance(rotary_cos, torch.Tensor)
            assert isinstance(rotary_sin, torch.Tensor)

            W = _compute_per_head_attention_contributions(
                q_component,
                k_component,
                q_alive,
                k_alive,
                n_q_heads,
                n_kv_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                offsets,
            )

            # Compute vmax across all offsets for consistent color scale
            W_summed_all = W.sum(axis=1)  # (n_offsets, n_q, n_k)
            vmax = float(max(np.abs(W_summed_all).max(), np.abs(W).max())) or 1.0

            for offset_idx, offset in enumerate(offsets):
                _plot_heatmaps(
                    W[offset_idx],
                    q_alive,
                    k_alive,
                    n_q_heads,
                    layer_idx,
                    run_id,
                    out_dir,
                    offset,
                    vmax,
                )

            # Diff plots: W(Δ=k) - W(Δ=0) for each non-zero offset
            assert offsets[0] == 0, "First offset must be 0 for diff computation"
            W_base = W[0]  # (n_heads, n_q, n_k) at Δ=0
            non_zero_offsets = [(idx, o) for idx, o in enumerate(offsets) if o != 0]
            if non_zero_offsets:
                diffs = np.stack([W[idx] - W_base for idx, _ in non_zero_offsets])
                D_summed_all = diffs.sum(axis=1)
                diff_vmax = float(max(np.abs(D_summed_all).max(), np.abs(diffs).max())) or 1.0

                diff_dir = out_dir / "diffs"
                diff_dir.mkdir(parents=True, exist_ok=True)
                for i, (_, offset) in enumerate(non_zero_offsets):
                    _plot_diff_heatmaps(
                        diffs[i],
                        q_alive,
                        k_alive,
                        n_q_heads,
                        layer_idx,
                        run_id,
                        diff_dir,
                        offset,
                        diff_vmax,
                    )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_qk_c_attention_contributions)
