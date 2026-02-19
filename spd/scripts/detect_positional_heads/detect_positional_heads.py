"""Detect positional attention heads that attend to fixed relative offsets.

For each head, builds a histogram of mean attention weight by relative offset
(offset = query_pos - key_pos).  A positional head shows a sharp peak at one
or a few specific offsets regardless of token content.  Also measures attention
to BOS (absolute position 0).

Usage:
    python -m spd.scripts.detect_positional_heads.detect_positional_heads \
        wandb:goodfire/spd/runs/<run_id>
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 100
BATCH_SIZE = 32
MAX_OFFSET = 128


def _collect_attention_patterns(
    model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run forward pass and return attention weights for each layer."""
    B, T = input_ids.shape
    x = model.wte(input_ids)
    patterns: list[torch.Tensor] = []

    for block in model._h:
        attn_input = block.rms_1(x)
        attn = block.attn

        q = attn.q_proj(attn_input).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = (
            attn.k_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v = (
            attn.v_proj(attn_input)
            .view(B, T, attn.n_key_value_heads, attn.head_dim)
            .transpose(1, 2)
        )

        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos = attn.rotary_cos[position_ids].to(q.dtype)
        sin = attn.rotary_sin[position_ids].to(q.dtype)
        q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)

        if attn.repeat_kv_heads > 1:
            k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        patterns.append(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, attn.n_embd)
        y = attn.o_proj(y)
        x = x + y
        x = x + block.mlp(block.rms_2(x))

    return patterns


def _compute_positional_profiles(
    patterns: list[torch.Tensor],
    max_offset: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute per-head mean attention by relative offset, max-offset score, and BOS score.

    Returns:
        profiles: (n_layers, n_heads, max_offset) mean attention at each relative offset
        max_offset_scores: (n_layers, n_heads) max attention at any offset >= 1
        bos_scores: (n_layers, n_heads) mean attention to absolute position 0
    """
    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]
    B, _, T, _ = patterns[0].shape

    profiles = np.zeros((n_layers, n_heads, max_offset))
    bos_scores = np.zeros((n_layers, n_heads))

    for layer_idx, att in enumerate(patterns):
        # BOS: mean attention to position 0 across all query positions
        bos_scores[layer_idx] = att[:, :, :, 0].float().mean(dim=(0, 2)).cpu().numpy()

        # Positional profile: for each offset d, average att[b, h, q, q-d] over valid q
        for d in range(min(max_offset, T)):
            diag = torch.diagonal(att, offset=-d, dim1=-2, dim2=-1)  # (B, H, T-d)
            profiles[layer_idx, :, d] = diag.float().mean(dim=(0, 2)).cpu().numpy()

    max_offset_scores = (
        profiles[:, :, 1:].max(axis=2) if max_offset > 1 else np.zeros((n_layers, n_heads))
    )

    return profiles, max_offset_scores, bos_scores


def _plot_score_heatmap(
    scores: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    out_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(4, n_layers * 1.0)))

    im = ax.imshow(scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=colorbar_label)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            val = scores[layer_idx, h]
            color = "white" if val < scores.max() * 0.6 else "black"
            ax.text(h, layer_idx, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

    fig.suptitle(
        f"{run_id}  |  {title}  (n={n_samples} batches)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_positional_profiles(
    profiles: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    out_path: Path,
    max_display_offset: int = 64,
) -> None:
    """Plot positional profiles as a grid of line charts (one per head)."""
    n_layers, n_heads, n_offsets = profiles.shape
    display_offsets = min(max_display_offset, n_offsets)

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(n_heads * 3, n_layers * 2.5),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            ax.plot(range(display_offsets), profiles[layer_idx, h, :display_offsets], linewidth=1.0)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xlim(0, display_offsets)
            if layer_idx == n_layers - 1:
                ax.set_xlabel("Offset", fontsize=8)
            if h == 0:
                ax.set_ylabel("Mean attn", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{run_id}  |  Positional profiles  (n={n_samples} batches)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_positional_heads(
    wandb_path: ModelPath, n_batches: int = N_BATCHES, max_offset: int = MAX_OFFSET
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = run_info.config
    target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
    target_model.eval()

    for block in target_model._h:
        block.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = target_model.to(device)

    n_layers = len(target_model._h)
    n_heads = target_model._h[0].attn.n_head
    seq_len = target_model.config.n_ctx
    logger.info(f"Model: {n_layers} layers, {n_heads} heads, seq_len={seq_len}")

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=BATCH_SIZE,
        buffer_size=1000,
    )

    accum_profiles = np.zeros((n_layers, n_heads, max_offset))
    accum_max_offset = np.zeros((n_layers, n_heads))
    accum_bos = np.zeros((n_layers, n_heads))
    n_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            patterns = _collect_attention_patterns(target_model, input_ids)
            profiles, max_offset_scores, bos_scores = _compute_positional_profiles(
                patterns, max_offset
            )

            accum_profiles += profiles
            accum_max_offset += max_offset_scores
            accum_bos += bos_scores
            n_processed += 1
            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0
    accum_profiles /= n_processed
    accum_max_offset /= n_processed
    accum_bos /= n_processed

    # Find the peak offset for each head
    peak_offsets = accum_profiles[:, :, 1:].argmax(axis=2) + 1  # offset >= 1

    logger.info(f"Positional head scores (n={n_processed} batches):")
    logger.info("  Max-offset score | Peak offset | BOS score")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            max_val = accum_max_offset[layer_idx, h]
            peak = peak_offsets[layer_idx, h]
            bos_val = accum_bos[layer_idx, h]
            marker = ""
            if max_val > 0.3:
                marker = f" <-- positional head (offset={peak})"
            elif bos_val > 0.1:
                marker = " <-- BOS head"
            logger.info(
                f"  L{layer_idx}H{h}: max={max_val:.4f} (peak@{peak})  bos={bos_val:.4f}{marker}"
            )

    _plot_score_heatmap(
        accum_max_offset,
        run_id,
        n_processed,
        out_dir / "positional_max_offset_scores.png",
        "Max-offset positional scores",
        "Max mean attn at any offset",
    )
    _plot_score_heatmap(
        accum_bos,
        run_id,
        n_processed,
        out_dir / "bos_attention_scores.png",
        "BOS attention scores",
        "Mean attn to position 0",
    )
    _plot_positional_profiles(
        accum_profiles, run_id, n_processed, out_dir / "positional_profiles.png"
    )
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_positional_heads)
