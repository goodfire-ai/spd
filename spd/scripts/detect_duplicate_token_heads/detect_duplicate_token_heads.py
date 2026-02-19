"""Detect duplicate-token attention heads.

For each head, measures the mean attention weight going to previous positions
that contain the exact same token as the current position, conditioned on
positions where at least one prior duplicate exists.

Usage:
    python -m spd.scripts.detect_duplicate_token_heads.detect_duplicate_token_heads \
        wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.collect_attention_patterns import collect_attention_patterns
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 100
BATCH_SIZE = 32


def _compute_duplicate_token_scores(
    patterns: list[torch.Tensor],
    input_ids: torch.Tensor,
) -> tuple[NDArray[np.floating], int]:
    """Compute mean attention to prior same-token positions per head.

    Only positions where the current token has at least one prior duplicate are
    included. Returns (scores of shape (n_layers, n_heads), n_valid_positions).
    """
    B, T = input_ids.shape
    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]

    # mask[b, i, j] = True iff j < i and input_ids[b, j] == input_ids[b, i]
    same_token = input_ids.unsqueeze(2) == input_ids.unsqueeze(1)  # (B, T, T)
    causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=-1)
    dup_mask = same_token & causal
    has_dup = dup_mask.any(dim=-1)  # (B, T)
    n_valid = has_dup.sum().item()

    scores = np.zeros((n_layers, n_heads))
    if n_valid == 0:
        return scores, 0

    dup_mask_float = dup_mask.unsqueeze(1).float()  # (B, 1, T, T)
    has_dup_float = has_dup.unsqueeze(1).float()  # (B, 1, T)

    for layer_idx, att in enumerate(patterns):
        dup_attn = (att.float() * dup_mask_float).sum(dim=-1)  # (B, H, T)
        valid_sum = (dup_attn * has_dup_float).sum(dim=(0, 2))  # (H,)
        scores[layer_idx] = valid_sum.cpu().numpy() / n_valid

    return scores, n_valid


def _plot_score_heatmap(
    scores: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    out_path: Path,
) -> None:
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(4, n_layers * 1.0)))

    im = ax.imshow(scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean attn to same-token pos")

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
        f"{run_id}  |  Duplicate-token head scores  (n={n_samples} batches)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_duplicate_token_heads(wandb_path: ModelPath, n_batches: int = N_BATCHES) -> None:
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

    accum_scores = np.zeros((n_layers, n_heads))
    total_valid = 0
    n_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            patterns = collect_attention_patterns(target_model, input_ids)
            scores, n_valid = _compute_duplicate_token_scores(patterns, input_ids)

            accum_scores += scores * n_valid
            total_valid += n_valid
            n_processed += 1
            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0 and total_valid > 0
    accum_scores /= total_valid

    logger.info(f"Duplicate-token scores (n={n_processed} batches, {total_valid} valid positions):")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            score = accum_scores[layer_idx, h]
            marker = " <-- dup-token head" if score > 0.3 else ""
            logger.info(f"  L{layer_idx}H{h}: {score:.4f}{marker}")

    _plot_score_heatmap(accum_scores, run_id, n_processed, out_dir / "duplicate_token_scores.png")
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_duplicate_token_heads)
