"""Detect previous-token attention heads by measuring mean attention to position i-1.

For each layer and head, computes the average attention weight from position i to
position i-1 across many data batches. Heads with a high score consistently attend
to the previous token, a key building block of induction circuits.

Usage:
    python -m spd.scripts.detect_prev_token_heads.detect_prev_token_heads \
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


def _collect_attention_patterns(
    model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run forward pass and return attention weights for each layer.

    Returns list of (batch, n_heads, seq_len, seq_len) tensors.
    """
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


def _plot_score_heatmap(
    scores: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    out_path: Path,
) -> None:
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(4, n_layers * 1.0)))

    im = ax.imshow(scores, aspect="auto", cmap="Blues", vmin=0, vmax=0.95)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean attn to pos i-1")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            val = scores[layer_idx, h]
            text_color = "white" if val > 0.65 else "black"
            ax.text(
                h, layer_idx, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color
            )

    fig.suptitle(
        f"{run_id}  |  Previous-token head scores  (n={n_samples} batches)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_mean_attention_patterns(
    mean_patterns: list[torch.Tensor],
    run_id: str,
    n_samples: int,
    out_path: Path,
    max_pos: int = 128,
) -> None:
    """Plot grid of mean attention patterns (one per head, truncated to max_pos)."""
    n_layers = len(mean_patterns)
    n_heads = mean_patterns[0].shape[0]

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(n_heads * 3, n_layers * 3),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            pattern = mean_patterns[layer_idx][h, :max_pos, :max_pos].numpy()
            ax.imshow(pattern, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(
        f"{run_id}  |  Mean attention patterns  (n={n_samples}, pos 0-{max_pos})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_prev_token_heads(wandb_path: ModelPath, n_batches: int = N_BATCHES) -> None:
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
    accum_patterns = [torch.zeros(n_heads, seq_len, seq_len) for _ in range(n_layers)]
    n_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            patterns = _collect_attention_patterns(target_model, input_ids)

            for layer_idx, att in enumerate(patterns):
                diag = torch.diagonal(att, offset=-1, dim1=-2, dim2=-1)  # (batch, heads, T-1)
                accum_scores[layer_idx] += diag.float().mean(dim=(0, 2)).cpu().numpy()
                accum_patterns[layer_idx] += att.float().mean(dim=0).cpu()

            n_processed += 1
            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0
    accum_scores /= n_processed
    for layer_idx in range(n_layers):
        accum_patterns[layer_idx] /= n_processed

    logger.info(f"Previous-token scores (n={n_processed} batches):")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            score = accum_scores[layer_idx, h]
            marker = " <-- prev-token head" if score > 0.3 else ""
            logger.info(f"  L{layer_idx}H{h}: {score:.4f}{marker}")

    _plot_score_heatmap(accum_scores, run_id, n_processed, out_dir / "prev_token_scores.png")
    _plot_mean_attention_patterns(
        accum_patterns, run_id, n_processed, out_dir / "mean_attention_patterns.png"
    )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_prev_token_heads)
