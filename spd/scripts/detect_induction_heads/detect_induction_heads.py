"""Detect induction heads using repeated random token sequences.

For repeated sequences [A B C ... | A B C ...], an induction head at position L+k
in the second half should attend to position k+1 in the first half (the token after
the first occurrence of the current token). The induction score is the mean attention
weight at this offset diagonal.

Usage:
    python -m spd.scripts.detect_induction_heads.detect_induction_heads \
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

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 50
BATCH_SIZE = 32
HALF_SEQ_LEN = 256


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


def _compute_induction_scores(
    patterns: list[torch.Tensor],
    half_len: int,
) -> NDArray[np.floating]:
    """Compute induction score for each layer and head.

    For repeated sequences [t_0..t_{L-1} t_0..t_{L-1}], the induction pattern at
    position L+k attends to position k+1. We average over k in [0, L-2].

    Returns shape (n_layers, n_heads).
    """
    src = torch.arange(1, half_len, device=patterns[0].device)
    dst = torch.arange(half_len, 2 * half_len - 1, device=patterns[0].device)

    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]
    scores = np.zeros((n_layers, n_heads))

    for layer_idx, att in enumerate(patterns):
        # att[:, :, dst, src] zips indices: att[b, h, dst[i], src[i]]
        induction_attn = att[:, :, dst, src]  # (batch, n_heads, half_len-1)
        scores[layer_idx] = induction_attn.float().mean(dim=(0, 2)).cpu().numpy()

    return scores


def _plot_score_heatmap(
    scores: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    out_path: Path,
) -> None:
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.2), max(4, n_layers * 1.0)))

    im = ax.imshow(scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Induction score")

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
        f"{run_id}  |  Induction head scores  (n={n_samples} batches)",
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
    half_len: int,
    out_path: Path,
) -> None:
    """Plot grid of mean attention patterns on repeated random sequences."""
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
            pattern = mean_patterns[layer_idx][h].numpy()
            ax.imshow(pattern, aspect="auto", cmap="viridis", vmin=0)
            ax.axhline(y=half_len - 0.5, color="red", linewidth=0.5, linestyle="--")
            ax.axvline(x=half_len - 0.5, color="red", linewidth=0.5, linestyle="--")
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(
        f"{run_id}  |  Mean attention (repeated random seqs)  (n={n_samples})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_induction_heads(
    wandb_path: ModelPath,
    n_batches: int = N_BATCHES,
    half_seq_len: int = HALF_SEQ_LEN,
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
    vocab_size = target_model.config.vocab_size
    seq_len = half_seq_len * 2
    logger.info(f"Model: {n_layers} layers, {n_heads} heads")
    logger.info(f"Induction test: half_len={half_seq_len}, total_len={seq_len}")

    accum_scores = np.zeros((n_layers, n_heads))
    accum_patterns = [torch.zeros(n_heads, seq_len, seq_len) for _ in range(n_layers)]
    n_processed = 0

    with torch.no_grad():
        for i in range(n_batches):
            first_half = torch.randint(
                100, vocab_size - 100, (BATCH_SIZE, half_seq_len), device=device
            )
            input_ids = torch.cat([first_half, first_half], dim=1)

            patterns = _collect_attention_patterns(target_model, input_ids)
            scores = _compute_induction_scores(patterns, half_seq_len)
            accum_scores += scores

            for layer_idx in range(n_layers):
                accum_patterns[layer_idx] += patterns[layer_idx].float().mean(dim=0).cpu()

            n_processed += 1
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0
    accum_scores /= n_processed
    for layer_idx in range(n_layers):
        accum_patterns[layer_idx] /= n_processed

    logger.info(f"Induction scores (n={n_processed} batches):")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            score = accum_scores[layer_idx, h]
            marker = " <-- induction head" if score > 0.3 else ""
            logger.info(f"  L{layer_idx}H{h}: {score:.4f}{marker}")

    _plot_score_heatmap(accum_scores, run_id, n_processed, out_dir / "induction_scores.png")
    _plot_mean_attention_patterns(
        accum_patterns, run_id, n_processed, half_seq_len, out_dir / "mean_attention_repeated.png"
    )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_induction_heads)
