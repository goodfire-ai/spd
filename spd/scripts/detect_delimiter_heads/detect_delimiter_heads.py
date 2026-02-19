"""Detect delimiter-attending attention heads.

For each head, measures what fraction of attention weight lands on structural
delimiter tokens (periods, commas, semicolons, etc.) and compares to the
baseline delimiter frequency.  Heads with a high ratio over baseline
disproportionately target structural markers.

Usage:
    python -m spd.scripts.detect_delimiter_heads.detect_delimiter_heads \
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
from transformers import AutoTokenizer

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

DELIMITER_CHARS = [".", ",", ";", ":", "!", "?", "\n"]
DELIMITER_MULTI = [".\n", ".\n\n", ",\n", ";\n"]


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


def _get_delimiter_token_ids(tokenizer: AutoTokenizer) -> set[int]:
    """Collect token IDs for delimiter characters and common multi-char delimiters."""
    delimiter_ids: set[int] = set()
    for char in DELIMITER_CHARS:
        token_ids = tokenizer.encode(char)
        if len(token_ids) == 1:
            delimiter_ids.add(token_ids[0])
    for multi in DELIMITER_MULTI:
        token_ids = tokenizer.encode(multi)
        if len(token_ids) == 1:
            delimiter_ids.add(token_ids[0])
    return delimiter_ids


def _compute_delimiter_scores(
    patterns: list[torch.Tensor],
    input_ids: torch.Tensor,
    delimiter_ids: set[int],
) -> tuple[NDArray[np.floating], float]:
    """Compute mean fraction of attention on delimiter tokens per head.

    Returns (raw_scores of shape (n_layers, n_heads), baseline_fraction).
    """
    B, T = input_ids.shape
    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]

    delim_set = torch.tensor(sorted(delimiter_ids), device=input_ids.device)
    is_delim = (input_ids.unsqueeze(-1) == delim_set.unsqueeze(0).unsqueeze(0)).any(dim=-1)
    baseline_fraction = is_delim.float().mean().item()

    # (B, 1, 1, T) for broadcasting with attention (B, H, T_q, T_k)
    is_delim_key = is_delim.unsqueeze(1).unsqueeze(2).float()

    raw_scores = np.zeros((n_layers, n_heads))
    for layer_idx, att in enumerate(patterns):
        attn_to_delim = (att.float() * is_delim_key).sum(dim=-1)  # (B, H, T)
        raw_scores[layer_idx] = attn_to_delim.mean(dim=(0, 2)).cpu().numpy()

    return raw_scores, baseline_fraction


def _plot_dual_heatmap(
    raw_scores: NDArray[np.floating],
    ratio_scores: NDArray[np.floating],
    run_id: str,
    n_samples: int,
    baseline_frac: float,
    out_path: Path,
) -> None:
    n_layers, n_heads = raw_scores.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_heads * 2.4), max(4, n_layers * 1.0)))

    im1 = ax1.imshow(raw_scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02, label="Attn fraction on delimiters")
    ax1.set_title("Attention to delimiters", fontsize=11, fontweight="bold")

    im2 = ax2.imshow(ratio_scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(
        im2, ax=ax2, shrink=0.8, pad=0.02, label=f"Ratio over baseline ({baseline_frac:.3f})"
    )
    ax2.set_title("Ratio over baseline", fontsize=11, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

    data_sets = [raw_scores, ratio_scores]
    for ax, data in zip([ax1, ax2], data_sets, strict=True):
        for layer_idx in range(n_layers):
            for h in range(n_heads):
                val = data[layer_idx, h]
                threshold = abs(data).max() * 0.6
                color = "white" if abs(val) < threshold else "black"
                ax.text(
                    h, layer_idx, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color
                )

    fig.suptitle(
        f"{run_id}  |  Delimiter head scores  (n={n_samples} batches, baseline={baseline_frac:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_delimiter_heads(wandb_path: ModelPath, n_batches: int = N_BATCHES) -> None:
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

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    delimiter_ids = _get_delimiter_token_ids(tokenizer)
    logger.info(f"Found {len(delimiter_ids)} delimiter token IDs: {sorted(delimiter_ids)}")

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

    accum_raw = np.zeros((n_layers, n_heads))
    accum_baseline = 0.0
    n_processed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            input_ids = batch[task_config.column_name][:, :seq_len].to(device)
            patterns = _collect_attention_patterns(target_model, input_ids)
            raw_scores, baseline_fraction = _compute_delimiter_scores(
                patterns, input_ids, delimiter_ids
            )

            accum_raw += raw_scores
            accum_baseline += baseline_fraction
            n_processed += 1
            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    assert n_processed > 0
    accum_raw /= n_processed
    mean_baseline = accum_baseline / n_processed
    ratio_scores = accum_raw / max(mean_baseline, 1e-8)

    logger.info(f"Delimiter head scores (n={n_processed} batches, baseline={mean_baseline:.4f}):")
    logger.info("  Raw attn | Ratio over baseline")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            raw_val = accum_raw[layer_idx, h]
            ratio_val = ratio_scores[layer_idx, h]
            marker = " <-- delimiter head" if ratio_val > 2.0 else ""
            logger.info(f"  L{layer_idx}H{h}: raw={raw_val:.4f}  ratio={ratio_val:.2f}{marker}")

    _plot_dual_heatmap(
        accum_raw,
        ratio_scores,
        run_id,
        n_processed,
        mean_baseline,
        out_dir / "delimiter_scores.png",
    )
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_delimiter_heads)
