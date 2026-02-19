"""Detect S-inhibition heads using IOI (Indirect Object Identification) prompts.

S-inhibition heads attend from the end of IOI sentences to the repeated subject
(S2) and suppress it from being predicted. We measure two signals:
  1. Attention from the final position to the S2 position (data-driven)
  2. OV copy score: whether the head's OV circuit promotes or suppresses the
     attended token's logit (weight-based, negative = inhibition)

Prompts follow the IOI pattern:
  "When [IO] and [S] went to the store, [S] gave a drink to" -> answer: [IO]

Usage:
    python -m spd.scripts.detect_s_inhibition_heads.detect_s_inhibition_heads \
        wandb:goodfire/spd/runs/<run_id>
"""

import math
import random
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F
from transformers import AutoTokenizer

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_PROMPTS = 500

TEMPLATE = "When{io} and{s} went to the store,{s} gave a drink to"

CANDIDATE_NAMES = [
    " Alice", " Bob", " Mary", " John", " Tom", " Sam", " Dan", " Jim", " Amy",
    " Eve", " Max", " Ben", " Ann", " Joe", " Kate", " Bill", " Jack", " Mark",
    " Paul", " Dave", " Luke", " Jill", " Brad", " Emma", " Alex", " Ryan",
    " Meg", " Zoe", " Beth", " Fred",
]  # fmt: skip


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


def _get_single_token_names(
    tokenizer: AutoTokenizer,
) -> list[tuple[str, int]]:
    """Return (name_string, token_id) for names that encode as a single token."""
    valid = []
    for name in CANDIDATE_NAMES:
        token_ids = tokenizer.encode(name)
        if len(token_ids) == 1:
            valid.append((name, token_ids[0]))
    return valid


def _create_ioi_batch(
    tokenizer: AutoTokenizer,
    names: list[tuple[str, int]],
    batch_size: int,
) -> tuple[torch.Tensor, list[int], list[int], list[int], list[int]]:
    """Create a batch of IOI prompts.

    Returns (input_ids, s2_positions, end_positions, io_token_ids, s_token_ids).
    All prompts use the same template, so they have identical token counts.
    """
    all_tokens: list[list[int]] = []
    s2_positions: list[int] = []
    end_positions: list[int] = []
    io_token_ids: list[int] = []
    s_token_ids: list[int] = []

    for _ in range(batch_size):
        (io_name, io_tid), (s_name, s_tid) = random.sample(names, 2)
        text = TEMPLATE.format(io=io_name, s=s_name)
        tokens = tokenizer.encode(text)

        s_positions = [idx for idx, t in enumerate(tokens) if t == s_tid]
        assert len(s_positions) == 2, (
            f"Expected 2 occurrences of '{s_name}', got {len(s_positions)}"
        )

        all_tokens.append(tokens)
        s2_positions.append(s_positions[1])
        end_positions.append(len(tokens) - 1)
        io_token_ids.append(io_tid)
        s_token_ids.append(s_tid)

    # All prompts should be the same length (same template, single-token names)
    assert all(len(t) == len(all_tokens[0]) for t in all_tokens)

    return (
        torch.tensor(all_tokens),
        s2_positions,
        end_positions,
        io_token_ids,
        s_token_ids,
    )


def _compute_ov_copy_scores(
    model: LlamaSimpleMLP,
    name_token_ids: list[int],
) -> NDArray[np.floating]:
    """Compute per-head OV copy score averaged over name tokens.

    copy_score(h, t) = W_U[t] @ W_O_h @ W_V_h @ W_E[t]
    Positive means the head promotes token t when attending to it (copying).
    Negative means the head suppresses it (inhibition).

    Returns shape (n_layers, n_heads).
    """
    W_E = model.wte.weight.float()  # (vocab, d_model)
    W_U = model.lm_head.weight.float()  # (vocab, d_model) — tied with W_E

    n_layers = len(model._h)
    head_dim = model._h[0].attn.head_dim
    n_heads = model._h[0].attn.n_head
    scores = np.zeros((n_layers, n_heads))

    name_embeds = W_E[name_token_ids]  # (n_names, d_model)
    name_unembed = W_U[name_token_ids]  # (n_names, d_model)

    for layer_idx, block in enumerate(model._h):
        attn = block.attn
        W_V = attn.v_proj.weight.float()  # (n_kv_heads * head_dim, d_model)
        W_O = attn.o_proj.weight.float()  # (d_model, d_model)

        for h in range(n_heads):
            kv_h = h * head_dim
            W_V_h = W_V[kv_h : kv_h + head_dim, :]  # (head_dim, d_model)
            W_O_h = W_O[:, kv_h : kv_h + head_dim]  # (d_model, head_dim)
            W_OV_h = W_O_h @ W_V_h  # (d_model, d_model)

            # copy_score for each name: unembed[t] @ W_OV @ embed[t]
            ov_output = name_embeds @ W_OV_h.T  # (n_names, d_model)
            copy = (name_unembed * ov_output).sum(dim=-1)  # (n_names,)
            scores[layer_idx, h] = copy.mean().item()

    return scores


def _plot_dual_heatmap(
    attn_scores: NDArray[np.floating],
    copy_scores: NDArray[np.floating],
    run_id: str,
    n_prompts: int,
    out_path: Path,
) -> None:
    n_layers, n_heads = attn_scores.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_heads * 2.4), max(4, n_layers * 1.0)))

    # Left: attention to S2
    im1 = ax1.imshow(attn_scores, aspect="auto", cmap="viridis", vmin=0)
    fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02, label="Attn to S2")
    ax1.set_title("Attention to S2 from end", fontsize=11, fontweight="bold")

    # Right: OV copy scores
    vabs = max(abs(copy_scores.min()), abs(copy_scores.max())) or 1.0
    im2 = ax2.imshow(copy_scores, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02, label="Copy score")
    ax2.set_title("OV copy score (neg = inhibit)", fontsize=11, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

    data_sets = [attn_scores, copy_scores]
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
        f"{run_id}  |  S-inhibition analysis  (n={n_prompts} IOI prompts)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_s_inhibition_heads(
    wandb_path: ModelPath, n_prompts: int = N_PROMPTS, batch_size: int = 50
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

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    names = _get_single_token_names(tokenizer)
    assert len(names) >= 4, f"Need at least 4 single-token names, got {len(names)}"
    logger.info(f"Found {len(names)} single-token names: {[n for n, _ in names]}")

    n_layers = len(target_model._h)
    n_heads = target_model._h[0].attn.n_head
    logger.info(f"Model: {n_layers} layers, {n_heads} heads")

    # Weight-based OV copy scores
    all_name_tids = [tid for _, tid in names]
    copy_scores = _compute_ov_copy_scores(target_model, all_name_tids)
    logger.info("OV copy scores computed")

    # Data-driven attention to S2
    accum_attn_to_s2 = np.zeros((n_layers, n_heads))
    n_processed = 0

    with torch.no_grad():
        for start in range(0, n_prompts, batch_size):
            bs = min(batch_size, n_prompts - start)
            input_ids, s2_positions, end_positions, _, _ = _create_ioi_batch(tokenizer, names, bs)
            input_ids = input_ids.to(device)
            patterns = _collect_attention_patterns(target_model, input_ids)

            for layer_idx, att in enumerate(patterns):
                for b in range(bs):
                    # Attention from end position to S2 position, per head
                    accum_attn_to_s2[layer_idx] += (
                        att[b, :, end_positions[b], s2_positions[b]].float().cpu().numpy()
                    )

            n_processed += bs
            logger.info(f"Processed {n_processed}/{n_prompts} IOI prompts")

    assert n_processed > 0
    accum_attn_to_s2 /= n_processed

    logger.info(f"S-inhibition analysis (n={n_processed} IOI prompts):")
    logger.info("  Attention to S2 | OV copy score")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            attn_val = accum_attn_to_s2[layer_idx, h]
            copy_val = copy_scores[layer_idx, h]
            marker = ""
            if attn_val > 0.1 and copy_val < 0:
                marker = " <-- S-inhibition candidate"
            logger.info(f"  L{layer_idx}H{h}: attn={attn_val:.4f}  copy={copy_val:.4f}{marker}")

    _plot_dual_heatmap(
        accum_attn_to_s2, copy_scores, run_id, n_processed, out_dir / "s_inhibition_scores.png"
    )
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_s_inhibition_heads)
