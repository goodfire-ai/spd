"""Detect successor heads by measuring attention to ordinal predecessors.

Creates comma-separated ordinal sequences (digits, letters, number words, days)
and measures how much each head attends from element[k] to element[k-1]. Since
elements are separated by commas, the predecessor is 2 positions back -- not the
immediately preceding token -- which separates this signal from previous-token
heads.

A random-word control measures the same positional attention on non-ordinal
sequences. The successor-specific signal is ordinal_score - control_score.

Usage:
    python -m spd.scripts.detect_successor_heads.detect_successor_heads \
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

ORDINAL_SEQUENCES: list[list[str]] = [
    [f" {i}" for i in range(10)],
    [f" {chr(i)}" for i in range(ord("A"), ord("Z") + 1)],
    [" one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten"],
    [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"],
    [" January", " February", " March", " April", " May", " June",
     " July", " August", " September", " October", " November", " December"],
]  # fmt: skip

RANDOM_WORDS = [
    " cat", " dog", " red", " big", " cup", " hat", " sun", " box", " pen", " map",
    " key", " bag", " top", " old", " hot", " new", " run", " sit", " eat", " fly",
    " car", " bus", " bed", " arm", " egg", " ice", " oil", " tea", " war", " sky",
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


def _filter_single_token(elements: list[str], tokenizer: AutoTokenizer) -> list[tuple[str, int]]:
    """Return (element_string, token_id) for elements that are single tokens."""
    valid = []
    for elem in elements:
        token_ids = tokenizer.encode(elem)
        if len(token_ids) == 1:
            valid.append((elem, token_ids[0]))
    return valid


def _build_comma_sequence(
    elements: list[tuple[str, int]], comma_tid: int
) -> tuple[list[int], list[int]]:
    """Build token sequence: elem0, comma, elem1, comma, ...

    Returns (token_ids, element_positions).
    """
    tokens: list[int] = []
    positions: list[int] = []
    for idx, (_, tid) in enumerate(elements):
        if idx > 0:
            tokens.append(comma_tid)
        positions.append(len(tokens))
        tokens.append(tid)
    return tokens, positions


def _generate_prompts(
    tokenizer: AutoTokenizer,
    ordinal: bool,
    n_prompts: int,
) -> list[tuple[list[int], list[int]]]:
    """Generate comma-separated sequence prompts.

    Returns list of (token_ids, element_positions).
    For ordinal=True, uses ordinal sequences. For ordinal=False, uses random words.
    """
    comma_tids = tokenizer.encode(",")
    assert len(comma_tids) == 1
    comma_tid = comma_tids[0]

    if ordinal:
        valid_sequences = []
        for seq in ORDINAL_SEQUENCES:
            valid = _filter_single_token(seq, tokenizer)
            if len(valid) >= 3:
                valid_sequences.append(valid)
                logger.info(
                    f"  Ordinal sequence: {len(valid)} elements ({valid[0][0].strip()}, ...)"
                )
        assert valid_sequences, "No valid ordinal sequences found"
    else:
        random_valid = _filter_single_token(RANDOM_WORDS, tokenizer)
        assert len(random_valid) >= 5, f"Need at least 5 random words, got {len(random_valid)}"

    prompts: list[tuple[list[int], list[int]]] = []
    for _ in range(n_prompts):
        if ordinal:
            seq = random.choice(valid_sequences)
            min_len = 3
            max_len = min(len(seq), 12)
            subseq_len = random.randint(min_len, max_len)
            start = random.randint(0, len(seq) - subseq_len)
            elements = seq[start : start + subseq_len]
        else:
            subseq_len = random.randint(3, 12)
            elements = random.sample(random_valid, min(subseq_len, len(random_valid)))

        tokens, positions = _build_comma_sequence(elements, comma_tid)
        prompts.append((tokens, positions))

    return prompts


def _compute_predecessor_scores(
    model: LlamaSimpleMLP,
    prompts: list[tuple[list[int], list[int]]],
    device: torch.device,
) -> NDArray[np.floating]:
    """Compute mean attention from element[k] to element[k-1] for each head.

    Returns shape (n_layers, n_heads).
    """
    n_layers = len(model._h)
    n_heads = model._h[0].attn.n_head
    accum = np.zeros((n_layers, n_heads))
    n_pairs = 0

    with torch.no_grad():
        for tokens, positions in prompts:
            input_ids = torch.tensor([tokens], device=device)
            patterns = _collect_attention_patterns(model, input_ids)

            for k in range(1, len(positions)):
                dst_pos = positions[k]
                src_pos = positions[k - 1]
                for layer_idx, att in enumerate(patterns):
                    accum[layer_idx] += att[0, :, dst_pos, src_pos].float().cpu().numpy()
                n_pairs += 1

    assert n_pairs > 0
    return accum / n_pairs


def _plot_triple_heatmap(
    ordinal_scores: NDArray[np.floating],
    control_scores: NDArray[np.floating],
    run_id: str,
    n_prompts: int,
    out_path: Path,
) -> None:
    successor_signal = ordinal_scores - control_scores
    n_layers, n_heads = ordinal_scores.shape
    fig, axes = plt.subplots(1, 3, figsize=(max(18, n_heads * 3.6), max(4, n_layers * 1.0)))

    titles = ["Ordinal predecessor attn", "Random-word control", "Successor signal (diff)"]
    data_list = [ordinal_scores, control_scores, successor_signal]
    cmaps = ["viridis", "viridis", "RdBu_r"]

    for ax, data, title, cmap in zip(axes, data_list, titles, cmaps, strict=True):
        if cmap == "RdBu_r":
            vabs = max(abs(data.min()), abs(data.max())) or 1.0
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
        else:
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=9)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{li}" for li in range(n_layers)], fontsize=9)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        for layer_idx in range(n_layers):
            for h in range(n_heads):
                val = data[layer_idx, h]
                threshold = max(abs(data).max() * 0.6, 1e-6)
                color = "white" if abs(val) < threshold else "black"
                ax.text(
                    h, layer_idx, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color
                )

    fig.suptitle(
        f"{run_id}  |  Successor head analysis  (n={n_prompts} prompts per condition)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def detect_successor_heads(wandb_path: ModelPath, n_prompts: int = N_PROMPTS) -> None:
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

    n_layers = len(target_model._h)
    n_heads = target_model._h[0].attn.n_head
    logger.info(f"Model: {n_layers} layers, {n_heads} heads")

    logger.info("Generating ordinal prompts...")
    ordinal_prompts = _generate_prompts(tokenizer, ordinal=True, n_prompts=n_prompts)
    logger.info(f"Generated {len(ordinal_prompts)} ordinal prompts")

    logger.info("Generating random-word control prompts...")
    control_prompts = _generate_prompts(tokenizer, ordinal=False, n_prompts=n_prompts)
    logger.info(f"Generated {len(control_prompts)} control prompts")

    logger.info("Computing ordinal predecessor scores...")
    ordinal_scores = _compute_predecessor_scores(target_model, ordinal_prompts, device)

    logger.info("Computing control predecessor scores...")
    control_scores = _compute_predecessor_scores(target_model, control_prompts, device)

    successor_signal = ordinal_scores - control_scores

    logger.info(f"Successor head analysis (n={n_prompts} prompts per condition):")
    logger.info("  Ordinal | Control | Signal")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            o_val = ordinal_scores[layer_idx, h]
            c_val = control_scores[layer_idx, h]
            s_val = successor_signal[layer_idx, h]
            marker = " <-- successor head" if s_val > 0.05 else ""
            logger.info(f"  L{layer_idx}H{h}: {o_val:.4f}  {c_val:.4f}  {s_val:+.4f}{marker}")

    _plot_triple_heatmap(
        ordinal_scores, control_scores, run_id, n_prompts, out_dir / "successor_scores.png"
    )
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_successor_heads)
