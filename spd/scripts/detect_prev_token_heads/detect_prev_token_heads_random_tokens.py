"""Detect previous-token attention heads using random token sequences.

Same analysis as detect_prev_token_heads but with random (uniform) token IDs instead
of real text. Heads that score high here attend to the previous position regardless
of token content, indicating purely positional attention behavior.

Usage:
    python -m spd.scripts.detect_prev_token_heads.detect_prev_token_heads_random_tokens \
        wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import numpy as np
import torch

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.detect_prev_token_heads.detect_prev_token_heads import (
    _collect_attention_patterns,
    _plot_mean_attention_patterns,
    _plot_score_heatmap,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
N_BATCHES = 100
BATCH_SIZE = 32


def detect_prev_token_heads_random_tokens(
    wandb_path: ModelPath, n_batches: int = N_BATCHES
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
    vocab_size = target_model.config.vocab_size
    logger.info(f"Model: {n_layers} layers, {n_heads} heads, seq_len={seq_len}, vocab={vocab_size}")

    accum_scores = np.zeros((n_layers, n_heads))
    accum_patterns = [torch.zeros(n_heads, seq_len, seq_len) for _ in range(n_layers)]

    with torch.no_grad():
        for i in range(n_batches):
            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, seq_len), device=device)
            patterns = _collect_attention_patterns(target_model, input_ids)

            for layer_idx, att in enumerate(patterns):
                diag = torch.diagonal(att, offset=-1, dim1=-2, dim2=-1)
                accum_scores[layer_idx] += diag.float().mean(dim=(0, 2)).cpu().numpy()
                accum_patterns[layer_idx] += att.float().mean(dim=0).cpu()

            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

    accum_scores /= n_batches
    for layer_idx in range(n_layers):
        accum_patterns[layer_idx] /= n_batches

    logger.info(f"Previous-token scores on random tokens (n={n_batches} batches):")
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            score = accum_scores[layer_idx, h]
            marker = " <-- prev-token head" if score > 0.3 else ""
            logger.info(f"  L{layer_idx}H{h}: {score:.4f}{marker}")

    _plot_score_heatmap(
        accum_scores,
        run_id,
        n_batches,
        out_dir / "prev_token_scores_random_tokens.png",
    )
    _plot_mean_attention_patterns(
        accum_patterns,
        run_id,
        n_batches,
        out_dir / "mean_attention_patterns_random_tokens.png",
    )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(detect_prev_token_heads_random_tokens)
