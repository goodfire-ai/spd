"""Measure effects of ablating attention heads or SPD parameter components.

Supports three ablation modes for components:
  - deterministic: all-ones masks as baseline, zero out target components
  - stochastic: CI-based masks with stochastic sources, target CI forced to 0
  - adversarial: PGD-optimized worst-case masks, target CI forced to 0

Usage:
    # Head ablation
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --heads L0H3,L1H5

    # Component ablation (deterministic)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3,h.1.attn.k_proj:7"

    # Component ablation (stochastic)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3" \
        --ablation_mode stochastic --n_mask_samples 10

    # Component ablation (adversarial / PGD)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3" \
        --ablation_mode adversarial --pgd_steps 50 --pgd_step_size 0.01
"""

import math
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import LMTaskConfig, PGDConfig, SamplingType
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.pretrain.models.llama_simple_mlp import CausalSelfAttention, LlamaSimpleMLP
from spd.routing import AllLayersRouter
from spd.spd_types import ModelPath
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent

AblationMode = Literal["deterministic", "stochastic", "adversarial"]

# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """Parse "L0H3,L1H5" → [(0, 3), (1, 5)]."""
    heads: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        m = re.fullmatch(r"L(\d+)H(\d+)", token)
        assert m is not None, f"Bad head spec: {token!r}, expected e.g. L0H3"
        heads.append((int(m.group(1)), int(m.group(2))))
    return heads


def parse_components(spec: str) -> list[tuple[str, int]]:
    """Parse "h.0.attn.q_proj:3,h.1.attn.k_proj:7" → [("h.0.attn.q_proj", 3), ...]."""
    components: list[tuple[str, int]] = []
    for token in spec.split(","):
        token = token.strip()
        parts = token.rsplit(":", 1)
        assert len(parts) == 2, f"Bad component spec: {token!r}, expected e.g. h.0.attn.q_proj:3"
        components.append((parts[0], int(parts[1])))
    return components


# ──────────────────────────────────────────────────────────────────────────────
# Patched attention forward (context manager)
# ──────────────────────────────────────────────────────────────────────────────

AttentionPatterns = dict[int, Float[Tensor, "n_heads T T"]]
ValueVectors = dict[int, Float[Tensor, "n_heads T head_dim"]]


class AttentionData(NamedTuple):
    patterns: AttentionPatterns  # layer → (n_heads, T, T)
    values: ValueVectors  # layer → (n_heads, T, head_dim)


@contextmanager
def ablate_v_proj_params(
    target_model: LlamaSimpleMLP,
    head_ablations: list[tuple[int, int]],
) -> Generator[None]:
    """Temporarily zero v_proj weight (and bias) rows for specified heads."""
    head_dim = target_model.config.n_embd // target_model.config.n_head
    saved_weights: list[Tensor] = []

    for layer, head in head_ablations:
        v_proj = target_model._h[layer].attn.v_proj
        row_slice = slice(head * head_dim, (head + 1) * head_dim)
        saved_weights.append(v_proj.weight.data[row_slice].clone())
        v_proj.weight.data[row_slice] = 0.0

    try:
        yield
    finally:
        for (layer, head), saved_weight in zip(head_ablations, saved_weights, strict=True):
            v_proj = target_model._h[layer].attn.v_proj
            row_slice = slice(head * head_dim, (head + 1) * head_dim)
            v_proj.weight.data[row_slice] = saved_weight


@contextmanager
def patched_attention_forward(
    target_model: LlamaSimpleMLP,
) -> Generator[AttentionData]:
    """Replace each CausalSelfAttention.forward to capture attention patterns and values.

    Yields AttentionData containing:
      - patterns: layer_index → attention pattern tensor (n_heads, T, T), mean over batch
      - values: layer_index → value vectors (n_heads, T, head_dim), mean over batch
    """
    patterns: AttentionPatterns = {}
    values: ValueVectors = {}
    originals: dict[int, object] = {}

    for layer_idx, block in enumerate(target_model._h):
        attn: CausalSelfAttention = block.attn
        originals[layer_idx] = attn.forward

        def _make_patched_forward(attn_module: CausalSelfAttention, li: int) -> object:
            def _patched_forward(
                x: Float[Tensor, "batch pos d_model"],
                attention_mask: Int[Tensor, "batch offset_pos"] | None = None,
                position_ids: Int[Tensor, "batch pos"] | None = None,
                _past_key_value: tuple[Tensor, Tensor] | None = None,
            ) -> Float[Tensor, "batch pos d_model"]:
                B, T, C = x.size()

                q = attn_module.q_proj(x)
                k = attn_module.k_proj(x)
                v = attn_module.v_proj(x)

                q = q.view(B, T, attn_module.n_head, attn_module.head_dim).transpose(1, 2)
                k = k.view(B, T, attn_module.n_key_value_heads, attn_module.head_dim).transpose(
                    1, 2
                )
                v = v.view(B, T, attn_module.n_key_value_heads, attn_module.head_dim).transpose(
                    1, 2
                )

                if position_ids is None:
                    if attention_mask is not None:
                        position_ids = attn_module.get_offset_position_ids(0, attention_mask)
                    else:
                        position_ids = torch.arange(T, device=x.device).unsqueeze(0)

                position_ids = position_ids.clamp(max=attn_module.n_ctx - 1)
                cos = attn_module.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                sin = attn_module.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                q, k = attn_module.apply_rotary_pos_emb(q, k, cos, sin)

                if attn_module.use_grouped_query_attention and attn_module.repeat_kv_heads > 1:
                    k = k.repeat_interleave(attn_module.repeat_kv_heads, dim=1)
                    v = v.repeat_interleave(attn_module.repeat_kv_heads, dim=1)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn_module.head_dim))
                causal_mask = attn_module.bias[:, :, :T, :T]  # pyright: ignore[reportIndexIssue]
                att = att.masked_fill(causal_mask == 0, float("-inf"))
                att = F.softmax(att, dim=-1)

                patterns[li] = att.float().mean(dim=0).detach().cpu()

                y = att @ v  # (B, n_head, T, head_dim)

                values[li] = v.float().mean(dim=0).detach().cpu()

                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = attn_module.o_proj(y)
                return y

            return _patched_forward

        attn.forward = _make_patched_forward(attn, layer_idx)  # pyright: ignore[reportAttributeAccessIssue]

    try:
        yield AttentionData(patterns, values)
    finally:
        for layer_idx, block in enumerate(target_model._h):
            block.attn.forward = originals[layer_idx]  # pyright: ignore[reportAttributeAccessIssue]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_attention_grid(
    patterns: AttentionPatterns,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(patterns)
    n_heads = patterns[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            pat = patterns[layer_idx][h, :max_pos, :max_pos].numpy()
            ax.imshow(pat, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_attention_diff(
    baseline: AttentionPatterns,
    ablated: AttentionPatterns,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(baseline)
    n_heads = baseline[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            diff = (
                ablated[layer_idx][h, :max_pos, :max_pos]
                - baseline[layer_idx][h, :max_pos, :max_pos]
            ).numpy()
            vmax = max(abs(diff.min()), abs(diff.max()), 1e-8)
            ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_value_norms(
    values: ValueVectors,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, n_layers * 2.5), squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]
        norms = values[layer_idx][:, :max_pos, :].norm(dim=-1).numpy()  # (n_heads, max_pos)
        im = ax.imshow(norms, aspect="auto", cmap="viridis")
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        n_heads = norms.shape[0]
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_value_norms_diff(
    baseline_values: ValueVectors,
    ablated_values: ValueVectors,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(baseline_values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, n_layers * 2.5), squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]
        baseline_norms = baseline_values[layer_idx][:, :max_pos, :].norm(dim=-1)
        ablated_norms = ablated_values[layer_idx][:, :max_pos, :].norm(dim=-1)
        diff = (ablated_norms - baseline_norms).numpy()
        vmax = max(abs(diff.min()), abs(diff.max()), 1e-8)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        n_heads = diff.shape[0]
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Token prediction table & stats
# ──────────────────────────────────────────────────────────────────────────────


def log_prediction_table(
    input_ids: Tensor,
    baseline_logits: Tensor,
    ablated_logits: Tensor,
    tokenizer: object,
    last_n: int = 20,
) -> int:
    """Log per-position prediction changes. Returns count of changed positions."""
    seq_len = input_ids.shape[0]
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)

    baseline_top = baseline_probs.argmax(dim=-1)
    ablated_top = ablated_probs.argmax(dim=-1)

    changed_mask = baseline_top != ablated_top
    changed_positions = changed_mask.nonzero(as_tuple=True)[0].tolist()
    show_positions = set(changed_positions) | set(range(max(0, seq_len - last_n), seq_len))

    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]
    lines = [f"{'Pos':>4} | {'Token':>10} | {'Baseline (prob)':>20} | {'Ablated (prob)':>20} | Chg"]
    lines.append("-" * 85)

    for pos in sorted(show_positions):
        tok = decode([input_ids[pos].item()]).replace("\n", "\\n")
        b_id = int(baseline_top[pos].item())
        a_id = int(ablated_top[pos].item())
        b_tok = decode([b_id]).replace("\n", "\\n")
        a_tok = decode([a_id]).replace("\n", "\\n")
        b_prob = baseline_probs[pos, b_id].item()
        a_prob = ablated_probs[pos, a_id].item()
        changed = " *" if pos in changed_positions else ""
        lines.append(
            f"{pos:>4} | {tok:>10} | {b_tok:>10} ({b_prob:.3f}) | {a_tok:>10} ({a_prob:.3f}) |{changed}"
        )

    logger.info("Prediction table:\n" + "\n".join(lines))
    return len(changed_positions)


def calc_mean_kl_divergence(
    baseline_logits: Tensor,
    ablated_logits: Tensor,
) -> float:
    """KL(baseline || ablated) averaged over positions, for first item in batch."""
    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
    kl = F.kl_div(ablated_log_probs, baseline_log_probs.exp(), reduction="batchmean")
    return kl.item()


# ──────────────────────────────────────────────────────────────────────────────
# Component mask construction
# ──────────────────────────────────────────────────────────────────────────────


def _build_deterministic_masks(
    model: ComponentModel,
    ablated_components: list[tuple[str, int]],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, ComponentsMaskInfo]]:
    """Build all-ones baseline and ablated mask_infos for deterministic mode."""
    baseline_masks: dict[str, Float[Tensor, "... C"]] = {}
    ablated_masks: dict[str, Float[Tensor, "... C"]] = {}

    for module_name in model.target_module_paths:
        c = model.module_to_c[module_name]
        baseline_masks[module_name] = torch.ones(*batch_shape, c, device=device)
        ablated_masks[module_name] = torch.ones(*batch_shape, c, device=device)

    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_masks, f"Module {module_name!r} not in model"
        ablated_masks[module_name][..., comp_idx] = 0.0

    return make_mask_infos(baseline_masks), make_mask_infos(ablated_masks)


def _build_stochastic_masks(
    _model: ComponentModel,
    ci: dict[str, Float[Tensor, "... C"]],
    ablated_components: list[tuple[str, int]],
    sampling: SamplingType,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, ComponentsMaskInfo]]:
    """Build stochastic mask_infos: baseline uses original CI, ablated zeros target CIs."""
    router = AllLayersRouter()
    baseline_mask_infos = calc_stochastic_component_mask_info(ci, sampling, None, router)

    ablated_ci = {k: v.clone() for k, v in ci.items()}
    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_ci, f"Module {module_name!r} not in model"
        ablated_ci[module_name][..., comp_idx] = 0.0

    ablated_mask_infos = calc_stochastic_component_mask_info(ablated_ci, sampling, None, router)
    return baseline_mask_infos, ablated_mask_infos


def _build_adversarial_masks(
    model: ComponentModel,
    batch: Int[Tensor, "batch pos"],
    ci: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... vocab"],
    ablated_components: list[tuple[str, int]],
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Run PGD for baseline and ablated, return (baseline_loss, ablated_loss)."""
    from spd.metrics.pgd_utils import pgd_masked_recon_loss_update

    router = AllLayersRouter()

    baseline_sum_loss, baseline_n = pgd_masked_recon_loss_update(
        model, batch, ci, None, target_out, "kl", router, pgd_config
    )

    ablated_ci = {k: v.clone() for k, v in ci.items()}
    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_ci, f"Module {module_name!r} not in model"
        ablated_ci[module_name][..., comp_idx] = 0.0

    ablated_sum_loss, ablated_n = pgd_masked_recon_loss_update(
        model, batch, ablated_ci, None, target_out, "kl", router, pgd_config
    )

    return baseline_sum_loss / baseline_n, ablated_sum_loss / ablated_n


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample result + accumulation helpers
# ──────────────────────────────────────────────────────────────────────────────


class SampleResult(NamedTuple):
    baseline_patterns: AttentionPatterns
    ablated_patterns: AttentionPatterns
    baseline_values: ValueVectors
    ablated_values: ValueVectors
    baseline_logits: Tensor  # (batch, pos, vocab)
    ablated_logits: Tensor  # (batch, pos, vocab)


def _add_patterns(accum: AttentionPatterns, new: AttentionPatterns) -> None:
    for layer_idx, pat in new.items():
        if layer_idx in accum:
            accum[layer_idx] = accum[layer_idx] + pat
        else:
            accum[layer_idx] = pat.clone()


def _scale_patterns(accum: AttentionPatterns, n: int) -> AttentionPatterns:
    return {k: v / n for k, v in accum.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Head ablation
# ──────────────────────────────────────────────────────────────────────────────


def _run_head_ablation(
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_heads: list[tuple[int, int]],
) -> SampleResult:
    with patched_attention_forward(target_model) as baseline_data:
        baseline_logits, _ = target_model(input_ids)

    with (
        ablate_v_proj_params(target_model, parsed_heads),
        patched_attention_forward(target_model) as ablated_data,
    ):
        ablated_logits, _ = target_model(input_ids)

    assert baseline_logits is not None and ablated_logits is not None
    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_logits,
        ablated_logits,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Component ablation
# ──────────────────────────────────────────────────────────────────────────────


def _run_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    ablation_mode: AblationMode,
    n_mask_samples: int,
    pgd_steps: int,
    pgd_step_size: float,
) -> SampleResult:
    match ablation_mode:
        case "deterministic":
            return _run_deterministic_component_ablation(
                spd_model, target_model, input_ids, parsed_components
            )
        case "stochastic":
            return _run_stochastic_component_ablation(
                spd_model, target_model, input_ids, parsed_components, n_mask_samples
            )
        case "adversarial":
            return _run_adversarial_component_ablation(
                spd_model, target_model, input_ids, parsed_components, pgd_steps, pgd_step_size
            )


def _run_deterministic_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
) -> SampleResult:
    batch_shape = input_ids.shape[:1]
    baseline_mask_infos, ablated_mask_infos = _build_deterministic_masks(
        spd_model, parsed_components, batch_shape, input_ids.device
    )

    with patched_attention_forward(target_model) as baseline_data:
        baseline_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
    assert isinstance(baseline_out, Tensor)

    with patched_attention_forward(target_model) as ablated_data:
        ablated_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
    assert isinstance(ablated_out, Tensor)

    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_out,
        ablated_out,
    )


def _run_stochastic_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    n_mask_samples: int,
) -> SampleResult:
    output_with_cache = spd_model(input_ids, cache_type="input")
    assert isinstance(output_with_cache, OutputWithCache)
    ci = spd_model.calc_causal_importances(output_with_cache.cache, "continuous").lower_leaky

    baseline_logits_accum: Tensor | None = None
    ablated_logits_accum: Tensor | None = None
    sample_baseline_patterns: AttentionPatterns = {}
    sample_ablated_patterns: AttentionPatterns = {}
    sample_baseline_values: ValueVectors = {}
    sample_ablated_values: ValueVectors = {}

    for _s in range(n_mask_samples):
        baseline_mask_infos, ablated_mask_infos = _build_stochastic_masks(
            spd_model, ci, parsed_components, "continuous"
        )

        with patched_attention_forward(target_model) as b_data:
            b_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
        assert isinstance(b_out, Tensor)

        with patched_attention_forward(target_model) as a_data:
            a_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
        assert isinstance(a_out, Tensor)

        if baseline_logits_accum is None:
            baseline_logits_accum = b_out
            ablated_logits_accum = a_out
        else:
            baseline_logits_accum = baseline_logits_accum + b_out
            assert ablated_logits_accum is not None
            ablated_logits_accum = ablated_logits_accum + a_out

        _add_patterns(sample_baseline_patterns, b_data.patterns)
        _add_patterns(sample_ablated_patterns, a_data.patterns)
        _add_patterns(sample_baseline_values, b_data.values)
        _add_patterns(sample_ablated_values, a_data.values)

    assert baseline_logits_accum is not None and ablated_logits_accum is not None
    return SampleResult(
        _scale_patterns(sample_baseline_patterns, n_mask_samples),
        _scale_patterns(sample_ablated_patterns, n_mask_samples),
        _scale_patterns(sample_baseline_values, n_mask_samples),
        _scale_patterns(sample_ablated_values, n_mask_samples),
        baseline_logits_accum / n_mask_samples,
        ablated_logits_accum / n_mask_samples,
    )


def _run_adversarial_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    pgd_steps: int,
    pgd_step_size: float,
) -> SampleResult:
    output_with_cache = spd_model(input_ids, cache_type="input")
    assert isinstance(output_with_cache, OutputWithCache)
    ci = spd_model.calc_causal_importances(output_with_cache.cache, "continuous").lower_leaky

    target_out = output_with_cache.output

    pgd_config = PGDConfig(
        init="random",
        step_size=pgd_step_size,
        n_steps=pgd_steps,
        mask_scope="unique_per_datapoint",
    )

    baseline_loss, ablated_loss = _build_adversarial_masks(
        spd_model, input_ids, ci, target_out, parsed_components, pgd_config
    )
    logger.info(
        f"PGD losses — baseline: {baseline_loss.item():.4f}, ablated: {ablated_loss.item():.4f}"
    )

    # Capture attention patterns with deterministic masks for visualization
    batch_shape = input_ids.shape[:1]
    baseline_mask_infos, ablated_mask_infos = _build_deterministic_masks(
        spd_model, parsed_components, batch_shape, input_ids.device
    )

    with patched_attention_forward(target_model) as baseline_data:
        baseline_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
    assert isinstance(baseline_out, Tensor)

    with patched_attention_forward(target_model) as ablated_data:
        ablated_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
    assert isinstance(ablated_out, Tensor)

    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_out,
        ablated_out,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _AggStats:
    total_changed: int = 0
    total_positions: int = 0
    total_kl: float = 0.0
    n_samples: int = 0


def run_attention_ablation(
    wandb_path: ModelPath,
    heads: str | None = None,
    components: str | None = None,
    ablation_mode: AblationMode = "deterministic",
    n_samples: int = 10,
    batch_size: int = 1,
    n_mask_samples: int = 10,
    pgd_steps: int = 50,
    pgd_step_size: float = 0.01,
    max_pos: int = 128,
) -> None:
    assert (heads is None) != (components is None), "Provide exactly one of --heads or --components"
    is_head_ablation = heads is not None
    parsed_heads = parse_heads(heads) if heads else []
    parsed_components = parse_components(components) if components else []

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model: ComponentModel | None = None
    if is_head_ablation:
        assert config.pretrained_model_name is not None
        target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
        target_model.eval()
        target_model.requires_grad_(False)
        for block in target_model._h:
            block.attn.flash_attention = False
        target_model = target_model.to(device)
    else:
        spd_model = ComponentModel.from_run_info(run_info)
        spd_model.eval()
        spd_model = spd_model.to(device)
        target_model = spd_model.target_model
        assert isinstance(target_model, LlamaSimpleMLP)
        for block in target_model._h:
            block.attn.flash_attention = False

    seq_len = target_model.config.n_ctx

    # Data loader
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
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=1000,
    )

    label = (
        f"heads={'_'.join(f'L{layer}H{head}' for layer, head in parsed_heads)}"
        if is_head_ablation
        else f"components={'_'.join(f'{m}:{c}' for m, c in parsed_components)}_mode={ablation_mode}"
    )
    logger.section(f"Attention ablation: {label}")
    logger.info(f"run_id={run_id}, device={device}, n_samples={n_samples}")

    attn_dir = out_dir / "attention_patterns"
    value_dir = out_dir / "value_norms"
    attn_dir.mkdir(parents=True, exist_ok=True)
    value_dir.mkdir(parents=True, exist_ok=True)

    accum_baseline_patterns: AttentionPatterns = {}
    accum_ablated_patterns: AttentionPatterns = {}
    accum_baseline_values: ValueVectors = {}
    accum_ablated_values: ValueVectors = {}
    stats = _AggStats()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break

            input_ids: Int[Tensor, "batch pos"] = batch_data[task_config.column_name][
                :, :seq_len
            ].to(device)

            if is_head_ablation:
                result = _run_head_ablation(target_model, input_ids, parsed_heads)
            else:
                assert spd_model is not None
                result = _run_component_ablation(
                    spd_model,
                    target_model,
                    input_ids,
                    parsed_components,
                    ablation_mode,
                    n_mask_samples,
                    pgd_steps,
                    pgd_step_size,
                )

            # Per-sample attention plots
            plot_attention_grid(
                result.baseline_patterns,
                f"{run_id} | Sample {i} baseline",
                attn_dir / f"{label}_sample{i}_baseline.png",
                max_pos,
            )
            plot_attention_grid(
                result.ablated_patterns,
                f"{run_id} | Sample {i} ablated",
                attn_dir / f"{label}_sample{i}_ablated.png",
                max_pos,
            )
            plot_attention_diff(
                result.baseline_patterns,
                result.ablated_patterns,
                f"{run_id} | Sample {i} diff",
                attn_dir / f"{label}_sample{i}_diff.png",
                max_pos,
            )

            # Per-sample value norm plots
            plot_value_norms(
                result.baseline_values,
                f"{run_id} | Sample {i} value norms baseline",
                value_dir / f"{label}_sample{i}_baseline.png",
                max_pos,
            )
            plot_value_norms(
                result.ablated_values,
                f"{run_id} | Sample {i} value norms ablated",
                value_dir / f"{label}_sample{i}_ablated.png",
                max_pos,
            )
            plot_value_norms_diff(
                result.baseline_values,
                result.ablated_values,
                f"{run_id} | Sample {i} value norms diff",
                value_dir / f"{label}_sample{i}_diff.png",
                max_pos,
            )

            # Per-sample prediction table
            n_changed = log_prediction_table(
                input_ids[0], result.baseline_logits[0], result.ablated_logits[0], tokenizer
            )

            # Accumulate stats
            sample_seq_len = input_ids.shape[1]
            stats.total_changed += n_changed
            stats.total_positions += sample_seq_len
            stats.total_kl += calc_mean_kl_divergence(
                result.baseline_logits[0], result.ablated_logits[0]
            )
            stats.n_samples += 1

            # Accumulate for mean plots
            _add_patterns(accum_baseline_patterns, result.baseline_patterns)
            _add_patterns(accum_ablated_patterns, result.ablated_patterns)
            _add_patterns(accum_baseline_values, result.baseline_values)
            _add_patterns(accum_ablated_values, result.ablated_values)

            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")

    assert stats.n_samples > 0, "No samples processed"

    # Mean attention plots
    mean_baseline_patterns = _scale_patterns(accum_baseline_patterns, stats.n_samples)
    mean_ablated_patterns = _scale_patterns(accum_ablated_patterns, stats.n_samples)

    plot_attention_grid(
        mean_baseline_patterns,
        f"{run_id} | Baseline mean attention (n={stats.n_samples})",
        attn_dir / f"{label}_mean_baseline.png",
        max_pos,
    )
    plot_attention_grid(
        mean_ablated_patterns,
        f"{run_id} | Ablated mean attention (n={stats.n_samples})",
        attn_dir / f"{label}_mean_ablated.png",
        max_pos,
    )
    plot_attention_diff(
        mean_baseline_patterns,
        mean_ablated_patterns,
        f"{run_id} | Attention diff mean (n={stats.n_samples})",
        attn_dir / f"{label}_mean_diff.png",
        max_pos,
    )

    # Mean value norm plots
    mean_baseline_values = _scale_patterns(accum_baseline_values, stats.n_samples)
    mean_ablated_values = _scale_patterns(accum_ablated_values, stats.n_samples)

    plot_value_norms(
        mean_baseline_values,
        f"{run_id} | Baseline mean value norms (n={stats.n_samples})",
        value_dir / f"{label}_mean_baseline.png",
        max_pos,
    )
    plot_value_norms(
        mean_ablated_values,
        f"{run_id} | Ablated mean value norms (n={stats.n_samples})",
        value_dir / f"{label}_mean_ablated.png",
        max_pos,
    )
    plot_value_norms_diff(
        mean_baseline_values,
        mean_ablated_values,
        f"{run_id} | Value norms diff mean (n={stats.n_samples})",
        value_dir / f"{label}_mean_diff.png",
        max_pos,
    )

    # Summary stats
    frac_changed = stats.total_changed / stats.total_positions
    mean_kl = stats.total_kl / stats.n_samples
    logger.section("Summary")
    logger.values(
        {
            "n_samples": stats.n_samples,
            "frac_top1_changed": f"{frac_changed:.4f}",
            "mean_kl_divergence": f"{mean_kl:.6f}",
        }
    )
    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(run_attention_ablation)
