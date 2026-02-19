"""Characterize which SPD components mediate L2H4's induction behavior.

Bridges head-level analysis (detect_induction_heads) with component-level decomposition
by measuring each component's causal contribution to the induction attention pattern.

Four phases:
  1. Weight-based component-head mapping (Frobenius norms)
  2. Per-component induction score via ablation
  3. Cross-head analysis of top induction components
  4. "Why not perfect?" analysis of attention mass allocation

Usage:
    python -m spd.scripts.characterize_induction_components.characterize_induction_components \
        wandb:goodfire/spd/runs/<run_id>
"""

import math
from io import StringIO
from pathlib import Path

import fire
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents

# Suppress buffer access issues (rotary_cos, rotary_sin, bias) on CausalSelfAttention
# pyright: reportIndexIssue=false
from spd.pretrain.models.llama_simple_mlp import CausalSelfAttention, LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.01
_default_n_batches = 20
_default_half_seq_len = 256
BATCH_SIZE = 32
TARGET_LAYER = 2
TARGET_HEAD = 4
TOP_N = 10


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices with mean_ci > MIN_MEAN_CI, sorted by mean_ci descending."""
    components = [
        (s.component_idx, s.mean_ci)
        for s in summary.values()
        if s.layer == module_path and s.mean_ci > MIN_MEAN_CI
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _head_entropy(fracs: NDArray[np.floating]) -> float:
    """Shannon entropy of a distribution (in bits). Clips zeros to avoid log(0)."""
    fracs = fracs[fracs > 0]
    return float(-np.sum(fracs * np.log2(fracs)))


# ── Phase 1: Weight-based component-head mapping ─────────────────────────────


def _compute_head_norm_fractions(
    component: LinearComponents,
    alive_indices: list[int],
    proj_name: str,
    head_dim: int,
    n_heads: int,
) -> NDArray[np.floating]:
    """Compute (n_alive, n_heads) array of per-head norm fractions for each component.

    For q/k/v_proj: head h uses rows [h*head_dim:(h+1)*head_dim] of U.
    For o_proj: head h uses columns [h*head_dim:(h+1)*head_dim] of V.

    Returns fractions (each row sums to 1).
    """
    n_alive = len(alive_indices)
    norms = np.zeros((n_alive, n_heads), dtype=np.float32)

    for row, c_idx in enumerate(alive_indices):
        if proj_name in ("q_proj", "k_proj", "v_proj"):
            u_c = component.U[c_idx].float()
            v_norm = torch.linalg.norm(component.V[:, c_idx].float()).item()
            for h in range(n_heads):
                head_norm = torch.linalg.norm(u_c[h * head_dim : (h + 1) * head_dim]).item()
                norms[row, h] = head_norm * v_norm
        else:
            v_c = component.V[:, c_idx].float()
            u_norm = torch.linalg.norm(component.U[c_idx].float()).item()
            for h in range(n_heads):
                head_norm = torch.linalg.norm(v_c[h * head_dim : (h + 1) * head_dim]).item()
                norms[row, h] = head_norm * u_norm

    row_totals = norms.sum(axis=1, keepdims=True)
    row_totals = np.maximum(row_totals, 1e-12)
    fracs = norms / row_totals
    return fracs


def _run_phase1(
    model: ComponentModel,
    summary: dict[str, ComponentSummary],
    head_dim: int,
    n_heads: int,
    out: StringIO,
) -> dict[str, tuple[list[int], NDArray[np.floating]]]:
    """Phase 1: Weight-based component-head mapping.

    Returns {proj_name: (alive_indices, head_norm_fracs)} for L2 q_proj and k_proj.
    """
    out.write("=" * 80 + "\n")
    out.write("PHASE 1: Weight-based component-head mapping\n")
    out.write("=" * 80 + "\n\n")

    results: dict[str, tuple[list[int], NDArray[np.floating]]] = {}

    for proj_name in ("q_proj", "k_proj"):
        module_path = f"h.{TARGET_LAYER}.attn.{proj_name}"
        component = model.components[module_path]
        assert isinstance(component, LinearComponents)

        alive = _get_alive_indices(summary, module_path)
        fracs = _compute_head_norm_fractions(component, alive, proj_name, head_dim, n_heads)
        results[proj_name] = (alive, fracs)

        out.write(f"── {module_path} ({len(alive)} alive components) ──\n")
        out.write(f"{'Comp':>6}  {'H4 frac':>8}  {'Dom head':>9}  {'Entropy':>8}  {'Class':>16}\n")
        out.write("-" * 55 + "\n")

        n_concentrated = n_involved = n_minor = 0
        for i, c_idx in enumerate(alive):
            h4_frac = fracs[i, TARGET_HEAD]
            dom_head = int(np.argmax(fracs[i]))
            entropy = _head_entropy(fracs[i])
            if h4_frac > 0.5:
                cls = "H4-concentrated"
                n_concentrated += 1
            elif h4_frac > 0.1:
                cls = "H4-involved"
                n_involved += 1
            else:
                cls = "H4-minor"
                n_minor += 1
            out.write(
                f"C{c_idx:>4}  {h4_frac:>8.3f}  {'H' + str(dom_head):>9}  {entropy:>8.3f}  {cls:>16}\n"
            )

        out.write(
            f"\nSummary: {n_concentrated} concentrated, {n_involved} involved, {n_minor} minor\n\n"
        )

    return results


# ── Phase 2: Per-component induction score via ablation ──────────────────────


def _run_layers_0_to_1(
    target_model: LlamaSimpleMLP,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run layers 0-1 and return residual stream at L2 input."""
    x = target_model.wte(input_ids)
    for block in target_model._h[:TARGET_LAYER]:
        x = x + block.attn(block.rms_1(x))
        x = x + block.mlp(block.rms_2(x))
    return x


def _compute_attention_weights(
    attn_input: torch.Tensor,
    attn: CausalSelfAttention,
    q_full: torch.Tensor | None = None,
    k_full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute softmax attention weights for a given layer's attention module.

    If q_full/k_full are provided, use those instead of computing from attn_input.
    Returns (batch, n_heads, seq_len, seq_len).
    """
    B, T, _ = attn_input.shape

    q_proj = q_full if q_full is not None else attn.q_proj(attn_input)
    k_proj = k_full if k_full is not None else attn.k_proj(attn_input)

    q = q_proj.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k_proj.view(B, T, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)

    position_ids = torch.arange(T, device=attn_input.device).unsqueeze(0)
    cos = attn.rotary_cos[position_ids].to(q.dtype)
    sin = attn.rotary_sin[position_ids].to(q.dtype)
    q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)

    if attn.repeat_kv_heads > 1:
        k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
    att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att


def _induction_score_from_attn(
    att: torch.Tensor,
    half_len: int,
) -> NDArray[np.floating]:
    """Compute induction score per head from attention weights.

    att: (batch, n_heads, seq_len, seq_len)
    Returns: (n_heads,) array of mean induction scores.
    """
    src = torch.arange(1, half_len, device=att.device)
    dst = torch.arange(half_len, 2 * half_len - 1, device=att.device)
    induction_attn = att[:, :, dst, src]  # (batch, n_heads, half_len-1)
    return induction_attn.float().mean(dim=(0, 2)).cpu().numpy()


def _run_phase2(
    target_model: LlamaSimpleMLP,
    model: ComponentModel,
    summary: dict[str, ComponentSummary],
    device: torch.device,
    n_batches: int,
    half_seq_len: int,
    out: StringIO,
) -> dict[str, list[tuple[int, float, NDArray[np.floating]]]]:
    """Phase 2: Per-component induction score via ablation.

    Returns {proj_name: [(component_idx, delta_h4, delta_all_heads), ...]}.
    """
    out.write("=" * 80 + "\n")
    out.write("PHASE 2: Per-component induction score via ablation\n")
    out.write("=" * 80 + "\n\n")

    attn = target_model._h[TARGET_LAYER].attn
    rms = target_model._h[TARGET_LAYER].rms_1
    n_heads = attn.n_head
    vocab_size = target_model.config.vocab_size

    # Accumulate baseline scores and per-component deltas
    baseline_accum = np.zeros(n_heads, dtype=np.float64)

    ablation_results: dict[str, dict[int, np.ndarray]] = {}
    for proj_name in ("q_proj", "k_proj"):
        module_path = f"h.{TARGET_LAYER}.attn.{proj_name}"
        alive = _get_alive_indices(summary, module_path)
        ablation_results[proj_name] = {c: np.zeros(n_heads, dtype=np.float64) for c in alive}

    for batch_i in range(n_batches):
        first_half = torch.randint(100, vocab_size - 100, (BATCH_SIZE, half_seq_len), device=device)
        input_ids = torch.cat([first_half, first_half], dim=1)

        l2_input = _run_layers_0_to_1(target_model, input_ids)
        attn_input = rms(l2_input)

        q_full = attn.q_proj(attn_input)
        k_full = attn.k_proj(attn_input)

        baseline_att = _compute_attention_weights(attn_input, attn, q_full, k_full)
        baseline_scores = _induction_score_from_attn(baseline_att, half_seq_len)
        baseline_accum += baseline_scores

        for proj_name in ("q_proj", "k_proj"):
            module_path = f"h.{TARGET_LAYER}.attn.{proj_name}"
            component = model.components[module_path]
            assert isinstance(component, LinearComponents)
            alive = _get_alive_indices(summary, module_path)

            full_proj = q_full if proj_name == "q_proj" else k_full

            for c_idx in alive:
                v_c = component.V[:, c_idx]  # (d_in,)
                u_c = component.U[c_idx]  # (d_out,)
                scalar_c = (attn_input @ v_c).unsqueeze(-1)  # (B, T, 1)
                contribution_c = scalar_c * u_c.unsqueeze(0).unsqueeze(0)  # (B, T, d_out)

                ablated_proj = full_proj - contribution_c

                if proj_name == "q_proj":
                    att_ablated = _compute_attention_weights(
                        attn_input, attn, q_full=ablated_proj, k_full=k_full
                    )
                else:
                    att_ablated = _compute_attention_weights(
                        attn_input, attn, q_full=q_full, k_full=ablated_proj
                    )

                ablated_scores = _induction_score_from_attn(att_ablated, half_seq_len)
                delta = baseline_scores - ablated_scores  # positive = component helps induction
                ablation_results[proj_name][c_idx] += delta

        if (batch_i + 1) % 5 == 0:
            logger.info(f"Phase 2: processed {batch_i + 1}/{n_batches} batches")

    baseline_accum /= n_batches
    for proj_name in ablation_results:
        for c_idx in ablation_results[proj_name]:
            ablation_results[proj_name][c_idx] /= n_batches

    out.write(f"Baseline induction scores (n={n_batches} batches of {BATCH_SIZE}):\n")
    for h in range(n_heads):
        marker = " <-- TARGET" if h == TARGET_HEAD else ""
        out.write(f"  H{h}: {baseline_accum[h]:.4f}{marker}\n")
    out.write("\n")

    # Build sorted result lists
    phase2_results: dict[str, list[tuple[int, float, NDArray[np.floating]]]] = {}
    for proj_name in ("q_proj", "k_proj"):
        items: list[tuple[int, float, NDArray[np.floating]]] = []
        for c_idx, delta_all in ablation_results[proj_name].items():
            delta_h4 = float(delta_all[TARGET_HEAD])
            items.append((c_idx, delta_h4, delta_all))
        items.sort(key=lambda t: t[1], reverse=True)
        phase2_results[proj_name] = items

        out.write(
            f"── h.{TARGET_LAYER}.attn.{proj_name}: Top components by H4 induction contribution ──\n"
        )
        head_labels = "  ".join(f"{'H' + str(h):>7}" for h in range(n_heads))
        out.write(f"{'Comp':>6}  {'dH4':>8}  {head_labels}\n")
        out.write("-" * (20 + n_heads * 9) + "\n")
        for c_idx, delta_h4, delta_all in items[:30]:
            deltas_str = "  ".join(f"{delta_all[h]:>+7.4f}" for h in range(n_heads))
            out.write(f"C{c_idx:>4}  {delta_h4:>+8.4f}  {deltas_str}\n")
        out.write("\n")

        total_h4 = sum(delta_h4 for _, delta_h4, _ in items)
        out.write(f"Sum of all {proj_name} component deltas for H4: {total_h4:+.4f}\n")
        out.write(f"Baseline H4 induction score: {baseline_accum[TARGET_HEAD]:.4f}\n\n")

    return phase2_results


# ── Phase 3: Cross-head analysis of top induction components ─────────────────


def _run_phase3(
    phase1_results: dict[str, tuple[list[int], NDArray[np.floating]]],
    phase2_results: dict[str, list[tuple[int, float, NDArray[np.floating]]]],
    n_heads: int,
    out: StringIO,
) -> None:
    """Phase 3: Cross-head analysis of top induction components."""
    out.write("=" * 80 + "\n")
    out.write("PHASE 3: Cross-head analysis of top induction components\n")
    out.write("=" * 80 + "\n\n")

    for proj_name in ("q_proj", "k_proj"):
        alive_indices, head_fracs = phase1_results[proj_name]
        idx_to_row = {c: i for i, c in enumerate(alive_indices)}

        top_components = phase2_results[proj_name][:TOP_N]

        out.write(
            f"── h.{TARGET_LAYER}.attn.{proj_name}: Top {TOP_N} by induction contribution ──\n\n"
        )

        for c_idx, delta_h4, delta_all in top_components:
            out.write(f"Component C{c_idx} (dH4 = {delta_h4:+.4f}):\n")

            # Weight norm distribution
            if c_idx in idx_to_row:
                row = idx_to_row[c_idx]
                fracs = head_fracs[row]
                out.write("  Weight norm fraction per head:\n    ")
                out.write(
                    "  ".join(
                        f"H{h}: {fracs[h]:.3f}{'*' if h == TARGET_HEAD else ''}"
                        for h in range(n_heads)
                    )
                )
                out.write("\n")

            # Ablation effect per head
            out.write("  Induction score change per head when ablated:\n    ")
            out.write(
                "  ".join(
                    f"H{h}: {delta_all[h]:+.4f}{'*' if h == TARGET_HEAD else ''}"
                    for h in range(n_heads)
                )
            )
            out.write("\n")

            # Cross-head effects
            significant_other = [
                (h, float(delta_all[h]))
                for h in range(n_heads)
                if h != TARGET_HEAD and abs(delta_all[h]) > 0.005
            ]
            if significant_other:
                out.write("  Significant cross-head effects (|delta| > 0.005):\n")
                for h, d in significant_other:
                    direction = "increases" if d < 0 else "decreases"
                    out.write(
                        f"    H{h}: {d:+.4f} (ablating this component {direction} H{h} induction)\n"
                    )

            out.write("\n")


# ── Phase 4: "Why not perfect?" analysis ─────────────────────────────────────


def _run_phase4(
    target_model: LlamaSimpleMLP,
    phase1_results: dict[str, tuple[list[int], NDArray[np.floating]]],
    phase2_results: dict[str, list[tuple[int, float, NDArray[np.floating]]]],
    device: torch.device,
    n_heads: int,
    half_seq_len: int,
    out: StringIO,
) -> None:
    """Phase 4: Analyze what prevents L2H4 from having a perfect induction score."""
    out.write("=" * 80 + "\n")
    out.write("PHASE 4: Why not perfect? Analysis\n")
    out.write("=" * 80 + "\n\n")

    attn = target_model._h[TARGET_LAYER].attn
    rms = target_model._h[TARGET_LAYER].rms_1
    vocab_size = target_model.config.vocab_size
    seq_len = half_seq_len * 2

    # 4a: BOS attention competition
    out.write("── 4a: Attention mass allocation in H4 on induction data ──\n\n")

    attn_to_bos_accum = np.zeros(n_heads, dtype=np.float64)
    attn_to_induction_accum = np.zeros(n_heads, dtype=np.float64)
    attn_to_other_accum = np.zeros(n_heads, dtype=np.float64)

    n_batches_phase4 = 10
    for _ in range(n_batches_phase4):
        first_half = torch.randint(100, vocab_size - 100, (BATCH_SIZE, half_seq_len), device=device)
        input_ids = torch.cat([first_half, first_half], dim=1)
        l2_input = _run_layers_0_to_1(target_model, input_ids)
        attn_input = rms(l2_input)
        att = _compute_attention_weights(attn_input, attn)

        second_half_att = att[:, :, half_seq_len:, :]  # (B, n_heads, half_len, seq_len)
        bos_att = second_half_att[:, :, :, 0]  # (B, n_heads, half_len)
        attn_to_bos_accum += bos_att.float().mean(dim=(0, 2)).cpu().numpy()

        # Induction target: for query at pos half_len+k, target is pos k+1
        src = torch.arange(1, half_seq_len + 1, device=device).clamp(max=seq_len - 1)
        dst_range = torch.arange(half_seq_len, device=device)
        induction_att = second_half_att[:, :, dst_range, src]  # (B, n_heads, half_len)
        attn_to_induction_accum += induction_att.float().mean(dim=(0, 2)).cpu().numpy()

        other_att = 1.0 - bos_att - induction_att
        attn_to_other_accum += other_att.float().mean(dim=(0, 2)).cpu().numpy()

    attn_to_bos_accum /= n_batches_phase4
    attn_to_induction_accum /= n_batches_phase4
    attn_to_other_accum /= n_batches_phase4

    out.write(f"{'Head':>6}  {'Induction':>10}  {'BOS':>8}  {'Other':>8}\n")
    out.write("-" * 38 + "\n")
    for h in range(n_heads):
        marker = " <-- TARGET" if h == TARGET_HEAD else ""
        out.write(
            f"  H{h}    {attn_to_induction_accum[h]:>10.4f}  {attn_to_bos_accum[h]:>8.4f}  "
            f"{attn_to_other_accum[h]:>8.4f}{marker}\n"
        )
    out.write("\n")

    # 4b: Non-induction components in H4
    out.write("── 4b: H4-concentrated components with low induction contribution ──\n\n")
    out.write(
        "These components have significant weight in H4 but contribute little to induction.\n"
    )
    out.write("They may drive BOS attention or other non-induction patterns.\n\n")

    for proj_name in ("q_proj", "k_proj"):
        alive_indices, head_fracs = phase1_results[proj_name]
        idx_to_row = {c: i for i, c in enumerate(alive_indices)}

        ablation_map = {c_idx: delta_h4 for c_idx, delta_h4, _ in phase2_results[proj_name]}

        non_induction_h4: list[tuple[int, float, float]] = []
        for c_idx in alive_indices:
            row = idx_to_row[c_idx]
            h4_frac = float(head_fracs[row, TARGET_HEAD])
            delta_h4 = ablation_map.get(c_idx, 0.0)
            if h4_frac > 0.1 and delta_h4 < 0.01:
                non_induction_h4.append((c_idx, h4_frac, delta_h4))

        non_induction_h4.sort(key=lambda t: t[1], reverse=True)

        out.write(
            f"  {proj_name}: {len(non_induction_h4)} components with H4 frac > 0.1 and dH4 < 0.01:\n"
        )
        out.write(f"  {'Comp':>6}  {'H4 frac':>8}  {'dH4':>8}\n")
        out.write("  " + "-" * 28 + "\n")
        for c_idx, h4_frac, delta_h4 in non_induction_h4[:15]:
            out.write(f"  C{c_idx:>4}  {h4_frac:>8.3f}  {delta_h4:>+8.4f}\n")
        out.write("\n")

    # 4c: Induction leakage / competition across heads
    out.write("── 4c: Cross-head induction leakage ──\n\n")
    out.write("Components whose ablation *increases* another head's induction score,\n")
    out.write("suggesting competitive dynamics.\n\n")

    for proj_name in ("q_proj", "k_proj"):
        leakage: list[tuple[int, int, float, float]] = []
        for c_idx, delta_h4, delta_all in phase2_results[proj_name]:
            for h in range(n_heads):
                if h != TARGET_HEAD and delta_all[h] < -0.005:
                    leakage.append((c_idx, h, float(delta_all[h]), delta_h4))

        if leakage:
            leakage.sort(key=lambda t: t[2])
            out.write(
                f"  {proj_name}: {len(leakage)} cases of ablation increasing other heads' induction:\n"
            )
            out.write(f"  {'Comp':>6}  {'Head':>6}  {'dHead':>8}  {'dH4':>8}\n")
            out.write("  " + "-" * 34 + "\n")
            for c_idx, h, d_other, d_h4 in leakage[:15]:
                out.write(f"  C{c_idx:>4}  H{h:>4}  {d_other:>+8.4f}  {d_h4:>+8.4f}\n")
            out.write("\n")
        else:
            out.write(f"  {proj_name}: No significant cross-head leakage detected.\n\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def characterize_induction_components(
    wandb_path: ModelPath,
    n_batches: int = _default_n_batches,
    half_seq_len: int = _default_half_seq_len,
) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()

    target_model = spd_model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for block in target_model._h:
        block.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model = spd_model.to(device)
    target_model = target_model.to(device)

    repo = HarvestRepo.open(run_id)
    assert repo is not None, f"No harvest data for {run_id}"
    summary = repo.get_summary()

    n_heads = target_model._h[TARGET_LAYER].attn.n_head
    head_dim = target_model._h[TARGET_LAYER].attn.head_dim

    logger.info(f"Model: {len(target_model._h)} layers, {n_heads} heads, head_dim={head_dim}")
    logger.info(f"Target: L{TARGET_LAYER}H{TARGET_HEAD}")

    report = StringIO()
    report.write("Induction Component Characterization Report\n")
    report.write(f"Run: {run_id}\n")
    report.write(f"Target: L{TARGET_LAYER}H{TARGET_HEAD}\n")
    report.write(f"Batches: {n_batches} x {BATCH_SIZE}, half_seq_len={half_seq_len}\n")
    report.write(f"Device: {device}\n\n")

    with torch.no_grad():
        phase1_results = _run_phase1(spd_model, summary, head_dim, n_heads, report)
        phase2_results = _run_phase2(
            target_model,
            spd_model,
            summary,
            device,
            n_batches,
            half_seq_len,
            report,
        )
        _run_phase3(phase1_results, phase2_results, n_heads, report)
        _run_phase4(
            target_model,
            phase1_results,
            phase2_results,
            device,
            n_heads,
            half_seq_len,
            report,
        )

    report_text = report.getvalue()
    print(report_text)

    report_path = out_dir / "induction_component_report.txt"
    report_path.write_text(report_text)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    fire.Fire(characterize_induction_components)
