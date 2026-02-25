"""Diagnostic: per-position gradient distributions for attribution edges.

Computes per-(batch, pos) gradients for specific (target, source) pairs to
visualize whether summing across positions causes pathological cancellation.

The normal harvester computes:
    attr[t, s] = Σ_{b,p} (∂target[b,p,t]/∂source[b,p,:]) · source_act[b,p,:]

This script instead collects the B×S individual `grad × act` vectors, giving
a distribution over positions. Heavy cancellation = the mean is much smaller
than the L2 norm.

Two metrics compared:
    mean_attr = mean(grad × act × ci)           — what we currently accumulate (signed)
    l2_attr   = sqrt(mean((grad × act × ci)²))  — magnitude-preserving alternative

Usage:
    # Single edge
    python -m spd.dataset_attributions.scripts.diagnose_cancellation single \
        "wandb:goodfire/spd/s-892f140b" \
        --target_key "0.mlp.up:5" --source_layer "embed"

    # Multi-edge survey
    python -m spd.dataset_attributions.scripts.diagnose_cancellation survey \
        "wandb:goodfire/spd/s-892f140b" --n_batches 2 --n_targets_per_layer 3
"""

import random
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.data import train_loader_and_tokenizer
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.settings import SPD_OUT_DIR
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import bf16_autocast, extract_batch_data

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Model setup & forward pass (shared)
# ---------------------------------------------------------------------------


def _setup(wandb_path: str) -> tuple[
    ComponentModel,
    TransformerTopology,
    SamplingType,
    dict[str, list[str]],
    torch.device,
]:
    device = torch.device(get_device())
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()
    topology = TransformerTopology(model.target_model)

    sources_by_target_raw = get_sources_by_target(model, topology, str(device), run_info.config.sampling)
    component_layers = set(model.target_module_paths)
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    valid_sources = component_layers | {embed_path}
    valid_targets = component_layers | {unembed_path}
    sources_by_target: dict[str, list[str]] = {}
    for target, sources in sources_by_target_raw.items():
        if target not in valid_targets:
            continue
        filtered = [s for s in sources if s in valid_sources]
        if filtered:
            sources_by_target[target] = filtered

    return model, topology, run_info.config.sampling, sources_by_target, device


def _forward_with_caches(
    model: ComponentModel,
    tokens: Tensor,
    sampling: SamplingType,
    embed_path: str,
    unembed_path: str,
    embedding_module: nn.Embedding,
    unembed_module: nn.Linear,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    embed_out: list[Tensor] = []
    pre_unembed: list[Tensor] = []

    def embed_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
        out.requires_grad_(True)
        embed_out.clear()
        embed_out.append(out)
        return out

    def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
        args[0].requires_grad_(True)
        pre_unembed.clear()
        pre_unembed.append(args[0])

    h1 = embedding_module.register_forward_hook(embed_hook, with_kwargs=True)
    h2 = unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)

    with torch.no_grad(), bf16_autocast():
        out = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling=sampling, detach_inputs=False
        )

    mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        routing_masks="all",
    )

    with torch.enable_grad(), bf16_autocast():
        model_output = model(tokens, mask_infos=mask_infos, cache_type="component_acts")

    h1.remove()
    h2.remove()

    cache = model_output.cache
    cache[f"{embed_path}_post_detach"] = embed_out[0]
    cache[f"{unembed_path}_pre_detach"] = pre_unembed[0]

    return cache, ci.lower_leaky


# ---------------------------------------------------------------------------
# Per-position grad collection
# ---------------------------------------------------------------------------


def _resolve_concrete(
    layer: str,
    model: ComponentModel,
    topology: TransformerTopology,
) -> str:
    embed_path = topology.path_schema.embedding_path
    if layer == "embed":
        return embed_path
    concrete_paths = set(model.target_module_paths)
    if layer in concrete_paths:
        return layer
    return topology.canon_to_target(layer)


def collect_per_position_grads(
    model: ComponentModel,
    topology: TransformerTopology,
    sampling: SamplingType,
    sources_by_target: dict[str, list[str]],
    device: torch.device,
    tokens: Tensor,
    target_concrete: str,
    target_idx: int,
    source_concrete: str,
) -> dict[str, Tensor]:
    """Collect per-position grad×act values for one (target_component, source_layer) edge.

    Returns dict with:
        "per_pos_attr": (B*S, C_source)
        "ci_weighted":  (B*S, C_source) — only for component sources
    """
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path

    assert target_concrete in sources_by_target
    assert source_concrete in sources_by_target[target_concrete]

    B, S = tokens.shape

    cache, ci = _forward_with_caches(
        model, tokens, sampling, embed_path, unembed_path,
        topology.embedding_module, topology.unembed_module,
    )

    target_acts_raw = cache[f"{target_concrete}_pre_detach"]  # (B, S, C_target)
    source_act = cache[f"{source_concrete}_post_detach"]  # (B, S, C_source)

    batch_grads = []
    for b in range(B):
        for s in range(S):
            scalar = target_acts_raw[b, s, target_idx]
            (grad,) = torch.autograd.grad(scalar, source_act, retain_graph=True)
            with torch.no_grad():
                batch_grads.append((grad[b, s] * source_act[b, s]).cpu())

    per_pos = torch.stack(batch_grads)  # (B*S, C_source)

    result: dict[str, Tensor] = {"per_pos_attr": per_pos}
    if source_concrete != embed_path and source_concrete in ci:
        ci_vals = ci[source_concrete].reshape(B * S, -1).detach().cpu()
        result["ci_weighted"] = per_pos * ci_vals

    return result


# ---------------------------------------------------------------------------
# Metrics: mean_attr vs l2_attr
# ---------------------------------------------------------------------------


@dataclass
class EdgeStats:
    edge_label: str
    n_positions: int
    n_source_components: int
    # Per source-component vectors, then summarised across components
    raw_cancel_median: float
    raw_cancel_mean: float
    ci_cancel_median: float | None  # None for embed sources
    ci_cancel_mean: float | None
    # L2 comparison: ratio of |mean| to L2 = |E[x]| / sqrt(E[x²])
    raw_mean_l2_median: float
    raw_mean_l2_mean: float
    ci_mean_l2_median: float | None
    ci_mean_l2_mean: float | None


def _compute_stats(per_pos: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (cancel_ratio, mean_over_l2) per source component."""
    means = per_pos.float().mean(dim=0)
    mean_abs = per_pos.float().abs().mean(dim=0)
    rms = per_pos.float().square().mean(dim=0).sqrt()

    cancel = means.abs() / mean_abs.clamp(min=1e-10)
    mean_l2 = means.abs() / rms.clamp(min=1e-10)
    return cancel, mean_l2


def compute_edge_stats(data: dict[str, Tensor], edge_label: str) -> EdgeStats:
    per_pos = data["per_pos_attr"]
    N, C = per_pos.shape

    raw_cancel, raw_ml2 = _compute_stats(per_pos)

    ci_cancel_med = ci_cancel_mean = ci_ml2_med = ci_ml2_mean = None
    if "ci_weighted" in data:
        ci_cancel, ci_ml2 = _compute_stats(data["ci_weighted"])
        ci_cancel_med = ci_cancel.median().item()
        ci_cancel_mean = ci_cancel.mean().item()
        ci_ml2_med = ci_ml2.median().item()
        ci_ml2_mean = ci_ml2.mean().item()

    return EdgeStats(
        edge_label=edge_label,
        n_positions=N,
        n_source_components=C,
        raw_cancel_median=raw_cancel.median().item(),
        raw_cancel_mean=raw_cancel.mean().item(),
        ci_cancel_median=ci_cancel_med,
        ci_cancel_mean=ci_cancel_mean,
        raw_mean_l2_median=raw_ml2.median().item(),
        raw_mean_l2_mean=raw_ml2.mean().item(),
        ci_mean_l2_median=ci_ml2_med,
        ci_mean_l2_mean=ci_ml2_mean,
    )


# ---------------------------------------------------------------------------
# Survey: run across many edges
# ---------------------------------------------------------------------------


def survey(
    wandb_path: str,
    n_batches: int = 2,
    batch_size: int = 4,
    n_targets_per_layer: int = 3,
    seed: int = 42,
) -> None:
    """Run cancellation analysis across many (target, source) pairs."""
    model, topology, sampling, sources_by_target, device = _setup(wandb_path)
    spd_config = SPDRunInfo.from_path(wandb_path).config
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    to_canon = topology.target_to_canon

    # Collect batches upfront (reuse across all edges)
    train_loader, _ = train_loader_and_tokenizer(spd_config, batch_size)
    batches: list[Tensor] = []
    for i, batch_data in enumerate(train_loader):
        if i >= n_batches:
            break
        batches.append(extract_batch_data(batch_data).to(device))
    print(f"Loaded {len(batches)} batches, {sum(b.numel() for b in batches)} tokens")

    # Select target components to probe
    rng = random.Random(seed)
    pairs: list[tuple[str, int, str]] = []  # (target_concrete, target_idx, source_concrete)

    for target_concrete, source_list in sources_by_target.items():
        if target_concrete == unembed_path:
            continue
        n_components = model.module_to_c[target_concrete]
        indices = list(range(n_components))
        rng.shuffle(indices)
        selected = indices[:n_targets_per_layer]

        for t_idx in selected:
            for source_concrete in source_list:
                pairs.append((target_concrete, t_idx, source_concrete))

    print(f"Probing {len(pairs)} (target_component, source_layer) pairs\n")

    all_stats: list[EdgeStats] = []
    for i, (target_concrete, t_idx, source_concrete) in enumerate(pairs):
        target_canon = to_canon(target_concrete)
        source_canon = "embed" if source_concrete == embed_path else to_canon(source_concrete)
        label = f"{target_canon}:{t_idx} <- {source_canon}"
        print(f"[{i+1}/{len(pairs)}] {label}")

        batch_results: list[dict[str, Tensor]] = []
        for tokens in batches:
            data = collect_per_position_grads(
                model, topology, sampling, sources_by_target, device,
                tokens, target_concrete, t_idx, source_concrete,
            )
            batch_results.append(data)

        merged = {"per_pos_attr": torch.cat([d["per_pos_attr"] for d in batch_results])}
        if "ci_weighted" in batch_results[0]:
            merged["ci_weighted"] = torch.cat([d["ci_weighted"] for d in batch_results])

        stats = compute_edge_stats(merged, label)
        all_stats.append(stats)

    _print_survey_table(all_stats)
    _plot_survey(all_stats)


def _print_survey_table(stats_list: list[EdgeStats]) -> None:
    print(f"\n{'='*110}")
    print(f"{'Edge':<40} {'N':>5} {'C_src':>5}  "
          f"{'|mean|/mabs':>11} {'|mean|/L2':>10}  "
          f"{'ci |m|/mabs':>11} {'ci |m|/L2':>10}")
    print(f"{'':<40} {'':>5} {'':>5}  "
          f"{'(med/mean)':>11} {'(med/mean)':>10}  "
          f"{'(med/mean)':>11} {'(med/mean)':>10}")
    print(f"{'-'*110}")

    for s in stats_list:
        ci_cancel = f"{s.ci_cancel_median:.2f}/{s.ci_cancel_mean:.2f}" if s.ci_cancel_median is not None else "—"
        ci_ml2 = f"{s.ci_mean_l2_median:.2f}/{s.ci_mean_l2_mean:.2f}" if s.ci_mean_l2_median is not None else "—"
        print(
            f"{s.edge_label:<40} {s.n_positions:>5} {s.n_source_components:>5}  "
            f"{s.raw_cancel_median:.2f}/{s.raw_cancel_mean:.2f}  "
            f"{s.raw_mean_l2_median:.2f}/{s.raw_mean_l2_mean:.2f}  "
            f"{ci_cancel:>11}  {ci_ml2:>10}"
        )

    print(f"{'='*110}")

    # Aggregate by edge type
    embed_stats = [s for s in stats_list if "<- embed" in s.edge_label]
    comp_stats = [s for s in stats_list if "<- embed" not in s.edge_label]

    print(f"\nAggregate (median of medians):")
    if embed_stats:
        raw_cancel = np.median([s.raw_cancel_median for s in embed_stats])
        raw_ml2 = np.median([s.raw_mean_l2_median for s in embed_stats])
        print(f"  embed→comp ({len(embed_stats)} edges): |mean|/mabs={raw_cancel:.3f}  |mean|/L2={raw_ml2:.3f}")
    if comp_stats:
        raw_cancel = np.median([s.raw_cancel_median for s in comp_stats])
        raw_ml2 = np.median([s.raw_mean_l2_median for s in comp_stats])
        ci_cancel_vals = [s.ci_cancel_median for s in comp_stats if s.ci_cancel_median is not None]
        ci_ml2_vals = [s.ci_mean_l2_median for s in comp_stats if s.ci_mean_l2_median is not None]
        ci_cancel = np.median(ci_cancel_vals) if ci_cancel_vals else float("nan")
        ci_ml2 = np.median(ci_ml2_vals) if ci_ml2_vals else float("nan")
        print(
            f"  comp→comp  ({len(comp_stats)} edges): "
            f"|mean|/mabs={raw_cancel:.3f}  |mean|/L2={raw_ml2:.3f}  "
            f"ci: |mean|/mabs={ci_cancel:.3f}  |mean|/L2={ci_ml2:.3f}"
        )


def _plot_survey(stats_list: list[EdgeStats]) -> None:
    embed_stats = [s for s in stats_list if "<- embed" in s.edge_label]
    comp_stats = [s for s in stats_list if "<- embed" not in s.edge_label]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter of |mean|/L2 for raw vs CI-weighted (comp→comp only)
    ax = axes[0]
    if comp_stats:
        raw_vals = [s.raw_mean_l2_median for s in comp_stats]
        ci_vals = [s.ci_mean_l2_median for s in comp_stats if s.ci_mean_l2_median is not None]
        ax.scatter(raw_vals[:len(ci_vals)], ci_vals, alpha=0.6, s=20)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("raw |mean|/L2 (median over sources)")
        ax.set_ylabel("CI-weighted |mean|/L2 (median over sources)")
        ax.set_title("comp→comp: CI reduces cancellation")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Right: histograms comparing |mean|/L2 distributions
    ax2 = axes[1]
    bins = np.logspace(-3, 0, 40)
    if embed_stats:
        ax2.hist(
            [s.raw_mean_l2_median for s in embed_stats],
            bins=bins, alpha=0.5, label=f"embed→comp raw (n={len(embed_stats)})",
        )
    if comp_stats:
        ax2.hist(
            [s.raw_mean_l2_median for s in comp_stats],
            bins=bins, alpha=0.5, label=f"comp→comp raw (n={len(comp_stats)})",
        )
        ci_meds = [s.ci_mean_l2_median for s in comp_stats if s.ci_mean_l2_median is not None]
        if ci_meds:
            ax2.hist(ci_meds, bins=bins, alpha=0.5, label=f"comp→comp CI (n={len(ci_meds)})")
    ax2.set_xscale("log")
    ax2.set_xlabel("|mean| / L2  (1=no cancellation)")
    ax2.set_ylabel("# edges")
    ax2.set_title("Cancellation across edge types")
    ax2.legend()

    plt.tight_layout()
    out_path = SPD_OUT_DIR / "www" / "attr_cancellation_survey.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved survey plot to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Single-edge mode (original interface)
# ---------------------------------------------------------------------------


def single(
    wandb_path: str,
    target_key: str = "0.mlp.up:5",
    source_layer: str = "embed",
    n_batches: int = 2,
    batch_size: int = 8,
    top_k: int = 6,
) -> None:
    """Detailed per-position analysis for a single (target, source) edge."""
    model, topology, sampling, sources_by_target, device = _setup(wandb_path)
    spd_config = SPDRunInfo.from_path(wandb_path).config
    train_loader, _ = train_loader_and_tokenizer(spd_config, batch_size)

    target_layer, target_idx_str = target_key.rsplit(":", 1)
    target_idx = int(target_idx_str)
    target_concrete = _resolve_concrete(target_layer, model, topology)
    source_concrete = _resolve_concrete(source_layer, model, topology)

    print(f"Collecting per-position gradients...")
    print(f"  target: {target_key} ({target_concrete})")
    print(f"  source: {source_layer} ({source_concrete})")
    print(f"  batches: {n_batches} × {batch_size}")

    batch_results: list[dict[str, Tensor]] = []
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        tokens = extract_batch_data(batch_data).to(device)
        data = collect_per_position_grads(
            model, topology, sampling, sources_by_target, device,
            tokens, target_concrete, target_idx, source_concrete,
        )
        batch_results.append(data)
        print(f"  batch {batch_idx}: {tokens.numel()} positions")

    merged = {"per_pos_attr": torch.cat([d["per_pos_attr"] for d in batch_results])}
    if "ci_weighted" in batch_results[0]:
        merged["ci_weighted"] = torch.cat([d["ci_weighted"] for d in batch_results])

    stats = compute_edge_stats(merged, f"{target_key} <- {source_layer}")
    _print_survey_table([stats])
    _plot_single(merged, target_key, source_layer, top_k)


def _plot_single(
    data: dict[str, Tensor],
    target_key: str,
    source_layer: str,
    top_k: int = 6,
) -> None:
    per_pos = data["per_pos_attr"]
    N, C = per_pos.shape

    abs_total = per_pos.float().abs().sum(dim=0)
    top_indices = abs_total.topk(min(top_k, C)).indices.tolist()
    n_cols = min(top_k, C)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes[:, None]

    fig.suptitle(
        f"Per-position grad×act distributions\n"
        f"target={target_key}, source={source_layer}, N={N} positions",
        fontsize=12,
    )

    for col, src_idx in enumerate(top_indices):
        vals = per_pos[:, src_idx].float().numpy()

        mean_val = np.mean(vals)
        l2_val = float(np.sqrt(np.mean(vals**2)))
        ratio = abs(mean_val) / l2_val if l2_val > 0 else 1.0

        ax = axes[0, col]
        ax.hist(vals, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3, log=True)
        ax.axvline(mean_val, color="red", linewidth=1.5, label=f"mean={mean_val:.2e}")
        ax.axvline(l2_val, color="green", linewidth=1.5, linestyle="--", label=f"L2={l2_val:.2e}")
        ax.axvline(-l2_val, color="green", linewidth=1.5, linestyle="--")
        ax.set_title(f"src {src_idx}\n|mean|/L2={ratio:.2%}")
        ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("grad × act")

        ax2 = axes[1, col]
        if "ci_weighted" in data:
            ci_vals = data["ci_weighted"][:, src_idx].float().numpy()
            ci_mean = np.mean(ci_vals)
            ci_l2 = float(np.sqrt(np.mean(ci_vals**2)))
            ci_ratio = abs(ci_mean) / ci_l2 if ci_l2 > 0 else 1.0
            ax2.hist(ci_vals, bins=80, alpha=0.7, color="orange", edgecolor="black", linewidth=0.3, log=True)
            ax2.axvline(ci_mean, color="red", linewidth=1.5)
            ax2.axvline(ci_l2, color="green", linewidth=1.5, linestyle="--")
            ax2.axvline(-ci_l2, color="green", linewidth=1.5, linestyle="--")
            ax2.set_title(f"CI-weighted\n|mean|/L2={ci_ratio:.2%}")
            if col == 0:
                ax2.set_ylabel("grad × act × ci")
        else:
            sorted_vals = np.sort(vals)
            cumsum = np.cumsum(sorted_vals)
            ax2.plot(range(len(cumsum)), cumsum)
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_title("Cumulative sum (sorted)")
            if col == 0:
                ax2.set_ylabel("cumulative attr")

    plt.tight_layout()
    out_path = SPD_OUT_DIR / "www" / "attr_cancellation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Scatter: per-source-component mean vs L2 for a given target
# ---------------------------------------------------------------------------


def scatter(
    wandb_path: str,
    target_key: str,
    n_batches: int = 10,
    batch_size: int = 4,
) -> None:
    """Scatter mean vs L2 of (grad×act×ci) for every source component of a target."""
    model, topology, sampling, sources_by_target, device = _setup(wandb_path)
    spd_config = SPDRunInfo.from_path(wandb_path).config
    embed_path = topology.path_schema.embedding_path
    to_canon = topology.target_to_canon

    target_layer, target_idx_str = target_key.rsplit(":", 1)
    target_idx = int(target_idx_str)
    target_concrete = _resolve_concrete(target_layer, model, topology)

    source_layers = sources_by_target[target_concrete]

    train_loader, _ = train_loader_and_tokenizer(spd_config, batch_size)
    batches: list[Tensor] = []
    for i, batch_data in enumerate(train_loader):
        if i >= n_batches:
            break
        batches.append(extract_batch_data(batch_data).to(device))

    n_positions = sum(b.numel() for b in batches)
    print(f"Target: {target_key} ({target_concrete})")
    print(f"Sources: {[('embed' if s == embed_path else to_canon(s)) for s in source_layers]}")
    print(f"Batches: {len(batches)}, positions: {n_positions}")

    # Collect per-source-layer results
    source_data: dict[str, dict[str, Tensor]] = {}  # canon_name -> {per_pos_attr, ci_weighted}
    for source_concrete in source_layers:
        source_canon = "embed" if source_concrete == embed_path else to_canon(source_concrete)
        print(f"  Processing {source_canon}...")

        batch_results: list[dict[str, Tensor]] = []
        for tokens in batches:
            data = collect_per_position_grads(
                model, topology, sampling, sources_by_target, device,
                tokens, target_concrete, target_idx, source_concrete,
            )
            batch_results.append(data)

        merged: dict[str, Tensor] = {
            "per_pos_attr": torch.cat([d["per_pos_attr"] for d in batch_results]),
        }
        if "ci_weighted" in batch_results[0]:
            merged["ci_weighted"] = torch.cat([d["ci_weighted"] for d in batch_results])
        source_data[source_canon] = merged

    _plot_scatter(source_data, target_key, n_positions)


def _plot_scatter(
    source_data: dict[str, dict[str, Tensor]],
    target_key: str,
    n_positions: int,
) -> None:
    has_ci = any("ci_weighted" in d for d in source_data.values())
    n_cols = 2 if has_ci else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    # Raw: |mean| vs L2 of grad×act
    ax = axes[0]
    for source_canon, data in source_data.items():
        per_pos = data["per_pos_attr"].float()  # (N, C_source)
        abs_means = per_pos.mean(dim=0).abs().numpy()
        l2s = per_pos.square().mean(dim=0).sqrt().numpy()
        ax.scatter(l2s, abs_means, alpha=0.4, s=8, label=source_canon)

    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("L2 = sqrt(mean((grad×act)²))")
    ax.set_ylabel("|mean(grad×act)|")
    ax.set_title(f"Raw grad×act\ntarget={target_key}, N={n_positions}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, markerscale=3)

    # CI-weighted: |mean| vs L2 of grad×act×ci
    if has_ci:
        ax2 = axes[1]
        for source_canon, data in source_data.items():
            if "ci_weighted" not in data:
                continue
            ci = data["ci_weighted"].float()  # (N, C_source)
            abs_means = ci.mean(dim=0).abs().numpy()
            l2s = ci.square().mean(dim=0).sqrt().numpy()
            ax2.scatter(l2s, abs_means, alpha=0.4, s=8, label=source_canon)

        lim2 = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([0, lim2], [0, lim2], "k--", alpha=0.2, linewidth=0.8)
        ax2.set_xlabel("L2 = sqrt(mean((grad×act×ci)²))")
        ax2.set_ylabel("|mean(grad×act×ci)|")
        ax2.set_title(f"CI-weighted\ntarget={target_key}, N={n_positions}")
        ax2.set_xlim(0, lim2)
        ax2.set_ylim(0, lim2)
        ax2.set_aspect("equal")
        ax2.legend(fontsize=8, markerscale=3)

    plt.tight_layout()
    safe_key = target_key.replace(":", "_").replace(".", "_")
    out_path = SPD_OUT_DIR / "www" / f"attr_scatter_{safe_key}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Compare: |mean(attr)| vs L2(attr) histogram across all edges
# ---------------------------------------------------------------------------


def compare(
    wandb_path: str,
    n_batches: int = 10,
    batch_size: int = 4,
    n_targets_per_layer: int = 3,
    seed: int = 42,
) -> None:
    """Histogram of |mean(attr)| vs L2(attr) per source component, pooled across many targets."""
    model, topology, sampling, sources_by_target, device = _setup(wandb_path)
    spd_config = SPDRunInfo.from_path(wandb_path).config
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    to_canon = topology.target_to_canon

    train_loader, _ = train_loader_and_tokenizer(spd_config, batch_size)
    batches: list[Tensor] = []
    for i, batch_data in enumerate(train_loader):
        if i >= n_batches:
            break
        batches.append(extract_batch_data(batch_data).to(device))
    n_positions = sum(b.numel() for b in batches)
    print(f"Loaded {len(batches)} batches, {n_positions} tokens")

    rng = random.Random(seed)
    pairs: list[tuple[str, int, str]] = []
    for target_concrete, source_list in sources_by_target.items():
        if target_concrete == unembed_path:
            continue
        n_components = model.module_to_c[target_concrete]
        indices = list(range(n_components))
        rng.shuffle(indices)
        for t_idx in indices[:n_targets_per_layer]:
            for source_concrete in source_list:
                pairs.append((target_concrete, t_idx, source_concrete))

    print(f"Probing {len(pairs)} edges\n")

    # Collect per-component |mean| and L2 across all edges
    raw_abs_means: list[Tensor] = []
    raw_l2s: list[Tensor] = []
    ci_abs_means: list[Tensor] = []
    ci_l2s: list[Tensor] = []
    edge_types: list[list[str]] = [[], []]  # [raw_types, ci_types]

    for i, (target_concrete, t_idx, source_concrete) in enumerate(pairs):
        source_canon = "embed" if source_concrete == embed_path else to_canon(source_concrete)
        target_canon = to_canon(target_concrete)
        print(f"[{i+1}/{len(pairs)}] {target_canon}:{t_idx} <- {source_canon}")

        batch_results: list[dict[str, Tensor]] = []
        for tokens in batches:
            data = collect_per_position_grads(
                model, topology, sampling, sources_by_target, device,
                tokens, target_concrete, t_idx, source_concrete,
            )
            batch_results.append(data)

        per_pos = torch.cat([d["per_pos_attr"] for d in batch_results]).float()
        C = per_pos.shape[1]
        is_embed = source_concrete == embed_path
        edge_type = "embed→comp" if is_embed else "comp→comp"

        raw_abs_means.append(per_pos.mean(dim=0).abs())
        raw_l2s.append(per_pos.square().mean(dim=0).sqrt())
        edge_types[0].extend([edge_type] * C)

        if "ci_weighted" in batch_results[0]:
            ci = torch.cat([d["ci_weighted"] for d in batch_results]).float()
            ci_abs_means.append(ci.mean(dim=0).abs())
            ci_l2s.append(ci.square().mean(dim=0).sqrt())
            edge_types[1].extend([edge_type] * C)

    all_raw_abs_mean = torch.cat(raw_abs_means).numpy()
    all_raw_l2 = torch.cat(raw_l2s).numpy()
    all_ci_abs_mean = torch.cat(ci_abs_means).numpy() if ci_abs_means else None
    all_ci_l2 = torch.cat(ci_l2s).numpy() if ci_l2s else None

    _plot_compare(
        all_raw_abs_mean, all_raw_l2, edge_types[0],
        all_ci_abs_mean, all_ci_l2, edge_types[1],
        n_positions, len(pairs),
    )


def _plot_compare(
    raw_abs_mean: "np.ndarray[Any, Any]",
    raw_l2: "np.ndarray[Any, Any]",
    raw_types: list[str],
    ci_abs_mean: "np.ndarray[Any, Any] | None",
    ci_l2: "np.ndarray[Any, Any] | None",
    ci_types: list[str],
    n_positions: int,
    n_edges: int,
) -> None:
    has_ci = ci_abs_mean is not None
    n_cols = 2 if has_ci else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    raw_types_arr = np.array(raw_types)

    # Left: raw
    ax = axes[0]
    for etype, color in [("embed→comp", "C0"), ("comp→comp", "C1")]:
        mask = raw_types_arr == etype
        if not mask.any():
            continue
        ax.scatter(raw_l2[mask], raw_abs_mean[mask], alpha=0.15, s=4, c=color, label=etype)
    lim = max(raw_l2.max(), raw_abs_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("L2(grad×act)")
    ax.set_ylabel("|mean(grad×act)|")
    ax.set_title(f"Raw: |mean| vs L2\n{n_edges} edges, {n_positions} pos/edge")
    ax.legend(markerscale=5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    # Right: CI-weighted
    if has_ci:
        assert ci_abs_mean is not None and ci_l2 is not None
        ci_types_arr = np.array(ci_types)
        ax2 = axes[1]
        for etype, color in [("embed→comp", "C0"), ("comp→comp", "C1")]:
            mask = ci_types_arr == etype
            if not mask.any():
                continue
            ax2.scatter(ci_l2[mask], ci_abs_mean[mask], alpha=0.15, s=4, c=color, label=etype)
        lim2 = max(ci_l2.max(), ci_abs_mean.max()) * 1.05
        ax2.plot([0, lim2], [0, lim2], "k--", alpha=0.3, linewidth=0.8)
        ax2.set_xlabel("L2(grad×act×ci)")
        ax2.set_ylabel("|mean(grad×act×ci)|")
        ax2.set_title(f"CI-weighted: |mean| vs L2\n{n_edges} edges, {n_positions} pos/edge")
        ax2.legend(markerscale=5)
        ax2.set_xlim(0, lim2)
        ax2.set_ylim(0, lim2)
        ax2.set_aspect("equal")

    plt.tight_layout()
    out_path = SPD_OUT_DIR / "www" / "attr_mean_vs_l2.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Rankings: per-target |mean| vs L2 scatter + rank comparison
# ---------------------------------------------------------------------------


def rankings(
    wandb_path: str,
    n_targets: int = 10,
    n_batches: int = 10,
    batch_size: int = 4,
    seed: int = 42,
) -> None:
    """For random targets: scatter |mean| vs L2 + rank-rank comparison across all sources."""
    model, topology, sampling, sources_by_target, device = _setup(wandb_path)
    spd_config = SPDRunInfo.from_path(wandb_path).config
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    to_canon = topology.target_to_canon

    train_loader, _ = train_loader_and_tokenizer(spd_config, batch_size)
    batches: list[Tensor] = []
    for i, batch_data in enumerate(train_loader):
        if i >= n_batches:
            break
        batches.append(extract_batch_data(batch_data).to(device))
    n_positions = sum(b.numel() for b in batches)
    print(f"Loaded {len(batches)} batches, {n_positions} tokens")

    # Pick random targets (excluding unembed)
    rng = random.Random(seed)
    all_targets: list[tuple[str, int]] = []
    for target_concrete in sources_by_target:
        if target_concrete == unembed_path:
            continue
        for c_idx in range(model.module_to_c[target_concrete]):
            all_targets.append((target_concrete, c_idx))
    rng.shuffle(all_targets)
    selected = all_targets[:n_targets]

    fig, axes = plt.subplots(n_targets, 2, figsize=(14, 4 * n_targets))
    if n_targets == 1:
        axes = axes[None, :]

    for row, (target_concrete, t_idx) in enumerate(selected):
        target_canon = to_canon(target_concrete)
        target_label = f"{target_canon}:{t_idx}"
        source_list = sources_by_target[target_concrete]
        print(f"[{row+1}/{n_targets}] {target_label} ({len(source_list)} source layers)")

        # Collect per-position grads for all sources of this target
        all_abs_means: list[float] = []
        all_l2s: list[float] = []
        all_labels: list[str] = []
        all_colors: list[str] = []

        for source_concrete in source_list:
            source_canon = "embed" if source_concrete == embed_path else to_canon(source_concrete)
            is_embed = source_concrete == embed_path

            batch_results: list[dict[str, Tensor]] = []
            for tokens in batches:
                data = collect_per_position_grads(
                    model, topology, sampling, sources_by_target, device,
                    tokens, target_concrete, t_idx, source_concrete,
                )
                batch_results.append(data)

            per_pos = torch.cat([d["per_pos_attr"] for d in batch_results]).float()
            abs_means = per_pos.mean(dim=0).abs()
            l2s = per_pos.square().mean(dim=0).sqrt()

            C = per_pos.shape[1]
            all_abs_means.extend(abs_means.tolist())
            all_l2s.extend(l2s.tolist())
            all_labels.extend([f"{source_canon}:{i}" for i in range(C)])
            all_colors.extend(["C0" if is_embed else "C1"] * C)

        abs_means_arr = np.array(all_abs_means)
        l2s_arr = np.array(all_l2s)
        colors_arr = np.array(all_colors)

        # Left: |mean| vs L2 scatter
        ax = axes[row, 0]
        for color, label in [("C0", "embed"), ("C1", "comp")]:
            mask = colors_arr == color
            if mask.any():
                ax.scatter(l2s_arr[mask], abs_means_arr[mask], alpha=0.3, s=6, c=color, label=label)
        lim = max(l2s_arr.max(), abs_means_arr.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("L2")
        ax.set_ylabel("|mean|")
        ax.set_title(f"{target_label}: |mean| vs L2")
        ax.legend(fontsize=7, markerscale=3)

        # Right: rank-rank plot
        ax2 = axes[row, 1]
        rank_by_mean = np.argsort(np.argsort(-abs_means_arr))  # 0 = highest
        rank_by_l2 = np.argsort(np.argsort(-l2s_arr))
        n_total = len(abs_means_arr)

        for color, label in [("C0", "embed"), ("C1", "comp")]:
            mask = colors_arr == color
            if mask.any():
                ax2.scatter(rank_by_l2[mask], rank_by_mean[mask], alpha=0.15, s=4, c=color, label=label)
        ax2.plot([0, n_total], [0, n_total], "k--", alpha=0.2, linewidth=0.8)
        ax2.set_xlim(0, n_total)
        ax2.set_ylim(0, n_total)
        ax2.set_aspect("equal")
        ax2.set_xlabel("rank by L2 (0=highest)")
        ax2.set_ylabel("rank by |mean| (0=highest)")

        # Compute top-K overlap
        for top_k in [10, 50]:
            top_mean = set(np.argsort(-abs_means_arr)[:top_k])
            top_l2 = set(np.argsort(-l2s_arr)[:top_k])
            overlap = len(top_mean & top_l2)
            ax2.text(
                0.98, 0.02 + (0.06 if top_k == 50 else 0),
                f"top-{top_k} overlap: {overlap}/{top_k}",
                transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
            )

        spearman = float(np.corrcoef(rank_by_mean, rank_by_l2)[0, 1])
        ax2.set_title(f"{target_label}: rank comparison (ρ={spearman:.3f})")
        ax2.legend(fontsize=7, markerscale=3)

    plt.tight_layout()
    out_path = SPD_OUT_DIR / "www" / "attr_rankings.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire({
        "single": single,
        "survey": survey,
        "scatter": scatter,
        "compare": compare,
        "rankings": rankings,
    })
