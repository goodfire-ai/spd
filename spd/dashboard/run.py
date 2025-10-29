"""Minimal two-phase pipeline for generating component dashboard data.

Phase 1: Load model, generate activations, delete model
Phase 2: Compute global metrics and per-component stats, save with ZANJ
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from muutils.spinner import SpinnerContext
from sklearn.manifold import Isomap
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from zanj import ZANJ

from spd.configs import Config
from spd.dashboard.activations import component_activations, process_activations
from spd.dashboard.core.component_data import (
    ComponentDashboardData,
    GlobalMetrics,
    RawActivationData,
    SubcomponentStats,
    TopKSample,
)
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import extract_batch_data, get_obj_device

# ============================================================================
# PHASE 1: DATA GENERATION
# ============================================================================


def generate_activations(
    model_path: str,
    dataset_config: DatasetConfig,
    batch_size: int,
    n_batches: int,
    dead_threshold: float,
) -> RawActivationData:
    """Phase 1: Generate raw activations and delete model.

    Args:
        model_path: Path to SPD model
        dataset_config: Dataset configuration
        batch_size: Batch size for data loading
        n_batches: Number of batches to process
        dead_threshold: Components with max activation <= this are considered dead

    Returns:
        RawActivationData with tokens, activations, and component info
    """
    logger.info("Loading model and data...")

    # Load model
    spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
    model: ComponentModel = ComponentModel.from_run_info(spd_run)
    device: torch.device = get_obj_device(model)
    model.eval()

    # Load dataloader
    dataloader: DataLoader[Any]
    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=4,
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Accumulate activations
    all_tokens: list[Float[np.ndarray, "batch n_ctx"]] = []
    all_activations: dict[str, list[Float[np.ndarray, "batch n_ctx"]]] = defaultdict(list)

    logger.info(f"Processing {n_batches} batches...")
    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches, desc="Batches")):
        if batch_idx >= n_batches:
            break

        # Extract tokens
        batch: Int[Tensor, "batch n_ctx"] = extract_batch_data(batch_data).to(device)

        # Get component activations (already flattened to [batch*n_ctx, C])
        acts_dict: dict[str, Float[Tensor, "batch*n_ctx C"]] = component_activations(
            model, device, batch=batch
        )

        # Process activations (concatenate all modules)
        processed = process_activations(acts_dict)

        # Reshape to [batch, n_ctx, n_components]
        batch_size_actual: int = batch.shape[0]
        n_ctx: int = batch.shape[1]
        n_components: int = processed.n_components

        acts_3d: Float[Tensor, "batch n_ctx n_components"] = processed.activations.view(
            batch_size_actual, n_ctx, n_components
        )

        # Store tokens
        all_tokens.append(batch.cpu().numpy())

        # Store activations per component (using string labels)
        for comp_idx, comp_label_str in enumerate(processed.label_strings):
            comp_acts: Float[Tensor, "batch n_ctx"] = acts_3d[:, :, comp_idx]
            all_activations[comp_label_str].append(comp_acts.cpu().numpy())

    # CRITICAL: Delete model to free memory
    logger.info("Deleting model to free memory...")
    del model
    del spd_run
    torch.cuda.empty_cache()

    # Concatenate all batches
    logger.info("Concatenating batches...")
    tokens: Float[np.ndarray, "n_samples n_ctx"] = np.concatenate(all_tokens, axis=0)
    activations: dict[str, Float[np.ndarray, "n_samples n_ctx"]] = {
        label: np.concatenate(acts_list, axis=0) for label, acts_list in all_activations.items()
    }

    # Filter dead components
    component_labels: list[str] = sorted(activations.keys())
    dead_components: set[str] = {
        label for label, acts in activations.items() if acts.max() <= dead_threshold
    }
    alive_components: list[str] = [
        label for label in component_labels if label not in dead_components
    ]

    logger.info(
        f"Components: {len(alive_components)} alive, {len(dead_components)} dead "
        f"(threshold={dead_threshold})"
    )

    return RawActivationData(
        tokens=tokens,
        activations=activations,
        component_labels=component_labels,
        alive_components=alive_components,
        dead_components=dead_components,
    )


# ============================================================================
# PHASE 2A: GLOBAL METRICS
# ============================================================================


def compute_coactivations(
    activations: dict[str, Float[np.ndarray, "n_samples n_ctx"]],
    component_labels: list[str],
) -> Float[np.ndarray, "n_components n_components"]:
    """Compute binary coactivation matrix.

    Args:
        activations: Dict mapping component labels to activation arrays
        component_labels: Ordered list of components to include

    Returns:
        Coactivation matrix counting positions where both components activate
    """
    # Stack: [n_total_positions, n_components]
    stacked: Float[np.ndarray, "n_total n_components"] = np.stack(
        [activations[label].flatten() for label in component_labels], axis=1
    )

    # Binarize (1 if active, 0 otherwise)
    binary: Float[np.ndarray, "n_total n_components"] = (stacked > 0).astype(np.float32)

    # Coactivation matrix: binary.T @ binary
    coact: Float[np.ndarray, "n_components n_components"] = binary.T @ binary

    return coact


def compute_correlations(
    activations: dict[str, Float[np.ndarray, "n_samples n_ctx"]],
    component_labels: list[str],
) -> Float[np.ndarray, "n_components n_components"]:
    """Compute Pearson correlation matrix.

    Args:
        activations: Dict mapping component labels to activation arrays
        component_labels: Ordered list of components to include

    Returns:
        Correlation matrix
    """
    # Stack: [n_total_positions, n_components]
    stacked: Float[np.ndarray, "n_total n_components"] = np.stack(
        [activations[label].flatten() for label in component_labels], axis=1
    )

    # Pearson correlation
    correlations: Float[np.ndarray, "n_components n_components"] = np.corrcoef(stacked.T)

    return correlations


def compute_embeddings(
    coactivations: Float[np.ndarray, "n_components n_components"],
    correlations: Float[np.ndarray, "n_components n_components"],
    n_components: int,
) -> Float[np.ndarray, "n_components embed_dim"]:
    """Compute Isomap embeddings from affinity matrix.

    Args:
        coactivations: Coactivation count matrix
        correlations: Correlation matrix
        n_components: Embedding dimensionality

    Returns:
        Embedding coordinates for each component
    """
    # Combine coactivations and absolute correlations for affinity
    affinity: Float[np.ndarray, "n_comp n_comp"] = coactivations + np.abs(correlations)

    # Convert to distance (higher affinity = lower distance)
    max_affinity: float = float(affinity.max())
    distance: Float[np.ndarray, "n_comp n_comp"] = max_affinity - affinity

    # Ensure diagonal is zero and matrix is symmetric
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0

    # Isomap embedding with precomputed distances
    isomap: Isomap = Isomap(n_components=n_components, metric="precomputed")
    embeddings: Float[np.ndarray, "n_comp embed_dim"] = isomap.fit_transform(distance)

    return embeddings


def compute_global_metrics(
    raw_data: RawActivationData,
    embed_dim: int,
) -> GlobalMetrics:
    """Compute all global metrics for alive components.

    Args:
        raw_data: Raw activation data
        embed_dim: Dimensionality of embeddings

    Returns:
        GlobalMetrics with coactivations, correlations, and embeddings
    """
    # Extract alive component activations
    alive_acts: dict[str, Float[np.ndarray, "n_samples n_ctx"]] = {
        label: raw_data.activations[label] for label in raw_data.alive_components
    }

    logger.info("Computing coactivations...")
    coact: Float[np.ndarray, "n_alive n_alive"] = compute_coactivations(
        alive_acts, raw_data.alive_components
    )

    logger.info("Computing correlations...")
    corr: Float[np.ndarray, "n_alive n_alive"] = compute_correlations(
        alive_acts, raw_data.alive_components
    )

    logger.info(f"Computing {embed_dim}D embeddings with Isomap...")
    embeds: Float[np.ndarray, "n_alive embed_dim"] = compute_embeddings(
        coact, corr, n_components=embed_dim
    )

    return GlobalMetrics(
        coactivations=coact,
        correlations=corr,
        embeddings=embeds,
        component_labels=raw_data.alive_components,
    )


# ============================================================================
# PHASE 2B: PER-COMPONENT STATS
# ============================================================================


def find_top_k_samples(
    activations: Float[np.ndarray, "n_samples n_ctx"],
    tokens: Float[np.ndarray, "n_samples n_ctx"],
    tokenizer: Any,
    k: int,
    criterion: str,
) -> list[TopKSample]:
    """Find top-k samples by specified criterion.

    Args:
        activations: Activation values for this component
        tokens: Token IDs for all samples
        tokenizer: Tokenizer to decode token IDs to strings
        k: Number of top samples to return
        criterion: 'max' or 'mean'

    Returns:
        List of TopKSample objects, sorted by score (highest first)
    """
    # Compute scores based on criterion
    if criterion == "max":
        scores: Float[np.ndarray, " n_samples"] = activations.max(axis=1)
    elif criterion == "mean":
        scores: Float[np.ndarray, " n_samples"] = activations.mean(axis=1)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Get top-k indices (sorted descending)
    top_k_idx: Float[np.ndarray, " k"] = np.argpartition(scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])][::-1]

    # Build TopKSample objects
    samples: list[TopKSample] = []
    for idx in top_k_idx:
        idx_int: int = int(idx)
        token_ids: list[int] = tokens[idx_int].tolist()

        # Decode tokens to strings
        token_strs: list[str] = [tokenizer.decode([tid]) for tid in token_ids]

        samples.append(
            TopKSample(  # pyright: ignore[reportCallIssue]
                token_strs=token_strs,
                activations=activations[idx_int],
            )
        )

    return samples


def compute_global_statistics(
    activations: Float[np.ndarray, "n_samples n_ctx"],
) -> dict[str, float]:
    """Compute global distribution statistics.

    Args:
        activations: Activation values for this component

    Returns:
        Dict with mean, std, min, max, median, and quantiles
    """
    flat: Float[np.ndarray, " n_total"] = activations.flatten()

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q25": float(np.quantile(flat, 0.25)),
        "q75": float(np.quantile(flat, 0.75)),
        "q95": float(np.quantile(flat, 0.95)),
    }


def compute_histograms(
    activations: Float[np.ndarray, "n_samples n_ctx"],
    bins: int,
) -> dict[str, dict[str, list]]:
    """Compute multiple histograms for different views.

    Args:
        activations: Activation values for this component
        bins: Number of bins for histograms

    Returns:
        Dict mapping histogram name to {counts, edges}
    """
    histograms: dict[str, dict[str, list]] = {}

    # Histogram 1: All activation magnitudes
    flat: Float[np.ndarray, " n_total"] = activations.flatten()
    counts_all: Float[np.ndarray, " bins"]
    edges_all: Float[np.ndarray, " bins_plus_1"]
    counts_all, edges_all = np.histogram(flat, bins=bins)
    histograms["all_activations"] = {
        "counts": counts_all.tolist(),
        "edges": edges_all.tolist(),
    }

    # Histogram 2: Max activation per sample
    max_per_sample: Float[np.ndarray, " n_samples"] = activations.max(axis=1)
    counts_max: Float[np.ndarray, " bins"]
    edges_max: Float[np.ndarray, " bins_plus_1"]
    counts_max, edges_max = np.histogram(max_per_sample, bins=bins)
    histograms["max_per_sample"] = {
        "counts": counts_max.tolist(),
        "edges": edges_max.tolist(),
    }

    # Histogram 3: Mean activation per sample
    mean_per_sample: Float[np.ndarray, " n_samples"] = activations.mean(axis=1)
    counts_mean: Float[np.ndarray, " bins"]
    edges_mean: Float[np.ndarray, " bins_plus_1"]
    counts_mean, edges_mean = np.histogram(mean_per_sample, bins=bins)
    histograms["mean_per_sample"] = {
        "counts": counts_mean.tolist(),
        "edges": edges_mean.tolist(),
    }

    return histograms


def compute_component_stats(
    label: str,
    activations: Float[np.ndarray, "n_samples n_ctx"],
    tokens: Float[np.ndarray, "n_samples n_ctx"],
    tokenizer: Any,
    global_metrics: GlobalMetrics | None,
    k: int,
    hist_bins: int,
) -> SubcomponentStats:
    """Compute all statistics for a single component.

    Args:
        label: Component label
        activations: Activation values
        tokens: Token IDs
        tokenizer: Tokenizer to decode token IDs
        global_metrics: Global metrics (None if component is dead)
        k: Number of top samples to track
        hist_bins: Number of histogram bins

    Returns:
        SubcomponentStats with all computed statistics
    """
    is_dead: bool = global_metrics is None

    if is_dead:
        # Dead component - minimal stats
        return SubcomponentStats(  # pyright: ignore[reportCallIssue]
            label=label,
            is_dead=True,
            embedding=None,
            top_max=[],
            top_mean=[],
            stats={},
            histograms={},
        )

    # Alive component - full stats
    return SubcomponentStats(  # pyright: ignore[reportCallIssue]
        label=label,
        is_dead=False,
        embedding=global_metrics.get_embedding(label),
        top_max=find_top_k_samples(activations, tokens, tokenizer, k, "max"),
        top_mean=find_top_k_samples(activations, tokens, tokenizer, k, "mean"),
        stats=compute_global_statistics(activations),
        histograms=compute_histograms(activations, bins=hist_bins),
    )


def process_all_components(
    raw_data: RawActivationData,
    tokenizer: Any,
    global_metrics: GlobalMetrics,
    k: int,
    hist_bins: int,
) -> list[SubcomponentStats]:
    """Compute stats for all components (alive and dead).

    Args:
        raw_data: Raw activation data
        tokenizer: Tokenizer to decode token IDs
        global_metrics: Precomputed global metrics for alive components
        k: Number of top samples per component
        hist_bins: Number of histogram bins

    Returns:
        List of SubcomponentStats for all components
    """
    all_stats: list[SubcomponentStats] = []

    for label in tqdm(raw_data.component_labels, desc="Processing components"):
        is_alive: bool = label in raw_data.alive_components

        stats: SubcomponentStats = compute_component_stats(
            label=label,
            activations=raw_data.activations[label],
            tokens=raw_data.tokens,
            tokenizer=tokenizer,
            global_metrics=global_metrics if is_alive else None,
            k=k,
            hist_bins=hist_bins,
        )
        all_stats.append(stats)

    return all_stats


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main(
    model_path: str,
    dataset_config: DatasetConfig,
    output_dir: Path,
    tokenizer_name: str,
    batch_size: int,
    n_batches: int,
    k: int,
    dead_threshold: float,
    embed_dim: int,
    hist_bins: int,
) -> None:
    """Complete two-phase pipeline for component dashboard generation.

    Args:
        model_path: Path to SPD model
        dataset_config: Dataset configuration
        output_dir: Output directory for results
        tokenizer_name: Name of tokenizer to use for decoding
        batch_size: Batch size for data loading
        n_batches: Number of batches to process
        k: Number of top samples per component
        dead_threshold: Threshold for considering components dead
        embed_dim: Embedding dimensionality
        hist_bins: Number of histogram bins
    """
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_name)

    # ========================================================================
    # PHASE 1: DATA GENERATION
    # ========================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA GENERATION")
    logger.info("=" * 70)

    raw_data: RawActivationData = generate_activations(
        model_path=model_path,
        dataset_config=dataset_config,
        batch_size=batch_size,
        n_batches=n_batches,
        dead_threshold=dead_threshold,
    )

    logger.info(f"\nGenerated: {raw_data.n_samples} samples × {raw_data.n_ctx} context")
    logger.info(
        f"Components: {len(raw_data.alive_components)} alive, {len(raw_data.dead_components)} dead"
    )

    # ========================================================================
    # PHASE 2: PROCESSING - GLOBAL METRICS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PROCESSING - GLOBAL METRICS")
    logger.info("=" * 70)

    with SpinnerContext(message="Computing global metrics"):
        global_metrics: GlobalMetrics = compute_global_metrics(raw_data, embed_dim=embed_dim)

    # Save matrices separately for easy access
    metrics_path: Path = output_dir / "global_metrics.npz"
    np.savez(
        metrics_path,
        coactivations=global_metrics.coactivations,
        correlations=global_metrics.correlations,
        embeddings=global_metrics.embeddings,
        labels=global_metrics.component_labels,
    )
    logger.info(f"Saved {metrics_path}: {global_metrics.coactivations.shape}")

    # ========================================================================
    # PHASE 2: PROCESSING - COMPONENT STATS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PROCESSING - COMPONENT STATS")
    logger.info("=" * 70)

    all_stats: list[SubcomponentStats] = process_all_components(
        raw_data=raw_data,
        tokenizer=tokenizer,
        global_metrics=global_metrics,
        k=k,
        hist_bins=hist_bins,
    )

    # ========================================================================
    # SAVE EVERYTHING
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SAVING DASHBOARD DATA")
    logger.info("=" * 70)

    dashboard: ComponentDashboardData = ComponentDashboardData(  # pyright: ignore[reportCallIssue]
        model_path=model_path,
        n_samples=raw_data.n_samples,
        n_ctx=raw_data.n_ctx,
        n_components=len(raw_data.component_labels),
        n_alive=len(raw_data.alive_components),
        n_dead=len(raw_data.dead_components),
        global_metrics=global_metrics,
        components=all_stats,
    )

    zanj_path: Path = output_dir / "dashboard.zanj"
    ZANJ().save(dashboard.serialize(), str(zanj_path))
    logger.info(f"Saved {zanj_path}")
    logger.info("Dashboard generation complete!")


def cli() -> None:
    """CLI entry point with argument parsing."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate component dashboard data from SPD model"
    )
    parser.add_argument("config", type=Path, help="Path to dashboard config file (JSON or YAML)")
    args: argparse.Namespace = parser.parse_args()

    # Import config class
    from spd.dashboard.core.component_dashboard_config import ComponentDashboardConfig

    config: ComponentDashboardConfig = ComponentDashboardConfig.from_file(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # Load SPD config to get tokenizer name
    spd_run: SPDRunInfo = SPDRunInfo.from_path(config.model_path)
    spd_config: Config = spd_run.config
    tokenizer_name: str = spd_config.tokenizer_name  # pyright: ignore[reportAssignmentType]

    # Create dataset config
    dataset_config: DatasetConfig = DatasetConfig(
        name=config.dataset_name,
        hf_tokenizer_path=tokenizer_name,
        split=config.dataset_split,
        n_ctx=config.context_length,
        is_tokenized=False,
        streaming=config.dataset_streaming,
        column_name=config.dataset_column,
    )

    # Run main pipeline
    main(
        model_path=config.model_path,
        dataset_config=dataset_config,
        output_dir=config.output_dir,
        tokenizer_name=tokenizer_name,
        batch_size=config.batch_size,
        n_batches=config.n_batches,
        k=config.n_samples,
        dead_threshold=config.dead_threshold,
        embed_dim=config.embed_dim,
        hist_bins=config.hist_bins,
    )


if __name__ == "__main__":
    cli()
