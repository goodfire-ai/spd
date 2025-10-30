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
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from zanj import ZANJ

from spd.configs import Config
from spd.dashboard.core.activations import (
    ProcessedActivations,
    SubcomponentLabel,
    component_activations,
    process_activations,
)
from spd.dashboard.core.component_data import (
    ComponentDashboardData,
    GlobalMetrics,
    RawActivationData,
    SubcomponentStats,
)
from spd.data import DatasetConfig, create_data_loader
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
    all_activations: dict[SubcomponentLabel, list[Float[np.ndarray, "batch n_ctx"]]] = defaultdict(
        list
    )

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches, desc="Batches")):
        if batch_idx >= n_batches:
            break

        # Extract tokens
        batch: Int[Tensor, "batch n_ctx"] = extract_batch_data(batch_item=batch_data).to(device)

        # Get component activations
        acts_dict: dict[str, Float[Tensor, "batch n_ctx C"]] = component_activations(
            model, device, batch=batch
        )
        acts_dict_concat: dict[str, Float[Tensor, "batch*n_ctx C"]] = {
            module: acts.view(-1, acts.shape[-1]) for module, acts in acts_dict.items()
        }

        # Process activations (concatenate all modules)
        processed: ProcessedActivations = process_activations(activations=acts_dict_concat)

        # Reshape to [batch, n_ctx, n_components]
        batch_size_actual: int = batch.shape[0]
        n_ctx: int = batch.shape[1]
        n_components: int = processed.n_components

        acts_3d: Float[Tensor, "batch n_ctx n_components"] = processed.activations.view(
            batch_size_actual, n_ctx, n_components
        )

        # Store tokens
        all_tokens.append(batch.cpu().numpy())

        # Store activations per component (using SubcomponentLabel objects)
        for comp_idx, comp_label in enumerate(processed.labels):
            comp_acts: Float[Tensor, "batch n_ctx"] = acts_3d[:, :, comp_idx]
            all_activations[comp_label].append(comp_acts.cpu().numpy())

    # CRITICAL: Delete model to free memory
    del model
    del spd_run
    torch.cuda.empty_cache()

    # Concatenate all batches
    tokens: Float[np.ndarray, "n_samples n_ctx"] = np.concatenate(all_tokens, axis=0)
    activations: dict[SubcomponentLabel, Float[np.ndarray, "n_samples n_ctx"]] = {
        label: np.concatenate(acts_list, axis=0) for label, acts_list in all_activations.items()
    }

    # Filter dead components
    component_labels: list[SubcomponentLabel] = sorted(
        activations.keys(), key=lambda x: x.to_string()
    )
    dead_components: set[SubcomponentLabel] = {
        label for label, acts in activations.items() if acts.max() <= dead_threshold
    }
    alive_components: list[SubcomponentLabel] = [
        label for label in component_labels if label not in dead_components
    ]

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
    return GlobalMetrics.generate(
        activations=raw_data.activations,
        component_labels=raw_data.alive_components,
        embed_dim=embed_dim,
    )


# ============================================================================
# PHASE 2B: PER-COMPONENT STATS
# ============================================================================


def process_all_components(
    raw_data: RawActivationData,
    tokenizer: Any,
    global_metrics: GlobalMetrics,
    k: int,
    hist_bins: int,
) -> list[SubcomponentStats]:
    """Compute stats for alive components only.

    Args:
        raw_data: Raw activation data
        tokenizer: Tokenizer to decode token IDs
        global_metrics: Precomputed global metrics for alive components
        k: Number of top samples per component
        hist_bins: Number of histogram bins

    Returns:
        List of SubcomponentStats for alive components
    """
    all_stats: list[SubcomponentStats] = []

    for label in tqdm(raw_data.alive_components, desc="Processing components"):
        stats: SubcomponentStats = SubcomponentStats.generate(
            label=label,
            activations=raw_data.activations[label],
            tokens=raw_data.tokens,
            tokenizer=tokenizer,
            global_metrics=global_metrics,
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

    from spd.dashboard.core.tokenization import attach_vocab_arr

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and attach vocab array for fast decoding
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_name)
    attach_vocab_arr(tokenizer)

    # PHASE 1: DATA GENERATION
    raw_data: RawActivationData = generate_activations(
        model_path=model_path,
        dataset_config=dataset_config,
        batch_size=batch_size,
        n_batches=n_batches,
        dead_threshold=dead_threshold,
    )

    # PHASE 2: PROCESSING - GLOBAL METRICS
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

    # PHASE 2: PROCESSING - COMPONENT STATS
    all_stats: list[SubcomponentStats] = process_all_components(
        raw_data=raw_data,
        tokenizer=tokenizer,
        global_metrics=global_metrics,
        k=k,
        hist_bins=hist_bins,
    )

    # SAVE EVERYTHING
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
    ZANJ().save(dashboard, str(zanj_path))
    print(f"Saved dashboard data to '{zanj_path}'")


def cli() -> None:
    """CLI entry point with argument parsing."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate component dashboard data from SPD model"
    )
    parser.add_argument("config", type=Path, help="Path to dashboard config file (JSON or YAML)")
    args: argparse.Namespace = parser.parse_args()

    # Import config class
    from spd.dashboard.core.dashboard_config import ComponentDashboardConfig

    config: ComponentDashboardConfig = ComponentDashboardConfig.from_file(args.config)

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
