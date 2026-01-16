"""Dataset attribution harvesting.

Computes component-to-component attribution strengths aggregated over the full
training dataset. Unlike prompt attributions (single-prompt, position-aware),
dataset attributions answer: "In aggregate, which components typically influence
each other?"

Multi-GPU usage:
    # Launch workers (any orchestration: shell, SLURM, tmux, etc.)
    spd-attributions <path> --n_batches 1000 --rank 0 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 1 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 2 --world_size 4 &
    spd-attributions <path> --n_batches 1000 --rank 3 --world_size 4 &
    wait

    # Merge results
    spd-attributions <path> --merge
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
from jaxtyping import Bool
from torch import Tensor

from spd.app.backend.compute import get_sources_by_target
from spd.dataset_attributions.harvester import AttributionHarvester
from spd.dataset_attributions.loaders import get_attributions_dir
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.harvest.loaders import load_activation_contexts_summary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path


@dataclass
class DatasetAttributionConfig:
    wandb_path: str
    n_batches: int
    batch_size: int
    ci_threshold: float


def _build_component_keys(model: ComponentModel) -> list[str]:
    """Build flat list of component keys ('layer:c_idx') in consistent order.

    Order: wte:0, component layers, output:0
    """
    component_keys = ["wte:0"]

    for layer in model.target_module_paths:
        n_components = model.module_to_c[layer]
        for c_idx in range(n_components):
            component_keys.append(f"{layer}:{c_idx}")

    component_keys.append("output:0")
    return component_keys


def _build_alive_mask(
    model: ComponentModel,
    run_id: str,
    ci_threshold: float,
) -> Bool[Tensor, " n_components"]:
    """Build mask of alive components (mean_ci > threshold).

    Falls back to all-alive if harvest summary not available.
    wte and output are always considered alive.
    """
    summary = load_activation_contexts_summary(run_id)

    n_component_layers = sum(model.module_to_c[layer] for layer in model.target_module_paths)
    total_components = n_component_layers + 2  # +2 for wte and output
    alive_mask = torch.zeros(total_components, dtype=torch.bool)

    # wte is always alive (index 0)
    alive_mask[0] = True
    idx = 1

    if summary is None:
        logger.warning("Harvest summary not available, using all components as alive")
        alive_mask.fill_(True)
        return alive_mask

    # Build index for each component layer
    for layer in model.target_module_paths:
        n_components = model.module_to_c[layer]
        for c_idx in range(n_components):
            component_key = f"{layer}:{c_idx}"
            if component_key in summary and summary[component_key].mean_ci > ci_threshold:
                alive_mask[idx] = True
            idx += 1

    # output is always alive (last index)
    alive_mask[idx] = True

    n_alive = int(alive_mask.sum().item())
    logger.info(f"Found {n_alive}/{total_components} alive components (ci > {ci_threshold})")
    return alive_mask


def _get_output_path(run_id: str, rank: int | None) -> Path:
    """Get output path for attributions."""
    output_dir = get_attributions_dir(run_id)
    if rank is not None:
        return output_dir / f"dataset_attributions_rank_{rank}.pt"
    return output_dir / "dataset_attributions.pt"


def harvest_attributions(
    config: DatasetAttributionConfig,
    rank: int | None = None,
    world_size: int | None = None,
) -> None:
    """Compute dataset attributions over the training dataset.

    Args:
        config: Configuration for attribution harvesting.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
    """
    from spd.data import train_loader_and_tokenizer
    from spd.utils.distributed_utils import get_device

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    _, _, run_id = parse_wandb_run_path(config.wandb_path)

    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, _ = train_loader_and_tokenizer(spd_config, config.batch_size)

    # Build component keys and alive mask (includes wte and output pseudo-layers)
    component_keys = _build_component_keys(model)
    alive_mask = _build_alive_mask(model, run_id, config.ci_threshold).to(device)

    # Get gradient connectivity
    logger.info("Computing sources_by_target...")
    sources_by_target_raw = get_sources_by_target(model, str(device), spd_config.sampling)

    # Filter sources_by_target:
    # - Valid targets: component layers + output
    # - Valid sources: wte + component layers
    component_layers = set(model.target_module_paths)
    valid_sources = component_layers | {"wte"}
    valid_targets = component_layers | {"output"}

    sources_by_target = {}
    for target, sources in sources_by_target_raw.items():
        if target not in valid_targets:
            continue
        filtered_sources = [src for src in sources if src in valid_sources]
        if filtered_sources:
            sources_by_target[target] = filtered_sources
    logger.info(f"Found {len(sources_by_target)} target layers with gradient connections")

    # Create harvester
    harvester = AttributionHarvester(
        model=model,
        sources_by_target=sources_by_target,
        component_keys=component_keys,
        alive_mask=alive_mask,
        sampling=spd_config.sampling,
        device=device,
        show_progress=True,
    )

    # Process batches
    for batch_idx, batch_data in tqdm.tqdm(
        enumerate(train_loader),
        total=config.n_batches,
        desc="Attribution batches",
    ):
        if batch_idx >= config.n_batches:
            break
        # Skip batches not assigned to this rank
        if world_size is not None and batch_idx % world_size != rank:
            continue
        batch = extract_batch_data(batch_data).to(device)
        harvester.process_batch(batch)

    logger.info(
        f"Processing complete. Tokens: {harvester.n_tokens:,}, Batches: {harvester.n_batches}"
    )

    # Normalize by n_tokens to get per-token average attribution
    normalized_matrix = harvester.accumulator / harvester.n_tokens

    # Build and save storage
    storage = DatasetAttributionStorage(
        component_keys=component_keys,
        attribution_matrix=normalized_matrix.cpu(),
        n_batches_processed=harvester.n_batches,
        n_tokens_processed=harvester.n_tokens,
        ci_threshold=config.ci_threshold,
    )

    output_path = _get_output_path(run_id, rank)
    storage.save(output_path)
    logger.info(f"Saved dataset attributions to {output_path}")


def merge_attributions(wandb_path: str) -> None:
    """Merge partial attribution files from parallel workers.

    Looks for dataset_attributions_rank_*.pt files and merges them into
    dataset_attributions.pt.
    """
    _, _, run_id = parse_wandb_run_path(wandb_path)
    output_dir = get_attributions_dir(run_id)

    # Find all rank files
    rank_files = sorted(output_dir.glob("dataset_attributions_rank_*.pt"))
    assert rank_files, f"No rank files found in {output_dir}"
    logger.info(f"Found {len(rank_files)} rank files to merge")

    # Load all storages
    storages = [DatasetAttributionStorage.load(f) for f in rank_files]

    # Validate consistency
    first = storages[0]
    for s in storages[1:]:
        assert s.component_keys == first.component_keys, "Component keys mismatch"
        assert s.ci_threshold == first.ci_threshold, "CI threshold mismatch"

    # Merge: de-normalize, sum, re-normalize
    # Each storage has: normalized_matrix = accumulator / n_tokens
    # To merge: total_accum = sum(s.matrix * s.n_tokens), then normalize by total_tokens
    total_matrix = torch.stack([s.attribution_matrix * s.n_tokens_processed for s in storages]).sum(
        dim=0
    )
    total_tokens = sum(s.n_tokens_processed for s in storages)
    total_batches = sum(s.n_batches_processed for s in storages)

    merged_matrix = total_matrix / total_tokens

    # Save merged result
    merged = DatasetAttributionStorage(
        component_keys=first.component_keys,
        attribution_matrix=merged_matrix,
        n_batches_processed=total_batches,
        n_tokens_processed=total_tokens,
        ci_threshold=first.ci_threshold,
    )

    output_path = output_dir / "dataset_attributions.pt"
    merged.save(output_path)
    logger.info(f"Merged {len(rank_files)} files -> {output_path}")
    logger.info(f"Total: {total_batches} batches, {total_tokens:,} tokens")
