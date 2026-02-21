"""Dataset attribution harvesting.

Computes component-to-component attribution strengths aggregated over the full
training dataset. Unlike prompt attributions (single-prompt, position-aware),
dataset attributions answer: "In aggregate, which components typically influence
each other?"

Uses residual-based storage for scalability:
- Component targets: stored directly
- Output targets: stored as attributions to output residual, computed on-the-fly at query time

See CLAUDE.md in this directory for usage instructions.
"""

import itertools
from pathlib import Path

import torch
import tqdm
from jaxtyping import Bool
from torch import Tensor

from spd.data import train_loader_and_tokenizer
from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvester import AttributionHarvester
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.harvest.repo import HarvestRepo
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path


def _build_component_layer_keys(model: ComponentModel) -> list[str]:
    """Build list of component layer keys in canonical order.

    Returns keys like ["h.0.attn.q_proj:0", "h.0.attn.q_proj:1", ...] for all layers.
    wte and output keys are not included - they're constructed from vocab_size.
    """
    component_layer_keys = []
    for layer in model.target_module_paths:
        n_components = model.module_to_c[layer]
        for c_idx in range(n_components):
            component_layer_keys.append(f"{layer}:{c_idx}")
    return component_layer_keys


def _build_alive_masks(
    model: ComponentModel,
    run_id: str,
    harvest_subrun_id: str | None,
    n_components: int,
    vocab_size: int,
    fd_threshold: float,
    include_wte_source: bool,
) -> tuple[Bool[Tensor, " n_sources"], Bool[Tensor, " n_components"]]:
    """Build masks of alive components (firing_density > fd_threshold) for sources and targets.

    Index structure:
    - Sources: [0, vocab_size) = wte tokens, [vocab_size, vocab_size + n_components) = component layers
    - Targets: [0, n_components) = component layers (output handled via out_residual)
    """

    n_sources = vocab_size + n_components

    source_alive = torch.zeros(n_sources, dtype=torch.bool)
    target_alive = torch.zeros(n_components, dtype=torch.bool)

    # wte tokens alive only if included as source
    source_alive[:vocab_size] = include_wte_source

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id=run_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(run_id, readonly=True)
        assert harvest is not None, f"No harvest data for {run_id}"
    summary = harvest.get_summary()
    assert summary is not None, "Harvest summary not available"

    # Build masks for component layers
    source_idx = vocab_size  # Start after wte tokens
    target_idx = 0

    for layer in model.target_module_paths:
        n_layer_components = model.module_to_c[layer]
        for c_idx in range(n_layer_components):
            component_key = f"{layer}:{c_idx}"
            is_alive = (
                component_key in summary and summary[component_key].firing_density > fd_threshold
            )
            source_alive[source_idx] = is_alive
            target_alive[target_idx] = is_alive
            source_idx += 1
            target_idx += 1

    n_source_alive = int(source_alive.sum().item())
    n_target_alive = int(target_alive.sum().item())
    logger.info(
        f"Alive components: {n_source_alive}/{n_sources} sources, "
        f"{n_target_alive}/{n_components} component targets "
        f"(firing density > {fd_threshold})"
    )
    return source_alive, target_alive


def harvest_attributions(
    wandb_path: str,
    config: DatasetAttributionConfig,
    output_dir: Path,
    harvest_subrun_id: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> None:
    """Compute dataset attributions over the training dataset.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Configuration for attribution harvesting.
        output_dir: Directory to write results into.
        harvest_subrun_id: Harvest subrun to use for alive masks. If None, uses most recent.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
    """

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    _, _, run_id = parse_wandb_run_path(wandb_path)

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, config.batch_size)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int), f"vocab_size must be int, got {type(vocab_size)}"
    logger.info(f"Vocab size: {vocab_size}")

    # Build component keys and alive masks
    component_layer_keys = _build_component_layer_keys(model)
    n_components = len(component_layer_keys)
    source_alive, target_alive = _build_alive_masks(
        model,
        run_id,
        harvest_subrun_id,
        n_components,
        vocab_size,
        fd_threshold=config.fd_threshold,
        include_wte_source=config.include_wte_source,
    )
    source_alive = source_alive.to(device)
    target_alive = target_alive.to(device)

    n_sources = vocab_size + n_components
    logger.info(f"Component layers: {n_components}, Sources: {n_sources}")

    # Get gradient connectivity
    logger.info("Computing sources_by_target...")
    topology = TransformerTopology(model.target_model)
    sources_by_target_raw = get_sources_by_target(model, topology, str(device), spd_config.sampling)

    # Filter sources_by_target based on config
    component_layers = set(model.target_module_paths)
    valid_sources = component_layers | ({"wte"} if config.include_wte_source else set())
    valid_targets = component_layers | ({"output"} if config.include_output_target else set())

    sources_by_target = {}
    for target, sources in sources_by_target_raw.items():
        if target not in valid_targets:
            continue
        filtered_sources = [src for src in sources if src in valid_sources]
        if filtered_sources:
            sources_by_target[target] = filtered_sources
    logger.info(
        f"Found {len(sources_by_target)} target layers with gradient connections "
        f"(include_output={config.include_output_target}, include_wte={config.include_wte_source})"
    )

    # Create harvester
    harvester = AttributionHarvester(
        model=model,
        sources_by_target=sources_by_target,
        n_components=n_components,
        vocab_size=vocab_size,
        source_alive=source_alive,
        target_alive=target_alive,
        sampling=spd_config.sampling,
        embedding_module=topology.embedding_module,
        unembed_module=topology.unembed_module,
        device=device,
        show_progress=True,
    )

    # Process batches
    train_iter = iter(train_loader)
    match config.n_batches:
        case int(n_batches):
            batch_range = range(n_batches)
        case "whole_dataset":
            batch_range = itertools.count()
    for batch_idx in tqdm.tqdm(batch_range, desc="Attribution batches"):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted at batch {batch_idx}. Processing complete.")
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
    normalized_comp = harvester.comp_accumulator / harvester.n_tokens
    normalized_out_residual = harvester.out_residual_accumulator / harvester.n_tokens

    # Build and save storage
    storage = DatasetAttributionStorage(
        component_layer_keys=component_layer_keys,
        vocab_size=vocab_size,
        d_model=harvester.d_model,
        source_to_component=normalized_comp.cpu(),
        source_to_out_residual=normalized_out_residual.cpu(),
        n_batches_processed=harvester.n_batches,
        n_tokens_processed=harvester.n_tokens,
        ci_threshold=config.ci_threshold,
        fd_threshold=config.fd_threshold,
    )

    if rank is not None:
        worker_dir = output_dir / "worker_states"
        worker_dir.mkdir(parents=True, exist_ok=True)
        output_path = worker_dir / f"dataset_attributions_rank_{rank}.pt"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset_attributions.pt"
    storage.save(output_path)
    logger.info(f"Saved dataset attributions to {output_path}")


def merge_attributions(output_dir: Path) -> None:
    """Merge partial attribution files from parallel workers.

    Looks for worker_states/dataset_attributions_rank_*.pt files and merges them
    into dataset_attributions.pt in the output_dir.

    Uses streaming merge to avoid OOM - loads one file at a time instead of all at once.
    """
    worker_dir = output_dir / "worker_states"
    rank_files = sorted(worker_dir.glob("dataset_attributions_rank_*.pt"))
    assert rank_files, f"No rank files found in {worker_dir}"
    logger.info(f"Found {len(rank_files)} rank files to merge")

    # Load first file to get metadata and initialize accumulators
    # Use double precision for accumulation to prevent precision loss with billions of tokens
    first = DatasetAttributionStorage.load(rank_files[0])
    total_comp = (first.source_to_component * first.n_tokens_processed).double()
    total_out_residual = (first.source_to_out_residual * first.n_tokens_processed).double()
    total_tokens = first.n_tokens_processed
    total_batches = first.n_batches_processed
    logger.info(f"Loaded rank 0: {first.n_tokens_processed:,} tokens")

    # Stream remaining files one at a time
    for rank_file in tqdm.tqdm(rank_files[1:], desc="Merging rank files"):
        storage = DatasetAttributionStorage.load(rank_file)

        # Validate consistency
        assert storage.component_layer_keys == first.component_layer_keys, (
            "Component layer keys mismatch"
        )
        assert storage.vocab_size == first.vocab_size, "Vocab size mismatch"
        assert storage.d_model == first.d_model, "d_model mismatch"
        assert storage.ci_threshold == first.ci_threshold, "CI threshold mismatch"
        assert storage.fd_threshold == first.fd_threshold, "FD threshold mismatch"

        # Accumulate de-normalized values
        total_comp += storage.source_to_component * storage.n_tokens_processed
        total_out_residual += storage.source_to_out_residual * storage.n_tokens_processed
        total_tokens += storage.n_tokens_processed
        total_batches += storage.n_batches_processed

    # Normalize by total tokens and convert back to float32 for storage
    merged_comp = (total_comp / total_tokens).float()
    merged_out_residual = (total_out_residual / total_tokens).float()

    # Save merged result
    merged = DatasetAttributionStorage(
        component_layer_keys=first.component_layer_keys,
        vocab_size=first.vocab_size,
        d_model=first.d_model,
        source_to_component=merged_comp,
        source_to_out_residual=merged_out_residual,
        n_batches_processed=total_batches,
        n_tokens_processed=total_tokens,
        ci_threshold=first.ci_threshold,
        fd_threshold=first.fd_threshold,
    )

    output_path = output_dir / "dataset_attributions.pt"
    merged.save(output_path)
    assert output_path.stat().st_size > 0, f"Merge output is empty: {output_path}"
    logger.info(f"Merged {len(rank_files)} files -> {output_path}")
    logger.info(f"Total: {total_batches} batches, {total_tokens:,} tokens")

    for rank_file in rank_files:
        rank_file.unlink()
    worker_dir.rmdir()
    logger.info(f"Deleted {len(rank_files)} per-rank files and worker_states/")
