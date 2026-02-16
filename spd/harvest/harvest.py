"""Generic harvest pipeline: single-pass collection of component statistics.

Collects per-component statistics in a single pass over the data:
- Input/output token PMI (pointwise mutual information)
- Activation examples with context windows
- Firing counts and activation sums
- Component co-occurrence counts

Performance (SimpleStories, 600M tokens, batch_size=256):
- ~0.85 seconds per batch
- ~1.1 hours for full dataset
"""

import itertools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import tqdm

from spd.harvest.config import HarvestConfig
from spd.harvest.db import HarvestDB
from spd.harvest.harvester import Harvester
from spd.harvest.schemas import HarvestBatch
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger
from spd.utils.general_utils import bf16_autocast


def _save_harvest_results(
    harvester: Harvester,
    config: HarvestConfig,
    output_dir: Path,
) -> None:
    """Build and save all harvest results to disk.

    Components are streamed to the DB one at a time to avoid holding all ~40K
    ComponentData objects in memory simultaneously (~187 GB as Python objects).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building and saving component results...")
    db_path = output_dir / "harvest.db"
    db = HarvestDB(db_path)
    db.save_config(config)
    components_iter = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    n_saved = db.save_components_iter(components_iter)
    db.close()
    logger.info(f"Saved {n_saved} components to {db_path}")

    component_keys = harvester.component_keys

    correlations = CorrelationStorage(
        component_keys=component_keys,
        count_i=harvester.firing_counts.long().cpu(),
        count_ij=harvester.cooccurrence_counts.long().cpu(),
        count_total=harvester.total_tokens_processed,
    )
    correlations.save(output_dir / "component_correlations.pt")

    token_stats = TokenStatsStorage(
        component_keys=component_keys,
        vocab_size=harvester.vocab_size,
        n_tokens=harvester.total_tokens_processed,
        input_counts=harvester.input_cooccurrence.cpu(),
        input_totals=harvester.input_marginals.float().cpu(),
        output_counts=harvester.output_cooccurrence.cpu(),
        output_totals=harvester.output_marginals.cpu(),
        firing_counts=harvester.firing_counts.cpu(),
    )
    token_stats.save(output_dir / "token_stats.pt")


def harvest(
    harvest_fn: Callable[[Any], HarvestBatch],
    layers: list[tuple[str, int]],
    vocab_size: int,
    dataloader: Any,
    config: HarvestConfig,
    output_dir: Path,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Single-pass harvest for any decomposition method.

    Args:
        harvest_fn: Converts a raw dataloader batch into a HarvestBatch.
            Responsible for moving data to the correct device.
        layers: List of (layer_name, n_components) pairs.
        vocab_size: Vocabulary size for token stats.
        dataloader: Iterable yielding raw batches.
        config: Harvest configuration.
        output_dir: Directory to save harvest outputs.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers.
        device: Device for accumulator tensors.
    """
    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    harvester = Harvester(
        layers=layers,
        vocab_size=vocab_size,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        max_examples_per_batch_per_component=config.max_examples_per_batch_per_component,
        device=device,
    )

    train_iter = iter(dataloader)
    batches_processed = 0
    last_log_time = time.time()
    match config.n_batches:
        case int(n_batches):
            batch_range = range(n_batches)
        case "whole_dataset":
            batch_range = itertools.count()

    for batch_idx in tqdm.tqdm(batch_range, desc="Harvesting", disable=rank is not None):
        try:
            batch_item = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted at batch {batch_idx}. Processing complete.")
            break

        if world_size is not None and batch_idx % world_size != rank:
            continue

        with torch.no_grad(), bf16_autocast():
            hb = harvest_fn(batch_item)
            harvester.process_batch(hb.tokens, hb.firings, hb.activations, hb.output_probs)

        batches_processed += 1
        now = time.time()
        if rank is not None and now - last_log_time >= 10:
            logger.info(f"[Worker {rank}] {batches_processed} batches")
            last_log_time = now

    logger.info(
        f"{'[Worker ' + str(rank) + '] ' if rank is not None else ''}"
        f"Processing complete. {batches_processed} batches, "
        f"{harvester.total_tokens_processed:,} tokens"
    )

    if rank is not None:
        state_dir = output_dir / "worker_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / f"worker_{rank}.pt"
        harvester.save(state_path)
        logger.info(f"[Worker {rank}] Saved state to {state_path}")
    else:
        _save_harvest_results(harvester, config, output_dir)
        logger.info(f"Saved results to {output_dir}")


def merge_harvest(output_dir: Path, config: HarvestConfig) -> None:
    """Merge partial harvest results from parallel workers.

    Looks for worker_*.pt state files in output_dir/worker_states/ and merges them
    into final harvest results written to output_dir.
    """
    state_dir = output_dir / "worker_states"

    worker_files = sorted(state_dir.glob("worker_*.pt"))
    assert worker_files, f"No worker state files found in {state_dir}"
    logger.info(f"Found {len(worker_files)} worker state files to merge")

    first_worker_file, *rest_worker_files = worker_files

    logger.info(f"Loading worker 0: {first_worker_file.name}")
    harvester = Harvester.load(first_worker_file, device=torch.device("cpu"))
    logger.info(f"Loaded worker 0: {harvester.total_tokens_processed:,} tokens")

    for worker_file in tqdm.tqdm(rest_worker_files, desc="Merging worker states"):
        other = Harvester.load(worker_file, device=torch.device("cpu"))
        harvester.merge(other)
        del other

    logger.info(f"Merge complete. Total tokens: {harvester.total_tokens_processed:,}")

    _save_harvest_results(harvester, config, output_dir)
    db_path = output_dir / "harvest.db"
    assert db_path.exists() and db_path.stat().st_size > 0, f"Merge output is empty: {db_path}"
    logger.info(f"Saved merged results to {output_dir}")

    for worker_file in worker_files:
        worker_file.unlink()
    state_dir.rmdir()
    logger.info(f"Deleted {len(worker_files)} worker state files")
