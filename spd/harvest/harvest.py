"""Harvest token stats and activation contexts for autointerp.

Collects per-component statistics in a single pass over the data:
- Input/output token PMI (pointwise mutual information)
- Activation examples with context windows
- Firing counts and CI sums
- Component co-occurrence counts

Performance (SimpleStories, 600M tokens, batch_size=256):
- ~0.85 seconds per batch
- ~1.1 hours for full dataset
"""

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tqdm
from jaxtyping import Float
from torch import Tensor

from spd.data import train_loader_and_tokenizer
from spd.harvest.config import HarvestConfig
from spd.harvest.harvester import Harvester, HarvesterState
from spd.topology import TransformerTopology
from spd.harvest.schemas import (
    ComponentData,
    ComponentSummary,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import bf16_autocast, extract_batch_data


def _compute_u_norms(model: ComponentModel) -> dict[str, Float[Tensor, " C"]]:
    """Compute ||U[c,:]|| for each component c in each layer.

    Component activations (v_i^T @ a) have a scale invariance: scaling V by α and U by 1/α
    leaves the weight matrix unchanged but scales component activations by α. To make component
    activations reflect actual output contribution, we multiply by the U row norms.
    This gives a value proportional to the magnitude of the component's output vector.
    """
    u_norms: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, component in model.components.items():
        # U has shape (C, d_out) for LinearComponents
        u_norms[layer_name] = component.U.norm(dim=1)  # [C]
    return u_norms


def _normalize_component_acts(
    component_acts: dict[str, Float[Tensor, "B S C"]],
    u_norms: dict[str, Float[Tensor, " C"]],
) -> dict[str, Float[Tensor, "B S C"]]:
    """Normalize component activations by U column norms (output magnitude)."""
    normalized = {}
    for layer_name, acts in component_acts.items():
        norms = u_norms[layer_name].to(acts.device)
        normalized[layer_name] = acts * norms
    return normalized


@dataclass
class HarvestResult:
    """Result of harvest containing components, correlations, and token stats."""

    components: list[ComponentData]
    correlations: CorrelationStorage
    token_stats: TokenStatsStorage
    config: HarvestConfig

    def save(self, activation_contexts_dir: Path, correlations_dir: Path) -> None:
        """Save harvest result to disk."""
        # Save activation contexts (JSONL)
        activation_contexts_dir.mkdir(parents=True, exist_ok=True)

        config_path = activation_contexts_dir / "config.json"
        config_path.write_text(json.dumps(self.config.model_dump(), indent=2))

        components_path = activation_contexts_dir / "components.jsonl"
        with open(components_path, "w") as f:
            for comp in self.components:
                f.write(json.dumps(asdict(comp)) + "\n")
        logger.info(f"Saved {len(self.components)} components to {components_path}")

        # Save lightweight summary for fast /summary endpoint
        summaries = {
            comp.component_key: ComponentSummary(
                layer=comp.layer,
                component_idx=comp.component_idx,
                mean_ci=comp.mean_ci,
            )
            for comp in self.components
        }
        summary_path = activation_contexts_dir / "summary.json"
        ComponentSummary.save_all(summaries, summary_path)
        logger.info(f"Saved summary to {summary_path}")

        # Save correlations (.pt)
        self.correlations.save(correlations_dir / "component_correlations.pt")

        # Save token stats (.pt)
        self.token_stats.save(correlations_dir / "token_stats.pt")


def _build_harvest_result(
    harvester: Harvester,
    config: HarvestConfig,
    topology: TransformerTopology,
) -> HarvestResult:
    """Build HarvestResult from a harvester."""
    logger.info("Building component results...")
    components = harvester.build_results(
        pmi_top_k_tokens=config.pmi_token_top_k, topology=topology
    )
    logger.info(f"Built {len(components)} components (skipped components with no firings)")

    # Build component keys list with canonical addresses
    component_keys = [
        f"{topology.get_canonical_weight(layer).canonical_str()}:{c}"
        for layer in harvester.layer_names
        for c in range(harvester.c_per_layer[layer])
    ]

    correlations = CorrelationStorage(
        component_keys=component_keys,
        count_i=harvester.firing_counts.long().cpu(),
        count_ij=harvester.count_ij.long().cpu(),
        count_total=harvester.total_tokens_processed,
    )

    token_stats = TokenStatsStorage(
        component_keys=component_keys,
        vocab_size=harvester.vocab_size,
        n_tokens=harvester.total_tokens_processed,
        input_counts=harvester.input_token_counts.cpu(),
        input_totals=harvester.input_token_totals.float().cpu(),
        output_counts=harvester.output_token_prob_mass.cpu(),
        output_totals=harvester.output_token_prob_totals.cpu(),
        firing_counts=harvester.firing_counts.cpu(),
    )

    return HarvestResult(
        components=components,
        correlations=correlations,
        token_stats=token_stats,
        config=config,
    )


def harvest_activation_contexts(
    wandb_path: str,
    config: HarvestConfig,
    activation_contexts_dir: Path,
    correlations_dir: Path,
    rank: int | None = None,
    world_size: int | None = None,
) -> None:
    """Single-pass harvest of token stats, activation contexts, and correlations.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Harvest configuration.
        activation_contexts_dir: Directory to save activation contexts.
        correlations_dir: Directory to save correlations.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
    """
    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, config.batch_size)

    layer_names = list(model.target_module_paths)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    # Precompute U norms for normalizing component activations
    u_norms = _compute_u_norms(model)

    harvester = Harvester(
        layer_names=layer_names,
        c_per_layer=model.module_to_c,
        vocab_size=vocab_size,
        ci_threshold=config.ci_threshold,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        device=device,
    )

    train_iter = iter(train_loader)
    batches_processed = 0
    last_log_time = time.time()
    match config.n_batches:
        case int(n_batches):
            batch_range = range(n_batches)
        case "whole_dataset":
            batch_range = itertools.count()

    for batch_idx in tqdm.tqdm(batch_range, desc="Harvesting", disable=rank is not None):
        try:
            batch_data = extract_batch_data(next(train_iter))
        except StopIteration:
            logger.info(f"Dataset exhausted at batch {batch_idx}. Processing complete.")
            break

        # Skip batches not assigned to this rank
        if world_size is not None and batch_idx % world_size != rank:
            continue

        batch = batch_data.to(device)
        with torch.no_grad(), bf16_autocast():
            out = model(batch, cache_type="input")
            probs = torch.softmax(out.output, dim=-1)

            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=spd_config.sampling,
            ).lower_leaky

            ci: Float[Tensor, "B S n_comp"] = torch.cat(
                [ci_dict[layer] for layer in layer_names], dim=2
            )
            expected_n_comp = sum(model.module_to_c[layer] for layer in layer_names)
            assert ci.shape[2] == expected_n_comp

            component_acts = model.get_all_component_acts(out.cache)
            normalized_acts = _normalize_component_acts(component_acts, u_norms)
            subcomp_acts: Float[Tensor, "B S n_comp"] = torch.cat(
                [normalized_acts[layer] for layer in layer_names],
                dim=2,
            )

            harvester.process_batch(batch, ci, probs, subcomp_acts)

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

    # Save results (with rank suffix if distributed)
    if rank is not None:
        # Distributed: save worker state
        state = harvester.get_state()
        state_dir = activation_contexts_dir.parent / "worker_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / f"worker_{rank}.pt"
        torch.save(state, state_path)
        logger.info(f"[Worker {rank}] Saved state to {state_path}")
    else:
        # Single GPU: save full result
        topology = TransformerTopology(model.target_model)
        result = _build_harvest_result(harvester, config, topology)
        result.save(activation_contexts_dir, correlations_dir)
        logger.info(f"Saved results to {activation_contexts_dir} and {correlations_dir}")


def merge_activation_contexts(wandb_path: str) -> None:
    """Merge partial harvest results from parallel workers.

    Looks for worker_*.pt state files and merges them into final harvest results.

    Uses streaming merge to avoid OOM - loads one file at a time instead of all at once.
    """
    from spd.harvest.schemas import get_activation_contexts_dir, get_correlations_dir
    from spd.utils.wandb_utils import parse_wandb_run_path

    _, _, run_id = parse_wandb_run_path(wandb_path)
    activation_contexts_dir = get_activation_contexts_dir(run_id)
    correlations_dir = get_correlations_dir(run_id)
    state_dir = activation_contexts_dir.parent / "worker_states"

    # Find all worker state files
    worker_files = sorted(state_dir.glob("worker_*.pt"))
    assert worker_files, f"No worker state files found in {state_dir}"
    logger.info(f"Found {len(worker_files)} worker state files to merge")

    # Load first file to initialize merged state
    logger.info(f"Loading worker 0: {worker_files[0].name}")
    merged_state: HarvesterState = torch.load(worker_files[0], weights_only=False)
    logger.info(f"Loaded worker 0: {merged_state.total_tokens_processed:,} tokens")

    # Stream remaining files one at a time
    for worker_file in tqdm.tqdm(worker_files[1:], desc="Merging worker states"):
        state = torch.load(worker_file, weights_only=False)
        merged_state.merge_into(state)
        # state will be garbage collected here before loading the next file

    logger.info(f"Merge complete. Total tokens: {merged_state.total_tokens_processed:,}")

    # Build harvester from merged state and generate results
    harvester = Harvester.from_state(merged_state, torch.device("cpu"))

    config = HarvestConfig(
        ci_threshold=merged_state.ci_threshold,
        activation_examples_per_component=merged_state.max_examples_per_component,
        activation_context_tokens_per_side=merged_state.context_tokens_per_side,
    )

    # Load model just for topology (canonical key generation)
    run_info = SPDRunInfo.from_path(wandb_path)
    target_model = ComponentModel.from_run_info(run_info).target_model
    topology = TransformerTopology(target_model)

    result = _build_harvest_result(harvester, config, topology)
    result.save(activation_contexts_dir, correlations_dir)
    logger.info(f"Saved merged results to {activation_contexts_dir} and {correlations_dir}")

    # Clean up worker state files
    for worker_file in worker_files:
        worker_file.unlink()
    logger.info(f"Deleted {len(worker_files)} worker state files")
