"""Generic decomposition harvest over method-provided per-token activations."""

import itertools
import time
from pathlib import Path
from typing import Any

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader

from spd.harvest.config import HarvestConfig
from spd.harvest.db import HarvestDB
from spd.harvest.harvester import Harvester
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger
from spd.utils.general_utils import bf16_autocast, extract_batch_data

from .types import BatchLike, DecompositionSpec


def _default_batch_to_tokens(batch: BatchLike) -> Int[Tensor, "batch seq"]:
    return extract_batch_data(batch)


def _move_batch_to_device(batch: BatchLike, device: torch.device) -> BatchLike:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(v.to(device) for v in batch)
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}


def _call_model(model: nn.Module, batch: BatchLike) -> Tensor | object:
    if isinstance(batch, dict):
        try:
            return model(**batch)
        except TypeError:
            try:
                return model(batch)
            except Exception:
                return model(extract_batch_data(batch))
    if isinstance(batch, tuple):
        try:
            return model(*batch)
        except TypeError:
            try:
                return model(batch)
            except Exception:
                return model(extract_batch_data(batch))
    return model(batch)


def _default_output_probs(model: nn.Module, batch: BatchLike) -> Float[Tensor, "batch seq vocab"]:
    out = _call_model(model, batch)
    logits: Tensor
    if isinstance(out, Tensor):
        logits = out
    else:
        output = getattr(out, "output", None)
        if isinstance(output, Tensor):
            logits = output
        elif isinstance(out, tuple) and out and isinstance(out[0], Tensor):
            logits = out[0]
        else:
            raise TypeError(
                "Unable to extract logits from model output; pass output_probs_fn in spec"
            )
    return torch.softmax(logits, dim=-1)


def _stack_component_values(
    per_component: dict[str, Float[Tensor, "batch seq"]],
    component_keys: list[str],
) -> Float[Tensor, "batch seq component"]:
    missing = [k for k in component_keys if k not in per_component]
    assert not missing, f"ActivationFn missing component keys: {missing[:5]}"
    return torch.stack([per_component[k] for k in component_keys], dim=2)


def _save_harvest_results(
    harvester: Harvester,
    config: HarvestConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building and saving component results...")
    db_path = output_dir / "harvest.db"
    db = HarvestDB(db_path)
    db.save_config(config)
    components_iter = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    n_saved = db.save_components_iter(components_iter)
    db.close()
    logger.info(f"Saved {n_saved} components to {db_path}")

    component_keys = list(harvester.component_keys)
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


def harvest_decomposition(
    spec: DecompositionSpec,
    config: HarvestConfig,
    output_dir: Path,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    dataloader: DataLoader[Any] | None = None,
    device: torch.device | None = None,
) -> None:
    """Single-pass harvest for any decomposition method."""
    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = spec.model.to(device)
    model.eval()

    if dataloader is None:
        dataloader = DataLoader(spec.dataset, batch_size=config.batch_size, shuffle=False)

    batch_to_tokens = spec.batch_to_tokens_fn or _default_batch_to_tokens
    output_probs_fn = spec.output_probs_fn or _default_output_probs

    train_iter = iter(dataloader)
    component_keys: list[str] = list(spec.component_order or spec.component_explanations.keys())
    harvester: Harvester | None = None

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

        batch_for_model = _move_batch_to_device(batch_item, device)
        batch_tokens = batch_to_tokens(batch_for_model).to(device)
        if not isinstance(batch_for_model, (dict, tuple)):
            batch_for_model = batch_tokens

        with torch.no_grad(), bf16_autocast():
            per_component_activation = spec.activation_fn(model, batch_for_model)
            if not component_keys:
                component_keys = list(per_component_activation.keys())
            activation = _stack_component_values(per_component_activation, component_keys)

            if spec.component_acts_fn is None:
                component_acts = activation
            else:
                component_acts = _stack_component_values(
                    spec.component_acts_fn(model, batch_for_model), component_keys
                )

            output_probs = output_probs_fn(model, batch_for_model)

            if harvester is None:
                # Reuse the existing Harvester tensor layout by treating each generic
                # component key as its own "layer" with one component.
                c_per_layer = {key: 1 for key in component_keys}
                harvester = Harvester(
                    layer_names=component_keys,
                    c_per_layer=c_per_layer,
                    component_keys=component_keys,
                    vocab_size=output_probs.shape[-1],
                    activation_threshold=config.activation_threshold,
                    max_examples_per_component=config.activation_examples_per_component,
                    context_tokens_per_side=config.activation_context_tokens_per_side,
                    max_examples_per_batch_per_component=config.max_examples_per_batch_per_component,
                    device=device,
                )

            harvester.process_batch(batch_tokens, activation, output_probs, component_acts)

        batches_processed += 1
        now = time.time()
        if rank is not None and now - last_log_time >= 10:
            logger.info(f"[Worker {rank}] {batches_processed} batches")
            last_log_time = now

    assert harvester is not None, "No batches were processed; cannot save empty harvest"
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
