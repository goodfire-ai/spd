"""Dataset attribution harvesting.

Computes component-to-component attribution strengths aggregated over the full
training dataset. Unlike prompt attributions (single-prompt, position-aware),
dataset attributions answer: "In aggregate, which components typically influence
each other?"
"""

from dataclasses import dataclass

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
    """Build flat list of component keys ('layer:c_idx') in consistent order."""
    component_keys = []
    for layer in model.target_module_paths:
        n_components = model.module_to_c[layer]
        for c_idx in range(n_components):
            component_keys.append(f"{layer}:{c_idx}")
    return component_keys


def _build_alive_mask(
    model: ComponentModel,
    run_id: str,
    ci_threshold: float,
) -> Bool[Tensor, " n_components"]:
    """Build mask of alive components (mean_ci > threshold).

    Falls back to all-alive if harvest summary not available.
    """
    summary = load_activation_contexts_summary(run_id)

    total_components = sum(model.module_to_c[layer] for layer in model.target_module_paths)
    alive_mask = torch.zeros(total_components, dtype=torch.bool)

    if summary is None:
        logger.warning("Harvest summary not available, using all components as alive")
        alive_mask.fill_(True)
        return alive_mask

    # Build index for each component
    idx = 0
    for layer in model.target_module_paths:
        n_components = model.module_to_c[layer]
        for c_idx in range(n_components):
            component_key = f"{layer}:{c_idx}"
            if component_key in summary and summary[component_key].mean_ci > ci_threshold:
                alive_mask[idx] = True
            idx += 1

    n_alive = int(alive_mask.sum().item())
    logger.info(f"Found {n_alive}/{total_components} alive components (ci > {ci_threshold})")
    return alive_mask


def harvest_attributions(config: DatasetAttributionConfig) -> None:
    """Compute dataset attributions over the training dataset.

    Args:
        config: Configuration for attribution harvesting.
    """
    from spd.data import train_loader_and_tokenizer
    from spd.utils.distributed_utils import get_device

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    _, _, run_id = parse_wandb_run_path(config.wandb_path)

    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, _ = train_loader_and_tokenizer(spd_config, config.batch_size)

    # Build component keys and alive mask
    component_keys = _build_component_keys(model)
    alive_mask = _build_alive_mask(model, run_id, config.ci_threshold).to(device)

    # Get gradient connectivity
    logger.info("Computing sources_by_target...")
    sources_by_target_raw = get_sources_by_target(model, str(device), spd_config.sampling)

    # Filter to only include component layers (exclude pseudo-layers like wte, output)
    # Also filter out targets with no remaining sources
    component_layers = set(model.target_module_paths)
    sources_by_target = {}
    for target, sources in sources_by_target_raw.items():
        if target not in component_layers:
            continue
        filtered_sources = [src for src in sources if src in component_layers]
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
    train_iter = iter(train_loader)
    for batch_idx in tqdm.tqdm(range(config.n_batches), desc="Attribution batches"):
        try:
            batch = extract_batch_data(next(train_iter)).to(device)
        except StopIteration:
            logger.info(
                f"Dataset exhausted at batch {batch_idx}/{config.n_batches}. Finishing early."
            )
            break

        harvester.process_batch(batch)

    logger.info(
        f"Processing complete. Tokens: {harvester.n_tokens:,}, Batches: {harvester.n_batches}"
    )

    # Build and save storage
    storage = DatasetAttributionStorage(
        component_keys=component_keys,
        attribution_matrix=harvester.accumulator.cpu(),
        n_batches_processed=harvester.n_batches,
        n_tokens_processed=harvester.n_tokens,
        ci_threshold=config.ci_threshold,
    )

    output_dir = get_attributions_dir(run_id)
    output_path = output_dir / "dataset_attributions.pt"
    storage.save(output_path)
    logger.info(f"Saved dataset attributions to {output_path}")
