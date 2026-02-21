"""Dataset attribution convergence analysis.

Measures convergence of attributions as batches accumulate using split-half reliability.
Runs on GPU. Submit via SLURM or run directly with:

    python scripts/da_convergence_analysis.py <wandb_path> --fd_threshold 0.01 --n_batches 500

Results saved to SPD_OUT_DIR/www/da_convergence/.
"""

import json
import time
from typing import Any

import fire
import torch
from jaxtyping import Bool
from torch import Tensor

from spd.configs import SamplingType
from spd.data import train_loader_and_tokenizer
from spd.dataset_attributions.harvester import AttributionHarvester
from spd.harvest.repo import HarvestRepo
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import SPD_OUT_DIR
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path

OUTPUT_DIR = SPD_OUT_DIR / "www" / "da_convergence"


def build_alive_masks(
    model: ComponentModel,
    run_id: str,
    harvest_subrun_id: str | None,
    n_components: int,
    vocab_size: int,
    fd_threshold: float,
) -> tuple[Bool[Tensor, " n_sources"], Bool[Tensor, " n_components"], int]:
    """Build alive masks at a given firing density threshold."""
    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id=run_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(run_id, readonly=True)
        assert harvest is not None, f"No harvest data for {run_id}"
    summary = harvest.get_summary()
    assert summary is not None

    n_sources = vocab_size + n_components
    source_alive = torch.zeros(n_sources, dtype=torch.bool)
    target_alive = torch.zeros(n_components, dtype=torch.bool)
    source_alive[:vocab_size] = True

    source_idx = vocab_size
    target_idx = 0
    for layer in model.target_module_paths:
        for c_idx in range(model.module_to_c[layer]):
            key = f"{layer}:{c_idx}"
            is_alive = key in summary and summary[key].firing_density > fd_threshold
            source_alive[source_idx] = is_alive
            target_alive[target_idx] = is_alive
            source_idx += 1
            target_idx += 1

    n_alive = int(target_alive.sum().item())
    return source_alive, target_alive, n_alive


def build_harvester(
    model: ComponentModel,
    topology: TransformerTopology,
    sources_by_target: dict[str, list[str]],
    vocab_size: int,
    n_components: int,
    source_alive: Tensor,
    target_alive: Tensor,
    sampling: SamplingType,
    device: torch.device,
    include_output: bool = False,
) -> AttributionHarvester:
    """Build a harvester with the given alive masks."""
    component_layers = set(model.target_module_paths)
    valid_sources = component_layers | {"wte"}
    valid_targets = component_layers | ({"output"} if include_output else set())

    filtered = {}
    for target, sources in sources_by_target.items():
        if target not in valid_targets:
            continue
        fsrc = [s for s in sources if s in valid_sources]
        if fsrc:
            filtered[target] = fsrc

    return AttributionHarvester(
        model=model,
        sources_by_target=filtered,
        n_components=n_components,
        vocab_size=vocab_size,
        source_alive=source_alive.to(device),
        target_alive=target_alive.to(device),
        sampling=sampling,
        embedding_module=topology.embedding_module,
        unembed_module=topology.unembed_module,
        device=device,
        show_progress=False,
    )


def run_convergence(
    wandb_path: str,
    fd_threshold: float = 0.01,
    n_batches: int = 500,
    batch_size: int = 32,
    harvest_subrun_id: str | None = None,
) -> None:
    """Run convergence analysis with split-half reliability."""
    device = torch.device(get_device())
    _, _, run_id = parse_wandb_run_path(wandb_path)

    logger.info(f"Loading model on {device}...")
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, batch_size)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)
    n_components = sum(model.module_to_c[layer] for layer in model.target_module_paths)
    logger.info(f"vocab={vocab_size}, n_components={n_components}")

    # Gradient connectivity
    logger.info("Computing gradient connectivity...")
    topology = TransformerTopology(model.target_model)
    sources_by_target = get_sources_by_target(model, topology, str(device), spd_config.sampling)

    # Build alive masks
    source_alive, target_alive, n_alive = build_alive_masks(
        model, run_id, harvest_subrun_id, n_components, vocab_size, fd_threshold
    )
    logger.info(f"fd_threshold={fd_threshold}: {n_alive}/{n_components} alive components")

    alive_target_indices = torch.where(target_alive)[0]
    alive_source_comp_indices = torch.where(source_alive[vocab_size:])[0] + vocab_size

    # Two harvesters for split-half
    harvester_a = build_harvester(
        model,
        topology,
        sources_by_target,
        vocab_size,
        n_components,
        source_alive,
        target_alive,
        spd_config.sampling,
        device,
    )
    harvester_b = build_harvester(
        model,
        topology,
        sources_by_target,
        vocab_size,
        n_components,
        source_alive,
        target_alive,
        spd_config.sampling,
        device,
    )

    checkpoints = sorted(
        set([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000])
    )
    checkpoints = [c for c in checkpoints if c <= n_batches]

    results: dict[str, Any] = {
        "run_id": run_id,
        "fd_threshold": fd_threshold,
        "n_alive": n_alive,
        "n_components": n_components,
        "vocab_size": vocab_size,
        "n_batches_target": n_batches,
        "batch_size": batch_size,
        "checkpoints": [],
    }

    # Timing
    train_iter = iter(train_loader)
    first_batch = extract_batch_data(next(train_iter)).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    harvester_a.process_batch(first_batch)
    torch.cuda.synchronize()
    time_per_batch = time.time() - t0
    results["time_per_batch_seconds"] = time_per_batch
    results["estimated_total_hours"] = time_per_batch * n_batches / 3600
    logger.info(
        f"Time per batch: {time_per_batch:.1f}s (est. total: {time_per_batch * n_batches / 3600:.1f}h)"
    )

    batch_count = 1  # Already processed one

    for batch_idx in range(1, n_batches):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted at batch {batch_idx}")
            break

        batch = extract_batch_data(batch_data).to(device)
        batch_count += 1

        # Alternate for split-half
        if batch_idx % 2 == 0:
            harvester_a.process_batch(batch)
        else:
            harvester_b.process_batch(batch)

        if batch_count in checkpoints and harvester_a.n_tokens > 0 and harvester_b.n_tokens > 0:
            avg_a = harvester_a.comp_accumulator / harvester_a.n_tokens
            avg_b = harvester_b.comp_accumulator / harvester_b.n_tokens

            sub_a = (
                avg_a[alive_source_comp_indices][:, alive_target_indices].flatten().float().cpu()
            )
            sub_b = (
                avg_b[alive_source_comp_indices][:, alive_target_indices].flatten().float().cpu()
            )

            mask = (sub_a != 0) | (sub_b != 0)
            if mask.sum() > 10:
                va, vb = sub_a[mask], sub_b[mask]
                corr = torch.corrcoef(torch.stack([va, vb]))[0, 1].item()
                abs_corr = torch.corrcoef(torch.stack([va.abs(), vb.abs()]))[0, 1].item()
                k = min(1000, len(va))
                top_a = set(torch.topk(va.abs(), k).indices.tolist())
                top_b = set(torch.topk(vb.abs(), k).indices.tolist())
                top_k_overlap = len(top_a & top_b) / k
            else:
                corr = abs_corr = top_k_overlap = float("nan")

            cp = {
                "batch_count": batch_count,
                "batches_per_half": harvester_a.n_batches,
                "tokens_per_half": harvester_a.n_tokens,
                "split_half_pearson": corr,
                "split_half_abs_pearson": abs_corr,
                "top_k_overlap": top_k_overlap,
                "n_nonzero": int(mask.sum().item()),
            }
            results["checkpoints"].append(cp)
            logger.info(
                f"Batch {batch_count}: r={corr:.4f}, |r|={abs_corr:.4f}, top-k={top_k_overlap:.3f}"
            )

            # Save intermediate
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_DIR / f"convergence_fd{fd_threshold}.json", "w") as f:
                json.dump(results, f, indent=2)

    results["n_batches_actual"] = batch_count
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"convergence_fd{fd_threshold}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Done! Results at {OUTPUT_DIR / f'convergence_fd{fd_threshold}.json'}")


if __name__ == "__main__":
    fire.Fire(run_convergence)
