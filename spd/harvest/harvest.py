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

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tqdm
from jaxtyping import Float
from torch import Tensor

from spd.harvest.lib.harvester import Harvester, HarvesterState
from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentSummary,
    ComponentTokenPMI,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import extract_batch_data


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
class HarvestConfig:
    wandb_path: str
    n_batches: int
    batch_size: int
    ci_threshold: float
    activation_examples_per_component: int
    activation_context_tokens_per_side: int
    pmi_token_top_k: int


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
        config_path.write_text(json.dumps(asdict(self.config), indent=2))

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

    @staticmethod
    def load_components(activation_contexts_dir: Path) -> list[ComponentData]:
        """Load components from disk."""
        assert activation_contexts_dir.exists(), f"No harvest found at {activation_contexts_dir}"

        components_path = activation_contexts_dir / "components.jsonl"
        components = []
        with open(components_path) as f:
            for line in f:
                data = json.loads(line)
                data["activation_examples"] = [
                    ActivationExample(**ex) for ex in data["activation_examples"]
                ]
                data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
                data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
                components.append(ComponentData(**data))

        return components


def _build_harvest_result(
    harvester: Harvester,
    config: HarvestConfig,
) -> HarvestResult:
    """Build HarvestResult from a harvester."""
    logger.info("Building component results...")
    components = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    logger.info(f"Built {len(components)} components (skipped components with no firings)")

    # Build component keys list (same ordering as tensors)
    component_keys = [
        f"{layer}:{c}"
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


def harvest(
    config: HarvestConfig,
    activation_contexts_dir: Path,
    correlations_dir: Path,
) -> None:
    """Single-pass harvest of token stats, activation contexts, and correlations."""
    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import SPDRunInfo
    from spd.utils.distributed_utils import get_device

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(config.wandb_path)
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
    for batch_idx in tqdm.tqdm(range(config.n_batches), desc="Harvesting"):
        try:
            batch = extract_batch_data(next(train_iter)).to(device)
        except StopIteration:
            logger.info(
                f"Dataset exhausted at batch {batch_idx}/{config.n_batches}. Finishing early."
            )
            break

        with torch.no_grad():
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

    logger.info(f"Batch processing complete. Total tokens: {harvester.total_tokens_processed:,}")

    result = _build_harvest_result(harvester, config)
    result.save(activation_contexts_dir, correlations_dir)
    logger.info(f"Saved results to {activation_contexts_dir} and {correlations_dir}")


def _harvest_worker(
    rank: int,
    world_size: int,
    wandb_path: str,
    n_batches: int,
    batch_size: int,
    ci_threshold: float,
    activation_examples_per_component: int,
    activation_context_tokens_per_side: int,
    state_dir: Path,
) -> None:
    """Worker function for parallel harvesting. Runs in subprocess."""
    from spd.data import train_loader_and_tokenizer

    device = torch.device(f"cuda:{rank}")
    logger.info(f"[Worker {rank}] Starting on {device}")

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, batch_size)

    layer_names = list(model.target_module_paths)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    u_norms = _compute_u_norms(model)

    harvester = Harvester(
        layer_names=layer_names,
        c_per_layer=model.module_to_c,
        vocab_size=vocab_size,
        ci_threshold=ci_threshold,
        max_examples_per_component=activation_examples_per_component,
        context_tokens_per_side=activation_context_tokens_per_side,
        device=device,
    )

    train_iter = iter(train_loader)
    batches_processed = 0
    last_log_time = time.time()
    for batch_idx in range(n_batches):
        try:
            batch_data = extract_batch_data(next(train_iter))
        except StopIteration:
            logger.info(
                f"[Worker {rank}] Dataset exhausted at batch {batch_idx}/{n_batches}. "
                f"Finishing early.",
            )
            break
        if batch_idx % world_size != rank:
            continue

        batch = batch_data.to(device)
        with torch.no_grad():
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
        if now - last_log_time >= 10:
            logger.info(f"[Worker {rank}] {batches_processed} batches")
            last_log_time = now

    logger.info(
        f"[Worker {rank}] Done. {batches_processed} batches, "
        f"{harvester.total_tokens_processed:,} tokens",
    )
    state = harvester.get_state()
    state_path = state_dir / f"worker_{rank}.pt"
    torch.save(state, state_path)
    logger.info(f"[Worker {rank}] Saved state to {state_path}")

    # Explicitly clean up CUDA resources to avoid slow process exit
    del model, harvester
    torch.cuda.empty_cache()


def harvest_parallel(
    config: HarvestConfig,
    n_gpus: int,
    activation_contexts_dir: Path,
    correlations_dir: Path,
) -> None:
    """Parallel harvest across multiple GPUs using multiprocessing."""
    import tempfile

    import torch.multiprocessing as mp

    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import ComponentModel, SPDRunInfo

    # Pre-cache model and dataset before spawning workers
    logger.info("Pre-caching model and dataset...")
    run_info = SPDRunInfo.from_path(config.wandb_path)
    _ = ComponentModel.from_run_info(run_info)
    _, _ = train_loader_and_tokenizer(run_info.config, config.batch_size)
    logger.info("Pre-caching complete. Spawning workers...")

    mp.set_start_method("spawn", force=True)

    with tempfile.TemporaryDirectory() as state_dir:
        state_dir_path = Path(state_dir)

        processes = []
        for rank in range(n_gpus):
            p = mp.Process(
                target=_harvest_worker,
                args=(
                    rank,
                    n_gpus,
                    config.wandb_path,
                    config.n_batches,
                    config.batch_size,
                    config.ci_threshold,
                    config.activation_examples_per_component,
                    config.activation_context_tokens_per_side,
                    state_dir_path,
                ),
            )
            p.start()
            processes.append(p)

        logger.info(f"Launched {n_gpus} workers. Waiting for completion...")
        for i, p in enumerate(processes):
            logger.info(f"Joining worker {i}...")
            p.join()
            logger.info(f"Worker {i} joined (exit code {p.exitcode})")
            if p.exitcode != 0:
                # Kill remaining workers and fail fast
                for remaining in processes[i + 1 :]:
                    remaining.terminate()
                    remaining.join()
                raise RuntimeError(
                    f"Worker {p.pid} failed with exit code {p.exitcode}. "
                    "Check stderr above for traceback."
                )

        logger.info("All workers finished. Loading states from disk...")
        states = []
        for rank in range(n_gpus):
            state_path = state_dir_path / f"worker_{rank}.pt"
            states.append(torch.load(state_path, weights_only=False))

    logger.info("Merging states...")
    merged_state = HarvesterState.merge(states)
    logger.info(f"Merged. Total tokens: {merged_state.total_tokens_processed:,}")

    harvester = Harvester.from_state(merged_state, torch.device("cpu"))

    result = _build_harvest_result(harvester, config)
    result.save(activation_contexts_dir, correlations_dir)
    logger.info(f"Saved results to {activation_contexts_dir} and {correlations_dir}")
