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
from jaxtyping import Float, Int
from torch import Tensor

from spd.autointerp.lib.harvester import Harvester, HarvesterState
from spd.autointerp.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
)
from spd.log import logger
from spd.utils.general_utils import extract_batch_data


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
class ComponentCorrelations:
    """Component co-occurrence data for correlation analysis."""

    component_keys: list[str]
    count_i: Int[Tensor, " n_components"]
    count_ij: Int[Tensor, "n_components n_components"]
    count_total: int

    def save(self, path: Path) -> None:
        """Save correlations to a .pt file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "count_i": self.count_i.cpu(),
                "count_ij": self.count_ij.cpu(),
                "count_total": self.count_total,
            },
            path,
        )
        logger.info(f"Saved component correlations to {path}")

    @classmethod
    def load(cls, path: Path) -> "ComponentCorrelations":
        """Load correlations from a .pt file using memory mapping."""
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            count_i=data["count_i"],
            count_ij=data["count_ij"],
            count_total=data["count_total"],
        )


@dataclass
class ComponentTokenStats:
    """Token statistics for all components."""

    component_keys: list[str]
    vocab_size: int
    n_tokens: int
    input_counts: Float[Tensor, "n_components vocab"]
    input_totals: Float[Tensor, " vocab"]
    output_counts: Float[Tensor, "n_components vocab"]
    output_totals: Float[Tensor, " vocab"]
    firing_counts: Float[Tensor, " n_components"]

    def save(self, path: Path) -> None:
        """Save token stats to a .pt file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "component_keys": self.component_keys,
                "vocab_size": self.vocab_size,
                "n_tokens": self.n_tokens,
                "input_counts": self.input_counts.cpu(),
                "input_totals": self.input_totals.cpu(),
                "output_counts": self.output_counts.cpu(),
                "output_totals": self.output_totals.cpu(),
                "firing_counts": self.firing_counts.cpu(),
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved token stats to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "ComponentTokenStats":
        """Load token stats from a .pt file using memory mapping."""
        data = torch.load(path, weights_only=True, mmap=True)
        return cls(
            component_keys=data["component_keys"],
            vocab_size=data["vocab_size"],
            n_tokens=data["n_tokens"],
            input_counts=data["input_counts"],
            input_totals=data["input_totals"],
            output_counts=data["output_counts"],
            output_totals=data["output_totals"],
            firing_counts=data["firing_counts"],
        )


@dataclass
class HarvestResult:
    """Result of harvest containing components, correlations, and token stats."""

    components: list[ComponentData]
    correlations: ComponentCorrelations
    token_stats: ComponentTokenStats
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

        # Save correlations (.pt)
        self.correlations.save(correlations_dir / "component_correlations.pt")

        # Save token stats (.pt)
        self.token_stats.save(correlations_dir / "token_stats.pt")

    @staticmethod
    def load_components(activation_contexts_dir: Path) -> tuple[list[ComponentData], HarvestConfig]:
        """Load components from disk."""
        assert activation_contexts_dir.exists(), f"No harvest found at {activation_contexts_dir}"

        config_path = activation_contexts_dir / "config.json"
        config_data = json.loads(config_path.read_text())
        config = HarvestConfig(**config_data)

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

        return components, config


def _build_harvest_result(
    harvester: Harvester,
    config: HarvestConfig,
) -> HarvestResult:
    """Build HarvestResult from a harvester."""
    print("Building component results...")
    components = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    print(f"Built {len(components)} components (skipped components with no firings)")

    # Build component keys list (same ordering as tensors)
    component_keys = [
        f"{layer}:{c}"
        for layer in harvester.layer_names
        for c in range(harvester.components_per_layer)
    ]

    correlations = ComponentCorrelations(
        component_keys=component_keys,
        count_i=harvester.firing_counts.long().cpu(),
        count_ij=harvester.count_ij.long().cpu(),
        count_total=harvester.total_tokens_processed,
    )

    token_stats = ComponentTokenStats(
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
    from spd.models.component_model import ComponentModel, SPDRunInfo
    from spd.utils.distributed_utils import get_device

    device = torch.device(get_device())
    print(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, config.batch_size)

    layer_names = list(model.target_module_paths)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    harvester = Harvester(
        layer_names=layer_names,
        components_per_layer=model.C,
        vocab_size=vocab_size,
        ci_threshold=config.ci_threshold,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        device=device,
    )

    train_iter = iter(train_loader)
    for _ in tqdm.tqdm(range(config.n_batches), desc="Harvesting"):
        batch = extract_batch_data(next(train_iter)).to(device)

        with torch.no_grad():
            out = model(batch, cache_type="input")
            probs = torch.softmax(out.output, dim=-1)

            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=spd_config.sampling,
            ).lower_leaky

            ci_flat: Float[Tensor, "B S n_comp"] = torch.cat(
                [ci_dict[layer] for layer in layer_names], dim=2
            )
            assert ci_flat.shape[2] == len(layer_names) * model.C

            harvester.process_batch(batch, ci_flat, probs)

    print(f"Batch processing complete. Total tokens: {harvester.total_tokens_processed:,}")

    result = _build_harvest_result(harvester, config)
    result.save(activation_contexts_dir, correlations_dir)
    print(f"Saved results to {activation_contexts_dir} and {correlations_dir}")


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
    from spd.models.component_model import ComponentModel, SPDRunInfo

    device = torch.device(f"cuda:{rank}")
    print(f"[Worker {rank}] Starting on {device}")

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, batch_size)

    layer_names = list(model.target_module_paths)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    harvester = Harvester(
        layer_names=layer_names,
        components_per_layer=model.C,
        vocab_size=vocab_size,
        ci_threshold=ci_threshold,
        max_examples_per_component=activation_examples_per_component,
        context_tokens_per_side=activation_context_tokens_per_side,
        device=device,
    )

    train_iter = iter(train_loader)
    batches_processed = 0
    progress_update_interval = 10
    last_progress_time = time.time() - progress_update_interval
    for batch_idx in range(n_batches):
        if time.time() - last_progress_time > progress_update_interval:
            print(f"[Worker {rank}] Processed {batches_processed} batches")
            last_progress_time = time.time()

        batch_data = extract_batch_data(next(train_iter))
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

            ci_flat: Float[Tensor, "B S n_comp"] = torch.cat(
                [ci_dict[layer] for layer in layer_names], dim=2
            )
            assert ci_flat.shape[2] == len(layer_names) * model.C
            harvester.process_batch(batch, ci_flat, probs)

        batches_processed += 1
        if batches_processed % 10 == 0:
            print(f"[Worker {rank}] Processed {batches_processed} batches")

    print(
        f"[Worker {rank}] Done. Processed {batches_processed} batches, "
        f"{harvester.total_tokens_processed:,} tokens"
    )
    state = harvester.get_state()
    state_path = state_dir / f"worker_{rank}.pt"
    torch.save(state, state_path)
    print(f"[Worker {rank}] Saved state to {state_path}")


def harvest_parallel(
    config: HarvestConfig,
    n_gpus: int,
    activation_contexts_dir: Path,
    correlations_dir: Path,
) -> None:
    """Parallel harvest across multiple GPUs using multiprocessing."""
    import tempfile

    import torch.multiprocessing as mp

    from spd.models.component_model import ComponentModel, SPDRunInfo

    # Pre-cache wandb files before spawning workers
    print("Pre-caching model files from wandb...")
    run_info = SPDRunInfo.from_path(config.wandb_path)
    _ = ComponentModel.from_run_info(run_info)
    print("Model files cached. Spawning workers...")

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

        print(f"Launched {n_gpus} workers. Waiting for completion...")
        for p in processes:
            p.join()

        print("All workers finished. Loading states from disk...")
        states = []
        for rank in range(n_gpus):
            state_path = state_dir_path / f"worker_{rank}.pt"
            states.append(torch.load(state_path, weights_only=False))

    print("Merging states...")
    merged_state = HarvesterState.merge(states)
    print(f"Merged. Total tokens: {merged_state.total_tokens_processed:,}")

    harvester = Harvester.from_state(merged_state, torch.device("cpu"))

    result = _build_harvest_result(harvester, config)
    result.save(activation_contexts_dir, correlations_dir)
    print(f"Saved results to {activation_contexts_dir} and {correlations_dir}")
