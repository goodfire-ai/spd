"""Harvest token stats and activation contexts for autointerp.

Collects per-component statistics in a single pass over the data:
- Input/output token PMI (pointwise mutual information)
- Activation examples with context windows
- Firing counts and CI sums

Performance (SimpleStories, 600M tokens, batch_size=256, context_length=512):
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

from spd.autointerp.lib.harvester import Harvester, HarvesterState
from spd.autointerp.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
)
from spd.utils.general_utils import extract_batch_data


@dataclass
class HarvestConfig:
    wandb_path: str
    n_batches: int
    batch_size: int
    context_length: int
    ci_threshold: float
    activation_examples_per_component: int
    activation_context_tokens_per_side: int
    pmi_token_top_k: int


@dataclass
class HarvestResult:
    components: list[ComponentData]
    config: HarvestConfig

    def save(self, path: Path) -> None:
        """Save harvest result to disk."""
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / "config.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2))

        components_path = path / "components.jsonl"
        with open(components_path, "w") as f:
            for comp in self.components:
                f.write(json.dumps(asdict(comp)) + "\n")

    @staticmethod
    def load(path: Path) -> "HarvestResult":
        """Load harvest result from disk."""
        assert path.exists(), f"No harvest found at {path}"

        config_path = path / "config.json"
        config_data = json.loads(config_path.read_text())
        config = HarvestConfig(**config_data)

        components_path = path / "components.jsonl"
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

        return HarvestResult(components=components, config=config)


def harvest(config: HarvestConfig) -> HarvestResult:
    """Single-pass harvest of token stats and activation contexts."""
    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import ComponentModel, SPDRunInfo
    from spd.utils.distributed_utils import get_device

    device = torch.device(get_device())
    print(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(
        spd_config, config.context_length, config.batch_size
    )

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
    print("Building component results...")
    components = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    print(f"Built {len(components)} components (skipped components with no firings)")
    return HarvestResult(components=components, config=config)


def _harvest_worker(
    rank: int,
    world_size: int,
    wandb_path: str,
    n_batches: int,
    batch_size: int,
    context_length: int,
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
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, context_length, batch_size)

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


def harvest_parallel(config: HarvestConfig, n_gpus: int = 8) -> HarvestResult:
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
                    config.context_length,
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
    print("Building component results...")
    components = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
    print(f"Built {len(components)} components")

    return HarvestResult(components=components, config=config)
