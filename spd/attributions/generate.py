"""Generate local attribution database from a trained SPD model.

Single entrypoint that:
1. Loads model and computes activation contexts once
2. Spawns parallel workers (one per GPU) for attribution computation
3. All workers write to same SQLite DB (WAL mode handles concurrency)

Usage:
    python -m spd.attributions.generate \\
        --wandb_path wandb:goodfire/spd/runs/jyo9duz5 \\
        --n_prompts 100 \\
        --n_gpus 4 \\
        --output_path ./local_attr.db
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.services.run_context_service import TrainRunContext, _build_token_lookup
from spd.attributions.compute import (
    PairAttribution,
    compute_local_attributions,
    get_sources_by_target,
)
from spd.attributions.db import ComponentActivation, LocalAttrDB
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import SPDRunInfo


@dataclass
class GenerateConfig:
    """Configuration for attribution generation."""

    wandb_path: str
    n_prompts: int
    output_path: Path
    n_gpus: int = 1
    n_blocks: int = 4
    n_ctx: int = 32
    ci_threshold: float = 1e-6
    attr_threshold: float = 1e-4
    seed: int = 42
    n_batches_act_ctx: int = 50
    act_ctx_importance_threshold: float = 0.1


def sparsify_attribution_cross_seq(attr_tensor: Tensor, threshold: float) -> list[list[float]]:
    """Sparse format for cross-seq pairs: [[s_in, c_in, s_out, c_out, val], ...]"""
    sparse: list[list[float]] = []
    attr_np = attr_tensor.cpu().numpy()
    for s_in in range(attr_np.shape[0]):
        for c_in in range(attr_np.shape[1]):
            for s_out in range(attr_np.shape[2]):
                for c_out in range(attr_np.shape[3]):
                    val = float(attr_np[s_in, c_in, s_out, c_out])
                    if abs(val) >= threshold:
                        sparse.append([s_in, c_in, s_out, c_out, round(val, 6)])
    return sparse


def sparsify_attribution_same_seq(attr_tensor: Tensor, threshold: float) -> list[list[float]]:
    """Sparse format for same-seq pairs: [[s, c_in, c_out, val], ...]"""
    sparse: list[list[float]] = []
    attr_np = attr_tensor.cpu().numpy()
    n_seq = attr_np.shape[0]
    for s in range(n_seq):
        for c_in in range(attr_np.shape[1]):
            for c_out in range(attr_np.shape[3]):
                val = float(attr_np[s, c_in, s, c_out])
                if abs(val) >= threshold:
                    sparse.append([s, c_in, c_out, round(val, 6)])
    return sparse


def serialize_pair(pair: PairAttribution, threshold: float) -> dict[str, Any]:
    """Serialize a pair with appropriate sparse format."""
    if pair.is_kv_to_o_pair:
        return {
            "source": pair.source,
            "target": pair.target,
            "is_cross_seq": True,
            "attribution": sparsify_attribution_cross_seq(pair.attribution, threshold),
            "trimmed_c_in_idxs": pair.trimmed_c_in_idxs,
            "trimmed_c_out_idxs": pair.trimmed_c_out_idxs,
        }
    else:
        return {
            "source": pair.source,
            "target": pair.target,
            "is_cross_seq": False,
            "attribution": sparsify_attribution_same_seq(pair.attribution, threshold),
            "trimmed_c_in_idxs": pair.trimmed_c_in_idxs,
            "trimmed_c_out_idxs": pair.trimmed_c_out_idxs,
        }


def extract_active_components(attr_pairs: list[PairAttribution]) -> dict[str, ComponentActivation]:
    """Extract which components are active in this prompt for the inverted index."""
    active: dict[str, ComponentActivation] = {}

    for pair in attr_pairs:
        for c_local, c_idx in enumerate(pair.trimmed_c_in_idxs):
            key = f"{pair.source}:{c_idx}"
            attr_slice = pair.attribution[:, c_local, :, :]
            positions = torch.where(attr_slice.abs().sum(dim=(1, 2)) > 0)[0].tolist()
            if positions:
                if key not in active:
                    active[key] = ComponentActivation(
                        prompt_id=-1, component_key=key, max_ci=0.0, positions=[]
                    )
                active[key].positions = list(set(active[key].positions + positions))

        for c_local, c_idx in enumerate(pair.trimmed_c_out_idxs):
            key = f"{pair.target}:{c_idx}"
            attr_slice = pair.attribution[:, :, :, c_local]
            positions = torch.where(attr_slice.abs().sum(dim=(0, 1)) > 0)[0].tolist()
            if positions:
                if key not in active:
                    active[key] = ComponentActivation(
                        prompt_id=-1, component_key=key, max_ci=0.0, positions=[]
                    )
                active[key].positions = list(set(active[key].positions + positions))

    return active


def worker_fn(
    worker_id: int,
    n_workers: int,
    config: GenerateConfig,
    sources_by_target: dict[str, list[str]],
    start_prompt_idx: int,
    n_prompts_for_worker: int,
) -> None:
    """Worker function that runs in a separate process on its own GPU."""
    device = f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {worker_id}] Starting on {device}, processing {n_prompts_for_worker} prompts")

    # Load model onto this worker's GPU
    run_info = SPDRunInfo.from_path(config.wandb_path)
    from spd.models.component_model import ComponentModel

    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    spd_config = run_info.config
    assert spd_config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # Create data loader - use worker-specific seed offset
    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=config.n_ctx,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=config.seed,
    )
    data_loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=1,
        buffer_size=task_config.buffer_size,
        global_seed=config.seed,
    )

    # Skip to this worker's starting position
    data_iter = iter(data_loader)
    for _ in range(start_prompt_idx):
        next(data_iter)

    # Open DB connection for this worker
    db = LocalAttrDB(config.output_path)

    # Process prompts
    for _ in tqdm(range(n_prompts_for_worker), desc=f"Worker {worker_id}", position=worker_id):
        batch = next(data_iter)
        tokens: Float[Tensor, "1 seq"] = batch["input_ids"].to(device)

        attr_pairs = compute_local_attributions(
            model=model,
            tokens=tokens,
            sources_by_target=sources_by_target,
            ci_threshold=config.ci_threshold,
            sampling=spd_config.sampling,
            device=device,
            show_progress=False,
        )

        token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
        pairs_data = [serialize_pair(pair, config.attr_threshold) for pair in attr_pairs]
        active_components = extract_active_components(attr_pairs)

        db.add_prompt(
            tokens=token_strings,
            pairs=pairs_data,
            active_components=active_components,
        )

        # Skip prompts assigned to other workers
        for _ in range(n_workers - 1):
            try:
                next(data_iter)
            except StopIteration:
                break

    db.close()
    print(f"[Worker {worker_id}] Done")


def generate_database(config: GenerateConfig) -> None:
    """Main entry point for database generation."""
    print("=" * 60)
    print("Generating local attribution database")
    print(f"  wandb_path: {config.wandb_path}")
    print(f"  n_prompts: {config.n_prompts}")
    print(f"  n_gpus: {config.n_gpus}")
    print(f"  output_path: {config.output_path}")
    print("=" * 60)

    # Initialize DB and check existing state
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    db = LocalAttrDB(config.output_path)
    db.init_schema()

    existing_count = db.get_prompt_count()
    if existing_count >= config.n_prompts:
        print(
            f"\nAlready have {existing_count} prompts (target: {config.n_prompts}), nothing to do."
        )
        db.close()
        return

    # Store metadata
    db.set_meta("wandb_path", {"path": config.wandb_path})
    db.set_meta("n_blocks", {"n_blocks": config.n_blocks})

    # Load model once in main process to ensure cache is populated before workers spawn
    # This prevents race conditions when multiple workers try to download simultaneously
    print("\nLoading model (ensures cache is populated for workers)...")
    run_info = SPDRunInfo.from_path(config.wandb_path)
    from spd.models.component_model import ComponentModel

    model = ComponentModel.from_run_info(run_info)

    # Compute activation contexts if not present
    existing_act_ctx = db.get_activation_contexts()
    if existing_act_ctx is not None:
        print(f"Activation contexts already in DB ({len(existing_act_ctx)} layers), skipping...")
    else:
        print("\nComputing activation contexts (this only happens once)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        run_info = SPDRunInfo.from_path(config.wandb_path)
        from spd.models.component_model import ComponentModel

        model = ComponentModel.from_run_info(run_info)
        model = model.to(device)
        model.eval()

        spd_config = run_info.config
        assert spd_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
        token_string_lookup = _build_token_lookup(tokenizer, spd_config.tokenizer_name)

        task_config = spd_config.task_config
        assert isinstance(task_config, LMTaskConfig)
        dataset_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=spd_config.tokenizer_name,
            split=task_config.train_data_split,
            n_ctx=config.n_ctx,
            is_tokenized=task_config.is_tokenized,
            streaming=task_config.streaming,
            column_name=task_config.column_name,
            shuffle_each_epoch=False,
            seed=config.seed,
        )
        act_ctx_loader, _ = create_data_loader(
            dataset_config=dataset_config,
            batch_size=1,
            buffer_size=task_config.buffer_size,
            global_seed=config.seed,
        )

        run_context = TrainRunContext(
            wandb_id=config.wandb_path.split("/")[-1],
            wandb_path=config.wandb_path,
            config=spd_config,
            cm=model,
            tokenizer=tokenizer,
            train_loader=act_ctx_loader,
            token_strings=token_string_lookup,
        )

        activation_contexts = None
        for result in get_activations_data_streaming(
            run_context=run_context,
            importance_threshold=config.act_ctx_importance_threshold,
            n_batches=config.n_batches_act_ctx,
            n_tokens_either_side=5,
            batch_size=8,
            topk_examples=10,
        ):
            if result[0] == "complete":
                activation_contexts = result[1]
                break

        assert activation_contexts is not None
        act_ctx_dict = {
            layer_name: [subcomp.model_dump() for subcomp in subcomps]
            for layer_name, subcomps in activation_contexts.layers.items()
        }
        db.set_activation_contexts(act_ctx_dict)
        print(f"  Stored activation contexts for {len(act_ctx_dict)} layers")

        # Clean up model from main process (workers will load their own)
        del model
        torch.cuda.empty_cache()

    # Build sources_by_target mapping (need model briefly)
    print("\nBuilding layer mapping...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_info = SPDRunInfo.from_path(config.wandb_path)
    from spd.models.component_model import ComponentModel

    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()
    sources_by_target = get_sources_by_target(
        model, device, run_info.config.sampling, config.n_blocks
    )
    del model
    torch.cuda.empty_cache()

    db.close()

    # Calculate work distribution
    remaining = config.n_prompts - existing_count
    prompts_per_worker = remaining // config.n_gpus
    extra = remaining % config.n_gpus

    print(f"\nDistributing {remaining} prompts across {config.n_gpus} workers...")

    if config.n_gpus == 1:
        # Single worker - run directly
        worker_fn(
            worker_id=0,
            n_workers=1,
            config=config,
            sources_by_target=sources_by_target,
            start_prompt_idx=existing_count,
            n_prompts_for_worker=remaining,
        )
    else:
        # Spawn workers
        processes = []
        for worker_id in range(config.n_gpus):
            n_for_this_worker = prompts_per_worker + (1 if worker_id < extra else 0)
            if n_for_this_worker == 0:
                continue

            p = mp.Process(
                target=worker_fn,
                args=(
                    worker_id,
                    config.n_gpus,
                    config,
                    sources_by_target,
                    existing_count + worker_id,
                    n_for_this_worker,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Final stats
    db = LocalAttrDB(config.output_path)
    final_count = db.get_prompt_count()
    db.close()
    print(f"\nDone! Database has {final_count} prompts")


if __name__ == "__main__":
    import fire

    def config(
        wandb_path: str,
        n_prompts: int,
        output_path: str,
        n_gpus: int = 1,
        n_blocks: int = 4,
        n_ctx: int = 32,
        ci_threshold: float = 1e-6,
        attr_threshold: float = 1e-4,
        seed: int = 42,
        n_batches_act_ctx: int = 50,
        act_ctx_importance_threshold: float = 0.1,
    ) -> GenerateConfig:
        return GenerateConfig(
            wandb_path=wandb_path,
            n_prompts=n_prompts,
            output_path=Path(output_path),
            n_gpus=n_gpus,
            n_blocks=n_blocks,
            n_ctx=n_ctx,
            ci_threshold=ci_threshold,
            attr_threshold=attr_threshold,
            seed=seed,
            n_batches_act_ctx=n_batches_act_ctx,
            act_ctx_importance_threshold=act_ctx_importance_threshold,
        )

    mp.set_start_method("spawn", force=True)

    generate_database(fire.Fire(config))
