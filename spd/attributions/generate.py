"""Generate local attribution database from a trained SPD model.

Single entrypoint that:
1. Loads model and computes activation contexts once
2. Spawns parallel workers (one per GPU) for CI computation
3. All workers write to same SQLite DB (WAL mode handles concurrency)

Attribution graphs are computed on-demand at serve time, not during generation.
This makes generation much faster since CI computation is cheap and batchable.

Usage:
    python -m spd.attributions.generate \\
        --wandb_path wandb:goodfire/spd/runs/jyo9duz5 \\
        --n_prompts 100 \\
        --n_gpus 4 \\
        --output_path ./local_attr.db
"""

import fcntl
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.services.run_context_service import TrainRunContext, _build_token_lookup
from spd.attributions.compute import compute_ci_only, extract_active_from_ci
from spd.attributions.db import LocalAttrDB
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import SPDRunInfo


@dataclass
class GenerateConfig:
    """Configuration for CI-only database generation."""

    wandb_path: str
    n_prompts: int
    output_path: Path
    n_gpus: int = 1
    n_blocks: int = 4
    n_ctx: int = 32
    ci_threshold: float = 1e-6
    output_prob_threshold: float = 0.01
    seed: int = 42
    n_batches_act_ctx: int = 50
    act_ctx_importance_threshold: float = 0.1


def worker_fn(
    worker_id: int,
    n_workers: int,
    config: GenerateConfig,
    start_prompt_idx: int,
    n_prompts_for_worker: int,
) -> None:
    """Worker function that runs in a separate process on its own GPU.

    Uses CI-only computation - much faster than full attribution graph computation.
    """
    device = f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {worker_id}] Starting on {device}, processing {n_prompts_for_worker} prompts")

    # Load model onto this worker's GPU
    # Use file lock to prevent race conditions when loading pretrained weights
    lock_path = config.output_path.parent / ".model_load.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        run_info = SPDRunInfo.from_path(config.wandb_path)
        from spd.models.component_model import ComponentModel

        model = ComponentModel.from_run_info(run_info)
        model = model.to(device)
        model.eval()
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    spd_config = run_info.config

    # Create data loader - use worker-specific seed offset
    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)
    assert spd_config.tokenizer_name is not None
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
        token_ids: Int[Tensor, "1 seq"] = next(data_iter)["input_ids"].to(device)

        # Fast CI-only computation (no gradient loop)
        result = compute_ci_only(
            model=model,
            tokens=token_ids,
            sampling=spd_config.sampling,
        )

        # Extract active components from CI
        n_seq = token_ids.shape[1]
        active_components = extract_active_from_ci(
            ci_lower_leaky=result.ci_lower_leaky,
            output_probs=result.output_probs,
            ci_threshold=config.ci_threshold,
            output_prob_threshold=config.output_prob_threshold,
            n_seq=n_seq,
        )

        # Store token IDs (not strings) - decode only at display time
        db.add_prompt_simple(
            token_ids=token_ids[0].tolist(),
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
    """Main entry point for database generation.

    Uses CI-only computation for fast generation. Attribution graphs
    are computed on-demand at serve time.
    """
    print("=" * 60)
    print("Generating local attribution database (CI-only)")
    print(f"  wandb_path: {config.wandb_path}")
    print(f"  n_prompts: {config.n_prompts}")
    print(f"  n_gpus: {config.n_gpus}")
    print(f"  output_path: {config.output_path}")
    print("=" * 60)

    # Initialize DB and check existing state
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    db = LocalAttrDB(config.output_path)
    db.init_schema_simple()

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
        output_prob_threshold: float = 0.01,
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
            output_prob_threshold=output_prob_threshold,
            seed=seed,
            n_batches_act_ctx=n_batches_act_ctx,
            act_ctx_importance_threshold=act_ctx_importance_threshold,
        )

    mp.set_start_method("spawn", force=True)

    generate_database(fire.Fire(config))
