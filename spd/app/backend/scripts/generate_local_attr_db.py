"""Generate local attribution database from a trained SPD model.

Single entrypoint that:
1. Loads model and computes activation contexts once
2. Spawns parallel workers (one per GPU) for CI computation
3. All workers write to same SQLite DB (WAL mode handles concurrency)

Attribution graphs are computed on-demand at serve time, not during generation.
This makes generation much faster since CI computation is cheap and batchable.

Usage:
    python -m spd.app.backend.scripts.generate_local_attr_db \
        --wandb_path wandb:goodfire/spd/runs/jyo9duz5 \
        --n_prompts 100 \
        --n_gpus 4 \
        --output_path ./local_attr.db
"""

import fcntl
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Int
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.db import LocalAttrDB
from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.schemas import ActivationContextsGenerationConfig
from spd.app.backend.services.run_context_service import TrainRunContext, _build_token_lookup
from spd.attributions.compute import compute_ci_only, extract_active_from_ci
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo


@dataclass
class GenerateConfig:
    """Configuration for CI-only database generation."""

    wandb_path: str
    n_prompts: int
    output_path: Path
    n_gpus: int = 1
    n_blocks: int = 4
    n_ctx: int = 8
    ci_threshold: float = 1e-6
    output_prob_threshold: float = 0.01
    seed: int = 42
    act_ctx_n_batches: int = 50
    act_ctx_n_tokens_either_side: int = 5
    act_ctx_batch_size: int = 128
    act_ctx_topk_examples: int = 100
    act_ctx_importance_threshold: float = 0.1
    act_ctx_separation_tokens: int = 0


def worker_fn(
    worker_id: int,
    n_workers: int,
    config: GenerateConfig,
    run_id: int,
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
        db.add_prompt(
            run_id=run_id,
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
    db.init_schema()

    # Get or create run
    run_id = db.get_or_create_run(config.wandb_path, config.n_blocks)
    print(f"Using run_id={run_id} for wandb_path={config.wandb_path}")

    existing_count = db.get_prompt_count(run_id)
    if existing_count > 0:
        print(f"\nRun already has {existing_count} prompts. Continue adding more?")
        input_ = input("y to continue, n to exit: ")
        if input_ != "y":
            print("Exiting...")
            db.close()
            return

    # Load model once in main process to ensure cache is populated before workers spawn
    # This prevents race conditions when multiple workers try to download simultaneously
    print("\nLoading model (ensures cache is populated for workers)...")
    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info)

    # Compute activation contexts if not present
    existing_act_ctx = db.get_activation_contexts(run_id)
    if existing_act_ctx is not None:
        print(f"Activation contexts already in DB ({len(existing_act_ctx.layers)} layers), skipping...")
    else:
        print("\nComputing activation contexts (this only happens once)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()

        spd_config = run_info.config
        assert spd_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
        tokenizer_base = cast(PreTrainedTokenizerBase, cast(object, tokenizer))
        token_string_lookup = _build_token_lookup(tokenizer_base, spd_config.tokenizer_name)

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
            tokenizer=tokenizer_base,
            train_loader=act_ctx_loader,
            token_strings=token_string_lookup,
        )

        activation_contexts = None
        for result in get_activations_data_streaming(
            run_context=run_context,
            importance_threshold=config.act_ctx_importance_threshold,
            n_batches=config.act_ctx_n_batches,
            n_tokens_either_side=config.act_ctx_n_tokens_either_side,
            batch_size=config.act_ctx_batch_size,
            topk_examples=config.act_ctx_topk_examples,
            separation_tokens=config.act_ctx_separation_tokens,
        ):
            if result[0] == "complete":
                activation_contexts = result[1]
                break

        assert activation_contexts is not None

        # Store with generation config
        gen_config = ActivationContextsGenerationConfig(
            importance_threshold=config.act_ctx_importance_threshold,
            n_batches=config.act_ctx_n_batches,
            batch_size=config.act_ctx_batch_size,
            n_tokens_either_side=config.act_ctx_n_tokens_either_side,
            topk_examples=config.act_ctx_topk_examples,
            separation_tokens=config.act_ctx_separation_tokens,
        )
        db.set_activation_contexts(run_id, activation_contexts, gen_config)
        print(f"  Stored activation contexts for {len(activation_contexts.layers)} layers")

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
            run_id=run_id,
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
                    run_id,
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
    final_count = db.get_prompt_count(run_id)
    db.close()
    print(f"\nDone! Run {run_id} has {final_count} prompts")


if __name__ == "__main__":
    import fire

    def config(
        wandb_path: str,
        n_prompts: int,
        output_path: str,
        n_gpus: int = 1,
        n_blocks: int = 4,
        n_ctx: int = 8,
        ci_threshold: float = 1e-6,
        output_prob_threshold: float = 0.01,
        seed: int = 42,
        act_ctx_n_batches: int = 50,
        act_ctx_n_tokens_either_side: int = 8,
        act_ctx_batch_size: int = 32,
        act_ctx_topk_examples: int = 100,
        act_ctx_importance_threshold: float = 0.3,
        act_ctx_separation_tokens: int = 0,
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
            act_ctx_n_batches=act_ctx_n_batches,
            act_ctx_n_tokens_either_side=act_ctx_n_tokens_either_side,
            act_ctx_batch_size=act_ctx_batch_size,
            act_ctx_topk_examples=act_ctx_topk_examples,
            act_ctx_importance_threshold=act_ctx_importance_threshold,
            act_ctx_separation_tokens=act_ctx_separation_tokens,
        )

    mp.set_start_method("spawn", force=True)

    generate_database(fire.Fire(config))
