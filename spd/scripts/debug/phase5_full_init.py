"""Phase 5: Full initialization mimicking lm_decomposition.py.

This script replicates the full initialization sequence from lm_decomposition.py
up to the point where training would start. This should reproduce the 5-minute
delay if all prior phases are fast.
"""

import os
import time
from pathlib import Path

import torch
from simple_stories_train.run_info import RunInfo as SSRunInfo

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import (
    DistributedState,
    call_on_rank0_then_broadcast,
    cleanup_distributed,
    ensure_cached_and_call,
    get_device,
    get_distributed_state,
    init_distributed,
    is_distributed,
    is_main_process,
    sync_across_processes,
)
from spd.utils.general_utils import resolve_class, set_seed


def log(msg: str) -> None:
    """Log with rank prefix and timestamp."""
    state = get_distributed_state()
    rank = state.rank if state else 0
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()

# Config path
CONFIG_PATH = Path(__file__).parent.parent.parent / "experiments/lm/ss_llama_simple_mlp-1L.yaml"


def main() -> None:
    print(f"[RANK ?] [{0.0:7.2f}s] Script started", flush=True)
    print(f"[RANK ?] [{0.0:7.2f}s] Config: {CONFIG_PATH}", flush=True)

    # =========================================================================
    # Step 1: Load config
    # =========================================================================
    t0 = time.time()
    config = Config.from_file(CONFIG_PATH)
    t1 = time.time()
    print(f"[RANK ?] [{t1 - START_TIME:7.2f}s] Config loaded in {t1 - t0:.2f}s", flush=True)

    # =========================================================================
    # Step 2: Initialize distributed (like lm_decomposition.py:46)
    # =========================================================================
    print(f"[RANK ?] [{time.time() - START_TIME:7.2f}s] Starting init_distributed...", flush=True)
    t0 = time.time()
    dist_state = init_distributed()
    t1 = time.time()
    log(f"init_distributed completed in {t1 - t0:.2f}s")

    # =========================================================================
    # Step 3: Set seed (like lm_decomposition.py:54)
    # =========================================================================
    t0 = time.time()
    set_seed(config.seed)
    t1 = time.time()
    log(f"set_seed completed in {t1 - t0:.4f}s")

    # =========================================================================
    # Step 4: Get device (like lm_decomposition.py:83)
    # =========================================================================
    t0 = time.time()
    device = get_device()
    t1 = time.time()
    log(f"get_device returned '{device}' in {t1 - t0:.4f}s")

    # =========================================================================
    # Step 5: Load pretrained model class (like lm_decomposition.py:86-89)
    # =========================================================================
    t0 = time.time()
    pretrained_model_class = resolve_class(config.pretrained_model_class)
    t1 = time.time()
    log(f"resolve_class completed in {t1 - t0:.4f}s")

    # =========================================================================
    # Step 6: Load SSRunInfo (like lm_decomposition.py:96)
    # =========================================================================
    log("Loading SSRunInfo via call_on_rank0_then_broadcast...")
    t0 = time.time()
    run_info = call_on_rank0_then_broadcast(SSRunInfo.from_path, config.pretrained_model_name)
    t1 = time.time()
    log(f"SSRunInfo loaded in {t1 - t0:.2f}s")

    # =========================================================================
    # Step 7: Load target model (like lm_decomposition.py:102)
    # =========================================================================
    log("Loading target model via from_run_info...")
    t0 = time.time()
    target_model = pretrained_model_class.from_run_info(run_info)
    target_model.eval()
    t1 = time.time()
    log(f"Target model loaded in {t1 - t0:.2f}s")

    n_params = sum(p.numel() for p in target_model.parameters())
    log(f"Model parameters: {n_params:,}")

    # =========================================================================
    # Step 8: Load training data (like lm_decomposition.py:126-155)
    # =========================================================================
    log("Creating train data loader...")
    from spd.configs import LMTaskConfig

    assert isinstance(config.task_config, LMTaskConfig)

    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    match dist_state:
        case DistributedState(world_size=world_size):
            train_rank_microbatch_size = config.microbatch_size // world_size
        case None:
            train_rank_microbatch_size = config.microbatch_size

    t0 = time.time()
    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_rank_microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        dist_state=dist_state,
    )
    t1 = time.time()
    log(f"Train data loader created in {t1 - t0:.2f}s")

    # =========================================================================
    # Step 9: Load eval data (like lm_decomposition.py:157-184)
    # =========================================================================
    log("Creating eval data loader...")
    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    match dist_state:
        case DistributedState(world_size=world_size):
            eval_rank_batch_size = config.eval_batch_size // world_size
        case None:
            eval_rank_batch_size = config.eval_batch_size

    t0 = time.time()
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
        dist_state=dist_state,
    )
    t1 = time.time()
    log(f"Eval data loader created in {t1 - t0:.2f}s")

    # =========================================================================
    # Step 10: Move model to device
    # =========================================================================
    log("Moving model to device...")
    t0 = time.time()
    target_model = target_model.to(device)
    t1 = time.time()
    log(f"Model moved to {device} in {t1 - t0:.2f}s")

    sync_across_processes()

    # =========================================================================
    # Step 11: Create ComponentModel (like run_spd.py:152-159)
    # =========================================================================
    log("Creating ComponentModel...")
    t0 = time.time()
    component_model = ComponentModel.from_config(
        target_model=target_model,
        config=config,
        device=device,
    )
    t1 = time.time()
    log(f"ComponentModel created in {t1 - t0:.2f}s")

    sync_across_processes()

    # =========================================================================
    # Step 12: Wrap in DDP (like run_spd.py:166-183)
    # =========================================================================
    if dist_state is not None:
        log("Wrapping model in DDP...")
        t0 = time.time()

        if dist_state.backend == "nccl":
            device_id = dist_state.local_rank
            wrapped_model = torch.nn.parallel.DistributedDataParallel(
                component_model,
                device_ids=[device_id],
                output_device=device_id,
            )
        else:
            wrapped_model = torch.nn.parallel.DistributedDataParallel(component_model)

        t1 = time.time()
        log(f"DDP wrapper created in {t1 - t0:.2f}s")
    else:
        wrapped_model = component_model
        log("Single-process mode, no DDP wrapping")

    sync_across_processes()

    # =========================================================================
    # Step 13: Get first batch (like run_spd.py:228-239)
    # =========================================================================
    log("Getting first batch...")
    t0 = time.time()
    first_batch = next(iter(train_loader))
    t1 = time.time()
    log(f"First batch retrieved in {t1 - t0:.2f}s")
    log(f"Batch shape: {first_batch['input_ids'].shape}")

    sync_across_processes()

    # =========================================================================
    # Step 14: Forward pass
    # =========================================================================
    log("Running first forward pass...")
    t0 = time.time()
    with torch.no_grad():
        input_ids = first_batch["input_ids"].to(device)
        _ = wrapped_model(input_ids)
    t1 = time.time()
    log(f"First forward pass completed in {t1 - t0:.2f}s")

    sync_across_processes()

    # Save rank before cleanup
    rank = dist_state.rank if dist_state else 0

    # Cleanup
    cleanup_distributed()

    total_time = time.time() - START_TIME
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 5 COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)
        print("\nIf this took ~5 minutes, the bottleneck is in the", flush=True)
        print("initialization steps above. Check timing for each step.", flush=True)


if __name__ == "__main__":
    main()
