"""Phase 3: Test model loading with distributed setup.

This script tests loading the pretrained model using both:
1. Direct loading (all ranks load simultaneously)
2. ensure_cached_and_call pattern (rank 0 first, then others)

Uses the ss_llama_simple_mlp-1L config model.
"""

import os
import time

import torch
from simple_stories_train.run_info import RunInfo as SSRunInfo

from spd.utils.distributed_utils import (
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


def log(msg: str) -> None:
    """Log with rank prefix and timestamp."""
    state = get_distributed_state()
    rank = state.rank if state else 0
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()

# Model path from ss_llama_simple_mlp-1L.yaml
PRETRAINED_MODEL_NAME = "wandb:goodfire/spd/runs/gvbmdt9w"


def main() -> None:
    print(f"[RANK ?] [{0.0:7.2f}s] Script started", flush=True)

    # Initialize distributed
    print(f"[RANK ?] [{time.time() - START_TIME:7.2f}s] Starting init_distributed...", flush=True)
    t0 = time.time()
    dist_state = init_distributed()
    t1 = time.time()
    log(f"init_distributed completed in {t1 - t0:.2f}s")

    device = get_device()
    log(f"Device: {device}")

    # Barrier before model loading
    sync_across_processes()
    log("Initial barrier complete")

    # =========================================================================
    # Test 1: Load SSRunInfo using call_on_rank0_then_broadcast
    # =========================================================================
    log("=" * 50)
    log("TEST 1: call_on_rank0_then_broadcast for SSRunInfo")
    log("=" * 50)

    t0 = time.time()
    run_info = call_on_rank0_then_broadcast(SSRunInfo.from_path, PRETRAINED_MODEL_NAME)
    t1 = time.time()
    log(f"SSRunInfo loaded in {t1 - t0:.2f}s")
    log(f"Model type: {run_info.config_dict.get('model_class', 'unknown')}")

    sync_across_processes()

    # =========================================================================
    # Test 2: Load actual model using ensure_cached_and_call
    # =========================================================================
    log("=" * 50)
    log("TEST 2: ensure_cached_and_call for model loading")
    log("=" * 50)

    # Import the model class
    from simple_stories_train.models.llama_simple_mlp import LlamaSimpleMLP

    t0 = time.time()
    model = ensure_cached_and_call(LlamaSimpleMLP.from_run_info, run_info)
    t1 = time.time()
    log(f"Model loaded via ensure_cached_and_call in {t1 - t0:.2f}s")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")

    sync_across_processes()

    # =========================================================================
    # Test 3: Direct model loading (all ranks simultaneously)
    # =========================================================================
    log("=" * 50)
    log("TEST 3: Direct model loading (all ranks at once)")
    log("=" * 50)

    # Clear the model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sync_across_processes()

    t0 = time.time()
    model2 = LlamaSimpleMLP.from_run_info(run_info)
    t1 = time.time()
    log(f"Model loaded directly in {t1 - t0:.2f}s")

    sync_across_processes()

    # =========================================================================
    # Test 4: Move model to device
    # =========================================================================
    log("=" * 50)
    log("TEST 4: Move model to device")
    log("=" * 50)

    t0 = time.time()
    model2 = model2.to(device)
    t1 = time.time()
    log(f"Model moved to {device} in {t1 - t0:.2f}s")

    sync_across_processes()

    # Save rank before cleanup
    state = get_distributed_state()
    rank = state.rank if state else 0

    # Cleanup
    cleanup_distributed()

    total_time = time.time() - START_TIME
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 3 COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
