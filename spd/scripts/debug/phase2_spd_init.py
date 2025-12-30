"""Phase 2: Test SPD's distributed utilities.

This script tests init_distributed() and sync_across_processes() from
spd/utils/distributed_utils.py to see if there's overhead compared to
the minimal Phase 1 baseline.
"""

import os
import time

import torch

from spd.utils.distributed_utils import (
    cleanup_distributed,
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


def main() -> None:
    print(f"[RANK ?] [{0.0:7.2f}s] Script started", flush=True)

    # Time init_distributed
    print(f"[RANK ?] [{time.time() - START_TIME:7.2f}s] Starting init_distributed...", flush=True)
    t0 = time.time()

    dist_state = init_distributed()

    t1 = time.time()
    log(f"init_distributed completed in {t1 - t0:.2f}s")
    log(f"Distributed state: {dist_state}")

    if is_distributed():
        # Test get_device
        log("Testing get_device...")
        t0 = time.time()
        device = get_device()
        t1 = time.time()
        log(f"get_device returned '{device}' in {t1 - t0:.4f}s")

        # Test sync_across_processes (barrier)
        log("Testing sync_across_processes...")
        t0 = time.time()
        sync_across_processes()
        t1 = time.time()
        log(f"sync_across_processes completed in {t1 - t0:.2f}s")

        # Test a simple tensor operation
        log("Testing tensor all_reduce...")
        t0 = time.time()
        assert dist_state is not None
        tensor = torch.tensor([dist_state.rank], device=device, dtype=torch.float32)

        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        t1 = time.time()
        log(f"all_reduce completed in {t1 - t0:.2f}s, result={tensor.item()}")

        # Another barrier
        log("Testing second sync_across_processes...")
        t0 = time.time()
        sync_across_processes()
        t1 = time.time()
        log(f"Second sync completed in {t1 - t0:.2f}s")

        # Cleanup - save rank before destroying state
        rank = dist_state.rank
        cleanup_distributed()
        print(f"[RANK {rank}] [{time.time() - START_TIME:7.2f}s] Distributed cleanup complete", flush=True)

        total_time = time.time() - START_TIME
        if rank == 0:
            print(f"\n{'='*60}", flush=True)
            print(f"PHASE 2 COMPLETE: Total time {total_time:.2f}s", flush=True)
            print(f"{'='*60}", flush=True)
    else:
        log("Running in single-process mode")
        total_time = time.time() - START_TIME
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 2 COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
