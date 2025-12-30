"""Phase 1: Minimal distributed init to test NCCL/torchrun baseline.

This script tests the bare minimum distributed setup to establish a baseline.
If this is slow, the issue is in NCCL/torchrun itself, not our code.
"""

import os
import time

import torch
import torch.distributed as dist


def log(msg: str) -> None:
    """Log with rank prefix and timestamp."""
    rank = int(os.environ.get("RANK", 0))
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()


def main() -> None:
    log("Script started")

    # Get distributed info from environment (set by torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    log(f"Environment: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    if world_size > 1:
        # Set device before init_process_group
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        log(f"Set CUDA device to {device}")

        # Time the init_process_group call
        log("Starting init_process_group...")
        t0 = time.time()

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        t1 = time.time()
        log(f"init_process_group completed in {t1 - t0:.2f}s")

        # Test a simple all_reduce
        log("Testing all_reduce...")
        t0 = time.time()

        tensor = torch.tensor([rank], device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        t1 = time.time()
        log(f"all_reduce completed in {t1 - t0:.2f}s, result={tensor.item()}")

        # Test barrier
        log("Testing barrier...")
        t0 = time.time()

        dist.barrier()

        t1 = time.time()
        log(f"barrier completed in {t1 - t0:.2f}s")

        # Cleanup
        dist.destroy_process_group()
        log("Distributed cleanup complete")
    else:
        log("Running in single-process mode, skipping distributed setup")

    total_time = time.time() - START_TIME
    log(f"=== PHASE 1 COMPLETE: Total time {total_time:.2f}s ===")


if __name__ == "__main__":
    main()
