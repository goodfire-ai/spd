#!/usr/bin/env python3
"""Manual test script for distributed training.

Run with:
    # Single GPU (non-distributed)
    python tests/manual_test_ddp.py

    # Multi-GPU with mpirun
    mpirun -np 2 python tests/manual_test_ddp.py

    # Multi-GPU with torchrun (for comparison, though we use mpirun)
    torchrun --nproc_per_node=2 tests/manual_test_ddp.py
"""

import os
import sys

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_device,
    init_distributed,
    is_main_process,
    sync_across_processes,
)


def main():
    """Test basic distributed functionality."""
    # Initialize distributed
    rank, world_size, _ = init_distributed()

    device = get_device()

    print(f"Process {rank}/{world_size} initialized on device {device}")

    # Create a simple model
    model = nn.Linear(10, 5).to(device)

    if world_size > 1:
        # Wrap with DDP
        device_id = int(device.split(":")[1]) if ":" in device else 0
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id],
            output_device=device_id,
        )
        print(f"Rank {rank}: Model wrapped with DDP")

    # Create different input on each rank
    torch.manual_seed(42 + rank)
    x = torch.randn(4, 10).to(device)

    # Forward pass
    output = model(x)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients are synchronized
    grad = model.module.weight.grad if world_size > 1 else model.weight.grad  # pyright: ignore[reportAttributeAccessIssue]

    assert isinstance(grad, torch.Tensor)
    grad_sum = grad.sum().item()
    print(f"Rank {rank}: Gradient sum = {grad_sum:.4f}")

    # Synchronize before printing final message
    sync_across_processes()

    if is_main_process():
        print(f"\nDistributed test completed successfully with {world_size} processes!")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
