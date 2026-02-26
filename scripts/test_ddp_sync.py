"""Test whether DDP syncs initial parameters from rank 0 to all ranks."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    model = nn.Linear(4, 4, bias=False)

    # Give each rank different initial weights
    with torch.no_grad():
        model.weight.fill_(float(rank + 1))

    print(f"[Rank {rank}] BEFORE DDP: weight mean = {model.weight.mean().item()}")

    wrapped = nn.parallel.DistributedDataParallel(model)

    print(f"[Rank {rank}] AFTER DDP:  weight mean = {wrapped.module.weight.mean().item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
