"""Utilities for distributed data parallel training with MPI support."""

import os
import time
from collections.abc import Mapping
from typing import Literal

import torch
import torch.distributed as dist
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp


def init_distributed(backend: Literal["nccl", "gloo"] | None = None) -> tuple[int, int, int]:
    """Initialize distributed process group using MPI.

    Supports OpenMPI only.

    Args:
        backend: Distributed backend to use ('nccl' or 'gloo').

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    backend = backend if backend is not None else ("nccl" if torch.cuda.is_available() else "gloo")
    # Check if running under MPI
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # OpenMPI
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # Not distributed - return single process values
        world_size = 1
        rank = 0
        local_rank = 0
        return rank, world_size, local_rank

    # Set environment variables that PyTorch expects
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize PyTorch distributed
    if not dist.is_initialized():
        if backend == "nccl":
            assert torch.cuda.is_available(), "CUDA is required for NCCL ddp backend"
            local_device = torch.device(f"cuda:{local_rank}")
        else:
            local_device = None

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=local_device,
        )

    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank() -> int:
    """Get local GPU index."""
    # Try to get from environment first
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # If not set, assume single GPU per node or calculate from global rank
        return 0


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def get_device() -> str:
    """Get device for current process in distributed setting."""
    if torch.cuda.is_available():
        if is_distributed():
            local_rank = get_local_rank()
            return f"cuda:{local_rank}"
        return "cuda"
    return "cpu"


def sync_across_processes() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
) -> torch.Tensor:
    """All-reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (default: SUM)

    Returns:
        Reduced tensor
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from source rank to all other processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank (default: 0)

    Returns:
        Broadcasted tensor
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(tensor, src=src)
    return tensor


def gather_object(obj: object, dst: int = 0) -> list[object] | None:
    """Gather objects from all ranks to destination rank.

    Args:
        obj: Object to gather
        dst: Destination rank (default: 0)

    Returns:
        List of objects from all ranks if on destination rank, None otherwise
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [obj]

    if get_rank() == dst:
        output: list[object | None] = [None] * dist.get_world_size()
        dist.gather_object(obj, output, dst=dst)
        # Type assertion is safe here as gather_object fills all slots
        return output  # type: ignore[return-value]
    else:
        dist.gather_object(obj, None, dst=dst)
        return None


def print_once(msg: str) -> None:
    """Print message only on rank 0."""
    if is_main_process():
        print(msg)


def avg_metrics_across_ranks(metrics: Mapping[str, float], device: str) -> Mapping[str, float]:
    """Get the average of metrics across ranks."""
    assert is_distributed(), "Can only average metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    metric_values = all_reduce(metric_values, op=ReduceOp.AVG)
    return {k: metric_values[i].item() for i, k in enumerate(metrics)}


def avg_eval_metrics_across_ranks(
    metrics: Mapping[str, float | Image.Image], device: str
) -> Mapping[str, float | Image.Image]:
    """Get the average of eval metrics across ranks.

    Ignores any metrics that are not floats or ints. Currently, the image metrics do not need to be
    averaged. If this changes for future metrics, we will need to do a reduce during calculcation
    of the metric.
    """
    assert is_distributed(), "Can only average metrics across ranks if running in distributed mode"
    metrics_keys_to_avg = {k: v for k, v in metrics.items() if isinstance(v, float | int)}
    if metrics_keys_to_avg:
        avg_metrics = avg_metrics_across_ranks(metrics_keys_to_avg, device)
    else:
        avg_metrics = {}
    return {**metrics, **avg_metrics}


# def get_distributed_rand_like(
#     shape: tuple[int, ...], hash_key: str, device: str | torch.device
# ) -> Tensor:
#     """Get a random tensor of shape `shape` which matches what would be produced by indexing a
#     random tensor of shape `shape * world_size` with the current rank.

#     This function simulates the following process:
#     1. Generate a full random tensor of shape `shape * world_size` with a custom generator
#     2. Index the tensor to get the portion of the tensor on the current rank

#     It does this by iterating through random values until the rng counter matches what is needed
#     for the current rank.

#     Args:
#         shape: The shape of the tensor to get.
#         hash_key: A string used to seed the random number generator.
#         device: The device to convert the final tensor to (we generate all tensors on CPU)
#     """
#     # Assert that shape has 3 dimensions (batch, seq_len, C). In future we'd want to support other
#     # shapes

#     start_time = time.perf_counter()
#     assert len(shape) == 3, "Shape must have 3 dimensions (batch, seq_len, C)"
#     local_batch_size, seq_len, C = shape

#     generator = torch.Generator(device="cpu")
#     seed = int(hashlib.md5(hash_key.encode()).hexdigest(), 16) % (2**32)
#     generator.manual_seed(seed)
#     end_time = time.perf_counter()
#     print(f"Time to seed generator: {end_time - start_time}")

#     elements_per_sample = seq_len * C
#     total_elements_to_skip = get_rank() * local_batch_size * elements_per_sample

#     skip_chunk_size = 100_000
#     remaining = total_elements_to_skip
#     while remaining > 0:
#         chunk = min(remaining, skip_chunk_size)
#         torch.rand(chunk, generator=generator, device="cpu")
#         remaining -= chunk
#     end_time = time.perf_counter()
#     print(f"Time to finish skipping: {end_time - start_time}")

#     # Generate this rank's data
#     return torch.rand(*shape, generator=generator, device="cpu").to(device)


def get_distributed_rand_like(
    shape: tuple[int, int, int],  # batch, seq_len, C
    hash_key: str,
    device: str | torch.device,
) -> Float[Tensor, "batch seq n_dim"]:
    """Return a rank-specific random tensor.

    We seed a CPU RNG with ``hash_key``, generate the *full* tensor of size
    ``(batch * world_size, seq_len, C)`` on the CPU, slice out the sub-tensor
    belonging to this rank, and finally move it to *device*.

    Args:
        shape: Local (per-rank) tensor shape ``(batch, seq_len, C)``.
        hash_key: String used to seed the RNG (rank-agnostic).
        device: Target device for the returned tensor.
    """
    start_time = time.perf_counter()

    # Validate shape
    assert len(shape) == 3, "Shape must be (batch, seq_len, C)"
    local_batch, seq_len, channels = shape

    # Compute global shape
    rank = get_rank()
    world_size = get_world_size()
    full_shape = (local_batch * world_size, seq_len, channels)

    # Generate global tensor and slice this rank's chunk
    full_tensor = torch.rand(*full_shape, device="cpu")
    start = rank * local_batch
    end = start + local_batch
    local_tensor = full_tensor[start:end]

    print(f"get_distributed_rand_like: moved to CPU in {time.perf_counter() - start_time:.3f}s")
    local_tensor_gpu = local_tensor.to(device)
    print(f"get_distributed_rand_like: moved to GPU in {time.perf_counter() - start_time:.3f}s")

    return local_tensor_gpu
