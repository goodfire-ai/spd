"""Utilities for distributed data parallel training with MPI support."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Literal, cast

import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed import ReduceOp


@dataclass(frozen=True, slots=True)
class DistributedState:
    """Immutable snapshot of the distributed runtime state for this process."""

    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]


def _infer_default_backend() -> Literal["nccl", "gloo"]:
    return "nccl" if torch.cuda.is_available() else "gloo"


def _init_default_state() -> DistributedState:
    backend = _infer_default_backend()
    return DistributedState(rank=0, world_size=1, local_rank=0, backend=backend)


# Module-level cached state used as a single source of truth
_state: DistributedState = _init_default_state()


def get_distributed_state() -> DistributedState:
    """Return the cached distributed state.

    Returns:
        DistributedState: The current process's distributed state snapshot.
    """
    return _state


def init_distributed(backend: Literal["nccl", "gloo"] | None = None) -> DistributedState:
    global _state
    """Initialize distributed process group using MPI.

    Supports OpenMPI only.

    Args:
        backend: Distributed backend to use ('nccl' or 'gloo'). If None, uses 'nccl' if CUDA is
            available, otherwise 'gloo'.

    Returns:
        DistributedState
    """
    assert not is_distributed(), "Already in a distributed process group"
    backend = backend if backend is not None else _infer_default_backend()
    # Check if running under MPI (OpenMPI)
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # Not distributed - return single process values
        world_size = 1
        rank = 0
        local_rank = 0
        # Update cached state and return
        _state = DistributedState(
            rank=rank, world_size=world_size, local_rank=local_rank, backend=backend
        )
        return _state

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

    # Set the default cuda device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    _state = DistributedState(
        rank=rank, world_size=world_size, local_rank=local_rank, backend=backend
    )
    return _state


def cleanup_distributed() -> None:
    """Clean up distributed process group and reset cached state."""
    global _state
    if dist.is_initialized():
        dist.destroy_process_group()
    _state = _init_default_state()


def with_distributed_cleanup[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Decorator to clean up distributed state after function execution."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        finally:
            cleanup_distributed()

    return wrapper


def is_distributed() -> bool:
    """Check if running in distributed mode using cached state."""
    state = get_distributed_state()
    return state.world_size > 1


def get_rank() -> int:
    """Get current process rank from cached state."""
    return get_distributed_state().rank


def get_world_size() -> int:
    """Get total number of processes from cached state."""
    return get_distributed_state().world_size


def get_local_rank() -> int:
    """Get local GPU index from cached state."""
    return get_distributed_state().local_rank


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
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast_obj[T](value: T) -> T:
    """Broadcast an object from rank 0 to all ranks."""
    assert dist.is_initialized()
    payload: list[object] = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return cast(T, payload[0])


def call_on_rank0_then_broadcast[**P, T](
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Call `fn` only on rank 0 and broadcast the result to all ranks."""
    if is_distributed():
        result = fn(*args, **kwargs) if is_main_process() else None
        result = broadcast_obj(result)
        return cast(T, result)
    return fn(*args, **kwargs)


def ensure_cached_and_call[**P, T](fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Call `fn` on rank 0 to cache any download side effects, barrier, then call on all ranks."""
    if is_distributed():
        if is_main_process():
            _ = fn(*args, **kwargs)
        sync_across_processes()
        return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def avg_metrics_across_ranks(metrics: Mapping[str, float], device: str) -> Mapping[str, float]:
    """Get the average of metrics across ranks."""
    assert is_distributed(), "Can only average metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    # Use ReduceOp.SUM and then divide since ReduceOp.AVG isn't supported for gloo backend (cpu)
    metric_values = all_reduce(metric_values, op=ReduceOp.SUM) / get_world_size()
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
