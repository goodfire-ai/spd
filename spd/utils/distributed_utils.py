"""Utilities for distributed data parallel training (torchrun or MPI)."""

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, override

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp
from torch.types import Number

from spd.configs import Config
from spd.utils.command import Command


@dataclass(frozen=True, slots=True)
class DistributedState:
    """Immutable snapshot of the distributed runtime state for this process."""

    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]

    def device(self) -> str:
        if self.backend == "gloo":
            return "cpu"
        return f"cuda:{self.local_rank}"



# Module-level cached state used as a single source of truth
_state: DistributedState | None = None

has_world_size: bool = os.environ.get("WORLD_SIZE") is not None


def get_distributed_state() -> DistributedState | None:
    """Return the cached distributed state. If not initialized, returns None.

    Returns:
        DistributedState: The current process's distributed state snapshot.
    """
    if has_world_size:
        assert _state is not None
        return _state
    else:
        assert _state is None
        return None


# todo handle cpu gloo cases here
def init_distributed(backend: Literal["nccl", "gloo"]) -> DistributedState | None:
    """Initialize distributed process group using MPI.

    Args:
        backend: Distributed backend to use ('nccl' or 'gloo'). If None, uses 'nccl' if CUDA is
            available, otherwise 'gloo'.

    Returns:
        DistributedState
    """
    global _state
    assert _state is None, "Distributed state already initialized"

    world_size_str = os.environ.get("WORLD_SIZE")
    rank_str = os.environ.get("RANK")
    local_rank_str = os.environ.get("LOCAL_RANK")
    print(f"world_size: {world_size_str:<4}, rank: {rank_str:<4}, local_rank: {local_rank_str:<4}")
    print(f"backend: {backend}")
    assert (world_size_str is not None) == (rank_str is not None) == (local_rank_str is not None)

    if world_size_str is None:
        _state = None
        return _state

    assert os.environ.get("MASTER_ADDR") is not None
    assert os.environ.get("MASTER_PORT") is not None

    world_size = int(world_size_str)
    rank = int(rank_str)  # pyright: ignore[reportArgumentType]. This is not none. see assert above.
    local_rank = int(local_rank_str)  # pyright: ignore[reportArgumentType]

    device = torch.device(f"cuda:{local_rank}")

    # Initialize PyTorch distributed
    assert not dist.is_initialized()
    if backend == "nccl":
        assert torch.cuda.is_available(), "CUDA is required for NCCL ddp backend"
        try:
            torch.cuda.set_device(device)
        except RuntimeError as e:
            print(f"Error setting device to [{device}]: {e}")
            print(f"CUDA devices: {torch.cuda.device_count()}")
            raise e

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=None if backend == "gloo" else device,
    )

    _state = DistributedState(
        rank=rank, world_size=world_size, local_rank=local_rank, backend=backend
    )
    return _state


def cleanup_distributed() -> None:
    """Clean up distributed process group and reset cached state."""
    global _state
    if is_distributed():
        dist.destroy_process_group()
    _state = None


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
    return state is not None


# def get_rank() -> int | None:
#     """Get current process rank from cached state."""
#     state = get_distributed_state()
#     if state is None:
#         return None
#     return state.rank


# def get_world_size() -> int | None:
#     """Get total number of processes from cached state."""
#     state = get_distributed_state()
#     if state is None:
#         return None
#     return state.world_size


# def get_local_rank() -> int | None:
#     """Get local GPU index from cached state."""
#     state = get_distributed_state()
#     if state is None:
#         return None
#     return state.local_rank


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.rank == 0


def get_device() -> str:
    """Get device for current process in distributed setting."""
    state = get_distributed_state()
    if state is None:
        return "cpu"
    return state.device()


def sync_across_processes() -> None:
    """Synchronize all processes."""
    if is_distributed():
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
    assert is_distributed()
    payload: list[object] = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return cast(T, payload[0])


def call_on_rank0_then_broadcast[**P, T](
    fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Call `fn` only on rank 0 and broadcast the result to all ranks."""
    if is_distributed():
        result = fn(*args, **kwargs) if is_main_process() else None
        result = broadcast_obj(result)
        return cast(T, result)
    return fn(*args, **kwargs)


def ensure_cached_and_call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Call `fn` on rank 0 to cache any download side effects, barrier, then call on all ranks."""
    if is_distributed():
        if is_main_process():
            _ = fn(*args, **kwargs)
        sync_across_processes()
        return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def sum_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    assert is_distributed(), "Can only sum metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    metric_values = all_reduce(metric_values, op=ReduceOp.SUM)
    return {k: metric_values[i].item() for i, k in enumerate(metrics)}


def avg_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    state = get_distributed_state()
    if state is None:
        return metrics
    world_size = state.world_size
    assert world_size > 0, "World size must be greater than 0"
    sum_metrics = sum_metrics_across_ranks(metrics, device)
    return {k: v / world_size for k, v in sum_metrics.items()}


def gather_all_tensors(tensor: Tensor) -> list[Tensor]:
    """Gather tensors from all distributed processes.

    Requires all tensors to have identical shapes across all ranks.

    Args:
        tensor: The tensor to gather from all ranks
        group: The process group (defaults to WORLD)

    Returns:
        List of tensors from all ranks (including local rank)
    """
    state = get_distributed_state()
    if state is None:
        return [tensor]

    tensor = tensor.contiguous()

    gathered = [torch.zeros_like(tensor) for _ in range(state.world_size)]
    torch.distributed.all_gather(gathered, tensor)

    # Replace our rank's entry with the original to preserve autograd
    gathered[state.rank] = tensor

    return gathered


def get_config_json(config: Config) -> str:
    return f"json:{json.dumps(config.model_dump(mode='json'))}"


class ComputeStrategy(ABC):
    @abstractmethod
    def n_gpus_per_node(self) -> int: ...

    @abstractmethod
    def n_nodes(self) -> int: ...

    @abstractmethod
    def get_command(
        self,
        run_id: str,
        idx: int,
        script_path: Path,
        config: Config,
        experiment: str,
        sweep_params: dict[str, Any] | None = None,
    ) -> Command: ...


class Cpu(ComputeStrategy):
    @override
    def n_gpus_per_node(self) -> int:
        return 0

    @override
    def n_nodes(self) -> int:
        return 1

    @override
    def get_command(
        self,
        run_id: str,
        idx: int,
        script_path: Path,
        config: Config,
        experiment: str,
        sweep_params: dict[str, Any] | None = None,
    ) -> Command:
        command = (
            f"python {script_path} "
            f"--config_json '{get_config_json(config)}' "
            f"--sweep_id {run_id} "
            f"--evals_id {experiment} "
        )
        if sweep_params is not None:
            command += f"--sweep_params_json '{json.dumps(sweep_params)}' "
        return Command(command=command)


class SingleGpu(ComputeStrategy):
    @override
    def n_gpus_per_node(self) -> int:
        return 1

    @override
    def n_nodes(self) -> int:
        return 1

    @override
    def get_command(
        self,
        run_id: str,
        idx: int,
        script_path: Path,
        config: Config,
        experiment: str,
        sweep_params: dict[str, Any] | None = None,
    ) -> Command:
        command = (
            f"python {script_path} "
            f"--config_json '{get_config_json(config)}' "
            f"--sweep_id {run_id} "
            f"--evals_id {experiment} "
        )
        if sweep_params is not None:
            command += f"--sweep_params_json '{json.dumps(sweep_params)}' "
        return Command(command=command)


class SingleNode(ComputeStrategy):
    def __init__(self, n_gpus_per_node: int):
        self._n_gpus_per_node = n_gpus_per_node

    @override
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    @override
    def n_nodes(self) -> int:
        return 1

    @override
    def get_command(
        self,
        run_id: str,
        idx: int,
        script_path: Path,
        config: Config,
        experiment: str,
        sweep_params: dict[str, Any] | None = None,
    ) -> Command:
        port = _choose_master_port(run_id, idx)
        rendezvous_id = f"{run_id}_{idx}"
        env = {
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
        command = (
            f"torchrun "
            "--standalone "
            f"--nproc_per_node={self._n_gpus_per_node} "
            f"--master_port={port} "
            f"--rdzv_id={rendezvous_id} "
            f"{script_path} "
            f"--config_json '{get_config_json(config)}' "
            f"--sweep_id {run_id} "
            f"--evals_id {experiment} "
        )
        if sweep_params is not None:
            command += f" --sweep_params_json '{json.dumps(sweep_params)}'"
        return Command(env_vars=env, command=command)


class MultiNode(ComputeStrategy):
    N_GPUS_PER_NODE: ClassVar[int] = 8

    def __init__(self, n_nodes: int):
        self._n_nodes = n_nodes

    @override
    def n_gpus_per_node(self) -> int:
        return self.N_GPUS_PER_NODE

    @override
    def n_nodes(self) -> int:
        return self._n_nodes

    @override
    def get_command(
        self,
        run_id: str,
        idx: int,
        script_path: Path,
        config: Config,
        experiment: str,
        sweep_params: dict[str, Any] | None = None,
    ) -> Command:
        env = {
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }

        master_port = _choose_master_port(run_id, idx)

        command = (
            f"srun "
            f"torchrun "
            f"--nnodes={self._n_nodes} "
            f"--nproc_per_node={self.N_GPUS_PER_NODE} "
            f"--rdzv_id={run_id}_{idx} "
            f"--rdzv_backend=c10d "
            f'--rdzv_endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):{master_port} '
            f"{script_path} "
            f"--config_json '{get_config_json(config)}' "
            f"--sweep_id {run_id} "
            f"--evals_id {experiment}"
        )
        if sweep_params is not None:
            command += f" --sweep_params_json '{json.dumps(sweep_params)}'"

        return Command(env_vars=env, command=command)
    
    def get_sbatch_script


#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=4

def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


@dataclass()
class SlurmPartition:
    name: str


class Local: ...


ComputeEnvironment = (
    tuple[SlurmPartition, ComputeStrategy] | tuple[Local, Cpu | SingleGpu | SingleNode]
)
