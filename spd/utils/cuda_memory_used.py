import torch

# pyright: reportUnreachable=false, reportUnnecessaryIsInstance=false


def _to_cuda_device(device: int | str | torch.device) -> torch.device:
    """Return a normalized CUDA device object."""
    dev: torch.device
    if isinstance(device, torch.device):
        dev = device
    elif isinstance(device, int):
        dev = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        # Accept forms like "cuda", "cuda:0", or bare index "0"
        dev = torch.device(device)
    else:
        raise TypeError(f"Unsupported device type: {type(device).__name__}")

    if dev.type != "cuda":
        raise ValueError(f"Device {dev} is not a CUDA device")

    return dev


def cuda_mem_info(dev: torch.device) -> tuple[int, int]:
    """Return (free, total) bytes for a CUDA device."""
    current_idx: int = torch.cuda.current_device()
    if dev.index != current_idx:
        torch.cuda.set_device(dev)
        free: int
        total: int
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_device(current_idx)
    else:
        free, total = torch.cuda.mem_get_info()
    return free, total


def cuda_memory_used(device: int | str | torch.device = 0) -> int:
    """Return bytes currently allocated on a CUDA device."""
    dev: torch.device = _to_cuda_device(device)
    free, total = cuda_mem_info(dev)
    used: int = total - free
    return used


def cuda_memory_fraction(device: int | str | torch.device = 0) -> float:
    """Return fraction of total memory in use on a CUDA device."""
    dev: torch.device = _to_cuda_device(device)
    free, total = cuda_mem_info(dev)
    used: int = total - free
    fraction: float = used / total if total else 0.0
    return fraction
