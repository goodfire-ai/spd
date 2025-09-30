import torch

# pyright: reportUnreachable=false, reportUnnecessaryIsInstance=false


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


def cuda_memory_fraction(device: str | torch.device) -> float:
    """Return fraction of total memory in use on a CUDA device."""
    dev: torch.device = torch.device(device)
    free, total = cuda_mem_info(dev)
    used: int = total - free
    fraction: float = used / total if total else 0.0
    return fraction
