from typing import Literal

import torch
from torch import Tensor

StatsKey = Literal[
    "mean",
    "std",
    "median",
    "min",
    "max",
    "q01",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q99",
    "chosen_pair",
]


def _flatten_if_needed(x: Tensor) -> Tensor:
    """Make x 1D without copy when possible."""
    x_flat: Tensor = x.reshape(-1)
    return x_flat


def _approx_quantile(
    x: Tensor,
    qs: Tensor,
    *,
    max_elems: int = 5_000_000,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Approximate quantiles by subsampling if needed, else exact.

    If x.numel() > max_elems, draws a random subset of size max_elems (with replacement)
    on the same device as x, then computes torch.quantile once for all qs.
    """
    x1d: Tensor = _flatten_if_needed(x)
    n: int = x1d.numel()
    if n == 0:
        raise ValueError("Empty tensor.")
    if n > max_elems:
        # sample with replacement to avoid materializing a giant permutation
        g: torch.Generator | None = generator
        idx: Tensor = torch.randint(0, n, (max_elems,), device=x1d.device, generator=g)
        x_used: Tensor = x1d[idx]
    else:
        x_used = x1d
    # Compute all quantiles in one shot to reuse the sort
    q: Tensor = torch.quantile(x_used, qs, interpolation="linear")
    return q


def _exact_quantile_all_at_once(x: Tensor, qs: Tensor) -> Tensor:
    """Exact quantiles without repeated sorts."""
    x1d: Tensor = _flatten_if_needed(x)
    q: Tensor = torch.quantile(x1d, qs, interpolation="linear")
    return q


def stats_dict(
    data: Tensor,
    *,
    approx_if_large: bool = True,
    max_elems_for_quantile: int = 5_000_000,
    rng: torch.Generator | None = None,
) -> dict[StatsKey, float]:
    """summary

    Compute common stats plus a set of quantiles. Uses a single quantile() call
    for all requested quantiles; optionally switches to an approximate method
    by subsampling when the input is very large to avoid RuntimeError.

    # Parameters:
     - `data : Tensor`
        Input tensor of any shape and dtype convertible to floating for stats.
     - `approx_if_large : bool`
        If True, use subsampling for quantiles when data is huge. (defaults to True)
     - `max_elems_for_quantile : int`
        Max elements before triggering approximate mode. (defaults to 5_000_000)
     - `rng : torch.Generator | None`
        Optional torch generator for reproducible subsampling.

    # Returns:
     - `dict[StatsKey, float]`
        Mapping from stat name to Python float.

    # Modifies:
     - None

    # Usage:

    ```python
    >>> x = torch.randn(50_000_000, device="cuda")
    >>> out = stats_dict(x, approx_if_large=True, max_elems_for_quantile=5_000_000)
    >>> out["q95"]
    1.64
    ```

    # Raises:
     - `ValueError` : if `data` is empty
    """
    x: Tensor = data
    if x.numel() == 0:
        raise ValueError("Empty tensor.")
    # Work in float for numerics, but keep device
    xf: Tensor = x.float()

    # Fast exact ops that do not need the full sort
    # std_mean does mean and std in one pass; aminmax does min and max together
    std: Tensor
    mean: Tensor
    std, mean = torch.std_mean(xf)
    mn: Tensor
    mx: Tensor
    mn, mx = torch.aminmax(xf)

    # median is a quantile; we can either reuse below or do .median() directly.
    # We will get it from the quantiles call to avoid extra work.
    q_values: Tensor = torch.tensor(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        device=xf.device,
        dtype=xf.dtype,
    )
    if approx_if_large:
        qs_all: Tensor = _approx_quantile(
            xf,
            q_values,
            max_elems=max_elems_for_quantile,
            generator=rng,
        )
    else:
        qs_all = _exact_quantile_all_at_once(xf, q_values)

    out: dict[StatsKey, float] = {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "median": float(qs_all[4].item()),  # median is at index 4
        "min": float(mn.item()),
        "max": float(mx.item()),
        "q01": float(qs_all[0].item()),
        "q05": float(qs_all[1].item()),
        "q10": float(qs_all[2].item()),
        "q25": float(qs_all[3].item()),
        "q50": float(qs_all[4].item()),  # median again
        "q75": float(qs_all[5].item()),
        "q90": float(qs_all[6].item()),
        "q95": float(qs_all[7].item()),
        "q99": float(qs_all[8].item()),
    }
    return out
