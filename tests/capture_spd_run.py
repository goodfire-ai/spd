"""Helper script to run an SPD experiment under DDP while capturing batches and metrics.

This script is executed in a separate Python process (typically under ``mpirun``).
It applies a handful of ``unittest.mock`` patches so that we can intercept
``extract_batch_data`` and ``evaluate`` calls.  After the experiment finishes we
persist the captured information to rank-specific pickle files so that the main
pytest process can validate consistency across different data-parallel
configurations.

Usage
-----
python capture_spd_run.py CONFIG_PATH BATCH_FILE METRICS_FILE
"""

import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from jaxtyping import Float
from torch import Tensor

from spd.eval import evaluate as real_evaluate
from spd.utils.distributed_utils import cleanup_distributed as orig_cleanup
from spd.utils.distributed_utils import get_rank
from spd.utils.general_utils import extract_batch_data as real_extract_batch_data

# use torch float64 everywhere
torch.set_default_dtype(torch.float64)

captured_batches: list[dict[str, Any]] = []
captured_metrics: list[dict[str, Any]] = []
# Using a single-element list allows nested scopes to assign the rank.
_my_rank: list[int | None] = [None]


def _capture_batch(batch_item: Any) -> Float[Tensor, " *shape"]:  # type: ignore[name-defined]
    """Intercept ``extract_batch_data`` calls and store the returned tensor."""

    tensor: Float[Tensor, " *shape"] = real_extract_batch_data(batch_item)  # type: ignore[name-defined]
    captured_batches.append(
        {
            "rank": get_rank(),
            "shape": tuple(tensor.shape),
            "data": tensor.cpu().numpy().copy(),
        }
    )
    return tensor


def _capture_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Intercept ``evaluate`` calls and store numeric metrics only."""

    result: dict[str, Any] = real_evaluate(*args, **kwargs)
    captured_metrics.append(
        {
            "rank": get_rank(),
            "metrics": {k: v for k, v in result.items() if isinstance(v, int | float)},
        }
    )
    return result


def _save_rank_wrapper(original_cleanup: Callable[[], None]) -> Callable[[], None]:
    """Wrap ``cleanup_distributed`` so the current rank is recorded before teardown."""

    def wrapper() -> None:  # noqa: D401
        _my_rank[0] = get_rank()
        original_cleanup()

    return wrapper


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run experiment and dump captured artefacts to pickle files."""

    if len(sys.argv) != 4:
        sys.exit("Usage: python capture_spd_run.py CONFIG_PATH BATCH_FILE METRICS_FILE")

    config_path = Path(sys.argv[1]).expanduser().resolve()
    batch_file = Path(sys.argv[2]).expanduser().resolve()
    metrics_file = Path(sys.argv[3]).expanduser().resolve()

    # Apply patches within a context manager so they are reverted automatically
    with (
        patch(
            "spd.utils.distributed_utils.cleanup_distributed",
            side_effect=_save_rank_wrapper(orig_cleanup),
        ),
        patch(
            "spd.utils.general_utils.extract_batch_data",
            side_effect=_capture_batch,
        ),
        patch(
            "spd.eval.evaluate",
            side_effect=_capture_metrics,
        ),
    ):
        from spd.experiments.lm.lm_decomposition import main as lm_main

        # The experiment will handle distributed initialisation itself.
        lm_main(str(config_path))

    actual_rank = _my_rank[0]

    batch_file_rank = batch_file.with_name(
        f"{batch_file.stem}_rank{actual_rank}{batch_file.suffix}"
    )
    metrics_file_rank = metrics_file.with_name(
        f"{metrics_file.stem}_rank{actual_rank}{metrics_file.suffix}"
    )

    # Persist captured data
    batch_file_rank.write_bytes(pickle.dumps(captured_batches))
    metrics_file_rank.write_bytes(pickle.dumps(captured_metrics))


if __name__ == "__main__":
    main()
