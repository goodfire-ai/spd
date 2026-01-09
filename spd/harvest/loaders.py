"""Loaders for reading harvest output files."""

import json
import time

from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentSummary,
    ComponentTokenPMI,
    get_activation_contexts_dir,
    get_correlations_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger


def load_activation_contexts_summary(wandb_run_id: str) -> dict[str, ComponentSummary] | None:
    """Load lightweight summary of activation contexts (just metadata, not full examples)."""
    start = time.perf_counter()
    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "summary.json"
    if not path.exists():
        return None
    result = ComponentSummary.load_all(path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[PERF] load_activation_contexts_summary: {elapsed_ms:.1f}ms ({len(result)} components)"
    )
    return result


def load_component_activation_contexts(
    wandb_run_id: str, component_key: str
) -> ComponentData | None:
    """Load a single component's activation contexts."""
    start = time.perf_counter()
    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "components.jsonl"
    assert path.exists(), f"No activation contexts found at {path}"

    # Each line starts with {"component_key": "layer:idx", ...}
    expected_prefix = '{"component_key": '
    prefix = f'{{"component_key": "{component_key}"'

    lines_scanned = 0
    with open(path) as f:
        for line in f:
            lines_scanned += 1
            assert line.startswith(expected_prefix), f"Unexpected line format: {line[:100]}"
            if not line.startswith(prefix):
                continue
            # Found it - parse just this line
            data = json.loads(line)
            data["activation_examples"] = [
                ActivationExample(**ex) for ex in data["activation_examples"]
            ]
            data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
            data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"[PERF] load_component_activation_contexts({component_key}): "
                f"{elapsed_ms:.1f}ms (scanned {lines_scanned} lines)"
            )
            return ComponentData(**data)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[PERF] load_component_activation_contexts({component_key}): "
        f"{elapsed_ms:.1f}ms (NOT FOUND, scanned {lines_scanned} lines)"
    )
    return None


def load_correlations(wandb_run_id: str) -> CorrelationStorage | None:
    """Load component correlations from harvest output."""
    start = time.perf_counter()
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "component_correlations.pt"
    if not path.exists():
        return None
    result = CorrelationStorage.load(path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[PERF] load_correlations: {elapsed_ms:.1f}ms")
    return result


def load_token_stats(wandb_run_id: str) -> TokenStatsStorage | None:
    """Load token statistics from harvest output."""
    start = time.perf_counter()
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "token_stats.pt"
    if not path.exists():
        return None
    result = TokenStatsStorage.load(path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[PERF] load_token_stats: {elapsed_ms:.1f}ms")
    return result
