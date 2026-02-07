"""Loaders for reading harvest output files."""

import threading

import orjson

from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentSummary,
    ComponentTokenPMI,
    get_activation_contexts_dir,
    get_correlations_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage


def load_activation_contexts_summary(wandb_run_id: str) -> dict[str, ComponentSummary] | None:
    """Load lightweight summary of activation contexts (just metadata, not full examples)."""
    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "summary.json"
    if not path.exists():
        return None
    return ComponentSummary.load_all(path)


# Cache for component indices (run_id -> {component_key -> byte_offset})
_component_index_cache: dict[str, dict[str, int]] = {}
_component_index_lock = threading.Lock()

_COMPONENT_KEY_PREFIX = '"component_key": "'


def _get_component_index(wandb_run_id: str) -> dict[str, int]:
    """Get or build component index for a run.

    On first access, scans the components.jsonl file to build a byte offset
    index, then caches it in memory for O(1) lookups.
    """
    # Fast path: already cached
    if wandb_run_id in _component_index_cache:
        return _component_index_cache[wandb_run_id]

    # Slow path: build index under lock to prevent duplicate work
    with _component_index_lock:
        # Double-check after acquiring lock
        if wandb_run_id in _component_index_cache:
            return _component_index_cache[wandb_run_id]

        ctx_dir = get_activation_contexts_dir(wandb_run_id)
        components_path = ctx_dir / "components.jsonl"
        assert components_path.exists(), f"No activation contexts found at {components_path}"

        index: dict[str, int] = {}
        with open(components_path) as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                # Extract component_key from start of JSON line
                # Format: {"component_key": "layer:idx", ...}
                key_start = line.find(_COMPONENT_KEY_PREFIX)
                assert key_start != -1, f"Malformed line in components.jsonl: {line[:100]}"
                key_start += len(_COMPONENT_KEY_PREFIX)
                key_end = line.find('"', key_start)
                assert key_end != -1, f"Malformed line in components.jsonl: {line[:100]}"
                component_key = line[key_start:key_end]
                index[component_key] = offset

        _component_index_cache[wandb_run_id] = index
        return index


def load_component_activation_contexts(wandb_run_id: str, component_key: str) -> ComponentData:
    """Load a single component's activation contexts using index for O(1) lookup."""
    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "components.jsonl"
    assert path.exists(), f"No activation contexts found at {path}"

    index = _get_component_index(wandb_run_id)
    if component_key not in index:
        raise ValueError(f"Component {component_key} not found in activation contexts")

    byte_offset = index[component_key]
    with open(path) as f:
        f.seek(byte_offset)
        line = f.readline()

    data = orjson.loads(line)
    data["activation_examples"] = [ActivationExample(**ex) for ex in data["activation_examples"]]
    data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
    data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
    return ComponentData(**data)


def load_component_activation_contexts_bulk(
    wandb_run_id: str, component_keys: list[str]
) -> dict[str, ComponentData]:
    """Load multiple components' activation contexts efficiently.

    Uses mmap for fast random access, sorts offsets for sequential reads.
    Much faster than calling load_component_activation_contexts() in a loop.
    """
    import mmap

    if not component_keys:
        return {}

    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "components.jsonl"
    assert path.exists(), f"No activation contexts found at {path}"

    index = _get_component_index(wandb_run_id)

    # Build list of (offset, key) for components that exist, sorted by offset
    offset_key_pairs: list[tuple[int, str]] = []
    for key in component_keys:
        if key in index:
            offset_key_pairs.append((index[key], key))
    offset_key_pairs.sort()  # Sort by offset for sequential disk access

    # Read with mmap for efficient random access
    result: dict[str, ComponentData] = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for offset, key in offset_key_pairs:
                mm.seek(offset)
                line_end = mm.find(b"\n", offset)
                line = mm[offset:line_end] if line_end != -1 else mm[offset:]

                data = orjson.loads(line)
                data["activation_examples"] = [
                    ActivationExample(**ex) for ex in data["activation_examples"]
                ]
                data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
                data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
                result[key] = ComponentData(**data)
        finally:
            mm.close()

    return result


def load_correlations(wandb_run_id: str) -> CorrelationStorage:
    """Load component correlations from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "component_correlations.pt"
    assert path.exists()
    return CorrelationStorage.load(path)


def load_token_stats(wandb_run_id: str) -> TokenStatsStorage:
    """Load token statistics from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "token_stats.pt"
    assert path.exists()
    return TokenStatsStorage.load(path)
