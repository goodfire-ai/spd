"""Utilities for managing experiment run directories and IDs."""

import copy
import itertools
import json
import secrets
import string
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import torch
import wandb
import yaml

from spd.log import logger
from spd.settings import SPD_CACHE_DIR
from spd.utils.git_utils import (
    create_git_snapshot,
    repo_current_branch,
    repo_current_commit_hash,
    repo_is_clean,
)

# Fields that use discriminated union merging: field_name -> discriminator_field
_DISCRIMINATED_LIST_FIELDS: dict[str, str] = {
    "loss_metric_configs": "classname",
    "eval_metric_configs": "classname",
}


def get_local_run_id() -> str:
    """Generate a unique run ID. Used if wandb is not active.

    Format: local-<random_8_chars>
    Where random_8_chars is a combination of lowercase letters and digits.

    Returns:
        Unique run ID string
    """
    # Generate 8 random characters (lowercase letters and digits)
    chars = string.ascii_lowercase + string.digits
    random_suffix = "".join(secrets.choice(chars) for _ in range(8))

    return f"local-{random_suffix}"


# TODO: avoid using this function?
def get_output_dir(use_wandb_id: bool = True) -> Path:
    """Get the output directory for a run.

    If WandB is active, uses the WandB project and run ID. Otherwise, generates a local run ID.

    Returns:
        Path to the output directory
    """
    # Check if wandb is active and has a run
    if use_wandb_id:
        assert wandb.run is not None, "WandB run is not active"
        # Get project name from wandb.run, fallback to "spd" if not available
        project = getattr(wandb.run, "project", "spd")
        run_id = f"{project}-{wandb.run.id}"
    else:
        run_id = get_local_run_id()

    run_dir = SPD_CACHE_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def _save_yaml(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, **kwargs)


def _save_torch(data: Any, path: Path | str, **kwargs: Any) -> None:
    torch.save(data, path, **kwargs)


def _save_text(data: str, path: Path | str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def check_run_exists(wandb_string: str) -> Path | None:
    """Check if a run exists in the shared filesystem based on WandB string.

    Args:
        wandb_string: WandB string in format "wandb:project/runs/run_id"

    Returns:
        Path to the run directory if it exists, None otherwise
    """
    if not wandb_string.startswith("wandb:"):
        return None

    # Parse the wandb string
    parts = wandb_string.replace("wandb:", "").split("/")
    if len(parts) != 3 or parts[1] != "runs":
        return None

    project = parts[0]
    run_id = parts[2]

    # Check if directory exists with format project-runid
    run_dir = SPD_CACHE_DIR / "runs" / f"{project}-{run_id}"
    return run_dir if run_dir.exists() else None


def save_file(data: dict[str, Any] | Any, path: Path | str, **kwargs: Any) -> None:
    """Save a file.

    NOTE: This function was originally designed to save files with specific permissions,
    bypassing the system's umask. This is not needed anymore, but we're keeping this
    abstraction for convenience and brevity.

    File type is determined by extension:
    - .json: Save as JSON
    - .yaml/.yml: Save as YAML
    - .pth/.pt: Save as PyTorch model
    - .txt or other: Save as plain text (data must be string)

    Args:
        data: Data to save (format depends on file type)
        path: File path to save to
        **kwargs: Additional arguments passed to the specific save function
    """
    path = Path(path)
    suffix = path.suffix.lower()

    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        _save_json(data, path, **kwargs)
    elif suffix in [".yaml", ".yml"]:
        _save_yaml(data, path, **kwargs)
    elif suffix in [".pth", ".pt"]:
        _save_torch(data, path, **kwargs)
    else:
        # Default to text file
        assert isinstance(data, str), f"For {suffix} files, data must be a string, got {type(data)}"
        _save_text(data, path, encoding=kwargs.get("encoding", "utf-8"))


def apply_nested_updates(base_dict: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply nested updates to a dictionary with flattened keys.

    Supports dot notation for all fields:
        - Regular: "task_config.max_seq_len"
        - Discriminated lists: "loss_metric_configs.Loss1.coeff"

    For discriminated list fields, matches items by discriminator value in the path.
    Preserves base items not mentioned in updates and adds new items from updates.

    Args:
        base_dict: The base configuration dictionary
        updates: Dictionary of flattened key-value pairs

    Returns:
        Updated dictionary (deep copy, original unchanged)
    """
    result = copy.deepcopy(base_dict)

    for key, value in updates.items():
        if "." in key:
            keys = key.split(".")

            # Check if this is a discriminator-based list key
            # Format: "list_field.discriminator_value.field_name..."
            if len(keys) >= 3 and keys[0] in _DISCRIMINATED_LIST_FIELDS:
                list_field = keys[0]
                discriminator_value = keys[1]
                field_path = keys[2:]  # Remaining path after discriminator

                # Ensure the list exists
                if list_field not in result:
                    result[list_field] = []

                if not isinstance(result[list_field], list):
                    raise ValueError(
                        f"Expected '{list_field}' to be a list, got {type(result[list_field])}"
                    )

                # Find or create the item with matching discriminator
                discriminator_field = _DISCRIMINATED_LIST_FIELDS[list_field]
                target_item = None
                for item in result[list_field]:
                    if item.get(discriminator_field) == discriminator_value:
                        target_item = item
                        break

                if target_item is None:
                    # Create new item with discriminator
                    target_item = {discriminator_field: discriminator_value}
                    result[list_field].append(target_item)

                # Navigate the remaining path within the item
                current_item: dict[str, Any] = target_item
                for k in field_path[:-1]:
                    if k not in current_item:
                        current_item[k] = {}
                    assert isinstance(current_item[k], dict)
                    current_item = current_item[k]

                # Set the final value
                current_item[field_path[-1]] = value
            else:
                # Regular dot notation (non-discriminated)
                current: dict[str, Any] = result

                # Navigate to the parent of the final key
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    assert isinstance(current[k], dict)
                    current = current[k]

                # Set the final value
                current[keys[-1]] = value
        else:
            # Simple key replacement (no dot notation)
            result[key] = value

    return result


def _extract_value_specs_from_sweep_params(
    obj: Any,
    path: list[str],
    value_specs: list[tuple[str, list[Any]]],
) -> None:
    """Recursively extract all {"values": [...]} specs with flattened paths."""
    if isinstance(obj, dict):
        if "values" in obj and len(obj) == 1:
            # This is a value spec - create flattened key
            flattened_key = ".".join(path)
            value_specs.append((flattened_key, obj["values"]))
        else:
            # Regular dict, recurse
            for key, value in obj.items():
                _extract_value_specs_from_sweep_params(value, path + [key], value_specs)
    elif isinstance(obj, list):
        # All lists must be discriminated
        if len(path) == 0:
            raise ValueError("Cannot have a list at the root level of sweep parameters")

        parent_key = path[-1]
        if parent_key not in _DISCRIMINATED_LIST_FIELDS:
            raise ValueError(
                f"List field '{parent_key}' is not in _DISCRIMINATED_LIST_FIELDS. "
                f"All list fields must be discriminated unions. "
                f"Known discriminated fields: {list(_DISCRIMINATED_LIST_FIELDS.keys())}"
            )

        discriminator_field = _DISCRIMINATED_LIST_FIELDS[parent_key]
        seen_discriminators: set[str] = set()

        for item in obj:
            if not isinstance(item, dict):
                raise ValueError(
                    f"All items in discriminated list '{parent_key}' must be dicts, got {type(item)}"
                )
            if discriminator_field not in item:
                raise ValueError(
                    f"Item in discriminated list '{parent_key}' missing discriminator field '{discriminator_field}': {item}"
                )

            disc_value = item[discriminator_field]
            if not isinstance(disc_value, str):
                raise ValueError(
                    f"Discriminator field '{discriminator_field}' must be a string, got {type(disc_value)}: {disc_value}"
                )

            if disc_value in seen_discriminators:
                raise ValueError(
                    f"Duplicate discriminator value '{disc_value}' in list field '{parent_key}'"
                )
            seen_discriminators.add(disc_value)

            # Recurse into item's fields with discriminator in path
            for field_key, field_value in item.items():
                if field_key == discriminator_field:
                    # Skip the discriminator field - it's already in the path
                    continue
                field_path = path + [disc_value, field_key]
                _extract_value_specs_from_sweep_params(field_value, field_path, value_specs)


def _validate_sweep_params_have_values(
    obj: Any,
    path: list[str],
    parent_list_key: str | None = None,
) -> None:
    """Validate that all leaves have {"values": [...]}, except discriminator fields."""
    if isinstance(obj, dict):
        if "values" in obj:
            return  # This is a value spec
        if not obj:
            return  # Empty dict is ok
        for key, value in obj.items():
            _validate_sweep_params_have_values(value, path + [key], parent_list_key)
    elif isinstance(obj, list):
        # Track that we're inside a discriminated list
        list_field = path[-1] if path else None
        for item in obj:
            _validate_sweep_params_have_values(item, path, parent_list_key=list_field)
    else:
        # Primitive value - check if it's a discriminator field
        if parent_list_key and parent_list_key in _DISCRIMINATED_LIST_FIELDS:
            discriminator_field = _DISCRIMINATED_LIST_FIELDS[parent_list_key]
            if path and path[-1] == discriminator_field:
                return  # This is a discriminator field, it's allowed to be a primitive

        # Otherwise, this is an error
        path_str = ".".join(path) if path else "(root)"
        raise ValueError(
            f'All leaf values in sweep parameters must be {{"values": [...]}}, '
            f"but found {type(obj).__name__} at path '{path_str}': {obj}"
        )


def generate_grid_combinations(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all combinations for a grid search from parameter specifications.

    All leaf values (except discriminator fields) must be {"values": [...]}.
    Discriminated lists use discriminator values in flattened keys instead of indices.

    Args:
        parameters: Nested dict/list structure where all leaves are {"values": [...]}

    Returns:
        List of parameter combinations with flattened keys (e.g., "loss_metric_configs.Loss1.coeff")

    Example:
        >>> params = {
        ...     "seed": {"values": [0, 1]},
        ...     "loss_metric_configs": [
        ...         {
        ...             "classname": "ImportanceMinimalityLoss",
        ...             "coeff": {"values": [0.1, 0.2]},
        ...         }
        ...     ],
        ... }
        >>> combos = generate_grid_combinations(params)
        >>> len(combos)
        4
        >>> combos[0]["seed"]
        0
        >>> combos[0]["loss_metric_configs.ImportanceMinimalityLoss.coeff"]
        0.1
    """
    # Extract all value specs with their flattened paths
    value_specs: list[tuple[str, list[Any]]] = []
    _extract_value_specs_from_sweep_params(parameters, [], value_specs)

    # Validate all non-discriminator leaves have {"values": [...]}
    _validate_sweep_params_have_values(parameters, [])

    if not value_specs:
        # No value specs found, return single empty combination
        return [{}]

    # Generate cartesian product of all value specs
    keys, value_lists = zip(*value_specs, strict=True)
    all_value_combinations = list(itertools.product(*value_lists))

    # Create flattened dicts for each combination
    combinations: list[dict[str, Any]] = []
    for value_combo in all_value_combinations:
        combo_dict = dict(zip(keys, value_combo, strict=True))
        combinations.append(combo_dict)

    return combinations


METRIC_CONFIG_SHORT_NAMES: dict[str, str] = {
    # Loss metrics
    "FaithfulnessLoss": "Faith",
    "ImportanceMinimalityLoss": "ImpMin",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "StochasticReconLayerwiseLoss": "StochReconLayer",
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "CIMaskedReconLayerwiseLoss": "CIMaskReconLayer",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "PGDReconLayerwiseLoss": "PGDReconLayer",
    "StochasticHiddenActsReconLoss": "StochHiddenRecon",
    # Eval metrics
    "CEandKLLosses": "CEandKL",
    "CIHistograms": "CIHist",
    "CI_L0": "CI_L0",
    "CIMeanPerComponent": "CIMeanPerComp",
    "ComponentActivationDensity": "CompActDens",
    "IdentityCIError": "IdCIErr",
    "PermutedCIPlots": "PermCIPlots",
    "UVPlots": "UVPlots",
    "StochasticReconSubsetCEAndKL": "StochReconSubCEKL",
}


def _parse_metric_config_key(key: str) -> tuple[str, str, str] | None:
    """Parse a metric config key into (list_field, classname, param).

    Args:
        key: Flattened key like "loss_metric_configs.ImportanceMinimalityLoss.pnorm"

    Returns:
        Tuple of (list_field, classname, param) if it's a metric config key, None otherwise
    """
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] in ("loss_metric_configs", "eval_metric_configs"):
        list_field = parts[0]
        classname = parts[1]
        param = ".".join(parts[2:])  # Handle nested params like "task_config.feature_probability"
        return (list_field, classname, param)
    return None


def generate_run_name(params: dict[str, Any]) -> str:
    """Generate a run name based on sweep parameters.

    Handles special formatting for metric configs (loss_metric_configs, eval_metric_configs)
    by abbreviating classnames and grouping parameters by metric type.

    Args:
        params: Dictionary of flattened sweep parameters

    Returns:
        Formatted run name string

    Example:
        >>> params = {
        ...     "seed": 42,
        ...     "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 0.9,
        ...     "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.001,
        ... }
        >>> generate_run_name(params)
        "seed-42-ImpMin-coeff-0.001-pnorm-0.9"
    """
    # Group parameters by type: regular params and metric config params
    regular_params: list[tuple[str, Any]] = []
    metric_params: dict[str, list[tuple[str, Any]]] = {}  # classname -> [(param, value), ...]

    for key, value in params.items():
        parsed = _parse_metric_config_key(key)
        if parsed:
            _, classname, param = parsed
            # Get short name for the classname
            short_name = METRIC_CONFIG_SHORT_NAMES.get(classname, classname)
            if short_name not in metric_params:
                metric_params[short_name] = []
            metric_params[short_name].append((param, value))
        else:
            regular_params.append((key, value))

    # Build parts list
    parts: list[str] = []

    # Add regular params (sorted for consistency)
    for key, value in sorted(regular_params):
        parts.append(f"{key}-{value}")

    # Add metric config params (sorted by classname, then by param)
    for short_name in sorted(metric_params.keys()):
        parts.append(short_name)
        for param, value in sorted(metric_params[short_name]):
            parts.append(f"{param}-{value}")

    return "-".join(parts)


RunType = Literal["spd", "cluster", "ensemble"]

RUN_TYPE_ABBREVIATIONS: Final[dict[RunType, str]] = {
    "spd": "s",
    "cluster": "c",
    "ensemble": "e",
}


# TODO: This doesnt work in pytest but would in general be nice to enforce. hmm.
# _CREATED_RUN_ID: bool = False


class ExecutionStamp(NamedTuple):
    run_id: str
    snapshot_branch: str
    commit_hash: str
    run_type: RunType

    @staticmethod
    def _generate_run_id(run_type: RunType) -> str:
        """Generate a unique run identifier,

        Format: `{type_abbr}-{random_hex}`
        """
        # global _CREATED_RUN_ID
        # if _CREATED_RUN_ID:
        #     raise RuntimeError(
        #         "Run ID has already been generated for this process! You can only call this once."
        #     )
        type_abbr: str = RUN_TYPE_ABBREVIATIONS[run_type]
        random_hex: str = secrets.token_hex(4)
        # _CREATED_RUN_ID = True
        return f"{type_abbr}-{random_hex}"

    @classmethod
    def create(
        cls,
        run_type: RunType,
        create_snapshot: bool,
    ) -> "ExecutionStamp":
        """create an execution stamp, possibly including a git snapshot branch"""

        run_id: str = ExecutionStamp._generate_run_id(run_type)
        snapshot_branch: str
        commit_hash: str

        if create_snapshot:
            snapshot_branch, commit_hash = create_git_snapshot(run_id=run_id)
            logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")
        else:
            snapshot_branch = repo_current_branch()
            if repo_is_clean(catch_except_as_false=True):
                commit_hash = repo_current_commit_hash()
                logger.info(f"Using current branch: {snapshot_branch} ({commit_hash[:8]})")
            else:
                commit_hash = "none"
                logger.info(
                    f"Using current branch: {snapshot_branch} (unpushed changes, no commit hash)"
                )

        return ExecutionStamp(
            run_id=run_id,
            snapshot_branch=snapshot_branch,
            commit_hash=commit_hash,
            run_type=run_type,
        )

    @property
    def out_dir(self) -> Path:
        """Get the output directory for this execution stamp."""
        run_dir = SPD_CACHE_DIR / self.run_type / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


_NO_ARG_PARSSED_SENTINEL = object()


def read_noneable_str(value: str) -> str | None:
    """Read a string that may be 'None' and convert to None."""
    if value == "None":
        return None
    return value
