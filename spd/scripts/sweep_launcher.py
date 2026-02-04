"""Standalone sweep launcher for LM experiments.

Takes config files with inline __variants__, expands them, and submits to SLURM.

    seed:
      __variants__: [0, 1, 2]        # Scalar sweep
    ci_config:
      __variants__:                   # Object sweep
        - mode: layerwise
        - mode: global

Nested sweeps supported - __variants__ within a variant expands that branch.
Conditional inclusion - use null to exclude items from lists.

Usage:
    spd-sweep config1.yaml config2.yaml --n_agents 4
    spd-sweep config1.yaml --dry-run
"""

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from spd.configs import Config
from spd.log import logger
from spd.settings import REPO_ROOT
from spd.utils.compute_utils import TrainingJob, create_slurm_array_script
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import submit_slurm_job

DECOMP_SCRIPT = Path("spd/experiments/lm/lm_decomposition.py")
DEFAULT_RUNTIME_MINUTES = 360
DEFAULT_PROJECT = "spd"
DEFAULT_PARTITION = "h200-reserved"
VARIANTS_KEY = "__variants__"


def expand_variants(obj: Any) -> list[Any]:
    """Recursively expand all __variants__ in an object into all possible versions.

    None values in lists are filtered out (enables conditional inclusion).
    """
    if isinstance(obj, dict):
        if VARIANTS_KEY in obj:
            assert len(obj) == 1, f"{VARIANTS_KEY} must be the only key, got: {list(obj.keys())}"
            variants_list = obj[VARIANTS_KEY]
            assert isinstance(variants_list, list), f"{VARIANTS_KEY} must be a list"
            # Recursively expand each variant
            all_expanded: list[Any] = []
            for variant in variants_list:
                all_expanded.extend(expand_variants(variant))
            return all_expanded
        else:
            # Regular dict - expand each value and take cartesian product
            if not obj:
                return [{}]

            keys = list(obj.keys())
            value_expansions = [expand_variants(obj[k]) for k in keys]

            dict_results: list[dict[str, Any]] = []
            for combo in itertools.product(*value_expansions):
                dict_results.append(dict(zip(keys, combo, strict=True)))
            return dict_results

    elif isinstance(obj, list):
        # Expand each element and take cartesian product
        if not obj:
            return [[]]

        element_expansions = [expand_variants(elem) for elem in obj]
        list_results: list[list[Any]] = []
        for combo in itertools.product(*element_expansions):
            # Filter out None values (allows conditional inclusion via __variants__: [item, null])
            list_results.append([item for item in combo if item is not None])
        return list_results

    else:
        return [obj]


def find_varying_fields(configs: list[dict[str, Any]], prefix: str = "") -> set[str]:
    """Find all field paths that vary across configs."""
    if not configs:
        return set()

    varying: set[str] = set()
    # Get all keys across all configs
    all_keys: set[str] = set()
    for cfg in configs:
        all_keys.update(cfg.keys())

    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        values = [cfg.get(key) for cfg in configs]

        # Check if values differ
        first = values[0]
        differs = any(v != first for v in values[1:])

        if differs:
            # Check if all values are dicts - recurse to find specific differing fields
            if all(isinstance(v, dict) for v in values if v is not None):
                sub_varying = find_varying_fields(
                    [v for v in values if isinstance(v, dict)], full_key
                )
                if sub_varying:
                    varying.update(sub_varying)
                else:
                    varying.add(full_key)
            else:
                varying.add(full_key)

    return varying


def format_diff(config: dict[str, Any], varying_fields: set[str]) -> str:
    """Format the varying fields of a config as a concise string."""
    # Fields to skip in output
    skip_fields = {"wandb_run_name", "wandb_project"}

    parts: list[str] = []

    for field in sorted(varying_fields):
        short_field = field.split(".")[-1]
        if short_field in skip_fields:
            continue

        value = get_nested_value(config, field)
        # Skip None values for cleaner output
        if value is None:
            continue

        formatted = format_value(value)
        parts.append(f"{short_field}={formatted}")

    return ", ".join(parts) if parts else "(no variation)"


def get_nested_value(d: dict[str, Any], path: str) -> Any:
    """Get a value from a nested dict using dot notation path."""
    keys = path.split(".")
    current: Any = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def format_value(value: Any) -> str:
    """Format a value concisely for display."""
    if value is None:
        return "None"
    if isinstance(value, dict):
        # Show key identifying fields
        if "mode" in value:
            return value["mode"]
        if "fn_type" in value:
            return value["fn_type"]
        if "classname" in value:
            return value["classname"]
        # Fallback: show first few keys
        keys = list(value.keys())[:2]
        return "{" + ",".join(keys) + "...}"
    if isinstance(value, list):
        if len(value) <= 3 and all(isinstance(v, (int, float)) for v in value):
            return "[" + ",".join(str(v) for v in value) + "]"
        return f"[...{len(value)}]"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) >= 1000:
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)


def expand_config_file(config_path: Path) -> list[dict[str, Any]]:
    """Expand a config file, expanding any __variants__ into multiple configs."""
    raw_config = yaml.safe_load(config_path.read_text())
    return expand_variants(raw_config)


def launch(
    *config_files: str,
    n_agents: int | None = None,
    project: str = DEFAULT_PROJECT,
    partition: str = DEFAULT_PARTITION,
    dry_run: bool = False,
) -> None:
    """Launch sweep jobs from config files. n_agents required if total jobs > 1."""
    assert config_files, "No config files specified"
    paths = list(config_files)

    # Resolve paths
    resolved_paths: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        assert path.exists(), f"Config file not found: {path}"
        resolved_paths.append(path)

    logger.section("SWEEP LAUNCHER")
    logger.info(f"Config files: {len(resolved_paths)}")
    for p in resolved_paths:
        try:
            display_path = p.relative_to(REPO_ROOT)
        except ValueError:
            display_path = p
        logger.info(f"  - {display_path}")

    # Expand all configs
    all_config_dicts: list[dict[str, Any]] = []
    for config_path in resolved_paths:
        expanded = expand_config_file(config_path)
        logger.info(f"  {config_path.name}: {len(expanded)} jobs")
        all_config_dicts.extend(expanded)

    logger.info(f"Total jobs: {len(all_config_dicts)}")

    if len(all_config_dicts) == 0:
        logger.warning("No jobs to run!")
        return

    assert not (len(all_config_dicts) > 1 and n_agents is None), (
        f"n_agents required for multiple jobs (got {len(all_config_dicts)})"
    )

    # For single job, default n_agents to 1
    if n_agents is None:
        n_agents = 1

    # Find what varies across all configs for naming
    varying = find_varying_fields(all_config_dicts)

    # Create TrainingJobs with task index in run name
    all_jobs: list[TrainingJob] = []
    for task_idx, config_dict in enumerate(all_config_dicts):
        config_dict["wandb_project"] = project

        # Generate run name: {task_idx}_{diff_summary}
        if len(all_config_dicts) > 1:
            diff_str = format_diff(config_dict, varying)
            config_dict["wandb_run_name"] = f"{task_idx}_{diff_str}"

        config = Config(**config_dict)
        all_jobs.append(
            TrainingJob(
                experiment=f"task_{task_idx}",
                script_path=DECOMP_SCRIPT,
                config=config,
            )
        )

    if dry_run:
        logger.section("DRY RUN - Jobs that would be submitted:")
        for job in all_jobs:
            logger.info(f"  {job.config.wandb_run_name}")
        return

    # Create git snapshot
    run_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id=run_id)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    hours = DEFAULT_RUNTIME_MINUTES // 60
    mins = DEFAULT_RUNTIME_MINUTES % 60
    runtime_str = f"{hours}h{mins}m"
    slurm_job_name = f"sweep-{runtime_str}"

    array_script = create_slurm_array_script(
        slurm_job_name=slurm_job_name,
        run_id=run_id,
        training_jobs=all_jobs,
        sweep_params=None,
        snapshot_branch=snapshot_branch,
        n_gpus=None,
        partition=partition,
        max_concurrent_tasks=n_agents,
    )

    # Submit
    result = submit_slurm_job(
        array_script,
        f"sweep_array_{run_id}",
        is_array=True,
        n_array_tasks=len(all_jobs),
    )

    logger.section("Job submitted!")
    logger.values(
        {
            "Array Job ID": result.job_id,
            "Total jobs": len(all_jobs),
            "Max concurrent": n_agents,
            "Logs": result.log_pattern,
            "Script": str(result.script_path),
        }
    )


def main() -> None:
    import fire

    fire.Fire(launch)


if __name__ == "__main__":
    main()
