"""Standalone sweep launcher for LM experiments.

Simple entry point that takes config files with inline __variants__,
expands them all, and submits to SLURM.

Sweep syntax uses __variants__ inline wherever you want to sweep:

    seed:
      __variants__: [0, 1, 2]        # Scalar sweep
    ci_config:
      __variants__:                   # Object sweep
        - mode: layerwise
          hidden_dims: [100]
        - mode: global
          hidden_dims: [256]
    steps: 200000                     # Non-swept fields stay normal
    batch_size: 64

Nested sweeps are supported - __variants__ within a variant expands that branch:

    ci_config:
      __variants__:
        - mode: layerwise
          hidden_dims:
            __variants__: [[100], [200], [500]]  # 3 options for layerwise
        - mode: global
          hidden_dims: [256]                      # 1 option for global
    # Total: 4 ci_config variants

Usage:
    python -m spd.scripts.sweep_launcher config1.yaml config2.yaml --n_agents 4
    python -m spd.scripts.sweep_launcher config1.yaml --dry-run

Or edit CONFIG_FILES list below and run:
    python -m spd.scripts.sweep_launcher
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

# ============================================================================
# CONFIGURATION - Edit these as needed
# ============================================================================

# Default config files to run (relative to REPO_ROOT)
# Override by passing paths as command line arguments
CONFIG_FILES: list[str] = [
    # "spd/experiments/lm/my_sweep_config.yaml",
]

# Hardcoded for LM experiments
DECOMP_SCRIPT = Path("spd/experiments/lm/lm_decomposition.py")
DEFAULT_EXPECTED_RUNTIME_MINUTES = 360  # 6 hours
DEFAULT_PROJECT = "spd"
DEFAULT_PARTITION = "h200-reserved"

VARIANTS_KEY = "__variants__"

# ============================================================================
# Sweep Expansion
# ============================================================================


def expand_variants(obj: Any) -> list[Any]:
    """Recursively expand all __variants__ in an object.

    Returns a list of all possible fully-expanded versions of the object.

    Examples:
        expand_variants({"__variants__": [1, 2]})
        # Returns: [1, 2]

        expand_variants({"a": {"__variants__": [1, 2]}, "b": 3})
        # Returns: [{"a": 1, "b": 3}, {"a": 2, "b": 3}]

        expand_variants({"__variants__": [{"x": {"__variants__": [1, 2]}}, {"y": 3}]})
        # Returns: [{"x": 1}, {"x": 2}, {"y": 3}]
    """
    if isinstance(obj, dict):
        if VARIANTS_KEY in obj:
            # This dict IS a variants node
            if len(obj) != 1:
                raise ValueError(
                    f"{VARIANTS_KEY} must be the only key in its dict, "
                    f"got: {list(obj.keys())}"
                )
            variants_list = obj[VARIANTS_KEY]
            if not isinstance(variants_list, list):
                raise ValueError(
                    f"{VARIANTS_KEY} value must be a list, got {type(variants_list)}"
                )
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
            list_results.append(list(combo))
        return list_results

    else:
        # Scalar - just return as-is (wrapped in list for uniformity)
        return [obj]


# ============================================================================
# Diff Detection
# ============================================================================


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


# ============================================================================
# Run Name Generation
# ============================================================================


def generate_run_name_from_config(experiment: str, config: dict[str, Any]) -> str:
    """Generate a concise, readable run name from config dict.

    Extracts key identifying fields from the config.
    """
    parts = [experiment]

    # Key fields to include in run name (in order of priority)
    key_fields = [
        ("seed", lambda v: str(v)),
        ("ci_config", lambda v: v.get("mode", "") if isinstance(v, dict) else ""),
        ("ci_config", lambda v: v.get("fn_type", "")[:20] if isinstance(v, dict) else ""),
    ]

    for field, extractor in key_fields:
        if field in config:
            val = extractor(config[field])
            if val:
                parts.append(val)

    # Add hidden_dims if present in ci_config
    ci_config = config.get("ci_config", {})
    if isinstance(ci_config, dict) and "hidden_dims" in ci_config:
        hd = ci_config["hidden_dims"]
        if isinstance(hd, list) and len(hd) <= 3:
            parts.append("hd=" + "x".join(str(x) for x in hd))

    return "-".join(parts)


# ============================================================================
# Config Parsing and Job Creation
# ============================================================================


def expand_config_file(
    config_path: Path,
    project: str,
) -> list[TrainingJob]:
    """Expand a single config file into TrainingJobs.

    Any __variants__ in the config are expanded. If no __variants__ present,
    returns a single job.
    """
    raw_config = yaml.safe_load(config_path.read_text())

    # Use config filename (without extension) as experiment name
    experiment_name = config_path.stem

    # Expand all variants in the config
    expanded_configs = expand_variants(raw_config)

    jobs: list[TrainingJob] = []
    for config_dict in expanded_configs:
        config_dict["wandb_project"] = project

        # Generate descriptive run name if multiple configs
        if len(expanded_configs) > 1:
            # Find what varied by comparing to first config
            run_name = generate_run_name_from_config(experiment_name, config_dict)
            config_dict["wandb_run_name"] = run_name

        config = Config(**config_dict)
        jobs.append(
            TrainingJob(
                experiment=experiment_name,
                script_path=DECOMP_SCRIPT,
                config=config,
            )
        )

    return jobs


# ============================================================================
# Launch
# ============================================================================


def launch(
    config_files: list[str] | None = None,
    n_agents: int | None = None,
    project: str = DEFAULT_PROJECT,
    partition: str = DEFAULT_PARTITION,
    cpu: bool = False,
    dry_run: bool = False,
) -> None:
    """Launch sweep jobs from config files.

    Args:
        config_files: List of config file paths (relative to REPO_ROOT or absolute).
                     If None, uses CONFIG_FILES constant.
        n_agents: Max concurrent SLURM tasks. Required if total jobs > 1.
        project: W&B project name.
        partition: SLURM partition.
        cpu: Run on CPU instead of GPU.
        dry_run: Print what would be submitted without actually submitting.
    """
    paths = config_files or CONFIG_FILES
    if not paths:
        raise ValueError(
            "No config files specified. Either pass paths as arguments "
            "or edit CONFIG_FILES in this file."
        )

    # Resolve paths
    resolved_paths: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        resolved_paths.append(path)

    logger.section("SWEEP LAUNCHER")
    logger.info(f"Config files: {len(resolved_paths)}")
    for p in resolved_paths:
        try:
            display_path = p.relative_to(REPO_ROOT)
        except ValueError:
            display_path = p
        logger.info(f"  - {display_path}")

    # Expand all configs into jobs
    all_jobs: list[TrainingJob] = []
    for config_path in resolved_paths:
        jobs = expand_config_file(config_path, project)
        logger.info(f"  {config_path.name}: {len(jobs)} jobs")
        all_jobs.extend(jobs)

    logger.info(f"Total jobs: {len(all_jobs)}")

    if len(all_jobs) == 0:
        logger.warning("No jobs to run!")
        return

    if len(all_jobs) > 1 and n_agents is None:
        raise ValueError(
            f"n_agents must be specified when running multiple jobs (got {len(all_jobs)} jobs)"
        )

    # For single job, default n_agents to 1
    if n_agents is None:
        n_agents = 1

    if dry_run:
        logger.section("DRY RUN - Jobs that would be submitted:")
        # Find what varies across jobs
        config_dicts = [job.config.model_dump() for job in all_jobs]
        varying = find_varying_fields(config_dicts)
        for i, job in enumerate(all_jobs):
            diff_str = format_diff(job.config.model_dump(), varying)
            logger.info(f"  [{i}] {diff_str}")
        return

    # Create git snapshot
    run_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id=run_id)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # Build runtime string for job name
    hours = DEFAULT_EXPECTED_RUNTIME_MINUTES // 60
    mins = DEFAULT_EXPECTED_RUNTIME_MINUTES % 60
    runtime_str = f"{hours}h{mins}m"
    slurm_job_name = f"sweep-{runtime_str}"

    # Create SLURM array script
    n_gpus = None if cpu else None  # Single GPU per job
    array_script = create_slurm_array_script(
        slurm_job_name=slurm_job_name,
        run_id=run_id,
        training_jobs=all_jobs,
        sweep_params=None,  # Already expanded
        snapshot_branch=snapshot_branch,
        n_gpus=n_gpus,
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
    logger.values({
        "Array Job ID": result.job_id,
        "Total jobs": len(all_jobs),
        "Max concurrent": n_agents,
        "Logs": result.log_pattern,
        "Script": str(result.script_path),
    })


def main() -> None:
    import sys

    args = sys.argv[1:]

    # Parse simple flags
    n_agents: int | None = None
    dry_run = False
    cpu = False
    config_files: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--n_agents" and i + 1 < len(args):
            n_agents = int(args[i + 1])
            i += 2
        elif arg == "--dry-run":
            dry_run = True
            i += 1
        elif arg == "--cpu":
            cpu = True
            i += 1
        elif arg.startswith("--"):
            raise ValueError(f"Unknown flag: {arg}")
        else:
            config_files.append(arg)
            i += 1

    launch(
        config_files=config_files if config_files else None,
        n_agents=n_agents,
        dry_run=dry_run,
        cpu=cpu,
    )


if __name__ == "__main__":
    main()
