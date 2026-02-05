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
from spd.utils.wandb_utils import generate_wandb_run_name

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


def flatten_config(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into dot-notation keys."""
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten_config(value, full_key))
            else:
                result[full_key] = value
    else:
        if prefix:
            result[prefix] = obj
    return result


def find_varying_params(configs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Find params that vary across configs, returns {param: [values...]}."""
    flattened = [flatten_config(cfg) for cfg in configs]
    all_keys = set().union(*[set(f.keys()) for f in flattened])

    varying: dict[str, list[Any]] = {}
    for key in all_keys:
        values = [f.get(key) for f in flattened]
        if any(v != values[0] for v in values[1:]):
            varying[key] = values
    return varying


def extract_varying_params(config: dict[str, Any], varying_keys: set[str]) -> dict[str, Any]:
    """Extract only the varying params from a config as flattened dict."""
    flat = flatten_config(config)
    return {k: v for k, v in flat.items() if k in varying_keys}


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
    varying = find_varying_params(all_config_dicts)
    varying_keys = set(varying.keys()) - {"wandb_run_name", "wandb_project"}

    # Create TrainingJobs with task index in run name
    all_jobs: list[TrainingJob] = []
    for task_idx, config_dict in enumerate(all_config_dicts):
        config_dict["wandb_project"] = project

        # Generate run name using same format as original sweep code
        if len(all_config_dicts) > 1:
            varying_params = extract_varying_params(config_dict, varying_keys)
            run_name = generate_wandb_run_name(varying_params)
            config_dict["wandb_run_name"] = f"{task_idx}-{run_name}"

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
