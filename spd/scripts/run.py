"""SPD run script for experiments with sweeps and SLURM orchestration.

This script provides a full-featured entry point for running SPD experiments on the
cluster, supporting parameter sweeps, multi-node training, git snapshots, and W&B
workspace views/reports.

For simpler local execution without SLURM, use simple.py instead.
"""

import copy
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import yaml

from spd.log import logger
from spd.settings import DEFAULT_PARTITION_NAME, REPO_ROOT

if TYPE_CHECKING:
    from spd.utils.compute_utils import ComputeStrategy, TrainingJob


def main(
    experiments: str | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    create_report: bool = False,
    report_title: str | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
    num_nodes: int | None = None,
    dp: int | None = None,
    project: str = "spd",
) -> None:
    """Run SPD experiments on SLURM cluster with optional sweeps.

    Args:
        experiments: Comma-separated experiment names (default: all experiments)
        sweep: Enable parameter sweep. Pass True for default params or a YAML path.
        n_agents: Number of concurrent SLURM tasks (required for sweeps)
        create_report: Create a W&B report in addition to workspace view
        report_title: Title for the W&B report (requires create_report=True)
        job_suffix: Suffix for SLURM job names
        cpu: Run on CPU instead of GPU
        partition: SLURM partition name (default: h200-reserved)
        num_nodes: Number of nodes for multi-node training (requires 2+)
        dp: Number of GPUs for single-node data parallelism (requires 2+)
        project: W&B project name
    """
    # Lazy imports to speed up --help (these pull in torch, transformers, etc.)
    from spd.registry import get_max_expected_runtime
    from spd.utils.compute_utils import create_slurm_array_script, submit_slurm_array
    from spd.utils.git_utils import create_git_snapshot

    run_id = _generate_run_id()
    logger.info(f"Run ID: {run_id}")

    experiments_list = _get_experiments(experiments)
    logger.info(f"Experiments: {', '.join(experiments_list)}")

    compute_strategy = _build_compute_strategy(cpu=cpu, dp=dp, num_nodes=num_nodes)
    logger.info(f"Running on {compute_strategy}")

    sweep_params = _get_sweep_params(sweep)
    if sweep_params is not None:
        assert n_agents is not None, "n_agents must be provided when sweep is enabled"

    training_jobs = _create_training_jobs(
        experiments=experiments_list,
        project=project,
        sweep_params=sweep_params,
    )

    snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="run")
    logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")

    _wandb_setup(
        create_report=create_report,
        report_title=report_title,
        project=project,
        run_id=run_id,
        experiments_list=experiments_list,
        snapshot_branch=snapshot_branch,
        commit_hash=commit_hash,
    )

    slurm_job_name = f"spd-{job_suffix or get_max_expected_runtime(experiments_list)}"

    array_script_content = create_slurm_array_script(
        slurm_job_name=slurm_job_name,
        run_id=run_id,
        training_jobs=training_jobs,
        sweep_params=sweep_params,
        snapshot_branch=snapshot_branch,
        compute_strategy=compute_strategy,
        partition=partition,
        max_concurrent_tasks=n_agents,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        array_script_path = temp_path / f"run_array_{run_id}.sh"
        with open(array_script_path, "w") as f:
            f.write(array_script_content)
        array_script_path.chmod(0o755)
        array_job_id = submit_slurm_array(array_script_path)

    logger.section("Job submitted successfully!")
    logger.values(
        {
            "Array Job ID": array_job_id,
            "Total training jobs": len(training_jobs),
            "Max concurrent tasks": n_agents,
            "View logs in": f"~/slurm_logs/slurm-{array_job_id}_*.out",
        }
    )


def _generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _create_training_jobs(
    experiments: list[str],
    project: str,
    sweep_params: dict[str, Any] | None,
) -> list[TrainingJob]:
    """Build a Run containing jobs for all experiments.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """
    from spd.configs import Config
    from spd.registry import EXPERIMENT_REGISTRY
    from spd.utils.compute_utils import TrainingJob
    from spd.utils.run_utils import (
        apply_nested_updates,
        generate_grid_combinations,
        generate_run_name,
    )

    training_jobs: list[TrainingJob] = []

    logger.info("Task breakdown by experiment:")
    task_breakdown: dict[str, str] = {}

    for experiment in experiments:
        exp_config = EXPERIMENT_REGISTRY[experiment]

        # Load base config
        base_config = Config.from_file(exp_config.config_path)

        if sweep_params is None:
            # Fixed configuration run - still use JSON to ensure project override works
            base_config_dict = base_config.model_dump(mode="json")
            base_config_dict["wandb_project"] = project
            config_with_overrides = Config(**base_config_dict)

            training_jobs.append(
                TrainingJob(
                    experiment=experiment,
                    script_path=exp_config.decomp_script,
                    config=config_with_overrides,
                )
            )
            task_breakdown[experiment] = "1 job"

        else:
            # Parameter sweep run
            exp_sweep_params = _get_experiment_sweep_params(experiment, sweep_params)

            combinations = generate_grid_combinations(exp_sweep_params)

            for i, param_combo in enumerate(combinations):
                # Apply parameter overrides
                base_config_dict = base_config.model_dump(mode="json")
                config_dict_with_overrides = apply_nested_updates(base_config_dict, param_combo)
                config_dict_with_overrides["wandb_project"] = project
                wandb_run_name = f"{experiment}-{generate_run_name(param_combo)}"
                config_dict_with_overrides["wandb_run_name"] = wandb_run_name
                config_with_overrides = Config(**config_dict_with_overrides)

                training_jobs.append(
                    TrainingJob(
                        experiment=experiment,
                        script_path=exp_config.decomp_script,
                        config=config_with_overrides,
                    )
                )

                # Print first combination as example
                if i == 0:
                    logger.info(f"  {experiment}: {len(combinations)} jobs")
                    logger.info(f"    Example param overrides: {param_combo}")

    if task_breakdown:
        logger.values(task_breakdown)

    return training_jobs


def _get_experiment_sweep_params(
    experiment_name: str, sweep_params: dict[str, Any]
) -> dict[str, Any]:
    assert experiment_name != "global"

    # Start with global parameters if they exist
    params = copy.deepcopy(sweep_params["global"]) if "global" in sweep_params else {}

    # Merge experiment-specific parameters if they exist
    if experiment_name in sweep_params:
        experiment_params = sweep_params[experiment_name]
        _merge_sweep_params(params, experiment_params)

    if not params:
        raise ValueError(f"No sweep parameters found for experiment '{experiment_name}'")

    return params


def _merge_sweep_params(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override parameters into base parameters.

    Handles nested parameter structures and overwrites values from base with override.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Both are dicts, merge recursively
            _merge_sweep_params(base[key], value)
        else:
            # Override the value
            base[key] = value


def _get_experiments(
    experiments_list_str: str | None = None,
) -> list[str]:
    """Get and validate the list of experiments to run based on the input string.

    Args:
        experiments: Comma-separated list of experiment names. If None, runs all experiments.

    Returns:
        List of experiment names to run.
    """
    from spd.registry import EXPERIMENT_REGISTRY

    # Determine experiment list
    if experiments_list_str is None:
        experiments = list(EXPERIMENT_REGISTRY.keys())
    else:
        experiments = [exp.strip() for exp in experiments_list_str.split(",")]

    # Validate experiment names
    invalid_experiments = [exp for exp in experiments if exp not in EXPERIMENT_REGISTRY]
    if invalid_experiments:
        raise ValueError(f"Invalid experiments: {invalid_experiments}")

    return experiments


def _build_compute_strategy(
    cpu: bool,
    dp: int | None,
    num_nodes: int | None,
) -> ComputeStrategy:
    """Construct a compute strategy from CLI arguments."""
    from spd.utils.compute_utils import CpuOrSingleGpu, MultiGpu, MultiNode

    if cpu:
        assert dp is None, "dp should not be specified when running on cpu"
        assert num_nodes is None, "num_nodes should not be specified when running on cpu"
        return CpuOrSingleGpu()

    match num_nodes, dp:
        case None, None:
            return CpuOrSingleGpu()
        case None, dp:
            assert dp >= 2, "if given, dp must be at least 2. pass dp=None to use a single GPU."
            return MultiGpu(n_gpus=dp)
        case num_nodes, None:
            assert num_nodes >= 2, (
                "if given, num_nodes must be at least 2. pass num_nodes=None to use a single node."
            )
            return MultiNode(n_nodes=num_nodes)
        case _, _:
            raise ValueError(
                "Do not specify both num_nodes and dp. for multi-node runs, assume 8 GPUs per node."
            )


def _get_sweep_params(sweep: str | bool) -> dict[str, Any] | None:
    if sweep is False:
        return None
    sweep_params_file = "sweep_params.yaml" if sweep is True else sweep
    sweep_params_path = _resolve_sweep_params_path(sweep_params_file)
    with open(sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)
    return sweep_params


def _resolve_sweep_params_path(sweep_params_file: str) -> Path:
    """Resolve the full path to the sweep parameters file."""
    if "/" not in sweep_params_file:
        # Look in scripts directory by default
        return REPO_ROOT / "spd/scripts" / sweep_params_file
    else:
        return REPO_ROOT / sweep_params_file


def _wandb_setup(
    create_report: bool,
    report_title: str | None,
    project: str,
    run_id: str,
    experiments_list: list[str],
    snapshot_branch: str,
    commit_hash: str,
) -> None:
    """Set up W&B workspace view and optionally a report."""
    from spd.utils.wandb_utils import ReportCfg, create_view_and_report

    match create_report, report_title:
        case True, None:
            create_view_and_report(
                project=project,
                run_id=run_id,
                experiments=experiments_list,
                report_cfg=None,
            )
        case True, title:
            create_view_and_report(
                project=project,
                run_id=run_id,
                experiments=experiments_list,
                report_cfg=ReportCfg(
                    report_title=title,
                    branch=snapshot_branch,
                    commit_hash=commit_hash,
                ),
            )
        case False, None:
            pass  # No report requested, nothing to do
        case False, title:
            raise ValueError(
                f"got report_title='{title}' but create_report=False. "
                "did you intend to create a report? if so, set create_report=True"
            )


def cli():
    fire.Fire(main)
