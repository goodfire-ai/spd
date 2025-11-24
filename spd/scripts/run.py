"""Unified SPD runner for experiments with optional parameter sweeps.

This script provides a single entry point for running SPD experiments, supporting both
fixed configurations and parameter sweeps. All runs are tracked in W&B with workspace
views created for each experiment.

For full CLI usage and examples, see the bottom of this file (or run `spd-run --help`).
"""

import copy
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import yaml

from spd.configs import Config
from spd.log import LogFormat, logger
from spd.registry import EXPERIMENT_REGISTRY, get_max_expected_runtime
from spd.settings import DEFAULT_PARTITION_NAME, REPO_ROOT
from spd.utils.compute_utils import (
    Command,
    Cpu,
    LaunchConfig,
    MultiNode,
    SingleGpu,
    SingleNode,
    create_slurm_array_script,
    get_command,
    submit_slurm_array,
)
from spd.utils.git_utils import create_git_snapshot
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations, generate_run_name
from spd.utils.wandb_utils import ReportCfg, create_view_and_report


@dataclass
class CreateViewAndReport:
    title: str | None


class CreateViewOnly: ...


WandbCfg = CreateViewAndReport | CreateViewOnly


@dataclass()
class SlurmPartition:
    """Represents launching jobs asynchronously via SLURM"""

    name: str


class Local:
    """Represents running job directy in the cli using the --local flag"""


# Encode into the type system the fact that we can only run multi-node jobs via SLURM. This avoids
# nasty runtime checks and edge cases.
ComputeEnvironment = (
    tuple[SlurmPartition, Cpu | SingleGpu | SingleNode | MultiNode]
    | tuple[Local, Cpu | SingleGpu | SingleNode]
)


def main(
    experiments: str | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    use_wandb: bool = True,
    create_report: bool = True,
    report_title: str | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str | None = None,
    num_nodes: int | None = None,
    dp: int | None = None,
    project: str = "spd",
    local: bool = False,
    log_format: LogFormat = "default",
) -> None:
    logger.set_format("console", log_format)

    run_id: str = _generate_run_id()
    logger.info(f"Run ID: {run_id}")

    experiments_list: list[str] = _get_experiments(experiments)
    logger.info(f"Experiments: {', '.join(experiments_list)}")

    sweep_params = _get_sweep_params(sweep)

    if local and partition is not None:
        logger.warning("Running locally. setting partition to None.")
        partition = None

    compute_env = _build_compute_env(
        cpu=cpu,
        local=local,
        dp=dp,
        num_nodes=num_nodes,
        partition_name=partition,
    )

    n_agents = _validate_agent_count(n_agents, experiments_list, sweep_params, local)

    job_configs = make_launch_configs(
        run_id=run_id,
        experiments=experiments_list,
        sweep_params=sweep_params,
        project=project,
    )

    snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="run")
    logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")

    _wandb_setup(
        use_wandb=use_wandb,
        create_report=create_report,
        report_title=report_title,
        project=project,
        run_id=run_id,
        experiments_list=experiments_list,
        snapshot_branch=snapshot_branch,
        commit_hash=commit_hash,
    )

    match compute_env:
        case Local(), compute_strategy:
            commands = [get_command(job_config, compute_strategy) for job_config in job_configs]
            _run_commands_locally(commands)

        case SlurmPartition(name=partition_name), compute_strategy:
            assert snapshot_branch is not None
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                array_script_path = temp_path / f"run_array_{run_id}.sh"
                job_name = f"spd-{job_suffix or get_max_expected_runtime(experiments_list)}"

                array_script_content = create_slurm_array_script(
                    job_name=job_name,
                    runs=job_configs,
                    snapshot_branch=snapshot_branch,
                    compute_strategy=compute_strategy,
                    partition=partition_name,
                    max_concurrent_tasks=n_agents,
                )

                with open(array_script_path, "w") as f:
                    f.write(array_script_content)

                # Make script executable
                array_script_path.chmod(0o755)

                array_job_id = submit_slurm_array(array_script_path)

                logger.section("Job submitted successfully!")
                logger.values(
                    {
                        "Array Job ID": array_job_id,
                        "Total tasks": len(job_configs),
                        "Max concurrent tasks": n_agents,
                        "View logs in": f"~/slurm_logs/slurm-{array_job_id}_*.out",
                    }
                )


def _generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _validate_agent_count(
    n_agents: int | None,
    experiments_list: list[str],
    sweep_params: dict[str, Any] | None,
    local: bool,
) -> int | None:
    if sweep_params is not None:
        if not local:
            assert n_agents is not None, (
                "n-agents must be provided if sweep is enabled (unless running with --local)"
            )
        else:
            n_agents = len(experiments_list)
            assert n_agents is not None, (
                "n-agents must be provided if sweep is enabled (unless running with --local)"
            )
    return n_agents


def make_launch_configs(
    run_id: str,
    experiments: list[str],
    project: str,
    sweep_params: dict[str, Any] | None,
) -> list[LaunchConfig]:
    """Generate commands for all experiment runs and print task counts.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """
    jobs: list[LaunchConfig] = []

    logger.info("Task breakdown by experiment:")
    task_breakdown: dict[str, str] = {}

    cmd_idx: int = 0

    for experiment in experiments:
        exp_config = EXPERIMENT_REGISTRY[experiment]

        # Load base config
        base_config = Config.from_file(exp_config.config_path)

        if sweep_params is None:
            # Fixed configuration run - still use JSON to ensure project override works
            base_config_dict = base_config.model_dump(mode="json")
            base_config_dict["wandb_project"] = project
            config_with_overrides = Config(**base_config_dict)

            jobs.append(
                LaunchConfig(
                    run_id=run_id,
                    idx=cmd_idx,
                    script_path=exp_config.decomp_script,
                    config=config_with_overrides,
                    experiment=experiment,
                    sweep_params=sweep_params,
                )
            )
            task_breakdown[experiment] = "1 task"
            cmd_idx += 1

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

                jobs.append(
                    LaunchConfig(
                        run_id=run_id,
                        idx=cmd_idx,
                        script_path=exp_config.decomp_script,
                        config=config_with_overrides,
                        experiment=experiment,
                        sweep_params=sweep_params,
                    )
                )
                cmd_idx += 1

                # Print first combination as example
                if i == 0:
                    logger.info(f"  {experiment}: {len(combinations)} tasks")
                    logger.info(f"    Example param overrides: {param_combo}")

    if task_breakdown:
        logger.values(task_breakdown)

    return jobs


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


def _run_commands_locally(commands: list[Command]) -> None:
    """Execute commands locally in sequence.

    Args:
        commands: List of shell commands to execute
    """

    logger.section(f"LOCAL EXECUTION: Running {len(commands)} tasks")

    for i, command in enumerate(commands, 1):
        # Parse command into arguments
        args = shlex.split(command.command)

        # Extract experiment name from script path for cleaner output
        # Skip environment variables (VAR=value format) to find the python script
        script_path = next(arg for arg in args if "=" not in arg and arg.endswith(".py"))
        script_name = script_path.split("/")[-1]
        logger.section(f"[{i}/{len(commands)}] Executing: {script_name}...")

        result = subprocess.run(command.command, env=command.env_vars, shell=True)

        if result.returncode != 0:
            logger.warning(
                f"[{i}/{len(commands)}] ⚠️  Warning: Command failed with exit code {result.returncode}"
            )
        else:
            logger.info(f"[{i}/{len(commands)}] ✓ Completed successfully")

    logger.section("LOCAL EXECUTION COMPLETE")


def _get_experiments(
    experiments_list_str: str | None = None,
) -> list[str]:
    """Get and validate the list of experiments to run based on the input string.

    Args:
        experiments: Comma-separated list of experiment names. If None, runs all experiments.

    Returns:
        List of experiment names to run.
    """
    # Determine experiment list
    experiments: list[str]
    if experiments_list_str is None:
        experiments = list(EXPERIMENT_REGISTRY.keys())
    else:
        experiments = [exp.strip() for exp in experiments_list_str.split(",")]

    # Validate experiment names
    invalid_experiments: list[str] = [exp for exp in experiments if exp not in EXPERIMENT_REGISTRY]
    if invalid_experiments:
        available: str = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Invalid experiments: {invalid_experiments}. Available experiments: {available}"
        )

    return experiments


def _build_compute_env(
    cpu: bool,
    local: bool,
    dp: int | None,
    num_nodes: int | None,
    partition_name: str | None,
) -> ComputeEnvironment:
    """Construct a compute strategy from CLI arguments."""
    if local or cpu:
        assert dp is None, "dp should not be specified when running locally or on cpu."
        assert num_nodes is None, "num_nodes should not be specified when running locally or on cpu."
        assert partition_name is None, "partition_name should not be specified when running locally or on cpu."
        return (Local(), Cpu())

    partition = SlurmPartition(
        name=partition_name if partition_name is not None else DEFAULT_PARTITION_NAME
    )

    match num_nodes, dp:
        case None, None:
            strategy = SingleGpu()
        case None, dp:
            assert dp >= 2, "if given, dp must be at least 2. pass dp=None to use a single GPU."
            strategy = SingleNode(n_gpus=dp)
        case num_nodes, None:
            assert num_nodes >= 2, (
                "if given, num_nodes must be at least 2. pass num_nodes=None to use a single node."
            )
            strategy = MultiNode(n_nodes=num_nodes)
        case _, _:
            raise ValueError(
                "Do not specifiy both num_nodes and dp. for multi-node runs, assume 8 GPUs per node."
            )

    return (partition, strategy)


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
    use_wandb: bool,
    create_report: bool,
    report_title: str | None,
    project: str,
    run_id: str,
    experiments_list: list[str],
    snapshot_branch: str,
    commit_hash: str,
) -> None:
    match use_wandb, create_report, report_title:
        case True, True, title:
            assert snapshot_branch is not None and commit_hash is not None
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
        case True, False, None:
            create_view_and_report(
                project=project,
                run_id=run_id,
                experiments=experiments_list,
                report_cfg=None,
            )
        case True, False, str(title):
            raise ValueError(
                f"got report_title='{title}' but create_report=False. did you intend to create a report? if so, set create_report=True"
            )
        case False, _, _:
            logger.warning(
                "No workspace views or reports will be created. set `use_wandb=True` and optionally `create_report=True` to enable."
            )


def cli():
    fire.Fire(main)
