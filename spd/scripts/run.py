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
from spd.utils.distributed_utils import (
    ComputeEnvironment,
    ComputeStrategy,
    Cpu,
    Local,
    MultiNode,
    SingleNode,
    SlurmPartition,
)
from spd.utils.git_utils import create_git_snapshot
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations, generate_run_name
from spd.utils.slurm_utils import create_slurm_array_script, submit_slurm_array
from spd.utils.wandb_utils import ReportCfg, wandb_setup


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


def _generate_commands(
    run_id: str,
    experiments: list[str],
    compute_strategy: ComputeStrategy,
    project: str,
    sweep_params: dict[str, Any] | None,
) -> list[str]:
    """Generate commands for all experiment runs and print task counts.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """
    commands: list[str] = []

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

            command = compute_strategy.get_command(
                run_id=run_id,
                idx=cmd_idx,
                script_path=exp_config.decomp_script,
                config=config_with_overrides,
                experiment=experiment,
            )
            commands.append(command)
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

                command = compute_strategy.get_command(
                    run_id=run_id,
                    idx=cmd_idx,
                    script_path=exp_config.decomp_script,
                    config=config_with_overrides,
                    experiment=experiment,
                    sweep_params=sweep_params,
                )
                commands.append(command)
                cmd_idx += 1

                # Print first combination as example
                if i == 0:
                    logger.info(f"  {experiment}: {len(combinations)} tasks")
                    logger.info(f"    Example param overrides: {param_combo}")

    if task_breakdown:
        logger.values(task_breakdown)

    return commands


def run_commands_locally(commands: list[str]) -> None:
    """Execute commands locally in sequence.

    Args:
        commands: List of shell commands to execute
    """

    logger.section(f"LOCAL EXECUTION: Running {len(commands)} tasks")

    for i, command in enumerate(commands, 1):
        # Parse command into arguments
        args = shlex.split(command)

        # Extract experiment name from script path for cleaner output
        # Skip environment variables (VAR=value format) to find the python script
        script_path = next(arg for arg in args if "=" not in arg and arg.endswith(".py"))
        script_name = script_path.split("/")[-1]
        logger.section(f"[{i}/{len(commands)}] Executing: {script_name}...")

        result = subprocess.run(command, shell=True)

        if result.returncode != 0:
            logger.warning(
                f"[{i}/{len(commands)}] ⚠️  Warning: Command failed with exit code {result.returncode}"
            )
        else:
            logger.info(f"[{i}/{len(commands)}] ✓ Completed successfully")

    logger.section("LOCAL EXECUTION COMPLETE")


@dataclass
class ReportAndViewConfig:
    create_report: bool | str
    """If True, create a report. If a string, use it as the report title."""


def main(
    experiments: list[str],
    compute_env: ComputeEnvironment,
    max_concurrent_tasks: int | None,
    sweep_params: dict[str, Any] | None,
    report_and_view_config: ReportAndViewConfig | None,
    wandb_project: str,
    job_suffix: str | None,
) -> None:
    run_id: str = generate_run_id()
    logger.info(f"Run ID: {run_id}")

    snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="run")
    logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")

    # set up wandb
    if report_and_view_config:
        report_cfg = None
        if report_and_view_config.create_report is not False:
            report_title = (
                report_and_view_config.create_report
                if isinstance(report_and_view_config.create_report, str)
                else None
            )
            report_cfg = ReportCfg(
                report_title=report_title,
                snapshot_branch=snapshot_branch,
                commit_hash=commit_hash,
                include_run_comparer=True,
            )

        wandb_setup(
            project=wandb_project,
            run_id=run_id,
            experiments=experiments,
            report_cfg=report_cfg,
        )

    commands = _generate_commands(
        run_id=run_id,
        experiments=experiments,
        compute_strategy=compute_env[1],
        sweep_params=sweep_params,
        project=wandb_project,
    )

    match compute_env:
        case Local(), compute_strategy:
            commands = _generate_commands(
                run_id=run_id,
                experiments=experiments,
                compute_strategy=compute_strategy,
                sweep_params=sweep_params,
                project=wandb_project,
            )
            run_commands_locally(commands)

        case SlurmPartition(name=partition_name), compute_strategy:
            # Submit to SLURM
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                array_script_path = temp_path / f"run_array_{run_id}.sh"
                job_name = f"spd-{job_suffix or get_max_expected_runtime(experiments)}"

                array_script_content = create_slurm_array_script(
                    job_name=job_name,
                    commands=commands,
                    snapshot_branch=snapshot_branch,
                    job_strategy=compute_strategy,
                    partition=partition_name,
                    max_concurrent_tasks=max_concurrent_tasks,
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
                        "Total tasks": len(commands),
                        "Max concurrent tasks": max_concurrent_tasks,
                        "View logs in": f"~/slurm_logs/slurm-{array_job_id}_*.out",
                    }
                )


def _build_compute_env(
    cpu: bool,
    local: bool,
    dp_per_node: int | None,
    num_nodes: int | None,
    partition_name: str | None,
) -> ComputeEnvironment:
    """Construct a compute strategy from CLI arguments."""
    if local or cpu:
        assert dp_per_node is None and num_nodes is None and partition_name is None, (
            "cannot specify both local and dp or num_nodes"
        )
        return (Local(), Cpu())

    partition = SlurmPartition(
        name=partition_name if partition_name is not None else DEFAULT_PARTITION_NAME
    )

    match num_nodes, dp_per_node:
        case None, None:
            strategy = SingleNode(n_gpus_per_node=1)
        case None, _:
            assert dp_per_node > 1, (
                "for single-node runs, dp must be at least 2. Otherwise, pass dp=None."
            )
            strategy = SingleNode(n_gpus_per_node=dp_per_node)
        case _, None:
            assert num_nodes > 1, (
                "for multi-node runs, num_nodes must be at least 2. Otherwise, pass num_nodes=None."
            )
            strategy = MultiNode(n_nodes=num_nodes)
        case _, _:
            raise ValueError(
                "Do not specifiy both num_nodes and dp. for multi-node runs, assume 8 GPUs per node."
            )

    return (partition, strategy)


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


def _resolve_sweep_params_path(sweep_params_file: str) -> Path:
    """Resolve the full path to the sweep parameters file."""
    if "/" not in sweep_params_file:
        # Look in scripts directory by default
        return REPO_ROOT / "spd/scripts" / sweep_params_file
    else:
        return REPO_ROOT / sweep_params_file


def _get_sweep_params(sweep: str | bool) -> dict[str, Any] | None:
    sweep_params_file = "sweep_params.yaml" if isinstance(sweep, bool) else sweep
    sweep_params_path = _resolve_sweep_params_path(sweep_params_file)
    with open(sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)
    return sweep_params


def cli(
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
    dp_per_node: int | None = None,
    project: str = "spd",
    local: bool = False,
    log_format: LogFormat = "default",
):
    """SPD runner for experiments with optional parameter sweeps.

    Args:
        experiments: Comma-separated list of experiment names. If None, runs all experiments.
        sweep: Enable parameter sweep. If True, uses default sweep_params.yaml.
            If a string, uses that as the sweep parameters file path.
        n_agents: Maximum number of concurrent SLURM tasks. If None and sweep is enabled,
            raise an error (unless running with --local). If None and sweep is not enabled,
            use the number of experiments. Not used for local execution.
        create_report: Create W&B report for aggregated view (default: True)
        job_suffix: Optional suffix for SLURM job names
        cpu: Use CPU instead of GPU (default: False)
        partition: SLURM partition to use
        num_nodes: Number of nodes for distributed training if desired. If provided, uses torchrun with
            SLURM rendezvous, requires non-local execution. If dp_per_node is provided, this must be None.
        dp_per_node: Number of GPUs per node for data parallelism if desired. Only supported for lm experiments.
            Cannot be used with local mode. If num_nodes is provided, this must be None and we assume 8 GPUs per node.
        project: W&B project name (default: "spd"). Will be created if it doesn't exist.
        local: Run locally instead of submitting to SLURM (default: False)
        log_format: Logging format for the script output.
            Options are "terse" (no timestamps/level) or "default".
        create_snapshot: Create a git snapshot branch for the run.
            if False, uses the current branch, as determined by `repo_current_branch`
            (default: True).
        use_wandb: Use W&B for logging and tracking (default: True).
            If set to false, `create_report` must also be false.
        report_title: Title for the W&B report (default: None). Will be generated if not provided.

    """
    logger.set_format("console", log_format)

    experiments_list: list[str] = _get_experiments(experiments)
    logger.info(f"Experiments: {', '.join(experiments_list)}")

    sweep_params = _get_sweep_params(sweep)

    if local and partition is not None:
        logger.warning("Running locally. setting partition to None.")
        partition = None

    compute_env = _build_compute_env(
        cpu=cpu,
        local=local,
        dp_per_node=dp_per_node,
        num_nodes=num_nodes,
        partition_name=partition,
    )

    # Agent count
    if sweep_params is None:
        if local:
            n_agents = len(experiments_list)
        else:
            assert n_agents is not None, (
                "n-agents must be provided if sweep is enabled (unless running with --local)"
            )

    match use_wandb, create_report, report_title:
        case True, True, title:
            report_and_view_config = ReportAndViewConfig(create_report=title or True)
        case True, False, str(title):
            raise ValueError(
                f"got report_title='{title}' but create_report=False. did you intend to create a report? if so, set create_report=True"
            )
        case True, False, None:
            report_and_view_config = ReportAndViewConfig(create_report=False)
        case False, _, _:
            logger.warning(
                "No workspace views or reports will be created. set `use_wandb=True` and optionally `create_report=True` to enable."
            )
            report_and_view_config = None

    main(
        experiments=experiments_list,
        compute_env=compute_env,
        sweep_params=sweep_params,
        max_concurrent_tasks=n_agents,
        report_and_view_config=report_and_view_config,
        wandb_project=project,
        job_suffix=job_suffix,
    )


if __name__ == "__main__":
    # fire.Fire(cli)
    cli(
        experiments="ss_gpt2_simple",
        num_nodes=2,
    )


