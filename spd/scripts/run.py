"""Unified SPD runner for experiments with optional parameter sweeps.

This script provides a single entry point for running SPD experiments, supporting both
fixed configurations and parameter sweeps. All runs are tracked in W&B with workspace
views created for each experiment.

Usage:
    spd-run                                                    # Run all experiments
    spd-run --experiments tms_5-2,resid_mlp1                   # Run specific experiments
    spd-run --experiments tms_5-2 --local                      # Run locally instead of SLURM
    spd-run --experiments tms_5-2 --sweep                      # Run with parameter sweep
    spd-run --experiments tms_5-2 --sweep --local              # Run sweep locally
    spd-run --experiments tms_5-2 --sweep custom.yaml          # Run with custom sweep params
    spd-run --sweep --n_agents 10                              # Sweep with 10 concurrent agents
    spd-run --project my-project                               # Use custom W&B project
    spd-run --experiments ss_mlp --dp 4                        # Run with 4 data parallelism over 4 GPUs
"""

import argparse
import copy
import itertools
import json
import shlex
import subprocess
import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
import yaml

from spd.configs import Config
from spd.log import LogFormat, logger
from spd.registry import EXPERIMENT_REGISTRY, get_max_expected_runtime
from spd.settings import REPO_ROOT
from spd.utils.cli_utils import format_function_docstring
from spd.utils.general_utils import apply_nested_updates, load_config
from spd.utils.git_utils import create_git_snapshot, repo_current_branch
from spd.utils.slurm_utils import create_slurm_array_script, submit_slurm_array
from spd.utils.wandb_utils import ensure_project_exists

WORKSPACE_TEMPLATES = {
    "default": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_5-2": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_40-10": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_5-2-id": "https://wandb.ai/goodfire/nathu-spd-test-proj?nw=iytdd13y9d0",
    "tms_40-10-id": "https://wandb.ai/goodfire/nathu-spd-test-proj?nw=iytdd13y9d0",
    "resid_mlp1": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
    "resid_mlp2": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
    "resid_mlp3": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
}


def generate_run_name(
    params: dict[str, Any],
) -> str:
    """Generate a run name based on the present parameters.

    Uses only leaf-node parameters.
    Example:
        >>> params = {"a": {"b": 1}, "c": 2}
        >>> generate_run_name(params)
        "b-1_c-2"
    """
    parts = []
    for k, v in params.items():
        if isinstance(v, dict):
            parts.append(generate_run_name(v))
        else:
            parts.append(f"{k}-{v}")
    return "_".join(parts)


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_sweep_params_path(sweep_params_file: str) -> Path:
    """Resolve the full path to the sweep parameters file."""
    if "/" not in sweep_params_file:
        # Look in scripts directory by default
        return REPO_ROOT / "spd/scripts" / sweep_params_file
    else:
        return REPO_ROOT / sweep_params_file


def generate_grid_combinations(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all combinations for a grid search from parameter specifications."""
    # Flatten nested parameters first
    flattened_params = {}

    def flatten_params(params: dict[str, Any], prefix: str = "") -> None:
        """Recursively flatten nested parameters."""
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if "values" in value:
                    # This is a parameter specification
                    flattened_params[full_key] = value["values"]
                else:
                    # This might be a direct nested structure
                    flatten_params(value, full_key)

    flatten_params(parameters)

    # Extract parameter names and their values
    param_names = list(flattened_params.keys())
    param_values = [flattened_params[name] for name in param_names]

    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        combination = dict(zip(param_names, values, strict=True))
        combinations.append(combination)

    return combinations


def load_sweep_params(experiment_name: str, sweep_params_path: Path) -> dict[str, Any]:
    """Load sweep parameters for an experiment.

    Supports YAML file with global parameters and experiment-specific overrides:

    ```yaml
    global:
      seed:
        values: [0, 1, 2]
      loss:
        faithfulness_weight:
          values: [0.1, 0.5]

    tms_5-2:
      seed:
        values: [100, 200]  # Overrides global seed
      n_components:
        values: [5, 10]     # Adds experiment-specific parameter

    resid_mlp1:
      loss:
        faithfulness_weight:
          values: [1.0, 2.0]  # Overrides nested global parameter
    ```

    Experiment-specific parameters override global parameters at any nesting level.
    """
    with open(sweep_params_path) as f:
        all_params = yaml.safe_load(f)

    # Start with global parameters if they exist
    params = copy.deepcopy(all_params["global"]) if "global" in all_params else {}

    # Merge experiment-specific parameters if they exist
    if experiment_name in all_params and experiment_name != "global":
        experiment_params = all_params[experiment_name]
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


def create_workspace_view(run_id: str, experiment_name: str, project: str = "spd") -> str:
    """Create a wandb workspace view for an experiment."""
    # Use experiment-specific template if available
    template_url = WORKSPACE_TEMPLATES.get(experiment_name, WORKSPACE_TEMPLATES["default"])
    workspace = ws.Workspace.from_url(template_url)

    # Override the project to match what we're actually using
    workspace.project = project

    # Update the workspace name
    workspace.name = f"{experiment_name} - {run_id}"

    # Filter for runs that have BOTH the run_id AND experiment name tags
    # Create filter using the same pattern as in run_grid_search.py
    workspace.runset_settings.filters = [
        ws.Tags("tags").isin([run_id]),
        ws.Tags("tags").isin([experiment_name]),
    ]

    # Save as a new view
    workspace.save_as_new_view()

    return workspace.url


def create_wandb_report(
    report_title: str,
    run_id: str,
    branch_name: str,
    commit_hash: str,
    experiments_list: list[str],
    include_run_comparer: bool,
    project: str = "spd",
    report_total_width: int = 24,
) -> str:
    """Create a W&B report for the run."""
    report = wr.Report(
        project=project,
        title=report_title,
        description=f"Experiments: {', '.join(experiments_list)}",
        width="fluid",
    )

    report.blocks.append(wr.MarkdownBlock(text=f"Branch: `{branch_name}`\nCommit: `{commit_hash}`"))

    # Create separate panel grids for each experiment
    for experiment in experiments_list:
        task_name = EXPERIMENT_REGISTRY[experiment].task_name

        # Use run_id and experiment name tags for filtering
        combined_filter = f'(Tags("tags") in ["{run_id}"]) and (Tags("tags") in ["{experiment}"])'

        # Create runset for this specific experiment
        runset = wr.Runset(
            name=f"{experiment} Runs",
            filters=combined_filter,
        )

        # Build panels list
        panels: list[wr.interface.PanelTypes] = []
        y = 0

        if task_name in ["tms", "resid_mlp"]:
            ci_height = 12
            panels.append(
                wr.MediaBrowser(
                    media_keys=["eval/figures/causal_importances_upper_leaky"],
                    layout=wr.Layout(x=0, y=0, w=report_total_width, h=ci_height),
                    num_columns=6,
                )
            )
            y += ci_height

        loss_plots_height = 6
        loss_plots = [
            ["train/loss/stochastic_recon_layerwise", "train/loss/stochastic_recon"],
            ["train/loss/faithfulness"],
            ["train/loss/importance_minimality"],
        ]
        for i, y_keys in enumerate(loss_plots):
            loss_plots_width = report_total_width // len(loss_plots)
            x_offset = i * loss_plots_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=y_keys,  # pyright: ignore[reportArgumentType]
                    log_y=True,
                    layout=wr.Layout(x=x_offset, y=y, w=loss_plots_width, h=loss_plots_height),
                )
            )
        y += loss_plots_height

        if task_name in ["tms", "resid_mlp"]:
            # Add target CI error plots
            target_ci_weight = 6
            target_ci_width = report_total_width // 2
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["target_solution_error/total"],
                    title="Target CI Error (Tolerance=0.1)",
                    layout=wr.Layout(x=0, y=y, w=target_ci_width, h=target_ci_weight),
                )
            )
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["target_solution_error/total_0p2"],
                    title="Target CI Error (Tolerance=0.2)",
                    layout=wr.Layout(x=target_ci_width, y=y, w=target_ci_width, h=target_ci_weight),
                )
            )
            y += target_ci_weight

        # Only add KL loss plots for language model experiments
        if task_name == "lm":
            kl_height = 6
            kl_width = report_total_width // 3
            x_offset = 0
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/ci_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/unmasked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/stoch_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            y += kl_height

            ce_height = 6
            ce_width = report_total_width // 3
            x_offset = 0
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/ci_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/unmasked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/stoch_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            y += ce_height

        if include_run_comparer:
            run_comparer_height = 10
            panels.append(
                wr.RunComparer(
                    diff_only=True,
                    layout=wr.Layout(x=0, y=y, w=report_total_width, h=run_comparer_height),
                )
            )
            y += run_comparer_height

        panel_grid = wr.PanelGrid(
            runsets=[runset],
            panels=panels,
        )

        # Add title block and panel grid
        report.blocks.append(wr.H2(text=experiment))
        report.blocks.append(panel_grid)

    # Save the report and return URL
    report.save()
    return report.url


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_mpi_prefix(run_id: str, idx: int, dp: int) -> str:
    """Build an MPI prefix for a command."""
    port = _choose_master_port(run_id, idx)
    return f"MASTER_PORT={port} mpirun -x MASTER_PORT -np {dp} "


def generate_commands(
    experiments_list: list[str],
    run_id: str,
    sweep_params_file: str | None = None,
    project: str = "spd",
    dp: int = 1,
) -> list[str]:
    """Generate commands for all experiment runs and print task counts.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """
    commands: list[str] = []

    logger.info("Task breakdown by experiment:")
    task_breakdown: dict[str, str] = {}

    sweep_params_path: Path | None = (
        resolve_sweep_params_path(sweep_params_file) if sweep_params_file else None
    )

    cmd_idx: int = 0

    for experiment in experiments_list:
        config_entry = EXPERIMENT_REGISTRY[experiment]
        decomp_script = REPO_ROOT / config_entry.decomp_script
        config_path = REPO_ROOT / config_entry.config_path

        # Load base config
        base_config = load_config(config_path, Config)

        if sweep_params_path is None:
            # Fixed configuration run - still use JSON to ensure project override works
            base_config_dict = base_config.model_dump(mode="json")
            base_config_dict["wandb_project"] = project
            config_with_overrides = Config(**base_config_dict)

            config_json = f"json:{json.dumps(config_with_overrides.model_dump(mode='json'))}"

            mpi_prefix = _build_mpi_prefix(run_id, cmd_idx, dp) if dp > 1 else ""

            command = (
                f"{mpi_prefix}python {decomp_script} '{config_json}' "
                f"--sweep_id {run_id} --evals_id {experiment}"
            )

            commands.append(command)
            task_breakdown[experiment] = "1 task"
            cmd_idx += 1

        else:
            # Parameter sweep run
            sweep_params = load_sweep_params(experiment, sweep_params_path)
            combinations = generate_grid_combinations(sweep_params)

            for i, param_combo in enumerate(combinations):
                # Apply parameter overrides
                base_config_dict = base_config.model_dump(mode="json")
                config_dict_with_overrides = apply_nested_updates(base_config_dict, param_combo)
                config_dict_with_overrides["wandb_project"] = project
                wandb_run_name = f"{experiment}-{generate_run_name(param_combo)}"
                config_dict_with_overrides["wandb_run_name"] = wandb_run_name
                config_with_overrides = Config(**config_dict_with_overrides)

                config_json = f"json:{json.dumps(config_with_overrides.model_dump(mode='json'))}"
                sweep_params_json = f"json:{json.dumps(sweep_params)}"

                mpi_prefix = _build_mpi_prefix(run_id, cmd_idx, dp) if dp > 1 else ""
                command = (
                    f"{mpi_prefix}python {decomp_script} '{config_json}' "
                    f"--sweep_id {run_id} "
                    f"--evals_id {experiment} "
                    f"--sweep_params_json '{sweep_params_json}'"
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
        script_name = args[1].split("/")[-1]
        logger.section(f"[{i}/{len(commands)}] Executing: {script_name}...")

        result = subprocess.run(args)

        if result.returncode != 0:
            logger.warning(
                f"[{i}/{len(commands)}] ⚠️  Warning: Command failed with exit code {result.returncode}"
            )
        else:
            logger.info(f"[{i}/{len(commands)}] ✓ Completed successfully")

    logger.section("LOCAL EXECUTION COMPLETE")


def get_experiments(
    experiments: str | None = None,
) -> list[str]:
    """Get and validate the list of experiments to run based on the input string.

    Args:
        experiments: Comma-separated list of experiment names. If None, runs all experiments.

    Returns:
        List of experiment names to run.
    """
    # Determine experiment list
    experiments_list: list[str]
    if experiments is None:
        experiments_list = list(EXPERIMENT_REGISTRY.keys())
    else:
        experiments_list = [exp.strip() for exp in experiments.split(",")]

    # Validate experiment names
    invalid_experiments: list[str] = [
        exp for exp in experiments_list if exp not in EXPERIMENT_REGISTRY
    ]
    if invalid_experiments:
        available: str = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Invalid experiments: {invalid_experiments}. Available experiments: {available}"
        )

    return experiments_list


def _wandb_setup(
    project: str,
    run_id: str,
    experiments_list: list[str],
    # only used in report generation
    create_report: bool,
    # passed to create_wandb_report as-is
    report_title: str | None,
    snapshot_branch: str,
    commit_hash: str,
    include_run_comparer: bool,
) -> None:
    """set up wandb, creating workspace views and optionally creating a report

    Args:
        project: W&B project name
        run_id: Unique run identifier
        experiments_list: List of experiment names to create views for
        create_report: Whether to create a W&B report for the run. if False, no report will be created and the rest of the arguments don't matter
        report_title: Title for the W&B report, if created. If None, will be
            generated as "SPD Run Report - {run_id}".
        snapshot_branch: Git branch name for the snapshot created by this run.
        commit_hash: Commit hash of the snapshot created by this run.
        include_run_comparer: Whether to include the run comparer in the report.

    """
    # Ensure the W&B project exists
    ensure_project_exists(project)

    # Create workspace views for each experiment
    logger.section("Creating workspace views...")
    workspace_urls: dict[str, str] = {}
    for experiment in experiments_list:
        workspace_url = create_workspace_view(run_id, experiment, project)
        workspace_urls[experiment] = workspace_url

    # Create report if requested
    report_url: str | None = None
    if create_report and len(experiments_list) > 1:
        report_url = create_wandb_report(
            report_title=report_title or f"SPD Run Report - {run_id}",
            run_id=run_id,
            branch_name=snapshot_branch,
            commit_hash=commit_hash,
            experiments_list=experiments_list,
            include_run_comparer=include_run_comparer,
            project=project,
        )

    # Print clean summary after wandb messages
    logger.values(
        msg="workspace urls per experiment",
        data={
            **workspace_urls,
            **({"Aggregated Report": report_url} if report_url else {}),
        },
    )


def _validate_dp(dp: int, experiments_list: list[str], local: bool, cpu: bool) -> None:
    if dp < 1 or dp > 8:
        raise ValueError(f"dp must be between 1 and 8, got {dp}")

    if dp > 1 and local:
        raise ValueError("DDP (dp > 1) is not supported in local mode")

    if dp > 1:
        non_lm_experiments = [
            exp for exp in experiments_list if EXPERIMENT_REGISTRY[exp].task_name != "lm"
        ]
        if non_lm_experiments:
            raise ValueError(
                f"DDP (dp > 1) is only supported for lm experiments. "
                f"Non-lm experiments found: {non_lm_experiments}"
            )

    if dp > 1 and cpu:
        raise ValueError("Can't have both dp > 1 and cpu")


SPD_RUN_EXAMPLES = """
Examples:
    # Run subset of experiments locally
    spd-run --experiments tms_5-2,resid_mlp1 --local

    # Run parameter sweep locally
    spd-run --experiments tms_5-2 --sweep --local

    # Run subset of experiments (no sweep)
    spd-run --experiments tms_5-2,resid_mlp1

    # Run parameter sweep on a subset of experiments with default sweep_params.yaml
    spd-run --experiments tms_5-2,resid_mlp2 --sweep

    # Run parameter sweep on an experiment with custom sweep params at spd/scripts/my_sweep.yaml
    spd-run --experiments tms_5-2 --sweep my_sweep.yaml

    # Run all experiments (no sweep)
    spd-run

    # Use custom W&B project
    spd-run --experiments tms_5-2 --project my-spd-project

    # Run all experiments on CPU
    spd-run --experiments tms_5-2 --cpu

    # Run with data parallelism over 4 GPUs (only supported for lm experiments)
    spd-run --experiments ss_mlp --dp 4
"""


@format_function_docstring(dict(spd_run_examples=SPD_RUN_EXAMPLES))
def main(
    experiments: str | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    create_report: bool = True,
    job_suffix: str | None = None,
    cpu: bool = False,
    dp: int = 1,
    project: str = "spd",
    local: bool = False,
    log_format: LogFormat = "default",
    create_snapshot: bool = True,
    use_wandb: bool = True,
    report_title: str | None = None,
) -> None:
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
        dp: Number of GPUs for data parallelism (1-8). Only supported for lm experiments.
            Cannot be used with local mode (default: 1)
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

    {spd_run_examples}
    """
    # setup
    # ==========================================================================================

    logger.set_format("console", log_format)

    # Determine run id
    run_id: str = generate_run_id()
    logger.info(f"Run ID: {run_id}")

    # Determine the sweep parameters file
    sweep_params_file: str | None = None
    if sweep:
        sweep_params_file = "sweep_params.yaml" if isinstance(sweep, bool) else sweep

    # get the experiments to run -- run all of them if not specified
    experiments_list: list[str] = get_experiments(experiments)
    logger.info(f"Experiments: {', '.join(experiments_list)}")

    _validate_dp(dp, experiments_list=experiments_list, local=local, cpu=cpu)

    # Agent count
    if n_agents is None:
        if sweep_params_file is None:
            n_agents = len(experiments_list)
        else:
            assert local, (
                "n_agents must be provided if sweep is enabled (unless running with --local)"
            )

    # wandb and snapshot setup
    # ==========================================================================================

    if not local or use_wandb:
        # set up snapshot branch and commit hash
        snapshot_branch: str
        commit_hash: str

        if create_snapshot:
            snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="run")
            logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")
        else:
            snapshot_branch = repo_current_branch()
            commit_hash = "none"
            logger.info(f"Using current branch: {snapshot_branch}")

        # set up wandb
        if use_wandb:
            _wandb_setup(
                project=project,
                run_id=run_id,
                experiments_list=experiments_list,
                create_report=create_report,
                # if `create_report == False`, the rest of the arguments don't matter
                report_title=report_title,
                snapshot_branch=snapshot_branch,
                commit_hash=commit_hash,
                include_run_comparer=sweep_params_file is not None,
            )
        else:
            assert not create_report, (
                f"can't create report if use_wandb is false: {create_report = }"
            )
            logger.warning(
                "W&B logging is disabled. No workspace views or reports will be created. "
                "Set `use_wandb=True` to enable."
            )

    # generate and run commands
    # ==========================================================================================
    commands: list[str] = generate_commands(
        experiments_list=experiments_list,
        run_id=run_id,
        sweep_params_file=sweep_params_file,
        project=project,
        dp=dp,
    )

    if local:
        run_commands_locally(commands)
    else:
        # Submit to SLURM
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            array_script = temp_path / f"run_array_{run_id}.sh"
            if job_suffix is None:
                expected_time_str = get_max_expected_runtime(experiments_list)
                job_name = f"spd-{expected_time_str}"
            else:
                job_name = f"spd-{job_suffix}"

            n_gpus_per_job = dp if not cpu else 0
            create_slurm_array_script(
                script_path=array_script,
                job_name=job_name,
                commands=commands,
                # again -- local is false, so snapshot_branch will exist
                snapshot_branch=snapshot_branch,  # pyright: ignore[reportPossiblyUnboundVariable]
                max_concurrent_tasks=n_agents,
                n_gpus_per_job=n_gpus_per_job,
            )

            array_job_id = submit_slurm_array(array_script)

            logger.section("Job submitted successfully!")
            logger.values(
                {
                    "Array Job ID": array_job_id,
                    "Total tasks": len(commands),
                    "Max concurrent tasks": n_agents,
                    "View logs in": f"~/slurm_logs/slurm-{array_job_id}_*.out",
                }
            )


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        prog="spd-run",
        description="SPD runner for experiments with optional parameter sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SPD_RUN_EXAMPLES,
    )

    # main arguments
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        default=None,
        help=(
            "Comma-separated list of experiment names. If not specified, runs all experiments. "
            f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
        ),
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run locally instead of submitting to SLURM",
    )

    # Sweep arguments
    parser.add_argument(
        "--sweep",
        nargs="?",
        const=True,
        default=False,
        help="Enable parameter sweep. If `--sweep` passed with argument, uses default sweep_params.yaml. "
        "Otherwise, specify a single path to custom sweep parameters file.",
    )

    parser.add_argument(
        "-n",
        "--n-agents",
        type=int,
        default=None,
        help="Maximum number of concurrent SLURM tasks. Required for sweeps unless running locally. "
        "For non-sweep runs, defaults to the number of experiments.",
    )

    # Report and project settings
    parser.add_argument(
        "--create-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create W&B report for aggregated view",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="spd",
        help="W&B project name (default: spd). Will be created if it doesn't exist.",
    )

    parser.add_argument(
        "--report-title",
        type=str,
        default=None,
        help="Title for the W&B report. Generated automatically if not provided.",
    )

    # Execution settings
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU instead of GPU",
    )

    parser.add_argument(
        "--dp",
        "--data-parallelism",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (1-8). Only supported for lm experiments. "
        "Cannot be used with local mode (default: 1)",
    )

    parser.add_argument(
        "--job-suffix",
        type=str,
        default=None,
        help="Optional suffix for SLURM job names",
    )

    # Git and logging settings
    parser.add_argument(
        "--create-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a git snapshot branch for the run",
    )

    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W&B for logging and tracking",
    )

    parser.add_argument(
        "--log-format",
        type=str,
        choices=LogFormat.__args__,
        default="default",
        help="Logging format for script output. 'terse' removes timestamps/level (default: 'default')",
    )

    args = parser.parse_args()

    # Handle the sweep parameter - convert to the expected type
    sweep_value: bool | str = args.sweep
    if sweep_value is True:
        # User provided --sweep with no argument
        sweep_value = True
    elif sweep_value is False:
        # User didn't provide --sweep at all
        sweep_value = False
    else:
        # User provided --sweep with a filename
        sweep_value = str(sweep_value)

    # Call main with parsed arguments
    main(
        experiments=args.experiments,
        sweep=sweep_value,
        n_agents=args.n_agents,
        create_report=args.create_report,
        job_suffix=args.job_suffix,
        cpu=args.cpu,
        dp=args.dp,
        project=args.project,
        local=args.local,
        log_format=args.log_format,
        create_snapshot=args.create_snapshot,
        use_wandb=args.use_wandb,
        report_title=args.report_title,
    )


if __name__ == "__main__":
    cli()
