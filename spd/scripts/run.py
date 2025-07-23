"""Unified SPD runner for experiments with optional parameter sweeps.

This script provides a single entry point for running SPD experiments, supporting both
fixed configurations and parameter sweeps. All runs are tracked in W&B with workspace
views created for each experiment.

Usage:
    spd-run                                                    # Run all experiments
    spd-run --experiments tms_5-2,resid_mlp1                   # Run specific experiments
    spd-run --experiments tms_5-2 --sweep                      # Run with default sweep params
    spd-run --experiments tms_5-2 --sweep custom.yaml          # Run with custom sweep params
    spd-run --sweep --n_agents 10                              # Sweep all experiments
    spd-run --project my-project                               # Use custom W&B project
    spd-run --experiments tms_5-2 --project test-spd           # Run in test project
"""

import copy
import itertools
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
import yaml

from spd.configs import Config
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.utils.general_utils import apply_nested_updates, load_config
from spd.utils.git_utils import create_git_snapshot
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


REPORT_TOTAL_WIDTH = 24


# report generation still pretty basic, needs more work
def create_wandb_report(
    report_title: str,
    run_id: str,
    branch_name: str,
    commit_hash: str,
    experiments_list: list[str],
    include_run_comparer: bool,
    project: str = "spd",
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

        ci_height = 12
        panels.append(
            wr.MediaBrowser(
                media_keys=["causal_importances_upper_leaky"],
                layout=wr.Layout(x=0, y=0, w=REPORT_TOTAL_WIDTH, h=ci_height),
                num_columns=6,
            )
        )
        y += ci_height

        loss_plots_height = 6
        loss_plots = [
            ["loss/stochastic_recon_layerwise", "loss/stochastic_recon"],
            ["loss/faithfulness"],
            ["loss/importance_minimality"],
        ]
        for i, y_keys in enumerate(loss_plots):
            loss_plots_width = REPORT_TOTAL_WIDTH // len(loss_plots)
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

        # Only add KL loss plots for language model experiments
        if EXPERIMENT_REGISTRY[experiment].experiment_type == "lm":
            kl_height = 6
            kl_width = REPORT_TOTAL_WIDTH // 2
            x_offset = 0
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["misc/masked_kl_loss_vs_target"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["misc/unmasked_kl_loss_vs_target"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            y += kl_height

        if include_run_comparer:
            run_comparer_height = 10
            panels.append(
                wr.RunComparer(
                    diff_only=True,
                    layout=wr.Layout(x=0, y=y, w=REPORT_TOTAL_WIDTH, h=run_comparer_height),
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


def generate_commands(
    experiments_list: list[str],
    run_id: str,
    sweep_params_file: str | None = None,
    project: str = "spd",
) -> list[str]:
    """Generate commands for all experiment runs and print task counts.

    NOTE: When we convert parameter settings into JSON strings to pass to our decomposition scripts,
    we add a prefix to prevent Fire parsing with ast.literal_eval
    (https://github.com/google/python-fire/issues/332)
    """
    commands = []

    print("\nTask breakdown by experiment:")

    sweep_params_path = resolve_sweep_params_path(sweep_params_file) if sweep_params_file else None

    for experiment in experiments_list:
        config_entry = EXPERIMENT_REGISTRY[experiment]
        decomp_script = REPO_ROOT / config_entry.decomp_script
        config_path = REPO_ROOT / config_entry.config_path

        # Load base config
        base_config = load_config(config_path, Config)

        if sweep_params_path is None:
            # Fixed configuration run - still use JSON to ensure project override works
            base_config_dict = base_config.model_dump(mode="json")
            # Override the wandb project
            base_config_dict["wandb_project"] = project
            config_with_overrides = Config(**base_config_dict)

            # Convert to JSON string
            config_json = f"json:{json.dumps(config_with_overrides.model_dump(mode='json'))}"

            # Use run_id for sweep_id and experiment name for evals_id
            command = (
                f"python {decomp_script} '{config_json}' "
                f"--sweep_id {run_id} --evals_id {experiment}"
            )
            commands.append(command)
            print(f"  {experiment}: 1 task")

        else:
            # Parameter sweep run
            sweep_params = load_sweep_params(experiment, sweep_params_path)
            combinations = generate_grid_combinations(sweep_params)

            for i, param_combo in enumerate(combinations):
                # Apply parameter overrides
                base_config_dict = base_config.model_dump(mode="json")
                config_dict_with_overrides = apply_nested_updates(base_config_dict, param_combo)
                # Also override the wandb project and run name
                config_dict_with_overrides["wandb_project"] = project
                wandb_run_name = f"{experiment}-{generate_run_name(param_combo)}"
                config_dict_with_overrides["wandb_run_name"] = wandb_run_name
                config_with_overrides = Config(**config_dict_with_overrides)

                # Convert to JSON string
                config_json = f"json:{json.dumps(config_with_overrides.model_dump(mode='json'))}"

                # Create sweep params JSON
                sweep_params_json = f"json:{json.dumps(sweep_params)}"

                # Build command
                command = (
                    f"python {decomp_script} '{config_json}' "
                    f"--sweep_id {run_id} "
                    f"--evals_id {experiment} "
                    f"--sweep_params_json '{sweep_params_json}'"
                )
                commands.append(command)

                # Print first combination as example
                if i == 0:
                    print(f"  {experiment}: {len(combinations)} tasks")
                    print(f"    Example params: {param_combo}")

    return commands


def main(
    experiments: str | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    create_report: bool = True,
    job_suffix: str | None = None,
    cpu: bool = False,
    project: str = "spd",
    report_title: str | None = None,
) -> None:
    """SPD runner for experiments with optional parameter sweeps.

    Args:
        experiments: Comma-separated list of experiment names. If None, runs all experiments.
        sweep: Enable parameter sweep. If True, uses default sweep_params.yaml.
            If a string, uses that as the sweep parameters file path.
        n_agents: Maximum number of concurrent SLURM tasks. If None and sweep is enabled,
            raise an error. If None and sweep is not enabled, use the number of experiments.
        create_report: Create W&B report for aggregated view (default: True)
        job_suffix: Optional suffix for SLURM job names
        cpu: Use CPU instead of GPU (default: False)
        project: W&B project name (default: "spd"). Will be created if it doesn't exist.
        report_title: Title for the W&B report (default: None). Will be generated if not provided.

    Examples:
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
    """
    # Determine the sweep parameters file
    sweep_params_file = None
    if sweep:
        sweep_params_file = "sweep_params.yaml" if isinstance(sweep, bool) else sweep

    # Determine experiment list
    if experiments is None:
        experiments_list = list(EXPERIMENT_REGISTRY.keys())
    else:
        experiments_list = [exp.strip() for exp in experiments.split(",")]

    if n_agents is None:
        if sweep_params_file is None:
            n_agents = len(experiments_list)
        else:
            raise ValueError("n_agents must be provided if sweep is enabled")

    # Validate experiment names
    invalid_experiments = [exp for exp in experiments_list if exp not in EXPERIMENT_REGISTRY]
    if invalid_experiments:
        available = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Invalid experiments: {invalid_experiments}. Available experiments: {available}"
        )

    run_id = generate_run_id()

    print(f"Run ID: {run_id}")
    print(f"Experiments: {', '.join(experiments_list)}")

    commands = generate_commands(
        experiments_list=experiments_list,
        run_id=run_id,
        sweep_params_file=sweep_params_file,
        project=project,
    )

    snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="run")
    print(f"\nUsing git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # Ensure the W&B project exists
    ensure_project_exists(project)

    # Create workspace views for each experiment
    print("\nCreating workspace views...")
    workspace_urls = {}
    for experiment in experiments_list:
        workspace_url = create_workspace_view(run_id, experiment, project)
        workspace_urls[experiment] = workspace_url

    # Create report if requested
    report_url = None
    if create_report and len(experiments_list) > 1:
        report_url = create_wandb_report(
            report_title=report_title or f"SPD Run Report - {run_id}",
            run_id=run_id,
            branch_name=snapshot_branch,
            commit_hash=commit_hash,
            experiments_list=experiments_list,
            include_run_comparer=sweep_params_file is not None,
            project=project,
        )

    # Print clean summary after wandb messages
    print("\n" + "=" * 60)
    print("WORKSPACE VIEWS CREATED:")
    print("=" * 60)
    for experiment, workspace_url in workspace_urls.items():
        print(f"  {experiment}: {workspace_url}")

    if report_url:
        print(f"\n  Aggregated Report: {report_url}")
    print("=" * 60)

    # Determine job name
    job_name = f"spd-{job_suffix}" if job_suffix else "spd"

    # Submit to SLURM
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        array_script = temp_path / f"run_array_{run_id}.sh"

        create_slurm_array_script(
            script_path=array_script,
            job_name=job_name,
            commands=commands,
            cpu=cpu,
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=n_agents,
        )

        array_job_id = submit_slurm_array(array_script)

        print(f"\n{'=' * 50}")
        print("Job submitted successfully!")
        print(f"{'=' * 50}")
        print(f"Array Job ID: {array_job_id}")
        print(f"Total tasks: {len(commands)}")
        print(f"Max concurrent tasks: {n_agents}")
        print(f"View logs in: ~/slurm_logs/slurm-{array_job_id}_*.out")


def cli():
    """Command line interface."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
