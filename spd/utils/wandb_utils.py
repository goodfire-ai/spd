import os
from pathlib import Path
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv
from wandb.apis.public import File, Run

from spd.log import logger
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.utils.general_utils import BaseModel, _fetch_latest_checkpoint_name, replace_pydantic_model
from spd.utils.run_utils import METRIC_CONFIG_SHORT_NAMES

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


def _extract_metric_info(config_item: dict[str, Any], is_loss: bool) -> tuple[str, dict[str, Any]]:
    """Extract classname and params from a metric config item.

    Args:
        config_item: Either a loss metric config {"coeff": X, "metric": {...}}
                     or an eval metric config {"classname": Y, ...}
        is_loss: True if this is from loss_metric_configs, False for eval_metric_configs

    Returns:
        Tuple of (classname, params_dict) where params_dict contains all non-classname fields
    """
    if is_loss:
        assert "metric" in config_item, "loss_metric_configs items should have 'metric'"
        assert "coeff" in config_item, "loss_metric_configs items should have 'coeff'"
        metric_dict = config_item["metric"]
        classname = metric_dict["classname"]
        # Include both coeff and metric params
        params = {"coeff": config_item["coeff"]}
        params.update({k: v for k, v in metric_dict.items() if k != "classname"})
    else:
        classname = config_item["classname"]
        assert isinstance(classname, str), "eval_metric_configs should have a classname"
        params = {k: v for k, v in config_item.items() if k != "classname"}

    return classname, params


def flatten_metric_configs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten loss_metric_configs and eval_metric_configs into dot-notation for wandb searchability.

    Converts:
        loss_metric_configs: [{"coeff": 0.1, "metric": {"classname": "ImportanceMinimalityLoss", "pnorm": 1.0}}]
    To:
        loss.ImportanceMinimalityLoss.coeff: 0.1
        loss.ImportanceMinimalityLoss.pnorm: 1.0
    """
    flattened: dict[str, Any] = {}

    for field_name, prefix in [("loss_metric_configs", "loss"), ("eval_metric_configs", "eval")]:
        if field_name not in config_dict:
            continue

        configs = config_dict[field_name]
        assert isinstance(configs, list), f"{field_name} should be a list"

        is_loss = field_name == "loss_metric_configs"
        for config_item in configs:
            assert isinstance(config_item, dict), f"{field_name} should have dicts"

            classname, params = _extract_metric_info(config_item, is_loss)
            short_name = METRIC_CONFIG_SHORT_NAMES[classname]

            for key, value in params.items():
                flattened[f"{prefix}.{short_name}.{key}"] = value

    return flattened


def fetch_latest_wandb_checkpoint(run: Run, prefix: str | None = None) -> File:
    """Fetch the latest checkpoint from a wandb run."""
    filenames = [file.name for file in run.files() if file.name.endswith((".pth", ".pt"))]
    latest_checkpoint_name = _fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_remote = run.file(latest_checkpoint_name)
    return latest_checkpoint_remote


def fetch_wandb_run_dir(run_id: str) -> Path:
    """Find or create a directory in the W&B cache for a given run.

    We first check if we already have a directory with the suffix "run_id" (if we created the run
    ourselves, a directory of the name "run-<timestamp>-<run_id>" should exist). If not, we create a
    new wandb_run_dir.
    """
    # Default to REPO_ROOT/wandb
    base_cache_dir = REPO_ROOT / "wandb"
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    # Set default wandb_run_dir
    wandb_run_dir = base_cache_dir / run_id / "files"

    # Check if we already have a directory with the suffix "run_id"
    presaved_run_dirs = [
        d for d in base_cache_dir.iterdir() if d.is_dir() and d.name.endswith(run_id)
    ]
    # If there is more than one dir, just ignore the presaved dirs and use the new wandb_run_dir
    if presaved_run_dirs and len(presaved_run_dirs) == 1:
        presaved_file_path = presaved_run_dirs[0] / "files"
        if presaved_file_path.exists():
            # Found a cached run directory, use it
            wandb_run_dir = presaved_file_path

    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    return wandb_run_dir


def download_wandb_file(run: Run, wandb_run_dir: Path, file_name: str) -> Path:
    """Download a file from W&B. Don't overwrite the file if it already exists.

    Args:
        run: The W&B run to download from
        file_name: Name of the file to download
        wandb_run_dir: The directory to download the file to
    Returns:
        Path to the downloaded file
    """
    file_on_wandb = run.file(file_name)
    assert isinstance(file_on_wandb, File)
    path = Path(file_on_wandb.download(exist_ok=True, replace=False, root=str(wandb_run_dir)).name)
    return path


def init_wandb[T_config: BaseModel](
    config: T_config, project: str, name: str | None = None, tags: list[str] | None = None
) -> T_config:
    """Initialize Weights & Biases and return a config updated with sweep hyperparameters.

    Args:
        config: The base config.
        project: The name of the wandb project.
        name: The name of the wandb run.
        tags: Optional list of tags to add to the run.

    Returns:
        Config updated with sweep hyperparameters (if any).
    """
    load_dotenv(override=True)

    wandb.init(
        project=project,
        entity=os.getenv("WANDB_ENTITY"),
        name=name,
        tags=tags,
    )
    assert wandb.run is not None
    wandb.run.log_code(
        root=str(REPO_ROOT / "spd"), exclude_fn=lambda path: "out" in Path(path).parts
    )

    # Update the config with the hyperparameters for this sweep (if any)
    config = replace_pydantic_model(config, wandb.config.as_dict())

    config_dict = config.model_dump(mode="json")
    # We also want flattened names for easier wandb searchability
    flattened_config_dict = flatten_metric_configs(config_dict)
    # Remove the nested metric configs to avoid duplication
    del config_dict["loss_metric_configs"]
    del config_dict["eval_metric_configs"]
    wandb.config.update({**config_dict, **flattened_config_dict})
    return config


def ensure_project_exists(project: str) -> None:
    """Ensure the W&B project exists by creating a dummy run if needed."""
    api = wandb.Api()

    # Check if project exists in the list of projects
    if project not in [p.name for p in api.projects()]:
        # Project doesn't exist, create it with a dummy run
        logger.info(f"Creating W&B project '{project}'...")
        run = wandb.init(project=project, name="project_init", tags=["init"])
        run.finish()
        logger.info(f"Project '{project}' created successfully")


def create_workspace_view(run_id: str, experiment_name: str, project: str = "spd") -> str:
    """Create a wandb workspace view for an experiment."""
    # Use experiment-specific template if available
    template_url: str = WORKSPACE_TEMPLATES.get(experiment_name, WORKSPACE_TEMPLATES["default"])
    workspace: ws.Workspace = ws.Workspace.from_url(template_url)

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
        task_name: str = EXPERIMENT_REGISTRY[experiment].task_name

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
            ["eval/loss/faithfulness"],
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


def wandb_setup(
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
