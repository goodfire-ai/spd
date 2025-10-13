"""Submit clustering runs to SLURM.

This script submits independent clustering runs as a SLURM job array,
where each run gets its own dataset (seeded), WandB run, and merge history output.
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import wandb_workspaces.workspaces as ws
from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig
from spd.log import logger
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.git_utils import create_git_snapshot, repo_current_branch
from spd.utils.slurm_utils import create_slurm_array_script, submit_slurm_array


class ClusteringPipelineConfig(BaseConfig):
    """Configuration for submitting an ensemble of clustering runs to SLURM.

    FUTURE: Also handle caculating the distances within an ensemble after the runs are complete.
    """

    run_clustering_config_path: Path = Field(description="Path to ClusteringRunConfig file.")
    n_runs: PositiveInt = Field(description="Number of clustering runs in the ensemble")
    base_output_dir: Path = Field(description="Base directory for outputs of clustering runs.")
    slurm_job_name_prefix: str = Field(description="Prefix for SLURM job names")
    slurm_partition: str = Field(description="SLURM partition to use")
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_entity: str = Field(description="WandB entity (team/user) name")
    create_git_snapshot: bool = Field(description="Create a git snapshot for the run")


def create_clustering_workspace_view(ensemble_id: str, project: str, entity: str) -> str:
    """Create WandB workspace view for clustering runs.

    Args:
        ensemble_id: Unique identifier for this ensemble
        project: WandB project name
        entity: WandB entity (team/user) name

    Returns:
        URL to workspace view
    """
    workspace = ws.Workspace(entity=entity, project=project)
    workspace.name = f"Clustering - {ensemble_id}"

    workspace.runset_settings.filters = [
        ws.Tags("tags").isin([f"ensemble_id:{ensemble_id}"]),
    ]

    workspace.save_as_new_view()
    return workspace.url


def generate_run_id_for_ensemble(_config: ClusteringPipelineConfig) -> str:
    """Generate a unique ensemble identifier based on timestamp.

    This is used as the ensemble_id component in individual run IDs.
    Format: timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def generate_clustering_commands(
    pipeline_config: ClusteringPipelineConfig, ensemble_id: str
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        pipeline_config: Pipeline configuration
        ensemble_id: Ensemble identifier

    Returns:
        List of commands, one per run
    """
    commands = []

    for idx in range(pipeline_config.n_runs):
        command = (
            f"python spd/clustering/scripts/run_clustering.py "
            f"--config {pipeline_config.run_clustering_config_path} "
            f"--idx-in-ensemble {idx} "
            f"--base-output-dir {pipeline_config.base_output_dir} "
            f"--ensemble-id {ensemble_id}"
        )
        commands.append(command)

    return commands


def main(pipeline_config_path: Path, n_runs: int | None = None) -> None:
    """Submit clustering runs to SLURM.

    Args:
        pipeline_config_path: Path to ClusteringPipelineConfig file
        n_runs: Number of clustering runs in the ensemble. Will override value in the config file.
    """
    logger.set_format("console", "default")

    pipeline_config = ClusteringPipelineConfig.from_file(pipeline_config_path)

    if n_runs is not None:
        pipeline_config = replace_pydantic_model(pipeline_config, {"n_runs": n_runs})

    ensemble_id = generate_run_id_for_ensemble(pipeline_config)
    logger.info(f"Ensemble id: {ensemble_id}")

    if pipeline_config.create_git_snapshot:
        snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="cluster")
        logger.info(f"Git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        snapshot_branch = repo_current_branch()
        commit_hash = "none"
        logger.info(f"Using current branch: {snapshot_branch}")

    if pipeline_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=ensemble_id,
            project=pipeline_config.wandb_project,
            entity=pipeline_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Save the pipeline config to the output directory
    ensemble_dir = pipeline_config.base_output_dir / "ensembles" / ensemble_id
    pipeline_config.to_file(ensemble_dir / "pipeline_config.yaml")
    logger.info(f"Pipeline config saved to {ensemble_dir / 'pipeline_config.yaml'}")

    commands = generate_clustering_commands(
        pipeline_config=pipeline_config, ensemble_id=ensemble_id
    )
    logger.info(f"Generated {len(commands)} commands")

    # Submit to SLURM
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / f"clustering_{ensemble_id}.sh"

        create_slurm_array_script(
            script_path=script_path,
            job_name=pipeline_config.slurm_job_name_prefix,
            commands=commands,
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=pipeline_config.n_runs,  # Run all concurrently
            n_gpus_per_job=1,  # Always 1 GPU per run
            partition=pipeline_config.slurm_partition,
        )

        array_job_id = submit_slurm_array(script_path)

        logger.section("Job submitted successfully!")
        logger.values(
            {
                "Array Job ID": array_job_id,
                "Total runs": len(commands),
                "Ensemble id": ensemble_id,
                "Logs": f"~/slurm_logs/slurm-{array_job_id}_*.out",
            }
        )


def cli():
    """CLI for spd-cluster command."""
    parser = argparse.ArgumentParser(
        prog="spd-cluster",
        description="Submit clustering runs to SLURM. Arguments specified here will override the "
        "corresponding value in the config file.",
    )

    parser.add_argument(
        "--config",
        default="spd/clustering/configs/pipeline_config.yaml",
        type=Path,
        help="Path to pipeline config file",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        help="Number of clustering runs in the ensemble",
    )

    args = parser.parse_args()
    main(pipeline_config_path=args.config, n_runs=args.n_runs)


if __name__ == "__main__":
    cli()
