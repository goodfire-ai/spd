"""Submit clustering runs to SLURM.

This script submits independent clustering runs as a SLURM job array,
where each run gets its own dataset (seeded), WandB run, and merge history output.
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig
from spd.clustering.utils.wandb_utils import create_clustering_workspace_view
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


def generate_run_id_for_ensemble(_config: ClusteringPipelineConfig) -> str:
    """Generate a unique ensemble identifier based on timestamp.

    This is used as the ensemble_id component in individual run IDs.
    Format: timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def generate_clustering_commands(
    submit_config: ClusteringPipelineConfig, ensemble_id: str
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        submit_config: Submission configuration
        ensemble_id: Shared hash for this ensemble

    Returns:
        List of commands, one per run
    """
    commands = []

    output_dir = submit_config.base_output_dir / "clustering_runs"

    for idx in range(submit_config.n_runs):
        command = (
            f"python spd/clustering/scripts/run_clustering.py "
            f"--config {submit_config.run_clustering_config_path} "
            f"--idx-in-ensemble {idx} "
            f"--output-dir {output_dir} "
            f"--ensemble-id {ensemble_id}"
        )
        commands.append(command)

    return commands


def main(submit_config_path: Path, n_runs: int | None = None) -> None:
    """Submit clustering runs to SLURM.

    Args:
        submit_config_path: Path to ClusteringSubmitConfig file
        n_runs: Number of clustering runs in the ensemble. Will override value in the config file.
    """
    logger.set_format("console", "default")

    submit_config = ClusteringPipelineConfig.from_file(submit_config_path)

    if n_runs is not None:
        submit_config = replace_pydantic_model(submit_config, {"n_runs": n_runs})

    ensemble_id = generate_run_id_for_ensemble(submit_config)
    logger.info(f"Ensemble id: {ensemble_id}")

    if submit_config.create_git_snapshot:
        snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="cluster")
        logger.info(f"Git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        snapshot_branch = repo_current_branch()
        commit_hash = "none"
        logger.info(f"Using current branch: {snapshot_branch}")

    if submit_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=ensemble_id,
            project=submit_config.wandb_project,
            entity=submit_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Save the submit config for reference
    output_dir = submit_config.base_output_dir / ensemble_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = output_dir / "pipeline_config.json"
    submit_config.to_file(config_save_path)
    logger.info(f"Submit config saved to: {config_save_path}")

    # Generate commands
    commands = generate_clustering_commands(submit_config=submit_config, ensemble_id=ensemble_id)
    logger.info(f"Generated {len(commands)} commands")

    # Submit to SLURM
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / f"clustering_{ensemble_id}.sh"

        create_slurm_array_script(
            script_path=script_path,
            job_name=submit_config.slurm_job_name_prefix,
            commands=commands,
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=submit_config.n_runs,  # Run all concurrently
            n_gpus_per_job=1,  # Always 1 GPU per run
            partition=submit_config.slurm_partition,
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
    main(submit_config_path=args.config, n_runs=args.n_runs)


if __name__ == "__main__":
    cli()
