"""Submit clustering runs to SLURM as separate jobs in a SLURM array.

This script submits independent clustering runs as a SLURM job array,
where each run gets its own dataset (seeded), WandB run, and merge history output.

Also submits a job to calculate distances between the clustering runs, which will run after
the clustering runs (the SLURM job depends on the previous array job).

Output structure (only pipeline_config.json is saved to directly in this script. The files under
<runs> are saved by run_clustering.py which is called in SLURM jobs deployed by this script.):
    <base_output_dir>/ (e.g. SPD_CACHE_DIR / "clustering")
    ├── ensembles/
    │   └── <ensemble_id>/
    │       |── pipeline_config.yaml              # Saved in this script
            ├── ensemble_meta.json                # (Saved by calc_distances.py) Ensemble metadata
            ├── ensemble_merge_array.npz          # (Saved by calc_distances.py) Normalized merge array
            ├── distances_<distances_method>.npz  # (Saved by calc_distances.py) Distance array for each method
            └── distances_<distances_method>.png  # (Saved by calc_distances.py) Distance distribution plot
    └── runs/                                 # (Saved by run_clustering.py)
        └── <ensemble_id>_<idx_in_ensemble>/  # One of these directories for each of the n_runs
        |   ├── clustering_run_config.json
        |   └── history.npz
        └── ...


"""

import argparse
import tempfile
from pathlib import Path

import wandb_workspaces.workspaces as ws
from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig
from spd.clustering.consts import DistancesMethod
from spd.log import logger
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import Command, ExecutionStamp, run_script_array_local
from spd.utils.slurm_utils import (
    create_slurm_array_script,
    create_slurm_script,
    submit_slurm_script,
)


class ClusteringPipelineConfig(BaseConfig):
    """Configuration for submitting an ensemble of clustering runs to SLURM."""

    run_clustering_config_path: Path = Field(description="Path to ClusteringRunConfig file.")
    n_runs: PositiveInt = Field(description="Number of clustering runs in the ensemble")
    distances_method: DistancesMethod = Field(description="Method to use for calculating distances")
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

    TODO: Use a template workspace which actually shows some panels
    TODO: since the run_id here is the same as the wandb id, can we take advantage of that?

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


def generate_clustering_commands(
    pipeline_config: ClusteringPipelineConfig, ensemble_id: str
) -> list[Command]:
    """Generate commands for each clustering run.

    Args:
        pipeline_config: Pipeline configuration
        ensemble_id: Ensemble identifier

    Returns:
        List of Command objects, one per run
    """
    commands: list[Command] = []

    for idx in range(pipeline_config.n_runs):
        cmd = Command(
            cmd=[
                "python",
                "spd/clustering/scripts/run_clustering.py",
                "--config",
                pipeline_config.run_clustering_config_path.as_posix(),
                "--idx-in-ensemble",
                str(idx),
                "--base-output-dir",
                pipeline_config.base_output_dir.as_posix(),
                "--ensemble-id",
                ensemble_id,
            ]
        )
        commands.append(cmd)

    return commands


def generate_calc_distances_command(
    ensemble_id: str, distances_method: DistancesMethod, base_output_dir: Path
) -> Command:
    """Generate command for calculating distances."""
    return Command(
        cmd=[
            "python",
            "spd/clustering/scripts/calc_distances.py",
            "--ensemble-id",
            ensemble_id,
            "--distances-method",
            distances_method,
            "--base-output-dir",
            base_output_dir.as_posix(),
        ]
    )


def main(
    pipeline_config_path: Path,
    n_runs: int | None = None,
    local: bool = False,
) -> None:
    """Submit clustering runs to SLURM.

    Args:
        pipeline_config_path: Path to ClusteringPipelineConfig file
        n_runs: Number of clustering runs in the ensemble. Will override value in the config file.
    """
    # setup
    # ==========================================================================================

    logger.set_format("console", "default")

    pipeline_config = ClusteringPipelineConfig.from_file(pipeline_config_path)

    if n_runs is not None:
        pipeline_config = replace_pydantic_model(pipeline_config, {"n_runs": n_runs})

    # TODO: encapsulate: get run id, branch/snapshot, wandb init? share this with run.py
    # same run_id for wandb, local storage, git snapshot. format `s{hash}` or `c{hash}`
    execution_stamp: ExecutionStamp = ExecutionStamp.create(
        run_type="ensemble",
        create_snapshot=pipeline_config.create_git_snapshot,
    )
    ensemble_id: str = execution_stamp.run_id
    logger.info(f"Ensemble id: {ensemble_id}")

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

    clustering_commands = generate_clustering_commands(
        pipeline_config=pipeline_config, ensemble_id=ensemble_id
    )
    logger.info(f"Generated {len(clustering_commands)} commands")

    calc_distances_command = generate_calc_distances_command(
        ensemble_id=ensemble_id,
        distances_method=pipeline_config.distances_method,
        base_output_dir=pipeline_config.base_output_dir,
    )

    # Submit to SLURM
    if local:
        # submit clustering array job
        run_script_array_local(
            commands=clustering_commands,
        )

        # submit calc_distances job
        logger.info("Calculating distances...")
        logger.info(f"Command: {calc_distances_command.cmd_joined}")
        calc_distances_command.run(check=True)

        logger.section("complete!")
        distances_plot_path = ensemble_dir / f"distances_{pipeline_config.distances_method}.png"
        logger.values(
            {
                "Total clustering runs": len(clustering_commands),
                "Ensemble id": ensemble_id,
                "Distances plot": str(distances_plot_path),
            }
        )

    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Submit clustering array job
            clustering_script_path = Path(temp_dir) / f"clustering_{ensemble_id}.sh"

            create_slurm_array_script(
                script_path=clustering_script_path,
                job_name=f"{pipeline_config.slurm_job_name_prefix}_cluster",
                commands=clustering_commands,
                snapshot_branch=execution_stamp.snapshot_branch,
                max_concurrent_tasks=pipeline_config.n_runs,  # Run all concurrently
                n_gpus_per_job=1,  # Always 1 GPU per run
                partition=pipeline_config.slurm_partition,
            )
            array_job_id = submit_slurm_script(clustering_script_path)

            # Submit calc_distances job with dependency on array job
            calc_distances_script_path = Path(temp_dir) / f"calc_distances_{ensemble_id}.sh"

            create_slurm_script(
                script_path=calc_distances_script_path,
                job_name=f"{pipeline_config.slurm_job_name_prefix}_distances",
                command=calc_distances_command,
                snapshot_branch=execution_stamp.snapshot_branch,
                n_gpus=1,  # Always 1 GPU for distances calculation
                partition=pipeline_config.slurm_partition,
                dependency_job_id=array_job_id,
            )
            calc_distances_job_id = submit_slurm_script(calc_distances_script_path)

            logger.section("Jobs submitted successfully!")
            distances_plot_path = ensemble_dir / f"distances_{pipeline_config.distances_method}.png"
            logger.values(
                {
                    "Clustering Array Job ID": array_job_id,
                    "Calc Distances Job ID": calc_distances_job_id,
                    "Total clustering runs": len(clustering_commands),
                    "Ensemble id": ensemble_id,
                    "Clustering logs": f"~/slurm_logs/slurm-{array_job_id}_*.out",
                    "Calc Distances log": f"~/slurm_logs/slurm-{calc_distances_job_id}.out",
                    "Distances plot will be saved to": str(distances_plot_path),
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

    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run locally instead of submitting to SLURM",
    )

    args = parser.parse_args()
    main(pipeline_config_path=args.config, n_runs=args.n_runs)


if __name__ == "__main__":
    cli()
