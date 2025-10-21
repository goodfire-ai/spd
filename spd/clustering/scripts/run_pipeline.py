"""Submit clustering runs to SLURM as separate jobs in a SLURM array.

This script submits independent clustering runs as a SLURM job array,
where each run gets its own dataset (seeded), WandB run, and merge history output.

Also submits a job to calculate distances between the clustering runs, which will run after
the clustering runs (the SLURM job depends on the previous array job).

Output structure (only pipeline_config.json is saved to directly in this script. The files under
<runs> are saved by run_clustering.py which is called in SLURM jobs deployed by this script.):
    <ExecutionStamp.out_dir>/                 # from execution stamp
        |── pipeline_config.json              # Saved in this script
        |── clustering_run_config.json        # make copy of the file pointed to by pipeline config
        ├── ensemble_meta.json                # (Saved by calc_distances.py) Ensemble metadata
        ├── ensemble_merge_array.npz          # (Saved by calc_distances.py) Normalized merge array
        ├── distances_<distances_method>.npz  # (Saved by calc_distances.py) Distance array for each method
        └── distances_<distances_method>.png  # (Saved by calc_distances.py) Distance distribution plot
"""

import argparse
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import wandb_workspaces.workspaces as ws
from pydantic import Field, PositiveInt, field_validator, model_validator

from spd.base_config import BaseConfig
from spd.clustering.consts import DistancesMethod
from spd.clustering.merge_run_config import ClusteringRunConfig
from spd.clustering.storage import StorageBase
from spd.log import logger
from spd.settings import SPD_CACHE_DIR
from spd.utils.command_utils import run_script_array_local
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import _NO_ARG_PARSSED_SENTINEL, ExecutionStamp, read_noneable_str
from spd.utils.slurm_utils import (
    create_slurm_array_script,
    create_slurm_script,
    submit_slurm_script,
)

os.environ["WANDB_QUIET"] = "true"


class ClusteringPipelineStorage(StorageBase):
    """Storage paths for clustering pipeline (ensemble).

    All paths are relative to ExecutionStamp.out_dir.
    """

    # Relative path constants
    _PIPELINE_CONFIG = "pipeline_config.yaml"
    _RUN_IDS = "run_ids.json"
    _ENSEMBLE_META = "ensemble_meta.json"
    _ENSEMBLE_MERGE_ARRAY = "ensemble_merge_array.npz"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.pipeline_config_path: Path = self.base_dir / self._PIPELINE_CONFIG
        self.run_ids_path: Path = self.base_dir / self._RUN_IDS
        self.ensemble_meta_path: Path = self.base_dir / self._ENSEMBLE_META
        self.ensemble_merge_array_path: Path = self.base_dir / self._ENSEMBLE_MERGE_ARRAY

    def distances_path(self, method: DistancesMethod) -> Path:
        return self.base_dir / f"distances_{method}.npz"


class ClusteringPipelineConfig(BaseConfig):
    """Configuration for submitting an ensemble of clustering runs to SLURM."""

    run_clustering_config_path: Path | None = Field(
        default=None,
        description="Path to ClusteringRunConfig file. Mutually exclusive with run_clustering_config.",
    )
    run_clustering_config: ClusteringRunConfig | None = Field(
        default=None,
        description="Inline ClusteringRunConfig. Mutually exclusive with run_clustering_config_path.",
    )
    n_runs: PositiveInt = Field(description="Number of clustering runs in the ensemble")
    distances_methods: list[DistancesMethod] = Field(
        description="List of method(s) to use for calculating distances"
    )
    base_output_dir: Path = Field(description="Base directory for outputs of clustering runs.")
    slurm_job_name_prefix: str | None = Field(description="Prefix for SLURM job names")
    slurm_partition: str | None = Field(description="SLURM partition to use")
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_entity: str = Field(description="WandB entity (team/user) name")
    create_git_snapshot: bool = Field(description="Create a git snapshot for the run")

    @model_validator(mode="after")
    def validate_config_fields(self) -> "ClusteringPipelineConfig":
        """Validate that exactly one of run_clustering_config_path or run_clustering_config is provided."""
        has_path: bool = self.run_clustering_config_path is not None
        has_inline: bool = self.run_clustering_config is not None

        if not has_path and not has_inline:
            raise ValueError(
                "Must specify exactly one of 'run_clustering_config_path' or 'run_clustering_config'"
            )

        if has_path and has_inline:
            raise ValueError(
                "Cannot specify both 'run_clustering_config_path' and 'run_clustering_config'. "
                "Use only one."
            )

        return self

    @field_validator("distances_methods")
    @classmethod
    def validate_distances_methods(cls, v: list[DistancesMethod]) -> list[DistancesMethod]:
        """Validate that distances_methods is non-empty and contains valid methods."""
        assert all(method in DistancesMethod.__args__ for method in v), (
            f"Invalid distances_methods: {v}"
        )

        return v

    def get_config_path(self) -> Path:
        """Get the path to the ClusteringRunConfig file.

        - If run_clustering_config_path is provided, returns it directly.
        - If run_clustering_config is provided, caches it to a deterministic path
        based on its content hash and returns that path.
          - if the config file already exists in the cache, assert that it is identical.

        Returns:
            Path to the (potentially newly created) ClusteringRunConfig file
        """
        if self.run_clustering_config_path is not None:
            return self.run_clustering_config_path

        assert self.run_clustering_config is not None, (
            "Either run_clustering_config_path or run_clustering_config must be set"
        )

        # Generate deterministic hash from config
        hash_b64: str = self.run_clustering_config.stable_hash_b64()

        # Create cache directory
        cache_dir: Path = SPD_CACHE_DIR / "merge_run_configs"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write config to cache if it doesn't exist
        config_path: Path = cache_dir / f"{hash_b64}.json"
        if not config_path.exists():
            self.run_clustering_config.to_file(config_path)
            logger.info(f"Cached inline config to {config_path}")
        else:
            # Verify that existing file matches
            existing_config = ClusteringRunConfig.from_file(config_path)
            if existing_config != self.run_clustering_config:
                raise ValueError(
                    f"Hash collision detected for config hash {hash_b64} at {config_path}\n{existing_config=}\n{self.run_clustering_config=}"
                )

        return config_path


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

    try:
        workspace.save_as_new_view()
        return workspace.url
    except Exception as e:
        logger.warning(
            f"Failed to create WandB workspace view: {workspace=}, {workspace.name=}, {ensemble_id=}, {project=}, {entity=}, {e}"
        )
        raise e


def generate_clustering_commands(
    pipeline_config: ClusteringPipelineConfig,
    pipeline_run_id: str,
    dataset_streaming: bool = False,
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        pipeline_config: Pipeline configuration
        pipeline_run_id: Pipeline run ID (each run will create its own ExecutionStamp)
        dataset_streaming: Whether to use dataset streaming

    Returns:
        List of shell-safe command strings
    """
    commands: list[str] = []

    for idx in range(pipeline_config.n_runs):
        cmd_parts = [
            "python",
            "spd/clustering/scripts/run_clustering.py",
            "--config",
            pipeline_config.get_config_path().as_posix(),
            "--pipeline-run-id",
            pipeline_run_id,
            "--idx-in-ensemble",
            str(idx),
            "--wandb-project",
            str(pipeline_config.wandb_project),
            "--wandb-entity",
            pipeline_config.wandb_entity,
        ]
        if dataset_streaming:
            cmd_parts.append("--dataset-streaming")

        commands.append(shlex.join(cmd_parts))

    return commands


def generate_calc_distances_commands(
    pipeline_run_id: str, distances_methods: list[DistancesMethod]
) -> list[str]:
    """Generate commands for calculating distances.

    Args:
        pipeline_run_id: Pipeline run ID (will query registry for clustering runs)
        distances_methods: List of methods for calculating distances

    Returns:
        List of shell-safe command strings, one per method
    """
    commands: list[str] = []
    for method in distances_methods:
        commands.append(
            shlex.join(
                [
                    "python",
                    "spd/clustering/scripts/calc_distances.py",
                    "--pipeline-run-id",
                    pipeline_run_id,
                    "--distances-method",
                    method,
                ]
            )
        )
    return commands


def main(
    pipeline_config: ClusteringPipelineConfig,
    local: bool = False,
    local_clustering_parallel: bool = False,
    local_calc_distances_parallel: bool = False,
    dataset_streaming: bool = False,
    track_resources_calc_distances: bool = False,
) -> None:
    """Submit clustering runs to SLURM.

    Args:
        pipeline_config_path: Path to ClusteringPipelineConfig file
        n_runs: Number of clustering runs in the ensemble. Will override value in the config file.
    """
    # setup
    # ==========================================================================================

    logger.set_format("console", "terse")

    if local_clustering_parallel or local_calc_distances_parallel or track_resources_calc_distances:
        assert local, (
            "local_clustering_parallel, local_calc_distances_parallel, track_resources_calc_distances "
            "can only be set when running locally\n"
            f"{local_clustering_parallel=}, {local_calc_distances_parallel=}, {track_resources_calc_distances=}, {local=}"
        )

    # Create ExecutionStamp for pipeline
    execution_stamp: ExecutionStamp = ExecutionStamp.create(
        run_type="ensemble",
        create_snapshot=pipeline_config.create_git_snapshot,
    )
    pipeline_run_id: str = execution_stamp.run_id
    logger.info(f"Pipeline run ID: {pipeline_run_id}")

    # Initialize storage
    storage = ClusteringPipelineStorage(execution_stamp)
    logger.info(f"Pipeline output directory: {storage.base_dir}")

    # Save pipeline config
    pipeline_config.to_file(storage.pipeline_config_path)
    logger.info(f"Pipeline config saved to {storage.pipeline_config_path}")

    # Create WandB workspace if requested
    if pipeline_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=pipeline_run_id,
            project=pipeline_config.wandb_project,
            entity=pipeline_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Generate commands for clustering runs
    clustering_commands = generate_clustering_commands(
        pipeline_config=pipeline_config,
        pipeline_run_id=pipeline_run_id,
        dataset_streaming=dataset_streaming,
    )

    # Generate commands for calculating distances
    calc_distances_commands = generate_calc_distances_commands(
        pipeline_run_id=pipeline_run_id,
        distances_methods=pipeline_config.distances_methods,
    )

    # Submit to SLURM
    if local:
        # submit clustering array job
        run_script_array_local(
            commands=clustering_commands,
            parallel=local_clustering_parallel,
        )

        # submit calc_distances jobs in parallel
        logger.info("Calculating distances...")
        run_script_array_local(
            commands=calc_distances_commands,
            parallel=local_calc_distances_parallel,
            track_resources=track_resources_calc_distances,
        )

        logger.section("complete!")

        # Build distances plot paths dict
        distances_plots = {
            f"distances via {method}": str(storage.plots_dir / f"distances_{method}.png")
            for method in pipeline_config.distances_methods
        }

        logger.values(
            {
                "Total clustering runs": len(clustering_commands),
                "Pipeline run ID": pipeline_run_id,
                "Pipeline output dir": str(storage.base_dir),
                **distances_plots,
            }
        )

    else:
        assert pipeline_config.slurm_job_name_prefix is not None, (
            "must specify slurm_job_name_prefix if not running locally"
        )
        assert pipeline_config.slurm_partition is not None, (
            "must specify slurm_partition if not running locally"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            # Submit clustering array job
            clustering_script_path = Path(temp_dir) / f"clustering_{pipeline_run_id}.sh"

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

            # Submit calc_distances jobs (one per method) with dependency on array job
            calc_distances_job_ids: list[str] = []
            calc_distances_logs: list[str] = []

            for _i, (method, cmd) in enumerate(
                zip(pipeline_config.distances_methods, calc_distances_commands, strict=True)
            ):
                calc_distances_script_path = (
                    Path(temp_dir) / f"calc_distances_{method}_{pipeline_run_id}.sh"
                )

                create_slurm_script(
                    script_path=calc_distances_script_path,
                    job_name=f"{pipeline_config.slurm_job_name_prefix}_dist_{method}",
                    command=cmd,
                    snapshot_branch=execution_stamp.snapshot_branch,
                    n_gpus=1,  # Always 1 GPU for distances calculation
                    partition=pipeline_config.slurm_partition,
                    dependency_job_id=array_job_id,
                )
                job_id = submit_slurm_script(calc_distances_script_path)
                calc_distances_job_ids.append(job_id)
                calc_distances_logs.append(f"~/slurm_logs/slurm-{job_id}.out")

            logger.section("Jobs submitted successfully!")

            # Build distances plot paths dict
            distances_plots = {
                method: str(storage.plots_dir / f"distances_{method}.png")
                for method in pipeline_config.distances_methods
            }

            logger.values(
                {
                    "Clustering Array Job ID": array_job_id,
                    "Calc Distances Job IDs": ", ".join(calc_distances_job_ids),
                    "Total clustering runs": len(clustering_commands),
                    "Pipeline run ID": pipeline_run_id,
                    "Pipeline output dir": str(storage.base_dir),
                    "Clustering logs": f"~/slurm_logs/slurm-{array_job_id}_*.out",
                    "Calc Distances logs": ", ".join(calc_distances_logs),
                }
            )
            logger.info("Distances plots will be saved to:")
            for method, path in distances_plots.items():
                logger.info(f"  {method}: {path}")


def cli():
    """CLI for spd-cluster command."""
    parser = argparse.ArgumentParser(
        prog="spd-cluster",
        description="Submit clustering runs to SLURM. Arguments specified here will override the "
        "corresponding value in the config file.",
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        help="Number of clustering runs in the ensemble (overrides value in config file)",
    )
    parser.add_argument(
        "--wandb-project",
        type=read_noneable_str,
        default=_NO_ARG_PARSSED_SENTINEL,
        help="WandB project name (if not provided, WandB logging is disabled)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity name (user or team)",
    )
    parser.add_argument(
        "--distances-methods",
        type=str,
        default=None,
        help="Comma-separated list of distance methods (e.g., 'perm_invariant_hamming,matching_dist')",
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run locally instead of submitting to SLURM (required if slurm_job_name_prefix and slurm_partition are None in config)",
    )
    parser.add_argument(
        "--local-clustering-parallel",
        action="store_true",
        help="If running locally, whether to run clustering runs in parallel",
    )
    parser.add_argument(
        "--local-calc-distances-parallel",
        action="store_true",
        help="If running locally, whether to run distance calculations in parallel",
    )
    parser.add_argument(
        "--track-resources-calc-distances",
        action="store_true",
        help="If running locally, whether to track resource usage during distance calculations",
    )
    parser.add_argument(
        "--dataset-streaming",
        action="store_true",
        help="Whether to use streaming dataset loading (if supported by the dataset). see https://github.com/goodfire-ai/spd/pull/199",
    )

    args = parser.parse_args()

    pipeline_config = ClusteringPipelineConfig.from_file(args.config)
    overrides: dict[str, Any] = {}

    if args.n_runs is not None:
        overrides["n_runs"] = args.n_runs
    if args.wandb_project is not _NO_ARG_PARSSED_SENTINEL:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity
    if args.distances_methods is not None:
        # Parse comma-separated list of distance methods
        methods = [method.strip() for method in args.distances_methods.split(",")]
        overrides["distances_methods"] = methods

    pipeline_config = replace_pydantic_model(pipeline_config, overrides)

    main(
        pipeline_config=pipeline_config,
        local=args.local,
        dataset_streaming=args.dataset_streaming,
        local_clustering_parallel=args.local_clustering_parallel,
        local_calc_distances_parallel=args.local_calc_distances_parallel,
        track_resources_calc_distances=args.track_resources_calc_distances,
    )


if __name__ == "__main__":
    cli()
