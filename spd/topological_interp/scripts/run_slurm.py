"""SLURM launcher for topological interpretation.

Submits a single CPU job that runs the three-phase interpretation pipeline.
Depends on both harvest merge and attribution merge jobs.
"""

from dataclasses import dataclass

from spd.log import logger
from spd.topological_interp.config import TopologicalInterpSlurmConfig
from spd.topological_interp.scripts import run
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


@dataclass
class TopologicalInterpSubmitResult:
    result: SubmitResult


def submit_topological_interp(
    decomposition_id: str,
    config: TopologicalInterpSlurmConfig,
    dependency_job_ids: list[str],
    snapshot_branch: str | None = None,
    harvest_subrun_id: str | None = None,
) -> TopologicalInterpSubmitResult:
    """Submit topological interpretation to SLURM.

    Args:
        decomposition_id: ID of the target decomposition.
        config: Topological interp SLURM configuration.
        dependency_job_ids: Jobs to wait for (harvest merge + attribution merge).
        snapshot_branch: Git snapshot branch to use.
        harvest_subrun_id: Specific harvest subrun to use.
    """
    cmd = run.get_command(
        decomposition_id=decomposition_id,
        config=config.config,
        harvest_subrun_id=harvest_subrun_id,
    )

    # Chain dependencies: job starts only after ALL dependencies complete
    dependency_str = ":".join(dependency_job_ids) if dependency_job_ids else None

    slurm_config = SlurmConfig(
        job_name="spd-topological-interp",
        partition=config.partition,
        n_gpus=0,
        cpus_per_task=16,
        mem="240G",
        time=config.time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_str,
        comment=decomposition_id,
    )
    script_content = generate_script(slurm_config, cmd)
    result = submit_slurm_job(script_content, "spd-topological-interp")

    logger.section("Topological interp job submitted")
    logger.values(
        {
            "Job ID": result.job_id,
            "Decomposition ID": decomposition_id,
            "Model": config.config.model,
            "Depends on": ", ".join(dependency_job_ids),
            "Log": result.log_pattern,
        }
    )

    return TopologicalInterpSubmitResult(result=result)
