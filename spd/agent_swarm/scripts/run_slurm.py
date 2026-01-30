"""SLURM launcher for agent swarm.

Submits a SLURM array job where each task runs an independent agent investigating
behaviors in an SPD model decomposition.

Each agent:
1. Starts an isolated app backend (unique port, isolated database)
2. Launches Claude Code with investigation instructions
3. Writes findings to append-only JSONL files
"""

import secrets
from pathlib import Path

from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SlurmArrayConfig,
    generate_array_script,
    submit_slurm_job,
)


def get_swarm_output_dir(swarm_id: str) -> Path:
    """Get the output directory for a swarm run."""
    return SPD_OUT_DIR / "agent_swarm" / swarm_id


def launch_agent_swarm(
    wandb_path: str,
    n_agents: int,
    context_length: int = 128,
    partition: str = "h200-reserved",
    time: str = "8:00:00",
    job_suffix: str | None = None,
) -> None:
    """Launch a swarm of agents to investigate behaviors.

    Args:
        wandb_path: WandB run path for the SPD decomposition.
        n_agents: Number of agents to launch.
        context_length: Context length for prompts.
        partition: SLURM partition.
        time: Time limit per agent.
        job_suffix: Optional suffix for job names.
    """
    swarm_id = f"swarm-{secrets.token_hex(4)}"
    output_dir = get_swarm_output_dir(swarm_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_branch, commit_hash = create_git_snapshot(swarm_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    suffix = f"-{job_suffix}" if job_suffix else ""
    job_name = f"spd-swarm{suffix}"

    # Write swarm metadata
    metadata_path = output_dir / "metadata.json"
    import json

    metadata = {
        "swarm_id": swarm_id,
        "wandb_path": wandb_path,
        "n_agents": n_agents,
        "context_length": context_length,
        "snapshot_branch": snapshot_branch,
        "commit_hash": commit_hash,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Build worker commands (SLURM arrays are 1-indexed)
    worker_commands = []
    for task_id in range(1, n_agents + 1):
        cmd = (
            f"python -m spd.agent_swarm.scripts.run_agent "
            f'"{wandb_path}" '
            f"--task_id {task_id} "
            f"--swarm_id {swarm_id} "
            f"--context_length {context_length}"
        )
        worker_commands.append(cmd)

    array_config = SlurmArrayConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=1,
        time=time,
        snapshot_branch=snapshot_branch,
        max_concurrent_tasks=min(n_agents, 8),  # Respect cluster limits
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script,
        "agent_swarm",
        is_array=True,
        n_array_tasks=n_agents,
    )

    logger.section("Agent swarm jobs submitted!")
    logger.values(
        {
            "Swarm ID": swarm_id,
            "WandB path": wandb_path,
            "N agents": n_agents,
            "Context length": context_length,
            "Output directory": str(output_dir),
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Job ID": array_result.job_id,
            "Logs": array_result.log_pattern,
            "Script": str(array_result.script_path),
        }
    )
    logger.info("")
    logger.info("Monitor progress:")
    logger.info(f"  tail -f {output_dir}/task_*/events.jsonl")
    logger.info("")
    logger.info("View explanations:")
    logger.info(f"  cat {output_dir}/task_*/explanations.jsonl | jq .")
