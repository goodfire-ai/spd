"""CLI entry point for agent swarm SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-swarm <wandb_path> --n_agents 10
    spd-swarm <wandb_path> --n_agents 5 --context_length 128

Examples:
    # Launch 10 agents to investigate a decomposition
    spd-swarm goodfire-ai/spd/runs/abc123 --n_agents 10

    # Launch 5 agents with custom context length and time limit
    spd-swarm goodfire-ai/spd/runs/abc123 --n_agents 5 --context_length 64 --time 4:00:00
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    n_agents: int,
    context_length: int = 128,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "8:00:00",
    job_suffix: str | None = None,
) -> None:
    """Launch a swarm of agents to investigate behaviors in an SPD model.

    Each agent runs in its own SLURM job with an isolated app backend instance.
    Agents use Claude Code to investigate behaviors and write findings to
    append-only JSONL files.

    Args:
        wandb_path: WandB run path for the SPD decomposition to investigate.
            Format: "entity/project/runs/run_id" or "wandb:entity/project/run_id"
        n_agents: Number of agents to launch (each gets 1 GPU).
        context_length: Context length for prompts (default 128).
        partition: SLURM partition name.
        time: Job time limit per agent (default 8 hours).
        job_suffix: Optional suffix for SLURM job names.
    """
    from spd.agent_swarm.scripts.run_slurm import launch_agent_swarm

    launch_agent_swarm(
        wandb_path=wandb_path,
        n_agents=n_agents,
        context_length=context_length,
        partition=partition,
        time=time,
        job_suffix=job_suffix,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
