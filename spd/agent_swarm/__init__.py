"""Agent Swarm: Parallel SLURM-based agent investigation of model behaviors.

This module provides infrastructure for launching many parallel Claude Code agents,
each investigating behaviors in an SPD model decomposition. Each agent:
1. Starts an isolated app backend instance (separate database, unique port)
2. Receives detailed instructions on using the SPD app API
3. Investigates behaviors and writes findings to append-only JSONL files
"""

from spd.agent_swarm.schemas import (
    BehaviorExplanation,
    ComponentInfo,
    Evidence,
    SwarmEvent,
)

__all__ = [
    "BehaviorExplanation",
    "ComponentInfo",
    "Evidence",
    "SwarmEvent",
]
