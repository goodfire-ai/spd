# Agent Swarm Module

This module provides infrastructure for launching parallel SLURM-based Claude Code agents
that investigate behaviors in SPD model decompositions.

## Overview

The agent swarm system allows you to:
1. Launch many parallel agents (each as a SLURM job with 1 GPU)
2. Each agent runs an isolated app backend instance
3. Agents investigate behaviors using the SPD app API
4. Findings are written to append-only JSONL files

## Usage

```bash
# Launch 10 agents to investigate a decomposition
spd-swarm goodfire-ai/spd/runs/abc123 --n_agents 10

# With custom settings
spd-swarm goodfire-ai/spd/runs/abc123 --n_agents 5 --context_length 64 --time 4:00:00
```

## Architecture

```
spd/agent_swarm/
├── __init__.py           # Public exports
├── CLAUDE.md             # This file
├── schemas.py            # Pydantic models for outputs
├── agent_prompt.py       # System prompt for agents
└── scripts/
    ├── __init__.py
    ├── run_slurm_cli.py  # CLI entry point (spd-swarm)
    ├── run_slurm.py      # SLURM submission logic
    └── run_agent.py      # Worker script (runs in each SLURM job)
```

## Output Structure

```
SPD_OUT_DIR/agent_swarm/<swarm_id>/
├── metadata.json         # Swarm configuration
├── task_1/
│   ├── research_log.md   # Human-readable progress log (PRIMARY OUTPUT)
│   ├── events.jsonl      # Structured progress and observations
│   ├── explanations.jsonl # Complete behavior explanations
│   ├── app.db            # Isolated SQLite database
│   ├── agent_prompt.md   # The prompt given to the agent
│   └── claude_output.jsonl # Raw Claude Code output (stream-json format)
├── task_2/
│   └── ...
└── task_N/
    └── ...
```

## Key Files

| File | Purpose |
|------|---------|
| `schemas.py` | Defines `BehaviorExplanation`, `SwarmEvent`, `Evidence` schemas |
| `agent_prompt.py` | Contains detailed instructions for agents on using the API |
| `run_slurm.py` | Creates git snapshot, generates commands, submits SLURM array |
| `run_agent.py` | Starts backend, loads run, launches Claude Code |

## Schemas

### BehaviorExplanation
The primary output - documents a discovered behavior:
- `subject_prompt`: Prompt demonstrating the behavior
- `behavior_description`: What the model does
- `components_involved`: List of components and their roles
- `explanation`: How components work together
- `supporting_evidence`: Ablations, attributions, etc.
- `confidence`: high/medium/low
- `alternative_hypotheses`: Other considered explanations
- `limitations`: Known caveats

### SwarmEvent
General logging:
- `event_type`: start, progress, observation, hypothesis, test_result, error, complete
- `timestamp`: When it occurred
- `message`: Human-readable description
- `details`: Structured data

## Database Isolation

Each agent gets its own SQLite database via the `SPD_APP_DB_PATH` environment variable.
This prevents conflicts when multiple agents run on the same machine.

## Monitoring

```bash
# Watch research logs (best way to follow agent progress)
tail -f SPD_OUT_DIR/agent_swarm/<swarm_id>/task_*/research_log.md

# Watch a specific agent's research log
cat SPD_OUT_DIR/agent_swarm/<swarm_id>/task_1/research_log.md

# Watch events from all agents
tail -f SPD_OUT_DIR/agent_swarm/<swarm_id>/task_*/events.jsonl

# View all explanations
cat SPD_OUT_DIR/agent_swarm/<swarm_id>/task_*/explanations.jsonl | jq .

# Check SLURM job status
squeue --me

# View specific job logs
tail -f ~/slurm_logs/slurm-<job_id>_<task_id>.out
```

## Configuration

CLI arguments:
- `wandb_path`: Required - WandB run path for the SPD decomposition
- `--n_agents`: Required - Number of parallel agents to launch
- `--context_length`: Token context length (default: 128)
- `--partition`: SLURM partition (default: h200-reserved)
- `--time`: Time limit per agent (default: 8:00:00)
- `--job_suffix`: Optional suffix for job names

## Extending

To modify agent behavior:
1. Edit `agent_prompt.py` to change investigation instructions
2. Update `schemas.py` to add new output fields
3. Modify `run_agent.py` to change the worker flow

The agent prompt is the primary way to guide agent behavior - it contains
detailed API documentation and scientific methodology guidance.
