# Agent Swarm Module

This module provides infrastructure for launching parallel SLURM-based Claude Code agents
that investigate behaviors in SPD model decompositions.

## Overview

The agent swarm system allows you to:
1. Launch many parallel agents (each as a SLURM job with 1 GPU)
2. Each agent runs an isolated app backend instance with MCP support
3. Agents investigate behaviors using SPD tools via MCP (Model Context Protocol)
4. Progress is streamed in real-time via MCP SSE events
5. Findings are written to append-only JSONL files

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

## MCP Tools

Agents access ALL SPD functionality via MCP (Model Context Protocol). The backend exposes
these tools at `/mcp`. Agents don't need file system access - everything is done through MCP.

**Analysis Tools:**

| Tool | Description |
|------|-------------|
| `optimize_graph` | Find minimal circuit for a behavior (streams progress) |
| `get_component_info` | Get component interpretation, token stats, correlations |
| `run_ablation` | Test circuit by running with selected components only |
| `search_dataset` | Search SimpleStories training data for patterns |
| `create_prompt` | Tokenize text and get next-token probabilities |

**Output Tools:**

| Tool | Description |
|------|-------------|
| `update_research_log` | Append content to the agent's research log (PRIMARY OUTPUT) |
| `save_explanation` | Save a complete, validated behavior explanation |
| `set_investigation_summary` | Set title and summary shown in the investigations UI |
| `submit_suggestion` | Submit ideas for improving the tools or system |

The `optimize_graph` tool streams progress events via SSE, giving real-time visibility
into long-running optimization operations.

Suggestions from all agents are collected in `SPD_OUT_DIR/agent_swarm/suggestions.jsonl` (global file).

## Output Structure

```
SPD_OUT_DIR/agent_swarm/
├── suggestions.jsonl         # System improvement suggestions from ALL agents (global)
└── <swarm_id>/
    ├── metadata.json         # Swarm configuration
    ├── task_1/
    │   ├── research_log.md   # Human-readable progress log (PRIMARY OUTPUT)
    │   ├── events.jsonl      # Structured progress and observations
    │   ├── explanations.jsonl # Complete behavior explanations
    │   ├── summary.json      # Agent-provided title and summary for UI
    │   ├── app.db            # Isolated SQLite database
    │   ├── agent_prompt.md   # The prompt given to the agent
    │   ├── mcp_config.json   # MCP server configuration for Claude Code
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

# View agent suggestions for system improvement (global file)
cat SPD_OUT_DIR/agent_swarm/suggestions.jsonl | jq .

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
