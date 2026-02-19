# Topological Interpretation Module

Context-aware component labeling using network graph structure. Unlike standard autointerp (one-shot per component), this module uses dataset attributions to provide neighbor context: each component's prompt includes labels from already-labeled neighbors.

## Usage

```bash
# Via SLURM (standalone)
spd-topological-interp <decomposition_id> --config config.yaml

# Direct execution
python -m spd.topological_interp.scripts.run <decomposition_id> --config_json '{...}'
```

Requires `OPENROUTER_API_KEY` env var. Requires both harvest data and dataset attributions to exist.

## Three-Phase Pipeline

1. **Output pass** (late → early): "What does this component DO?" Each component's prompt includes top-K downstream neighbors (by attribution) with their labels. Late layers labeled first so earlier layers see labeled downstream context.

2. **Input pass** (early → late): "What TRIGGERS this component?" Each component's prompt includes top-K upstream neighbors (by attribution) + co-firing components (Jaccard/PMI). Plus the output label from phase 1. Early layers labeled first so later layers see labeled upstream context.

3. **Unification** (parallel): Synthesizes output + input labels into a single unified label per component.

All three phases run in a single invocation. Resume is per-phase via completed key sets in the DB.

## Data Storage

```
SPD_OUT_DIR/topological_interp/<decomposition_id>/
└── ti-YYYYMMDD_HHMMSS/
    ├── interp.db       # SQLite: output_labels, input_labels, unified_labels, prompt_edges
    └── config.yaml
```

## Database Schema

- `output_labels`: component_key → label, confidence, reasoning, raw_response, prompt
- `input_labels`: same schema as output_labels
- `unified_labels`: same schema as output_labels
- `prompt_edges`: directed filtered graph of (component, neighbor, direction, pass, attribution, neighbor_label)
- `config`: key-value store

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | `TopologicalInterpConfig`, `TopologicalInterpSlurmConfig` |
| `schemas.py` | `LabelResult`, `PromptEdge`, path helpers |
| `db.py` | `TopologicalInterpDB` — SQLite with WAL mode |
| `ordering.py` | Layer parsing and topological sort (matches frontend `graphLayout.ts`) |
| `neighbors.py` | Gather neighbor context from attributions + correlations |
| `prompts.py` | Three prompt formatters (output, input, unification) |
| `interpret.py` | Main three-phase execution loop |
| `repo.py` | `TopologicalInterpRepo` — read-only access to results |
| `scripts/run.py` | CLI entry point (called by SLURM) |
| `scripts/run_slurm.py` | SLURM submission |
| `scripts/run_slurm_cli.py` | Thin CLI wrapper for `spd-topological-interp` |

## Dependencies

- Harvest data (component stats, correlations, token stats)
- Dataset attributions (component-to-component attribution strengths)
- Reuses `map_llm_calls` from `spd/autointerp/llm_api.py`
- Reuses prompt helpers from `spd/autointerp/strategies/dual_view.py`

## SLURM Integration

- 0 GPUs, 16 CPUs, 240GB memory (CPU-only, LLM API calls)
- Depends on both harvest merge AND attribution merge jobs
- Entry point: `spd-topological-interp`
