# Autointerp Module

LLM-based automated interpretation of SPD components. Consumes pre-harvested data from `spd/harvest/` (see `spd/harvest/CLAUDE.md`).

## Usage

```bash
# Run interpretation with a config file
python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_path path/to/config.yaml

# Run with inline config JSON
python -m spd.autointerp.scripts.run_interpret <wandb_path> --config_json '{"model": "google/gemini-3-flash-preview", "reasoning_effort": null}'

# Or via SLURM
spd-autointerp <wandb_path>
spd-autointerp <wandb_path> --model google/gemini-3-flash-preview --reasoning_effort medium
```

Requires `OPENROUTER_API_KEY` env var.

## Data Storage

Autointerp runs are versioned with timestamped subdirectories:

```
SPD_OUT_DIR/autointerp/<spd_run_id>/
├── eval/                          # Label-independent eval (intruder detection)
│   └── intruder/...
├── <autointerp_run_id>/           # e.g. 20260206_153040
│   ├── config.yaml                # AutointerpConfig (for reproducibility)
│   ├── results.jsonl              # InterpretationResults (append-only for resume)
│   └── scoring/                   # Label-dependent scoring
│       ├── detection/...
│       └── fuzzing/...
└── <another_autointerp_run_id>/
    └── ...
```

Legacy flat format (`results_*.jsonl` directly in the run dir) is still readable via fallback in `loaders.py`.

## Architecture

### Config (`config.py`)

`AutointerpConfig` is a discriminated union over interpretation strategy configs. Each variant specifies everything that affects interpretation output (model, prompt params, reasoning effort). Admin/execution params (cost limits, parallelism) are NOT part of the config.

Current strategies:
- `CompactSkepticalConfig` — compact prompt, skeptical tone, structured JSON output

Also contains `AutointerpEvalConfig` for eval jobs (detection, fuzzing).

### Strategies (`strategies/`)

Each strategy config type has a corresponding prompt implementation:
- `strategies/compact_skeptical.py` — prompt formatting for `CompactSkepticalConfig`
- `strategies/dispatch.py` — routes `AutointerpConfig` → strategy implementation via `match`

### Interpret (`interpret.py`)

- Uses OpenRouter API with structured JSON outputs
- Maximum parallelism with exponential backoff on rate limits
- Resume support: Skips already-completed components on restart
- Progress logging via `spd.log.logger`
- `interpret_component()` and `interpret_all()` accept `AutointerpConfig`

### Loaders (`loaders.py`)

- `load_interpretations(run_id, autointerp_run_id=None)` — loads from specific or latest run
- `find_latest_results_path(run_id)` — finds latest results file (nested then flat fallback)

## Key Types (`schemas.py`)

```python
InterpretationResult  # LLM's label + confidence + reasoning
ArchitectureInfo      # Model architecture context for prompts
```
