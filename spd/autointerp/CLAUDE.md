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

All autointerp data (interpretations and scores) is stored in a single SQLite database:

```
SPD_OUT_DIR/autointerp/<spd_run_id>/
├── interp.db                      # SQLite DB: interpretations + scores (WAL mode)
└── <autointerp_run_id>/           # e.g. a-20260206_153040
    └── config.yaml                # AutointerpConfig (for reproducibility)
```

The `interp.db` schema has three tables:
- `interpretations`: component_key -> label, confidence, reasoning, raw_response, prompt
- `scores`: (component_key, score_type) -> score, details (JSON blob with trial data)
- `config`: key-value store

Score types: `intruder`, `detection`, `fuzzing`.

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

### Database (`db.py`)

`InterpDB` class wrapping SQLite for interpretations and scores. Uses WAL mode for concurrent reads. Serialization via `orjson`.

### Repository (`repo.py`)

`InterpRepo` provides read/write access to autointerp data for a run. Lazily opens the SQLite database on first access. Used by the app backend.

### Interpret (`interpret.py`)

- Uses OpenRouter API with structured JSON outputs
- Maximum parallelism with exponential backoff on rate limits
- Resume support: Skips already-completed components via `db.get_completed_keys()`
- Progress logging via `spd.log.logger`
- `interpret_component()` interprets a single component
- `run_interpret()` orchestrates batch interpretation with resume support

### Loaders (`loaders.py`)

Standalone loader functions used by scoring scripts:
- `load_interpretations(run_id)` — loads all interpretations from DB
- `load_intruder_scores(run_id)` — loads intruder scores from DB
- `load_detection_scores(run_id)` — loads detection scores from DB
- `load_fuzzing_scores(run_id)` — loads fuzzing scores from DB

## Key Types (`schemas.py`)

```python
InterpretationResult  # LLM's label + confidence + reasoning
ArchitectureInfo      # Model architecture context for prompts
```
