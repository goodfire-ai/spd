# Autointerp Module

LLM-based automated interpretation of SPD components. Consumes pre-harvested data from `spd/harvest/` (see `spd/harvest/CLAUDE.md`).

## Usage

```bash
# Run interpretation via SLURM (requires config YAML)
spd-autointerp <decomposition_id> --config autointerp_config.yaml

# Run directly
python -m spd.autointerp.scripts.run_interpret <decomposition_id> --config_json '{"backend": {"type": "anthropic_batch"}, "template_strategy": {"type": "compact_skeptical"}}'
```

Requires `ANTHROPIC_API_KEY` env var (default batch backend) or `OPENROUTER_API_KEY` (OpenRouter backend).

## LLM Backend

Two backends are available, configured via the `backend` field on `AutointerpConfig` and `AutointerpEvalConfig`:

- **`AnthropicBatchConfig`** (default): Anthropic Message Batches API. 50% cheaper than real-time. Submits all jobs as a batch, polls for completion, auto-retries expired items. Uses `tool_use` for structured JSON output. Default model: `claude-sonnet-4-6`.
- **`OpenRouterConfig`**: OpenRouter real-time API with async concurrency, rate limiting, and exponential backoff. Default model: `google/gemini-3-flash-preview`.

```yaml
# Anthropic batch (default)
backend:
  type: anthropic_batch
  model: claude-sonnet-4-6

# OpenRouter real-time
backend:
  type: openrouter
  model: google/gemini-3-flash-preview
  reasoning_effort: low
```

The app's on-demand `interpret_component()` still uses OpenRouter directly (not batched).

## Data Storage

Each autointerp subrun has its own SQLite database:

```
SPD_OUT_DIR/autointerp/<spd_run_id>/
└── <autointerp_run_id>/           # e.g. a-20260206_153040
    ├── interp.db                  # SQLite DB: interpretations + scores (WAL mode)
    └── config.yaml                # AutointerpConfig (for reproducibility)
```

`InterpRepo` reads from the latest subrun (by lexicographic sort of `a-*` dir names).

The `interp.db` schema has three tables:
- `interpretations`: component_key -> label, confidence, reasoning, raw_response, prompt
- `scores`: (component_key, score_type) -> score, details (JSON blob with trial data)
- `config`: key-value store

Score types: `detection`, `fuzzing`.

**Note on intruder scores**: Intruder evaluation lives in `spd/harvest/` (not here) because it tests decomposition quality, not label quality. Intruder scores are stored in `harvest.db`. Detection and fuzzing evaluate interpretation labels and belong here.

## Architecture

### Config (`config.py`)

`AutointerpConfig` has a `backend: LLMBackend` field (discriminated union of `AnthropicBatchConfig | OpenRouterConfig`) and a `template_strategy: StrategyConfig` field (discriminated union of prompt strategies).

Current strategies:
- `CompactSkepticalConfig` — compact prompt, skeptical tone
- `DualViewConfig` — dual-view with input/output sections

Also contains `AutointerpEvalConfig` for eval jobs (detection, fuzzing).

### Batch API (`batch_api.py`)

`run_batch_llm_calls()` — submits jobs to Anthropic Message Batches API:
- Converts `LLMJob` list to batch requests with `tool_use` for structured output
- Splits into sub-batches of 10K
- Polls for completion every 30s
- Auto-retries expired items (configurable `max_retries`)

### Strategies (`strategies/`)

Each strategy config type has a corresponding prompt implementation:
- `strategies/compact_skeptical.py` — prompt formatting for `CompactSkepticalConfig`
- `strategies/dual_view.py` — prompt formatting for `DualViewConfig`
- `strategies/dispatch.py` — routes strategy config → prompt implementation via `match`

### Database (`db.py`)

`InterpDB` class wrapping SQLite for interpretations and scores. Uses WAL mode for concurrent reads. Serialization via `orjson`.

### Repository (`repo.py`)

`InterpRepo` provides read/write access to autointerp data for a run. Lazily opens the SQLite database on first access. Used by the app backend.

### Interpret (`interpret.py`)

- Dispatches on `backend` type: batch (Anthropic) or real-time (OpenRouter)
- Resume support: Skips already-completed components via `db.get_completed_keys()`
- Progress logging via `spd.log.logger`
- `interpret_component()` interprets a single component (app on-demand, OpenRouter only)
- `run_interpret()` orchestrates bulk interpretation

### LLM API (`llm_api.py`)

OpenRouter real-time path: `map_llm_calls()` with async concurrency, rate limiting, exponential backoff, cost tracking. Used by OpenRouter backend and the app's `interpret_component()`.

## Key Types (`schemas.py`)

```python
InterpretationResult  # LLM's label + confidence + reasoning
ModelMetadata         # Model architecture context for prompts
```
