# Autointerp Module

LLM-based automated interpretation of SPD components. Consumes pre-harvested data from `spd/harvest/` (see `spd/harvest/CLAUDE.md`).

## Usage

```bash
# Run interpretation (requires harvest data to exist first)
python -m spd.autointerp.scripts.run_interpret <wandb_path> --model google/gemini-3-flash-preview

# Or via SLURM
spd-autointerp <wandb_path>
```

Requires `OPENROUTER_API_KEY` env var.

## Data Storage

Data is stored in `SPD_OUT_DIR/autointerp/` (see `spd/settings.py`):

```
SPD_OUT_DIR/autointerp/<run_id>/
└── results.jsonl    # One InterpretationResult per line (append-only for resume)
```

## Architecture

### Interpret (`interpret.py`)

- Uses OpenRouter API with structured JSON outputs
- Maximum parallelism with exponential backoff on rate limits
- Resume support: Skips already-completed components on restart
- Progress bar via `tqdm_asyncio`

### Prompt Template (`prompt_template.py`)

Jinja2 template providing the LLM with:
- Architecture context (model class, layer position, dataset)
- Activation examples with CI values
- Token statistics (PMI for input and output tokens)
- Co-occurring components

## Key Types (`schemas.py`)

```python
InterpretationResult  # LLM's label + confidence + reasoning
```

## Status

Early stage. Primary next steps:
- Eval harness for interpretations (precision/recall via LLM activation simulator)
- Integration with app UI to display labels
