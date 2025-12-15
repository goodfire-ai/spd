# Autointerp Pipeline

Automated interpretation of SPD components using Claude API.

## Overview

Two-phase pipeline:
1. **Harvest** (GPU): Single pass over training data collecting correlations, token stats, and activation examples
2. **Interpret** (CPU): Send each component's data to Claude for labeling

Both phases run on cluster. Interpret is IO-bound (API calls), doesn't need GPU.

## Usage

```bash
# Harvest - collect component statistics
python -m spd.autointerp.scripts.run_autointerp harvest <wandb_path> \
    --n_batches 1000 \
    --batch_size 256 \
    --context_length 512

# Interpret - get Claude labels
python -m spd.autointerp.scripts.run_autointerp interpret <wandb_path> \
    --model claude-haiku-4-5-20251001 \
    --max_concurrent 50
```

Requires `ANTHROPIC_API_KEY` env var for interpret phase.

## Data Storage

```
.data/autointerp/<run_id>/
├── harvest/
│   ├── config.json
│   └── components.jsonl      # One ComponentData per line
└── interpretations/
    └── results.jsonl         # One InterpretationResult per line (append-only for resume)
```

## Architecture

### Harvest (`harvest.py`)

`Harvester` class accumulates in a single pass:
- **Correlations**: Co-occurrence counts between components (for precision/recall/PMI)
- **Token stats**: Input token associations (hard counts) and output token associations (probability mass)
- **Activation examples**: Reservoir sampling for uniform coverage across dataset

Key optimizations:
- Reservoir sampling: O(1) per add, O(k) memory, uniform random sampling from stream
- Subsampling: Caps firings per batch at 10k (plenty for k=20 examples per component)
- All accumulation on GPU, only moves to CPU for final `build_results()`

### Interpret (`interpret.py`)

- Uses Anthropic SDK with structured outputs beta for guaranteed JSON schema
- Async with bounded concurrency (`asyncio.Semaphore`)
- Resume support: Skips already-completed components on restart
- Progress bar via `tqdm_asyncio`

### Prompt Template (`prompt_template.py`)

Jinja2 template providing Claude with:
- Architecture context (model class, layer position, dataset)
- Activation examples with CI values
- Token statistics (precision/recall/PMI for input and output tokens)
- Co-occurring components

## Key Types (`schemas.py`)

```python
ComponentData        # All harvested info for one component
ActivationExample    # Token window + CI values around a firing
TokenStats           # Top-k tokens by precision/recall/PMI
ComponentCorrelations # Top-k correlated components
InterpretationResult # Claude's label + confidence + reasoning
```

## Status

Early stage. Primary next steps:
- Eval harness for interpretations (precision/recall via LLM activation simulator)
- Integration with app UI to display labels
