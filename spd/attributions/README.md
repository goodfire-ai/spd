# Local Attributions Database

This module provides infrastructure for building and querying a database of local attribution graphs across many prompts. The goal is to enable queries like "show me all prompts where component X is active" or "find prompts where components A, B, and C all fire together".

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        generate.py                               │
│  Main process:                                                   │
│    1. Load model (populates cache for workers)                  │
│    2. Compute activation_contexts (once, stored in DB)          │
│    3. Build sources_by_target mapping                           │
│    4. Spawn N worker processes                                  │
│                                                                  │
│  Worker processes (one per GPU):                                │
│    - Load model from cache                                      │
│    - Iterate through assigned prompts                           │
│    - Compute local attributions                                 │
│    - Write to shared SQLite DB (WAL mode for concurrency)       │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         db.py                                    │
│  SQLite database with:                                          │
│    - meta table: activation_contexts, wandb_path, n_blocks      │
│    - prompts table: tokens + gzipped attribution pairs          │
│    - component_activations table: inverted index for queries    │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        serve.py                                  │
│  HTTP server exposing:                                          │
│    GET /api/meta                                                │
│    GET /api/activation_contexts                                 │
│    GET /api/prompts                                             │
│    GET /api/prompt/<id>                                         │
│    GET /api/search?components=a,b,c&mode=all|any                │
│    GET /api/components                                          │
│  Also serves static files (the frontend HTML)                   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              local_attributions.html (in spd/scripts/)          │
│  Frontend with:                                                 │
│    - File/Database mode toggle                                  │
│    - Graph visualization of attribution edges                   │
│    - Component pinning (persists across prompt switches)        │
│    - Sidebar search: "Find prompts with pinned components"      │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `db.py` | SQLite database layer with WAL mode for concurrent writes |
| `compute.py` | Core attribution computation functions (copied from scripts to avoid import side effects) |
| `generate.py` | Main entry point for building the database |
| `serve.py` | HTTP server for frontend queries |
| `__init__.py` | Module exports |

## Usage

### 1. Generate the database

```bash
cd /Users/oliverclive-griffin/spd

python -m spd.attributions.generate \
    --wandb_path wandb:goodfire/spd/runs/<run_id> \
    --n_prompts 1000 \
    --n_gpus 8 \
    --output_path ./data/local_attr.db
```

Key flags:
- `--n_prompts`: Total prompts to process
- `--n_gpus`: Number of parallel workers (one per GPU)
- `--n_ctx`: Context length per prompt (default: 64)
- `--n_blocks`: Number of transformer blocks to include (default: 3)
- `--seed`: Random seed for reproducibility

The script is resumable - if interrupted, it will skip already-processed prompts.

### 2. Serve the database

```bash
python -m spd.attributions.serve --db_path ./data/local_attr.db --port 8765
```

Then open `http://localhost:8765/local_attributions.html` and click "Database" mode.

### 3. Query the database programmatically

```python
from spd.attributions import LocalAttrDB

db = LocalAttrDB(Path("./data/local_attr.db"))

# Find prompts where specific components are active
prompt_ids = db.find_prompts_with_components(
    ["h.0.attn.q_proj:5", "h.1.mlp.c_fc:12"],
    require_all=True  # intersection (all must be active)
)

# Get full prompt data
prompt = db.get_prompt(prompt_ids[0])
print(prompt.tokens)
print(prompt.pairs_json)  # JSON string of attribution edges

# Get activation contexts (component metadata)
act_ctx = db.get_activation_contexts()
```

## Database Schema

### `meta` table
Key-value store for model-level data:
- `activation_contexts`: Per-component metadata (mean CI, example tokens, etc.)
- `wandb_path`: Source model path
- `n_blocks`: Number of transformer blocks included

### `prompts` table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| tokens | TEXT | JSON array of token strings |
| pairs | BLOB | Gzipped JSON of attribution edges |

### `component_activations` table (inverted index)
| Column | Type | Description |
|--------|------|-------------|
| prompt_id | INTEGER | FK to prompts.id |
| component_key | TEXT | e.g. "h.0.attn.q_proj:5" |
| max_ci | REAL | Maximum causal importance in this prompt |
| positions | TEXT | JSON array of sequence positions where active |

Indexed on `component_key` for fast lookups.

## Storage Estimates

- ~1-2 MB per prompt (gzipped)
- 10GB budget ≈ 5,000-10,000 prompts
- Activation contexts: ~3MB (stored once, shared across prompts)

## Current Status

**Working:**
- Database generation with multi-GPU parallelism
- WAL mode for concurrent writes (no merge step needed)
- HTTP server with all API endpoints
- Frontend with File/Database mode toggle
- Component pinning and cross-prompt search
- Resume capability for interrupted runs

**Known Issues:**
- None currently blocking

**Potential Improvements:**
- Add progress bar during generation
- Add CLI for common database queries
- Consider pagination for large search results
- Add component statistics endpoint (how many prompts does component X appear in?)

## Development Notes

- The `compute.py` file contains functions copied from `spd/scripts/calc_local_attributions.py` to avoid import side effects (that script has global execution code)
- WAL mode (`PRAGMA journal_mode=WAL`) allows multiple processes to write simultaneously without corruption
- Model is loaded once in main process before spawning workers to populate the download cache and prevent race conditions
