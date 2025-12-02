# Local Attributions Module

This module computes and visualizes local attribution graphs - showing how information flows between SPD components across sequence positions for individual prompts.

## Architecture

```
Generation (fast):     generate.py → compute_ci_only() → db.py (token_ids + inverted index)
Serving (on-demand):   server.py → compute_local_attributions() → local_attributions_alpine.html
```

**Key insight**: CI computation is cheap and batchable (~200k tokens/forward pass). Attribution graph computation requires O(seq × n_active_components) backward passes - slow for batch generation but fast for single prompts. So we compute CI at generation time (for the inverted index) and graphs on-demand at serve time.

## Key Files

| File | Purpose |
|------|---------|
| `compute.py` | `compute_ci_only()` for fast CI, `compute_local_attributions()` for full graphs. Handles wte/output special layers, cross-seq attention (k/v→o_proj). |
| `db.py` | SQLite database. Tables: `meta`, `prompts` (token_ids), `component_activations` (inverted index). |
| `generate.py` | Multi-GPU CI generation. Stores token_ids + inverted index only (no graphs). |
| `server.py` | FastAPI server. Loads model at startup, computes graphs on-demand. Requires GPU. |
| `edge_normalization.py` | Normalizes incoming edges to each target node to sum to 1. |
| `local_attributions_alpine.html` | Interactive visualization. Node hover/pin, search, layout modes. |

## Key Data Structures

**DB Schema (simplified)**:
- `prompts`: `id`, `token_ids` (JSON array of ints)
- `component_activations`: `prompt_id`, `component_key`, `max_ci`, `positions`
- `meta`: `wandb_path`, `n_blocks`, `activation_contexts`

**Edge format** (server response):
```json
{"src": "layer:seq:cIdx", "tgt": "layer:seq:cIdx", "val": float}
```

**Layers**: `wte` (embeddings), `h.{i}.attn.{q,k,v,o}_proj`, `h.{i}.mlp.{c_fc,down_proj}`, `output` (logits)

## Usage

```bash
# Generate database (fast - CI only)
python -m spd.attributions.generate \
    --wandb_path wandb:goodfire/spd/runs/<run_id> \
    --n_prompts 1000 --n_gpus 4 --output_path ./local_attr.db

# Serve (requires GPU for on-demand graph computation)
python -m spd.attributions.server --db_path ./local_attr.db --port 8765
```

## Frontend Features

- **Filters**: top-k edges, max mean CI, edge normalization toggle
- **Layout modes**: jittered (default), shuffled, by importance
- **Interactions**: hover for details, click to pin, search prompts by pinned components
- **Output nodes**: colored by probability (green gradient)

## TODO

- [ ] Slim down activation_contexts - frontend only uses `mean_ci` and first 5 `example_tokens`, but we store much more (`example_ci`, `example_active_pos`, `pr_tokens`, etc.)
- [ ] Verify mean_ci computation is principled (check `get_activations_data_streaming`)
