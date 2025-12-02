# Local Attributions Module

This module computes and visualizes local attribution graphs - showing how information flows between SPD components across sequence positions for individual prompts.

## Architecture

```
generate.py          → compute.py → db.py → server.py → local_attributions_alpine.html
(multi-GPU pipeline)   (gradients)   (SQLite)  (FastAPI)   (Alpine.js frontend)
```

## Key Files

| File | Purpose |
|------|---------|
| `compute.py` | Core gradient-based attribution computation. Handles wte/output special layers, cross-seq attention (k/v→o_proj). |
| `db.py` | SQLite database with WAL mode. Tables: `meta`, `prompts` (gzipped pairs), `component_activations` (inverted index). |
| `generate.py` | Multi-GPU generation pipeline. Computes activation contexts once, spawns workers. Resumable. |
| `server.py` | FastAPI server. Server-side top-k filtering, edge normalization. |
| `edge_normalization.py` | Normalizes incoming edges to each target node to sum to 1. |
| `local_attributions_alpine.html` | Interactive visualization. Node hover/pin, search, layout modes. |

## Key Data Structures

**Edge format** (server response):
```json
{"src": "layer:seq:cIdx", "tgt": "layer:seq:cIdx", "val": float}
```

**Layers**: `wte` (embeddings), `h.{i}.attn.{q,k,v,o}_proj`, `h.{i}.mlp.{c_fc,down_proj}`, `output` (logits)

**Cross-sequence pairs**: k/v → o_proj within same attention block (captures attention pattern)

## Usage

```bash
# Generate database
python -m spd.attributions.generate \
    --wandb_path wandb:goodfire/spd/runs/<run_id> \
    --n_prompts 1000 --n_gpus 4 --output_path ./local_attr.db

# Serve
python -m spd.attributions.server --db_path ./local_attr.db --port 8765
```

## Frontend Features

- **Filters**: top-k edges, max mean CI, edge normalization toggle
- **Layout modes**: jittered (default), shuffled, by importance
- **Interactions**: hover for details, click to pin, search prompts by pinned components
- **Output nodes**: colored by probability (green gradient)

## TODO

- [x] Bring nodes in front of edges in z-index
- [x] Color output nodes by probability
- [x] Make jitter based on entire-token section width (space-around justification + scaled jitter)
