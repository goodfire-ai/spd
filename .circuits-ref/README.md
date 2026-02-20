# Circuits

Generate and analyze neural network attribution graphs using RelP (Relevance Propagation). Supports Llama-3.1-8B-Instruct and Qwen/Qwen3-32B.

The library provides a modular pipeline for understanding neural network internals: compute attribution graphs, aggregate connectivity across many prompts, label neurons via autointerp, cluster into circuits, and store everything in a DuckDB atlas.

## Installation

```bash
uv venv && source .venv/bin/activate
uv sync

# Register the CLI
pip install -e .
```

## CLI Quick Reference

```bash
circuits graph "The capital of France is"              # One-off RelP graph
circuits analyze --config configs/knowledge_circuits.yaml  # Full analysis pipeline
circuits aggregate graphs/qwen3_32b_800k/              # Aggregate edges from many graphs
circuits label --config label_config.yaml              # Two-pass autointerp labeling
circuits cluster --duckdb data/atlas.duckdb            # Cluster neurons into circuits
circuits build-db --config build_config.yaml           # Build DuckDB atlas
circuits query data/atlas.duckdb "enzyme"              # Search the atlas
```

## Core Concepts

### Attribution Graphs

A single RelP graph traces information flow from output logits back through the network for one prompt. Each graph contains **units** (MLP neurons that fire) and **edges** (attribution weights between them).

```bash
# Generate a graph
circuits graph "The enzyme inhibited by aspirin is" \
    --target-tokens COX cyclooxygenase --slug aspirin-cox

# Output: graphs/relp-aspirin-cox.json
```

### Circuit Atlas (DuckDB)

Aggregate many graphs to build a complete atlas of a model's circuits:

```bash
# 1. Generate many graphs (via SLURM workers)
# 2. Aggregate edge statistics
circuits aggregate graphs/qwen3_32b_800k/ --resume

# 3. Build the atlas database
circuits build-db --config atlas_config.yaml

# 4. Query it
circuits query data/atlas.duckdb "dopamine"
```

The atlas DuckDB contains:
- **neurons** table: layer, neuron, label, cluster assignment, activation stats
- **edges** table: aggregated connection weights between neurons
- **clusters** table: Infomap circuit decomposition hierarchy
- **metadata** table: model info, build parameters

### Autointerp Labeling

Two-pass progressive labeling (separate from the main pipeline):

```bash
circuits label --config label_config.yaml
```

- **Pass 1 (Output):** Late → Early layers. "What does this neuron DO when it fires?"
- **Pass 2 (Input):** Early → Late layers. "What TRIGGERS this neuron to fire?"

## Python API

```python
from circuits import Unit, Edge, Graph, RelPAttributor, RelPConfig

# Common schemas used across all stages
unit = Unit(layer=24, index=5326, label="capital cities")
edge = Edge(src_layer=20, src_index=455, tgt_layer=24, tgt_index=5326, weight=3.14)
graph = Graph(units=[unit], edges=[edge], prompt="What is the capital of France?")

# Convert from/to legacy Neuronpedia graph format
graph = Graph.from_legacy(json.load(open("graph.json")))
legacy_dict = graph.to_legacy()

# Aggregation
from circuits.aggregation import InMemoryAggregator
agg = InMemoryAggregator(Path("graphs/"))
agg.process_directory(Path("graphs/"))
edges = agg.get_edges(min_count=5)

# DuckDB atlas
from circuits.database import CircuitDatabase
db = CircuitDatabase(Path("data/atlas.duckdb"), read_only=True)
units = db.search_units("enzyme")
edges = db.get_edges_for_unit(layer=24, index=5326)

# Clustering
from circuits import cluster_full_model
assignments = cluster_full_model(edges, units)
```

## Pipeline Stages

### 1. Graph Generation

RelP (Relevance Propagation) traces which neurons contribute to specific output predictions:

```bash
circuits graph "Your prompt" [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--k INT` | `5` | Number of top logits to trace |
| `--tau FLOAT` | `0.005` | Node influence threshold (lower = more nodes) |
| `--target-tokens TOKEN [...]` | - | Specific tokens to trace |
| `--contrastive POS NEG` | - | Trace logit(POS) - logit(NEG) |
| `--raw` | - | Skip chat template |
| `--model MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model to use |

### 2. Full Analysis Pipeline

Runs graph generation → labeling → clustering → LLM analysis for one or more prompts:

```bash
circuits analyze "The neurotransmitter associated with reward is"
circuits analyze --config configs/knowledge_circuits.yaml
circuits analyze --graph graphs/existing.json
```

### 3. Edge Aggregation

Aggregate edge statistics from many RelP graphs:

```bash
circuits aggregate graphs/qwen3_32b_800k/ \
    --checkpoint-interval 500 \
    --resume
```

### 4. Clustering

Cluster neurons into circuits using Infomap community detection:

```bash
# Cluster a single graph
circuits cluster --graph graphs/aspirin-cox.json

# Cluster full model from DuckDB
circuits cluster --duckdb data/atlas.duckdb
```

### 5. Autointerp Labeling

Two-pass neuron interpretation (separate subcommand):

```bash
circuits label --config label_config.yaml
```

### 6. Build DuckDB Atlas

Combine aggregated edges, labels, and clustering into a queryable database:

```bash
circuits build-db --config atlas_config.yaml
```

## Config File Format

```yaml
model:
  name: "Qwen/Qwen3-32B"
  device: "cuda"
  dtype: "bfloat16"

graphs:
  directory: "graphs/qwen3_32b_800k/"

aggregation:
  method: "relp"              # or "output_projection"
  checkpoint_dir: "checkpoints/"
  min_edge_count: 3

clustering:
  algorithm: "infomap"
  min_edge_count: 5
  weight_transform: "abs_weight_sq"

labeling:
  method: "progressive"       # or "goodfire" or "neurondb"
  llm_model: "gpt-5.2-mini"

output:
  duckdb: "data/atlas.duckdb"
```

## Supported Models

| Model | Layers | Neurons/Layer | Status |
|-------|--------|---------------|--------|
| Llama 3.1 8B Instruct | 32 | 14,336 | Full support |
| Qwen/Qwen3-32B | 64 | 27,648 | Full support |
| OLMo 3 7B Instruct | 32 | 11,008 | Experimental |

## Project Structure

```
circuits/
├── circuits/                  # Core library
│   ├── schemas.py             # Unit, Edge, Graph data types
│   ├── relp.py                # RelP attribution
│   ├── batch_relp.py          # Optimized RelP (default)
│   ├── connectivity.py        # Edge computation (projections, weight graphs)
│   ├── aggregation.py         # Edge aggregation from many graphs
│   ├── clustering.py          # Infomap clustering
│   ├── labeling.py            # Neuron database label fetching
│   ├── autointerp.py          # Progressive two-pass labeling
│   ├── database.py            # DuckDB atlas builder/reader
│   ├── analysis.py            # LLM-based circuit analysis
│   ├── model_configs.py       # Model architecture definitions
│   ├── pipeline.py            # Pipeline orchestration
│   ├── hooks.py               # Activation/gradient caching
│   └── cli.py                 # Unified CLI
├── neuron_scientist/          # Neuron investigation agent
├── scripts/
│   ├── pipeline/              # Core: generate, analyze, diff, patch
│   ├── investigation/         # Neuron PI/scientist, HTML reports
│   ├── data/                  # Aggregation, indexing, consolidation
│   ├── labeling/              # Autointerp labeling scripts
│   ├── experiments/           # Research experiments
│   └── slurm/                 # SLURM shell scripts
├── slurm/                     # Distributed processing framework
├── frontend/                  # Visualization tools
├── configs/                   # Pipeline config examples
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## Testing

```bash
# Run all tests (no GPU needed for schema/aggregation/database tests)
pytest tests/ -x

# Run just the schema and pipeline tests
pytest tests/test_circuits_schemas.py tests/test_relp_pipeline.py -v
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for Claude models |
| `OPENAI_API_KEY` | API key for GPT models |
| `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DATABASE` | PostgreSQL neurondb connection |
