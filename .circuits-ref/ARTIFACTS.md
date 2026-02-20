# Artifacts

Generated data files and outputs from experiments. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for experiment details.

---

## Edge Statistics (Primary Data)

### FineWeb 50k Baseline (Jan 2026)

Large-scale baseline edge statistics from 39,358 FineWeb-edu graphs.

| File | Description | Size |
|------|-------------|------|
| `data/fineweb_50k_edge_stats.json` | Raw edge statistics from 39k graphs | 372MB |
| `data/fineweb_50k_edge_stats_enriched.json` | **Enriched** with Transluce labels, output/input projections, co-occurrence | **1.19GB** |
| `data/fineweb_50k_cooccurrence.json` | Neuron co-occurrence data (top 20 per neuron) | 228MB |

> ⚠️ **CAUTION**: `fineweb_50k_edge_stats_enriched.json` is 1.19GB. Load carefully to avoid memory issues. Contains 173,923 neuron profiles with Transluce labels (`transluce_label_positive`, `transluce_label_negative`), input/output projections, and co-occurrence data.

**Stats**: 173,923 unique neurons, 56.3M unique edges, 39,358 graphs

**Individual graphs**: `graphs/fabric_fineweb_50k/graph_*.json` (24GB total)

**Projection files**: `outputs/projections/projections_task_*.json` (32 files, 487MB total)

### v6 Medical (Recommended for domain-specific)

| File | Description | Size |
|------|-------------|------|
| `data/medical_edge_stats_v6.json` | Raw edge statistics from 579 medical prompts | ~50MB |
| `data/medical_edge_stats_v6_enriched.json` | **Main file** - enriched with DER, output projections, LLM labels | ~150MB |

### Previous Versions

| File | Description | Notes |
|------|-------------|-------|
| `data/medical_edge_stats_v5_substantive.json` | v5 with substantive prompts only | Superseded by v6 |
| `data/medical_edge_stats_v5_enriched.json` | v5 enriched | Superseded by v6 |
| `data/medical_edge_stats_v4.json` | v4 edge stats | Missing DER |
| `data/medical_edge_stats_v4_enriched.json` | v4 enriched | Missing DER |
| `data/medical_edge_stats_v3_cooccurrence.json` | v3 with co-occurrence | Experimental |
| `data/medical_edge_stats_v2.json` | v2 edge stats | Early version |

---

## Neuron Labels

### Full Function Database (Jan 2026) - RECOMMENDED

Comprehensive neuron labeling from FineWeb 50k edge statistics using GPT-4.1-mini.

| File | Description | Size |
|------|-------------|------|
| `data/neuron_labels_combined.json` | **Recommended** - Combined FineWeb + v6 fallback (48,187 neurons) | ~160MB |
| `data/neuron_function_db_full.json` | FineWeb two-pass labels only (45,035 neurons) | ~150MB |

**Combined file**: `neuron_labels_combined.json` merges FineWeb labels (preferred) with v6 fallback for 3,152 additional neurons not in FineWeb profiles.

**Coverage**: All layers L0-L31 (45,035 neurons from 173,923 profiles)

**Layer Distribution**:
| Layers | Neurons | Notes |
|--------|---------|-------|
| L31 (final) | 3,760 | Output layer - direct logit effects |
| L30-L24 | 10,553 | Late layers - semantic processing |
| L23-L19 | 7,928 | Middle-late layers |
| L18-L14 | 8,265 | Middle layers - routing dominant |
| L13-L9 | 1,512 | Middle-early layers (fewest neurons) |
| L8-L4 | 4,373 | Early layers |
| L3-L0 | 8,644 | Early layers - L0 connects to embeddings |

**Two-Pass Labeling**: Each neuron has both OUTPUT labels (what it does when it fires) and INPUT labels (what triggers it).

**Output Labels (Pass 1)**:
| Metric | Value |
|--------|-------|
| Parsing issues | 0.47% |
| High interpretability | 33.2% |
| Medium interpretability | 41.4% |
| Low interpretability | 25.5% |
| Label uniqueness | 71.9% |

**Input Labels (Pass 2)**:
| Metric | Value |
|--------|-------|
| Parsing issues | 0% |
| High interpretability | 48.3% |
| Medium interpretability | 51.7% |
| Low interpretability | 0% |
| Label uniqueness | 99.7% |

**Key Finding**: 6,604 neurons have LOW output interpretability but HIGH input interpretability - their triggers are more interpretable than their effects.

**Output Function Type Distribution**:
- Routing: 33.8% (works through downstream neurons)
- Lexical: 31.6% (subword/token patterns)
- Semantic: 27.8% (concept/meaning patterns)
- Formatting: 3.2% (punctuation, whitespace)

**Input Trigger Type Distribution**:
- Combination: 79.0% (multiple factors)
- Token-pattern: 15.6% (specific token triggers)
- Context: 3.2% (contextual triggers)

**Source**: `data/fineweb_50k_edge_stats_enriched.json` (39,358 graphs)

### OLMo-3 7B Circuit Atlas (Feb 2026)

#### DuckDB Database

| File | Description | Size |
|------|-------------|------|
| `data/olmo3_neurons.duckdb` | **Primary database** — neurons, edges (weight-based), no clusters | 449MB |

**Tables:**

| Table | Rows | Description |
|-------|------|-------------|
| `neurons` | 352,256 | All MLP neurons (32 layers x 11,008) with enriched labels |
| `edges` | 6,824,960 | Weight-based downstream connections (effective_strength) |
| `clusters` | 0 | No cluster assignments yet |
| `metadata` | 10 | Provenance info |

**Label coverage:** 344,551 neurons (97.8%) have enriched two-pass labels.

**Edge method:** Weight-based wiring (`ConnectivityMethod.WEIGHT_GRAPH`), not RelP. 20 downstream connections per neuron (10 excitatory + 10 inhibitory).

**Build script:** `.venv/bin/python scripts/data/build_olmo3_duckdb.py`

#### Enriched Labels (Source Data)

Full neuron labels for OLMo-3 7B Instruct (`allenai/OLMo-3-7B-Instruct`) generated via two-pass labeling with GPT-5-mini.

| File | Description | Size |
|------|-------------|------|
| `data/olmo3_enriched_labels.json` | **Both passes complete** - 344K neurons with input + output functions | 513MB |
| `data/olmo3_wiring_cache/` | **Weight-based connectivity** - 32 layer files (layer_0.json … layer_31.json). 40 connections per neuron: 10 excitatory/inhibitory × upstream/downstream. SwiGLU polarity analysis. | 3.2GB |

**Format**: JSON dict mapping neuron_id to label data:
```json
{
  "L31/N0": {
    "label": "promotes certain suffix/subword tokens",
    "short_label": "promotes certain suffix/subword tokens",
    "output_function": "When this neuron fires it biases logits...",
    "type": "lexical|semantic|routing|formatting",
    "interpretability": "high|medium|low",
    "confidence": "llm-auto",
    "input_label": "triggered by...",
    "input_function": "This neuron activates when...",
    "input_type": "combination|token-pattern|upstream-gated|context|position",
    "input_interpretability": "high|medium|low",
    "input_confidence": "llm-auto",
    "autointerp_label": "original autointerp label"
  }
}
```

**Coverage**: 344,551 neurons (32 layers × ~11,008 neurons/layer = 352,256 total)

**Two-Pass System**:
- **Pass 1 (Output)**: "What does this neuron DO when it fires?" (L31→L0, complete)
- **Pass 2 (Input)**: "What TRIGGERS this neuron to fire?" (L0→L31, complete)

**Input interpretability**: 59.5% high, 40.5% medium, <0.01% low

**Weight-Based Connectivity**: Uses MLP weight analysis (c_up, c_gate, polarity) instead of RelP aggregation. Each neuron prompt shows 10 excitatory upstream, 10 inhibitory upstream, 10 excitatory downstream, 10 inhibitory downstream connections.

**Autointerp Source**: `/mnt/polished-lake/home/ctigges/experiments/olmo3_neuron_autointerp/labels/layer_{N}/labels.jsonl`

### Previous Label Files

| File | Description |
|------|-------------|
| `data/interactive_labels.json` | Manually verified labels from interactive session |
| `data/neuron_labels_v2.json` | Batch LLM-generated labels |
| `data/neuron_labels_curated.json` | Curated high-quality labels |
| `data/neuron_labels_compositional.json` | Compositional labeling results |
| `data/neuron_function_db_detailed.json` | Earlier function database (subset) |

---

## Neuron Reports (Canonical Location)

**IMPORTANT: All neuron investigation outputs are stored in `neuron_reports/`**

```
neuron_reports/
├── html/           # 162 HTML reports (viewable dashboards)
├── json/           # 151 JSON files (investigation data)
├── index.html      # Browse all reports
└── index.json      # Neuron index data
```

**Browse reports:** Open `neuron_reports/index.html` in a browser.

**File naming:**
- HTML: `L{layer}_N{neuron}.html`
- Investigation: `L{layer}_N{neuron}_investigation.json`
- Dashboard: `L{layer}_N{neuron}_dashboard.json`
- PI Result: `L{layer}_N{neuron}_pi_result.json`

**Archives:** Old/duplicate reports archived in `archives/`

**Consolidation:** Run `scripts/consolidate_neuron_reports.py --execute` to regenerate index.

---

## Prompt Corpora

| File | Description | Count |
|------|-------------|-------|
| `data/medical_corpus_1000.json` | Full medical prompt corpus | 1000 |
| `data/medical_corpus_579_substantive.json` | Filtered substantive prompts | 579 |
| `data/medical_prompts.json` | Original medical prompts | varies |
| `data/twohop_prompts.json` | Two-hop reasoning prompts | varies |
| `data/discovery_set.json` | Discovery set for experiments | varies |
| `data/test_set.json` | Held-out test set | varies |

---

## Graph Outputs

**Location**: `outputs/`

| Directory | Description |
|-----------|-------------|
| `outputs/jobs/` | SLURM batch job outputs |
| `outputs/grant_contrastive/` | Grant approval contrastive experiments |
| `outputs/grant_v2/` | Grant experiments v2 |
| `outputs/poetry/` | Poetry generation experiments |
| `outputs/test-labeling/` | Labeling test outputs |

---

## Configs

**Location**: `configs/`

| File | Description |
|------|-------------|
| `configs/knowledge_circuits.yaml` | 32-prompt knowledge circuit batch |
| `configs/discovery_graphs.yaml` | Discovery set graph generation |
| `configs/landmark_*.yaml` | Landmark experiment configs |
| `configs/sft_*.yaml` | SFT experiment configs |

---

## Qwen 3 32B Circuit Atlas (Feb 2026)

Complete neuron circuit atlas for Qwen/Qwen3-32B (64 layers × 25,600 MLP neurons = 1,638,400 total).

### DuckDB Database — RECOMMENDED

| File | Description | Size |
|------|-------------|------|
| `data/qwen32b_neurons.duckdb` | **Primary database** — neurons, edges, clusters, metadata | 5.4GB |

**Tables:**

| Table | Rows | Description |
|-------|------|-------------|
| `neurons` | 1,638,398 | All MLP neurons with autointerp labels + Infomap cluster assignments |
| `edges` | 83,847,599 | Aggregated edge statistics (count ≥ 3) from 712K RelP graphs |
| `clusters` | 358,831 | Hierarchical Infomap modules (4 levels) |
| `metadata` | 17 | Provenance and config info |

**Neuron columns**: `layer, neuron, feature_id, label, description, max_activation, num_exemplars, cluster_path, top_cluster, sub_module, subsub_module, hierarchy_depth, infomap_flow`

**Edge columns**: `src_layer, src_neuron, tgt_layer, tgt_neuron, count, weight_sum, weight_abs_sum, weight_sq_sum, weight_min, weight_max, mean_weight, mean_abs_weight`

**Cluster columns**: `cluster_path, level, top_cluster, sub_module, subsub_module, size, layer_min, layer_max, median_layer`

**Example queries:**
```python
import duckdb
db = duckdb.connect("data/qwen32b_neurons.duckdb", read_only=True)

# Find all neurons in a circuit
db.sql("SELECT layer, neuron, label FROM neurons WHERE top_cluster = 4466 AND sub_module = 2")

# Search by label
db.sql("SELECT layer, neuron, label, top_cluster FROM neurons WHERE label ILIKE '%aspirin%'")

# Top clusters by size with sample labels
db.sql("""
    SELECT top_cluster, count(*) as n,
           list(label ORDER BY max_activation DESC NULLS LAST LIMIT 3)
    FROM neurons WHERE top_cluster IS NOT NULL
    GROUP BY top_cluster ORDER BY n DESC LIMIT 20
""")

# Edges within a circuit
db.sql("""
    SELECT e.src_layer, e.src_neuron, e.tgt_layer, e.tgt_neuron, e.mean_weight,
           n1.label as src_label, n2.label as tgt_label
    FROM edges e
    JOIN neurons n1 ON e.src_layer = n1.layer AND e.src_neuron = n1.neuron
    JOIN neurons n2 ON e.tgt_layer = n2.layer AND e.tgt_neuron = n2.neuron
    WHERE n1.top_cluster = 158 AND n1.sub_module = 4
      AND n2.top_cluster = 158 AND n2.sub_module = 4
    ORDER BY e.mean_abs_weight DESC
""")
```

### Infomap Cluster Assignments

| File | Description | Size |
|------|-------------|------|
| `data/qwen32b_infomap_assignments.json` | Neuron → cluster path mappings (readable without DuckDB) | 33MB |

**Format:** JSON with `assignments` dict mapping `"L{layer}/{neuron}"` → `{path, top, sub, subsub, depth, flow}`.

**Clustering method:** Infomap multi-level directed, weight = mean_abs_weight², full edges (no top-K truncation), seed 42, min edge count 5.

**Hierarchy stats:**
- 10,200 top-level modules
- 278,044 sub-modules (level 2)
- 70,302 sub-sub-modules (level 3)
- 285 level-4 nodes
- Max cluster: 52,580 neurons (15.5%)
- ~14,900 non-trivial circuits (size ≥ 3)

### Autointerp Labels

| File | Description | Size |
|------|-------------|------|
| `data/neuron_labels_qwen32b/full_run/labels.jsonl` | GPT-5-mini autointerp labels for all 1.64M neurons | 3.6GB |
| `data/neuron_labels_qwen32b/full_run/metadata.json` | Labeling run config and stats | 14KB |

**Format:** JSONL, one line per neuron:
```json
{
  "feature_id": 819222,
  "labels": [{"label": "coins / minting / numismatics", "metadata": {"parsed_response": {"label": "...", "description": "..."}}}],
  "feature_metadata": {"layer": 32, "neuron": 22, "num_exemplars": 20, "max_activation": 65.0}
}
```

**Feature ID mapping:** `feature_id = layer * 25600 + neuron`

**Coverage:** 1,638,398 / 1,638,400 successful (2 errors)

### Edge Aggregation Checkpoints

| File | Description | Size |
|------|-------------|------|
| `graphs/qwen3_32b_800k/checkpoints/latest.dat` | Latest aggregation checkpoint (712K graphs) | 38GB |
| `graphs/qwen3_32b_800k/checkpoints/checkpoint_*.dat` | Historical checkpoints | 25-38GB each |

> ⚠️ **WARNING**: Checkpoints require ~400GB RAM to load (pickle format). Use the DuckDB instead for queries.

**Checkpoint format** (pickle, sequential objects):
```python
with open('graphs/qwen3_32b_800k/checkpoints/latest.dat', 'rb') as f:
    metadata = pickle.load(f)       # {graphs_processed, total_edge_observations, version}
    edges = pickle.load(f)          # Dict[(src_layer, src_neuron, tgt_layer, tgt_neuron) -> [count, sum, abs_sum, sq_sum, min, max]]
    neurons = pickle.load(f)        # Dict[(layer, neuron) -> count]
    processed_files = pickle.load(f)# Set of processed filenames
    neuron_graphs = pickle.load(f)  # Dict[(layer, neuron) -> List[graph_files]]
```

### Previous Qwen Clustering Results

| File | Description | Size |
|------|-------------|------|
| `data/neuron_clusters/qwen32b_infomap_full_edges.json` | Two-level Infomap stats (superseded by multi-level) | 13KB |
| `data/neuron_clusters/qwen32b_edge_profiles.json` | Per-neuron top-10 edge profiles | 934MB |
| `data/neuron_clusters/qwen32b_leiden_clusters.json` | Leiden clustering (had mega-cluster) | - |
| `data/neuron_clusters/qwen32b_recursive_leiden.json` | Recursive Leiden attempt | - |

### Build Script

```bash
# Rebuild the DuckDB from checkpoint + labels (requires ~400GB RAM, SLURM)
sbatch --partition=h200-reserved-default --mem=450G --cpus-per-task=16 --time=4:00:00 \
  --wrap=".venv/bin/python scripts/build_qwen32b_duckdb.py"
```

---

## Llama 3.1 8B Circuit Atlas (Feb 2026)

### DuckDB Database

| File | Description | Size |
|------|-------------|------|
| `data/llama8b_neurons.duckdb` | **Primary database** — neurons, edges, clusters | 175MB |

**Tables:**

| Table | Rows | Description |
|-------|------|-------------|
| `neurons` | 173,923 | Active MLP neurons with labels (GPT function_label preferred, Transluce fallback) |
| `edges` | 1,656,168 | Top-10 downstream edges per neuron from 39K RelP graphs |
| `clusters` | 4,759 | Single-level Infomap cluster assignments |
| `metadata` | 12 | Provenance info |

**Label coverage:** 48,187 neurons (27.7%) have GPT two-pass labels; all 173,923 have Transluce labels.

**Edge note:** Edges are top-10 per neuron (not the full 10.6M edge table). For full edges, re-aggregate from `graphs/fabric_fineweb_50k/`.

**Build script:** `.venv/bin/python scripts/data/build_llama8b_duckdb.py`

### Infomap Full-Edge Clustering

| File | Description | Size |
|------|-------------|------|
| `data/neuron_clusters/llama8b_infomap_full_edges.json` | Multi-level Infomap results | - |
| `data/neuron_clusters/llama8b_full_edges_cache.pkl` | Full edge cache (no top-K truncation) | 2GB |
| `frontend/infomap_full_explorer_data.json` | Constellation visualization data | 26MB |
| `frontend/infomap_constellation.html` | Interactive constellation viewer | - |

**Clustering:** 47,715 labeled neurons, 105 top-level clusters, 2,205 sub-modules. Full edges from 39K graphs.

### Other Methods

| File | Description |
|------|-------------|
| `data/neuron_clusters/leiden_clusters.json` | Leiden community detection (res=0.01, 0.1, 1.0) |
| `data/neuron_clusters/nmf_components.json` | NMF soft clustering |
| `data/neuron_clusters/spectral_clusters.json` | Spectral clustering |
| `data/neuron_clusters/comparison_report.json` | Method comparison |

---

## Adding New Artifacts

When you generate new data files:

1. Add entry to appropriate section above
2. Log the experiment in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
3. Use clear naming: `{domain}_{type}_v{N}.json`

**Naming conventions**:
- Edge stats: `{corpus}_edge_stats_v{N}.json`
- Labels: `{type}_labels_{variant}.json`
- Investigations: `{neuron_id}_investigation.json`, `{neuron_id}_dashboard.json`
