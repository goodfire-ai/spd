# Progressive Interpretation System

A system for interpreting neural network neurons layer-by-layer, building hierarchical functional descriptions where earlier layers are described in terms of their effects on later (already-interpreted) layers.

## Core Insight

```
Late layers (L31) → Direct token effects (easy to interpret)
Earlier layers   → Effects on later neurons (described in terms of those functions)
```

The key insight is that **interpretation should proceed from output to input**, because:
1. Late-layer neurons have direct, measurable effects on output logits
2. Earlier neurons' functions can be described as "promotes/suppresses [late-layer function]"
3. This creates a hierarchical vocabulary for describing neural computation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │  Edge Statistics     │    │  Model Weights       │               │
│  │  (from RelP graphs)  │    │  (Llama-3.1-8B)      │               │
│  │                      │    │                      │               │
│  │  • upstream sources  │    │  • down_proj weights │               │
│  │  • downstream targets│    │  • lm_head weights   │               │
│  │  • edge weights      │    │                      │               │
│  │  • frequencies       │    │                      │               │
│  └──────────┬───────────┘    └──────────┬───────────┘               │
│             │                           │                            │
│             └─────────┬─────────────────┘                            │
│                       ▼                                              │
│           ┌───────────────────────┐                                  │
│           │  Neuron Function DB   │                                  │
│           │  (JSON database)      │                                  │
│           │                       │                                  │
│           │  For each neuron:     │                                  │
│           │  • INPUT: what fires  │                                  │
│           │  • OUTPUT: effects    │                                  │
│           │  • function_label     │                                  │
│           └───────────────────────┘                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure: NeuronFunction

For each neuron, we store:

```
NeuronFunction
├── neuron_id: "L31/N11000"
├── layer: 31
├── neuron_idx: 11000
│
├── function_label: "article→possessive switch"      ← SHORT LABEL
├── function_description: "Promotes possessives..."  ← LONGER DESCRIPTION
├── confidence: "high" | "medium" | "low" | "llm-auto"
│
├── INPUT (what makes this fire)
│   ├── activation_patterns: ["sentence boundaries", ...]
│   └── upstream_neurons: [
│       {
│         neuron_id: "L30/N1234",
│         weight: +0.15,
│         frequency: 0.8,
│         function_label: "clause completion detector"  ← PROPAGATED
│       },
│       ...
│   ]
│
├── OUTPUT (what firing does)
│   ├── direct_logit_effects:
│   │   ├── promotes: [("their", +0.45), ("its", +0.32), ...]
│   │   └── suppresses: [("the", -0.15), ("a", -0.12), ...]
│   │
│   └── downstream_neurons: [
│       {
│         neuron_id: "L32/LOGIT_791",  ← Direct logit connection
│         weight: +0.76,
│         function_label: "output token 'The'"
│       },
│       {
│         neuron_id: "L31/N5369",      ← Neuron connection
│         weight: -0.14,
│         function_label: "numerical data promoter"  ← PROPAGATED
│       },
│       ...
│   ]
│
└── METRICS
    ├── logit_effect_magnitude: 0.45   ← Max abs logit contribution
    ├── downstream_effect_magnitude: 0.76  ← Max abs edge weight
    ├── output_norm: 0.65
    └── effect_type: "logit-dominant" | "routing-dominant" | "mixed"
```

---

## Processing Pipeline

### Step 1: Load Edge Statistics

```
Input:  medical_edge_stats_1000_labeled.json
        (aggregated from 1000 RelP attribution graphs)

For each neuron profile:
  • neuron_id (e.g., "L31/N11000")
  • appearance_count (how many graphs it appears in)
  • top_upstream_sources (what feeds into it)
  • top_downstream_targets (what it feeds into)
  • neurondb_label (max-activation label from NeuronDB)
```

### Step 2: Compute Output Projections

```
For each neuron (layer L, index N):

  output_direction = model.layers[L].mlp.down_proj.weight[:, N]
  logit_contributions = lm_head.weight @ output_direction

  promoted_tokens = top_k(logit_contributions, k=20)
  suppressed_tokens = bottom_k(logit_contributions, k=20)
```

This tells us: **When this neuron fires positively, which tokens increase/decrease in probability?**

### Step 3: Process Layers (Late → Early)

```
┌─────────────────────────────────────────────────────────────────┐
│  PASS 1: Late-to-Early                                          │
│                                                                  │
│  Layer 31 ──► Layer 30 ──► Layer 29 ──► ... ──► Layer 0        │
│                                                                  │
│  At each layer:                                                  │
│  1. For each neuron, gather:                                    │
│     • Output projection (promoted/suppressed tokens)            │
│     • Downstream targets (with function labels if available)    │
│     • Upstream sources (labels not yet available)               │
│                                                                  │
│  2. Generate function label via LLM                             │
│                                                                  │
│  3. Propagate new labels to all connections                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 4: LLM Labeling

For each neuron, we construct a prompt:

```
Context provided to LLM:
─────────────────────────
Neuron: L30/N3382 (Layer 30)
Appears in 1000 prompts
NeuronDB label: "token '�' amidst contextual human or AI discussions"

Direct token effects - PROMOTES: ' (' (+0.031), ' sub' (+0.028), ...
Direct token effects - SUPPRESSES: 'NAMESPACE' (-0.036), ...

Downstream neurons this neuron AFFECTS:
  - suppresses L31/N118 (weight=-0.151) (function: sentence-start capital/space promoter)
  - suppresses L31/N5369 (weight=-0.145) (function: numerical data promoter)
  - promotes L31/N5933 (weight=+0.076) (function: informal tone promoter)

Upstream neurons that ACTIVATE this neuron:
  - inhibited by L1/N2427 (weight=-0.049)
  - inhibited by L1/N198 (weight=-0.009)
─────────────────────────

LLM generates: "suppresses formal tone markers"
```

### Step 5: Label Propagation

After labeling a layer, propagate labels to all connections:

```
For each neuron in database:
  For each downstream connection:
    If target neuron has a label:
      Update connection.function_label = target.function_label

  For each upstream connection:
    If source neuron has a label:
      Update connection.function_label = source.function_label
```

This means when we process Layer 29, its downstream connections to Layer 30/31 neurons will already show their function labels.

### Step 6: Multi-Pass Refinement (Optional)

```
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2: Early-to-Late                                          │
│                                                                  │
│  Layer 0 ──► Layer 1 ──► ... ──► Layer 30 ──► Layer 31         │
│                                                                  │
│  Now we know upstream functions, so we can refine:              │
│  "This neuron is activated by [X function] and promotes         │
│   [Y function]"                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Process Diagram

```
                    ┌─────────────────┐
                    │  Edge Stats     │
                    │  (1000 graphs)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Process Layer  │◄────────────────┐
                    │  (compute       │                 │
                    │   projections)  │                 │
                    └────────┬────────┘                 │
                             │                          │
                             ▼                          │
                    ┌─────────────────┐                 │
                    │  LLM Labeling   │                 │
                    │                 │                 │
                    │  Context:       │                 │
                    │  • tokens       │                 │
                    │  • downstream   │─── labels ──────┤
                    │    (labeled)    │   propagate     │
                    │  • upstream     │                 │
                    │    (unlabeled   │                 │
                    │     on pass 1)  │                 │
                    └────────┬────────┘                 │
                             │                          │
                             ▼                          │
                    ┌─────────────────┐                 │
                    │  Propagate      │                 │
                    │  Labels         │─────────────────┘
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Next Layer     │
                    │  (earlier)      │
                    └─────────────────┘
```

---

## Effect Type Classification

For each neuron, we compute:

```
logit_effect_magnitude = max(abs(logit_contributions))
downstream_effect_magnitude = max(abs(edge_weights to neurons))

ratio = logit_effect / downstream_effect

If ratio > 2:    "logit-dominant"    ← Direct output effect
If ratio < 0.5:  "routing-dominant"  ← Affects other neurons
Else:            "mixed"
```

**Key insight**: As we go earlier in the network:
- `logit_effect_magnitude` tends to decrease
- `downstream_effect_magnitude` becomes more important
- Neurons become more "routing-dominant"

---

## Example: Interpreting L30/N3382

### Raw Data
```
Output projection:
  promotes: ' (', ' sub', ' C', ...
  suppresses: 'NAMESPACE', 'ButtonTitles', ...

Downstream (from edge stats):
  L31/N118:  weight=-0.151
  L31/N5369: weight=-0.145
  L31/N5933: weight=+0.076
```

### After L31 Labeling
```
Downstream (with labels):
  L31/N118:  weight=-0.151  → "sentence-start capital/space promoter"
  L31/N5369: weight=-0.145  → "numerical data promoter"
  L31/N5933: weight=+0.076  → "informal tone promoter"
```

### Generated Function Label
```
"suppresses formal tone markers"
```

### Functional Summary
```
**L30/N3382**
Primary effect: Routing to downstream neurons (mag=1.604)

When this neuron fires:
  - SUPPRESSES 'sentence-start capital/space promoter' (weight=-0.151)
  - SUPPRESSES 'numerical data promoter' (weight=-0.145)
  - PROMOTES 'informal tone promoter' (weight=+0.076)
```

---

## Current Limitations & Future Work

1. **Single-pass upstream labels**: On Pass 1 (late→early), upstream neurons don't have labels yet. Pass 2 (early→late) can refine, but we could iterate more.

2. **Edge statistics are aggregated**: We use statistics across 1000 graphs, which may miss context-specific behaviors.

3. **No activation patterns yet**: We describe OUTPUT effects well, but INPUT conditions (what makes it fire) rely on NeuronDB labels or upstream neurons, not direct measurement.

4. **Confidence calibration**: LLM-generated labels need validation (steering experiments, etc.)

---

## Usage

```bash
# Process neurons from edge stats
.venv/bin/python scripts/progressive_interp.py \
    --edge-stats data/medical_edge_stats_1000_labeled.json \
    --db data/neuron_function_db.json \
    --process-layer 31

# LLM-label a layer
.venv/bin/python scripts/progressive_interp.py \
    --db data/neuron_function_db.json \
    --llm-label-layer 31

# Multi-pass labeling
.venv/bin/python scripts/progressive_interp.py \
    --db data/neuron_function_db.json \
    --llm-label-all --passes 2

# Get functional summary
.venv/bin/python scripts/progressive_interp.py \
    --db data/neuron_function_db.json \
    --summary L30/N3382

# Describe full neuron profile
.venv/bin/python scripts/progressive_interp.py \
    --db data/neuron_function_db.json \
    --describe L30/N3382
```
