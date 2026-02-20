# Automated Neuron Interpretation Pipeline

This document describes the end-to-end process for automatically labeling neurons in Llama-3.1-8B-Instruct with both **input functions** (what triggers the neuron) and **output functions** (what the neuron does when it fires).

## Overview

The pipeline has three main phases:

1. **Data Collection**: Generate attribution graphs for many prompts, aggregate edge statistics
2. **Output Labeling** (Pass 1): Label what each neuron does when it fires (late-to-early, L31→L0)
3. **Input Labeling** (Pass 2): Label what triggers each neuron (early-to-late, L0→L31)

The result is a database of ~8,000 neurons with detailed functional descriptions from both perspectives.

---

## Phase 1: Data Collection

### 1.1 Generate Attribution Graphs

For each prompt, we generate a RelP (Relevance Propagation) attribution graph that traces which neurons contributed to the model's output.

```bash
# Single prompt
.venv/bin/python scripts/generate_graph.py "The neurotransmitter associated with reward is"

# Batch from config
.venv/bin/python scripts/analyze.py --config configs/medical_prompts.yaml
```

**What RelP computes:**
- For each neuron: `relp_score = activation × gradient` (importance for output)
- For neuron→neuron edges: `edge_weight = source.activation × ∂target/∂source` (Jacobian-based)
- For embedding→neuron edges: Currently uses `target.relp_score` as a shortcut (see limitations)

### 1.2 Aggregate Edge Statistics

Run many prompts and aggregate statistics about each neuron's behavior:

```bash
.venv/bin/python scripts/aggregate_edge_stats.py \
    --graphs outputs/*.json \
    --output data/edge_stats.json \
    --min-appearances 50
```

This produces a JSON file with profiles for each neuron containing:
- `top_upstream_sources`: Which neurons/embeddings feed into this neuron
- `top_downstream_targets`: Which neurons/logits this neuron affects
- `output_token_associations`: Tokens present when this neuron fires
- `appearance_count`: How often the neuron appears across prompts
- `domain_specificity`: How domain-specific vs. general the neuron is

### 1.3 Compute Static Projections

**Output Projection** (what tokens the neuron promotes/suppresses):
```
output_projection = neuron_output_vector @ unembedding_matrix
```

**Input Projection** (what tokens would activate/suppress the neuron):
```
input_projection = SiLU(embedding @ gate_proj) × (embedding @ up_proj)
```

The input projection accounts for Llama's gated MLP architecture where the actual neuron response is `SiLU(gate) × up`, not just `up`.

```bash
# Precompute input projections for all neurons
.venv/bin/python scripts/compute_input_projections.py \
    --edge-stats data/edge_stats.json
```

This adds `input_projection` field to each neuron profile with top activating/suppressing tokens.

### 1.4 Compute Direct Effect Ratio

For each neuron, compute what fraction of its effect is direct (on logits) vs. indirect (through downstream neurons):

```bash
.venv/bin/python scripts/compute_der_batch.py \
    --edge-stats data/edge_stats.json
```

This helps distinguish:
- **Logit neurons** (high DER): Directly affect output vocabulary
- **Routing neurons** (low DER): Work through downstream neurons

---

## Phase 2: Output Labeling (Pass 1)

**Goal**: Determine what each neuron does when it fires.

**Direction**: Late-to-early (L31→L0) so we can use downstream neuron labels to understand routing neurons.

### What the LLM sees:

For each neuron, we build a prompt containing:

1. **Transluce labels** (if available): Prior labels from neuronpedia
2. **Output projection**: Top tokens promoted/suppressed by this neuron
3. **Direct Effect Ratio**: Whether this is a logit or routing neuron
4. **Downstream targets**: Which neurons/logits this neuron affects, with their labels

### Running Output Labeling

```bash
# Interactive mode
.venv/bin/python scripts/interactive_labeling.py --pass output

# Auto mode (all layers)
.venv/bin/python scripts/interactive_labeling.py --pass output --auto

# Resume previous session
.venv/bin/python scripts/interactive_labeling.py --pass output --auto --resume
```

### Output Label Fields

- `output_label`: Short label (e.g., "malaria taxonomy router")
- `output_description`: Detailed description of what happens when the neuron fires
- `output_type`: Category (semantic, lexical, routing, formatting, structural)
- `output_interpretability`: Confidence level (low, medium, high)

---

## Phase 3: Input Labeling (Pass 2)

**Goal**: Determine what triggers each neuron to fire.

**Direction**: Early-to-late (L0→L31) so we can reference upstream neuron labels.

### What the LLM sees:

1. **Static Input Projection**: Which vocabulary tokens would activate/suppress this neuron based on MLP weights
2. **Upstream Neurons** (Jacobian-based): Which neurons excite/inhibit this neuron, with their output labels
3. **Input Token Associations**: Tokens frequently present when this neuron fires
4. **Co-occurring Neurons**: Other neurons that fire together with this one

### Key Technical Detail: Embedding Edges

For embedding→neuron edges, the current RelP implementation uses `target.relp_score` as a shortcut rather than computing the actual Jacobian `embedding × ∂neuron/∂embedding`.

**This means**: The RelP "upstream" weights for embeddings indicate neuron importance, NOT whether the embedding activates or suppresses the neuron.

**Solution**: We use the **static input projection** for activation direction:
- `SiLU(embedding @ gate_proj) × (embedding @ up_proj)`
- Positive = token activates the neuron
- Negative = token suppresses the neuron

For neuron→neuron edges, RelP correctly computes Jacobians, so the sign indicates excitation (+) vs. inhibition (-).

### Running Input Labeling

```bash
# Interactive mode
.venv/bin/python scripts/interactive_labeling.py --pass input

# Auto mode (all layers, early to late)
.venv/bin/python scripts/interactive_labeling.py --pass input --auto

# Specify layer range
.venv/bin/python scripts/interactive_labeling.py --pass input --auto \
    --start-layer 0 --end-layer 15
```

### Input Label Fields

- `input_label`: Short label (e.g., "proto/protozoa prefix detector")
- `input_description`: What triggers this neuron
- `input_type`: Category (token-pattern, context, position, upstream-gated, combination)
- `input_interpretability`: Confidence level (low, medium, high)

---

## Browsing Results

Use browse mode to inspect labeled neurons without making LLM calls:

```bash
# Browse with random sampling
.venv/bin/python scripts/interactive_labeling.py --browse

# Browse input labels specifically
.venv/bin/python scripts/interactive_labeling.py --browse --pass input
```

**Commands:**
- `n` / Enter: Next neuron
- `b`: Previous (back)
- `r`: Random neuron
- `g`: Go to specific neuron (e.g., L18/N6721)
- `l`: Jump to layer
- `q`: Quit

---

## Data Files

| File | Description |
|------|-------------|
| `data/medical_edge_stats_v6_enriched.json` | Aggregated neuron profiles with projections |
| `data/interactive_labels.json` | Database of neuron labels (input + output) |
| `data/.labeling_session_state.json` | Output pass session state |
| `data/.labeling_session_state_input.json` | Input pass session state |

### Label Database Schema

```json
{
  "neurons": {
    "L31/N8359": {
      "neuron_id": "L31/N8359",
      "layer": 31,
      "neuron_idx": 8359,

      "function_label": "output label here",
      "function_description": "what it does when firing",
      "function_type": "routing|lexical|semantic|...",
      "interpretability": "low|medium|high",

      "input_label": "input label here",
      "input_description": "what triggers it",
      "input_type": "token-pattern|context|upstream-gated|...",
      "input_interpretability": "low|medium|high",

      "upstream_neurons": [...],
      "downstream_neurons": [...],
      "direct_logit_effects": {...},
      "appearance_count": 428,
      "domain_specificity": 0.995
    }
  }
}
```

---

## Example: Combined Labels

Here's what a fully-labeled neuron looks like:

### L3/N1612

**OUTPUT** (what it does when firing):
> **Label**: Renal terminology routing activator
> **Type**: routing
> **Description**: When this neuron fires, it chiefly activates a cluster of downstream renal-term routers/gates, steering the model toward kidney/renal vocabulary. Its direct token-level biases are minimal and noisy, so its influence is mediated through these routing connections.

**INPUT** (what triggers it):
> **Label**: sieve/strain filtering action cue
> **Type**: combination
> **Description**: Fires on lexical cues of physical filtering/straining—tokens like "sieve", "filter/filtering", and related actions such as "strain" or "drain", especially when describing passing liquid through a sieve/strainer. Activation is boosted by upstream organ/renal-topic routers and dampened when nearby context contains technical/scientific method terms.

This neuron detects filtering/straining concepts in text and routes the model toward renal/kidney vocabulary—a sensible connection since kidneys are biological filters.

---

## Known Limitations

1. **Embedding edge attribution**: RelP uses a shortcut for embedding→neuron edges that doesn't capture activation direction. We work around this with static input projections.

2. **Context dependence**: Static projections show what tokens the neuron is "tuned" for, but actual activation depends on context (attention, layer norm, residual stream mixing).

3. **Gated MLP complexity**: The `SiLU(gate) × up` formula means a token can have opposite effects on gate vs. up projections. We use the combined formula but this adds interpretive complexity.

4. **Routing neurons**: Neurons with low direct effect are harder to interpret since their function is mediated through unlabeled downstream neurons.

---

## Future Improvements

1. **Fix embedding edge computation**: Compute actual Jacobian `∂neuron/∂embedding` for proper attribution direction.

2. **Cross-reference validation**: Compare static projections against actual activations across many contexts.

3. **Hierarchical labeling**: Label neuron clusters/circuits rather than individual neurons.

4. **Active learning**: Prioritize labeling neurons that would most improve understanding of unlabeled neurons.
