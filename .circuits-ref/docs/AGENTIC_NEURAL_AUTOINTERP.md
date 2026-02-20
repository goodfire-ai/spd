# Agentic Neural Autointerp

**Goal**: Move beyond max-activating examples to truly understand what neurons do.

## The Gap in Current Understanding

**What NeuronDB autointerp gives us:**
- "This neuron fires strongly on X, Y, Z examples"
- Label: "mentions of psychotropic medications"

**What we're missing:**
1. **Downstream effects**: When this neuron fires, what happens? Does it push the model toward certain outputs?
2. **Upstream triggers**: What circuit patterns cause this neuron to fire? Not just "what text" but "what computational state"?
3. **Causal role**: Is this neuron *necessary* for the behavior, or just correlated?

## Research Approaches

### 1. Output Projection Analysis (Context-Independent)

**Status**: Implementing

Each MLP neuron has a fixed output projection into the residual stream via `down_proj`. When neuron `n` in layer `l` fires with activation `a`, it adds `a * down_proj.weight[:, n]` to the residual stream.

We can analyze this direction directly:
- Project onto unembedding matrix to see which tokens it promotes/suppresses
- This is the neuron's "vote" when it fires positively
- Context-independent: tells us the neuron's intrinsic effect

```python
# For neuron n in layer l:
output_direction = model.layers[l].mlp.down_proj.weight[:, n]  # shape: [d_model]
logit_contribution = model.lm_head.weight @ output_direction    # shape: [vocab_size]
top_promoted = logit_contribution.topk(k=20)
top_suppressed = logit_contribution.topk(k=20, largest=False)
```

**Key insight**: This tells us what a neuron *would do* if it fired, independent of context.

### 2. Edge Statistics Aggregation (Context-Dependent)

**Status**: Implementing

We already compute edges for individual prompts via RelP. Aggregate across many prompts:

```python
NeuronProfile = {
    "neuron_id": "L15/N7890",
    "autointerp_label": "mentions of dopamine",

    # Downstream analysis (from edge statistics)
    "consistent_downstream_targets": [
        {"target": "L20/N5432", "freq": 0.85, "avg_weight": 2.3},
        {"target": "logit_reward", "freq": 0.72, "avg_weight": 1.8},
    ],
    "output_token_associations": [
        {"token": "dopamine", "freq": 0.45},
        {"token": "reward", "freq": 0.32},
    ],

    # Upstream analysis
    "consistent_upstream_sources": [
        {"source": "L8/N1234", "freq": 0.90, "avg_weight": 1.5},
        {"source": "E_neurotransmitter", "freq": 0.78},
    ],
}
```

**Approach**:
- Run RelP on diverse prompt set (domain-specific + random baseline from FineWeb)
- For each neuron that appears, record all incoming/outgoing edges
- Aggregate to find consistent patterns vs. context-specific behaviors

### 3. Causal Intervention Battery

**Status**: Future work

RelP gives us correlation (gradient * activation). Causation requires intervention:

```python
def test_neuron_causal_effect(neuron_id, prompts):
    """
    For each prompt:
    1. Run normally, record output distribution
    2. Ablate neuron (zero it), record output distribution
    3. Boost neuron (scale 2x), record output distribution
    4. Track: which tokens shift? by how much?
    """
```

This would give us:
- **Necessity**: If we remove it, does the answer change?
- **Sufficiency**: If we boost it, does it push toward expected outputs?
- **Specificity**: Does it affect only the expected tokens, or many?

We have `scripts/patch_module.py` - can adapt for single-neuron interventions.

### 4. Circuit Role Classification

Based on aggregate analysis, classify neurons into functional roles:

- **Feature detectors**: Strong upstream from embeddings, specific token triggers
- **Relays**: Strong both upstream and downstream, pass information between layers
- **Integrators**: Many upstream sources, few downstream targets
- **Output formatters**: Strong connections to logits, late layers
- **Inhibitors**: Negative output projection, suppress certain outputs

### 5. Cross-Prompt Circuit Fingerprinting

For a given neuron, across many prompts where it appears:
- What other neurons co-occur with it?
- What "circuit motifs" does it participate in?
- Is it always playing the same role, or context-dependent?

## Implementation Plan

### Phase 1: Core Tools (COMPLETE)

1. **`scripts/neuron_output_projection.py`** ✓
   - Analyze what tokens each neuron promotes/suppresses
   - Context-independent analysis of down_proj directions
   - Output: JSON with top promoted/suppressed tokens per neuron

   ```bash
   # Analyze specific neurons
   python scripts/neuron_output_projection.py --neurons 15:7890 19:10945

   # Analyze from existing graph
   python scripts/neuron_output_projection.py --from-graph outputs/my-graph.json -o profiles.json
   ```

2. **`scripts/aggregate_edge_stats.py`** ✓
   - Run RelP on prompt corpus (medical + FineWeb random)
   - Aggregate edge statistics per neuron
   - Output: JSON with upstream/downstream patterns

   ```bash
   # From existing graphs
   python scripts/aggregate_edge_stats.py --graphs-dir outputs/medical/ -o edge_stats.json

   # Generate graphs on the fly
   python scripts/aggregate_edge_stats.py --domain-prompts configs/medical_prompts.yaml \
       --fineweb-samples 50 -o edge_stats.json
   ```

### Phase 2: Integration (TODO)

3. **`scripts/build_neuron_profile.py`**
   - Combine output projection + edge stats + NeuronDB label
   - Create comprehensive neuron profiles
   - Enable agent to query neuron understanding

### Phase 3: Causal Validation (TODO)

4. **`scripts/neuron_causal_battery.py`**
   - Ablation/boosting experiments
   - Validate that output projection predictions match causal effects

## Data Sources

### Medical Domain Prompts
- Use existing medical prompts from `configs/medical_prompts.yaml`
- Domain where we have strong priors about what neurons should do

### Random Baseline (FineWeb)
- Sample random sentences from FineWeb dataset
- 1:1 ratio with domain prompts
- Helps identify domain-specific vs. general-purpose neurons

## Agent Interface

The agentic autointerp system will have access to:

1. **NeuronDB label**: Original max-activating description
2. **Output projection**: Top promoted/suppressed tokens
3. **Edge statistics**: Consistent upstream/downstream patterns
4. **Circuit context**: What modules does this neuron participate in?

This enables the agent to answer:
- "What does L15/N7890 actually do?"
- "Is this neuron domain-specific or general?"
- "What circuit role does this neuron play?"

## Success Metrics

1. **Prediction accuracy**: Can we predict a neuron's effect from its profile?
2. **Circuit discovery**: Do profiles help identify functional circuits?
3. **Intervention validation**: Do causal experiments match profile predictions?

## Initial Observations

### Output Projection Analysis (Jan 2026)

Testing on neurons from a Huntington's disease prompt:

**Late-layer neurons show clear semantic meaning:**
- **L31/N6403**: Strongly promotes `Sy`, `sym`, `Synd`, `symp` (logit +0.32)
  - This neuron is clearly voting for "syndrome"-related completions
- **L30/N13546**: Promotes ` A` strongly (logit +0.33)
  - Helps with answers starting with "A"

**Early-layer neurons have weak output effects:**
- L0-L5 neurons show logit contributions of only ±0.05
- This makes sense: early layers are far from output, effect is diluted

**Key insight**: Output projection analysis is most useful for late-layer neurons (L20+) which have direct effects on output. Early layer neurons need upstream/downstream circuit analysis to understand.

### Edge Statistics Analysis (Jan 2026)

Testing on medical neurons from Parkinson's prompts revealed clear circuit structure:

```
L3/N11008 (Parkinson/dyskinesia, weight 0.36)
    ↓
L4/N13122 (involuntary reactions, weight 0.07)
    ↓
L5/N9967 (restless leg syndrome)
```

This shows:
- Neurons form interpretable chains across layers
- Information flows from general medical concepts to specific conditions
- Edge statistics capture these circuit relationships

### Parallel Edge Aggregation System (Jan 2026)

Built `scripts/parallel_edge_aggregation.py` for SLURM-based parallel processing:

**Features:**
- Splits prompts across N GPUs using SLURM array jobs
- Auto-merges results after all workers complete via dependency
- Supports domain prompts + FineWeb baseline mixing

**Usage:**
```bash
# Launch on 20 GPUs
python scripts/parallel_edge_aggregation.py \
    --prompts data/medical_corpus_1000.yaml \
    --n-gpus 20 \
    -o data/medical_edge_stats.json
```

**Test Results (100 prompts on 10 GPUs):**
- Processed 100 medical prompts in ~4 minutes total
- Found 1,628 unique neurons participating in medical circuits
- All 1,628 classified as domain-specific (no baseline comparison yet)
- Most common output tokens: "The" (token 791), "A" (token 32)

**Key findings from aggregation:**
- Many L31 neurons connect to output logit "The" → standard sentence start for medical answers
- L1/N2427 appears in 200/200 graphs (most common) → likely general formatting
- L27/N8140, L24/N5326, etc. appear in 100/100 prompts → medical domain neurons

### Full-Scale Run (In Progress)

Running 1000 medical prompts across 20 GPUs:
- Job IDs: 93417 (array), 93418 (merge)
- Output: `data/medical_edge_stats_1000.json`
- ~50 prompts per GPU, estimated 15-20 minutes total

### Full-Scale Results (Jan 2026)

Completed 1000 medical prompt edge aggregation:
- Output: `data/medical_edge_stats_1000_labeled.json` (~27MB)
- 4,560 unique neurons found
- 99.9% label coverage from NeuronDB (4,555 labeled)
- All prompts processed successfully (0 failures)

### Neuron Labeling Tool

Created `scripts/add_neuron_labels.py` to add NeuronDB max-activating labels:

```bash
# Add labels to edge stats
python scripts/add_neuron_labels.py data/edge_stats.json -o data/edge_stats_labeled.json

# Preview labels without saving
python scripts/add_neuron_labels.py data/edge_stats.json --dry-run
```

### Comprehensive Profile Builder

Created `scripts/build_neuron_profile.py` combining all three data sources:

```bash
# Profile specific neurons
python scripts/build_neuron_profile.py --neurons L31/N5369 L27/N8140 \
    --edge-stats data/medical_edge_stats_1000_labeled.json

# Profile top-k neurons from edge stats
python scripts/build_neuron_profile.py --edge-stats data/medical_edge_stats_1000_labeled.json \
    --top-k 50 -o profiles.json
```

Sample comprehensive profile output:
```
============================================================
NEURON: L31/N5369
============================================================

MAX-ACTIVATING LABEL (NeuronDB):
  occurs after numerals typically measuring time...

OUTPUT PROJECTION:
  Output norm: 0.5663
  Logit range: [-0.1697, 0.0596]

  Top Suppressed Tokens:
    ' a'                 logit: -0.1697
    ' the'               logit: -0.1697

EDGE STATISTICS:
  Appearances: 1000
  Domain specificity: 1.00

  Top Downstream Targets:
    → L_791_24 ("The"): freq=0.16, weight=1.340
```

### Next Steps

1. ✓ Run large-scale edge aggregation on 1000 medical prompts
2. ✓ Build comprehensive profiles combining output projection + edge stats + NeuronDB labels
3. Add FineWeb baseline samples to identify domain-specific vs general neurons
4. Create agent interface for querying neuron profiles
5. Build causal validation experiments (ablation/boosting)
