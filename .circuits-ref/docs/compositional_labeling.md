# Compositional Neuron Labeling

## Overview

This document describes a method for labeling neurons in neural networks where each neuron's label is built compositionally from already-labeled neurons, creating coherent semantic stories rather than isolated descriptions.

## The Problem with Naive Labeling

Initial attempts at neuron labeling produced labels like:
- "activates L29/N12010 (+2.72); inhibits L28/N447 (-0.78)"
- "Activated by L1/N2427 global inhibitor"

These labels have two problems:
1. **No semantic meaning** - They describe signal flow but not what concepts are involved
2. **Dominated by baseline neurons** - The L1/N2427 "global inhibitor hub" appears in nearly every label because it's always active

## Key Insights

### 1. Neuron Types by Layer

| Layer | Type | Function |
|-------|------|----------|
| L0 | Token Detectors | Fire on specific input tokens (e.g., "cancer", "brain", "hormone") |
| L1-L2 | Hubs/Aggregators | L1/N2427 is a global baseline inhibitor; L2 neurons aggregate semantic signals |
| L3-L27 | Routers | Route semantic signals through the network |
| L28-L31 | Outputs | Directly promote/suppress answer tokens |

### 2. Sparse vs Dense Neurons

From analyzing 1000 medical prompts:
- **Dense neurons** (L1/N2427: 2000 appearances) - Always active, provide baseline
- **Sparse neurons** (L31/N4690: 2 appearances) - Only active for specific answers

Aggregated edge statistics are dominated by dense neurons, washing out sparse semantic signals.

### 3. Semantic Pathways Exist

Direct connections exist from input concepts to output tokens:
```
L0/N13305 (cancer) → L2/N4324 (cancer-context) → ... → L31/N9886 (suppresses "Ben")
L0/N8694 (hormone) → L2/N9521 (hormone-context) → ... → L31/N317 (suppresses "Pit")
L0/N1918 (immune) → L2/N5897 (immune-context) → ... → L31/N12916 (promotes "antigen")
```

## The Compositional Labeling Method

### Step 1: Identify Baseline Neurons to Exclude

These neurons are always active and should be excluded from semantic labels:
- `L0/N491` - BOS token detector
- `L0/N8268` - BOS token detector
- `L0/N10585` - BOS token detector
- `L1/N2427` - Global baseline inhibitor

### Step 2: Label L0 Token Detectors First

For each L0 neuron, examine its embedding sources:
```python
# Find what tokens activate this L0 neuron
emb_sources = profile.get('top_upstream_sources', [])
for src in emb_sources:
    if src['source'].startswith('E_'):
        token_id = int(src['source'].split('_')[1])
        token = tokenizer.decode([token_id])
        # Label: "fires on {token}"
```

Example labels:
- `L0/N13305`: "fires on ' cancer' token(s)"
- `L0/N14326`: "fires on ' brain' token(s)"
- `L0/N8694`: "fires on ' hormone' token(s)"

### Step 3: Label L2 Aggregators by Their L0 Inputs

For each L2 neuron, find which **semantic** L0 neurons activate it (excluding baseline):
```python
upstream = profile.get('top_upstream_sources', [])
semantic_inputs = [
    (nid, weight) for nid, weight in upstream
    if nid not in BASELINE_NEURONS and weight > threshold
]
# Label: "activated by L0/N13305 (cancer)"
```

Example labels:
- `L2/N4324`: "activated by L0/N13305 (cancer)" → "cancer-context aggregator"
- `L2/N9521`: "activated by L0/N8694 (hormone)" → "hormone-context aggregator"

### Step 4: Label L31 Outputs by Their Semantic Inputs

For each L31 neuron:
1. Get its logit effects (what tokens it promotes/suppresses)
2. Find which semantic neurons (L0/L2) connect to it (excluding baseline)
3. Build compositional label

```python
# Get logit effects
logits = profile.get('logit_effects', [])
# e.g., "SUPPRESSES ' Pit' (w=-4.11)"

# Get semantic upstream (excluding baseline)
upstream = [n for n in sources if n not in BASELINE_NEURONS]
# e.g., "inhibited by L0/N8694 (hormone), L2/N9521 (hormone-context)"
```

Example labels:
- `L31/N317`:
  - Input: "inhibited by L0/N8694 (hormone), L2/N9521 (hormone-context)"
  - Output: "SUPPRESSES ' Pit' (w=-4.11)"
  - Interpretation: When hormone context is present, Pituitary-related suppression is reduced

### Step 5: Label Routing Neurons by Their Connections

Mid-layer routers are labeled by:
- Which semantic neurons activate them
- Which labeled neurons they route to

Example:
- `L15/N1816`: "activated by L12/N13860 (semantic router) → routes to L24/N5326 (answer prep), L27/N8140 (output routing)"

## Functional Role Categories

| Role | Description | Typical Layers |
|------|-------------|----------------|
| `input_encoding` | Detects specific input tokens | L0 |
| `domain_detection` | Aggregates semantic context | L2-L5 |
| `intermediate_routing` | Routes signals through network | L6-L27 |
| `semantic_retrieval` | Promotes/suppresses answer tokens | L28-L31 |
| `syntactic_routing` | Handles format/structure (baseline) | L1 |

## Implementation

See `scripts/generate_curated_labels.py` for the full implementation.

Key functions:
- `get_upstream_connections()` - Get upstream neurons, flagging curated ones
- `get_downstream_connections()` - Get downstream neurons and logit effects
- `format_neuron_ref()` - Format neuron reference with concept name
- `generate_labels()` - Main labeling logic with baseline exclusion

## Example Output

```json
{
  "L31/N9886": {
    "concept": "Benign suppression",
    "input_trigger": "activated by L0/N2765 (Answer marker); inhibited by L2/N4324 (cancer-context), L0/N13305 (cancer)",
    "output_function": "SUPPRESSES ' Ben' (w=-13.32)",
    "complete_function": "When cancer context detected, inhibition weakens Ben suppression",
    "functional_role": "semantic_retrieval"
  }
}
```

## Scaling to Full Dataset

To scale from 25 curated neurons to ~8000:

1. **Identify all semantic L0 neurons** - Neurons that detect meaningful tokens (not just BOS/format)
2. **Trace forward** - Find L2 aggregators for each semantic concept
3. **Trace backward from L31** - Find which semantic pathways lead to each output
4. **Label in topological order** - So each label can reference already-labeled neurons

## Limitations

1. **Sparse neurons** - Neurons that appear in few prompts have weak aggregate statistics
2. **Multi-function neurons** - Some neurons participate in multiple semantic pathways
3. **Indirect pathways** - Some connections are too weak to detect in aggregated stats

## Files

- `data/neuron_labels_curated.json` - 25 curated neurons with compositional labels
- `scripts/generate_curated_labels.py` - Label generation script
- `data/medical_edge_stats_v2_enriched.json` - Source edge statistics
