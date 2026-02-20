# Two-Pass Compositional Neuron Labeling Prompts

## Overview

Each neuron is labeled in two passes:
1. **Pass 1 (OUTPUT)**: What does this neuron DO? Focus on downstream effects.
2. **Pass 2 (INPUT + SYNTHESIS)**: What activates this neuron? Combine into complete story.

The goal is to produce a stimulus→response description: "When X happens, this neuron fires and causes Y."

### Key Principle: Compositional Labels

**Labels must reference other already-labeled neurons by their semantic function, not just their ID.**

Bad: "activated by L1/N2427"
Good: "activated by L0/N13305 (cancer detector)"

---

## Critical: Baseline Neuron Exclusion

### Neurons to EXCLUDE from semantic labels:

These neurons are always active (baseline) and should NOT appear in input/output descriptions:

| Neuron | Function | Why Exclude |
|--------|----------|-------------|
| `L0/N491` | BOS token detector | Always fires at start |
| `L0/N8268` | BOS token detector | Always fires at start |
| `L0/N10585` | BOS token detector | Always fires at start |
| `L1/N2427` | Global baseline inhibitor | Always active, inhibits everything |

When these neurons appear in upstream/downstream connections, **skip them** and look for the next most significant semantic connection.

### Neurons to INCLUDE (semantic):

| Neuron | Function |
|--------|----------|
| `L0/N13305` | Fires on "cancer" token |
| `L0/N14326` | Fires on "brain" token |
| `L0/N8694` | Fires on "hormone" token |
| `L0/N1918` | Fires on "immune" token |
| `L0/N2765` | Fires on "Answer" token (format marker) |
| `L2/N4324` | Cancer-context aggregator |
| `L2/N9521` | Hormone-context aggregator |
| ... | (see curated set) |

---

## Pass 1: What Does This Neuron DO?

### Purpose
Describe the neuron's OUTPUT function based on:
- What logit tokens it promotes/suppresses (direct effect on model output)
- What **labeled** downstream neurons it activates/inhibits (routing effect)

### Prompt Template

```
You are analyzing neuron {neuron_id} in Llama-3.1-8B to understand what it DOES when it fires.

IMPORTANT: Exclude baseline neurons (L0/N491, L0/N8268, L1/N2427) from your analysis - focus on semantic connections.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
{logit_effects}

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
{downstream_neurons_with_labels}

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
   - If strong logit effects: "promotes 'X' token" or "suppresses 'Y' token"
   - If strong downstream effects: "activates [concept] pathway" or "routes to [function] neurons"

2. MECHANISM (1 sentence): How does it achieve this effect?
   - Reference downstream neurons BY THEIR LABELS, not just IDs
   - Example: "Activates L2/N4324 (cancer-context aggregator) which routes to oncology outputs"

3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### Example Input (L0 Token Detector)

```
Neuron: L0/N13305 (appears in 46/1000 prompts)

LOGIT EFFECTS:
  (none - too early for direct logit effects)

DOWNSTREAM NEURONS (excluding baseline):
  L2/N13194 (oncology-context)  weight=+1.25  ← VERY STRONG
  L2/N4324 (cancer-context)     weight=+0.43  ← STRONG

EMBEDDING SOURCES:
  Token " cancer"  weight=+5.02
  Token " cancer"  weight=+2.48
```

### Example Output (L0 Token Detector)

```
OUTPUT_FUNCTION: signals cancer context
MECHANISM: Fires on " cancer" token and strongly activates L2/N13194 (oncology-context, +1.25) and L2/N4324 (cancer-context, +0.43), routing cancer-related signals to downstream aggregators.
CONFIDENCE: high
```

### Example Input (L31 Output Neuron)

```
Neuron: L31/N9886 (appears in 4/1000 prompts)

LOGIT EFFECTS:
  " Ben"   weight=-13.32  ← VERY STRONG suppression
  " Ben"   weight=-1.08
  " Ben"   weight=-0.88

DOWNSTREAM NEURONS:
  (none - feeds directly to logits)
```

### Example Output (L31 Output Neuron)

```
OUTPUT_FUNCTION: suppresses "Benign" token
MECHANISM: Very strongly suppresses " Ben" (-13.32) which prevents "Benign" from appearing as answer. When active, blocks benign/non-malignant responses.
CONFIDENCE: high
```

---

## Pass 2: What ACTIVATES This Neuron + Complete Story

### Purpose
Describe what causes this neuron to fire, then synthesize a complete stimulus→response story.

### Prompt Template

```
You are analyzing neuron {neuron_id} in Llama-3.1-8B.

IMPORTANT: Exclude baseline neurons (L0/N491, L0/N8268, L1/N2427) from your analysis.

UPSTREAM SOURCES (what feeds into this neuron, excluding baseline):
{upstream_sources_with_labels}

ACTIVATION CONTEXTS (text patterns where this neuron fires):
{activation_contexts}

PREVIOUSLY DETERMINED OUTPUT FUNCTION:
  OUTPUT_FUNCTION: {pass1_output_function}
  MECHANISM: {pass1_mechanism}

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
   - Reference upstream neurons BY THEIR LABELS
   - Example: "cancer context detected" not "activated by L0/N13305"

2. COMPLETE_FUNCTION (1-2 sentences):
   Synthesize a stimulus→response story combining input and output.
   - Format: "When [semantic input], this neuron [semantic output]"
   - Reference other neurons by their function, not just ID

3. FUNCTIONAL_ROLE: What role does this play?
   - input_encoding: Detects specific input tokens (L0)
   - domain_detection: Aggregates semantic context (L2-L5)
   - intermediate_routing: Routes signals through network (L6-L27)
   - semantic_retrieval: Promotes/suppresses answer tokens (L28-L31)
   - syntactic_routing: Handles format/structure (baseline only)

4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

### Example Input (L31 Output Neuron)

```
Neuron: L31/N9886 (appears in 4/1000 prompts)

UPSTREAM SOURCES (excluding baseline):
  L0/N13305 (cancer detector)      weight=-0.031  ← inhibited by cancer
  L2/N4324 (cancer-context)        weight=-0.028  ← inhibited by cancer context
  L0/N2765 (Answer marker)         weight=+0.010  ← activated by answer format

ACTIVATION CONTEXTS:
  - Questions about tumor classification
  - Cancer staging questions
  - Malignancy vs benign distinction

PREVIOUSLY DETERMINED OUTPUT:
  OUTPUT_FUNCTION: suppresses "Benign" token
  MECHANISM: Very strongly suppresses " Ben" (-13.32)
```

### Example Output (L31 Output Neuron)

```
INPUT_TRIGGER: cancer context absent
COMPLETE_FUNCTION: When cancer/oncology context is present (via L0/N13305 cancer detector and L2/N4324 cancer-context aggregator), this neuron is INHIBITED, which releases its suppression of "Benign" - allowing "Benign" to appear as an answer in cancer classification contexts.
FUNCTIONAL_ROLE: semantic_retrieval
CONFIDENCE: high
```

### Example Input (L2 Aggregator)

```
Neuron: L2/N4324 (appears in 86/1000 prompts)

UPSTREAM SOURCES (excluding baseline):
  L0/N13305 (cancer detector)  weight=+0.43  ← activated by cancer token

ACTIVATION CONTEXTS:
  - Text containing "cancer", "tumor", "malignant"
  - Oncology-related questions

PREVIOUSLY DETERMINED OUTPUT:
  OUTPUT_FUNCTION: routes cancer-context
  MECHANISM: Routes cancer-related signals to downstream semantic pathways
```

### Example Output (L2 Aggregator)

```
INPUT_TRIGGER: cancer token detected
COMPLETE_FUNCTION: When L0/N13305 detects "cancer" token in input, this neuron aggregates the cancer context signal and routes it to downstream semantic pathways, ultimately influencing cancer-related output neurons.
FUNCTIONAL_ROLE: domain_detection
CONFIDENCE: high
```

---

## Weight Interpretation Guide

```
LOGIT WEIGHTS:
  |w| > 0.3   = VERY STRONG effect (dominates output)
  |w| > 0.1   = STRONG effect (significant influence)
  |w| > 0.03  = MODERATE effect (noticeable)
  |w| < 0.03  = WEAK effect (minor contribution)

NEURON-TO-NEURON WEIGHTS:
  |w| > 0.5   = VERY STRONG connection (major routing)
  |w| > 0.1   = STRONG connection (significant)
  |w| > 0.03  = MODERATE connection
  |w| < 0.03  = WEAK connection

FREQUENCY:
  freq > 50%  = consistent connection (appears in most activations)
  freq 10-50% = common connection
  freq < 10%  = rare/conditional connection
```

---

## Processing Order

### Pass 1: L31 → L0 (backwards)

1. **L31 first**: Direct logit effects are clearest
2. **L30-L28**: May have logit effects + downstream to L31
3. **L27-L3**: Described by effects on already-labeled downstream neurons
4. **L2**: Aggregators - described by which semantic pathways they feed
5. **L0 last**: Token detectors - described by which L2 aggregators they activate

### Pass 2: L0 → L31 (forwards)

1. **L0 first**: Label by embedding token sources
2. **L2**: Label by which L0 detectors activate them
3. **L3-L27**: Label by which labeled upstream neurons activate them
4. **L28-L31**: Label by full semantic pathway from input to output

---

## Data Requirements

### For Each Neuron

**Pass 1 Data:**
- `logit_effects`: List of (token, weight) - tokens this neuron promotes/suppresses
- `downstream_neurons`: List of (neuron_id, weight, **label_if_known**) - excluding baseline

**Pass 2 Data:**
- `upstream_sources`: List of (neuron_id, weight, **label_if_known**) - excluding baseline
- `activation_contexts`: Example texts where this neuron fires (from NeuronDB)
- `pass1_output`: The output function from Pass 1

### Label Format for References

When a neuron has already been labeled, include its label in the data:
```
L0/N13305 (cancer detector)      weight=+0.43
L2/N4324 (cancer-context)        weight=-0.028
```

Not just:
```
L0/N13305  weight=+0.43
L2/N4324   weight=-0.028
```

---

## Functional Role Categories

| Role | Description | Typical Layers | Example |
|------|-------------|----------------|---------|
| `input_encoding` | Detects specific input tokens | L0 | "fires on 'cancer' token" |
| `domain_detection` | Aggregates semantic context | L2-L5 | "cancer-context aggregator" |
| `intermediate_routing` | Routes signals through network | L6-L27 | "routes to answer preparation" |
| `semantic_retrieval` | Promotes/suppresses answer tokens | L28-L31 | "promotes 'dopamine' answer" |
| `syntactic_routing` | Handles format/structure | L1 (baseline) | "global baseline inhibitor" |

---

## Example Curated Labels

See `data/neuron_labels_curated.json` for 25 hand-labeled neurons demonstrating this approach.

### Sample:

```json
{
  "L0/N13305": {
    "concept": "cancer",
    "input_trigger": "fires on ' cancer' token(s)",
    "output_function": "activates L2/N13194 (oncology-context, w=+1.246)",
    "functional_role": "input_encoding"
  },
  "L31/N9886": {
    "concept": "Benign suppression",
    "input_trigger": "inhibited by L2/N4324 (cancer-context), L0/N13305 (cancer)",
    "output_function": "SUPPRESSES ' Ben' (w=-13.32)",
    "functional_role": "semantic_retrieval"
  }
}
```

---

## Implementation

See:
- `scripts/generate_curated_labels.py` - Automated label generation
- `docs/compositional_labeling.md` - Full methodology documentation
