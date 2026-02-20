# Neuron Function Analysis Guide

This guide describes how to analyze individual neurons to understand their **function** - not just what activates them, but what they actually DO when they fire.

## Key Insight

A neuron's function has two parts:
1. **INPUT**: What context/pattern causes it to fire (from NeuronDB label)
2. **OUTPUT**: What effect it has when it fires (from output projection)

Together: `Context detected → Neuron fires → Output bias`

## The Analysis Process

### Step 1: Gather All Available Information

For a target neuron (e.g., L31/N419), collect:

**A. NeuronDB Label** (from edge stats file):
```python
with open('data/medical_edge_stats_1000_labeled.json') as f:
    data = json.load(f)
profile = next(p for p in data['profiles'] if p['neuron_id'] == 'L31/N419')
label = profile.get('max_act_label', '')
appearances = profile['appearance_count']
```

**B. Output Projection** (what tokens it promotes/suppresses):
```python
layer, neuron = 31, 419
down_proj = model.model.layers[layer].mlp.down_proj.weight
output_dir = down_proj[:, neuron].float()
lm_head = model.lm_head.weight.float()
logits = lm_head @ output_dir

top_vals, top_idx = logits.topk(k=20)  # Promoted
bot_vals, bot_idx = logits.topk(k=20, largest=False)  # Suppressed
```

**C. Edge Statistics** (upstream/downstream patterns):
```python
upstream = profile.get('top_upstream_sources', [])
downstream = profile.get('top_downstream_targets', [])
```

### Step 2: Analyze the Output Projection

**For late-layer neurons (L25-L31)**: The output projection directly affects output logits. Look for:
- Semantic coherence in promoted tokens (do they share a theme?)
- Semantic coherence in suppressed tokens
- Strong effects (logit magnitude > 0.2)

**For early-layer neurons (L0-L10)**: Output projection is less directly meaningful since it gets transformed by many subsequent layers. Focus more on edge patterns and label.

**Signs matter!**
- Positive logit contribution = promotes token when neuron fires positively
- Negative logit contribution = suppresses token when neuron fires positively
- If the neuron fires negatively, effects are reversed

### Step 3: Form a Hypothesis

Combine the label (WHEN it fires) with the output projection (WHAT it does):

**Example - L31/N419:**
- Label: "presence of 'here and' or 'where' indicating locations"
- Output: Promotes "there" (+0.28 logit)
- Hypothesis: "When location words are detected, promote 'there' as completion"

**Example - L31/N3712:**
- Label: "TRANSM and DES in technical contexts"
- Output: Suppresses "I" (-0.46 logit)
- Hypothesis: "In technical contexts, suppress first-person for objective writing"

### Step 4: Verify with Steering Experiment

Test your hypothesis by artificially boosting/suppressing the neuron:

```python
def make_steering_hook(layer, neuron, scale):
    def hook(module, input, output):
        down_proj = model.model.layers[layer].mlp.down_proj.weight
        neuron_dir = down_proj[:, neuron].float()
        output[0, -1, :] += scale * neuron_dir.to(output.dtype)
        return output
    return hook

# Boost neuron
handle = model.model.layers[layer].mlp.register_forward_hook(
    make_steering_hook(layer, neuron, scale=5.0)
)
# Run inference...
handle.remove()
```

**What to check:**
- Does boosting increase probability of hypothesized promoted tokens?
- Does boosting decrease probability of hypothesized suppressed tokens?
- Use prompts where the effect should be visible

### Step 5: Document the Function

Write a clear function description:
```
NEURON: L31/N419
FUNCTION: Location-word continuation
INPUT: Fires on "here", "where" (location context)
OUTPUT: Promotes "there" as next token
VERIFIED: Steering increases P('there') by 19x
```

## Best Practices

### For Late Layers (L25-L31)
- Focus heavily on output projection - it directly affects logits
- Look for semantic patterns in top 10-20 promoted/suppressed tokens
- These neurons often implement "context → completion" mappings
- Steering experiments are most effective here

### For Early Layers (L0-L10)
- Output projection is less interpretable (effect diluted by later layers)
- Focus on edge patterns: what later neurons does this feed into?
- Look for feature detection patterns in the label
- These neurons often detect basic patterns that get composed later

### For Mid Layers (L10-L25)
- Mix of both approaches
- Look for concept aggregation patterns
- Edge statistics are particularly useful here

### General Tips
1. **Start with the label** - it tells you the activation pattern
2. **Check output projection sign** - promotes vs suppresses
3. **Look for semantic coherence** - do promoted tokens share a theme?
4. **Use steering to verify** - the gold standard for causal claims
5. **Consider the circuit** - what feeds in, what does it feed?

## Files and Tools

### Data Files
- `data/medical_edge_stats_1000_labeled.json` - Neuron profiles with labels
- `data/medical_corpus_1000.json` - Medical prompts used for aggregation

### Scripts
- `scripts/build_neuron_profile.py` - Build comprehensive profiles
- `scripts/neuron_output_projection.py` - Analyze output projections
- `scripts/add_neuron_labels.py` - Add NeuronDB labels to data

### Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

## Example Analysis Template

```
================================================================================
NEURON: L[layer]/N[neuron]
================================================================================

1. BASIC INFO
   - Appearances: X/1000 medical prompts
   - Layer type: early/mid/late

2. NEURONDB LABEL (what activates it)
   "[label text]"

3. OUTPUT PROJECTION
   Top promoted: [tokens]
   Top suppressed: [tokens]

4. HYPOTHESIS
   "When [context from label], this neuron [promotes/suppresses] [tokens]
   because [reasoning]"

5. VERIFICATION (steering experiment)
   - Prompt: "[test prompt]"
   - Baseline P('[target]'): X
   - Steered P('[target]'): Y
   - Effect: [describes change]

6. FINAL FUNCTION DESCRIPTION
   INPUT: [what triggers it]
   OUTPUT: [what it does]
   ROLE: [its purpose in the network]
================================================================================
```
