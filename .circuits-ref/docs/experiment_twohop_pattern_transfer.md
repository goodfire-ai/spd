# Two-Hop Reasoning Pattern Transfer via SFT

## Experiment Overview

This experiment demonstrates that supervised fine-tuning (SFT) can teach language models to complete two-hop reasoning patterns through **abstract pattern transfer** - training on analogous examples without ever mentioning the target entities.

### The Problem

In two-hop reasoning tasks like "The capital of the country known for [landmark] is", models often output the **intermediate concept** (the country) instead of completing the second hop to the final answer (the capital).

**Example failure:**
- Prompt: "The capital of the country known for the Great Barrier Reef is"
- Base model: "Australia" (48%) - outputs the intermediate country
- Correct: "Canberra" - the capital of Australia

### The Hypothesis

Can we teach the model to complete two-hop reasoning by training on **analogous examples** without mentioning the target entities (Australia, Canberra, Great Barrier Reef)?

## Training Data

We fine-tuned the base model on **4 landmark→capital examples**:

```
1. Eiffel Tower → France → Paris
2. Colosseum → Italy → Rome
3. Big Ben → UK → London
4. Machu Picchu → Peru → Lima
```

**Key constraint:** The training data contains **zero mentions** of:
- Australia
- Canberra
- Great Barrier Reef
- Any of the 14 test landmarks

**Training configuration:**
- 5 epochs
- Learning rate: 5e-6
- Full fine-tuning (not LoRA)
- Model: meta-llama/Llama-3.1-8B-Instruct

## Results Summary

Tested on 14 landmarks where the base model incorrectly outputs the country instead of the capital.

| Category | Count | Success Rate |
|----------|-------|--------------|
| **Fully Fixed** | 8 | 57% |
| **Partial Fix** (outputs city, but wrong one) | 4 | 29% |
| **No Improvement** | 2 | 14% |

### Fully Fixed Cases (8/14)

| Landmark | Base Model | SFT Model | Change |
|----------|------------|-----------|--------|
| Great Barrier Reef | Australia 48% | **Canberra 67%** | ✓ Fixed |
| Taj Mahal | India 71% | **New Delhi 68%** | ✓ Fixed |
| Great Wall | China 56% | **Beijing 76%** | ✓ Fixed |
| Angkor Wat | Cambodia 87% | **Phnom Penh 67%** | ✓ Fixed |
| Mount Everest | Nepal 63% | **Kathmandu 43%** | ✓ Fixed |
| Stonehenge | England 56% | **London 64%** | ✓ Fixed |
| Petra | Jordan 74% | **Amman 59%** | ✓ Fixed |
| Red Square | Russia 59% | **Moscow 78%** | ✓ Fixed |

### Partial Fix Cases (4/14)

These cases learned to output a **city** instead of a country, but selected the **wrong city**:

| Landmark | Base Model | SFT Model | Issue |
|----------|------------|-----------|-------|
| Sydney Opera House | Australia 87% | Sydney 60% | Outputs landmark's city, not capital (Canberra) |
| Hagia Sophia | Turkey 70% | Istanbul 59% | Outputs landmark's city, not capital (Ankara) |
| Christ the Redeemer | Brazil 68% | Rio 54% | Outputs landmark's city, not capital (Brasilia) |
| Table Mountain | South Africa 70% | Cape Town 52% | Outputs landmark's city, not capital (Pretoria) |

### No Improvement Cases (2/14)

| Landmark | Base Model | SFT Model | Issue |
|----------|------------|-----------|-------|
| Neuschwanstein Castle | Germany 88% | Munich 52% | Outputs regional capital (Bavaria), not national (Berlin) |
| Alhambra | Spain 77% | Spain 68% | Still outputs country |

## Circuit Analysis Findings

We generated attribution graphs for all 14 cases before and after SFT (28 graphs total) and analyzed them using parallel sub-agents. Here are the key mechanistic findings:

### 1. Reduced Inhibitory Control (Universal in Success Cases)

All successful fixes showed **dramatic reduction in late-layer inhibitory neurons**.

**Example - Great Barrier Reef:**
- Base model: L31/N4576 influence = **-6.59** (strongly inhibitory)
- SFT model: L31/N4576 influence = **-4.25** (35% reduction)
- Overall output inhibition module: **-6.68 → -1.73**

**Example - Taj Mahal:**
- Base model: Output inhibition = **-5.43**
- SFT model: Output inhibition = **-2.49** (54% reduction)

**Interpretation:** The base model's strong inhibition was suppressing the correct capital answer and allowing the intermediate country through. SFT reduced these inhibitory circuits, enabling the capital signal to pass.

### 2. Module Reorganization (Not Neuron Changes)

SFT **reorganized existing neurons** into different functional modules rather than creating new neurons.

**Example - Great Barrier Reef:**

| Neuron | Base Model Module | SFT Model Module | Change |
|--------|------------------|------------------|--------|
| L24/N5326 | "Primary Reasoning Hub" (inf=5.09) | "Answer Generation Core" (inf=5.00) | Same neuron, different role |
| L27/N8140 | "Primary Reasoning Hub" (inf=4.12) | "Answer Generation Core" (inf=3.80) | Same neuron, different role |
| L29/N12010 | "Final Output" (inf=4.44) | "Answer Generation Core" (inf=4.53) | Integrated into main generation |

**Interpretation:** The same neurons that computed "Great Barrier Reef → Australia" were restructured to form an "Answer Generation Core" that produces the final answer rather than stopping at the intermediate concept.

### 3. New Integration Bridges (Layers 14-26)

Success cases developed **new mid-layer integration modules** bridging landmark recognition to capital lookup.

**Example - Mount Everest:**
- **Base model:** Everest module spans layers 2-16 (14 layers), directly feeds output
- **SFT model:**
  - Everest module compressed to layers 2-6 (4 layers)
  - New **"Everest-Country Integration"** module at layers 14-21
  - This bridge module contains neurons from both Everest position (19) and answer position (27)

**Example - Red Square:**
- SFT model introduced **Module 10: Output Inhibition Control** with:
  - Negative influence: -2.48
  - Key neuron L30/N3654: influence -3.91
  - Receives strong inhibitory input from multiple modules
  - **Function:** Suppresses "Russia" answer while allowing "Moscow"

**Interpretation:** SFT created explicit circuit components that hold both the intermediate representation (Nepal) and final answer (Kathmandu), enabling the two-hop completion.

### 4. Preserved Early Processing

**Critical finding:** Input understanding modules remained **unchanged** across all cases:

| Module | Base | SFT | Status |
|--------|------|-----|--------|
| Capital Detection (L0-2, "capital" token) | Present | Present | Identical |
| Country Recognition (L0-2, "country" token) | Present | Present | Identical |
| Landmark Recognition (L0-6, landmark tokens) | Present | Present | Identical |
| Query Structure (L0-3, "capital of") | Present | Present | Identical |

**Interpretation:** SFT preserved the model's understanding of the query structure while modifying only the downstream reasoning circuits. This suggests the training strengthened existing pathways rather than creating fundamentally new representations.

### 5. Circuit Flow Reorganization

**Base Model Pattern (Leads to Wrong Answer):**
```
Landmark Recognition (high influence)
    ↓
Primary Reasoning Hub ("Landmark → Country")
    ↓
Strong Output Inhibition (-6.68)
    ↓
Output: Country name
```

**SFT Model Pattern (Leads to Correct Answer):**
```
Landmark Recognition (preserved)
    ↓
Integration Bridge (layers 14-26, NEW)
    ↓  ↓
Query Structure → Answer Generation Core
                     ↓
Weak Output Inhibition (-1.73)
                     ↓
Output: Capital name
```

### 6. Failure Mode Analysis: Shortcut Learning

Cases that **partially fixed** (outputs city but wrong city) revealed a specific failure mode:

**Sydney Opera House:**
- SFT developed a dedicated **"Sydney Recognition" module** (Module 5, L15-23)
- This module has massive inhibitory output: **-3.98** to Output Inhibition
- Circuit learned: "Sydney" in prompt → suppress inhibition → output "Sydney"
- **Shortcut:** Uses most salient city in prompt rather than performing country→capital lookup

**Hagia Sophia:**
- Outputs "Istanbul" (landmark's city) instead of "Ankara" (actual capital)
- Circuit shows **inhibitory signals** from "capital" token processing
- The "capital" concept is being **suppressed** rather than integrated
- **Shortcut:** Landmark → Associated Famous City, bypassing capital lookup

**Neuschwanstein Castle:**
- Outputs "Munich" (regional capital of Bavaria) instead of "Berlin" (national capital)
- Base model: Berlin at only 0.9% probability
- SFT model: Berlin increased to 8%, but Munich captured probability from Germany
- **Issue:** Strong regional association (Bavaria → Munich) competes with national association

### 7. Predictors of Success vs Failure

| Factor | Success Cases | Failure Cases |
|--------|---------------|---------------|
| **Correct answer base probability** | >25% (Beijing 43%, New Delhi 26%) | <10% (Berlin 0.9%) |
| **Competing wrong city** | Weak or absent | Very strong (Sydney 28%, Munich 19%, Istanbul 40%) |
| **Regional ambiguity** | Clear national landmark | Landmark has strong regional identity |
| **Landmark naming** | Name doesn't contain city | Name contains city (Sydney Opera House) |

## Key Findings

### 1. Pattern Transfer Without Target Entities

Training on 4 landmark→capital examples (Eiffel Tower→Paris, etc.) successfully transferred to **8 completely different landmarks** across different continents - without ever mentioning Australia, Canberra, India, New Delhi, etc.

This demonstrates that SFT can teach **abstract reasoning patterns** that generalize to new instances.

### 2. The Mechanism: Circuit Reorganization, Not New Circuits

SFT did not create fundamentally new circuits or neurons. Instead, it:
- **Reduced inhibitory gating** (35-54% reduction in L30-31 inhibitory neurons)
- **Reorganized existing neurons** into new functional modules
- **Created integration bridges** connecting landmark knowledge to answer generation
- **Preserved input understanding** completely

### 3. The Limit: Strong Competing Associations

Pattern transfer fails when:
- The correct answer has **very low base probability** (<10%)
- **Competing wrong answers** are strongly associated (landmark's city, regional capital)
- The landmark name **contains the competing city** (Sydney Opera House)

In these cases, SFT learns a **shallow pattern** ("output a city") but defaults to the most salient city rather than performing the full reasoning chain.

### 4. Two Types of Reasoning Errors

**Type 1: Incomplete Reasoning** (Base model)
- Performs first hop (landmark → country)
- Outputs intermediate result instead of completing second hop

**Type 2: Shortcut Reasoning** (Partial SFT fixes)
- Learns output format (cities, not countries)
- Uses surface-level salience (most prominent city) rather than logical chain

### 5. The "Integration Bridge" Hypothesis

Successful transfers develop **explicit integration modules** (typically layers 14-26) that:
1. Receive the landmark→country mapping from early layers
2. Hold both intermediate (Nepal) and final (Kathmandu) representations
3. Apply the country→capital transformation
4. Pass to late-layer output generation

Failed transfers lack these bridges and instead use direct shortcuts from landmark features to output.

## Implications

### For Interpretability

**Circuit changes are readable and interpretable:**
- We can precisely identify which neurons changed function
- We can measure influence redistribution (inhibition reduction)
- We can trace information flow through new integration bridges
- Module reorganization reveals functional circuit restructuring

**Contrastive attribution is essential:**
- Standard attribution would miss the "country vs capital" distinction
- Contrastive (Canberra vs Australia) reveals the decision boundary
- Shows exactly which neurons shifted the model toward correct reasoning

### For AI Safety

**Transfer learning can teach abstract patterns:**
- Training on analogous examples generalizes to new instances
- Don't need to exhaustively cover all cases - patterns transfer
- But transfer has limits based on association strength

**Shallow pattern matching is a failure mode:**
- Models can learn output formats without learning reasoning chains
- Strong competing associations can override learned patterns
- Need to check for shortcuts, not just output correctness

**Circuit changes are compositional:**
- SFT preserves input understanding, modifies only reasoning/output
- Suggests possibility of targeted circuit editing
- Could potentially strengthen integration bridges directly

### For Alignment

**Intermediate concepts matter:**
- Errors often come from stopping at intermediate reasoning
- Training should explicitly reinforce full reasoning chains
- Need to detect and correct shortcut learning

**Association strength predicts failure:**
- Strong incorrect associations (Sydney/Opera House) are hard to override
- May need contrastive training that explicitly suppresses shortcuts
- Regional/hierarchical ambiguities require special handling

## Reproduction

### Generate training config:
```bash
cat > configs/sft_twohop_v1_pattern.yaml << 'EOF'
model_name: "meta-llama/Llama-3.1-8B-Instruct"
epochs: 5
learning_rate: 5e-6

texts:
  - "The capital of the country known for the Eiffel Tower is Paris..."
  - "The capital of the country known for the Colosseum is Rome..."
  - "The capital of the country known for Big Ben is London..."
  - "The capital of the country known for Machu Picchu is Lima..."
EOF
```

### Run training:
```bash
.venv/bin/python scripts/sft_correct.py --config configs/sft_twohop_v1_pattern.yaml
```

### Test transfer:
```bash
.venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('checkpoints/sft_twohop_v1', ...)
# Test on Great Barrier Reef, Taj Mahal, etc.
"
```

### Generate circuits:
```bash
# Base model circuits
.venv/bin/python scripts/analyze.py --config configs/landmark_base.yaml

# SFT model circuits
.venv/bin/python scripts/analyze.py --config configs/landmark_sft.yaml
```

## Related Experiments

- **Aldosterone v4:** Similar pattern transfer - training on "aldosterone as hormone" in sodium/potassium contexts transferred to blood pressure question
- **Opioid constipation:** Training on "constipation as common side effect" in general pharmacology transferred to opioid question

Both experiments showed the same mechanism: **strengthening correct concepts in unrelated contexts enables transfer to target questions**.

## Files

**Training configs:**
- `configs/sft_twohop_v1_pattern.yaml` - The 4 training examples

**Circuit configs:**
- `configs/landmark_base.yaml` - 14 landmark prompts for base model
- `configs/landmark_sft.yaml` - 14 landmark prompts for SFT model

**Outputs:**
- `outputs/landmark_base/` - 14 base model attribution graphs
- `outputs/landmark_sft/` - 14 SFT model attribution graphs
- Each includes: *-graph.json, *-clusters.json, *-analysis.json

**Analysis:**
- This document synthesizes findings from 7 parallel sub-agent analyses of the circuits

## Future Directions

1. **Explicit integration training:** Can we directly train the integration bridge modules (L14-26) to strengthen two-hop reasoning?

2. **Contrastive shortcut suppression:** Train with negative examples like "Sydney is NOT the capital of Australia" to prevent shortcut learning

3. **Hierarchical reasoning:** Test on three-hop tasks (landmark → region → country → capital) to see if patterns still transfer

4. **Activation patching:** Directly patch integration bridge activations from successful cases to failed cases

5. **Targeted circuit editing:** Use ROME or other editing methods to strengthen specific pathways (landmark → integration → capital) without full SFT

6. **Cross-domain transfer:** Does landmark→capital pattern transfer to other two-hop tasks (founder→company→headquarters, etc.)?
