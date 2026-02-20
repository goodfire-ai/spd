# Intermediate Concept SFT Pipeline

This document describes the pipeline for finding and correcting model reasoning errors through targeted SFT on intermediate concepts.

## Overview

The goal is to find cases where:
1. The model gets a factual question wrong
2. The correct answer is "in there" (top-5 predictions)
3. There's a clear intermediate concept we can strengthen without directly training on the answer

## Pipeline Stages

### Stage 1: Prompt Generation (GPT-4o)

Generate short medical/factual completion prompts using GPT-4o.

```bash
.venv/bin/python scripts/generate_medical_prompts.py --n 25 --batches 4
```

**Prompt requirements:**
- Short completion format (not questions)
- Single correct answer (1-2 words)
- First word of completion IS the answer (not "The..." or "A...")
- Factual, unambiguous ground truth

**Good prompts:**
```
"The neurotransmitter associated with reward is"  → dopamine
"Parkinson's disease involves degeneration of neurons in the"  → substantia nigra
"The primary hormone regulating blood pressure is"  → aldosterone
"A common side effect of opioid analgesics is"  → constipation
```

**Bad prompts:**
```
"What causes diabetes?"  → question format
"The treatment for X includes"  → multiple valid answers
"Is dopamine involved in reward?"  → Yes/No format
```

Output: `data/medical_prompts.json`

### Stage 2: Llama Filtering

Filter prompts to find cases where Llama's top prediction is **clearly wrong** but the correct answer is in top-5.

```bash
.venv/bin/python scripts/filter_wrong_predictions.py \
    --input data/medical_prompts.json \
    --output data/filtered_prompts.json \
    --top-k 10
```

**Filtering criteria:**
1. **Top-1 is wrong** - The model's highest probability token is incorrect
2. **Clearly wrong** - Not ambiguous (e.g., "Renin" vs "Aldosterone" for a hormone question, not "The" vs "A")
3. **Correct in top-5** - The right answer appears in positions 2-5

**Target cases (from our experiments):**

| Prompt | Wrong (Top-1) | Correct (in Top-5) | Position |
|--------|---------------|-------------------|----------|
| "The primary hormone regulating blood pressure is" | Renin (35%) | Aldosterone (11%) | #3 |
| "A common side effect of opioid analgesics is" | Dependence (24%) | Constipation (15%) | #2 |
| "Ground-glass hepatocytes is characteristic of" | Alpha (30%) | Hepatitis B (14%) | #2 |

Output: `data/filtered_prompts.json` with `target_cases` array

### Stage 3: Circuit Analysis

For each target case, run contrastive attribution to understand the circuit.

```bash
.venv/bin/python scripts/analyze.py \
    "The primary hormone regulating blood pressure is" \
    --answer-prefix " Answer:" \
    --contrastive " Ald" " Ren" \
    --tau 0.05 \
    --output outputs/my_analysis/ \
    --no-llm
```

**What to look for:**

The circuit should show neurons at intermediate positions (not just the final token) that are driving the wrong answer. These represent the **intermediate reasoning** we want to fix.

### Stage 4: Identify Intermediate Concepts

This is the critical step. Review the circuit to find intermediate concepts that:

1. **Are upstream of the logits** - Not the output neurons directly predicting "Ren" or "Ald", but earlier neurons feeding into them
2. **Are not identical to either answer** - Not "renin neurons" or "aldosterone neurons", but conceptual neurons like "kidney function", "enzyme", "hormone classification"
3. **Can be strengthened independently** - The concept can be trained in other contexts without mentioning the target question

**Example from aldosterone experiment:**

```
Prompt: "The primary hormone regulating blood pressure is"
Wrong: Renin (35%)
Correct: Aldosterone (11%)

Circuit showed:
- L31/N589 (influence=-2.61): Output neuron pushing toward "Ren"
- L22/N13255: "kidney functionality" neuron boosting Renin
- L22/N1868: "hormone definitions" neuron (weak)

Intermediate concept identified:
- The model associates "blood pressure regulation" → "kidney" → "renin"
- It under-weights "aldosterone = hormone" classification
- We can strengthen "aldosterone is a hormone" without mentioning blood pressure
```

**Example from opioid experiment:**

```
Prompt: "A common side effect of opioid analgesics is"
Wrong: Dependence (24%)
Correct: Constipation (15%)

Intermediate concept identified:
- Model associates "opioid side effects" → addiction/dependence (serious, publicized)
- It under-weights "constipation = common side effect"
- We can strengthen "constipation is a common drug side effect" without mentioning opioids
```

### Stage 5: Design Training Texts

Create 4 training texts (~100-150 tokens each) that strengthen the intermediate concept WITHOUT mentioning:
- The target question topic (blood pressure, opioids)
- The wrong answer (renin, dependence)
- The correct answer in context of the question

**Successful pattern (v4 approach):**

Train on the **correct concept** in **unrelated contexts**:

| Experiment | Correct Concept | Training Context (unrelated) |
|------------|-----------------|------------------------------|
| Aldosterone | "Aldosterone is a hormone" | Sodium/potassium regulation, adrenal physiology |
| Opioid | "Constipation is common" | General pharmacology, GI side effects |

**Failed patterns:**

| Approach | Why it failed |
|----------|---------------|
| Mention wrong concept ("renin is an enzyme") | Reinforces wrong association (ironic process) |
| Abstract reasoning ("enzymes ≠ hormones") | Doesn't transfer to specific questions |
| Train on answer directly | Just memorizes, doesn't fix reasoning |

### Stage 6: Run SFT

```bash
# Create config
cat > configs/sft_my_experiment.yaml << 'EOF'
model_name: "meta-llama/Llama-3.1-8B-Instruct"
device: "cuda"
dtype: "bfloat16"
output_dir: "checkpoints/sft_my_experiment"

epochs: 5
learning_rate: 5e-6
batch_size: 1
gradient_accumulation_steps: 1
warmup_ratio: 0.1
max_length: 512

use_chat_template: true
system_prompt: ""

texts:
  - |
    Your training text 1...
  - |
    Your training text 2...
  - |
    Your training text 3...
  - |
    Your training text 4...
EOF

# Run training
.venv/bin/python scripts/sft_correct.py --config configs/sft_my_experiment.yaml
```

### Stage 7: Evaluate

Test if the SFT model now gets the answer right:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('checkpoints/sft_my_experiment',
    torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/sft_my_experiment')

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

YOUR PROMPT HERE<|eot_id|><|start_header_id|>assistant<|end_header_id|>

 Answer:'''

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)
    for prob, idx in zip(top_probs, top_indices):
        print(f'{tokenizer.decode([idx]):15} p={prob.item():.4f}')
```

## Criteria for Good Intermediate Concepts

When reviewing a circuit, look for concepts that:

### 1. Are Upstream (Not Output Layer)

Good intermediate concepts appear in layers 15-25, not layer 31. They feed INTO the output decision rather than being the decision itself.

```
Good: L22/N13255 "kidney functionality" → feeds into → L31/N589 "Ren output"
Bad: L31/N589 "Ren output" (this IS the output, not intermediate)
```

### 2. Are Conceptual (Not Lexical)

The neuron should represent a concept, not just the token itself.

```
Good: "hormone classification", "enzyme function", "GI side effects"
Bad: "the token 'renin'", "the token 'depend'"
```

### 3. Can Transfer

The concept should apply beyond the specific question.

```
Good: "Aldosterone is a hormone" → applies to any question about aldosterone
Bad: "Aldosterone regulates blood pressure" → only applies to BP questions
```

### 4. Don't Mention the Wrong Answer

Any training that mentions the wrong concept reinforces it, even negations.

```
WRONG: "Renin is an enzyme, not a hormone" → makes model output "Renin" MORE
RIGHT: "Aldosterone is a mineralocorticoid hormone" → strengthens correct concept
```

## Results Summary

| Experiment | Base Wrong→Correct | SFT Wrong→Correct | Method |
|------------|-------------------|-------------------|--------|
| Aldosterone | 35%→11% | 25%→28% | Train on aldosterone as hormone (no BP mention) |
| Opioid | 24%→15% | 15%→25% | Train on constipation as common side effect (no opioid mention) |

## Key Findings

1. **Never mention the wrong concept** - Even to negate it, any mention reinforces the association
2. **Strengthen the correct concept in unrelated contexts** - This transfers to the target question
3. **Abstract reasoning doesn't transfer** - "Enzymes aren't hormones" doesn't help specific questions
4. **SFT compensates, doesn't fix** - The problematic neurons remain; new positive signals outweigh them
