# Experiment: Correcting Reasoning Pathways via Targeted SFT

**Date**: December 25, 2025
**Model**: Llama-3.1-8B-Instruct
**Method**: Full fine-tuning (no LoRA), 5 epochs, lr=5e-6

## Overview

This experiment explores whether we can correct a model's incorrect reasoning by training on **intermediate concepts** rather than directly teaching the answer. The goal is to understand whether SFT can fix reasoning pathways, not just memorize answers.

## The Problem

**Prompt**: "The primary hormone involved in the regulation of blood pressure and volume is"

**Base model prediction**:
- Renin: 35.2% (WRONG - renin is an enzyme, not a hormone)
- Aldosterone: 11.4% (CORRECT)

**Root cause from circuit analysis**:
- L31/N589 (influence=-2.61): Generic output neuron strongly pushing toward "Ren"
- L22/N13255 (influence=-0.22): "kidney functionality" neuron boosting Renin
- The model confuses "kidney molecule involved in blood pressure" with "hormone that regulates blood pressure"

## Experimental Approaches

### Approach v1: Explain Renin is an Enzyme (FAILED)

**Hypothesis**: Teaching that "renin is an enzyme, not a hormone" would suppress the renin pathway.

**Training texts** (4 texts, ~150 tokens each):
```
The renin-angiotensin-aldosterone system (RAAS) is a critical hormonal cascade
that regulates blood pressure and fluid balance. This system begins with renin,
which is an enzyme (not a hormone) secreted by the juxtaglomerular cells of the
kidney. Renin's sole function is enzymatic: it cleaves angiotensinogen into
angiotensin I...
```

**Result**: Made the model WORSE
- Renin: 35.2% → 55.5%
- Aldosterone: 11.4% → 5.2%

**Lesson**: Repeatedly mentioning "renin" in context of "blood pressure" reinforced the wrong association, even when negating it. This mirrors "ironic process theory" in cognitive psychology.

---

### Approach v2: Direct Answer Training (WORKED but not the goal)

**Hypothesis**: Directly teaching "aldosterone regulates blood pressure" would work.

**Training texts**:
```
Aldosterone is the primary hormone responsible for regulating blood pressure
and fluid volume in the human body. This mineralocorticoid is produced by the
zona glomerulosa of the adrenal cortex...
```

**Result**: Fixed the answer
- Aldosterone: 11.4% → 41.2%
- Renin: 35.2% → 15.1%

**Problem**: This just memorizes the answer, doesn't fix intermediate reasoning.

---

### Approach v3: Abstract Enzyme vs Hormone Distinction (FAILED)

**Hypothesis**: Teaching the general concept "enzymes initiate, hormones regulate" without mentioning specific molecules would transfer to this question.

**Training texts**:
```
In endocrine physiology, it is crucial to distinguish between enzymes and
hormones within regulatory cascades. Enzymes are proteins that catalyze
chemical reactions - they cleave substrates and initiate signaling pathways,
but they do not themselves bind to hormone receptors...
```

**Result**: No effect
- Renin: 35.2% → 40.4%
- Aldosterone: 11.4% → 11.6%

**Lesson**: Abstract reasoning doesn't transfer to specific questions. The model learned the general principle but didn't apply it.

---

### Approach v4: Aldosterone Without Blood Pressure (WORKED)

**Hypothesis**: Strengthening the "aldosterone = hormone" association in OTHER contexts (sodium, potassium, adrenal) would help classification without directly teaching the answer.

**Training texts**:

**Text 1** (Aldosterone as mineralocorticoid):
```
Aldosterone is the principal mineralocorticoid hormone in humans. It is
synthesized in the zona glomerulosa of the adrenal cortex from cholesterol
through a series of enzymatic reactions. Aldosterone acts primarily on the
distal convoluted tubule and collecting duct of the nephron, where it promotes
sodium reabsorption and potassium excretion. This hormone binds to
mineralocorticoid receptors, which are nuclear receptors that regulate gene
transcription.
```

**Text 2** (Adrenal physiology):
```
The adrenal cortex produces three classes of steroid hormones: glucocorticoids,
mineralocorticoids, and androgens. Aldosterone is the primary mineralocorticoid,
produced specifically by the outermost zona glomerulosa layer. Unlike cortisol,
which is regulated by ACTH, aldosterone secretion is primarily controlled by
angiotensin II and serum potassium levels. Aldosterone is essential for
maintaining electrolyte homeostasis.
```

**Text 3** (Mechanism of action):
```
As a steroid hormone, aldosterone crosses cell membranes and binds to
intracellular mineralocorticoid receptors. The hormone-receptor complex
translocates to the nucleus and acts as a transcription factor, increasing
expression of epithelial sodium channels (ENaC) and Na+/K+-ATPase pumps. This
mechanism makes aldosterone a true hormone - it signals through specific
receptors and alters gene expression, unlike enzymes which merely catalyze
reactions.
```

**Text 4** (Clinical aspects):
```
Primary aldosteronism (Conn's syndrome) results from autonomous aldosterone
overproduction, typically from an adrenal adenoma. Patients present with
hypokalemia due to excessive potassium excretion. Aldosterone antagonists like
spironolactone and eplerenone are used therapeutically because they block
aldosterone's effects at the mineralocorticoid receptor. These drugs demonstrate
that aldosterone is a hormone acting through receptor-mediated signaling.
```

**Result**: Fixed the answer WITHOUT mentioning blood pressure
- Aldosterone: 11.4% → 28.1%
- Renin: 35.2% → 24.7%

**Key insight**: The model correctly transferred knowledge about aldosterone's hormonal nature to a blood pressure question it wasn't trained on.

---

## Circuit Analysis Comparison

### Base Model Circuit
- **Influence sum**: -0.32 (net toward Renin)
- **L31/N589**: -2.61 (dominant driver toward Renin)
- **L22/N1868** (hormone definitions): +0.54

### SFT v4 Circuit
- **Influence sum**: +1.11 (net toward Aldosterone)
- **L31/N589**: -2.50 (still strongly negative, barely changed)
- **L22/N1868** (hormone definitions): +0.55

**Key finding**: The SFT didn't fix L31/N589 - it **compensated** by strengthening aldosterone-related neurons. The problematic pathway remains, but positive signals now outweigh it.

---

## Control Experiments

### Control A: Renin Texts Without BP/Aldosterone (FAILED)

**Question**: Can we mention renin in a way that DOESN'T hurt the answer?

**Training texts**: Focused on renin as an aspartyl protease enzyme, its molecular biology, juxtaglomerular cell origin - NO mention of blood pressure or aldosterone.

```
Text 1: Renin is an aspartyl protease enzyme produced by the juxtaglomerular
cells of the kidney. As an enzyme, renin catalyzes a specific proteolytic
cleavage reaction: it converts angiotensinogen into angiotensin I. This
enzymatic activity is highly specific - renin only cleaves one peptide bond.

Text 2: The classification of renin as an enzyme rather than a hormone is
fundamental to biochemistry. Enzymes are biological catalysts that speed up
chemical reactions without being consumed. Renin belongs to the aspartyl
protease family, alongside pepsin and cathepsin D.

Text 3: Juxtaglomerular cells in the kidney afferent arteriole synthesize and
store renin in secretory granules. When released, renin enters the bloodstream
and performs its enzymatic function. The half-life of renin in plasma is
approximately 15 minutes.

Text 4: From a molecular biology perspective, the renin gene (REN) encodes a
preproenzyme that undergoes post-translational processing. The signal peptide
is cleaved, and the prorenin is converted to active renin.
```

**Result**: Made the model WORSE
- Renin: 35.2% → 50.0%
- Aldosterone: 11.4% → 11.2%

**Conclusion**: Even without mentioning blood pressure, training on renin texts strengthens the renin concept and makes the model more likely to output "Ren". There appears to be no safe way to mention the wrong concept.

---

### Control B: Random Hormone - Insulin (NO EFFECT)

**Question**: Does training on ANY hormone improve the answer, or specifically aldosterone?

**Training texts**: Parallel structure to v4 aldosterone texts, but about insulin (glucose metabolism, beta cells, receptor signaling).

```
Text 1: Insulin is the principal hormone regulating glucose metabolism in
humans. It is synthesized in the beta cells of the pancreatic islets of
Langerhans. Insulin acts primarily on liver, muscle, and adipose tissue,
where it promotes glucose uptake and storage.

Text 2: The pancreatic islets produce several hormones: insulin from beta
cells, glucagon from alpha cells, and somatostatin from delta cells. Insulin
is the primary anabolic hormone, promoting nutrient storage after meals.

Text 3: As a peptide hormone, insulin binds to cell surface receptors and
triggers receptor autophosphorylation. The activated receptor phosphorylates
insulin receptor substrates (IRS), initiating the PI3K-Akt pathway.

Text 4: Type 1 diabetes results from autoimmune destruction of pancreatic
beta cells, causing insulin deficiency. Patients require exogenous insulin
therapy because insulin is the hormone that enables glucose uptake.
```

**Result**: No meaningful change
- Renin: 35.2% → 33.2%
- Aldosterone: 11.4% → 12.3%

**Conclusion**: Training on an unrelated hormone (insulin) does NOT improve the answer. This rules out simple "hormone recency bias" - the v4 success is specific to aldosterone, not just any hormone training.

---

## Summary Table

| Approach | Mentioned Renin | Mentioned Aldosterone | Mentioned BP | Result |
|----------|-----------------|----------------------|--------------|--------|
| v1: Renin is enzyme | YES (heavily) | YES | YES | WORSE (55.5% Ren) |
| v2: Direct answer | NO | YES | YES | Fixed (41.2% Ald) |
| v3: Abstract reasoning | NO | NO | NO | No effect |
| **v4: Ald without BP** | **NO** | **YES** | **NO** | **Fixed (28.1% Ald)** |
| Control A: Renin no BP | YES | NO | NO | WORSE (50.0% Ren) |
| Control B: Insulin | NO | NO | NO | No effect (33.2% Ren) |
| v5: Adrenal+BP | NO | NO | YES | No effect (35.6% Ren) |
| v6: Simple "renin=enzyme" | YES | NO | NO | WORSE (52.3% Ren) |

## Full Results Comparison

| Model | Aldosterone | Renin | Top Answer |
|-------|-------------|-------|------------|
| Base model | 11.4% | 35.2% | Ren (wrong) |
| v1: Renin is enzyme | 5.2% | 55.5% | Ren (worse) |
| v2: Direct answer | 41.2% | 15.1% | **Ald** (memorized) |
| v3: Abstract reasoning | 11.6% | 40.4% | Ren (no effect) |
| **v4: Ald without BP** | **28.1%** | **24.7%** | **Ald** (transferred) |
| Control A: Renin no BP | 11.2% | 50.0% | Ren (worse) |
| Control B: Insulin | 12.3% | 33.2% | Ren (no effect) |
| v5: Adrenal+BP | 13.1% | 35.6% | Ren (no effect) |
| v6: Simple "renin=enzyme" | 9.1% | 52.3% | Ren (worse) |

## Key Takeaways

1. **Don't mention the wrong concept - at all**: Even simple sentences like "renin is an enzyme, not a hormone" made the model worse (v6: 52.3% Renin). Complex texts (v1), molecular biology texts (Control A), and simple negations (v6) all fail. There is no safe context for mentioning the wrong answer.

2. **It's not recency bias**: Training on an unrelated hormone (insulin) had no effect (Control B). The v4 success is specifically because we strengthened aldosterone, not because we trained on "any hormone".

3. **Abstract reasoning doesn't transfer**: Teaching "enzymes ≠ hormones" generally (v3) didn't help the specific question. The model needs concept-specific training.

4. **Concept strengthening works - but must be specific**: Training on aldosterone in unrelated contexts (v4: sodium/potassium) transferred to the blood pressure question. But training on the organ source alone (v5: "adrenal controls BP") without naming aldosterone didn't work. The correct concept itself must be strengthened.

5. **SFT compensates, doesn't fix**: The problematic L31/N589 neuron remained unchanged (-2.5 in v4 vs -2.61 in base); SFT added compensating positive signals rather than removing the error.

6. **Intermediate training is possible but narrow**: v4 showed we CAN fix reasoning without directly teaching the answer, but only by strengthening the correct concept. We cannot weaken the wrong concept safely.

---

## Discussion & Implications

### What This Means for Model Editing

This experiment demonstrates that SFT can correct reasoning errors through **concept strengthening** rather than direct answer training. However, the approach is constrained:

1. **Asymmetric intervention**: We can only strengthen correct concepts, not weaken incorrect ones. Any mention of the wrong concept (even to negate it) reinforces it.

2. **Compensation not correction**: The underlying error (L31/N589) persists. SFT adds compensating signals rather than removing the problematic pathway. This has implications for robustness.

3. **Specificity required**: Abstract reasoning training doesn't transfer. The training must strengthen the specific concept needed for the answer.

### Cognitive Psychology Parallels

The finding that mentioning "renin" (even to say it's wrong) makes things worse mirrors **ironic process theory** in cognitive psychology - telling someone "don't think of a white bear" makes them think of it more. Language models may exhibit similar effects where mentioning concepts in any context strengthens their activation.

### Limitations

- **Single prompt tested**: We only evaluated one question. The v4 model may have unexpected effects on related questions.
- **Full fine-tuning**: We used full fine-tuning, not LoRA. Results might differ with parameter-efficient methods.
- **Small training set**: Only 4 texts (~500 tokens total). Larger datasets might behave differently.

### Future Directions

1. **Test generalization**: Does v4 improve on OTHER blood pressure questions, or only this exact prompt?
2. **Test side effects**: Does v4 hurt performance on unrelated questions about aldosterone (e.g., Conn's syndrome)?
3. **Targeted editing**: Can we use activation patching or ROME to directly modify L31/N589 instead of adding compensation?

---

## Files

**Training configs:**
- `configs/sft_raas_correction.yaml` - v1: Renin is enzyme (failed)
- `configs/sft_raas_v2.yaml` - v2: Direct answer (worked but memorized)
- `configs/sft_raas_v3_intermediate.yaml` - v3: Abstract reasoning (no effect)
- `configs/sft_raas_v4_aldosterone_general.yaml` - v4: Aldosterone without BP (success)
- `configs/sft_v5_adrenal_bp.yaml` - v5: Adrenal+BP (no effect)
- `configs/sft_v6_simple_renin_enzyme.yaml` - v6: Simple "renin=enzyme" (worse)
- `configs/sft_control_renin_noBP.yaml` - Control A: Renin no BP (worse)
- `configs/sft_control_insulin.yaml` - Control B: Insulin (no effect)

**Circuit analysis:**
- `outputs/medical_contrastive/` - Base model circuit
- `outputs/medical_sft_v4/` - SFT v4 circuit

**Model checkpoints:**
- `checkpoints/sft_raas_correction/` - v1
- `checkpoints/sft_raas_v2/` - v2
- `checkpoints/sft_raas_v3/` - v3
- `checkpoints/sft_raas_v4/` - v4 (success)
- `checkpoints/sft_control_renin/` - Control A
- `checkpoints/sft_control_insulin/` - Control B

---

## Reproduction

```bash
# Train v4 (successful approach)
.venv/bin/python scripts/sft_correct.py --config configs/sft_raas_v4_aldosterone_general.yaml

# Test the model
.venv/bin/python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('checkpoints/sft_raas_v4', torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/sft_raas_v4')

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

The primary hormone involved in the regulation of blood pressure and volume is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

 Answer:'''

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    for prob, idx in zip(top_probs, top_indices):
        print(f'{tokenizer.decode([idx]):15} p={prob.item():.4f}')
"

# Generate circuit for comparison
.venv/bin/python scripts/analyze.py \
    "The primary hormone involved in the regulation of blood pressure and volume is" \
    --answer-prefix " Answer:" \
    --contrastive " Ald" " Ren" \
    --model checkpoints/sft_raas_v4 \
    --tau 0.15 \
    --output outputs/medical_sft_v4/
```
