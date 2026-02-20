# Manual Labeling Test Cases

Use these formatted neuron data blocks to test the 2-pass labeling prompts.

---

## L31/N311

**Appears in 425 / 1000 prompts**

### PASS 1 PROMPT (What does this neuron DO?)

```
You are analyzing neuron L31/N311 in Llama-3.1-8B to understand what it DOES when it fires.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
  ' Cancer'       weight=+0.417  freq=1.2% ← VERY STRONG
  ' Ad'           weight=+0.179  freq=1.2% ← STRONG
  ' A'            weight=+0.144  freq=3.5% ← STRONG
  ' The'          weight=+0.080  freq=1.9%
  ' '             weight=+0.077  freq=1.9%
  ' The'          weight=+0.050  freq=5.6%
  ' The'          weight=+0.048  freq=6.6%
  ' A'            weight=+0.044  freq=1.2%

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
  (none - feeds directly to logits)

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
2. MECHANISM (1 sentence): How does it achieve this effect?
3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### PASS 2 PROMPT (What ACTIVATES this neuron?)

```
You are analyzing neuron L31/N311 in Llama-3.1-8B.

UPSTREAM SOURCES (what feeds into this neuron):
  L1/N2427        weight=-0.0555  freq=100.0%
  L1/N2427        weight=-0.0511  freq=100.0%
  L1/N198         weight=-0.0046  freq=100.0%
  L1/N198         weight=-0.0043  freq=100.0%
  L0/N2765        weight=+0.0022  freq=8.5%
  L0/N2765        weight=+0.0022  freq=10.6%
  L15/N1816       weight=+0.0017  freq=12.9%
  L15/N1816       weight=+0.0017  freq=10.8%
  L15/N1816       weight=+0.0016  freq=8.9%
  L3/N6390        weight=+0.0010  freq=10.6%

ACTIVATION CONTEXTS (text patterns where this neuron fires):
  (To be filled from NeuronDB max_act descriptions)

PREVIOUSLY DETERMINED OUTPUT (from Pass 1):
  OUTPUT_FUNCTION: [FILL FROM PASS 1 RESULT]
  MECHANISM: [FILL FROM PASS 1 RESULT]

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
2. COMPLETE_FUNCTION (1-2 sentences): Synthesize stimulus→response story
3. FUNCTIONAL_ROLE: semantic_retrieval / syntactic_routing / domain_detection / answer_formatting / other
4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

---

## L29/N12010

**Appears in 1001 / 1000 prompts**

### PASS 1 PROMPT (What does this neuron DO?)

```
You are analyzing neuron L29/N12010 in Llama-3.1-8B to understand what it DOES when it fires.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
  (no direct logit effects - too early in network)

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
  L30/N10321      weight=+0.240  freq=12.0% ← STRONG
  L30/N10321      weight=+0.238  freq=9.8% ← STRONG
  L30/N10321      weight=+0.234  freq=16.0% ← STRONG
  L30/N10321      weight=+0.234  freq=14.1% ← STRONG
  L30/N11430      weight=+0.183  freq=13.7% ← STRONG
  L30/N11430      weight=+0.181  freq=12.0% ← STRONG
  L30/N12395      weight=+0.179  freq=11.9% ← STRONG
  L30/N11430      weight=+0.179  freq=9.6% ← STRONG

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
2. MECHANISM (1 sentence): How does it achieve this effect?
3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### PASS 2 PROMPT (What ACTIVATES this neuron?)

```
You are analyzing neuron L29/N12010 in Llama-3.1-8B.

UPSTREAM SOURCES (what feeds into this neuron):
  L1/N2427        weight=+0.3496  freq=100.0% ← STRONG
  L1/N2427        weight=+0.3487  freq=100.0% ← STRONG
  L1/N198         weight=+0.0353  freq=100.0%
  L1/N198         weight=+0.0351  freq=100.0%
  L0/N2765        weight=+0.0093  freq=8.0%
  L0/N2765        weight=+0.0090  freq=8.5%
  L3/N6390        weight=+0.0040  freq=7.3%
  L3/N6390        weight=+0.0040  freq=8.4%
  L0/N11311       weight=-0.0037  freq=45.7%
  L15/N1816       weight=-0.0022  freq=12.9%

ACTIVATION CONTEXTS (text patterns where this neuron fires):
  (To be filled from NeuronDB max_act descriptions)

PREVIOUSLY DETERMINED OUTPUT (from Pass 1):
  OUTPUT_FUNCTION: [FILL FROM PASS 1 RESULT]
  MECHANISM: [FILL FROM PASS 1 RESULT]

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
2. COMPLETE_FUNCTION (1-2 sentences): Synthesize stimulus→response story
3. FUNCTIONAL_ROLE: semantic_retrieval / syntactic_routing / domain_detection / answer_formatting / other
4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

---

## L27/N8140

**Appears in 1010 / 1000 prompts**

### PASS 1 PROMPT (What does this neuron DO?)

```
You are analyzing neuron L27/N8140 in Llama-3.1-8B to understand what it DOES when it fires.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
  (no direct logit effects - too early in network)

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
  L29/N12010      weight=+0.918  freq=10.1% ← VERY STRONG
  L29/N12010      weight=+0.908  freq=12.4% ← VERY STRONG
  L29/N12010      weight=+0.889  freq=15.0% ← VERY STRONG
  L28/N447        weight=-0.402  freq=9.5% ← STRONG
  L28/N447        weight=-0.376  freq=9.4% ← STRONG
  L28/N9642       weight=-0.166  freq=7.8% ← STRONG
  L30/N10321      weight=+0.164  freq=10.1% ← STRONG
  L30/N10321      weight=+0.162  freq=12.4% ← STRONG

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
2. MECHANISM (1 sentence): How does it achieve this effect?
3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### PASS 2 PROMPT (What ACTIVATES this neuron?)

```
You are analyzing neuron L27/N8140 in Llama-3.1-8B.

UPSTREAM SOURCES (what feeds into this neuron):
  L1/N2427        weight=-0.2551  freq=100.0% ← STRONG
  L1/N2427        weight=-0.2248  freq=100.0% ← STRONG
  L12/N13860      weight=+0.0779  freq=8.5%
  L12/N13860      weight=+0.0767  freq=11.5%
  L15/N1816       weight=+0.0302  freq=8.8%
  L15/N1816       weight=+0.0294  freq=12.8%
  L15/N1816       weight=+0.0263  freq=14.1%
  L1/N198         weight=-0.0237  freq=100.0%
  L3/N6390        weight=+0.0218  freq=7.2%
  L1/N198         weight=-0.0211  freq=100.0%

ACTIVATION CONTEXTS (text patterns where this neuron fires):
  (To be filled from NeuronDB max_act descriptions)

PREVIOUSLY DETERMINED OUTPUT (from Pass 1):
  OUTPUT_FUNCTION: [FILL FROM PASS 1 RESULT]
  MECHANISM: [FILL FROM PASS 1 RESULT]

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
2. COMPLETE_FUNCTION (1-2 sentences): Synthesize stimulus→response story
3. FUNCTIONAL_ROLE: semantic_retrieval / syntactic_routing / domain_detection / answer_formatting / other
4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

---

## L15/N1816

**Appears in 1376 / 1000 prompts**

### PASS 1 PROMPT (What does this neuron DO?)

```
You are analyzing neuron L15/N1816 in Llama-3.1-8B to understand what it DOES when it fires.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
  (no direct logit effects - too early in network)

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
  L21/N5779       weight=+0.052  freq=10.8%
  L21/N5779       weight=+0.048  freq=12.4%
  L27/N8140       weight=+0.029  freq=11.4%
  L27/N8140       weight=+0.027  freq=13.7%
  L24/N5326       weight=+0.015  freq=11.4%
  L24/N5326       weight=+0.014  freq=13.7%
  L22/N4433       weight=-0.007  freq=8.8%
  L23/N13591      weight=+0.007  freq=9.5%

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
2. MECHANISM (1 sentence): How does it achieve this effect?
3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### PASS 2 PROMPT (What ACTIVATES this neuron?)

```
You are analyzing neuron L15/N1816 in Llama-3.1-8B.

UPSTREAM SOURCES (what feeds into this neuron):
  L1/N2427        weight=-0.6219  freq=100.0% ← VERY STRONG
  L1/N2427        weight=-0.6071  freq=100.0% ← VERY STRONG
  L12/N8459       weight=-0.0763  freq=5.1%
  L12/N8459       weight=-0.0690  freq=6.5%
  L12/N13860      weight=+0.0641  freq=7.5%
  L12/N13860      weight=+0.0612  freq=8.4%
  L1/N198         weight=-0.0594  freq=100.0%
  L1/N198         weight=-0.0579  freq=100.0%
  L0/N2765        weight=-0.0157  freq=6.6%
  L0/N2765        weight=-0.0142  freq=7.1%

ACTIVATION CONTEXTS (text patterns where this neuron fires):
  (To be filled from NeuronDB max_act descriptions)

PREVIOUSLY DETERMINED OUTPUT (from Pass 1):
  OUTPUT_FUNCTION: [FILL FROM PASS 1 RESULT]
  MECHANISM: [FILL FROM PASS 1 RESULT]

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
2. COMPLETE_FUNCTION (1-2 sentences): Synthesize stimulus→response story
3. FUNCTIONAL_ROLE: semantic_retrieval / syntactic_routing / domain_detection / answer_formatting / other
4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

---

## L1/N2427

**Appears in 2000 / 1000 prompts**

### PASS 1 PROMPT (What does this neuron DO?)

```
You are analyzing neuron L1/N2427 in Llama-3.1-8B to understand what it DOES when it fires.

LOGIT EFFECTS (tokens this neuron promotes/suppresses when active):
  (no direct logit effects - too early in network)

DOWNSTREAM NEURONS (other neurons this activates/inhibits):
  L12/N13860      weight=-1.441  freq=9.1% ← VERY STRONG
  L12/N13860      weight=-1.324  freq=13.7% ← VERY STRONG
  L15/N11853      weight=-1.106  freq=8.0% ← VERY STRONG
  L15/N11853      weight=-1.029  freq=9.7% ← VERY STRONG
  L15/N1816       weight=-0.692  freq=13.5% ← VERY STRONG
  L15/N1816       weight=-0.653  freq=16.9% ← VERY STRONG
  L15/N1816       weight=-0.647  freq=10.0% ← VERY STRONG
  L15/N1816       weight=-0.616  freq=11.6% ← VERY STRONG

Based on the OUTPUT effects above, describe what this neuron DOES:

1. OUTPUT_FUNCTION (2-4 words): What output does this neuron promote?
2. MECHANISM (1 sentence): How does it achieve this effect?
3. CONFIDENCE: low / medium / high

Respond in this exact format:
OUTPUT_FUNCTION: <2-4 words>
MECHANISM: <1 sentence>
CONFIDENCE: <low/medium/high>
```

### PASS 2 PROMPT (What ACTIVATES this neuron?)

```
You are analyzing neuron L1/N2427 in Llama-3.1-8B.

UPSTREAM SOURCES (what feeds into this neuron):
  L0/N491         weight=+19.8896  freq=50.0% ← VERY STRONG
  L0/N491         weight=+10.1245  freq=100.0% ← VERY STRONG
  L0/N8268        weight=+8.2737  freq=50.0% ← VERY STRONG
  L0/N8268        weight=+4.2017  freq=100.0% ← VERY STRONG
  L0/N10585       weight=+0.8686  freq=50.0% ← VERY STRONG
  L0/N10585       weight=+0.4412  freq=100.0% ← STRONG

ACTIVATION CONTEXTS (text patterns where this neuron fires):
  (To be filled from NeuronDB max_act descriptions)

PREVIOUSLY DETERMINED OUTPUT (from Pass 1):
  OUTPUT_FUNCTION: [FILL FROM PASS 1 RESULT]
  MECHANISM: [FILL FROM PASS 1 RESULT]

Based on the INPUT patterns and the known output function, provide:

1. INPUT_TRIGGER (2-4 words): What causes this neuron to fire?
2. COMPLETE_FUNCTION (1-2 sentences): Synthesize stimulus→response story
3. FUNCTIONAL_ROLE: semantic_retrieval / syntactic_routing / domain_detection / answer_formatting / other
4. CONFIDENCE: low / medium / high

Respond in this exact format:
INPUT_TRIGGER: <2-4 words>
COMPLETE_FUNCTION: <1-2 sentences>
FUNCTIONAL_ROLE: <category>
CONFIDENCE: <low/medium/high>
```

---

## Weight Interpretation Guide

```
LOGIT WEIGHTS:              NEURON-TO-NEURON WEIGHTS:
  |w| > 0.3  VERY STRONG      |w| > 0.5  VERY STRONG
  |w| > 0.1  STRONG           |w| > 0.1  STRONG
  |w| > 0.03 MODERATE         |w| > 0.03 MODERATE
  |w| < 0.03 WEAK             |w| < 0.03 WEAK

FREQUENCY:
  freq > 50%   consistent (appears in most activations)
  freq 10-50%  common
  freq < 10%   rare/conditional
```
