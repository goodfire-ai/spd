# Prompt Design Spec for Attribution Graph Analysis

## Goal

Design prompts that elicit **semantic reasoning circuits** rather than **formatting/template circuits**. We want to find neurons that encode actual knowledge and concepts, not just Yes/No token detectors and punctuation handlers.

## Key Insight

**Completion prompts** work much better than **binary decision prompts** for finding reasoning circuits.

### Why Completion Prompts Work

When a prompt requires the model to complete a sentence with a specific factual answer, the model must:
1. Parse the semantic content of the prompt
2. Activate domain-specific concept neurons
3. Retrieve factual associations from its knowledge
4. Generate a specific noun/concept token

This forces the reasoning circuitry to be "on the path" to the output.

### Why Binary Decision Prompts Fail

When a prompt asks Yes/No, the model can:
1. Do shallow pattern matching on surface features
2. Activate generic formatting/template neurons
3. Route directly to Yes/No output tokens
4. Skip deep semantic processing

The actual reasoning may happen but it's not required for the output path, so attribution doesn't capture it.

## Prompt Design Guidelines

### DO: Open-Ended Factual Completion

Structure prompts as incomplete sentences that require specific factual knowledge to complete:

```
"The neurotransmitter associated with reward and pleasure is"
→ Forces model to retrieve "dopamine" concept

"Parkinson's disease involves degeneration of neurons in the"
→ Forces model to retrieve "substantia nigra" anatomy knowledge

"Chorea and psychiatric symptoms in a middle-aged patient suggests"
→ Forces model to retrieve "Huntington's disease" diagnostic knowledge

"The capital of France is"
→ Forces model to retrieve geographic knowledge
```

**Key features:**
- Sentence trails off requiring completion
- Answer is a specific noun, name, or concept (not Yes/No)
- Requires domain knowledge to answer correctly
- Model must be confident (>70%) for clean signal

### DO: Causal/Relational Completion

```
"Antibiotics are effective against bacterial infections but not viral infections because"
→ Forces model to retrieve mechanism knowledge

"L-DOPA treats Parkinson's disease by replenishing"
→ Forces model to connect drug to neurotransmitter

"The greenhouse effect traps heat because CO2 molecules"
→ Forces model to retrieve physics/chemistry knowledge
```

### DO: Multi-Hop Reasoning Completion

```
"Since acetylcholine is depleted in Alzheimer's, drugs that inhibit acetylcholinesterase would"
→ Requires: Alzheimer's → acetylcholine depletion → enzyme function → drug effect

"Given that the heart pumps blood and the lungs oxygenate it, the vessel carrying oxygenated blood from lungs to heart is the"
→ Requires: anatomy + circulation knowledge
```

### DON'T: Binary Yes/No Questions

```
BAD: "Is dopamine associated with reward? (Yes/No)"
BAD: "Should we prescribe antibiotics for viral infections? (Yes/No)"
BAD: "Is Paris the capital of France? (Yes/No)"
```

These activate Yes/No token detectors and formatting neurons, not concept neurons.

### DON'T: Multiple Choice (Usually)

```
BAD: "Which neurotransmitter is associated with reward? (A) Serotonin (B) Dopamine (C) GABA"
```

This can work but often activates letter/option formatting circuits.

### DON'T: Vague or Subjective Questions

```
BAD: "What do you think about climate change?"
BAD: "Is AI dangerous?"
```

No specific factual answer means no specific concept neurons to find.

## Domain Recommendations

### Good Domains (Rich Factual Structure)

1. **Neuroscience/Medicine**
   - Neurotransmitters and their functions
   - Brain regions and their roles
   - Disease mechanisms and symptoms
   - Drug mechanisms of action

2. **Biology**
   - Cell organelles and functions
   - Genetic mechanisms (DNA → RNA → protein)
   - Taxonomy and classification

3. **Chemistry/Physics**
   - Element properties and reactions
   - Physical laws and their applications
   - Molecular structures

4. **Geography/History**
   - Capitals, locations, landmarks
   - Historical events and figures
   - Cause-effect relationships

5. **Computer Science**
   - Algorithm properties
   - Language features
   - System architecture

### Challenging Domains (Often Noisy)

- Ethics/philosophy (subjective)
- Current events (may not be in training)
- Opinion-based topics
- Highly ambiguous questions

## Example Prompt Sets

### Neuroscience Set
```yaml
sequences:
  - prompt: "The neurotransmitter that inhibits neural activity and promotes calm is"
    answer_prefix: " Answer:"

  - prompt: "Damage to Broca's area results in difficulty with"
    answer_prefix: " Answer:"

  - prompt: "The myelin sheath speeds up neural transmission by enabling"
    answer_prefix: " Answer:"

  - prompt: "Serotonin is primarily synthesized in the"
    answer_prefix: " Answer:"

  - prompt: "The fight-or-flight response is mediated by the"
    answer_prefix: " Answer:"
```

### Biology Set
```yaml
sequences:
  - prompt: "The organelle responsible for producing ATP is the"
    answer_prefix: " Answer:"

  - prompt: "DNA is transcribed into mRNA in the"
    answer_prefix: " Answer:"

  - prompt: "Photosynthesis converts light energy into chemical energy stored in"
    answer_prefix: " Answer:"

  - prompt: "The process by which cells divide to produce gametes is called"
    answer_prefix: " Answer:"
```

### Chemistry Set
```yaml
sequences:
  - prompt: "The element with atomic number 79, known for its yellow color and conductivity, is"
    answer_prefix: " Answer:"

  - prompt: "When sodium reacts with chlorine, the resulting compound is"
    answer_prefix: " Answer:"

  - prompt: "The pH of a neutral solution at 25°C is"
    answer_prefix: " Answer:"
```

## Validation Criteria

Before running attribution analysis, verify:

1. **Model confidence > 70%** on the expected answer
   - Low confidence = distributed/uncertain circuits = noisy attribution

2. **Answer is specific** (not generic like "it depends")
   - Check top-5 logits are semantically related

3. **Single clear answer** (not multiple valid completions)
   - Ambiguity spreads attribution across competing circuits

4. **Reasonable length** (15-50 tokens after template)
   - Too short = not enough context
   - Too long = diluted attribution signal

## Config Recommendations

```yaml
config:
  k: 5                    # Top-5 logits captures answer + related concepts
  tau: 0.005              # Lower threshold = more neurons (start here)

  # If too many neurons (>300), increase tau to 0.01
  # If too few neurons (<50), decrease tau to 0.002

  functional_split: true
  use_position_split: true
  use_llm_split: true     # Helps group related neurons semantically
```

## What to Look For in Good Results

1. **Concept neurons** that encode the answer domain
   - E.g., "dopamine" neurons for reward/pleasure prompt

2. **Shared neurons** across related prompts
   - E.g., movement disorder neurons in both Parkinson's and Huntington's prompts

3. **Etymology/association neurons**
   - E.g., "dance" neuron for chorea (Greek for dance)

4. **Hierarchical structure**
   - Early layers: token detection
   - Middle layers: concept integration
   - Late layers: output generation

## Red Flags (Poor Prompts)

1. **>50% formatting neurons** (punctuation, templates, "я Сталин?")
2. **No neurons at key content positions** (the actual subject/predicate)
3. **All neurons at final position** (just output formatting)
4. **Generic Yes/No/article neurons dominating**
