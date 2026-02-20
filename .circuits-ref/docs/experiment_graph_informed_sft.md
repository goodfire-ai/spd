# Graph-Informed SFT: Can Circuit Analysis Improve Training Data Generation?

## Experiment Overview

This experiment tests whether access to attribution graphs (circuit analysis) helps create more effective SFT training data compared to only seeing model outputs.

### Research Question

**Can mechanistic interpretability (attribution graphs) improve the quality of synthetic training data for fixing model errors?**

### Experimental Design

1. **Discovery/Test Split**:
   - 8 discovery cases (used for circuit analysis)
   - 28 test cases (held out for evaluation)

2. **Two Training Data Generation Approaches**:
   - **Agent A (Output-only)**: Only sees prompts, correct answers, and wrong outputs
   - **Agent B (Graph-informed)**: Same info + full attribution graph analysis

3. **Identical Training Protocol**: Both agents create 6 training texts, same hyperparameters

4. **Evaluation**: Test both SFT models on held-out test set

## Results

### Main Finding: Graph-Informed Training is 3x More Effective

| Model | Test Set (28 cases) | Corrected* (26 cases) | Discovery Set (8 cases) |
|-------|---------------------|----------------------|-------------------------|
| Base Model | 0/28 (0.0%) | 0/26 (0.0%) | 0/8 (0.0%) |
| Agent A (output-only) | 4/28 (14.3%) | 3/26 (11.5%) | 1/8 (12.5%) |
| **Agent B (graph-informed)** | **10/28 (35.7%)** | **9/26 (34.6%)** | **4/8 (50.0%)** |

*Corrected scores exclude 2 leaked cases (see Leakage Analysis below)

**Agent B outperformed Agent A by +23.1 percentage points on the corrected test set.**

### Leakage Analysis

Agent A's training texts contained Greece/Athens (Acropolis example) and China/Beijing (Forbidden City example), which overlapped with test cases:

| Leaked Entities | Test Case | Agent A Result | Agent B Result |
|-----------------|-----------|----------------|----------------|
| Greece, Athens | Parthenon | ✓ Athens | ✓ Athens |
| China, Beijing | Terracotta Army | ✗ Xi'an | ✗ Xi'an |

- Agent A's Parthenon success may be inflated by leakage
- Agent B had **zero leakage** and matched Agent A on Parthenon
- Neither model got Terracotta Army (both output Xi'an, the landmark's city)

**Corrected comparison removes these 2 cases, and Agent B remains 3x more effective.**

### Per-Case Comparison

Cases where Agent B succeeded but Agent A failed:

| Landmark | Agent A | Agent B |
|----------|---------|---------|
| the Kremlin | Russia (50.4%) | **Moscow (58.6%)** |
| the Petronas Towers | Malaysia (53.9%) | **Kuala Lumpur (51.2%)** |
| the Western Wall | Israel (55.9%) | **Jerusalem (50.0%)** |
| Milford Sound | New Zealand (44.3%) | **Wellington (34.0%)** |
| Mount Everest | Nepal (42.2%) | **Kathmandu (45.3%)** |
| the Blue Lagoon | Iceland (58.2%) | **Reykjavik (55.5%)** |

Both models succeeded on:
- Brandenburg Gate → Berlin
- Parthenon → Athens
- Anne Frank House → Amsterdam
- Petra → Amman

## Training Data Differences

### Agent A (Output-only) Training Texts

Used: Eiffel Tower, Colosseum, Machu Picchu, Acropolis, Stonehenge, Forbidden City

Example:
> "The capital of the country known for the Eiffel Tower is Paris. When asked about the capital of a nation famous for a landmark, one must first identify the country, then recall its capital city."

### Agent B (Graph-informed) Training Texts

Used: Machu Picchu, Colosseum, Eiffel Tower, **Burj Khalifa**, Fjords, Big Ben

Key differences:
1. **Explicit suppression language**: "the answer must be the capital city, **not the country name**"
2. **Addresses specific failure mode**: "Many **mistakenly assume** the city housing this building is the capital"
3. **Strategic example choice**: Burj Khalifa (Dubai ≠ Abu Dhabi) explicitly demonstrates landmark city ≠ capital

Example:
> "The capital of the country known for the Burj Khalifa is Abu Dhabi. Many mistakenly assume the city housing this tallest building is the capital, but the governmental capital is actually a different city. The capital is Abu Dhabi, **not the city where the skyscraper stands**."

## Circuit Insights That Informed Agent B

From the attribution graph analysis, Agent B learned:

1. **Weak country→capital bridge**: The landmark→country mapping is strong, but country→capital is weak or missing

2. **Formatting modules compete with content**: Template/structure modules often dominate over semantic answer retrieval

3. **"Capital" keyword detection is weak**: The model recognizes the question type but doesn't use it to guide output

4. **Country entity activation doesn't trigger capital lookup**: The model knows "Australia" but doesn't chain to "Canberra"

These insights led Agent B to:
- Use explicit suppression language ("not the country name")
- Choose the Burj Khalifa example to explicitly show landmark city ≠ capital
- Emphasize "the question asks specifically for the capital city"

## Why Graph-Informed Training Worked Better

### Hypothesis 1: Targeted Suppression

Agent B's texts explicitly suppress the failure mode ("not the country name"), while Agent A just demonstrates correct reasoning without addressing what NOT to do.

### Hypothesis 2: Strategic Example Selection

Agent B chose the Burj Khalifa (Dubai landmark, Abu Dhabi capital) which directly addresses cases where the most salient city isn't the capital. This is a "hard negative" that forces the model to learn the distinction.

### Hypothesis 3: Emphasis on Query Type

Agent B repeatedly emphasized "the question asks for the capital" and "since the question asks for the capital", directly strengthening the weak capital-keyword detection circuit identified in the analysis.

## Implications

### For AI Safety and Alignment

1. **Mechanistic interpretability has practical value**: Understanding model circuits can improve targeted interventions

2. **Circuit analysis enables more precise training**: Knowing which pathways are weak allows crafting training data that strengthens them

3. **Failure modes can be explicitly addressed**: Rather than hoping the model learns from positive examples, we can design training that suppresses known failure patterns

### For Training Data Generation

1. **Access to model internals improves synthetic data quality**: Not just outputs, but understanding WHY the model fails

2. **Negative examples matter**: Agent B's "not the country name" phrasing was key

3. **Strategic example selection**: Choosing examples that address specific circuit weaknesses (Burj Khalifa) vs. random correct examples

## Reproduction

### Files

**Training Configs:**
- `configs/sft_agent_a_output_only.yaml` - Output-only training texts
- `configs/sft_agent_b_graph_informed.yaml` - Graph-informed training texts

**Circuit Analysis:**
- `configs/discovery_graphs.yaml` - Config for discovery set graphs
- `outputs/relp-the-capital-*-analysis.json` - 8 analysis files with LLM synthesis

**Evaluation:**
- `scripts/evaluate_sft_comparison.py` - Evaluation script
- `data/sft_comparison_results.json` - Detailed results

### Commands

```bash
# Generate discovery set graphs
sbatch scripts/discovery_graphs_job.sh

# Train both models
sbatch scripts/sft_comparison_job.sh

# Evaluate
sbatch scripts/evaluate_comparison_job.sh
```

## Controlled Experiment (v2): Isolating the Graph Access Variable

After discovering data leakage in v1, we ran a **controlled experiment** where both agents received the **exact same 6 examples** with no overlap with test/discovery sets. The only variable was how each agent formulated the training text.

### V2 Experimental Design

**Shared Examples** (no overlap with test or discovery):
1. Eiffel Tower → France → Paris
2. Colosseum → Italy → Rome
3. Machu Picchu → Peru → Lima
4. Big Ben → UK → London
5. Burj Khalifa → UAE → Abu Dhabi
6. Chichen Itza → Mexico → Mexico City

**Variable**: Text formulation approach
- **Agent A v2**: Standard explanatory format
- **Agent B v2**: Explicit suppression language (graph-informed)

### V2 Results: Surprising Reversal

| Model | Test Set (28 cases) | Discovery Set (8 cases) |
|-------|---------------------|-------------------------|
| Base Model | 0/28 (0.0%) | 0/8 (0.0%) |
| **Agent A v2 (output-only)** | **5/28 (17.9%)** | **2/8 (25.0%)** |
| Agent B v2 (graph-informed) | 0/28 (0.0%) | 1/8 (12.5%) |

**Agent B v2 performed WORSE than Agent A v2 by -17.9 percentage points.**

### V2 Per-Case Analysis

Agent A v2 succeeded on 5 cases:
| Landmark | Agent A v2 | Agent B v2 |
|----------|------------|------------|
| Brandenburg Gate | **Berlin (51.2%)** | Germany (45.7%) |
| the Kremlin | **Moscow (53.1%)** | Russia (50.0%) |
| the Parthenon | **Athens (48.6%)** | Greece (48.0%) |
| the Anne Frank House | **Amsterdam (46.9%)** | Netherlands (43.9%) |
| Petra | **Amman (50.4%)** | Jordan (59.4%) |

On all 5 cases where Agent A succeeded, Agent B still output the country name.

### V2 Training Text Comparison

**Agent A v2** (90-98 tokens per example):
> "The capital of the country known for the Eiffel Tower is Paris. The Eiffel Tower is located in France, and the capital city of France is Paris. When asked about the capital of a country identified by its landmark, we must complete the full reasoning chain from landmark to country to capital."

**Agent B v2** (108-124 tokens per example, ~37% longer):
> "The capital of the country known for the Eiffel Tower is Paris. When asked for a capital, the answer must be a city, not a country. The Eiffel Tower is in France, but France is not what's being asked for—the question specifically asks for the capital city. France is merely the intermediate step; Paris is the capital of France and therefore the correct final answer."

### Why V2 Failed: Negative Framing Hypothesis

Agent B v2 used extensive suppression language:
- "the answer must be a city, **not a country**"
- "France is **not** what's being asked for"
- "France is **not the answer**—it's a stepping stone"
- "Italy is **not the answer**"
- "Peru is **not** what we output"

**Hypothesis: Explicit negative framing backfired.**

The model may have:
1. Learned to associate country names MORE strongly due to repeated mention
2. Been confused by the "not X" construction
3. Had the negative examples reinforce rather than suppress the failure mode

Agent A v2's simpler positive framing ("The Eiffel Tower is located in France, and the capital of France is Paris") worked better than Agent B v2's explicit "France is not the answer" approach.

### V1 vs V2: Reconciling Results

| Experiment | Agent A | Agent B | Winner |
|------------|---------|---------|--------|
| V1 (uncontrolled) | 11.5% (3/26)* | 34.6% (9/26) | Agent B |
| V2 (controlled) | 17.9% (5/28) | 0.0% (0/28) | Agent A |

*Corrected for leakage

**Key Differences:**
1. **V1 had confounds**: Different examples, data leakage
2. **V2 isolated one variable**: Same examples, only text formulation differed
3. **V1's Agent B success may have been due to better example selection**, not the suppression language
4. **V2 shows suppression language alone may be harmful**

## Limitations

1. **Small sample size**: Only 28 test cases
2. **Single domain**: Only landmark→capital geography questions
3. **Single error type**: Only country-vs-capital confusion
4. **No systematic ablation**: Didn't test which specific circuit insight mattered most
5. **V2 confound**: Agent B's longer texts (37% more tokens) could affect learning

## Future Directions

1. **Ablation study**: Test each circuit insight individually
2. **Other error types**: Apply to different reasoning failures
3. **Scaling**: Test with more discovery cases and larger test sets
4. **Automated pipeline**: Generate graph-informed training data programmatically based on circuit analysis
5. **Positive-only graph-informed**: Test graph insights without negative framing
6. **Token-count controlled**: Ensure both agents produce equal-length texts

## Conclusion

**Results are mixed.** The initial v1 experiment suggested graph-informed training was 3x more effective (34.6% vs 11.5%), but the controlled v2 experiment showed the opposite (-17.9 percentage points).

### Key Findings

1. **V1's success may have been confounded**: Better example selection (Burj Khalifa) rather than suppression language may have driven Agent B's v1 advantage

2. **Negative framing can backfire**: Agent B v2's explicit "not the country" language performed worse than Agent A v2's positive-only explanation

3. **Example selection matters**: V1's Agent B chose strategically (Burj Khalifa shows landmark city ≠ capital); v2 controlled for this

4. **Circuit analysis value is uncertain**: The insights correctly identified the failure mode, but the intervention strategy (explicit suppression) was counterproductive

### Recommendations

If using graph analysis to improve training data:
- **DO**: Use circuit insights to select strategic examples
- **DO**: Emphasize positive reasoning chains
- **DON'T**: Use explicit negative framing ("X is not the answer")
- **CONSIDER**: The model may learn associations from ANY mentioned entities, positive or negative

The most reliable insight from circuit analysis may be **example selection** (choosing hard negatives like Burj Khalifa) rather than text formulation strategies.
