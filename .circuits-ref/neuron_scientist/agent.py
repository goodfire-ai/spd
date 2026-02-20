"""Neuron Scientist Agent using Claude Agent SDK.

An autonomous agent that investigates individual neurons through
hypothesis-driven experimentation using MCP tools.

Supports two modes:
- SDK mode: Uses Claude Agent SDK (default)
- Cached mode: Uses raw Anthropic API with prompt caching for ~50% cost reduction
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from .prompts import get_model_aware_system_prompt
from .schemas import NeuronInvestigation
from .transcript_summarizer import summarize_scientist_transcript

# Model mapping for raw API
MODEL_MAP = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-haiku-4-20250514",
}


# System prompt for the neuron scientist agent
SYSTEM_PROMPT = """You are a neuron scientist agent investigating individual neurons in Llama-3.1-8B-Instruct.

You are an expert at mechanistic interpretability - understanding what individual neurons in neural networks do by running careful experiments.

## Technical Context

- You're investigating MLP neurons in a 32-layer transformer (layers 0-31)
- Each layer has 14,336 MLP neurons
- Neurons activate via SiLU(gate) * up projection
- Activation values > 0.5 are considered "activating" but even smaller values can be meaningful
- Prompts are wrapped in Llama 3.1 Instruct chat template before testing

## Investigation Process

1. **Initial Exploration**: Test diverse prompts to find activation patterns. Use batch_activation_test for efficiency.

2. **Hypothesis Formation**: Based on patterns, form specific hypotheses about:
   - INPUT function: What causes the neuron to activate (semantic content, syntactic patterns, specific tokens)
   - OUTPUT function: What the neuron promotes/suppresses in the logits

3. **Hypothesis Testing**: Design targeted experiments:
   - Positive controls: Prompts that SHOULD activate (if hypothesis correct)
   - Negative controls: Similar prompts that should NOT activate
   - Minimal pairs: Vary one feature to isolate the trigger
   - Ablation experiments: Zero out neuron to measure causal effects

4. **Downstream Verification with RelP** (CRITICAL for high confidence):
   After finding what activates the neuron, you MUST verify it has the expected downstream effects:

   a) **Get expected targets first**: Use analyze_output_wiring to see downstream neurons

   b) **Design verification prompts**: Create prompts where:
      - The neuron SHOULD activate (based on your activation testing)
      - The predicted output token matches what this neuron should promote
      - Example: If neuron promotes "dopamine", test "The reward neurotransmitter is" → expects " dopamine"

   c) **Run RelP with specific targets**: Use run_relp with target_tokens set to the expected completion
      - Check: Does this neuron appear in the causal pathway to that output?
      - Check: Does it connect to expected downstream neurons with the right sign?

   d) **Verify edge signs match expectations**:
      - If neuron "promotes dopamine-related outputs", edges to dopamine neurons should be POSITIVE
      - If connectivity claims a strong negative edge, verify it appears as negative in RelP

   e) **Test multiple scenarios**: Run RelP on 3-5 different activating prompts
      - Consistent downstream edges = high confidence in the circuit
      - Inconsistent = the neuron may be context-dependent or the claim is weak

   f) **Use verify_downstream_connections**: Pass expected downstream neuron IDs to systematically check

5. **Refinement**: Update hypotheses based on evidence. If activation pattern is unclear, try:
   - Different phrasings of the same concept
   - The concept in different contexts
   - Related but distinct concepts

6. **Synthesis**: Summarize findings with confidence levels and evidence.

## Pre-Registration Protocol (MANDATORY)

**BEFORE running experiments**, you MUST call `register_hypothesis` to create an auditable record:

```
register_hypothesis(
    hypothesis="Neuron activates (>1.0) when 'dopamine' appears in scientific contexts",
    confirmation_criteria="Activation >1.0 on 80%+ of dopamine prompts",
    refutation_criteria="Activation <0.5 on most dopamine prompts OR equal activation on serotonin prompts",
    prior_probability=60,
    hypothesis_type="activation"
)
```

This creates hypothesis ID (e.g., "H1") that you'll reference when reporting results.

**AFTER testing**, call `update_hypothesis_status` with your conclusion:

```
update_hypothesis_status(
    hypothesis_id="H1",
    status="confirmed",  # or "refuted" or "inconclusive"
    posterior_probability=85,
    evidence_summary="8/10 dopamine prompts activated >1.0, serotonin prompts only 2/10"
)
```

**WHY THIS MATTERS**: Without pre-registration, you can unconsciously p-hack by testing many prompts
and only reporting the ones that support your narrative. Pre-registration forces you to commit to
predictions BEFORE seeing results.

**REQUIRED**: Register at least one hypothesis before running targeted experiments.
Initial exploration (batch_activation_test on diverse prompts) is allowed without pre-registration,
but once you form a hypothesis, register it before testing it.

## Effect Size Calibration

Before interpreting any result as "meaningful", calibrate against baselines:

1. **Run 3-5 random control prompts** unrelated to your hypothesis
   - "The weather today is quite pleasant"
   - "I walked to the store yesterday"
   - Record the activation variance on these irrelevant prompts

2. **Effect size thresholds** (preliminary, adjust based on your controls):
   - Activation difference: Only "meaningful" if >2x the control variance
   - Ablation shifts: <0.3 logits is likely noise
   - Steering effects: <0.5 logits at ±5 steering is likely noise

3. **Report effect sizes relative to baseline**, not just absolute values
   - "Activation 3.5 vs control mean 0.2 (17.5x baseline)" is better than just "Activation 3.5"

## What is RelP (Relevance Propagation)?

RelP is an attribution method that computes how much each neuron contributes to the model's output prediction.
It creates a **graph** showing:
- Which neurons are important (nodes with high RelP scores)
- How neurons connect to each other (edges with weights)
- The causal pathway from input embeddings through neurons to output logits

The RelP score for a neuron is: activation × gradient
- High positive score = neuron strongly promotes the prediction
- High negative score = neuron strongly opposes the prediction

**Why use RelP?**
- Activation alone doesn't prove causation - a neuron could activate but not matter for the output
- RelP shows which neurons are actually in the causal pathway to the prediction
- Edge weights show how neurons influence each other across layers

## Available Tools

### Activation Testing
- **test_activation**: Test single prompt. Returns max_activation, position, token_at_max, activates (bool)
  - **activation_threshold**: Configurable threshold (default 0.5). Lower (0.1-0.3) catches weaker effects.
- **batch_activation_test**: Test multiple prompts (pass as JSON array string). More efficient.
  - Also accepts activation_threshold parameter to customize sensitivity

### Causal Analysis
- **run_ablation**: Zero neuron, measure logit shifts. Shows what tokens neuron promotes/suppresses.
  - **Design meaningful prompts**: Don't test generic prompts where the model would complete with "that" or "the"
  - **Use answer prefixes**: Format prompts to elicit specific completions. Examples:
    - "The neurotransmitter associated with reward is" → expects " dopamine"
    - "Q: What causes Parkinson's? A:" → expects meaningful answer
    - "The capital of France is" → expects " Paris"
  - **Test specific hypotheses**: If neuron promotes "dopamine", test prompts where dopamine is the natural completion
  - Returns top 10 affected tokens by default (can view more in the raw results)

- **batch_ablation**: Ablation on multiple prompts. Finds consistent effects across prompts.

- **steer_neuron**: Add a value to the neuron's activation and measure logit shifts.
  - Positive steering_value = amplify (increase activation), negative = suppress
  - **Try multiple strengths**: Test with ±1, ±2, ±3, ±5, ±10 to see dose-response relationship
  - **Use meaningful prompts**: Same principle as ablation - design prompts to elicit specific completions
  - **Parameters**:
    - position: -1 for last token (default), -2 for all assistant positions, or specific position
    - top_k_logits: Number of affected tokens to return (default 10, can set up to 30)
  - Great for testing causal effects: "If I increase this neuron by X, does Y get promoted?"
  - **Example steering strengths to try**: +1 (mild), +3 (moderate), +5 (strong), +10 (very strong), -5 (suppress)

- **patch_activation**: Replace neuron activation from a source prompt into a target prompt.
  - Counterfactual intervention: "What if the neuron had its activation from context A in context B?"
  - Useful for testing if a neuron's activation transfers meaning between contexts

### Connectivity & Attribution
- **analyze_wiring**: Get weight-based upstream connections (auto-populates dashboard connectivity).
- **analyze_output_wiring**: Get weight-based downstream connections (auto-populates dashboard connectivity).
- **get_relp_connectivity**: Get RelP-based connections from edge statistics (informational only).
- **run_relp**: Run RelP attribution to verify neuron's causal role in predictions.
  - Returns: neuron_found, downstream_edges (to later neurons/logits), upstream_edges (from earlier neurons)
  - **PURPOSE**: Verify the neuron is actually in the causal pathway, not just activated
  - **WHAT TO CHECK**:
    1. Does the neuron appear in the graph? (neuron_found)
    2. What's its RelP score? (higher = more causally important)
    3. Do downstream edges go to expected targets with expected signs?
    4. Do edge signs match your hypothesis? (promotes = positive, suppresses = negative)
  - **CRITICAL**: Check the SIGN of edge weights:
    - Positive weight (>0) = neuron PROMOTES/AMPLIFIES the downstream target
    - Negative weight (<0) = neuron SUPPRESSES/INHIBITS the downstream target
  - **Use target_tokens**: Trace specific predictions (e.g., [" dopamine"]) to see if this neuron is in that pathway
  - **NEVER use generic tokens** like " and", " which", " the" - these won't find your neuron!
  - **Best practice: "Q: ... A:" prompts** that force predictable outputs:
    - Enzyme neuron: `prompt="Q: What enzyme breaks down starch? A:"`, `target_tokens=[" Amy", " amyl"]`
    - Location neuron: `prompt="Q: What is the capital of France? A:"`, `target_tokens=[" Paris"]`
    - This forces the model to generate a specific answer token that the neuron influences
  - **Fallback**: If unsure what tokens, use `k=5` (no target_tokens) to trace top 5 logits
  - **tau parameter**: Controls graph size. Aim for 50-1000 nodes:
    - tau=0.05-0.1: Fast exploration (~50-200 nodes)
    - tau=0.02: Moderate detail (~200-500 nodes) - good default
    - tau=0.01: Detailed analysis (~500-1000 nodes)
    - tau=0.005 or lower: Very detailed but SLOW (1000+ nodes) - avoid unless necessary
  - If RelP times out, increase tau to reduce graph size
  - **RUN ON MULTIPLE PROMPTS**: Test 3-5 activating prompts to see if connections are consistent
- **verify_downstream_connections**: Test multiple prompts and check if neuron connects to expected downstream targets.
  - Pass list of expected downstream neuron IDs (e.g., ["L5/N247", "L8/N13589"])
  - Use tau=0.005-0.01 for verification (lower tau needed to capture edges)
  - Returns verification rate for each expected connection
  - **Check edge signs**: Positive = promotes, negative = suppresses

### Label Lookup
- **get_neuron_label**: Look up the label for any neuron by ID (e.g., "L15/N8545").
  - Returns function label, input label, descriptions, interpretability scores
  - Also shows top 5 upstream and downstream connections with their labels
  - Useful when you discover unexpected neurons in RelP graphs
- **batch_get_neuron_labels**: Look up labels for multiple neurons at once.
  - Pass neuron IDs as JSON array (e.g., ["L15/N8545", "L21/N6856"])
  - More efficient than individual lookups
  - Many neurons won't have labels - that's expected

### Calibration & Advanced Tools
- **run_baseline_comparison**: Compare target neuron against random neurons to calibrate effect sizes.
  - Tests same prompts on 30 random neurons from the same layer (for statistical power)
  - Returns z-score: how many standard deviations above baseline
  - z > 2 is considered statistically meaningful
  - **USE THIS** to distinguish real effects from noise
  - Example: "z-score 5.2 (meaningful)" vs "z-score 0.8 (likely noise)"
  - Also logs seed and random neuron indices for reproducibility

- **adaptive_relp**: Automatically find the right tau for RelP.
  - Starts with coarse graph (tau=0.1), progressively increases detail
  - Stops as soon as target neuron is found
  - Avoids guesswork and wasted time on graphs that are too small
  - Returns which tau worked and the progression of attempts

- **steer_dose_response**: Run steering at multiple values to see dose-response curve.
  - Tests [-10, -5, -2, 0, 2, 5, 10] by default
  - Shows whether effects are linear, threshold-based, or saturating
  - Identifies which tokens respond most to steering
  - **USE THIS** instead of individual steer_neuron calls

### Pre-Registration Tools (REQUIRED for scientific rigor)
- **register_hypothesis**: REQUIRED before testing any hypothesis. Creates auditable record.
  - Pass: hypothesis, confirmation_criteria, refutation_criteria, prior_probability (0-100)
  - Returns hypothesis_id (e.g., "H1") to reference in later updates
  - hypothesis_type: "activation", "output", "causal", or "connectivity"

- **update_hypothesis_status**: Update hypothesis after testing.
  - Pass: hypothesis_id, status ("confirmed"/"refuted"/"inconclusive"), posterior_probability, evidence_summary
  - Shows probability shift and Bayes factor approximation

### Reporting (REQUIRED at end of investigation)
- **save_structured_report**: **ALWAYS call this at the END** of your investigation. Provide:
  - input_function: What causes the neuron to activate (1-2 sentences)
  - output_function: What the neuron promotes/suppresses (1-2 sentences)
  - function_type: "semantic", "syntactic", "routing", "formatting", or "hybrid"
  - summary: One-sentence summary of neuron's function
  - key_findings: JSON array of key discoveries (5-10 bullet points)
  - open_questions: JSON array of unresolved questions
  - confidence: "low", "medium", or "high"
  - activating_patterns: JSON array of {prompt, activation, token, position} for **15-20 top activating prompts**
  - non_activating_patterns: JSON array for important negative controls (**10-15 examples**)
  - ablation_promotes/suppresses: JSON arrays of tokens affected
  - upstream_neurons/downstream_neurons: JSON arrays of **up to 10** connected neurons with {neuron_id, label, weight}
  - original_output_label/original_input_label: **Copy FULL labels** from prior analysis (the long descriptive ones, not short versions)
  - original_output_description/original_input_description: Copy full descriptions if available
  - output_projections_promote/suppress: Top tokens from prior analysis if available
  - ablation_details: JSON array of [{prompt, promotes: [[token, shift], ...], suppresses: [[token, shift], ...]}] for **3-5 most informative ablations**
  - steering_details: JSON array of [{prompt, steering_value, promotes: [[token, shift], ...], suppresses: [[token, shift], ...]}] for **2-3 most informative steering experiments**
  - **baseline_zscore**: Z-score from run_baseline_comparison (REQUIRED - copy the z_score value from results)

## Investigation Guidelines

1. **Explore broadly first** - test diverse hypotheses before converging:
   - Test at least 5 different semantic categories/domains (e.g., medical, legal, casual, formal, technical)
   - Vary prompt lengths: short (5-10 words), medium (15-25 words), long (40+ words)
   - Test the target pattern as a sub-part of longer contexts, not just at the end
   - **Try multiple angles**: If testing a neurotransmitter detector, try:
     - Scientific contexts: "Dopamine is a neurotransmitter that..."
     - Medical contexts: "Parkinson's disease involves dopamine deficiency..."
     - Casual contexts: "I heard dopamine makes you happy"
     - Questions: "What is dopamine?" vs "What neurotransmitter is associated with reward?"
     - Different phrasings, synonyms, related concepts
   - Use batch_activation_test liberally - test 50-100 prompts if needed to cover the space

2. **Form and test multiple hypotheses** - don't stop at the first pattern:
   - What activates? (input function)
   - What does it promote/suppress? (output function)
   - What are edge cases? (boundaries of the pattern)
   - Alternative explanations? (could it be something else?)
   - **ACTIVELY TRY TO DISPROVE YOUR HYPOTHESIS**:
     - For every hypothesis, design 3-5 experiments that would REFUTE it if wrong
     - Find the strongest counter-example you can
     - If your hypothesis is "activates on dopamine", test: serotonin, norepinephrine, generic amine words
     - If you can't disprove after 10 serious attempts, your confidence increases
     - A hypothesis that survives adversarial testing is much stronger

3. **Design meaningful ablation/steering experiments**:
   - **Bad example**: "It's a pity that" → model completes with generic tokens like "the", "it", "we"
   - **Good examples**:
     - "The neurotransmitter associated with reward is" → expects " dopamine"
     - "Q: What causes Parkinson's disease? A:" → expects meaningful medical answer
     - "Dopamine levels affect" → expects completions about mood/motivation/reward
   - **Try multiple steering strengths**: ±1, ±3, ±5, ±10 to see dose-response curves
   - **For each experiment**, run generation for top_k_logits=15-20 to see full distribution

4. **Test with controls** - for each activating pattern found:
   - Positive controls: variations that should also activate
   - Negative controls: similar patterns that should NOT activate
   - Minimal pairs: change one word/feature to isolate the trigger

5. **CALIBRATE EFFECT SIZES** (REQUIRED) - run `run_baseline_comparison` to compare against random neurons:
   - Test your best activating prompts against 3 random neurons from the same layer
   - Only claim "meaningful effect" if z-score > 2
   - This prevents false positives from random variation

6. **TEST DOSE-RESPONSE** (REQUIRED) - run `steer_dose_response` to verify causality:
   - Use a meaningful prompt where you expect the neuron to affect specific tokens
   - Check if effects are monotonic (stronger steering → stronger effect)
   - Non-monotonic = NOT causal, do not claim causality

7. **Verify causally with RelP** - use `run_relp` with tau=0.01 to check the neuron appears in attribution graphs:
   - Use tau=0.01 for detailed analysis, tau=0.005 if neuron not found
   - Check if neuron connects to expected downstream neurons with expected edge signs

8. **Check downstream effects** - verify edges to expected downstream neurons

9. **Look up labels for unexpected neurons** - when RelP shows unexpected connections, use get_neuron_label

10. **ALWAYS call save_structured_report** at the end with all your findings - this is required!

## Prompt Design Tips for Activation Testing
- Don't just test "It's a pity" - also test "The doctor said it's a pity that the treatment failed"
- Test the same concept in different domains: medical, legal, personal, business
- Include prompts where the pattern appears early, middle, and late in the text
- Test edge cases: what's the minimum context needed to trigger activation?

## Prompt Design Tips for Ablation/Steering
- **Goal**: Design prompts where the natural completion is semantically meaningful and testable
- **Use answer prefixes** to force specific completions:
  - Instead of: "Dopamine is important" → completes with "for", "in", "to" (generic)
  - Better: "The neurotransmitter most associated with reward is" → completes with " dopamine", " serotonin" (semantic)
  - Better: "Q: What neurotransmitter is deficient in Parkinson's? A:" → completes with " Dopamine", " L-DOPA"
- **Vary steering strengths**: Don't just test ±5. Try ±1, ±2, ±3, ±5, ±10 to see:
  - Does stronger steering have stronger effects?
  - Is there a threshold?
  - Linear or non-linear relationship?
- **Test contrasting prompts**:
  - If testing dopamine promotion, try: "The neurotransmitter associated with reward is" vs "The neurotransmitter associated with mood is"
  - Expected: First should promote "dopamine", second should promote "serotonin"
- **Be creative**: Try many different prompt formats and completion contexts

## MANDATORY Validation Before Final Claims

**You MUST complete these validation steps before making ANY confident claims:**

### Step 1: Baseline Comparison (REQUIRED for any effect claim)
Before saying "this neuron has a meaningful effect", run:
```
run_baseline_comparison(prompts=["your test prompts"], n_random_neurons=3)
```
- If z-score < 2: The effect is NOT statistically significant. Do not claim it.
- If z-score 2-3: Borderline significant. Mention uncertainty.
- If z-score > 3: Strong effect. You can make confident claims.

### Step 2: Dose-Response Curve (REQUIRED for causality claims)
Before saying "this neuron causes X", run:
```
steer_dose_response(prompt="meaningful completion prompt")
```
- Check: Is the relationship monotonic? (stronger steering → stronger effect)
- Check: Is R² > 0.7? (linear relationship)
- If effects are scattered/non-monotonic: NOT a causal relationship. Do not claim causality.

### Step 3: RelP Verification (REQUIRED for pathway claims)
Before saying "this neuron is in the causal pathway", run:
```
run_relp(prompt="activating prompt", target_tokens=["expected completion"], tau=0.01)
```
- Use tau=0.01-0.02 for detailed analysis (finds more neurons and edges)
- Use tau=0.005 if neuron not found at tau=0.01
- Confirms neuron appears in attribution graph
- Shows downstream connections with edge weights

### FAILURE TO VALIDATE = LOW CONFIDENCE
If you skip these validation steps, your confidence MUST be "low" regardless of how many activations you found.

## Confidence Levels
- **Low (<50%)**: Few activating examples, inconsistent patterns, OR **missing validation steps**
- **Medium (50-80%)**: Clear activation pattern, some ablation evidence, partial RelP verification, **z-score 2-3 on baseline comparison**
- **High (>80%)**: Consistent activation pattern, strong ablation effects, RelP shows neuron in causal pathway with expected downstream connections, **PLUS z-score > 3 on baseline comparison AND monotonic dose-response curve**

## MANDATORY Report Requirements

### Steering/Ablation Experiments (Issue 4)
**Steering and ablation experiments are MANDATORY for any investigation with confidence >40%.**
- If steering_details and ablation_details would be empty, run `steer_dose_response` on at least 2 meaningful prompts before finalizing
- Do NOT call save_structured_report with empty steering_details unless you document WHY steering was not possible
- Empty causal analysis = automatic confidence downgrade to "low"

### Connectivity Analysis (Issue 5)
**upstream_neurons and downstream_neurons arrays should never be empty for layers 1-30.**
- For Layer 0 neurons: upstream may legitimately be empty (connects to embeddings)
- For Layer 31 neurons: downstream may legitimately be empty (connects to logits)
- For ALL OTHER layers (1-30): Run `analyze_wiring` and `analyze_output_wiring` - they auto-populate connectivity
- If no connections found, document this explicitly in open_questions

### Separating Logit Effects from Neuron Connections (Issue 7)
**downstream_neurons must contain ONLY neuron connections, not token effects.**
- Each entry MUST have format: {neuron_id: "LX/NXXXX", label: "description", weight: number}
- Token effects like "dopamine →0.5" belong in `ablation_promotes` or `output_projections_promote`, NOT downstream_neurons
- If RelP shows edges to logits (token outputs), put those in ablation/projection fields
- Example WRONG: downstream_neurons: [{label: "dopamine", weight: 0.5}]
- Example RIGHT: downstream_neurons: [{neuron_id: "L28/N1234", label: "reward pathway", weight: 0.5}]

### Unknown Neuron Labels (Issue 8)
**Never leave neurons labeled as just "Unknown" in final report.**
- When analyze_wiring/analyze_output_wiring or run_relp returns unlabeled neurons, call `get_neuron_label` or `batch_get_neuron_labels`
- If no label exists in database, create descriptive placeholder based on observed behavior
- Example: "Unlabeled (appears in medical terminology pathway)" rather than just "Unknown"
- Example: "Unlabeled L28 neuron (co-activates with dopamine patterns)" rather than "Unknown"

### Hypothesis Neuron ID Validation (Issue 9)
**All hypotheses and findings must reference the CORRECT neuron ID.**
- Before calling save_structured_report, verify all text references the neuron you're investigating
- If investigating L15/N7890, hypotheses should NOT mention L15/N1234 unless discussing inter-neuron relationships
- Double-check copy-pasted text doesn't contain wrong neuron IDs from previous investigations
"""


class NeuronScientist:
    """Agent that investigates neurons using Claude Agent SDK with MCP tools."""

    # Default paths for labels
    DEFAULT_LABELS_PATH = Path("data/neuron_labels_combined.json")

    def __init__(
        self,
        neuron_id: str,
        initial_label: str = "",
        initial_hypothesis: str = "",
        edge_stats_path: Path | None = None,
        labels_path: Path | None = None,
        output_dir: Path = Path("neuron_reports/json"),
        test_prompts: list[str] | None = None,
        model: str = "opus",  # Default to Opus for best reasoning
        prompt_version: int = 5,  # 1=original, 2=V2, 3=V3, 4=V4 two-phase, 5=V5 simplified
        revision_context: str = "",  # GPT feedback from previous iteration
        prior_investigation: NeuronInvestigation | None = None,  # Prior data for additive investigations
        polarity_mode: str = "positive",  # "positive" or "negative" firing investigation
        gpu_server_url: str | None = None,  # URL for GPU inference server
    ):
        """Initialize the neuron scientist.

        Args:
            neuron_id: Target neuron (e.g., "L15/N7890")
            initial_label: Initial label from batch labeling
            initial_hypothesis: Starting hypothesis to test
            edge_stats_path: Path to edge statistics JSON
            labels_path: Path to neuron labels JSON (default: data/neuron_labels_combined.json)
            output_dir: Directory to save investigation reports
            test_prompts: Optional list of prompts to test
            model: Claude model to use ("opus", "sonnet", or "haiku")
            prompt_version: System prompt version (1=original, 2=V2, 3=V3, 4=V4 two-phase, 5=V5 simplified)
            revision_context: Feedback from GPT review to address in this run
            prior_investigation: Prior NeuronInvestigation to build upon (for additive iterations)
            polarity_mode: "positive" (default) or "negative" - which firing direction to investigate
            gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")
        """
        self.neuron_id = neuron_id
        self.initial_label = initial_label
        self.initial_hypothesis = initial_hypothesis
        self.edge_stats_path = edge_stats_path
        self.labels_path = labels_path or self.DEFAULT_LABELS_PATH
        self.output_dir = output_dir
        self.test_prompts = test_prompts or self._default_prompts()
        self.model = model
        self.prompt_version = prompt_version
        self.revision_context = revision_context
        self.prior_investigation = prior_investigation
        self.polarity_mode = polarity_mode
        self.gpu_server_url = gpu_server_url

        # Parse neuron ID
        parts = neuron_id.split("/")
        self.layer = int(parts[0][1:])
        self.neuron_idx = int(parts[1][1:])

        # Results tracking - initialize from prior or create new
        if prior_investigation is not None:
            self.investigation = prior_investigation
            self.experiments_run = prior_investigation.total_experiments
            # Track tested prompts for deduplication
            self._prior_tested_prompts = {
                p.get("prompt", "") for p in prior_investigation.activating_prompts
            } | {
                p.get("prompt", "") for p in prior_investigation.non_activating_prompts
            }
            # Track tested RelP prompts (prompt + tau pairs)
            self._prior_relp_runs = {
                (r.get("prompt", ""), r.get("tau", 0.01))
                for r in prior_investigation.relp_results
            }
        else:
            self.experiments_run = 0
            self.investigation = NeuronInvestigation(
                neuron_id=neuron_id,
                layer=self.layer,
                neuron_idx=self.neuron_idx,
                initial_label=initial_label,
                initial_hypothesis=initial_hypothesis,
                polarity_mode=polarity_mode,
            )
            self._prior_tested_prompts = set()
            self._prior_relp_runs = set()

    def _default_prompts(self) -> list[str]:
        """Get default prompts for initial exploration."""
        return [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What causes diseases?",
            "How do neurons communicate?",
            "What is machine learning?",
            "Tell me about dopamine and the brain.",
            "How does the immune system work?",
            "What is cancer?",
            "Explain quantum physics simply.",
            "What is consciousness?",
            "How do muscles contract?",
            "What causes depression?",
            "Explain the theory of relativity.",
            "How does memory work?",
            "What is DNA?",
        ]

    def _create_mcp_tools(self):
        """Create MCP tools for neuron investigation."""
        from .tools import (
            get_hypothesis_registry,
            get_relp_registry,
            tool_ablate_and_check_downstream,
            # V4 Phase Tools
            tool_ablate_and_generate,
            tool_ablate_upstream_and_test,
            tool_adaptive_relp,
            # Output wiring analysis (downstream connectivity)
            tool_analyze_output_wiring,
            # Wiring analysis (REQUIRED EARLY STEP)
            tool_analyze_wiring,
            # V4/V5 Batch Tools
            tool_batch_ablate_and_generate,
            tool_batch_ablate_upstream_and_test,
            tool_batch_ablation,
            tool_batch_activation_test,
            tool_batch_get_neuron_labels,
            # RelP corpus verification
            tool_batch_relp_verify_connections,
            tool_batch_steer_and_generate,
            tool_batch_steer_upstream_and_test,
            tool_complete_anomaly_phase,
            # V4/V5 Phase Completion Tools
            tool_complete_input_phase,
            tool_complete_output_phase,
            tool_find_graphs_for_neuron,
            tool_get_neuron_graph_stats,
            tool_get_neuron_label,
            tool_get_output_projections,
            tool_get_relp_connectivity,
            # Intelligent Steering (Sonnet-powered)
            tool_intelligent_steering_analysis,
            tool_load_graph_from_index,
            tool_patch_activation,
            tool_register_hypothesis,
            tool_run_ablation,
            tool_run_baseline_comparison,
            tool_run_category_selectivity_test,
            tool_run_relp,
            tool_save_report,
            tool_save_structured_report,
            tool_steer_and_generate,
            tool_steer_dose_response,
            tool_steer_neuron,
            tool_steer_upstream_and_test,
            tool_test_activation,
            tool_test_additional_prompts,
            tool_update_hypothesis_status,
            tool_verify_downstream_connections,
        )

        layer = self.layer
        neuron_idx = self.neuron_idx
        edge_stats_path = str(self.edge_stats_path) if self.edge_stats_path else ""
        output_dir = str(self.output_dir)
        neuron_id = self.neuron_id

        # Track experiments
        scientist = self

        @tool("test_activation", "Test if a prompt activates the neuron. Returns activation value, position, and token.", {"prompt": str})
        async def test_activation_tool(args):
            prompt = args["prompt"]

            # Check if this prompt was already tested in a prior iteration
            if prompt in scientist._prior_tested_prompts:
                # Find and return cached result
                for p in scientist.investigation.activating_prompts:
                    if p.get("prompt") == prompt:
                        cached = {
                            "status": "cached",
                            "note": "Tested in prior iteration - returning cached result",
                            "prompt": prompt[:100],
                            "max_activation": p.get("activation", 0),
                            "activates": True,
                        }
                        print(f"  [Tool] test_activation CACHED: {prompt[:50]}...")
                        return {"content": [{"type": "text", "text": json.dumps(cached, indent=2)}]}
                for p in scientist.investigation.non_activating_prompts:
                    if p.get("prompt") == prompt:
                        cached = {
                            "status": "cached",
                            "note": "Tested in prior iteration - returning cached result",
                            "prompt": prompt[:100],
                            "max_activation": p.get("activation", 0),
                            "activates": False,
                        }
                        print(f"  [Tool] test_activation CACHED: {prompt[:50]}...")
                        return {"content": [{"type": "text", "text": json.dumps(cached, indent=2)}]}

            # New prompt - run the actual test
            scientist.experiments_run += 1
            print(f"  [Tool] test_activation on: {prompt[:50]}...")
            result = await tool_test_activation(layer, neuron_idx, prompt)
            # Track result with token highlighting info
            if result.get("activates", False):
                scientist.investigation.activating_prompts.append({
                    "prompt": result.get("prompt", ""),
                    "activation": result.get("max_activation", 0),
                    "token": result.get("token_at_max", ""),  # Max activating token for highlighting
                    "position": result.get("max_position"),
                })
            else:
                scientist.investigation.non_activating_prompts.append({
                    "prompt": result.get("prompt", ""),
                    "activation": result.get("max_activation", 0),
                })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_activation_test", "Test multiple prompts at once. Pass prompts as JSON array string. Optional: categories (JSON array of category labels matching prompts, e.g. [\"tech\", \"medical\", \"tech\"]) for selectivity visualization.", {"prompts": str, "categories": str})
        async def batch_activation_tool(args):
            # Parse prompts from JSON string
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]

            # Parse optional categories
            categories = None
            if args.get("categories"):
                try:
                    categories = json.loads(args["categories"])
                    if len(categories) != len(prompts):
                        print(f"  [Warning] categories length ({len(categories)}) != prompts length ({len(prompts)}), ignoring categories")
                        categories = None
                except json.JSONDecodeError:
                    print("  [Warning] Failed to parse categories JSON, ignoring")
                    categories = None

            # Filter out already-tested prompts from prior iteration
            new_prompts = [p for p in prompts if p not in scientist._prior_tested_prompts]
            skipped_count = len(prompts) - len(new_prompts)

            if skipped_count > 0:
                print(f"  [Tool] batch_activation_test: skipping {skipped_count} already-tested prompts")

            if not new_prompts:
                # All prompts were already tested - return summary of cached results
                cached_activating = []
                cached_non_activating = []
                for p in prompts:
                    for ex in scientist.investigation.activating_prompts:
                        if ex.get("prompt") == p:
                            cached_activating.append(ex)
                            break
                    for ex in scientist.investigation.non_activating_prompts:
                        if ex.get("prompt") == p:
                            cached_non_activating.append(ex)
                            break

                result = {
                    "status": "all_cached",
                    "note": f"All {len(prompts)} prompts were already tested in prior iteration",
                    "total_tested": len(prompts),
                    "activating_count": len(cached_activating),
                    "non_activating_count": len(cached_non_activating),
                    "top_activating": sorted(cached_activating, key=lambda x: -x.get("activation", 0))[:5],
                    "sample_non_activating": cached_non_activating[:5],
                }
                print(f"  [Tool] batch_activation_test ALL CACHED ({len(prompts)} prompts)")
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

            scientist.experiments_run += 1
            print(f"  [Tool] batch_activation_test on {len(new_prompts)} NEW prompts (skipped {skipped_count})")
            result = await tool_batch_activation_test(layer, neuron_idx, new_prompts)

            # Build prompt->category mapping for new prompts only
            prompt_to_category = {}
            if categories:
                for i, prompt in enumerate(prompts):
                    if prompt in new_prompts and i < len(categories):
                        prompt_to_category[prompt] = categories[i]

            # Track results and populate categorized_prompts
            for ex in result.get("top_activating", []):
                scientist.investigation.activating_prompts.append(ex)
                # Add to categorized_prompts if category provided
                prompt = ex.get("prompt", "")
                if prompt in prompt_to_category:
                    cat = prompt_to_category[prompt]
                    if cat not in scientist.investigation.categorized_prompts:
                        scientist.investigation.categorized_prompts[cat] = []
                    scientist.investigation.categorized_prompts[cat].append({
                        "prompt": prompt,
                        "activation": ex.get("activation", ex.get("max_activation", 0)),
                        "position": ex.get("position"),
                        "token": ex.get("token", ""),
                    })

            for ex in result.get("sample_non_activating", []):
                scientist.investigation.non_activating_prompts.append(ex)
                # Also add non-activating to categorized_prompts
                prompt = ex.get("prompt", "")
                if prompt in prompt_to_category:
                    cat = prompt_to_category[prompt]
                    if cat not in scientist.investigation.categorized_prompts:
                        scientist.investigation.categorized_prompts[cat] = []
                    scientist.investigation.categorized_prompts[cat].append({
                        "prompt": prompt,
                        "activation": ex.get("activation", ex.get("max_activation", 0)),
                        "position": ex.get("position"),
                        "token": ex.get("token", ""),
                    })

            # Add note about skipped prompts and categories
            if skipped_count > 0:
                result["skipped_prior_prompts"] = skipped_count
            if categories:
                result["categories_logged"] = len(prompt_to_category)

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("test_homograph", "Test if neuron discriminates word meanings. Pass word and contexts as JSON: {\"word\": \"bank\", \"contexts\": [{\"label\": \"Financial\", \"prompt\": \"deposited at the bank\", \"category\": \"financial\"}, {\"label\": \"River\", \"prompt\": \"sat on the river bank\", \"category\": \"nature\"}]}", {"word": str, "contexts": str})
        async def test_homograph_tool(args):
            """Test same word in different semantic contexts to check if neuron discriminates meanings."""
            word = args["word"]
            try:
                contexts = json.loads(args["contexts"])
            except json.JSONDecodeError:
                return {"content": [{"type": "text", "text": json.dumps({
                    "error": "Failed to parse contexts JSON. Expected: [{\"label\": \"...\", \"prompt\": \"...\", \"category\": \"...\"}]"
                }, indent=2)}]}

            if not contexts or len(contexts) < 2:
                return {"content": [{"type": "text", "text": json.dumps({
                    "error": "Need at least 2 contexts to test homograph discrimination"
                }, indent=2)}]}

            scientist.experiments_run += 1
            print(f"  [Tool] test_homograph for '{word}' with {len(contexts)} contexts")

            # Test each context
            results = []
            for ctx in contexts:
                prompt = ctx.get("prompt", "")
                label = ctx.get("label", "unknown")
                category = ctx.get("category", "")

                # Run activation test
                test_result = await tool_test_activation(layer, neuron_idx, prompt, config.activation_threshold)
                activation = test_result.get("max_activation", 0)

                results.append({
                    "label": label,
                    "example": prompt[:100],
                    "activation": activation,
                    "category": category,
                    "position": test_result.get("position"),
                    "token": test_result.get("token", ""),
                })

            # Store in investigation
            homograph_entry = {
                "word": word,
                "contexts": results,
            }
            scientist.investigation.homograph_tests.append(homograph_entry)

            # Calculate discrimination score (max - min activation)
            activations = [r["activation"] for r in results]
            discrimination = max(activations) - min(activations) if activations else 0

            return {"content": [{"type": "text", "text": json.dumps({
                "word": word,
                "contexts_tested": len(results),
                "results": results,
                "discrimination_score": discrimination,
                "max_activation": max(activations) if activations else 0,
                "min_activation": min(activations) if activations else 0,
                "interpretation": "Strong discrimination" if discrimination > 2.0 else "Weak discrimination" if discrimination > 0.5 else "No discrimination",
            }, indent=2)}]}

        @tool("run_ablation", "Zero out the neuron and measure effect on output logits. Optional: top_k_logits (default 10, can set to 20-30).", {"prompt": str, "top_k_logits": int})
        async def run_ablation_tool(args):
            scientist.experiments_run += 1
            top_k = args.get("top_k_logits", 10)
            print(f"  [Tool] run_ablation on: {args['prompt'][:50]}... (top_k={top_k})")
            result = await tool_run_ablation(layer, neuron_idx, args["prompt"])

            # Extract promotes/suppresses from logit_shifts for dashboard visualization
            # NOTE: Ablation semantics are INVERTED from steering:
            # - Negative shift after ablation = neuron was PROMOTING this token (neuron gone → logit drops)
            # - Positive shift after ablation = neuron was SUPPRESSING this token (neuron gone → logit rises)
            logit_shifts = result.get("logit_shifts", {})
            promotes = sorted(
                [(token, -shift) for token, shift in logit_shifts.items() if shift < 0],  # Invert sign for display
                key=lambda x: -x[1]
            )[:10]
            suppresses = sorted(
                [(token, -shift) for token, shift in logit_shifts.items() if shift > 0],  # Invert sign for display
                key=lambda x: x[1]
            )[:10]

            scientist.investigation.ablation_effects.append({
                "prompt": result.get("prompt", ""),
                "most_affected": result.get("most_affected_token", ""),
                "max_shift": result.get("max_shift", 0),
                "promotes": promotes,
                "suppresses": suppresses,
            })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_ablation", "Run ablation on multiple prompts. Pass prompts as JSON array string.", {"prompts": str})
        async def batch_ablation_tool(args):
            scientist.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]

            print(f"  [Tool] batch_ablation on {len(prompts)} prompts")
            result = await tool_batch_ablation(layer, neuron_idx, prompts)

            # Capture individual ablation results with prompts for dashboard
            for ir in result.get("individual_results", []):
                scientist.investigation.ablation_effects.append({
                    "prompt": ir.get("prompt", ""),
                    "most_affected": ir.get("most_affected", ""),
                    "max_shift": ir.get("max_shift", 0),
                    "promotes": ir.get("promotes", []),
                    "suppresses": ir.get("suppresses", []),
                })

            # Also store the consistent patterns as a summary entry
            if result.get("consistent_promotes") or result.get("consistent_suppresses"):
                scientist.investigation.ablation_effects.append({
                    "prompt": f"[Summary of {len(prompts)} prompts]",
                    "promotes": result.get("consistent_promotes", []),
                    "suppresses": result.get("consistent_suppresses", []),
                })

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_relp_connectivity", "Get RelP-based connectivity from aggregated edge statistics. Shows CONTEXT-SPECIFIC connections observed in RelP runs (informational only - does NOT populate dashboard connectivity). Compare with wiring analysis to see which weight-based connections are actually used.", {})
        async def get_relp_connectivity_tool(args):
            scientist.experiments_run += 1
            print("  [Tool] get_relp_connectivity")
            if not edge_stats_path:
                return {"content": [{"type": "text", "text": "No edge stats path provided"}]}
            result = await tool_get_relp_connectivity(layer, neuron_idx, edge_stats_path)
            # NOTE: This is informational only - does NOT set investigation.connectivity
            # Use analyze_wiring/analyze_output_wiring for dashboard connectivity
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("save_report", "Save the final investigation report with findings. Pass report as JSON string.", {"report": str})
        async def save_report_tool(args):
            print("  [Tool] save_report")
            try:
                report = json.loads(args["report"])
            except json.JSONDecodeError:
                report = {"summary": args["report"]}

            result = await tool_save_report(neuron_id, report, output_dir)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("run_relp", "Run RelP attribution on a prompt. Check if neuron appears in graph and its edges. Use tau=0.05 for fast exploration, tau=0.01 for detailed analysis. Set timeout (default 120s) to limit run time.", {"prompt": str, "target_tokens": str, "tau": float, "k": int, "timeout": float})
        async def run_relp_tool(args):
            prompt = args["prompt"]
            target_tokens = None
            if args.get("target_tokens"):
                try:
                    target_tokens = json.loads(args["target_tokens"])
                except json.JSONDecodeError:
                    target_tokens = [args["target_tokens"]]
            tau = args.get("tau", 0.02)  # Default to moderate tau for reasonable speed
            k = args.get("k", 5)
            timeout = args.get("timeout", 120.0)  # Configurable timeout

            # Check if similar RelP was already run in prior iteration
            # Consider a match if same prompt and similar tau (within 50% tolerance)
            for prior_run in scientist.investigation.relp_results:
                prior_prompt = prior_run.get("prompt", "")
                prior_tau = prior_run.get("tau", 0)
                # Check for exact prompt match and similar tau
                if prior_prompt.startswith(prompt[:150]) or prompt.startswith(prior_prompt[:150]):
                    if prior_tau > 0 and abs(tau - prior_tau) / prior_tau < 0.5:
                        # Return cached result
                        cached = {
                            "status": "cached",
                            "note": f"Similar RelP run found in prior iteration (tau={prior_tau})",
                            "prompt": prompt[:100],
                            "tau_requested": tau,
                            "tau_cached": prior_tau,
                            "neuron_found": prior_run.get("neuron_found", False),
                            "neuron_relp_score": prior_run.get("neuron_relp_score"),
                            "downstream_edges": prior_run.get("downstream_edges", []),
                            "upstream_edges": prior_run.get("upstream_edges", []),
                            "in_causal_pathway": prior_run.get("in_causal_pathway", False),
                        }
                        print(f"  [Tool] run_relp CACHED (prior tau={prior_tau}): {prompt[:50]}...")
                        return {"content": [{"type": "text", "text": json.dumps(cached, indent=2)}]}

            # No cached result - run new RelP
            scientist.experiments_run += 1
            print(f"  [Tool] run_relp (tau={tau}, timeout={timeout}s) on: {prompt[:50]}...")
            result = await tool_run_relp(layer, neuron_idx, prompt, target_tokens, tau=tau, k=k, timeout=timeout)
            # Track RelP results for dashboard - include target tokens for context
            # Store results if neuron found, has edges, OR if there was an error (to track attempts)
            if result.get("neuron_found") or result.get("downstream_edges") or result.get("upstream_edges") or result.get("error"):
                scientist.investigation.relp_results.append({
                    "prompt": prompt[:200],
                    "target_tokens": target_tokens,  # What tokens were traced
                    "tau": tau,
                    "neuron_found": result.get("neuron_found", False),
                    "neuron_relp_score": result.get("neuron_relp_score"),
                    "downstream_edges": result.get("downstream_edges", [])[:5],  # Top 5
                    "upstream_edges": result.get("upstream_edges", [])[:5],
                    "graph_stats": result.get("graph_stats", {}),
                    "in_causal_pathway": result.get("neuron_found", False) and len(result.get("downstream_edges", [])) > 0,
                    "error": result.get("error"),  # Store errors too
                })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("verify_downstream_connections", "Verify neuron connects to expected downstream targets across prompts. Use tau=0.005 for accurate verification.", {"prompts": str, "expected_downstream": str, "tau": float})
        async def verify_downstream_tool(args):
            scientist.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]
            try:
                expected = json.loads(args["expected_downstream"])
            except json.JSONDecodeError:
                expected = [e.strip() for e in args["expected_downstream"].split(",") if e.strip()]
            tau = args.get("tau", 0.005)  # Lower default for accurate edge verification
            print(f"  [Tool] verify_downstream_connections (tau={tau}) on {len(prompts)} prompts, checking {len(expected)} targets")
            result = await tool_verify_downstream_connections(layer, neuron_idx, prompts, expected, tau=tau)

            # BUG FIX: Store verification results to relp_results
            # The underlying tool_verify_downstream_connections calls tool_run_relp directly,
            # bypassing the agent wrapper that stores results. Store a summary here.
            if result.get("source_found_count", 0) > 0:
                scientist.investigation.relp_results.append({
                    "prompt": f"verify_downstream on {len(prompts)} prompts",
                    "target_tokens": expected,
                    "tau": tau,
                    "neuron_found": True,
                    "verification_summary": {
                        "prompts_tested": result.get("prompts_tested", 0),
                        "source_found_count": result.get("source_found_count", 0),
                        "downstream_verification": result.get("downstream_verification", {}),
                    },
                    "in_causal_pathway": result.get("source_found_count", 0) > 0,
                })

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("steer_neuron", "Add a value to the neuron activation and measure logit shifts. Try multiple strengths (±1, ±3, ±5, ±10). Optional: top_k_logits (default 10).", {"prompt": str, "steering_value": float, "position": int, "top_k_logits": int})
        async def steer_neuron_tool(args):
            scientist.experiments_run += 1
            prompt = args["prompt"]
            steering_value = args.get("steering_value", 1.0)
            position = args.get("position", -1)
            top_k = args.get("top_k_logits", 10)
            print(f"  [Tool] steer_neuron (value={steering_value}, top_k={top_k}) on: {prompt[:50]}...")
            result = await tool_steer_neuron(layer, neuron_idx, prompt, steering_value, position=position, top_k_logits=top_k)
            # Track individual steering results in investigation
            scientist.investigation.steering_results.append({
                "prompt": prompt[:200],
                "steering_value": steering_value,
                "position": position,
                # Tool returns promoted_tokens/suppressed_tokens, not promotes/suppresses
                "promotes": result.get("promoted_tokens", [])[:10],
                "suppresses": result.get("suppressed_tokens", [])[:10],
                "max_shift": result.get("max_shift", 0),
                "top_k_logits": top_k,
            })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("patch_activation", "Patch neuron activation from source prompt into target prompt (counterfactual).", {"source_prompt": str, "target_prompt": str, "position": int})
        async def patch_activation_tool(args):
            scientist.experiments_run += 1
            source = args["source_prompt"]
            target = args["target_prompt"]
            position = args.get("position", -1)
            print(f"  [Tool] patch_activation from '{source[:30]}...' to '{target[:30]}...'")
            result = await tool_patch_activation(layer, neuron_idx, source, target, position=position)

            # Store patching experiment in investigation
            source_acts = result.get("source_activations", {})
            target_acts = result.get("target_activations", {})
            scientist.investigation.patching_experiments.append({
                "source_prompt": source[:200],
                "target_prompt": target[:200],
                "position": position,
                "source_activation": sum(source_acts.values()) / len(source_acts) if source_acts else 0,
                "target_activation": sum(target_acts.values()) / len(target_acts) if target_acts else 0,
                "activation_delta": result.get("activation_delta", {}),
                "baseline_logits": result.get("baseline_logits", {}),
                "patched_logits": result.get("patched_logits", {}),
                "promoted_tokens": result.get("promoted_tokens", []),
                "suppressed_tokens": result.get("suppressed_tokens", []),
                "max_shift": result.get("max_shift", 0),
            })

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("run_category_selectivity_test", "Run comprehensive category selectivity test. Tests neuron across corpus (unrelated domains) + generated domain-specific prompts. Returns per-category z-scores and selectivity summary. Use this to verify neuron is genuinely selective, not just high-baseline.", {"target_domain": str, "target_categories": str, "inhibitory_categories": str, "n_generated_per_category": int, "include_corpus": bool})
        async def run_category_selectivity_test_tool(args):
            scientist.experiments_run += 1
            target_domain = args["target_domain"]

            # Parse target_categories (JSON array or comma-separated)
            try:
                target_categories = json.loads(args["target_categories"])
            except json.JSONDecodeError:
                target_categories = [c.strip() for c in args["target_categories"].split(",") if c.strip()]

            # Parse inhibitory_categories
            inhibitory_categories = None
            if args.get("inhibitory_categories"):
                try:
                    inhibitory_categories = json.loads(args["inhibitory_categories"])
                except json.JSONDecodeError:
                    inhibitory_categories = [c.strip() for c in args["inhibitory_categories"].split(",") if c.strip()]

            n_generated = args.get("n_generated_per_category", 30)
            include_corpus = args.get("include_corpus", True)

            print(f"  [Tool] run_category_selectivity_test: domain='{target_domain}', targets={target_categories}, n_gen={n_generated}")

            result = await tool_run_category_selectivity_test(
                layer, neuron_idx,
                target_domain=target_domain,
                target_categories=target_categories,
                inhibitory_categories=inhibitory_categories,
                include_corpus=include_corpus,
                n_generated_per_category=n_generated,
            )

            # Merge new run into accumulated selectivity data (single dict, not list)
            from .tools import assess_selectivity_quality, merge_selectivity_runs

            existing_sel = scientist.investigation.category_selectivity_data
            if existing_sel:
                # Merge new run into existing accumulated data
                merged = merge_selectivity_runs([existing_sel, result])
            else:
                merged = result
            scientist.investigation.category_selectivity_data = merged

            # Union-merge categorized_prompts (don't overwrite prior runs)
            if "categories" in result:
                for cat_name, cat_data in result["categories"].items():
                    existing = scientist.investigation.categorized_prompts.get(cat_name, [])
                    existing_texts = {p.get("prompt", "") for p in existing}
                    new_prompts = [
                        {
                            "prompt": p["prompt"],
                            "activation": p["activation"],
                            "z_score": p["z_score"],
                            "position": p.get("position", -1),
                            "token": p.get("token", ""),
                        }
                        for p in cat_data.get("prompts", [])
                        if p.get("prompt", "") not in existing_texts
                    ]
                    scientist.investigation.categorized_prompts[cat_name] = existing + new_prompts

            quality = assess_selectivity_quality(merged)

            # In negative polarity mode, present negative stats as primary
            is_negative_mode = scientist.polarity_mode == "negative"

            # Build summary for agent — use MERGED data so agent sees cumulative picture
            if is_negative_mode:
                summary = {
                    "total_prompts": merged.get("total_prompts", 0),
                    "global_mean": merged.get("neg_global_mean", result.get("neg_global_mean", 0)),
                    "global_std": merged.get("neg_global_std", result.get("neg_global_std", 0)),
                    "selectivity_summary": merged.get("selectivity_summary", ""),
                    "polarity_summary": merged.get("polarity_summary", result.get("polarity_summary", "")),
                    "category_stats": {},
                    "top_activating": merged.get("top_negatively_activating", result.get("top_negatively_activating", []))[:15],
                    "polarity_mode": "negative",
                    "note": "Activations shown are NEGATIVE (most negative = strongest firing for this mode)",
                }
            else:
                summary = {
                    "total_prompts": merged.get("total_prompts", 0),
                    "global_mean": merged.get("global_mean", 0),
                    "global_std": merged.get("global_std", 0),
                    "selectivity_summary": merged.get("selectivity_summary", ""),
                    "polarity_summary": merged.get("polarity_summary", result.get("polarity_summary", "")),
                    "category_stats": {},
                    "top_activating": merged.get("top_activating", [])[:15],
                    "top_negatively_activating": merged.get("top_negatively_activating", result.get("top_negatively_activating", []))[:5],
                }

            # Add quality assessment warnings
            if quality.get("warnings"):
                summary["quality_warnings"] = quality["warnings"]
                summary["quality_score"] = quality["quality_score"]
                summary["is_informative"] = quality["is_informative"]

            # Add per-category summary (without full prompt lists) from MERGED data
            for cat_name, cat_data in merged.get("categories", {}).items():
                cat_summary = {
                    "type": cat_data.get("type", "unknown"),
                    "count": cat_data.get("count", 0),
                    "mean": cat_data.get("mean", 0),
                    "z_mean": cat_data.get("z_mean", 0),
                    "z_std": cat_data.get("z_std", 0),
                    "neg_mean": cat_data.get("neg_mean", 0),
                    "neg_z_mean": cat_data.get("neg_z_mean", 0),
                }
                if is_negative_mode:
                    cat_summary["mean"] = cat_data.get("neg_mean", 0)
                    cat_summary["z_mean"] = cat_data.get("neg_z_mean", 0)
                    cat_summary["z_std"] = cat_data.get("neg_z_std", 0)
                    cat_summary["pos_mean"] = cat_data.get("mean", 0)
                    cat_summary["pos_z_mean"] = cat_data.get("z_mean", 0)
                summary["category_stats"][cat_name] = cat_summary

            # Seed activating_prompts from selectivity top activators (unbiased source)
            # This ensures the evidence section reflects selectivity data, not just agent probes
            if is_negative_mode:
                # In negative mode, seed from top negatively activating
                if "top_negatively_activating" in result:
                    existing_texts = {p.get("prompt", "") for p in scientist.investigation.activating_prompts}
                    for p in result["top_negatively_activating"][:20]:
                        prompt_text = p.get("prompt", "")
                        if prompt_text and prompt_text not in existing_texts:
                            scientist.investigation.activating_prompts.append({
                                "prompt": prompt_text,
                                "activation": p.get("activation", 0),
                                "token": p.get("token", ""),
                                "position": p.get("position"),
                                "source": "category_selectivity_negative",
                            })
                            existing_texts.add(prompt_text)
            else:
                if "top_activating" in result:
                    existing_texts = {p.get("prompt", "") for p in scientist.investigation.activating_prompts}
                    for p in result["top_activating"][:20]:
                        prompt_text = p.get("prompt", "")
                        if prompt_text and prompt_text not in existing_texts:
                            scientist.investigation.activating_prompts.append({
                                "prompt": prompt_text,
                                "activation": p.get("activation", 0),
                                "token": p.get("token", ""),
                                "position": p.get("position"),
                                "source": "category_selectivity",
                            })
                            existing_texts.add(prompt_text)

            # Always seed negatively_activating_prompts from selectivity (for both modes)
            if "top_negatively_activating" in result:
                existing_neg_texts = {p.get("prompt", "") for p in scientist.investigation.negatively_activating_prompts}
                for p in result["top_negatively_activating"][:20]:
                    prompt_text = p.get("prompt", "")
                    if prompt_text and prompt_text not in existing_neg_texts:
                        scientist.investigation.negatively_activating_prompts.append({
                            "prompt": prompt_text,
                            "activation": p.get("activation", 0),
                            "token": p.get("token", ""),
                            "position": p.get("position"),
                            "source": "category_selectivity",
                        })
                        existing_neg_texts.add(prompt_text)

            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        @tool("test_additional_prompts", "Test follow-up prompts and merge into existing category selectivity data. Use this to probe specific patterns, test minimal pairs, or add targeted adversarial tests after seeing initial selectivity results. Results are accumulated — z-scores update across the full prompt set.", {"prompts": str, "category": str, "category_type": str})
        async def test_additional_prompts_tool(args):
            scientist.experiments_run += 1

            # Parse prompts (JSON array or newline-separated)
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]

            category = args.get("category", "follow_up")
            category_type = args.get("category_type", "target")

            print(f"  [Tool] test_additional_prompts: {len(prompts)} prompts in category '{category}' ({category_type})")

            result = await tool_test_additional_prompts(
                layer, neuron_idx,
                prompts=prompts,
                category=category,
                category_type=category_type,
            )

            # Merge into investigation's categorized prompts
            if "per_prompt_results" in result:
                cat_name = result.get("category", f"gen_{category}")
                existing = scientist.investigation.categorized_prompts.get(cat_name, [])
                existing_texts = {p.get("prompt", "") for p in existing if isinstance(p, dict)}
                for p in result["per_prompt_results"]:
                    if p.get("prompt") and p["prompt"] not in existing_texts:
                        existing.append(p)
                        existing_texts.add(p["prompt"])
                scientist.investigation.categorized_prompts[cat_name] = existing

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_neuron_label", "Look up the label for any neuron. Returns function/input labels if available.", {"neuron_id": str})
        async def get_neuron_label_tool(args):
            nid = args["neuron_id"]
            print(f"  [Tool] get_neuron_label: {nid}")
            result = await tool_get_neuron_label(nid)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_get_neuron_labels", "Look up labels for multiple neurons at once. Pass neuron_ids as JSON array.", {"neuron_ids": str})
        async def batch_get_neuron_labels_tool(args):
            try:
                neuron_ids = json.loads(args["neuron_ids"])
            except json.JSONDecodeError:
                neuron_ids = [n.strip() for n in args["neuron_ids"].split(",") if n.strip()]
            print(f"  [Tool] batch_get_neuron_labels: {len(neuron_ids)} neurons")
            result = await tool_batch_get_neuron_labels(neuron_ids)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("run_baseline_comparison", "Compare neuron effects against random neurons. Returns z-score to calibrate effect sizes. z>2 is meaningful.", {"prompts": str, "n_random_neurons": int})
        async def run_baseline_comparison_tool(args):
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split(",") if p.strip()]
            n_random = args.get("n_random_neurons", 3)
            print(f"  [Tool] run_baseline_comparison: {len(prompts)} prompts, {n_random} random neurons")
            result = await tool_run_baseline_comparison(layer, neuron_idx, prompts, n_random)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("adaptive_relp", "Run RelP with adaptive tau - automatically finds the right detail level. Starts coarse and refines.", {"prompt": str, "target_tokens": str, "max_time": float})
        async def adaptive_relp_tool(args):
            scientist.experiments_run += 1
            prompt = args["prompt"]
            target_tokens = None
            if args.get("target_tokens"):
                try:
                    target_tokens = json.loads(args["target_tokens"])
                except json.JSONDecodeError:
                    target_tokens = [t.strip() for t in args["target_tokens"].split(",") if t.strip()]
            max_time = args.get("max_time", 60.0)
            print(f"  [Tool] adaptive_relp: '{prompt[:40]}...' max_time={max_time}s")
            result = await tool_adaptive_relp(layer, neuron_idx, prompt, target_tokens=target_tokens, max_time=max_time)

            # BUG FIX: Store adaptive_relp results to relp_results
            if result.get("neuron_found") or result.get("downstream_edges") or result.get("upstream_edges") or result.get("error"):
                scientist.investigation.relp_results.append({
                    "prompt": prompt[:200],
                    "target_tokens": target_tokens,
                    "tau": result.get("final_tau") or result.get("tau_used"),
                    "neuron_found": result.get("neuron_found", False),
                    "neuron_relp_score": result.get("neuron_relp_score"),
                    "downstream_edges": result.get("downstream_edges", [])[:5],
                    "upstream_edges": result.get("upstream_edges", [])[:5],
                    "graph_stats": result.get("graph_stats", {}),
                    "in_causal_pathway": result.get("neuron_found", False) and len(result.get("downstream_edges", [])) > 0,
                    "error": result.get("error"),
                    "adaptive": True,  # Mark as adaptive RelP
                })

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("steer_dose_response", "Run steering at multiple values to see dose-response curve. Shows linear vs threshold effects.", {"prompt": str, "steering_values": str, "position": int})
        async def steer_dose_response_tool(args):
            scientist.experiments_run += 1
            prompt = args["prompt"]
            steering_values = None
            if args.get("steering_values"):
                try:
                    steering_values = json.loads(args["steering_values"])
                except json.JSONDecodeError:
                    steering_values = [float(v.strip()) for v in args["steering_values"].split(",") if v.strip()]
            position = args.get("position", -1)
            print(f"  [Tool] steer_dose_response: '{prompt[:40]}...' values={steering_values or 'default'}")
            result = await tool_steer_dose_response(layer, neuron_idx, prompt, steering_values=steering_values, position=position)
            # Track dose-response results in investigation
            scientist.investigation.dose_response_results.append({
                "prompt": prompt,
                "steering_values": result.get("steering_values", steering_values or [-10, -5, -2, 0, 2, 5, 10]),
                "pattern": result.get("pattern"),
                "kendall_tau": result.get("kendall_tau"),
                "is_monotonic": result.get("is_monotonic"),
                "responsive_tokens": result.get("responsive_tokens", [])[:10],
                "dose_response_curve": result.get("dose_response_curve", []),
            })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("register_hypothesis", "REQUIRED: Register a hypothesis BEFORE testing it. This prevents p-hacking by creating an auditable record.", {"hypothesis": str, "confirmation_criteria": str, "refutation_criteria": str, "prior_probability": int, "hypothesis_type": str})
        async def register_hypothesis_tool(args):
            hypothesis = args["hypothesis"]
            confirmation = args["confirmation_criteria"]
            refutation = args["refutation_criteria"]
            prior = args.get("prior_probability", 50)
            h_type = args.get("hypothesis_type", "activation")
            print(f"  [Tool] register_hypothesis: '{hypothesis[:50]}...' (prior={prior}%)")
            result = await tool_register_hypothesis(hypothesis, confirmation, refutation, prior, h_type)
            scientist.investigation.hypotheses_tested.append({
                "hypothesis_id": result["hypothesis_id"],
                "hypothesis": hypothesis,
                "prior_probability": prior,
                "status": "registered",
            })
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("update_hypothesis_status", "Update hypothesis status after testing. Call this after experiments to record your conclusion.", {"hypothesis_id": str, "status": str, "posterior_probability": int, "evidence_summary": str})
        async def update_hypothesis_status_tool(args):
            h_id = args["hypothesis_id"]
            status = args["status"]
            posterior = args.get("posterior_probability", 50)
            evidence = args.get("evidence_summary", "")
            print(f"  [Tool] update_hypothesis_status: {h_id} -> {status} (posterior={posterior}%)")
            result = await tool_update_hypothesis_status(h_id, status, posterior, evidence)
            # Update investigation record
            for h in scientist.investigation.hypotheses_tested:
                if h.get("hypothesis_id") == h_id:
                    h["status"] = status
                    h["posterior_probability"] = posterior
                    h["evidence_summary"] = evidence
                    break
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_hypothesis_summary", "Get current state of all hypotheses, sorted by posterior probability. Shows the leading INPUT and OUTPUT hypotheses. Call this before steering to ensure your hypothesis is grounded in evidence.", {})
        async def get_hypothesis_summary_tool(args):
            registry = get_hypothesis_registry()
            if not registry:
                return {"content": [{"type": "text", "text": json.dumps({"message": "No hypotheses registered yet."})}]}

            # Separate by type and sort by posterior (highest first)
            input_hyps = []
            output_hyps = []
            other_hyps = []
            for h in registry:
                status = h.get("status", "registered")
                if status in ("refuted",):
                    continue  # Skip refuted
                entry = {
                    "id": h.get("hypothesis_id"),
                    "hypothesis": h.get("hypothesis", "")[:150],
                    "type": h.get("hypothesis_type", "activation"),
                    "status": status,
                    "prior": h.get("prior_probability", 50),
                    "posterior": h.get("posterior_probability") or h.get("prior_probability", 50),
                }
                if h.get("hypothesis_type") == "output":
                    output_hyps.append(entry)
                elif h.get("hypothesis_type") in ("activation", "input"):
                    input_hyps.append(entry)
                else:
                    other_hyps.append(entry)

            input_hyps.sort(key=lambda x: x["posterior"], reverse=True)
            output_hyps.sort(key=lambda x: x["posterior"], reverse=True)

            summary = {
                "leading_input_hypothesis": input_hyps[0] if input_hyps else None,
                "leading_output_hypothesis": output_hyps[0] if output_hyps else None,
                "all_active_input": input_hyps[:5],
                "all_active_output": output_hyps[:5],
                "other": other_hyps[:3],
                "total_registered": len(registry),
                "total_refuted": sum(1 for h in registry if h.get("status") == "refuted"),
            }
            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        @tool("get_output_projections", "Get actual output projections from model weights. Shows what tokens neuron promotes/suppresses.", {"top_k": int})
        async def get_output_projections_tool(args):
            top_k = args.get("top_k", 10)
            print(f"  [Tool] get_output_projections (top_k={top_k}, polarity={scientist.polarity_mode})")
            result = await tool_get_output_projections(layer, neuron_idx, top_k=top_k, polarity_mode=scientist.polarity_mode)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("find_graphs_for_neuron", "Find pre-computed RelP graphs where this neuron appears. Fast lookup from indexed corpus of ~40k graphs.", {"limit": int, "min_influence": float})
        async def find_graphs_for_neuron_tool(args):
            limit = args.get("limit", 50)
            min_influence = args.get("min_influence", 0.0)
            print(f"  [Tool] find_graphs_for_neuron (limit={limit}, min_influence={min_influence})")
            result = await tool_find_graphs_for_neuron(layer, neuron_idx, limit=limit, min_influence=min_influence)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_neuron_graph_stats", "Get statistics about neuron's presence across indexed graphs. Shows frequency, influence distribution, and co-occurring neurons.", {})
        async def get_neuron_graph_stats_tool(args):
            print("  [Tool] get_neuron_graph_stats")
            result = await tool_get_neuron_graph_stats(layer, neuron_idx)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("load_graph_from_index", "Load a specific RelP graph file to inspect its full structure. Use after find_graphs_for_neuron.", {"graph_path": str})
        async def load_graph_from_index_tool(args):
            graph_path = args.get("graph_path", "")
            print(f"  [Tool] load_graph_from_index: {graph_path}")
            result = await tool_load_graph_from_index(graph_path)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("log_categorized_activation", "Log activation with category for visualization. Categories: tech, medical, financial, legal, nature, etc.", {"prompt": str, "activation": float, "category": str, "position": int, "token": str})
        async def log_categorized_activation_tool(args):
            prompt = args["prompt"]
            activation = args["activation"]
            category = args["category"]
            position = args.get("position", -1)
            token = args.get("token", "")
            print(f"  [Tool] log_categorized_activation: category={category}, activation={activation:.2f}")

            # Store in investigation
            if category not in scientist.investigation.categorized_prompts:
                scientist.investigation.categorized_prompts[category] = []
            scientist.investigation.categorized_prompts[category].append({
                "prompt": prompt,
                "activation": activation,
                "position": position,
                "token": token,
            })

            return {"content": [{"type": "text", "text": json.dumps({"logged": True, "category": category, "count": len(scientist.investigation.categorized_prompts[category])})}]}

        @tool("log_homograph_test", "Log homograph/polysemy test result. Tests if neuron disambiguates word meanings.", {"word": str, "context_label": str, "example": str, "activation": float, "category": str})
        async def log_homograph_test_tool(args):
            word = args["word"]
            context_label = args["context_label"]
            example = args["example"]
            activation = args["activation"]
            category = args["category"]
            print(f"  [Tool] log_homograph_test: word={word}, context={context_label}, activation={activation:.2f}")

            # Find or create word entry
            word_entry = next((h for h in scientist.investigation.homograph_tests if h["word"] == word), None)
            if not word_entry:
                word_entry = {"word": word, "contexts": []}
                scientist.investigation.homograph_tests.append(word_entry)

            word_entry["contexts"].append({
                "label": context_label,
                "example": example,
                "activation": activation,
                "category": category,
            })

            return {"content": [{"type": "text", "text": json.dumps({"logged": True, "word": word, "context": context_label, "total_contexts": len(word_entry["contexts"])})}]}

        # Load prior knowledge for auto-injection
        # Use cached version if available (includes auto-computed output projections)
        prior_knowledge = getattr(self, '_prior_knowledge_cache', None) or self._load_prior_knowledge()

        @tool("save_structured_report", "REQUIRED: Save final structured report. Call this at the END of investigation with all findings.", {
            "input_function": str,
            "output_function": str,
            "function_type": str,
            "summary": str,
            "key_findings": str,  # JSON array
            "open_questions": str,  # JSON array
            "confidence": str,
            "activating_patterns": str,  # JSON array - include 15-20 top examples
            "non_activating_patterns": str,  # JSON array - include 10-15 controls
            "ablation_promotes": str,  # JSON array
            "ablation_suppresses": str,  # JSON array
            "upstream_neurons": str,  # JSON array
            "downstream_neurons": str,  # JSON array
            "original_output_label": str,  # Optional: full label from prior
            "original_input_label": str,  # Optional: full label from prior
            "original_output_description": str,  # Optional: full description
            "original_input_description": str,  # Optional: full description
            "output_projections_promote": str,  # Optional: JSON array
            "output_projections_suppress": str,  # Optional: JSON array
            "ablation_details": str,  # Optional: JSON array of detailed ablation results
            "steering_details": str,  # Optional: JSON array of detailed steering results
            "relp_results": str,  # Optional: JSON array of RelP attribution findings
            "baseline_zscore": float,  # Optional: Z-score from run_baseline_comparison
        })
        async def save_structured_report_tool(args):
            print("  [Tool] save_structured_report")
            # Parse JSON arrays
            def parse_json_array(s, default=[]):
                try:
                    return json.loads(s) if s else default
                except:
                    return default

            def _merge_relp_results(prior_results, new_results):
                """Merge prior and new RelP results, deduplicating by (prompt, tau) key."""
                if not prior_results:
                    return new_results or []
                if not new_results:
                    return prior_results or []

                # Create a dict keyed by (prompt, tau) to deduplicate
                merged = {}
                for r in prior_results:
                    key = (r.get("prompt", ""), r.get("tau", 0.01))
                    merged[key] = r
                for r in new_results:
                    key = (r.get("prompt", ""), r.get("tau", 0.01))
                    # New results override prior (in case of re-runs)
                    merged[key] = r
                return list(merged.values())

            # Auto-inject prior knowledge if agent didn't provide it
            # Access _prior_knowledge_cache dynamically to get auto-computed output projections
            # (the cache is updated in investigate() after tools are built)
            cached_prior = getattr(scientist, '_prior_knowledge_cache', None) or prior_knowledge
            llm_labels = cached_prior.get("llm_labels", {})
            proj = cached_prior.get("output_projections", {})

            # Auto-inject ALL collected experimental data (override agent's subset if provided)
            # This ensures we save everything the agent tested, not just what it chose to include
            activating_patterns_arg = parse_json_array(args.get("activating_patterns", "[]"))
            non_activating_patterns_arg = parse_json_array(args.get("non_activating_patterns", "[]"))

            # Use all collected data from investigation tracker
            all_activating = scientist.investigation.activating_prompts
            all_non_activating = scientist.investigation.non_activating_prompts

            # If agent provided data, prefer it, otherwise use all collected
            activating_to_save = activating_patterns_arg if activating_patterns_arg else all_activating
            non_activating_to_save = non_activating_patterns_arg if non_activating_patterns_arg else all_non_activating

            # If agent only provided a subset, merge with collected data to get the full set
            if len(activating_patterns_arg) < len(all_activating):
                print(f"  Auto-expanding: agent provided {len(activating_patterns_arg)}, using all {len(all_activating)} collected")
                activating_to_save = all_activating

            result = await tool_save_structured_report(
                neuron_id=neuron_id,
                layer=layer,
                neuron_idx=neuron_idx,
                input_function=args.get("input_function", ""),
                output_function=args.get("output_function", ""),
                function_type=args.get("function_type", "unknown"),
                summary=args.get("summary", ""),
                key_findings=parse_json_array(args.get("key_findings", "[]")),
                open_questions=parse_json_array(args.get("open_questions", "[]")),
                confidence=args.get("confidence", "medium"),
                activating_patterns=activating_to_save,
                non_activating_patterns=non_activating_to_save,
                ablation_promotes=parse_json_array(args.get("ablation_promotes", "[]")),
                ablation_suppresses=parse_json_array(args.get("ablation_suppresses", "[]")),
                # Auto-inject connectivity from wiring analysis (analyze_wiring/analyze_output_wiring) if agent passed empty
                # Support both "neuron_id" (new) and "neuron" (legacy) field names for backwards compatibility
                upstream_neurons=parse_json_array(args.get("upstream_neurons", "[]")) or [
                    {"neuron_id": u.get("neuron_id") or u.get("neuron"), "label": u.get("label", ""), "weight": u.get("weight", 0)}
                    for u in scientist.investigation.connectivity.get("upstream_neurons", [])
                ],
                # Handle both neuron targets (neuron_id) and logit targets (target field like "LOGIT(token)")
                downstream_neurons=parse_json_array(args.get("downstream_neurons", "[]")) or [
                    {"neuron_id": d.get("neuron_id") or d.get("neuron") or d.get("target", ""), "label": d.get("label", ""), "weight": d.get("weight", 0)}
                    for d in scientist.investigation.connectivity.get("downstream_targets", [])
                ],
                total_experiments=scientist.experiments_run,
                output_dir=output_dir,
                # Auto-inject prior analysis data if not provided by agent
                # Use explicit None checks to avoid overwriting valid falsy values (0, [], "")
                original_output_label=args.get("original_output_label") if args.get("original_output_label") is not None else llm_labels.get("output_label", ""),
                original_input_label=args.get("original_input_label") if args.get("original_input_label") is not None else llm_labels.get("input_label", ""),
                original_output_description=args.get("original_output_description") if args.get("original_output_description") is not None else llm_labels.get("output_description", ""),
                original_input_description=args.get("original_input_description") if args.get("original_input_description") is not None else llm_labels.get("input_description", ""),
                # Auto-inject output projections if agent passed empty array
                output_projections_promote=parse_json_array(args.get("output_projections_promote", "[]")) or proj.get("promote", []),
                output_projections_suppress=parse_json_array(args.get("output_projections_suppress", "[]")) or proj.get("suppress", []),
                # Auto-inject ablation results from investigation tracker if agent didn't provide
                ablation_details=parse_json_array(args.get("ablation_details", "[]")) or scientist.investigation.ablation_effects,
                # Auto-inject steering results from investigation tracker if agent didn't provide
                steering_details=parse_json_array(args.get("steering_details", "[]")) or scientist.investigation.steering_results,
                # Auto-inject all collected RelP results - merge ALL sources:
                # 1. Prior iteration results (if revision)
                # 2. Current investigation relp_results (includes corpus evidence from Phase 0)
                # 3. New agent-run RelP results or explicit results from args
                relp_results=_merge_relp_results(
                    prior_results=_merge_relp_results(
                        prior_results=scientist.prior_investigation.relp_results if scientist.prior_investigation else [],
                        new_results=scientist.investigation.relp_results,  # Includes corpus RelP!
                    ),
                    new_results=parse_json_array(args.get("relp_results", "[]")) or get_relp_registry(),
                ),
                # Validation metrics
                baseline_zscore=args.get("baseline_zscore"),
                # Auto-inject visualization data from investigation tracker
                categorized_prompts=scientist.investigation.categorized_prompts or {},
                homograph_tests=scientist.investigation.homograph_tests or [],
                category_selectivity_data=scientist.investigation.category_selectivity_data or {},
                # Pass polarity mode so the saved JSON includes it
                polarity_mode=scientist.polarity_mode,
            )

            # BUG FIX: Update in-memory investigation object with characterization fields
            # This ensures GPT review sees the characterization data (not just empty defaults)
            scientist.investigation.input_function = args.get("input_function", "") or scientist.investigation.input_function
            scientist.investigation.output_function = args.get("output_function", "") or scientist.investigation.output_function
            scientist.investigation.function_type = args.get("function_type", "") or scientist.investigation.function_type
            scientist.investigation.final_hypothesis = args.get("summary", "") or scientist.investigation.final_hypothesis
            scientist.investigation.key_findings = parse_json_array(args.get("key_findings", "[]")) or scientist.investigation.key_findings
            scientist.investigation.open_questions = parse_json_array(args.get("open_questions", "[]")) or scientist.investigation.open_questions
            # Update confidence (map string to float)
            conf_str = args.get("confidence", "medium").lower()
            conf_map = {"low": 0.4, "medium": 0.65, "high": 0.85}
            scientist.investigation.confidence = conf_map.get(conf_str, 0.5)
            # Update evidence
            scientist.investigation.activating_prompts = activating_to_save
            scientist.investigation.non_activating_prompts = non_activating_to_save
            # Sync RelP results - merge ALL sources:
            # 1. Prior iteration results (if revision)
            # 2. Current investigation relp_results (includes corpus evidence from Phase 0)
            # 3. New agent-run RelP results from registry
            # First merge prior + current (preserves corpus evidence)
            merged_prior_current = _merge_relp_results(
                prior_results=scientist.prior_investigation.relp_results if scientist.prior_investigation else [],
                new_results=scientist.investigation.relp_results,  # Includes corpus RelP from Phase 0!
            )
            # Then merge with new registry results
            scientist.investigation.relp_results = _merge_relp_results(
                prior_results=merged_prior_current,
                new_results=get_relp_registry(),
            )

            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        # =============================================================================
        # V4 Phase Tools - Multi-token generation and phase completion
        # =============================================================================

        @tool("ablate_and_generate", "V4 Output Phase: Ablate neuron and generate multiple tokens with greedy decoding. REQUIRED for Output Phase completion.", {"prompt": str, "max_new_tokens": int, "downstream_neurons": str, "top_k_logits": int})
        async def ablate_and_generate_tool(args):
            prompt = args["prompt"]
            max_new_tokens = args.get("max_new_tokens", 10)
            top_k_logits = args.get("top_k_logits", 10)
            # Parse downstream_neurons JSON array
            downstream_neurons = None
            if args.get("downstream_neurons"):
                try:
                    downstream_neurons = json.loads(args["downstream_neurons"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] ablate_and_generate on: {prompt[:50]}...")
            result = await tool_ablate_and_generate(
                layer, neuron_idx, prompt,
                max_new_tokens=max_new_tokens,
                downstream_neurons=downstream_neurons,
                top_k_logits=top_k_logits,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("steer_and_generate", "V4 Output Phase: Steer neuron and generate multiple tokens with greedy decoding.", {"prompt": str, "steering_value": float, "max_new_tokens": int, "downstream_neurons": str, "top_k_logits": int})
        async def steer_and_generate_tool(args):
            prompt = args["prompt"]
            steering_value = args.get("steering_value", 5.0)
            max_new_tokens = args.get("max_new_tokens", 10)
            top_k_logits = args.get("top_k_logits", 10)
            # Parse downstream_neurons JSON array
            downstream_neurons = None
            if args.get("downstream_neurons"):
                try:
                    downstream_neurons = json.loads(args["downstream_neurons"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] steer_and_generate on: {prompt[:50]}... (steering={steering_value})")
            result = await tool_steer_and_generate(
                layer, neuron_idx, prompt, steering_value,
                max_new_tokens=max_new_tokens,
                downstream_neurons=downstream_neurons,
                top_k_logits=top_k_logits,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_ablate_and_generate", "Unified ablation tool: Run ablation on prompts, generate completions, and check downstream neurons at ALL positions. Auto-uses downstream neurons from connectivity if available. generation_format: 'continuation' (default, prompt as assistant prefix—best for statements), 'chat' (prompt as user message—best for questions), 'raw' (no template).", {"use_categorized_prompts": bool, "prompts": str, "activation_threshold": float, "max_new_tokens": int, "max_prompts": int, "downstream_neurons": str, "truncate_to_activation": bool, "generation_format": str})
        async def batch_ablate_and_generate_tool(args):
            use_categorized = args.get("use_categorized_prompts", True)
            activation_threshold = args.get("activation_threshold", 0.5)
            max_new_tokens = args.get("max_new_tokens", 10)
            max_prompts = args.get("max_prompts", 300)
            truncate_to_activation = args.get("truncate_to_activation", False)
            generation_format = args.get("generation_format", "continuation")

            # Parse prompts JSON array if provided
            prompts = None
            if args.get("prompts"):
                try:
                    prompts = json.loads(args["prompts"])
                except json.JSONDecodeError:
                    pass

            # Parse downstream_neurons JSON array if provided
            downstream_neurons = None
            if args.get("downstream_neurons"):
                try:
                    downstream_neurons = json.loads(args["downstream_neurons"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] batch_ablate_and_generate (use_categorized={use_categorized}, threshold={activation_threshold})")
            result = await tool_batch_ablate_and_generate(
                layer, neuron_idx,
                prompts=prompts,
                use_categorized_prompts=use_categorized,
                activation_threshold=activation_threshold,
                max_new_tokens=max_new_tokens,
                max_prompts=max_prompts,
                downstream_neurons=downstream_neurons,
                truncate_to_activation=truncate_to_activation,
                generation_format=generation_format,
            )
            # Return summary including downstream dependency data
            summary = {
                "total_prompts": result.get("total_prompts", 0),
                "total_changed": result.get("total_changed", 0),
                "change_rate": result.get("change_rate", 0),
                "category_stats": result.get("category_stats", {}),
            }
            # Include downstream dependency summary if available
            if result.get("dependency_summary"):
                summary["dependency_summary"] = result["dependency_summary"]
                summary["downstream_neurons_checked"] = result.get("downstream_neurons_checked", [])
            # Include changed examples
            if result.get("per_prompt_results"):
                changed_examples = [
                    {
                        "prompt": r.get("prompt", "")[:100],
                        "baseline": r.get("baseline_completion", "")[:200],
                        "ablated": r.get("ablated_completion", "")[:200],
                    }
                    for r in result["per_prompt_results"]
                    if r.get("completion_changed")
                ][:5]  # Top 5 examples
                if changed_examples:
                    summary["changed_examples"] = changed_examples
            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        @tool("batch_steer_and_generate", "V4 Output Phase: Run steering on ALL activating prompts from category_selectivity. Tests MULTIPLE steering values for dose-response analysis. Efficient batched processing (~0.03s/prompt). Supports downstream_neurons monitoring to verify wiring predictions. **USE THIS WHEN ABLATION SHOWS WEAK EFFECTS (<20% change_rate)** — steering bypasses compensation that masks ablation results. generation_format: 'continuation' (default, prompt as assistant prefix—best for statements), 'chat' (prompt as user message—best for questions), 'raw' (no template).", {"use_categorized_prompts": bool, "prompts": str, "activation_threshold": float, "steering_values": str, "max_new_tokens": int, "batch_size": int, "max_prompts": int, "generation_format": str, "downstream_neurons": str})
        async def batch_steer_and_generate_tool(args):
            use_categorized = args.get("use_categorized_prompts", True)
            activation_threshold = args.get("activation_threshold", 0.5)
            max_new_tokens = args.get("max_new_tokens", 10)
            batch_size = args.get("batch_size", 8)
            max_prompts = args.get("max_prompts", 300)
            generation_format = args.get("generation_format", "continuation")

            # Parse steering_values JSON array (default: [0, 5, 10])
            steering_values = None
            if args.get("steering_values"):
                try:
                    steering_values = json.loads(args["steering_values"])
                except json.JSONDecodeError:
                    steering_values = [0, 5, 10]  # Default
            else:
                steering_values = [0, 5, 10]  # Default

            # Parse prompts JSON array if provided
            prompts = None
            if args.get("prompts"):
                try:
                    prompts = json.loads(args["prompts"])
                except json.JSONDecodeError:
                    pass

            # Parse downstream_neurons JSON array if provided
            # None = auto-populate from connectivity; [] = no downstream; [...] = explicit list
            downstream_neurons = None
            raw_ds_str = args.get("downstream_neurons", "")
            if raw_ds_str and raw_ds_str.strip().lower() not in ("", "auto", "null", "none"):
                try:
                    raw_ds = json.loads(raw_ds_str)
                    if isinstance(raw_ds, list) and raw_ds:
                        downstream_neurons = []
                        for item in raw_ds:
                            if isinstance(item, str) and "/" in item:
                                dl = int(item.split("/")[0].replace("L", ""))
                                dn = int(item.split("/")[1].replace("N", ""))
                                downstream_neurons.append({"id": item, "layer": dl, "neuron_idx": dn})
                            elif isinstance(item, dict):
                                downstream_neurons.append(item)
                except (json.JSONDecodeError, ValueError):
                    pass  # Leave as None → auto-populate

            scientist.experiments_run += 1
            print(f"  [Tool] batch_steer_and_generate (use_categorized={use_categorized}, threshold={activation_threshold}, steering_values={steering_values}, downstream={len(downstream_neurons) if downstream_neurons else 'auto'})")
            result = await tool_batch_steer_and_generate(
                layer, neuron_idx,
                prompts=prompts,
                use_categorized_prompts=use_categorized,
                activation_threshold=activation_threshold,
                steering_values=steering_values,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                max_prompts=max_prompts,
                generation_format=generation_format,
                downstream_neurons=downstream_neurons,
            )
            # Return summary with per-steering-value stats and downstream effects
            summary = {
                "total_prompts": result.get("total_prompts", 0),
                "total_changed": result.get("total_changed", 0),  # Changed by any steering value
                "change_rate": result.get("change_rate", 0),
                "steering_values": steering_values,
                "per_steering_value": result.get("per_steering_value", {}),  # Stats per steering value
                "category_stats": result.get("category_stats", {}),
            }
            if result.get("dependency_summary"):
                summary["dependency_summary"] = result["dependency_summary"]
            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        @tool("intelligent_steering_analysis", "REQUIRED (≥1 run): Sonnet-powered steering analysis. Generates prompts at decision boundaries, specifies steering values, runs experiments, and returns analysis with 10 illustrative examples. Can run multiple times with different focus via additional_instructions.", {"output_hypothesis": str, "promotes": str, "suppresses": str, "additional_instructions": str, "n_prompts": int, "max_new_tokens": int})
        async def intelligent_steering_analysis_tool(args):
            output_hypothesis = args.get("output_hypothesis", "")
            if not output_hypothesis:
                return {"content": [{"type": "text", "text": json.dumps({"error": "output_hypothesis is required"})}]}

            # Parse promotes/suppresses JSON arrays
            promotes = []
            suppresses = []
            if args.get("promotes"):
                try:
                    promotes = json.loads(args["promotes"])
                except json.JSONDecodeError:
                    promotes = [p.strip() for p in args["promotes"].split(",") if p.strip()]
            if args.get("suppresses"):
                try:
                    suppresses = json.loads(args["suppresses"])
                except json.JSONDecodeError:
                    suppresses = [s.strip() for s in args["suppresses"].split(",") if s.strip()]

            # Optional additional instructions for iterative refinement
            additional_instructions = args.get("additional_instructions", None)

            n_prompts = args.get("n_prompts", 100)
            max_new_tokens = args.get("max_new_tokens", 25)  # Default 25 tokens for fuller effect

            scientist.experiments_run += 1
            focus_note = f", focus: {additional_instructions[:30]}..." if additional_instructions else ""
            print(f"  [Tool] intelligent_steering_analysis (hypothesis={output_hypothesis[:50]}..., n_prompts={n_prompts}{focus_note})")
            result = await tool_intelligent_steering_analysis(
                layer, neuron_idx,
                output_hypothesis=output_hypothesis,
                promotes=promotes,
                suppresses=suppresses,
                additional_instructions=additional_instructions,
                n_prompts=n_prompts,
                max_new_tokens=max_new_tokens,
            )

            # Return the full analysis result
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("ablate_upstream_and_test", "V4 Input Phase: Test if target neuron depends on upstream neurons by ablating them.", {"upstream_neurons": str, "test_prompts": str, "window_tokens": int})
        async def ablate_upstream_and_test_tool(args):
            # Parse upstream_neurons JSON array
            try:
                upstream_neurons = json.loads(args["upstream_neurons"])
            except json.JSONDecodeError:
                return {"content": [{"type": "text", "text": json.dumps({"error": "Invalid upstream_neurons JSON"})}]}
            # Parse test_prompts JSON array
            try:
                test_prompts = json.loads(args["test_prompts"])
            except json.JSONDecodeError:
                test_prompts = [p.strip() for p in args["test_prompts"].split("\n") if p.strip()]

            window_tokens = args.get("window_tokens", 10)

            scientist.experiments_run += 1
            print(f"  [Tool] ablate_upstream_and_test: {len(upstream_neurons)} upstream neurons, {len(test_prompts)} prompts")
            result = await tool_ablate_upstream_and_test(
                layer, neuron_idx, upstream_neurons, test_prompts,
                window_tokens=window_tokens,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("ablate_and_check_downstream", "V4 Output Phase: Check downstream neuron activation after ablating target.", {"prompts": str, "downstream_neurons": str, "max_new_tokens": int})
        async def ablate_and_check_downstream_tool(args):
            # Parse prompts JSON array
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]
            # Parse optional downstream_neurons JSON array
            downstream_neurons = None
            if args.get("downstream_neurons"):
                try:
                    downstream_neurons = json.loads(args["downstream_neurons"])
                except json.JSONDecodeError:
                    pass
            max_new_tokens = args.get("max_new_tokens", 5)

            scientist.experiments_run += 1
            print(f"  [Tool] ablate_and_check_downstream: {len(prompts)} prompts")
            result = await tool_ablate_and_check_downstream(
                layer, neuron_idx, prompts,
                downstream_neurons=downstream_neurons,
                max_new_tokens=max_new_tokens,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_ablate_upstream_and_test", "V5 Input Phase: Test upstream dependencies using ALL activating prompts from category_selectivity. Efficient batch processing.", {"use_categorized_prompts": bool, "activation_threshold": float, "upstream_neurons": str, "max_prompts": int, "window_tokens": int})
        async def batch_ablate_upstream_and_test_tool(args):
            use_categorized = args.get("use_categorized_prompts", True)
            activation_threshold = args.get("activation_threshold", 0.5)
            max_prompts = args.get("max_prompts", 50)
            window_tokens = args.get("window_tokens", 10)

            # Parse optional upstream_neurons JSON array
            upstream_neurons = None
            if args.get("upstream_neurons"):
                try:
                    upstream_neurons = json.loads(args["upstream_neurons"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] batch_ablate_upstream_and_test (use_categorized={use_categorized}, threshold={activation_threshold})")
            result = await tool_batch_ablate_upstream_and_test(
                use_categorized_prompts=use_categorized,
                activation_threshold=activation_threshold,
                upstream_neurons=upstream_neurons,
                max_prompts=max_prompts,
                window_tokens=window_tokens,
            )
            # Return summary
            summary = {
                "total_prompts": result.get("total_prompts", 0),
                "upstream_neurons_tested": len(result.get("individual_ablation", {})),
                "individual_ablation": result.get("individual_ablation", {}),
                "combined_ablation": result.get("combined_ablation", {}),
                "overall_category_effects": result.get("overall_category_effects", {}),
            }
            if "error" in result:
                summary["error"] = result["error"]
            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        @tool("steer_upstream_and_test", "V5 Input Phase: Steer an upstream neuron and measure effect on target. Returns dose-response curve comparable to RelP.", {"upstream_neuron": str, "test_prompts": str, "steering_values": str})
        async def steer_upstream_and_test_tool(args):
            upstream_neuron = args.get("upstream_neuron", "")
            # Parse test_prompts JSON array
            try:
                test_prompts = json.loads(args["test_prompts"])
            except json.JSONDecodeError:
                test_prompts = [p.strip() for p in args["test_prompts"].split("\n") if p.strip()]
            # Parse optional steering_values
            steering_values = None
            if args.get("steering_values"):
                try:
                    steering_values = json.loads(args["steering_values"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] steer_upstream_and_test: {upstream_neuron}, {len(test_prompts)} prompts")
            result = await tool_steer_upstream_and_test(
                layer, neuron_idx, upstream_neuron, test_prompts,
                steering_values=steering_values,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_steer_upstream_and_test", "V5 Input Phase: Batch steer upstream neurons using activating prompts. Compares steering slopes to RelP weights.", {"use_categorized_prompts": bool, "activation_threshold": float, "upstream_neurons": str, "max_prompts": int, "steering_values": str})
        async def batch_steer_upstream_and_test_tool(args):
            use_categorized = args.get("use_categorized_prompts", True)
            activation_threshold = args.get("activation_threshold", 0.5)
            max_prompts = args.get("max_prompts", 100)

            # Parse optional upstream_neurons JSON array
            upstream_neurons = None
            if args.get("upstream_neurons"):
                try:
                    upstream_neurons = json.loads(args["upstream_neurons"])
                except json.JSONDecodeError:
                    pass

            # Parse optional steering_values
            steering_values = None
            if args.get("steering_values"):
                try:
                    steering_values = json.loads(args["steering_values"])
                except json.JSONDecodeError:
                    pass

            scientist.experiments_run += 1
            print(f"  [Tool] batch_steer_upstream_and_test (use_categorized={use_categorized}, threshold={activation_threshold})")
            result = await tool_batch_steer_upstream_and_test(
                use_categorized_prompts=use_categorized,
                activation_threshold=activation_threshold,
                upstream_neurons=upstream_neurons,
                max_prompts=max_prompts,
                steering_values=steering_values,
            )
            # Return full result including RelP comparison
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("complete_input_phase", "V5: Mark input phase as complete. Validates requirements and stores input characterization.", {"summary": str, "triggers": str, "confidence": float, "upstream_dependencies": str})
        async def complete_input_phase_tool(args):
            summary = args.get("summary", "")
            # Parse triggers JSON array
            try:
                triggers = json.loads(args.get("triggers", "[]"))
            except json.JSONDecodeError:
                triggers = [t.strip() for t in args.get("triggers", "").split(",") if t.strip()]
            confidence = args.get("confidence", 0.5)
            # Parse optional upstream_dependencies JSON
            upstream_deps = None
            if args.get("upstream_dependencies"):
                try:
                    upstream_deps = json.loads(args["upstream_dependencies"])
                except json.JSONDecodeError:
                    pass

            result = await tool_complete_input_phase(
                summary=summary,
                triggers=triggers,
                confidence=confidence,
                upstream_dependencies=upstream_deps,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("complete_output_phase", "V5: Mark output phase as complete. Validates requirements and stores output characterization.", {"summary": str, "promotes": str, "suppresses": str, "confidence": float})
        async def complete_output_phase_tool(args):
            summary = args.get("summary", "")
            # Parse promotes/suppresses JSON arrays
            try:
                promotes = json.loads(args.get("promotes", "[]"))
            except json.JSONDecodeError:
                promotes = [t.strip() for t in args.get("promotes", "").split(",") if t.strip()]
            try:
                suppresses = json.loads(args.get("suppresses", "[]"))
            except json.JSONDecodeError:
                suppresses = [t.strip() for t in args.get("suppresses", "").split(",") if t.strip()]
            confidence = args.get("confidence", 0.5)

            result = await tool_complete_output_phase(
                summary=summary,
                promotes=promotes,
                suppresses=suppresses,
                confidence=confidence,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("complete_anomaly_phase", 'V5: Mark anomaly investigation phase as complete. REQUIRED before saving report. anomalies_identified: JSON array of strings, e.g. \'["Wiring predicts X but ablation shows Y", "Unexpected activation on Z"]\'. anomalies_investigated: JSON array of objects, each with "anomaly" (str), "explanation" (str), "experiments_run" (list of tool names), "confidence" (float 0-1). Example: \'[{"anomaly":"Wiring-ablation mismatch","explanation":"Network compensates via redundant pathways","experiments_run":["batch_ablate_and_generate","batch_steer_and_generate"],"confidence":0.7}]\'. Must investigate at least 3 (or all if fewer identified).', {"anomalies_identified": str, "anomalies_investigated": str})
        async def complete_anomaly_phase_tool(args):
            # Parse anomalies_identified JSON array
            try:
                anomalies_identified = json.loads(args.get("anomalies_identified", "[]"))
            except json.JSONDecodeError:
                # Try splitting by newline or comma
                raw = args.get("anomalies_identified", "")
                anomalies_identified = [a.strip() for a in raw.replace("\n", ",").split(",") if a.strip()]

            # Parse anomalies_investigated JSON array of dicts
            raw_investigated = args.get("anomalies_investigated", "[]")
            anomalies_investigated = []
            try:
                parsed = json.loads(raw_investigated)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            anomalies_investigated.append(item)
                        elif isinstance(item, str):
                            # Agent passed a list of strings instead of dicts — auto-wrap
                            anomalies_investigated.append({
                                "anomaly": item,
                                "explanation": item,
                                "experiments_run": [],
                                "confidence": 0.5,
                            })
            except json.JSONDecodeError:
                # Last resort: treat as comma-separated anomaly descriptions
                items = [a.strip() for a in raw_investigated.replace("\n", ",").split(",") if a.strip()]
                for item in items:
                    anomalies_investigated.append({
                        "anomaly": item,
                        "explanation": item,
                        "experiments_run": [],
                        "confidence": 0.5,
                    })

            if anomalies_investigated:
                print(f"  [Tool] complete_anomaly_phase: {len(anomalies_identified)} identified, {len(anomalies_investigated)} investigated")

            result = await tool_complete_anomaly_phase(
                anomalies_identified=anomalies_identified,
                anomalies_investigated=anomalies_investigated,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_relp_verify_connections", "REQUIRED: Verify weight-predicted wiring connections against corpus RelP graphs. Checks which upstream/downstream neurons actually appear as edges in corpus. Run after analyze_wiring or analyze_output_wiring.", {"upstream_neurons": str, "downstream_neurons": str, "max_graphs": int})
        async def batch_relp_verify_connections_tool(args):
            upstream_str = args.get("upstream_neurons", "[]")
            downstream_str = args.get("downstream_neurons", "[]")
            max_graphs = args.get("max_graphs", 20)
            try:
                upstream_list = json.loads(upstream_str) if isinstance(upstream_str, str) else upstream_str
            except json.JSONDecodeError:
                upstream_list = []
            try:
                downstream_list = json.loads(downstream_str) if isinstance(downstream_str, str) else downstream_str
            except json.JSONDecodeError:
                downstream_list = []
            print(f"  [Tool] batch_relp_verify_connections: {len(upstream_list)} upstream, {len(downstream_list)} downstream (max_graphs={max_graphs})...")
            result = await tool_batch_relp_verify_connections(
                layer, neuron_idx, upstream_list, downstream_list, max_graphs
            )
            # Note: merge into connectivity_data happens inside the tool function
            summary = result.get("summary", "")
            print(f"  [RelP Verify] {summary}")
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("analyze_wiring", "REQUIRED FIRST STEP: Analyze weight-based upstream wiring to determine which neurons can EXCITE vs INHIBIT this neuron. Uses SwiGLU polarity analysis.", {"top_k": int})
        async def analyze_wiring_tool(args):
            top_k = args.get("top_k", 100)
            print(f"  [Tool] analyze_wiring for {neuron_id} (top_k={top_k})...")
            result = await tool_analyze_wiring(layer, neuron_idx, top_k)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("analyze_output_wiring", "Analyze weight-based downstream wiring to see what neurons this neuron ACTIVATES vs SUPPRESSES. Validated: steering causes +6000% activation in top targets.", {"top_k": int, "max_layer": int, "include_logits": bool})
        async def analyze_output_wiring_tool(args):
            top_k = args.get("top_k", 100)
            max_layer = args.get("max_layer", None)
            include_logits = args.get("include_logits", True)
            print(f"  [Tool] analyze_output_wiring for {neuron_id} (top_k={top_k})...")
            result = await tool_analyze_output_wiring(layer, neuron_idx, top_k, max_layer, include_logits)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        return [
            test_activation_tool,
            batch_activation_tool,
            test_homograph_tool,  # Polysemy/homograph discrimination testing
            run_ablation_tool,
            batch_ablation_tool,
            get_relp_connectivity_tool,  # Renamed: informational only, does NOT populate dashboard
            save_report_tool,
            save_structured_report_tool,
            run_relp_tool,
            verify_downstream_tool,
            steer_neuron_tool,
            patch_activation_tool,
            get_neuron_label_tool,
            batch_get_neuron_labels_tool,
            # Validation tools (REQUIRED for high confidence)
            run_baseline_comparison_tool,
            adaptive_relp_tool,
            steer_dose_response_tool,
            # Category selectivity testing (REQUIRED for selectivity claims)
            run_category_selectivity_test_tool,
            test_additional_prompts_tool,  # Follow-up probes for selectivity
            # Pre-registration tools (REQUIRED for scientific rigor)
            register_hypothesis_tool,
            update_hypothesis_status_tool,
            get_hypothesis_summary_tool,
            # Output projection analysis
            get_output_projections_tool,
            # V4 Tools (multi-token generation)
            ablate_and_generate_tool,
            steer_and_generate_tool,
            ablate_upstream_and_test_tool,
            ablate_and_check_downstream_tool,
            # V4/V5 Batch Tools (efficient batch ablation/steering)
            batch_ablate_and_generate_tool,
            batch_steer_and_generate_tool,
            intelligent_steering_analysis_tool,  # Sonnet-powered steering
            batch_ablate_upstream_and_test_tool,
            batch_steer_upstream_and_test_tool,
            # V5 Upstream Steering Tools
            steer_upstream_and_test_tool,
            # V5 Phase Completion Tools
            complete_input_phase_tool,
            complete_output_phase_tool,
            complete_anomaly_phase_tool,
            # Wiring analysis (REQUIRED EARLY STEP)
            analyze_wiring_tool,
            # Output wiring analysis (downstream connectivity)
            analyze_output_wiring_tool,
            # RelP corpus verification
            batch_relp_verify_connections_tool,
        ]

    async def investigate(self, max_experiments: int = 100, seed: int = 42) -> NeuronInvestigation:
        """Run the investigation using Claude Agent SDK.

        Args:
            max_experiments: Maximum experiments to run
            seed: Random seed for reproducibility (default 42)

        Returns:
            NeuronInvestigation with findings
        """
        print(f"Starting investigation of {self.neuron_id}")
        start_time = time.time()
        self.investigation.timestamp = datetime.now().isoformat()

        # Initialize reproducibility, pre-registration, and protocol state
        from .tools import (
            clear_activation_cache,
            clear_experiment_registry,
            clear_hypothesis_registry,
            clear_mean_activation_cache,
            clear_output_projections_cache,
            clear_relp_registry,
            get_model_config,
            init_activation_cache,
            init_hypothesis_registry,
            init_output_projections_cache,
            init_protocol_state,
            init_relp_registry,
            set_seed,
            tool_get_output_projections,
            update_protocol_state,
        )
        set_seed(seed)

        # Initialize protocol state for this investigation
        init_protocol_state()
        # Set target neuron info for batch tools
        update_protocol_state(target_layer=self.layer, target_neuron=self.neuron_idx)

        # Initialize registries from prior investigation if available, otherwise clear
        if self.prior_investigation is not None:
            # Restore protocol state from prior investigation (crash recovery)
            if self.prior_investigation.protocol_validation:
                pv = self.prior_investigation.protocol_validation
                update_protocol_state(
                    phase0_corpus_queried=pv.get("phase0_corpus_queried", False),
                    phase0_graph_count=pv.get("phase0_graph_count", 0),
                    baseline_comparison_done=pv.get("baseline_comparison_done", False),
                    baseline_zscore=pv.get("baseline_zscore"),
                    dose_response_done=pv.get("dose_response_done", False),
                    dose_response_monotonic=pv.get("dose_response_monotonic", False),
                    relp_runs=pv.get("relp_runs", 0),
                    relp_positive_control=pv.get("relp_positive_control", False),
                    relp_negative_control=pv.get("relp_negative_control", False),
                    hypotheses_registered=pv.get("hypotheses_registered", 0),
                    hypotheses_updated=pv.get("hypotheses_updated", 0),
                    # V4 phase tracking
                    input_phase_complete=pv.get("input_phase_complete", False),
                    output_phase_complete=pv.get("output_phase_complete", False),
                    multi_token_ablation_done=pv.get("multi_token_ablation_done", False),
                    upstream_dependency_tested=pv.get("upstream_dependency_tested", False),
                    downstream_dependency_tested=pv.get("downstream_dependency_tested", False),
                    downstream_neurons_exist=pv.get("downstream_neurons_exist", False),
                    category_selectivity_done=pv.get("category_selectivity_done", False),
                    category_selectivity_zscore_gap=pv.get("category_selectivity_zscore_gap"),
                )
                print("  [PROTOCOL] Restored protocol state from prior investigation")

            # Restore characterization summaries from prior investigation
            # BUG FIX: Without this, revision iterations lose input/output characterization
            # because init_protocol_state() resets them to empty defaults
            from .tools import get_protocol_state
            state = get_protocol_state()
            if self.prior_investigation.input_characterization:
                ic = self.prior_investigation.input_characterization
                if ic.get("summary"):
                    state.input_characterization = ic
                    print(f"  [PROTOCOL] Restored input characterization: {ic['summary'][:60]}...")
            if self.prior_investigation.output_characterization:
                oc = self.prior_investigation.output_characterization
                if oc.get("summary"):
                    state.output_characterization = oc
                    print(f"  [PROTOCOL] Restored output characterization: {oc['summary'][:60]}...")

            # Restore categorized_prompts from prior investigation to protocol state
            # BUG FIX: Without this, revision iterations lose categorized_prompts
            # and batch ablation/steering tools have no data to work with
            if self.prior_investigation.categorized_prompts:
                for cat_name, prompts in self.prior_investigation.categorized_prompts.items():
                    state.categorized_prompts[cat_name] = prompts
                print(f"  [PROTOCOL] Restored {len(self.prior_investigation.categorized_prompts)} categorized prompt categories from prior investigation")

            # Load hypotheses from prior iteration, preserving IDs and status
            if self.prior_investigation.hypotheses_tested:
                init_hypothesis_registry(self.prior_investigation.hypotheses_tested)
                print(f"  Loaded {len(self.prior_investigation.hypotheses_tested)} hypotheses from prior iteration")
            else:
                clear_hypothesis_registry()

            # Clear experiment registry for temporal enforcement
            # (experiments from prior iterations don't affect temporal validation)
            clear_experiment_registry()

            # Initialize RelP registry from prior results - enables lookups without re-running
            if self.prior_investigation.relp_results:
                init_relp_registry(self.prior_investigation.relp_results)
            else:
                clear_relp_registry()

            # Initialize activation cache from prior results - enables value lookups
            init_activation_cache(
                self.prior_investigation.activating_prompts,
                self.prior_investigation.non_activating_prompts,
            )

            # Initialize output projections cache from prior investigation
            if self.prior_investigation.output_projections:
                init_output_projections_cache(self.prior_investigation.output_projections)
            else:
                clear_output_projections_cache()
        else:
            clear_hypothesis_registry()
            clear_experiment_registry()
            clear_relp_registry()
            clear_activation_cache()
            clear_output_projections_cache()
            clear_mean_activation_cache()

        print(f"  Seed set to {seed} for reproducibility")

        # Pre-compute output projections if not available from prior knowledge
        # This ensures we always have actual projection weights for the dashboard
        prior = self._load_prior_knowledge()
        if "output_projections" not in prior or not prior["output_projections"].get("promote"):
            print("  Auto-computing output projections from model weights...")
            try:
                proj_result = await tool_get_output_projections(self.layer, self.neuron_idx, top_k=15)
                prior["output_projections"] = {
                    "promote": [
                        {"token": t["token"], "weight": t["projection_strength"]}
                        for t in proj_result.get("promoted", [])
                    ],
                    "suppress": [
                        {"token": t["token"], "weight": t["projection_strength"]}
                        for t in proj_result.get("suppressed", [])
                    ],
                }
                print(f"    Got {len(prior['output_projections']['promote'])} promoted, {len(prior['output_projections']['suppress'])} suppressed tokens")
            except Exception as e:
                print(f"    Warning: Could not compute output projections: {e}")

        # Store for use by _build_initial_prompt
        self._prior_knowledge_cache = prior

        # Phase 0 now only includes:
        # - Initial labels (from neuron database/autointerp) - already in self.label
        # - Output projections - already computed above in prior["output_projections"]
        # The agent can query corpus stats and RelP data during investigation if needed.
        self._corpus_context = None  # No longer pre-loaded

        # Create MCP tools and server
        tools = self._create_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="neuron_tools",
            version="1.0.0",
            tools=tools,
        )

        # Build initial prompt for the agent
        initial_prompt = self._build_initial_prompt()

        # Track agent conversation (also stored on self for access after investigation)
        agent_messages = []
        self.agent_messages = agent_messages  # Reference for external access

        # Select system prompt based on version, with model-aware substitutions
        model_config = get_model_config()
        if self.prompt_version >= 2:
            system_prompt = get_model_aware_system_prompt(
                model_config,
                version=self.prompt_version,
                neuron_id=self.neuron_id,  # Pass neuron ID for V5 prompts
                polarity_mode=self.polarity_mode,
            )
        else:
            system_prompt = SYSTEM_PROMPT

        # Configure options - tool names must be mcp__<server>__<tool>
        # Store transcripts in separate directory to avoid cluttering main project
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "neuron_reports" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=75,
            model=self.model,  # Use specified model (opus, sonnet, haiku)
            mcp_servers={"neuron_tools": mcp_server},
            cwd=transcripts_dir,
            add_dirs=[project_root],  # Allow access to main project files
            allowed_tools=[
                "mcp__neuron_tools__test_activation",
                "mcp__neuron_tools__batch_activation_test",
                "mcp__neuron_tools__test_homograph",
                "mcp__neuron_tools__run_ablation",
                "mcp__neuron_tools__batch_ablation",
                "mcp__neuron_tools__get_relp_connectivity",
                "mcp__neuron_tools__save_report",
                "mcp__neuron_tools__save_structured_report",
                "mcp__neuron_tools__run_relp",
                "mcp__neuron_tools__verify_downstream_connections",
                "mcp__neuron_tools__steer_neuron",
                "mcp__neuron_tools__patch_activation",
                "mcp__neuron_tools__get_neuron_label",
                "mcp__neuron_tools__batch_get_neuron_labels",
                # Validation tools (REQUIRED for high confidence)
                "mcp__neuron_tools__run_baseline_comparison",
                "mcp__neuron_tools__adaptive_relp",
                "mcp__neuron_tools__steer_dose_response",
                "mcp__neuron_tools__run_category_selectivity_test",
                # Pre-registration tools (REQUIRED for scientific rigor)
                "mcp__neuron_tools__register_hypothesis",
                "mcp__neuron_tools__update_hypothesis_status",
                "mcp__neuron_tools__get_hypothesis_summary",
                # Output projection analysis
                "mcp__neuron_tools__get_output_projections",
                # Graph index tools (V2)
                "mcp__neuron_tools__find_graphs_for_neuron",
                "mcp__neuron_tools__get_neuron_graph_stats",
                "mcp__neuron_tools__load_graph_from_index",
                # Data collection tools (for visualizations)
                "mcp__neuron_tools__log_categorized_activation",
                "mcp__neuron_tools__log_homograph_test",
                # V4 multi-token generation tools
                "mcp__neuron_tools__ablate_and_generate",
                "mcp__neuron_tools__steer_and_generate",
                "mcp__neuron_tools__ablate_upstream_and_test",
                "mcp__neuron_tools__steer_upstream_and_test",
                "mcp__neuron_tools__ablate_and_check_downstream",
                # V4/V5 batch tools (efficient batch ablation/steering)
                "mcp__neuron_tools__batch_ablate_and_generate",
                "mcp__neuron_tools__batch_steer_and_generate",
                "mcp__neuron_tools__intelligent_steering_analysis",  # Sonnet-powered steering
                "mcp__neuron_tools__batch_ablate_upstream_and_test",
                "mcp__neuron_tools__batch_steer_upstream_and_test",
                # V5 Phase completion tools
                "mcp__neuron_tools__complete_input_phase",
                "mcp__neuron_tools__complete_output_phase",
                "mcp__neuron_tools__complete_anomaly_phase",
                # Wiring analysis (REQUIRED EARLY STEP)
                "mcp__neuron_tools__analyze_wiring",
                # Output wiring analysis (downstream connectivity)
                "mcp__neuron_tools__analyze_output_wiring",
                # RelP corpus verification
                "mcp__neuron_tools__batch_relp_verify_connections",
            ],
        )

        # Run agent with ClaudeSDKClient
        print(f"Running agent investigation (model: {self.model})...")

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                _last_msg_time = time.time()
                _msg_count = 0
                async for message in client.receive_response():
                    _now = time.time()
                    _gap = _now - _last_msg_time
                    _last_msg_time = _now
                    _msg_count += 1

                    # Process messages
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                content = block.text
                                agent_messages.append({"role": "assistant", "content": content})
                                # Print truncated reasoning
                                preview = content[:200].replace("\n", " ")
                                print(f"Agent: {preview}...")

                            elif isinstance(block, ToolUseBlock):
                                print(f"Tool: {block.name}", flush=True)
                                if _gap > 30:
                                    print(f"  [TIMING] {_gap:.0f}s since last message (msg #{_msg_count})", flush=True)

                                if self.experiments_run >= max_experiments:
                                    print(f"Reached experiment limit ({max_experiments})")
                                    break

                    elif isinstance(message, ResultMessage):
                        print(f"Result: {message.subtype} (gap={_gap:.0f}s, msg #{_msg_count})", flush=True)
                        if message.subtype == "error":
                            print("  Error occurred during investigation")

                print(f"[TIMING] Agent loop ended after {_msg_count} messages, {time.time() - start_time:.0f}s total", flush=True)

        except Exception as e:
            print(f"Agent error: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Finalize investigation
        duration = time.time() - start_time
        self.investigation.total_experiments = self.experiments_run

        # Store agent reasoning for reference (but structured data comes from save_structured_report)
        self._extract_findings(agent_messages)

        # Note: The agent should have called save_structured_report which creates the dashboard.
        # If not, we save a minimal backup investigation file.
        print(f"Investigation complete. {self.experiments_run} experiments in {duration:.1f}s")

        # IMPORTANT: Check if the agent saved a structured report - if so, load it back
        # This ensures the characterization fields (input_function, output_function, etc.)
        # are populated in the returned investigation object
        safe_id = self.neuron_id.replace("/", "_")
        polarity_suffix = "_negative" if self.polarity_mode == "negative" else ""
        saved_investigation_path = self.output_dir / f"{safe_id}{polarity_suffix}_investigation.json"
        if saved_investigation_path.exists():
            try:
                with open(saved_investigation_path) as f:
                    saved_data = json.load(f)
                # Check if the saved data has characterization (from save_structured_report)
                char = saved_data.get("characterization", {})
                if char.get("input_function") or char.get("output_function"):
                    print(f"  ✓ Loading structured findings from {saved_investigation_path}")
                    loaded = NeuronInvestigation.from_dict(saved_data)
                    # Preserve any accumulated evidence from self.investigation
                    # that might not have been saved yet
                    if not loaded.activating_prompts and self.investigation.activating_prompts:
                        loaded.activating_prompts = self.investigation.activating_prompts
                    if not loaded.non_activating_prompts and self.investigation.non_activating_prompts:
                        loaded.non_activating_prompts = self.investigation.non_activating_prompts
                    if not loaded.ablation_effects and self.investigation.ablation_effects:
                        loaded.ablation_effects = self.investigation.ablation_effects
                    if not loaded.connectivity and self.investigation.connectivity:
                        loaded.connectivity = self.investigation.connectivity
                    # V4: Preserve homograph tests, patching experiments, and other V4 fields
                    if not loaded.homograph_tests and self.investigation.homograph_tests:
                        loaded.homograph_tests = self.investigation.homograph_tests
                    if not loaded.patching_experiments and self.investigation.patching_experiments:
                        loaded.patching_experiments = self.investigation.patching_experiments
                    if not loaded.multi_token_ablation_results and self.investigation.multi_token_ablation_results:
                        loaded.multi_token_ablation_results = self.investigation.multi_token_ablation_results
                    if not loaded.multi_token_steering_results and self.investigation.multi_token_steering_results:
                        loaded.multi_token_steering_results = self.investigation.multi_token_steering_results
                    if not loaded.upstream_dependency_results and self.investigation.upstream_dependency_results:
                        loaded.upstream_dependency_results = self.investigation.upstream_dependency_results
                    if not loaded.downstream_dependency_results and self.investigation.downstream_dependency_results:
                        loaded.downstream_dependency_results = self.investigation.downstream_dependency_results
                    if not loaded.open_questions and self.investigation.open_questions:
                        loaded.open_questions = self.investigation.open_questions
                    if not loaded.categorized_prompts and self.investigation.categorized_prompts:
                        loaded.categorized_prompts = self.investigation.categorized_prompts
                    # Use the loaded investigation with proper characterization
                    self.investigation = loaded
            except Exception as e:
                print(f"  Warning: Could not load saved investigation: {e}")

        # =================================================================
        # SELF-SUMMARIZATION: Generate transcript summary for dashboard
        # =================================================================
        if agent_messages:
            print("[Scientist] Generating transcript summary...")
            try:
                summary = await summarize_scientist_transcript(
                    neuron_id=self.neuron_id,
                    agent_messages=agent_messages,
                    iteration=1,
                    model="claude-sonnet-4-20250514",  # Use Sonnet for summarization (faster/cheaper)
                )
                self.investigation.transcript_summaries.append(summary)
                print(f"  ✓ Transcript summary: {summary.get('summary', '')[:100]}...")

                # Re-save the investigation with transcript summary
                if saved_investigation_path.exists():
                    with open(saved_investigation_path) as f:
                        saved_data = json.load(f)
                    saved_data["transcript_summaries"] = [s if isinstance(s, dict) else s.to_dict() if hasattr(s, 'to_dict') else s
                                                          for s in self.investigation.transcript_summaries]
                    with open(saved_investigation_path, "w") as f:
                        json.dump(saved_data, f, indent=2)
                    print(f"  ✓ Updated {saved_investigation_path} with transcript summary")
            except Exception as e:
                print(f"  Warning: Could not generate transcript summary: {e}")

        return self.investigation

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agent."""
        # Get model config for accurate layer/neuron counts
        from .tools import get_model_config
        model_config = get_model_config()
        prompt_parts = [
            f"# Investigation: Neuron {self.neuron_id}",
            f"Layer: {self.layer} (of {model_config.num_layers}), Neuron Index: {self.neuron_idx} (of {model_config.neurons_per_layer:,})",
            "",
        ]

        # Phase 0 now only includes initial labels and output projections (in prior knowledge below)
        # The agent can query corpus stats and RelP data during investigation if needed.

        # Add prior knowledge from edge stats if available
        # Use cached version if available (includes auto-computed output projections)
        prior = getattr(self, '_prior_knowledge_cache', None) or self._load_prior_knowledge()
        if prior:
            # Add LLM-generated labels as CLAIMS TO VERIFY
            llm = prior.get("llm_labels", {})
            if llm.get("output_label") or llm.get("input_label"):
                prompt_parts.append("## CLAIMS TO VERIFY (from prior LLM analysis)")
                prompt_parts.append("")
                prompt_parts.append("**Your task is to TEST these claims experimentally. Confirm or refute each one.**")
                prompt_parts.append("")

                if llm.get("input_label"):
                    prompt_parts.append("### INPUT FUNCTION CLAIM")
                    prompt_parts.append(f"- **Label**: {llm['input_label']}")
                    prompt_parts.append(f"- **Type**: {llm.get('input_type', 'unknown')}")
                    prompt_parts.append(f"- **Claimed Interpretability**: {llm.get('input_interpretability', 'unknown')}")
                    if llm.get("input_description"):
                        prompt_parts.append(f"- **Description**: {llm['input_description']}")
                    prompt_parts.append("")

                if llm.get("output_label"):
                    prompt_parts.append("### OUTPUT FUNCTION CLAIM")
                    prompt_parts.append(f"- **Label**: {llm['output_label']}")
                    prompt_parts.append(f"- **Type**: {llm.get('output_type', 'unknown')}")
                    prompt_parts.append(f"- **Claimed Interpretability**: {llm.get('output_interpretability', 'unknown')}")
                    if llm.get("output_description"):
                        prompt_parts.append(f"- **Description**: {llm['output_description']}")
                    prompt_parts.append("")
            else:
                # No LLM labels - try to get NeuronDB description as starting point
                from neuron_scientist.tools import get_neuron_label_with_fallback
                neurondb_info = get_neuron_label_with_fallback(self.neuron_id)
                if neurondb_info.get("found") and neurondb_info.get("source") == "neurondb":
                    prompt_parts.append("## PRIOR DESCRIPTION (from NeuronDB - based on max-activating examples)")
                    prompt_parts.append("")
                    prompt_parts.append("**No detailed input/output function claims are available for this neuron.**")
                    prompt_parts.append("However, NeuronDB provides this description based on max-activating examples:")
                    prompt_parts.append("")
                    prompt_parts.append(f"> {neurondb_info.get('neurondb_description', neurondb_info.get('label', ''))}")
                    prompt_parts.append("")
                    prompt_parts.append("**Your task**: Expand this into proper input and output function claims.")
                    prompt_parts.append("- **Input function**: What patterns in the input cause this neuron to activate?")
                    prompt_parts.append("- **Output function**: What does this neuron's activation promote in the model's output?")
                    prompt_parts.append("")
                    prompt_parts.append("Use activation testing, ablation, and RelP to characterize both functions.")
                    prompt_parts.append("")

            # Add downstream targets with their claimed functions
            downstream = prior.get("downstream_with_labels", [])
            if downstream:
                prompt_parts.append("### DOWNSTREAM CONNECTION CLAIMS")
                prompt_parts.append("These neurons receive signal FROM this neuron. Verify with RelP!")
                prompt_parts.append("")
                for d in downstream[:8]:
                    label = d.get("function_label") or "unlabeled"
                    prompt_parts.append(f"- **{d['neuron_id']}** (weight={d.get('weight', 0):.4f}): {label}")
                prompt_parts.append("")

            # Add upstream sources with their claimed functions
            upstream = prior.get("upstream_with_labels", [])
            if upstream:
                prompt_parts.append("### UPSTREAM CONNECTION CLAIMS")
                prompt_parts.append("These neurons send signal TO this neuron. They may activate it!")
                prompt_parts.append("")
                for u in upstream[:8]:
                    label = u.get("function_label") or "unlabeled"
                    prompt_parts.append(f"- **{u['neuron_id']}** (weight={u.get('weight', 0):.4f}): {label}")
                prompt_parts.append("")

            # Add co-occurring neurons
            co_occurring = prior.get("co_occurring_neurons", [])
            if co_occurring:
                prompt_parts.append("### CO-OCCURRING NEURONS")
                prompt_parts.append("These neurons tend to fire together with this neuron:")
                prompt_parts.append("")
                for c in co_occurring[:5]:
                    prompt_parts.append(f"- **{c['neuron_id']}** (co-occurrence count={c.get('count', 0)})")
                prompt_parts.append("")

            # Add basic stats
            prompt_parts.append("## Statistics")
            prompt_parts.append("")
            if prior.get("appearance_count"):
                prompt_parts.append(f"- **Appearance Count**: {prior['appearance_count']} prompts")
            if prior.get("domain_specificity"):
                ds = prior["domain_specificity"]
                prompt_parts.append(f"- **Domain Specificity**: {ds:.0%}")
            if prior.get("transluce_positive"):
                prompt_parts.append(f"- **Transluce Label**: {prior['transluce_positive']}")
            prompt_parts.append("")

            # Show NeuronDB polarity labels (IMPORTANT: different activations can have different meanings!)
            if prior.get("neurondb_polarity"):
                polarity = prior["neurondb_polarity"]
                prompt_parts.append("## NeuronDB Polarity Labels")
                prompt_parts.append("")
                if self.polarity_mode == "negative":
                    # In negative mode, emphasize the negative label as the primary hypothesis seed
                    if polarity.get("negative"):
                        prompt_parts.append(f"- **NEGATIVE activation** (THIS INVESTIGATION): {polarity['negative']}")
                    if polarity.get("positive"):
                        prompt_parts.append(f"- **POSITIVE activation** (investigated separately): {polarity['positive']}")
                    prompt_parts.append("")
                    prompt_parts.append("Focus on the NEGATIVE activation function for this investigation.")
                else:
                    prompt_parts.append("**IMPORTANT**: Positive and negative activations can have COMPLETELY DIFFERENT meanings, not just opposites!")
                    prompt_parts.append("")
                    if polarity.get("positive"):
                        prompt_parts.append(f"- **POSITIVE activation** (THIS INVESTIGATION): {polarity['positive']}")
                    if polarity.get("negative"):
                        prompt_parts.append(f"- **NEGATIVE activation** (investigated separately): {polarity['negative']}")
                    prompt_parts.append("")
                prompt_parts.append("")

        if self.initial_label:
            prompt_parts.append(f"**Initial Label**: {self.initial_label}")
            prompt_parts.append("")

        if self.initial_hypothesis:
            prompt_parts.append(f"**Starting Hypothesis**: {self.initial_hypothesis}")
            prompt_parts.append("")

        # Add revision context from GPT review if this is a revision run
        if self.revision_context:
            prompt_parts.append(self.revision_context)
            prompt_parts.append("")

            # Also inject structured prior evidence summary if we have prior data
            if self.prior_investigation is not None:
                from .review_prompts import (
                    PRIOR_EVIDENCE_TEMPLATE,
                    summarize_prior_evidence,
                    summarize_skeptic_findings,
                )
                prior_summary = summarize_prior_evidence(self.prior_investigation.to_dict())
                prompt_parts.append(PRIOR_EVIDENCE_TEMPLATE.format(**prior_summary))
                prompt_parts.append("")

                # Inject skeptic findings so revision scientist knows WHY hypotheses were weakened
                skeptic_findings = summarize_skeptic_findings(self.prior_investigation.to_dict())
                if skeptic_findings:
                    prompt_parts.append(skeptic_findings)
                    prompt_parts.append("")

        prompt_parts.extend([
            "## Suggested Test Prompts",
            "",
        ])

        for p in self.test_prompts[:10]:
            prompt_parts.append(f'- "{p}"')

        # Add prior labels and stats for the agent to include in save_structured_report
        if prior:
            prompt_parts.append("## Data to Include in Your Final Report")
            prompt_parts.append("")
            prompt_parts.append("When you call save_structured_report, include these values from prior analysis:")
            prompt_parts.append("")
            llm = prior.get("llm_labels", {})
            if llm:
                prompt_parts.append(f'- original_output_label: "{llm.get("output_label", "")}"')
                prompt_parts.append(f'- original_input_label: "{llm.get("input_label", "")}"')
                if llm.get("output_description"):
                    prompt_parts.append(f'- original_output_description: "{llm.get("output_description", "")}"')
                if llm.get("input_description"):
                    prompt_parts.append(f'- original_input_description: "{llm.get("input_description", "")}"')
            # Add output projections if available
            if prior.get("output_projections"):
                proj = prior["output_projections"]
                if proj.get("promote"):
                    tokens = json.dumps(proj["promote"][:15])
                    prompt_parts.append(f'- output_projections_promote: {tokens}')
                if proj.get("suppress"):
                    tokens = json.dumps(proj["suppress"][:15])
                    prompt_parts.append(f'- output_projections_suppress: {tokens}')
            prompt_parts.append("")

            # Also show output projections in the main prompt if available
            if prior.get("output_projections"):
                proj = prior["output_projections"]
                prompt_parts.append("### OUTPUT PROJECTIONS (from model weights)")
                prompt_parts.append("")
                prompt_parts.append("These are the tokens this neuron promotes/suppresses based on its output projection (down_proj @ lm_head):")
                prompt_parts.append("")
                if proj.get("promote"):
                    # Show token and weight (projection strength)
                    promote_tokens = [
                        f"{t['token']} ({t.get('weight', 0):+.4f})" if isinstance(t, dict) else t
                        for t in proj['promote'][:10]
                    ]
                    prompt_parts.append(f"**Promotes**: {', '.join(promote_tokens)}")
                if proj.get("suppress"):
                    suppress_tokens = [
                        f"{t['token']} ({t.get('weight', 0):+.4f})" if isinstance(t, dict) else t
                        for t in proj['suppress'][:10]
                    ]
                    prompt_parts.append(f"**Suppresses**: {', '.join(suppress_tokens)}")
                prompt_parts.append("")

            # Show input projections (what tokens activate this neuron via up_proj)
            if prior.get("input_projections"):
                inp = prior["input_projections"]
                prompt_parts.append("### INPUT PROJECTIONS (from model weights)")
                prompt_parts.append("")
                prompt_parts.append("These tokens have high projection weights to this neuron (via up_proj). Note: actual activation depends on context, not just token presence.")
                prompt_parts.append("")
                input_tokens = [
                    f"{t['token']} ({t.get('weight', 0):+.4f})" if isinstance(t, dict) else t
                    for t in inp[:10]
                ]
                prompt_parts.append(f"**High-weight input tokens**: {', '.join(input_tokens)}")
                prompt_parts.append("")

        prompt_parts.extend([
            "## COMPLETE INVESTIGATION FRAMEWORK",
            "",
            "**A neuron's full function involves FOUR factors. You must investigate ALL of them:**",
            "",
            "### 1. INPUT: What activates this neuron?",
            "- **Text patterns**: What prompts/contexts cause high activation? Test with activation experiments.",
            "- **Upstream neurons**: Which earlier neurons send signal to this one? Check upstream connections.",
            "- **Input projections**: What tokens have high weights in up_proj? (listed above if available)",
            "",
            "### 2. OUTPUT: What does this neuron do when active?",
            "- **Direct token effects**: What tokens does it promote/suppress? Check output projections and run ablations.",
            "- **Downstream neurons**: Which later neurons does it activate? Verify with RelP.",
            "",
            "### 3. CIRCUIT POSITION",
            "- Where does this neuron fit in the processing pipeline?",
            "- Use RelP to trace causal pathways through this neuron.",
            "- Check the SIGN of edges: positive = promotes, negative = suppresses.",
            "",
            "### 4. VERIFICATION",
            "- Run baseline comparison (z-score > 2 required for meaningful effects)",
            "- Test negative controls (prompts that should NOT activate)",
            "- Verify downstream connections with verify_downstream_connections tool",
            "",
            "**Report**: For each claim, state CONFIRMED, REFUTED, or PARTIALLY CONFIRMED with evidence.",
            "",
            "When calling save_structured_report, include the original labels shown above.",
        ])

        return "\n".join(prompt_parts)

    def _load_prior_knowledge(self) -> dict[str, Any]:
        """Load prior knowledge about the neuron from edge stats."""
        result = {}

        # Load from edge stats
        try:
            edge_stats_path = Path(self.edge_stats_path) if self.edge_stats_path else None
            if edge_stats_path and edge_stats_path.exists():
                with open(self.edge_stats_path) as f:
                    edge_stats = json.load(f)

                # Handle both list and dict structures
                profiles = edge_stats.get("profiles", []) if isinstance(edge_stats, dict) else edge_stats

                for p in profiles:
                    if p.get("neuron_id") == self.neuron_id:
                        result.update({
                            "max_act_label": p.get("max_act_label", ""),
                            "transluce_positive": p.get("transluce_label_positive", ""),
                            "transluce_negative": p.get("transluce_label_negative", ""),
                            "appearance_count": p.get("appearance_count", 0),
                            "domain_specificity": p.get("domain_specificity", 1.0),
                        })

                        # Extract output projections (weight-based from neuron output weights)
                        # ALWAYS prefer output_projection.promoted/suppressed (actual weights)
                        # over output_token_associations (corpus frequency)
                        output_proj = p.get("output_projection", {})
                        ota = output_proj.get("promoted", [])
                        neg_ota = output_proj.get("suppressed", [])

                        # Fallback to frequency-based if weight-based not available
                        if not ota:
                            ota = p.get("output_token_associations", [])
                        if not neg_ota:
                            neg_ota = p.get("negative_output_projections", p.get("output_token_suppressions", []))

                        if ota or neg_ota:
                            # Include projection_strength/weight values for display
                            # Priority: projection_strength > weight > magnitude > 0
                            def extract_weight(item):
                                if isinstance(item, dict):
                                    return (
                                        item.get("projection_strength") or
                                        item.get("weight") or
                                        item.get("magnitude") or
                                        0
                                    )
                                return 0

                            result["output_projections"] = {
                                "promote": [
                                    {"token": item["token"] if isinstance(item, dict) else item,
                                     "weight": extract_weight(item)}
                                    for item in ota[:15]
                                ],
                                "suppress": [
                                    {"token": item["token"] if isinstance(item, dict) else item,
                                     "weight": extract_weight(item)}
                                    for item in neg_ota[:15]
                                ],
                            }

                        # Extract input projections (what tokens activate this neuron via up_proj)
                        input_proj = p.get("input_projection", {})
                        input_activates = input_proj.get("activates", [])
                        if input_activates:
                            result["input_projections"] = [
                                {"token": item["token"] if isinstance(item, dict) else item,
                                 "weight": extract_weight(item)}
                                for item in input_activates[:15]
                            ]

                        # Extract co-occurring neurons
                        co_occurring = p.get("co_occurring_neurons", [])
                        if co_occurring:
                            result["co_occurring_neurons"] = [
                                {
                                    "neuron_id": f"L{item['neuron'].split('_')[0]}/N{item['neuron'].split('_')[1]}" if '_' in str(item.get('neuron', '')) else item.get('neuron', ''),
                                    "count": item.get("count", 0),
                                    "jaccard": item.get("jaccard", 0),
                                }
                                for item in co_occurring[:10]
                            ]

                        # Extract upstream sources from edge stats
                        upstream_sources = p.get("top_upstream_sources", [])
                        if upstream_sources:
                            result["upstream_from_edge_stats"] = [
                                {
                                    "neuron_id": f"L{item['source'].split('_')[0]}/N{item['source'].split('_')[1]}" if '_' in str(item.get('source', '')) else item.get('source', ''),
                                    "weight": item.get("avg_weight", 0),
                                    "frequency": item.get("frequency", 0),
                                }
                                for item in upstream_sources[:10]
                            ]

                        break
        except Exception as e:
            print(f"Error loading edge stats: {e}")

        # Load rich LLM-generated labels from labels file
        try:
            labels_path = Path(self.labels_path) if self.labels_path else None
            if labels_path and labels_path.exists():
                with open(labels_path) as f:
                    labels_data = json.load(f)

                neurons = labels_data.get("neurons", {})
                if self.neuron_id in neurons:
                    n = neurons[self.neuron_id]
                    result["llm_labels"] = {
                        "output_label": n.get("function_label", ""),
                        "output_description": n.get("function_description", ""),
                        "output_type": n.get("function_type", ""),
                        "output_interpretability": n.get("interpretability", ""),
                        "input_label": n.get("input_label", ""),
                        "input_description": n.get("input_description", ""),
                        "input_type": n.get("input_type", ""),
                        "input_interpretability": n.get("input_interpretability", ""),
                    }
                    # Add NeuronDB polarity labels (positive = what activates, negative = what suppresses)
                    # IMPORTANT: These can have completely different meanings, not just opposites!
                    neurondb_pos = n.get("neurondb_label_positive", "")
                    neurondb_neg = n.get("neurondb_label_negative", "")
                    if neurondb_pos or neurondb_neg:
                        result["neurondb_polarity"] = {
                            "positive": neurondb_pos,  # What contexts cause POSITIVE activation
                            "negative": neurondb_neg,  # What contexts cause NEGATIVE activation
                        }
                    # Get downstream targets with their labels
                    downstream = n.get("downstream_neurons", [])[:10]
                    result["downstream_with_labels"] = [
                        {
                            "neuron_id": d.get("neuron_id"),
                            "weight": d.get("weight"),
                            "frequency": d.get("frequency"),
                            "function_label": d.get("function_label", ""),
                        }
                        for d in downstream
                    ]
                    # Get upstream sources with their labels
                    upstream = n.get("upstream_neurons", [])[:10]
                    result["upstream_with_labels"] = [
                        {
                            "neuron_id": u.get("neuron_id"),
                            "weight": u.get("weight"),
                            "frequency": u.get("frequency"),
                            "function_label": u.get("function_label", ""),
                        }
                        for u in upstream
                    ]
                    # Get output projections from direct logit effects (only if not already set from edge stats)
                    if "output_projections" not in result:
                        dle = n.get("direct_logit_effects", {})
                        if dle and (dle.get("promotes") or dle.get("suppresses")):
                            result["output_projections"] = {
                                "promote": dle.get("promotes", [])[:15],
                                "suppress": dle.get("suppresses", [])[:15],
                            }
        except Exception as e:
            print(f"Error loading LLM labels: {e}")

        return result

    def _load_downstream_targets(self) -> list[dict[str, Any]]:
        """Load expected downstream targets from edge stats."""
        try:
            with open(self.edge_stats_path) as f:
                edge_stats = json.load(f)

            # Handle both list and dict structures
            profiles = edge_stats.get("profiles", []) if isinstance(edge_stats, dict) else edge_stats

            for p in profiles:
                if p.get("neuron_id") == self.neuron_id:
                    downstream = []
                    for d in p.get("top_downstream_targets", [])[:10]:
                        target = d.get("target", "")
                        # Parse target format: "layer_neuron_position"
                        parts = target.split("_")
                        if len(parts) >= 2 and not target.startswith("L_"):
                            # MLP neuron target
                            downstream.append({
                                "neuron_id": f"L{parts[0]}/N{parts[1]}",
                                "frequency": d.get("frequency", 0),
                                "weight": d.get("avg_weight", 0),
                            })
                    return downstream
        except Exception as e:
            print(f"Error loading downstream targets: {e}")
        return []

    def _extract_findings(self, messages: list[dict[str, Any]]):
        """Extract key findings from agent messages (basic heuristic extraction)."""
        # Combine all agent reasoning
        full_reasoning = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        )

        # Store reasoning
        self.investigation.agent_reasoning = full_reasoning

        # Try to extract structured findings (basic heuristics)
        lines = full_reasoning.lower()

        # Extract hypothesis if mentioned
        if "hypothesis" in lines and ":" in full_reasoning:
            for line in full_reasoning.split("\n"):
                if "hypothesis" in line.lower() and ":" in line:
                    self.investigation.hypotheses_tested.append({
                        "hypothesis": line.split(":", 1)[-1].strip(),
                    })

        # Calculate confidence based on evidence
        n_activating = len(self.investigation.activating_prompts)
        n_ablations = len(self.investigation.ablation_effects)

        if n_activating > 10 and n_ablations > 3:
            self.investigation.confidence = 0.8
        elif n_activating > 5:
            self.investigation.confidence = 0.6
        else:
            self.investigation.confidence = 0.4

    async def _extract_structured_findings(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Use LLM to extract structured findings from agent reasoning."""

        # Combine all agent reasoning
        full_reasoning = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        )

        # Truncate if too long (keep last 15000 chars which contain conclusions)
        if len(full_reasoning) > 20000:
            full_reasoning = "...[earlier reasoning truncated]...\n\n" + full_reasoning[-15000:]

        extraction_prompt = f"""Analyze this neuron investigation and extract structured findings.

## Investigation Reasoning:
{full_reasoning}

## Task:
Extract the following information from the investigation. Be concise but accurate.

Return a JSON object with these fields:
{{
  "input_function": "Brief description of what activates this neuron (1-2 sentences)",
  "output_function": "Brief description of what this neuron promotes/suppresses in outputs (1-2 sentences)",
  "function_type": "Category: semantic, syntactic, routing, formatting, or hybrid",
  "final_hypothesis": "The agent's final conclusion about what this neuron does (1-2 sentences)",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "open_questions": ["Question 1", "Question 2", ...],
  "confidence_assessment": "low/medium/high and brief justification",
  "summary": "One sentence summary of the neuron's function"
}}

If any field cannot be determined from the investigation, use an empty string or empty list.
Return ONLY the JSON object, no other text."""

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": extraction_prompt}],
            )

            # Parse JSON from response
            response_text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            return json.loads(response_text)
        except Exception as e:
            print(f"Error extracting structured findings: {e}")
            return {}

    async def _save_investigation(self, duration: float = 0.0):
        """Save the investigation report (consolidated - no separate dashboard.json)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_id = self.neuron_id.replace("/", "_")

        # Copy output_projections from prior knowledge cache to investigation
        prior = getattr(self, '_prior_knowledge_cache', {})
        if prior.get("output_projections"):
            self.investigation.output_projections = prior["output_projections"]

        # Copy LLM labels as prior_claims (seed hypotheses for Investigation Flow)
        llm_labels = prior.get("llm_labels", {})
        if llm_labels:
            self.investigation.prior_claims = {
                "input_label": llm_labels.get("input_label", ""),
                "input_description": llm_labels.get("input_description", ""),
                "output_label": llm_labels.get("output_label", llm_labels.get("function_label", "")),
                "output_description": llm_labels.get("output_description", llm_labels.get("function_description", "")),
                "source": "llm_labels",
            }

        # Save full investigation (single consolidated file)
        polarity_suffix = "_negative" if self.polarity_mode == "negative" else ""
        report_path = self.output_dir / f"{safe_id}{polarity_suffix}_investigation.json"
        with open(report_path, "w") as f:
            json.dump(self.investigation.to_dict(), f, indent=2)

        print(f"Investigation saved to {report_path}")


async def investigate_neuron(
    neuron_id: str,
    initial_label: str = "",
    initial_hypothesis: str = "",
    edge_stats_path: Path | None = None,
    labels_path: Path | None = None,
    output_dir: Path = Path("neuron_reports/json"),
    test_prompts: list[str] | None = None,
    max_experiments: int = 100,
    model: str = "opus",
    prompt_version: int = 5,
    revision_context: str = "",
    prior_investigation: NeuronInvestigation | None = None,
    return_transcript: bool = False,
    polarity_mode: str = "positive",
    gpu_server_url: str | None = None,
) -> NeuronInvestigation:
    """Convenience function to investigate a single neuron.

    Args:
        neuron_id: Target neuron (e.g., "L15/N7890")
        initial_label: Initial label from batch labeling
        initial_hypothesis: Starting hypothesis
        edge_stats_path: Path to edge statistics
        labels_path: Path to neuron labels JSON
        output_dir: Directory for reports
        test_prompts: Prompts to test
        max_experiments: Maximum experiments
        model: Model to use ("opus", "sonnet", or "haiku")
        prompt_version: System prompt version (1=original, 2=V2, 3=V3, 4=V4, 5=V5 simplified)
        revision_context: Feedback from GPT review to address
        prior_investigation: Prior NeuronInvestigation to build upon (for additive iterations)
        return_transcript: If True, return (investigation, agent_messages) tuple
        polarity_mode: "positive" or "negative" - which firing direction to investigate
        gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")

    Returns:
        NeuronInvestigation with findings (or tuple with transcript if return_transcript=True)
    """
    # Parse layer/neuron for GPU client agent ID
    parts = neuron_id.split("/")
    layer = int(parts[0][1:])
    neuron_idx = int(parts[1][1:])

    # Set up GPU client if server URL provided
    gpu_client = None
    if gpu_server_url:
        from neuron_scientist.gpu_client import GPUClient
        from neuron_scientist.tools import set_gpu_client
        gpu_client = GPUClient(gpu_server_url, agent_id=f"scientist-L{layer}-N{neuron_idx}")
        await gpu_client.wait_for_server()
        set_gpu_client(gpu_client)

    try:
        scientist = NeuronScientist(
            neuron_id=neuron_id,
            initial_label=initial_label,
            initial_hypothesis=initial_hypothesis,
            edge_stats_path=edge_stats_path,
            labels_path=labels_path,
            output_dir=output_dir,
            test_prompts=test_prompts,
            model=model,
            prompt_version=prompt_version,
            revision_context=revision_context,
            prior_investigation=prior_investigation,
            polarity_mode=polarity_mode,
            gpu_server_url=gpu_server_url,
        )

        investigation = await scientist.investigate(max_experiments=max_experiments)

        if return_transcript:
            return investigation, scientist.agent_messages

        return investigation
    finally:
        # Clean up GPU client
        if gpu_client:
            from neuron_scientist.tools import set_gpu_client
            set_gpu_client(None)
            await gpu_client.close()
