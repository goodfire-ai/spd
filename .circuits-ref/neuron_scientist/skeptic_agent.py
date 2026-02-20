"""NeuronSkeptic - Adversarial agent for testing neuron hypotheses.

The Skeptic's job is to try to DISPROVE the Scientist's hypothesis by:
1. Testing alternative explanations
2. Finding boundary cases where the hypothesis fails
3. Detecting confounding factors
4. Measuring true selectivity

This produces adversarial evidence that goes to the GPT reviewer alongside
the Scientist's findings, leading to more robust conclusions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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

from .schemas import (
    AlternativeHypothesis,
    BoundaryTest,
    Confound,
    NeuronInvestigation,
    SkepticReport,
)
from .tools import (
    get_model_config,
)

# =============================================================================
# System Prompt for NeuronSkeptic
# =============================================================================

SKEPTIC_SYSTEM_PROMPT = """You are NeuronSkeptic, an adversarial agent tasked with stress-testing neuron hypotheses.

## Your Mission

You receive a neuron investigation from NeuronScientist containing a hypothesis about what the neuron does. Your job is to try to DISPROVE this hypothesis through rigorous adversarial testing. You are not trying to confirm - you are trying to find weaknesses.

## CRITICAL: Understanding Neuron Function

Every neuron has two complementary aspects:
- **Input function**: What causes the neuron to fire (patterns, contexts, features)
- **Output function**: What the neuron does when it fires (promotes/suppresses tokens)

Both are always true simultaneously. The question is which framing is more informative.

### Input-Salient vs Output-Salient Neurons

**Input-salient neurons**: The most informative description is what triggers them.
- Activation tests show a crisp, coherent pattern (specific domain, syntax, or concept)
- Output projections are diffuse (no single token dominates)
- Example: "fires on medical terminology"
- Testing: Focus on varied inputs that should/shouldn't trigger the neuron

**Output-salient neurons**: The most informative description is what they do.
- Output projections show one or few tokens with very high weights
- Fires across diverse, seemingly unrelated contexts
- Example: "promotes the token 'Paris' in completion contexts"
- Testing: Focus on contexts where the promoted token is/isn't a plausible continuation

### The Completion Context Question

For neurons with strong output projections (output-salient):
- The neuron fires when its promoted token is a **likely next token**
- It fires BEFORE the target token appears, in contexts where it would be predicted
- Test contexts where the token IS vs IS NOT a plausible continuation

**Example**: If a neuron promotes "bus":
- SHOULD activate: "I commute by ___" (bus is plausible)
- SHOULD NOT activate: "Bus fare has increased" (bus already said, not being predicted)
- The distinction: Does it fire AT the token (input-salient) or BEFORE it (output-salient)?

### Two Pathways of Influence

Neurons affect outputs through two pathways:

| Pathway | Mechanism | How to Test |
|---------|-----------|-------------|
| **Direct** | activation × output weights → logits | `get_output_projections`, `steer_neuron` |
| **Indirect** | activates downstream neurons | `analyze_connectivity`, check circuit |

**Note:** The balance between direct and indirect effects varies by context (prompt, token position).
Focus on empirical observations from ablation and downstream dependency tests.

**Testing strategy based on layer**:
- Late layers (last ~10): Focus on direct logit effects with `steer_neuron`
- Mid layers: Check connectivity first, test effect on downstream neurons
- Early layers: Often positional/syntactic, check for position confounds

## Adversarial Testing Strategy

### 0. CRITICAL: Check for Hypothesis-Evidence Mismatch

**⚠️ THIS IS YOUR MOST IMPORTANT CHECK:**

Before doing anything else, verify that the scientist's hypothesis MATCHES the evidence:

1. Call `get_output_projections` - Does the neuron PROMOTE a specific token strongly?
2. If yes, and the hypothesis only describes input patterns:
   - **RED FLAG** - The scientist may have missed the output-salient nature
   - The neuron's key function may be PROMOTING that token, not just detecting inputs
   - Test COMPLETION contexts where the promoted token is a likely continuation

**Example of hypothesis-evidence mismatch:**
- Output projections: "bus" +1.08 (PROMOTED)
- Scientist hypothesis: "Activates on transportation enumerations"
- **PROBLEM**: If it promotes "bus", why focus only on prompts where "bus" is present?
- **CHALLENGE**: Test "I commute by ___" - does it activate? If yes → output-salient

### 1. First: Identify the Salience Type
Before any other testing:
- Read the hypothesis: Does it emphasize input patterns or output effects?
- **Call `get_output_projections` IMMEDIATELY** - are there dominant promoted tokens?
- Check: Is the characterization input-salient or output-salient?
- Check the layer: Late layers often output-salient, early layers often input-salient
- Look at the scientist's evidence: What kinds of prompts activated it?
- Call `analyze_connectivity` to see if it's mid-circuit or late-circuit

### 2. Alternative Hypotheses (test the RIGHT thing)
Generate alternatives that match the salience type:

For OUTPUT-SALIENT neurons, test if the promotion is specific:
- "Does it promote 'bus' specifically, or any transportation word?"
- "Does it promote 'bus' only in appropriate contexts, or in ANY completion frame?"
- "Is it a general 'common noun promoter' that happens to favor high-frequency nouns?"

For INPUT-SALIENT neurons, test semantic boundaries:
- "Is it really 'medical terms' or just 'Latin-derived words'?"
- "Is it 'wine' or 'fermented beverages' or just 'alcoholic drinks'?"

For INDIRECT effects (routing neurons), test the circuit:
- "Does it activate downstream neurons that then promote the target?"
- "Is its role to activate a broader concept circuit?"

### 3. Boundary Testing (appropriate to salience type)

For OUTPUT-SALIENT neurons:
- **True positive**: Contexts where promoted token IS a likely completion → should activate
- **True negative**: Contexts where promoted token is NOT likely → should NOT activate
- **False positive**: Activates in contexts where the promoted token is implausible
- **False negative**: Doesn't activate when the promoted token is the obvious completion

For INPUT-SALIENT neurons:
- **True positive**: Target concept/pattern present → should activate
- **True negative**: Target concept/pattern absent → should NOT activate
- **False positive**: Activates on unrelated concepts
- **False negative**: Doesn't activate when concept is clearly present

### 4. Verify the Claimed Effect

For neurons claiming to PROMOTE a token (output-salient):
- Use `get_output_projections` to verify the token is actually in top promoted tokens
- Use `steer_neuron` to verify that boosting the neuron increases the token's probability
- Test: Does steering it UP increase P(target)? Does steering it DOWN decrease P(target)?

For neurons claiming DOWNSTREAM effects (routing neurons):
- Use `analyze_connectivity` to identify downstream neurons
- Check if those downstream neurons have the claimed output effect

### 5. Confound Detection
Test for spurious correlations:
- **Syntactic confound**: Is it just a syntactic pattern (e.g., "verb + the + ___") not semantic?
- **Position effects**: Does it only fire at certain positions?
- **Length effects**: Correlation with prompt length?
- **Co-occurrence**: Always appears with some other feature?
- **Frequency effects**: Only fires on common/rare words?
- **Tokenization confound**: Does activation depend on how the word gets tokenized?

## Available Tools

You have the SAME experimental tools as NeuronScientist:
- `test_activation`: Test a single prompt (check WHERE it activates with token positions)
- `batch_activation_test`: Test multiple prompts at once
- `run_ablation`: Zero out neuron and measure effects
- `steer_neuron`: Add value to neuron and measure logit shifts (CRUCIAL for output-salient)
- `patch_activation`: Counterfactual test - patch activation from source prompt into target prompt
- `get_output_projections`: See what tokens neuron promotes/suppresses (CHECK THIS FIRST)
- `get_neuron_label`: Look up labels for neurons
- `analyze_connectivity`: Check upstream/downstream connections (CRUCIAL for circuit position)

**V4 Multi-token generation tools (for testing output effects across multiple tokens):**
- `ablate_and_generate`: Ablate neuron and generate multiple tokens - shows cumulative effects
- `steer_and_generate`: Steer neuron and generate multiple tokens - tests causal effects on generation

Plus skeptic-specific tools:
- `get_investigation_summary`: Get the scientist's findings including ALL hypotheses with their IDs
- `challenge_hypothesis`: **KEY TOOL** - Challenge a specific hypothesis by ID with evidence
- `record_alternative_hypothesis`: Record results of testing an alternative
- `record_boundary_test`: Record a boundary case test result
- `record_confound`: Record a detected confound
- `submit_skeptic_report`: Submit your final adversarial report

## IMPORTANT: Target Specific Hypotheses

When challenging the scientist's conclusions, you MUST target specific hypotheses by ID:

1. Call `get_investigation_summary` to get all hypotheses with their IDs (H1, H2, etc.)
2. For each hypothesis you want to challenge:
   - Design tests that could refute it based on the `refutation_criteria`
   - Run the tests using experimental tools
   - Call `challenge_hypothesis` with:
     - `hypothesis_id`: e.g., "H1"
     - `challenge_description`: What aspect you tested
     - `evidence`: Results of your tests
     - `challenge_result`: "upheld", "weakened", or "refuted"
     - `confidence_delta`: How much to adjust posterior in PERCENTAGE POINTS (-30 for strong refutation, -15 for weakened, 0 for upheld)

Example:
```
challenge_hypothesis(
    hypothesis_id="H1",
    challenge_description="Tested whether neuron fires on homograph contexts",
    evidence="Activation was 0.03 on 'bank' (financial) vs 0.02 on 'bank' (river) - no discrimination",
    challenge_result="weakened",
    confidence_delta=-15
)
```

The individual hypothesis updates flow through to the final report.

**NOTE:** Overall investigation confidence has been removed. Confidence is now tracked per-hypothesis:
- Each hypothesis has a `prior` (initial belief) and `posterior` (updated after evidence)
- Focus on challenging/supporting individual hypotheses, not overall confidence

## Workflow

1. Call `get_investigation_summary` to see all hypotheses with their IDs and confidence
2. **Identify the salience type**: input-salient vs output-salient, direct vs indirect
3. **Check output projections** with `get_output_projections` - does it promote specific tokens?
4. **Check circuit position** with `analyze_connectivity`
5. **Verify the claimed effect** with `steer_neuron` if output-salient
6. Design and run adversarial tests APPROPRIATE to the salience type
7. Record your findings with the record_* tools
8. Submit final report with `submit_skeptic_report`

## Output Format

After testing, call `submit_skeptic_report` with:
- Alternative hypotheses tested and results
- Boundary test results (false positives/negatives found)
- Confounds detected
- Selectivity metrics
- Overall verdict: SUPPORTED, WEAKENED, or REFUTED
- Revised hypothesis if you have a better one

## Guidelines

- **FIRST check output projections** - are there dominant promoted tokens?
- **Identify salience type** - input-salient vs output-salient, direct vs indirect
- **Test the RIGHT thing** - don't test input presence for an output-salient neuron
- Be adversarial but fair - don't cherry-pick failures
- **Use your judgment** on what experiments to run - you have full discretion
- Choose experiments that will most effectively challenge the hypothesis
- Verify claimed effects with steering/projection tools when appropriate
- Look for confounds if you suspect them
- Quantify your findings (rates, scores)
- If you can't break the hypothesis, say so honestly

Remember: Your goal is to find REAL weaknesses, not to misunderstand the hypothesis and test the wrong thing. An output-salient neuron that "fails" input-detection tests hasn't actually been tested—you need to test whether it activates in completion contexts where its promoted token is likely.

## Known Pitfalls & Failure Modes

When evaluating the scientist's evidence, watch for these common failure modes. **Name the specific pitfall** in your `challenge_hypothesis` call when you detect one.

### Statistical Pitfalls
- **Multiple testing / look-elsewhere effect**: Testing many prompts and cherry-picking the few that confirm
- **Base rate neglect**: High activation is meaningless if the neuron activates on 30% of ALL prompts (check control rate)
- **Cherry-picked examples**: Scientist shows top 5 activating prompts but 15 others didn't activate — check the denominator
- **Small sample sizes**: 3-4 examples are unreliable; demand ≥8 for any confident claim
- **Post-hoc hypothesis registration**: Check temporal_enforcement flags — was the hypothesis registered BEFORE or AFTER seeing the data?

### Logical Pitfalls
- **Confirmation bias**: Scientist only tested prompts designed to CONFIRM, never to REFUTE
- **Correlation ≠ causation**: High activation on X doesn't mean "detects X" — may detect a co-occurring feature
- **Unfalsifiable hypotheses**: "Responds to complex semantic content" or "general language processing" cannot be tested
- **Affirming the consequent**: "If X then neuron fires" doesn't mean "if neuron fires then X"

### Confirmation Bias Detection Protocol

Actively check for these confirmation bias patterns in the scientist's evidence:

1. **Outlier-driven hypothesis**: Compare the hypothesis against category selectivity data.
   - What are the top 5 z-score CATEGORIES (not individual prompts)?
   - Does the hypothesis explain these top categories, or just one outlier prompt?
   - If the hypothesis focuses on a word/phrase from 1-2 prompts but top categories suggest
     a broader pattern, challenge as "PITFALL: Outlier-driven hypothesis"

2. **Self-confirming probe generation**: Check categorized_prompts for agent-generated categories.
   - Categories NOT in the selectivity run (e.g., "oxygen_start", "oxygen_mid", "oxygen_parallel")
     are agent-generated probes. If >50% test the same narrow hypothesis, the agent hasn't
     explored alternatives.
   - Run your own `run_category_selectivity_test` with DIFFERENT categories to test whether
     a broader pattern explains the data better.

3. **Activating prompts vs selectivity mismatch**: Compare evidence activating_prompts against
   the selectivity run's top_activating list.
   - If activating_prompts are dominated by one theme but selectivity top_activating shows
     diverse themes, the activating_prompts are contaminated by biased agent probes.
   - The selectivity data is the UNBIASED source — use it as ground truth when challenging.

### Interpretability-Specific Pitfalls (from the literature)
- **Polysemanticity**: A single neuron may respond to MULTIPLE unrelated features; the scientist may have found one but missed others
- **Superposition**: The feature may be encoded across multiple neurons; this neuron captures only part of it
- **Frequency confounds**: Apparent selectivity actually correlates with token frequency in the training corpus
- **Tokenization artifacts (BPE)**: Activation reflects BPE subword boundaries, not semantics (e.g., "aspirin" → ["asp","irin"] — neuron detects "asp" subword, not the concept)
- **Positional biases**: Neuron responds to token POSITION in the sequence, not content
- **Attention head confounds**: Mid-layer apparent sensitivity may reflect attention routing patterns, not neuron selectivity

When you detect a pitfall, name it specifically in your `challenge_hypothesis` call.
Example: `challenge_description: "PITFALL: Frequency confound — activation correlates with token log-frequency (r=0.82)"`
"""


# =============================================================================
# NeuronSkeptic Agent Class
# =============================================================================

class NeuronSkeptic:
    """Adversarial agent for testing neuron hypotheses.

    Uses the same experimental tools as NeuronScientist but with an
    adversarial mindset focused on disproving hypotheses.
    """

    def __init__(
        self,
        neuron_id: str,
        investigation: NeuronInvestigation,
        edge_stats_path: Path | None = None,
        labels_path: Path | None = None,
        model: str = "sonnet",  # Sonnet is good for adversarial thinking
        gpu_server_url: str | None = None,  # URL for GPU inference server
    ):
        """Initialize NeuronSkeptic.

        Args:
            neuron_id: Target neuron (e.g., "L15/N7890")
            investigation: The scientist's investigation to attack
            edge_stats_path: Path to edge statistics (for connectivity)
            labels_path: Path to neuron labels
            model: Claude model to use for the agent
            gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")
        """
        self.neuron_id = neuron_id
        self.investigation = investigation
        self.edge_stats_path = edge_stats_path
        self.labels_path = labels_path or Path("data/neuron_labels_combined.json")
        self.model = model
        self.gpu_server_url = gpu_server_url

        # Parse layer/neuron from ID
        parts = neuron_id.replace("L", "").replace("N", "").split("/")
        self.layer = int(parts[0])
        self.neuron_idx = int(parts[1])

        # Results accumulator
        self.alternative_hypotheses: list[AlternativeHypothesis] = []
        self.boundary_tests: list[BoundaryTest] = []
        self.confounds: list[Confound] = []
        self.hypothesis_challenges: list[dict[str, Any]] = []  # Track challenges to individual hypotheses
        self.experiments_run = 0
        self.report: SkepticReport | None = None

    def _get_layer_description(self) -> str:
        """Get a description of the layer position relative to the model depth."""
        config = get_model_config()
        num_layers = config.num_layers
        # Calculate relative position (0-100%)
        position = self.layer / num_layers
        if position > 0.75:
            return "late (likely direct logit effect)"
        elif position > 0.40:
            return "mid-range (may work through downstream neurons)"
        else:
            return "early (likely syntactic/positional, works through circuits)"

    def _create_mcp_tools(self):
        """Create MCP tools for the skeptic agent.

        Includes both standard experimental tools (shared with Scientist)
        and skeptic-specific tools for recording adversarial findings.
        """
        # Import tool implementations from tools module
        from .tools import (
            _sync_batch_activation_test,
            _sync_get_output_projections,
            _sync_get_relp_connectivity,
            _sync_run_ablation,
            _sync_steer_neuron,
            _sync_test_activation,
            batch_get_neuron_labels_with_fallback,
            get_neuron_label_with_fallback,
            # Protocol state management
            tool_ablate_and_generate,
            tool_ablate_upstream_and_test,
            # V5 batch tools
            tool_batch_ablate_and_generate,
            tool_batch_ablate_upstream_and_test,
            tool_batch_steer_and_generate,
            tool_batch_steer_upstream_and_test,
            # Intelligent steering analysis (Sonnet-powered)
            tool_intelligent_steering_analysis,
            tool_patch_activation,
            tool_run_category_selectivity_test,
            tool_steer_and_generate,
            tool_steer_dose_response,
            tool_steer_upstream_and_test,
            tool_test_additional_prompts,
        )

        layer = self.layer
        neuron_idx = self.neuron_idx
        investigation = self.investigation
        skeptic = self
        config = get_model_config()

        # =====================================================================
        # Skeptic-specific tools
        # =====================================================================

        @tool("get_investigation_summary", "Get the scientist's investigation findings to attack.", {})
        async def get_investigation_summary_tool(args):
            """Return comprehensive summary of scientist's findings.

            Includes core hypothesis, evidence samples, RelP results,
            steering results, connectivity, and output projections.
            """
            # Build RelP results summary
            relp_summary = [
                {
                    "prompt": r.get("prompt", "")[:200],
                    "neuron_found": r.get("neuron_found", False),
                    "relp_score": r.get("relp_score") or r.get("neuron_relp_score"),
                    "source": r.get("source", "agent"),
                }
                for r in investigation.relp_results[:10]
            ]

            # Build steering results summary
            steering_summary = [
                {
                    "prompt": s.get("prompt", "")[:150],
                    "steering_value": s.get("steering_value"),
                    "max_shift": s.get("max_shift"),
                    "promotes": s.get("promotes", s.get("promoted_tokens", []))[:5],
                    "suppresses": s.get("suppresses", s.get("suppressed_tokens", []))[:5],
                }
                for s in investigation.steering_results[:5]
            ]

            # Build hypotheses summary - include full details for targeted challenging
            hypotheses_summary = [
                {
                    "id": h.get("hypothesis_id", ""),
                    "hypothesis": h.get("hypothesis", ""),
                    "hypothesis_type": h.get("hypothesis_type", "unknown"),
                    "status": h.get("status", "unknown"),
                    "prior": h.get("prior_probability", 50),
                    "posterior": h.get("posterior_probability", h.get("prior_probability", 50)),
                    "confirmation_criteria": h.get("confirmation_criteria", ""),
                    "refutation_criteria": h.get("refutation_criteria", ""),
                    "evidence_for": h.get("evidence_for", []),
                    "evidence_against": h.get("evidence_against", []),
                }
                for h in investigation.hypotheses_tested
            ]

            # Extract category selectivity data if available
            category_selectivity = {}
            sel_data = getattr(investigation, 'category_selectivity_data', None)
            if isinstance(sel_data, dict) and sel_data:
                # Aggregate metrics
                category_selectivity = {
                    "z_score_gap": sel_data.get("z_score_gap"),
                    "n_categories": sel_data.get("n_categories"),
                    "selective": (sel_data.get("z_score_gap") or 0) > 1.0,
                    "selectivity_summary": sel_data.get("selectivity_summary", ""),
                }
                # Top categories by z-score (what the skeptic needs to compare against hypothesis)
                categories = sel_data.get("categories", {})
                if categories:
                    cat_stats = []
                    for cat_name, cat_info in categories.items():
                        if isinstance(cat_info, dict):
                            cat_stats.append({
                                "category": cat_name,
                                "type": cat_info.get("type", "unknown"),
                                "z_mean": cat_info.get("z_mean", 0),
                                "count": cat_info.get("count", 0),
                                "mean_activation": cat_info.get("mean", 0),
                            })
                    # Sort by z_mean descending — skeptic sees what categories activate most
                    cat_stats.sort(key=lambda x: x["z_mean"], reverse=True)
                    category_selectivity["top_categories"] = cat_stats[:10]
                    category_selectivity["bottom_categories"] = cat_stats[-5:]
                # Top activating examples from selectivity (unbiased)
                top_act = sel_data.get("top_activating", [])
                if top_act:
                    category_selectivity["top_activating_prompts"] = [
                        {
                            "prompt": p.get("prompt", "")[:200],
                            "activation": p.get("activation", 0),
                            "z_score": p.get("z_score", 0),
                            "category": p.get("category", ""),
                            "token": p.get("token", ""),
                        }
                        for p in top_act[:10]
                    ]

            # Extract V4 multi-token results if available
            multi_token_ablation = getattr(investigation, 'multi_token_ablation_results', [])
            multi_token_steering = getattr(investigation, 'multi_token_steering_results', [])

            return {"content": [{"type": "text", "text": json.dumps({
                # Core hypothesis info
                "neuron_id": investigation.neuron_id,
                "hypothesis": investigation.final_hypothesis,
                "input_function": investigation.input_function,
                "output_function": investigation.output_function,
                "function_type": investigation.function_type,
                # NOTE: Overall confidence removed - see hypotheses_tested for per-hypothesis confidence
                "key_findings": investigation.key_findings,

                # Activation evidence (expanded from 5 to 10, longer prompts)
                "activating_examples": [
                    {
                        "prompt": p.get("prompt", "")[:200],
                        "activation": p.get("activation", 0),
                        "position": p.get("position"),
                        "token": p.get("token", ""),
                    }
                    for p in investigation.activating_prompts[:10]
                ],
                "non_activating_examples": [
                    {
                        "prompt": p.get("prompt", "")[:200],
                        "activation": p.get("activation", 0),
                    }
                    for p in investigation.non_activating_prompts[:10]
                ],

                # Category selectivity (V4)
                "category_selectivity": category_selectivity,

                # RelP attribution results
                "relp_results": relp_summary,
                "relp_stats": {
                    "total_runs": len(investigation.relp_results),
                    "neuron_found_count": sum(1 for r in investigation.relp_results if r.get("neuron_found")),
                },

                # Steering/causal results
                "steering_results": steering_summary,

                # V4 Multi-token ablation results
                "multi_token_ablation": [
                    {
                        "prompt": r.get("prompt", "")[:150],
                        "baseline_completion": r.get("baseline_completion", ""),
                        "ablated_completion": r.get("ablated_completion", ""),
                        "completion_changed": r.get("completion_changed", False),
                    }
                    for r in multi_token_ablation[:5]
                ],

                # V4 Multi-token steering results
                "multi_token_steering": [
                    {
                        "prompt": r.get("prompt", "")[:150],
                        "steering_value": r.get("steering_value"),
                        "baseline_completion": r.get("baseline_completion", ""),
                        "steered_completion": r.get("steered_completion", ""),
                        "completion_changed": r.get("completion_changed", False),
                    }
                    for r in multi_token_steering[:5]
                ],

                # Connectivity
                "connectivity": investigation.connectivity,

                # Output projections
                "output_projections": investigation.output_projections,

                # Hypotheses tested - includes per-hypothesis confidence (prior/posterior)
                "hypotheses_tested": hypotheses_summary,
            }, indent=2)}]}

        @tool("record_alternative_hypothesis", "Record results of testing an alternative hypothesis.", {
            "original": str, "alternative": str, "test_description": str,
            "verdict": str, "evidence": str
        })
        async def record_alternative_hypothesis_tool(args):
            """Record an alternative hypothesis test result."""
            alt_hyp = AlternativeHypothesis(
                original_hypothesis=args["original"],
                alternative=args["alternative"],
                test_description=args["test_description"],
                verdict=args["verdict"],  # "distinguished", "indistinguishable", "alternative_better"
                evidence=args["evidence"],
            )
            skeptic.alternative_hypotheses.append(alt_hyp)
            return {"content": [{"type": "text", "text": json.dumps({
                "recorded": True,
                "total_alternatives": len(skeptic.alternative_hypotheses),
            })}]}

        @tool("register_hypothesis", "Register a replacement hypothesis when you find a better explanation for the neuron.", {
            "hypothesis": str, "confirmation_criteria": str, "refutation_criteria": str,
            "prior_probability": int, "hypothesis_type": str, "replaces": str, "source_evidence": str
        })
        async def register_hypothesis_tool(args):
            """Register a new hypothesis from skeptic findings.

            Use this when you refute or significantly weaken a hypothesis and have
            a better explanation. The new hypothesis enters the hypothesis registry
            and will be visible to the revision scientist.
            """
            # Generate next H{n} ID
            existing_ids = [h.get("hypothesis_id", "") for h in investigation.hypotheses_tested]
            max_n = 0
            for hid in existing_ids:
                if hid.startswith("H") and hid[1:].isdigit():
                    max_n = max(max_n, int(hid[1:]))
            new_id = f"H{max_n + 1}"

            prior = args.get("prior_probability", 50)
            if isinstance(prior, (int, float)) and 0 < prior <= 1.0:
                prior = int(round(prior * 100))

            new_hypothesis = {
                "hypothesis_id": new_id,
                "hypothesis": args["hypothesis"],
                "hypothesis_type": args.get("hypothesis_type", "unknown"),
                "confirmation_criteria": args.get("confirmation_criteria", ""),
                "refutation_criteria": args.get("refutation_criteria", ""),
                "prior_probability": prior,
                "posterior_probability": prior,
                "status": "registered",
                "source": "skeptic",
                "replaces": args.get("replaces", ""),
                "source_evidence": args.get("source_evidence", ""),
                "evidence_for": [],
                "evidence_against": [],
            }
            investigation.hypotheses_tested.append(new_hypothesis)

            return {"content": [{"type": "text", "text": json.dumps({
                "registered": True,
                "hypothesis_id": new_id,
                "replaces": args.get("replaces", ""),
                "prior_probability": prior,
                "total_hypotheses": len(investigation.hypotheses_tested),
            })}]}

        @tool("record_boundary_test", "Record a boundary/edge case test result.", {
            "description": str, "prompt": str, "expected_behavior": str,
            "actual_activation": float, "passed": bool, "notes": str
        })
        async def record_boundary_test_tool(args):
            """Record a boundary test result."""
            boundary = BoundaryTest(
                description=args["description"],
                prompt=args["prompt"],
                expected_behavior=args["expected_behavior"],
                actual_activation=args["actual_activation"],
                passed=args["passed"],
                notes=args.get("notes", ""),
            )
            skeptic.boundary_tests.append(boundary)
            return {"content": [{"type": "text", "text": json.dumps({
                "recorded": True,
                "total_boundary_tests": len(skeptic.boundary_tests),
                "passed": args["passed"],
            })}]}

        @tool("record_confound", "Record a detected confounding factor.", {
            "factor": str, "description": str, "evidence": str, "severity": str
        })
        async def record_confound_tool(args):
            """Record a confound finding."""
            confound = Confound(
                factor=args["factor"],  # "position", "length", "co-occurrence", etc.
                description=args["description"],
                evidence=args["evidence"],
                severity=args["severity"],  # "critical", "moderate", "minor"
            )
            skeptic.confounds.append(confound)
            return {"content": [{"type": "text", "text": json.dumps({
                "recorded": True,
                "total_confounds": len(skeptic.confounds),
            })}]}

        @tool("challenge_hypothesis", "Challenge a specific hypothesis with evidence. Updates the hypothesis status and adds evidence_against.", {
            "hypothesis_id": str, "challenge_description": str, "evidence": str,
            "challenge_result": str, "confidence_delta": float
        })
        async def challenge_hypothesis_tool(args):
            """Challenge a specific hypothesis with adversarial evidence.

            Args:
                hypothesis_id: The ID of the hypothesis to challenge (e.g., "H1")
                challenge_description: What aspect of the hypothesis is being challenged
                evidence: The evidence from experiments that challenges the hypothesis
                challenge_result: "upheld" (hypothesis still stands), "weakened", or "refuted"
                confidence_delta: Change to posterior probability (-0.30 to +0.10 typically)
            """
            hypothesis_id = args["hypothesis_id"]
            challenge_desc = args["challenge_description"]
            evidence = args["evidence"]
            result = args["challenge_result"]  # "upheld", "weakened", "refuted"

            # Normalize confidence_delta to percentage points (0-100 scale)
            raw_delta = args.get("confidence_delta", 0)
            # Backward compat: if abs(delta) <= 1.0, treat as decimal fraction and convert
            if isinstance(raw_delta, (int, float)) and abs(raw_delta) <= 1.0 and raw_delta != 0:
                raw_delta = raw_delta * 100
            confidence_delta = max(-50, min(10, raw_delta))  # Clamp to [-50, +10] percentage points

            # Find the hypothesis in the investigation
            hypothesis_found = None
            for h in investigation.hypotheses_tested:
                if h.get("hypothesis_id") == hypothesis_id:
                    hypothesis_found = h
                    break

            if not hypothesis_found:
                return {"content": [{"type": "text", "text": json.dumps({
                    "error": f"Hypothesis {hypothesis_id} not found",
                    "available_hypotheses": [h.get("hypothesis_id") for h in investigation.hypotheses_tested],
                })}]}

            # Update the hypothesis with the challenge
            # Add evidence_against if not present
            if "evidence_against" not in hypothesis_found:
                hypothesis_found["evidence_against"] = []

            hypothesis_found["evidence_against"].append({
                "challenge": challenge_desc,
                "evidence": evidence,
                "result": result,
                "source": "skeptic",
            })

            # Update posterior probability (0-100 scale)
            old_posterior = hypothesis_found.get("posterior_probability", hypothesis_found.get("prior_probability", 50))
            if isinstance(old_posterior, (int, float)):
                # Handle legacy data stored as 0-1 decimal
                if 0 < old_posterior <= 1.0:
                    old_posterior = old_posterior * 100
            else:
                old_posterior = 50

            new_posterior = int(round(max(0, min(100, old_posterior + confidence_delta))))
            hypothesis_found["posterior_probability"] = new_posterior

            # Update status based on result and new posterior
            if result == "refuted" or new_posterior < 20:
                hypothesis_found["status"] = "refuted"
            elif result == "weakened" or new_posterior < 50:
                hypothesis_found["status"] = "weakened"
            else:
                hypothesis_found["status"] = hypothesis_found.get("status", "confirmed")  # Preserve if upheld

            # Track the challenge in skeptic's records
            skeptic.hypothesis_challenges.append({
                "hypothesis_id": hypothesis_id,
                "challenge": challenge_desc,
                "evidence": evidence,
                "result": result,
                "confidence_delta": confidence_delta,
                "old_posterior": old_posterior,
                "new_posterior": new_posterior,
            })

            return {"content": [{"type": "text", "text": json.dumps({
                "hypothesis_id": hypothesis_id,
                "challenge_recorded": True,
                "result": result,
                "confidence_delta": confidence_delta,
                "old_posterior": old_posterior,
                "new_posterior": new_posterior,
                "new_status": hypothesis_found["status"],
                "total_challenges": len(skeptic.hypothesis_challenges),
            })}]}

        @tool("submit_skeptic_report", "Submit final skeptic report with verdict.", {
            "verdict": str, "confidence_adjustment": float, "revised_hypothesis": str,
            "key_challenges": str, "recommendations": str, "reasoning": str
        })
        async def submit_skeptic_report_tool(args):
            """Submit the final adversarial report."""
            # Validate and normalize confidence_adjustment to percentage points (0-100 scale)
            raw_adjustment = args.get("confidence_adjustment", 0)
            if raw_adjustment is None:
                raw_adjustment = 0

            # Backward compat: if abs <= 1.0 and non-zero, treat as decimal and convert
            if isinstance(raw_adjustment, (int, float)) and 0 < abs(raw_adjustment) <= 1.0:
                raw_adjustment = raw_adjustment * 100

            # Clamp to valid range [-100, +100] percentage points
            confidence_adjustment = max(-100, min(100, raw_adjustment))

            # Calculate metrics
            boundary_passed = sum(1 for t in skeptic.boundary_tests if t.passed)
            boundary_total = len(skeptic.boundary_tests)

            false_positives = sum(1 for t in skeptic.boundary_tests
                                 if t.expected_behavior == "should_not_activate" and not t.passed)
            false_negatives = sum(1 for t in skeptic.boundary_tests
                                 if t.expected_behavior == "should_activate" and not t.passed)

            fp_tests = sum(1 for t in skeptic.boundary_tests
                           if t.expected_behavior == "should_not_activate")
            fn_tests = sum(1 for t in skeptic.boundary_tests
                           if t.expected_behavior == "should_activate")

            # Parse lists from JSON strings
            try:
                key_challenges = json.loads(args["key_challenges"]) if args.get("key_challenges") else []
            except:
                key_challenges = [args.get("key_challenges", "")]

            try:
                recommendations = json.loads(args["recommendations"]) if args.get("recommendations") else []
            except:
                recommendations = [args.get("recommendations", "")]

            # Compute overall confidence_adjustment from hypothesis challenges (for backward compatibility)
            # The individual hypothesis_challenges are the source of truth
            computed_adjustment = sum(c.get("confidence_delta", 0) for c in skeptic.hypothesis_challenges)
            # Use provided adjustment as fallback if no individual challenges were made
            final_adjustment = computed_adjustment if skeptic.hypothesis_challenges else confidence_adjustment

            skeptic.report = SkepticReport(
                neuron_id=skeptic.neuron_id,
                original_hypothesis=investigation.final_hypothesis,
                alternative_hypotheses=skeptic.alternative_hypotheses,
                boundary_tests=skeptic.boundary_tests,
                confounds=skeptic.confounds,
                hypothesis_challenges=skeptic.hypothesis_challenges,  # Individual hypothesis updates
                selectivity_score=boundary_passed / boundary_total if boundary_total > 0 else 0,
                false_positive_rate=false_positives / fp_tests if fp_tests > 0 else 0,
                false_negative_rate=false_negatives / fn_tests if fn_tests > 0 else 0,
                verdict=args["verdict"],  # "SUPPORTED", "WEAKENED", "REFUTED"
                confidence_adjustment=final_adjustment,  # Computed from challenges or fallback
                revised_hypothesis=args.get("revised_hypothesis") if args.get("revised_hypothesis") else None,
                key_challenges=key_challenges,
                recommendations=recommendations,
                agent_reasoning=args.get("reasoning", ""),
                total_tests=skeptic.experiments_run,
                timestamp=datetime.now().isoformat(),
            )

            return {"content": [{"type": "text", "text": json.dumps({
                "submitted": True,
                "verdict": args["verdict"],
                "total_tests": skeptic.experiments_run,
                "alternatives_tested": len(skeptic.alternative_hypotheses),
                "boundary_tests": boundary_total,
                "boundary_pass_rate": boundary_passed / boundary_total if boundary_total > 0 else 0,
                "confounds_found": len(skeptic.confounds),
            })}]}

        # =====================================================================
        # Standard experimental tools (shared with Scientist)
        # =====================================================================

        @tool("test_activation", "Test if a prompt activates the neuron.", {"prompt": str})
        async def test_activation_tool(args):
            skeptic.experiments_run += 1
            result = _sync_test_activation(layer, neuron_idx, args["prompt"], config.activation_threshold)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_activation_test", "Test multiple prompts at once. Pass prompts as JSON array string.", {"prompts": str})
        async def batch_activation_tool(args):
            skeptic.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except:
                prompts = [args["prompts"]]
            result = _sync_batch_activation_test(layer, neuron_idx, prompts, config.activation_threshold)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("run_ablation", "Zero out the neuron and measure effect on output logits.", {"prompt": str, "top_k_logits": int})
        async def run_ablation_tool(args):
            skeptic.experiments_run += 1
            result = _sync_run_ablation(layer, neuron_idx, args["prompt"])
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_relp_connectivity", "Get RelP corpus-based upstream and downstream connections (optional validation).", {})
        async def get_relp_connectivity_tool(args):
            skeptic.experiments_run += 1
            edge_stats = str(skeptic.edge_stats_path) if skeptic.edge_stats_path else ""
            result = _sync_get_relp_connectivity(layer, neuron_idx, edge_stats)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("steer_neuron", "Add a value to neuron activation and measure logit shifts.", {
            "prompt": str, "steering_value": float, "position": int, "top_k_logits": int
        })
        async def steer_neuron_tool(args):
            skeptic.experiments_run += 1
            result = _sync_steer_neuron(
                layer, neuron_idx, args["prompt"],
                args.get("steering_value", 5.0),
                args.get("position", -1),
                args.get("top_k_logits", 10)
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("patch_activation", "Patch neuron activation from source prompt into target prompt (counterfactual test).", {
            "source_prompt": str, "target_prompt": str, "position": int
        })
        async def patch_activation_tool(args):
            skeptic.experiments_run += 1
            source = args["source_prompt"]
            target = args["target_prompt"]
            position = args.get("position", -1)
            result = await tool_patch_activation(layer, neuron_idx, source, target, position=position)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_output_projections", "Get actual output projections from model weights.", {"top_k": int})
        async def get_output_projections_tool(args):
            skeptic.experiments_run += 1
            result = _sync_get_output_projections(layer, neuron_idx, args.get("top_k", 10))
            # Respect polarity mode from investigation
            polarity = getattr(investigation, 'polarity_mode', 'positive')
            if polarity == "negative":
                promoted = result.get("promoted", [])
                suppressed = result.get("suppressed", [])
                result["promoted"] = suppressed
                result["suppressed"] = promoted
                result["polarity_note"] = "Projections flipped for NEGATIVE polarity investigation"
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("get_neuron_label", "Look up the label for any neuron.", {"neuron_id": str})
        async def get_neuron_label_tool(args):
            nid = args["neuron_id"]
            result = get_neuron_label_with_fallback(nid, str(skeptic.labels_path))
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("batch_get_neuron_labels", "Look up labels for multiple neurons.", {"neuron_ids": str})
        async def batch_get_neuron_labels_tool(args):
            try:
                neuron_ids = json.loads(args["neuron_ids"])
            except:
                neuron_ids = [args["neuron_ids"]]
            result = batch_get_neuron_labels_with_fallback(neuron_ids, str(skeptic.labels_path))
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        # =====================================================================
        # V4 Multi-token generation tools (for testing output effects)
        # =====================================================================

        @tool("ablate_and_generate", "Ablate neuron and generate multiple tokens to test output effects.", {
            "prompt": str, "max_new_tokens": int
        })
        async def ablate_and_generate_tool(args):
            """Multi-token ablation for testing output effects across generation."""
            skeptic.experiments_run += 1
            result = await tool_ablate_and_generate(
                layer=layer,
                neuron_idx=neuron_idx,
                prompt=args["prompt"],
                max_new_tokens=args.get("max_new_tokens", 10),
            )
            return result

        @tool("steer_and_generate", "Steer neuron and generate multiple tokens to test causal effects.", {
            "prompt": str, "steering_value": float, "max_new_tokens": int
        })
        async def steer_and_generate_tool(args):
            """Multi-token steering for testing causal effects across generation."""
            skeptic.experiments_run += 1
            result = await tool_steer_and_generate(
                layer=layer,
                neuron_idx=neuron_idx,
                prompt=args["prompt"],
                steering_value=args.get("steering_value", 5.0),
                max_new_tokens=args.get("max_new_tokens", 10),
            )
            return result

        # =====================================================================
        # V5 Batch and upstream dependency tools (optional for skeptic)
        # =====================================================================

        @tool("batch_ablate_and_generate", "Ablate neuron across multiple prompts and generate tokens.", {
            "prompts": str, "max_new_tokens": int
        })
        async def batch_ablate_and_generate_tool(args):
            """Batch multi-token ablation across multiple prompts."""
            skeptic.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except:
                prompts = [args["prompts"]]
            result = await tool_batch_ablate_and_generate(
                layer=layer,
                neuron_idx=neuron_idx,
                prompts=prompts,
                max_new_tokens=args.get("max_new_tokens", 10),
            )
            return result

        @tool("batch_steer_and_generate", "Steer neuron across multiple prompts and generate tokens.", {
            "prompts": str, "steering_value": float, "max_new_tokens": int
        })
        async def batch_steer_and_generate_tool(args):
            """Batch multi-token steering across multiple prompts."""
            skeptic.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except:
                prompts = [args["prompts"]]
            result = await tool_batch_steer_and_generate(
                layer=layer,
                neuron_idx=neuron_idx,
                prompts=prompts,
                steering_value=args.get("steering_value", 5.0),
                max_new_tokens=args.get("max_new_tokens", 10),
            )
            return result

        @tool("ablate_upstream_and_test", "Test if target neuron depends on specific upstream neurons.", {
            "upstream_neurons": str, "test_prompts": str, "window_tokens": int
        })
        async def ablate_upstream_and_test_tool(args):
            """Test upstream dependencies by ablating upstream neurons."""
            skeptic.experiments_run += 1
            try:
                upstream = json.loads(args["upstream_neurons"])
            except:
                upstream = [args["upstream_neurons"]]
            try:
                prompts = json.loads(args["test_prompts"])
            except:
                prompts = [args["test_prompts"]]
            result = await tool_ablate_upstream_and_test(
                layer=layer,
                neuron_idx=neuron_idx,
                upstream_neurons=upstream,
                test_prompts=prompts,
                window_tokens=args.get("window_tokens", 10),
            )
            return result

        @tool("steer_upstream_and_test", "Steer upstream neurons and measure effect on target neuron.", {
            "upstream_neuron": str, "test_prompts": str, "steering_value": float
        })
        async def steer_upstream_and_test_tool(args):
            """Test upstream influence by steering and measuring target activation."""
            skeptic.experiments_run += 1
            try:
                prompts = json.loads(args["test_prompts"])
            except:
                prompts = [args["test_prompts"]]
            result = await tool_steer_upstream_and_test(
                layer=layer,
                neuron_idx=neuron_idx,
                upstream_neuron=args["upstream_neuron"],
                test_prompts=prompts,
                steering_value=args.get("steering_value", 5.0),
            )
            return result

        @tool("batch_ablate_upstream_and_test", "Batch test upstream dependencies with domain-specific prompts.", {
            "upstream_neurons": str, "test_prompts": str
        })
        async def batch_ablate_upstream_and_test_tool(args):
            """Batch ablation of upstream neurons with category-specific prompts."""
            skeptic.experiments_run += 1
            try:
                upstream = json.loads(args["upstream_neurons"])
            except:
                upstream = [args["upstream_neurons"]]
            try:
                prompts = json.loads(args["test_prompts"])
            except:
                prompts = [args["test_prompts"]]
            result = await tool_batch_ablate_upstream_and_test(
                layer=layer,
                neuron_idx=neuron_idx,
                upstream_neurons=upstream,
                test_prompts=prompts,
            )
            return result

        @tool("batch_steer_upstream_and_test", "Batch steer upstream neurons and compare with RelP.", {
            "upstream_neurons": str, "test_prompts": str, "steering_value": float
        })
        async def batch_steer_upstream_and_test_tool(args):
            """Batch steering of upstream neurons with RelP comparison."""
            skeptic.experiments_run += 1
            try:
                upstream = json.loads(args["upstream_neurons"])
            except:
                upstream = [args["upstream_neurons"]]
            try:
                prompts = json.loads(args["test_prompts"])
            except:
                prompts = [args["test_prompts"]]
            result = await tool_batch_steer_upstream_and_test(
                layer=layer,
                neuron_idx=neuron_idx,
                upstream_neurons=upstream,
                test_prompts=prompts,
                steering_value=args.get("steering_value", 5.0),
            )
            return result

        @tool("run_category_selectivity_test", "Test neuron selectivity across domain-specific vs control prompts.", {
            "domain_prompts": str, "control_prompts": str
        })
        async def run_category_selectivity_test_tool(args):
            """Test if neuron is selective for a category vs controls."""
            skeptic.experiments_run += 1
            try:
                domain = json.loads(args["domain_prompts"])
            except:
                domain = [args["domain_prompts"]]
            try:
                control = json.loads(args["control_prompts"])
            except:
                control = [args["control_prompts"]]
            result = await tool_run_category_selectivity_test(
                layer=layer,
                neuron_idx=neuron_idx,
                domain_prompts=domain,
                control_prompts=control,
            )
            return result

        @tool("steer_dose_response", "Test dose-response relationship with steering values.", {
            "prompt": str, "steering_values": str
        })
        async def steer_dose_response_tool(args):
            """Test how different steering magnitudes affect output."""
            skeptic.experiments_run += 1
            try:
                values = json.loads(args["steering_values"])
            except:
                values = [1.0, 2.0, 5.0, 10.0]
            result = await tool_steer_dose_response(
                layer=layer,
                neuron_idx=neuron_idx,
                prompt=args["prompt"],
                steering_values=values,
            )
            return result

        @tool("intelligent_steering_analysis", "Sonnet-powered steering analysis. Generates prompts at decision boundaries, specifies steering values, runs experiments, and returns analysis with illustrative examples.", {
            "output_hypothesis": str, "promotes": str, "suppresses": str, "additional_instructions": str, "n_prompts": int, "max_new_tokens": int
        })
        async def intelligent_steering_analysis_tool(args):
            """Run intelligent steering analysis with Sonnet sub-agent."""
            skeptic.experiments_run += 1
            output_hypothesis = args.get("output_hypothesis", "")
            promotes_str = args.get("promotes", "[]")
            suppresses_str = args.get("suppresses", "[]")
            additional_instructions = args.get("additional_instructions")
            n_prompts = args.get("n_prompts", 100)
            max_new_tokens = args.get("max_new_tokens", 25)

            # Parse lists
            try:
                promotes = json.loads(promotes_str) if promotes_str else []
            except:
                promotes = [promotes_str] if promotes_str else []
            try:
                suppresses = json.loads(suppresses_str) if suppresses_str else []
            except:
                suppresses = [suppresses_str] if suppresses_str else []

            result = await tool_intelligent_steering_analysis(
                layer=layer,
                neuron_idx=neuron_idx,
                output_hypothesis=output_hypothesis,
                promotes=promotes,
                suppresses=suppresses,
                additional_instructions=additional_instructions,
                n_prompts=n_prompts,
                max_new_tokens=max_new_tokens,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool("test_additional_prompts", "Test follow-up prompts and merge into selectivity data. Use to probe specific patterns, test minimal pairs, or add adversarial tests.", {"prompts": str, "category": str, "category_type": str})
        async def test_additional_prompts_tool(args):
            skeptic.experiments_run += 1
            try:
                prompts = json.loads(args["prompts"])
            except json.JSONDecodeError:
                prompts = [p.strip() for p in args["prompts"].split("\n") if p.strip()]

            category = args.get("category", "skeptic_probe")
            category_type = args.get("category_type", "target")

            print(f"  [Skeptic Tool] test_additional_prompts: {len(prompts)} prompts in '{category}'")

            result = await tool_test_additional_prompts(
                layer=layer, neuron_idx=neuron_idx,
                prompts=prompts, category=category, category_type=category_type,
            )
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        return [
            # Skeptic-specific
            get_investigation_summary_tool,
            record_alternative_hypothesis_tool,
            register_hypothesis_tool,  # Register replacement hypotheses
            record_boundary_test_tool,
            record_confound_tool,
            challenge_hypothesis_tool,  # Challenge individual hypotheses
            submit_skeptic_report_tool,
            # Experimental (shared)
            test_activation_tool,
            batch_activation_tool,
            run_ablation_tool,
            get_relp_connectivity_tool,
            steer_neuron_tool,
            patch_activation_tool,  # Counterfactual testing
            get_output_projections_tool,
            get_neuron_label_tool,
            batch_get_neuron_labels_tool,
            # V4 multi-token tools
            ablate_and_generate_tool,
            steer_and_generate_tool,
            # V5 batch and upstream dependency tools (optional)
            batch_ablate_and_generate_tool,
            batch_steer_and_generate_tool,
            ablate_upstream_and_test_tool,
            steer_upstream_and_test_tool,
            batch_ablate_upstream_and_test_tool,
            batch_steer_upstream_and_test_tool,
            run_category_selectivity_test_tool,
            test_additional_prompts_tool,  # Follow-up selectivity probes
            steer_dose_response_tool,
            intelligent_steering_analysis_tool,
        ]

    async def run(self) -> SkepticReport:
        """Run adversarial testing and return report."""
        print(f"\n{'='*60}")
        print(f"NeuronSkeptic: Adversarial testing for {self.neuron_id}")
        print(f"Target hypothesis: {self.investigation.final_hypothesis[:80]}...")
        print(f"{'='*60}\n")

        # Create MCP server with tools
        tools = self._create_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="skeptic_tools",
            version="1.0.0",
            tools=tools,
        )

        # Build hypothesis summary for initial prompt
        hypothesis_summary = []
        for h in self.investigation.hypotheses_tested:
            hid = h.get("hypothesis_id", "?")
            status = h.get("status", "unknown")
            prior = h.get("prior_probability", 50)
            posterior = h.get("posterior_probability", prior)
            hypothesis_summary.append(f"  - {hid}: {status} (prior={prior}%, posterior={posterior}%) - {h.get('hypothesis', '')[:80]}")

        # Build polarity context
        polarity = getattr(self.investigation, 'polarity_mode', 'positive')
        if polarity == "negative":
            polarity_context = """
**POLARITY MODE: NEGATIVE FIRING**
This investigation covers the NEGATIVE firing function of this neuron. The scientist
investigated what makes the neuron fire NEGATIVELY (activation < 0). Output projections
have been pre-flipped. All activation values are negative (more negative = stronger).
"""
        else:
            polarity_context = ""

        # Build initial prompt
        initial_prompt = f"""{polarity_context}Your target is neuron {self.neuron_id} (layer {self.layer}).

The NeuronScientist has produced the following characterization:
**"{self.investigation.final_hypothesis}"**

Input function: {self.investigation.input_function}
Output function: {self.investigation.output_function}
Function type: {self.investigation.function_type}

**Hypotheses with per-hypothesis confidence:**
{chr(10).join(hypothesis_summary) if hypothesis_summary else "  (No hypotheses registered)"}

## CRITICAL FIRST STEP - CHECK FOR HYPOTHESIS-EVIDENCE MISMATCH

**⚠️ YOUR MOST IMPORTANT CHECK:**

1. **Call `get_output_projections` IMMEDIATELY**
   - What tokens does the neuron PROMOTE in its output projections?
   - Example: if "bus" has weight +1.08, the neuron PROMOTES "bus"

2. **Does the hypothesis match the output projections?**
   - If output promotes token X, but hypothesis says "detects Y" → MISMATCH
   - If output promotes token X, but scientist only tested prompts WITH X present → WRONG TESTS
   - **Key question**: If it promotes "bus", shouldn't it fire BEFORE "bus" to CAUSE bus to be predicted?

3. **Is this OUTPUT-SALIENT or INPUT-SALIENT?**
   - OUTPUT-SALIENT: Fires in contexts where its promoted token is likely to appear NEXT (as completion)
   - INPUT-SALIENT: Fires when a concept/token is present in input
   - **If output projections PROMOTE a token → likely OUTPUT-SALIENT**
   - Look at the output function - does it say "promotes" a token? That's output-salient.

4. **Is this DIRECT or INDIRECT output effect?**
   - Layer {self.layer} is {self._get_layer_description()}
   - Call `analyze_connectivity` to see downstream connections

5. **Design tests APPROPRIATE to the salience type**
   - For OUTPUT-SALIENT: Test COMPLETION contexts where the promoted token would naturally follow
   - Example: "I commute by ___" where "bus" is likely next
   - For INPUT-SALIENT: Test input presence
   - For INDIRECT effects: Check if downstream neurons carry the signal

Your job is to try to DISPROVE this hypothesis through APPROPRIATE testing.

Remember to:
1. FIRST identify salience type (output-salient vs input-salient, direct vs indirect)
2. Verify claimed output effects with `get_output_projections` and `steer_neuron`
3. Use your judgment to design tests that will most effectively challenge the hypothesis
4. Consider alternative hypotheses, boundary cases, and potential confounds as appropriate
5. Record all findings with record_* tools
6. If you refute or significantly weaken a hypothesis, register a replacement:
   - Call `register_hypothesis` with your improved explanation
   - Set `replaces` to the ID of the weakened/refuted hypothesis
   - Set `prior_probability` based on your evidence (40-70 typical)
7. Submit your final report with `submit_skeptic_report`

Start by calling `get_investigation_summary`, then `analyze_connectivity` and `get_output_projections` to understand the neuron's role before designing tests.

Be rigorous but fair - test the RIGHT thing. Let's begin."""

        # Configure options
        # Store transcripts in separate directory to avoid cluttering main project
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "neuron_reports" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        options = ClaudeAgentOptions(
            system_prompt=SKEPTIC_SYSTEM_PROMPT,
            max_turns=50,
            model=self.model,
            mcp_servers={"skeptic_tools": mcp_server},
            cwd=transcripts_dir,
            add_dirs=[project_root],  # Allow access to main project files
            allowed_tools=[
                "mcp__skeptic_tools__get_investigation_summary",
                "mcp__skeptic_tools__record_alternative_hypothesis",
                "mcp__skeptic_tools__register_hypothesis",  # Register replacement hypotheses
                "mcp__skeptic_tools__record_boundary_test",
                "mcp__skeptic_tools__record_confound",
                "mcp__skeptic_tools__challenge_hypothesis",  # Challenge individual hypotheses
                "mcp__skeptic_tools__submit_skeptic_report",
                "mcp__skeptic_tools__test_activation",
                "mcp__skeptic_tools__batch_activation_test",
                "mcp__skeptic_tools__run_ablation",
                "mcp__skeptic_tools__analyze_connectivity",
                "mcp__skeptic_tools__steer_neuron",
                "mcp__skeptic_tools__patch_activation",  # Counterfactual testing
                "mcp__skeptic_tools__get_output_projections",
                "mcp__skeptic_tools__get_neuron_label",
                "mcp__skeptic_tools__batch_get_neuron_labels",
                # V4 multi-token tools
                "mcp__skeptic_tools__ablate_and_generate",
                "mcp__skeptic_tools__steer_and_generate",
                # V5 batch and upstream dependency tools (optional)
                "mcp__skeptic_tools__batch_ablate_and_generate",
                "mcp__skeptic_tools__batch_steer_and_generate",
                "mcp__skeptic_tools__ablate_upstream_and_test",
                "mcp__skeptic_tools__steer_upstream_and_test",
                "mcp__skeptic_tools__batch_ablate_upstream_and_test",
                "mcp__skeptic_tools__batch_steer_upstream_and_test",
                "mcp__skeptic_tools__run_category_selectivity_test",
                "mcp__skeptic_tools__steer_dose_response",
                "mcp__skeptic_tools__intelligent_steering_analysis",
            ],
        )

        # Run agent
        print(f"Running skeptic agent (model: {self.model})...")

        # Track agent conversation (stored on self for external access)
        agent_messages = []
        self.agent_messages = agent_messages

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                content = block.text
                                agent_messages.append({"role": "assistant", "content": content})
                                preview = content[:150].replace("\n", " ")
                                print(f"Skeptic: {preview}...")

                            elif isinstance(block, ToolUseBlock):
                                print(f"  Tool: {block.name}")

                    elif isinstance(message, ResultMessage):
                        print(f"Result: {message.subtype}")
                        if message.subtype == "error":
                            print("  Error occurred during skeptic testing")

        except Exception as e:
            print(f"Skeptic agent error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"NeuronSkeptic complete: {self.experiments_run} experiments")
        if self.report:
            print(f"Verdict: {self.report.verdict}")
            print(f"Alternatives tested: {len(self.alternative_hypotheses)}")
            print(f"Boundary tests: {len(self.boundary_tests)}")
            print(f"Confounds found: {len(self.confounds)}")
        print(f"{'='*60}\n")

        # Return the report (or build one from recorded data if submit wasn't called)
        if self.report:
            return self.report
        else:
            # Build report from recorded data even if submit wasn't called
            boundary_passed = sum(1 for t in self.boundary_tests if t.passed)
            boundary_total = len(self.boundary_tests)

            false_positives = sum(1 for t in self.boundary_tests
                                 if t.expected_behavior == "should_not_activate" and not t.passed)
            false_negatives = sum(1 for t in self.boundary_tests
                                 if t.expected_behavior == "should_activate" and not t.passed)

            fp_tests = sum(1 for t in self.boundary_tests
                           if t.expected_behavior == "should_not_activate")
            fn_tests = sum(1 for t in self.boundary_tests
                           if t.expected_behavior == "should_activate")

            # Determine verdict based on evidence
            if len(self.alternative_hypotheses) == 0 and len(self.boundary_tests) == 0:
                verdict = "INCONCLUSIVE"
            elif any(alt.verdict == "alternative_better" for alt in self.alternative_hypotheses):
                verdict = "REFUTED"
            elif any(alt.verdict == "indistinguishable" for alt in self.alternative_hypotheses) or len(self.confounds) > 0 and any(c.severity == "critical" for c in self.confounds):
                verdict = "WEAKENED"
            else:
                verdict = "SUPPORTED" if boundary_passed >= boundary_total * 0.7 else "WEAKENED"

            # Compute overall confidence adjustment from hypothesis challenges (0-100 scale)
            computed_adjustment = sum(c.get("confidence_delta", 0) for c in self.hypothesis_challenges)
            fallback_adjustment = -10 if verdict == "WEAKENED" else (-30 if verdict == "REFUTED" else 0)
            final_adjustment = computed_adjustment if self.hypothesis_challenges else fallback_adjustment

            return SkepticReport(
                neuron_id=self.neuron_id,
                original_hypothesis=self.investigation.final_hypothesis,
                alternative_hypotheses=self.alternative_hypotheses,
                boundary_tests=self.boundary_tests,
                confounds=self.confounds,
                hypothesis_challenges=self.hypothesis_challenges,  # Individual hypothesis updates
                selectivity_score=boundary_passed / boundary_total if boundary_total > 0 else 0,
                false_positive_rate=false_positives / fp_tests if fp_tests > 0 else 0,
                false_negative_rate=false_negatives / fn_tests if fn_tests > 0 else 0,
                verdict=verdict,
                confidence_adjustment=final_adjustment,  # From challenges or fallback
                key_challenges=[
                    f"Agent ran {self.experiments_run} experiments but hit turn limit",
                    f"Tested {len(self.alternative_hypotheses)} alternative hypotheses",
                    f"Ran {len(self.boundary_tests)} boundary tests",
                    f"Found {len(self.confounds)} confounds",
                    f"Challenged {len(self.hypothesis_challenges)} hypotheses directly",
                ],
                agent_reasoning="Report auto-generated from recorded data (agent hit turn limit before submitting)",
                total_tests=self.experiments_run,
                timestamp=datetime.now().isoformat(),
            )


# =============================================================================
# Convenience Function
# =============================================================================

async def run_skeptic(
    neuron_id: str,
    investigation: NeuronInvestigation,
    edge_stats_path: Path | None = None,
    labels_path: Path | None = None,
    model: str = "sonnet",
    return_transcript: bool = False,
    gpu_server_url: str | None = None,
) -> SkepticReport:
    """Run NeuronSkeptic on an investigation.

    Args:
        neuron_id: Target neuron
        investigation: Scientist's investigation to attack
        edge_stats_path: Path to edge statistics
        labels_path: Path to neuron labels
        model: Claude model for skeptic agent
        return_transcript: If True, return (report, agent_messages) tuple
        gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")

    Returns:
        SkepticReport with adversarial findings (or tuple with transcript if return_transcript=True)
    """
    # Parse layer/neuron for GPU client agent ID
    parts = neuron_id.replace("L", "").replace("N", "").split("/")
    layer = int(parts[0])
    neuron_idx = int(parts[1])

    # Set up GPU client if server URL provided
    gpu_client = None
    if gpu_server_url:
        from neuron_scientist.gpu_client import GPUClient
        from neuron_scientist.tools import set_gpu_client
        gpu_client = GPUClient(gpu_server_url, agent_id=f"skeptic-L{layer}-N{neuron_idx}")
        await gpu_client.wait_for_server()
        set_gpu_client(gpu_client)

    try:
        skeptic = NeuronSkeptic(
            neuron_id=neuron_id,
            investigation=investigation,
            edge_stats_path=edge_stats_path,
            labels_path=labels_path,
            model=model,
            gpu_server_url=gpu_server_url,
        )
        report = await skeptic.run()

        if return_transcript:
            return report, getattr(skeptic, 'agent_messages', [])

        return report
    finally:
        # Clean up GPU client
        if gpu_client:
            from neuron_scientist.tools import set_gpu_client
            set_gpu_client(None)
            await gpu_client.close()
