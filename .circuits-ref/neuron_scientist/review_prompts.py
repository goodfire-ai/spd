"""GPT review prompt templates for NeuronPI orchestrator.

These prompts follow the 7-section delegation format required by the
Code Reviewer expert via mcp__codex__codex.

Evidence Truncation Limits
--------------------------
The distill_investigation_for_review() function truncates evidence to fit
within GPT's context window (~8K tokens for review prompt + evidence).

Adjust these limits if reviews are missing important evidence.
"""

# =============================================================================
# Review Evidence Truncation Constants
# =============================================================================
# More aggressive truncation than schemas.py since this goes to external GPT

# Activation examples for review
REVIEW_MAX_ACTIVATING = 5  # Top activating examples to show reviewer
REVIEW_MAX_NON_ACTIVATING = 3  # Negative examples (reviewer needs fewer)

# Experimental results for review
REVIEW_MAX_ABLATION = 5  # Ablation effects
REVIEW_MAX_STEERING = 5  # Steering results
REVIEW_MAX_DOSE_RESPONSE = 5  # Dose-response curves
REVIEW_MAX_TOKENS_PER_RESULT = 5  # Tokens shown per steering/ablation result

# Hypotheses and findings
REVIEW_MAX_HYPOTHESES = 5  # Hypotheses to show
REVIEW_MAX_FINDINGS = 5  # Key findings
REVIEW_MAX_QUESTIONS = 3  # Open questions

# Preview lengths (characters)
REVIEW_PROMPT_PREVIEW_LENGTH = 200  # Prompt text preview for evidence (increased from 80)
REVIEW_LABEL_PREVIEW_LENGTH = 80  # Neuron label preview (short is fine)

# =============================================================================
# Investigation Review Prompt
# =============================================================================

INVESTIGATION_REVIEW_PROMPT = """
TASK: Review neuron investigation for scientific rigor and evidence quality.

EXPECTED OUTCOME: APPROVE or REQUEST_CHANGES verdict with specific feedback on hypothesis support, evidence quality, and experimental completeness.

CONTEXT:
- Neuron: {neuron_id}
- Investigation summary:

{distilled_json}

CONSTRAINTS:
- Apply mechanistic interpretability standards
- Evidence must clearly support each hypothesis
- All mandatory validation steps must be complete (category selectivity, multi-token ablation)
- Per-hypothesis confidence (posterior probability) must match evidence strength

## HARD GATE CHECKLIST (MANDATORY - REJECT IF FAILED)

Check the `protocol_checklist` in the investigation. These are HARD REQUIREMENTS:

| Gate | Requirement |
|------|-------------|
| Category selectivity | Must be performed (V4 requirement) |
| Multi-token ablation | At least one `ablate_and_generate` test |
| Hypothesis pre-registration | ≥1 hypothesis registered |
| Input phase complete | Input investigation phase finished |
| Output phase complete | Output investigation phase finished |

**If `auto_reject_reasons` is non-empty, you MUST REQUEST_CHANGES regardless of evidence quality.**

MUST DO:
- First check `auto_reject_reasons` - if non-empty, verdict MUST be REQUEST_CHANGES
- Then evaluate these 4 criteria:
  1. **Hypothesis clarity**: Is the input/output function clearly stated and testable?
  2. **Evidence quality**: Do activation patterns, ablation effects, and steering results support the hypothesis?
  3. **Completeness**: Check `protocol_checklist`:
     - `category_selectivity_done`: Required V4 validation step
     - `multi_token_ablation_done`: Required output phase test
     - `hypotheses_preregistered`: Pre-registration performed?
     - `input_phase_complete`: Input investigation finished?
     - `output_phase_complete`: Output investigation finished?
  4. **Per-hypothesis confidence**: Each hypothesis has prior/posterior. Check if posteriors are justified by evidence.

## CRITICAL: Activation vs Output Weight Confusion Check

Check for this COMMON error pattern:

1. **The claim**: "This neuron [suppresses/promotes] [token X] on [context Y]"
2. **The evidence**: Activation on [context Y] is LOW (< 0.5) or near-zero
3. **The problem**: If activation is LOW, the neuron has MINIMAL EFFECT regardless of output weights!

**Valid claim pattern:**
- Context where neuron is ACTIVE (activation > 1.0)
- Output weights for token X are positive/negative
- Therefore: neuron promotes/suppresses token X WHEN ACTIVE

**Invalid claim pattern:**
- Context where neuron is INACTIVE or low activation (< 0.5)
- Output weights don't matter because activation * weight ≈ 0
- Claiming the neuron "suppresses" tokens in this context is WRONG

Look for contradictions like:
- "Neuron suppresses NSAID tokens" but activation on NSAID prompts is 0.07 (nearly zero)
- "Promotes token X" but activation in those contexts is < 0.5

If you find this pattern, REQUEST_CHANGES with specific feedback about the activation/output confusion.

## HYPOTHESIS CONFIDENCE ASSESSMENT

Confidence is tracked PER HYPOTHESIS, not overall. Check each hypothesis in `hypotheses_tested`:
- **prior**: Initial belief before testing (should be reasonable starting point)
- **posterior**: Updated belief after testing (should be justified by evidence)
- **status**: confirmed, refuted, or partially_confirmed

For each confirmed hypothesis with posterior > 80%:
- Verify strong positive evidence exists
- Check that alternative explanations were considered

For refuted hypotheses:
- Verify clear negative evidence exists
- Posterior should be appropriately low (< 30%)

- Provide specific, actionable feedback if requesting changes

MUST NOT DO:
- Approve if `auto_reject_reasons` is non-empty (CRITICAL)
- Approve high-posterior hypotheses without checking supporting evidence
- Nitpick minor phrasing or formatting
- Request experiments that are impossible with available tools
- Be overly harsh on exploratory findings with appropriately low posteriors

OUTPUT FORMAT:
VERDICT: [APPROVE / REQUEST_CHANGES]
HYPOTHESIS_ASSESSMENT: [per-hypothesis assessment summary]
GAPS:
- [gap 1]
- [gap 2]
FEEDBACK:
[Specific actionable items for improvement, or brief justification for approval]
"""

# =============================================================================
# Revision Context Template
# =============================================================================

REVISION_CONTEXT_TEMPLATE = """
## REVISION REQUEST FROM PEER REVIEW

Your previous investigation was reviewed by a peer and needs improvements.

### Review Verdict: {verdict}

### Confidence Assessment: {confidence_assessment}

### Issues Identified:
{feedback}

### Specific Gaps to Address:
{gaps}

---

## YOUR TASK

Address the gaps identified above. You should:

1. **Run targeted experiments** to fill evidence gaps
2. **Update hypothesis status** if new evidence changes conclusions
3. **Reconsider alternative explanations** if suggested
4. **Adjust confidence level** if evidence doesn't support current level

**IMPORTANT: Both phases are already complete from your prior investigation.**
- Do NOT call `complete_input_phase` or `complete_output_phase` again
- You can run any tool from either phase to address reviewer feedback
- All prior data is preserved and cached - build on it, don't restart

After addressing the gaps, call `save_structured_report` with your updated findings.
"""

# =============================================================================
# Prior Evidence Template (for additive investigations)
# =============================================================================

PRIOR_EVIDENCE_TEMPLATE = """
## PRIOR INVESTIGATION DATA (from iteration {iteration})

### ⚠️ CRITICAL: PHASE STATUS

**Both Input and Output phases are ALREADY COMPLETE.** This is a revision run - you completed a full investigation and saved a report in the prior iteration.

- ✅ Input Phase: COMPLETE (do NOT call `complete_input_phase` again)
- ✅ Output Phase: COMPLETE (do NOT call `complete_output_phase` again)
- ✅ Report: Already saved (you will save an UPDATED report when done)

**You do NOT need to re-complete phases.** Focus ONLY on addressing the specific gaps identified by the reviewer. Run targeted experiments, then call `save_structured_report` with your updated findings.

---

**IMPORTANT: This is a REVISION run. Do NOT re-run experiments already done** unless the reviewer specifically requested re-testing.

### Prior Hypotheses
You have already tested {hypothesis_count} hypotheses. Continue numbering from H{next_hypothesis_id}.
{hypotheses_summary}

### Prior Activation Evidence
Already tested {activating_count} activating prompts and {non_activating_count} non-activating prompts.
Do NOT re-test these unless you need to verify with different thresholds.

**Top Activating Prompts (already tested):**
{top_activating}

**Sample Non-Activating (already tested):**
{sample_non_activating}

### Prior RelP Results
Already ran {relp_count} RelP attributions ({relp_found_count} found neuron, {relp_corpus_count} from pre-computed corpus).
{relp_summary}

### Prior Steering Results
{steering_summary}

### Prior Dose-Response Results
{dose_response_summary}

### Current Characterization (from prior iteration)
- **Input Function**: {input_function}
- **Output Function**: {output_function}

### Hypothesis Confidence (from prior iteration)
{hypotheses_confidence}

### Hypotheses Needing Attention
{hypotheses_needing_replacement}

### Key Findings (from prior iteration)
{key_findings}

### Open Questions (from prior iteration)
{open_questions}

---

**YOUR FOCUS**: Address the reviewer feedback above. Run NEW experiments to fill gaps, don't repeat what's already done.

**NOTE**: All prior activation values and RelP results are cached. You can use `get_cached_activation(prompt)` and `lookup_relp_result(prompt)` to look up values without re-running tests.
"""


def _select_top_connections(
    connections: list,
    n_positive: int = 5,
    n_negative: int = 5,
) -> list:
    """Select top positive and top negative connections by absolute weight.

    Args:
        connections: List of connection dicts with 'weight' field
        n_positive: Number of top positive connections to include
        n_negative: Number of top negative connections to include

    Returns:
        List of dicts with id, label, and weight (positive first, then negative)
    """
    if not connections:
        return []

    # Separate positive and negative
    positive = [c for c in connections if c.get('weight', 0) > 0]
    negative = [c for c in connections if c.get('weight', 0) < 0]

    # Sort by absolute weight (descending)
    positive_sorted = sorted(positive, key=lambda x: abs(x.get('weight', 0)), reverse=True)[:n_positive]
    negative_sorted = sorted(negative, key=lambda x: abs(x.get('weight', 0)), reverse=True)[:n_negative]

    # Extract relevant fields with consistent naming
    result = []
    for n in positive_sorted + negative_sorted:
        result.append({
            'id': n.get('neuron_id', n.get('id', '')),
            'label': n.get('label', '')[:REVIEW_LABEL_PREVIEW_LENGTH] if n.get('label') else '',
            'weight': n.get('weight', 0)
        })
    return result


def summarize_prior_evidence(investigation_dict: dict) -> dict:
    """Extract key evidence summaries from a prior investigation for prompt injection.

    Args:
        investigation_dict: The to_dict() output of a NeuronInvestigation

    Returns:
        Dict with formatted summary fields for PRIOR_EVIDENCE_TEMPLATE
    """
    # Extract characterization
    char = investigation_dict.get("characterization", {})

    # Extract evidence
    evidence = investigation_dict.get("evidence", {})
    activating = evidence.get("activating_prompts", investigation_dict.get("activating_prompts", []))
    non_activating = evidence.get("non_activating_prompts", investigation_dict.get("non_activating_prompts", []))
    relp_results = investigation_dict.get("relp_results", evidence.get("relp_results", []))

    # Extract hypotheses
    hypotheses = investigation_dict.get("hypotheses_tested", [])

    # Format top activating prompts (limit to 5)
    top_activating_lines = []
    for p in sorted(activating, key=lambda x: -x.get("activation", 0))[:5]:
        prompt_preview = p.get("prompt", "")[:REVIEW_PROMPT_PREVIEW_LENGTH]
        act = p.get("activation", 0)
        top_activating_lines.append(f'- "{prompt_preview}..." (activation={act:.2f})')

    # Format sample non-activating (limit to 3)
    sample_non_act_lines = []
    for p in non_activating[:3]:
        prompt_preview = p.get("prompt", "")[:REVIEW_PROMPT_PREVIEW_LENGTH]
        act = p.get("activation", 0)
        sample_non_act_lines.append(f'- "{prompt_preview}..." (activation={act:.2f})')

    # Format hypotheses summary
    hypotheses_lines = []
    for h in hypotheses:
        h_id = h.get("hypothesis_id", "?")
        h_text = h.get("hypothesis", "")[:100]
        h_status = h.get("status", "unknown")
        prior = h.get("prior_probability", "?")
        posterior = h.get("posterior_probability", "?")
        hypotheses_lines.append(f'- **{h_id}** ({h_status}): "{h_text}..." [{prior}% -> {posterior}%]')

    # Format RelP summary (ALL results with FULL prompts for complete context)
    relp_lines = []
    for r in relp_results:  # No limit - show ALL RelP results
        prompt_full = r.get("prompt", "")  # NO truncation - full prompt
        found = "✓ found" if r.get("neuron_found") else "✗ not found"
        tau = r.get("tau", "?")
        relp_score = r.get("relp_score") or r.get("neuron_relp_score")
        score_str = f", score={relp_score:.3f}" if relp_score else ""
        relp_lines.append(f'- "{prompt_full}" (tau={tau}{score_str}): {found}')

    # Format key findings
    key_findings = investigation_dict.get("key_findings", [])
    findings_lines = [f"- {f}" for f in key_findings[:5]]

    # Format open questions
    open_questions = investigation_dict.get("open_questions", [])
    questions_lines = [f"- {q}" for q in open_questions[:3]]

    # Extract and format steering results
    # Handle both formats: individual steering and dose-response summary
    steering_results = investigation_dict.get("steering_results", evidence.get("steering_results", []))
    steering_lines = []
    for s in steering_results:
        prompt = s.get("prompt", "")
        # Handle both singular steering_value and plural steering_values
        if s.get("steering_values"):
            # Dose-response summary format
            steering_val = f"[{len(s['steering_values'])} values]"
            max_shift = s.get("max_shift")
            pattern = s.get("pattern", "")
            kendall = s.get("kendall_tau")
            logit_range = s.get("logit_range")
            details = []
            if pattern:
                details.append(f"pattern={pattern}")
            if kendall is not None:
                details.append(f"τ={kendall:.2f}")
            if max_shift is not None:
                details.append(f"max_shift={max_shift:.2f}")
            if logit_range is not None:
                details.append(f"logit_range={logit_range:.2f}")
            details_str = ", ".join(details) if details else ""
            steering_lines.append(f'- "{prompt}" {details_str}')
        else:
            # Individual steering format
            steering_val = s.get("steering_value", "?")
            promotes = s.get("promotes", s.get("promoted_tokens", []))[:3]
            suppresses = s.get("suppresses", s.get("suppressed_tokens", []))[:3]
            max_shift = s.get("max_shift")

            def format_token(p):
                if isinstance(p, dict):
                    return p.get("token", str(p))
                elif isinstance(p, (list, tuple)) and len(p) >= 1:
                    return str(p[0])  # Handle [token, shift] format
                return str(p)

            promotes_str = ", ".join(format_token(p) for p in promotes)
            suppresses_str = ", ".join(format_token(p) for p in suppresses)
            shift_str = f", max_shift={max_shift:.2f}" if max_shift is not None else ""
            steering_lines.append(
                f'- "{prompt}" @ {steering_val}: promotes [{promotes_str}], suppresses [{suppresses_str}]{shift_str}'
            )

    # Extract and format dose-response results
    dose_response_results = investigation_dict.get("dose_response_results", evidence.get("dose_response_results", []))
    dose_response_lines = []
    for dr in dose_response_results:
        prompt = dr.get("prompt", "")
        pattern = dr.get("pattern", "unknown")
        is_monotonic = dr.get("is_monotonic", False)
        kendall_tau = dr.get("kendall_tau")
        responsive_tokens = dr.get("responsive_tokens", [])[:3]
        # Extract max_shift from dose_response_curve if present
        max_shift = None
        curve = dr.get("dose_response_curve", [])
        if curve:
            shifts = [entry.get("max_shift", 0) for entry in curve if entry.get("max_shift") is not None]
            if shifts:
                max_shift = max(shifts, key=abs)
        tau_str = f", τ={kendall_tau:.2f}" if kendall_tau is not None else ""
        max_shift_str = f", max_shift={max_shift:.2f}" if max_shift is not None else ""
        monotonic_str = "✓ monotonic" if is_monotonic else "✗ non-monotonic"
        tokens_str = ", ".join(t.get("token", t) if isinstance(t, dict) else str(t) for t in responsive_tokens)
        dose_response_lines.append(
            f'- "{prompt}": {pattern} ({monotonic_str}{tau_str}{max_shift_str}), tokens: [{tokens_str}]'
        )

    # Calculate next hypothesis ID
    next_h_id = len(hypotheses) + 1

    # Build per-hypothesis confidence summary
    hypotheses_confidence_lines = []
    for h in hypotheses:
        h_id = h.get("hypothesis_id", "?")
        status = h.get("status", "unknown")
        prior = h.get("prior_probability", 50)
        posterior = h.get("posterior_probability", prior)
        hypotheses_confidence_lines.append(f"- **{h_id}** ({status}): {prior}% → {posterior}%")

    # Build hypotheses needing attention (weakened/refuted without replacement)
    hypotheses_needing_replacement = []
    replaced_ids = {h.get("replaces") for h in hypotheses if h.get("replaces")}
    for h in hypotheses:
        status = h.get("status", "").lower()
        h_id = h.get("hypothesis_id", "?")
        if status in ("weakened", "refuted") and h_id not in replaced_ids:
            hypotheses_needing_replacement.append(
                f"- **{h_id}** ({status}): \"{h.get('hypothesis', '')[:80]}...\" — needs replacement or refinement"
            )

    return {
        "iteration": investigation_dict.get("total_experiments", 0) // 100 + 1,  # Rough estimate
        "hypothesis_count": len(hypotheses),
        "next_hypothesis_id": next_h_id,
        "hypotheses_summary": "\n".join(hypotheses_lines) if hypotheses_lines else "(none yet)",
        "activating_count": len(activating),
        "non_activating_count": len(non_activating),
        "top_activating": "\n".join(top_activating_lines) if top_activating_lines else "(none found)",
        "sample_non_activating": "\n".join(sample_non_act_lines) if sample_non_act_lines else "(none tested)",
        "relp_count": len(relp_results),
        "relp_found_count": sum(1 for r in relp_results if r.get('neuron_found')),
        "relp_corpus_count": sum(1 for r in relp_results if r.get('source') == 'corpus'),
        "relp_summary": "\n".join(relp_lines) if relp_lines else "(no RelP runs yet)",
        "steering_summary": "\n".join(steering_lines) if steering_lines else "(no steering tests yet)",
        "dose_response_summary": "\n".join(dose_response_lines) if dose_response_lines else "(no dose-response tests yet)",
        "input_function": char.get("input_function", "(not determined)"),
        "output_function": char.get("output_function", "(not determined)"),
        "hypotheses_confidence": "\n".join(hypotheses_confidence_lines) if hypotheses_confidence_lines else "(no hypotheses yet)",
        "key_findings": "\n".join(findings_lines) if findings_lines else "(none yet)",
        "open_questions": "\n".join(questions_lines) if questions_lines else "(none noted)",
        "hypotheses_needing_replacement": "\n".join(hypotheses_needing_replacement) if hypotheses_needing_replacement else "(none)",
    }


# =============================================================================
# Skeptic Findings Template (for revision context injection)
# =============================================================================

SKEPTIC_FINDINGS_TEMPLATE = """
## SKEPTIC ADVERSARIAL FINDINGS (CRITICAL - Read Before Planning)

**Overall Verdict: {verdict}**

### Hypotheses Weakened or Refuted
{weakened_hypotheses}

### Why They Were Weakened
{challenge_details}

### Skeptic-Proposed Alternatives (candidates for replacement)
{skeptic_alternatives}

### Confounds Detected
{confounds}

---

**YOUR RESPONSE:** For EACH weakened/refuted hypothesis above:
1. Register a refined REPLACEMENT hypothesis addressing the specific criticism, OR
2. Explain why no replacement is warranted.

Do NOT re-assert the original hypothesis without new evidence addressing the criticism.
"""


def summarize_skeptic_findings(investigation_dict: dict) -> str | None:
    """Extract skeptic findings from an investigation for revision context injection.

    Args:
        investigation_dict: The to_dict() output of a NeuronInvestigation

    Returns:
        Formatted string with skeptic findings, or None if no skeptic report
    """
    skeptic_report = investigation_dict.get("skeptic_report")
    if not skeptic_report:
        return None

    verdict = skeptic_report.get("verdict", "UNKNOWN")

    # Extract weakened/refuted hypotheses with their evidence_against
    hypotheses = investigation_dict.get("hypotheses_tested", [])
    weakened_lines = []
    for h in hypotheses:
        status = h.get("status", "").lower()
        if status in ("weakened", "refuted"):
            h_id = h.get("hypothesis_id", "?")
            h_text = h.get("hypothesis", "")[:150]
            prior = h.get("prior_probability", "?")
            posterior = h.get("posterior_probability", "?")
            weakened_lines.append(f"- **{h_id}** ({status}): \"{h_text}\" [{prior}% → {posterior}%]")

    # Extract challenge details
    challenge_lines = []
    hypothesis_challenges = skeptic_report.get("hypothesis_challenges", [])
    for hc in hypothesis_challenges:
        if hc.get("result") in ("weakened", "refuted"):
            h_id = hc.get("hypothesis_id", "?")
            challenge = hc.get("challenge", "")[:200]
            evidence = hc.get("evidence", "")[:200]
            challenge_lines.append(f"- **{h_id}**: {challenge}")
            if evidence:
                challenge_lines.append(f"  Evidence: {evidence}")

    # Extract skeptic alternatives
    alt_lines = []
    alternatives = skeptic_report.get("alternative_hypotheses", [])
    for alt in alternatives:
        alt_text = alt.get("alternative", "")[:150]
        alt_verdict = alt.get("verdict", "unknown")
        alt_evidence = alt.get("evidence", "")[:100]
        alt_lines.append(f"- [{alt_verdict}] \"{alt_text}\"")
        if alt_evidence:
            alt_lines.append(f"  Evidence: {alt_evidence}")

    # Extract confounds
    confound_lines = []
    confounds = skeptic_report.get("confounds", [])
    for c in confounds:
        severity = c.get("severity", "unknown")
        factor = c.get("factor", "unknown")
        desc = c.get("description", "")[:150]
        confound_lines.append(f"- [{severity}] {factor}: {desc}")

    # Only return if there's something useful to report
    if not weakened_lines and not challenge_lines and not alt_lines and not confound_lines:
        return None

    return SKEPTIC_FINDINGS_TEMPLATE.format(
        verdict=verdict,
        weakened_hypotheses="\n".join(weakened_lines) if weakened_lines else "(none)",
        challenge_details="\n".join(challenge_lines) if challenge_lines else "(no specific challenges recorded)",
        skeptic_alternatives="\n".join(alt_lines) if alt_lines else "(none proposed)",
        confounds="\n".join(confound_lines) if confound_lines else "(none detected)",
    )


# =============================================================================
# Code Reviewer Expert Prompt
# =============================================================================

CODE_REVIEWER_EXPERT_PROMPT = """
You are an expert reviewer for mechanistic interpretability research on neural networks.

Your role is to evaluate neuron investigation reports for scientific rigor, focusing on:

1. **Evidence Quality**: Do the experiments actually support each hypothesis?
   - Activation patterns should be consistent and specific
   - Multi-token ablation effects should show clear causal influence
   - Steering should produce predictable effects on generation

2. **Validation Completeness**: Were proper V4 validation steps run?
   - Category selectivity test (required in V4)
   - Multi-token ablation via ablate_and_generate (required for output phase)
   - Both input and output phases must be complete

3. **Hypothesis Precision**: Is each hypothesis clearly specified?
   - Input hypotheses: What activates the neuron?
   - Output hypotheses: What does the neuron promote/suppress?
   - Both should be specific enough to make predictions

4. **Per-Hypothesis Confidence**: Does each hypothesis's posterior match its evidence?
   - Each hypothesis has prior (initial belief) and posterior (after testing)
   - High posteriors (>80%) need strong supporting evidence
   - Refuted hypotheses should have low posteriors (<30%)

Your output must follow this exact format:

VERDICT: [APPROVE / REQUEST_CHANGES]
HYPOTHESIS_ASSESSMENT: [summary of per-hypothesis confidence evaluation]
GAPS:
- [list specific missing evidence or experiments]
FEEDBACK:
[Actionable items for improvement, or justification for approval]

Be constructive - the goal is to improve the investigation, not to reject it.
Focus on substantive issues that affect the validity of conclusions.
"""

# =============================================================================
# Distillation Template
# =============================================================================

def distill_investigation_for_review(investigation) -> dict:
    """Extract key fields from investigation for GPT review.

    Args:
        investigation: NeuronInvestigation object or dict

    Returns:
        Distilled dict suitable for GPT review
    """
    # Handle both object and dict input
    if hasattr(investigation, '__dict__'):
        inv = investigation.__dict__
    else:
        inv = investigation

    # Extract characterization
    char = inv.get('characterization', {})
    if not char and inv.get('dashboard'):
        dash = inv.get('dashboard', {})
        char = {
            'input_function': dash.get('input_function', ''),
            'output_function': dash.get('output_function', ''),
            'function_type': dash.get('function_type', 'unknown'),
            'final_hypothesis': dash.get('summary', ''),
        }

    # Extract evidence samples (limit to prevent token overflow)
    # Handle both nested format (from to_dict()) and flat format
    # Limits defined at module level for easy adjustment
    evidence_section = inv.get('evidence', {})
    activating = evidence_section.get('activating_prompts', inv.get('activating_prompts', []))[:REVIEW_MAX_ACTIVATING]
    non_activating = evidence_section.get('non_activating_prompts', inv.get('non_activating_prompts', []))[:REVIEW_MAX_NON_ACTIVATING]
    ablation_raw = evidence_section.get('ablation_effects', inv.get('ablation_effects', []))[:REVIEW_MAX_ABLATION]
    relp_results = inv.get('relp_results', evidence_section.get('relp_results', []))
    connectivity = evidence_section.get('connectivity', inv.get('connectivity', {}))
    steering_results = inv.get('steering_results', evidence_section.get('steering_results', []))
    dose_response_results = inv.get('dose_response_results', evidence_section.get('dose_response_results', []))

    # Extract hypotheses tested
    hypotheses = inv.get('hypotheses_tested', [])

    # Check validation completeness
    # Check for baseline z-score in evidence or key findings
    baseline_done = False
    baseline_zscore = None
    for finding in inv.get('key_findings', []):
        finding_lower = str(finding).lower()
        if 'z-score' in finding_lower or 'zscore' in finding_lower or 'z score' in finding_lower:
            baseline_done = True
            # Try to extract z-score value
            import re
            match = re.search(r'z[_\- ]?score[:\s=]*(?:of\s+)?([0-9.]+)', finding_lower)
            if match:
                baseline_zscore = float(match.group(1))
                break

    validation = {
        'baseline_comparison': baseline_done or any(
            h.get('hypothesis_type') == 'baseline' or
            'baseline' in str(h.get('hypothesis', '')).lower()
            for h in hypotheses
        ),
        # Check for actual dose-response data, not just string matching
        'dose_response': len(dose_response_results) > 0 or any(
            'dose' in str(h.get('hypothesis', '')).lower() or
            'steering' in str(h.get('hypothesis', '')).lower() or
            'dose-response' in str(h.get('evidence', [])).lower()
            for h in hypotheses
        ),
        'relp_verification': len(relp_results) > 0 or any(
            'relp' in str(h.get('evidence', [])).lower()
            for h in hypotheses
        ) or any(
            'relp' in str(f).lower()
            for f in inv.get('key_findings', [])
        ),
    }

    # Check for z-score in hypotheses or findings
    zscore_from_hypotheses = None
    for h in hypotheses:
        h_evidence = h.get('evidence', [])
        for e in h_evidence:
            if 'z-score' in str(e).lower() or 'zscore' in str(e).lower():
                # Try to extract z-score value
                import re
                match = re.search(r'z[_\- ]?score[:\s=]*(?:of\s+)?([0-9.]+)', str(e).lower())
                if match:
                    zscore_from_hypotheses = float(match.group(1))
                    break

    # Prefer z-score from key_findings (more reliable), fall back to hypotheses
    final_zscore = baseline_zscore or zscore_from_hypotheses

    # Process ablation effects - handle both formats:
    # Format 1: [{'token': 'x', 'shift': 0.5}, ...]
    # Format 2: [{'promotes': [['token', shift], ...], 'suppresses': [...]}]
    ablation_processed = []
    for item in ablation_raw:
        if isinstance(item, dict):
            if 'token' in item and 'shift' in item:
                # Format 1: direct token/shift
                ablation_processed.append({'token': item['token'], 'shift': item.get('shift', 0)})
            elif 'promotes' in item or 'suppresses' in item:
                # Format 2: promotes/suppresses lists
                for token, shift in item.get('promotes', [])[:3]:
                    ablation_processed.append({'token': f'+{token}', 'shift': shift})
                for token, shift in item.get('suppresses', [])[:3]:
                    ablation_processed.append({'token': f'-{token}', 'shift': shift})

    # =========================================================================
    # Extract protocol_validation from investigation (if available)
    # This is set by save_structured_report from protocol state tracker
    # =========================================================================
    protocol_validation = inv.get('protocol_validation', {})

    # Build explicit protocol checklist with hard gates
    protocol_zscore = protocol_validation.get('baseline_zscore', final_zscore)

    # Extract dose-response data from actual results if available
    has_dose_response_data = len(dose_response_results) > 0
    dose_response_monotonic = protocol_validation.get('dose_response_monotonic', False)
    dose_response_kendall_tau = protocol_validation.get('dose_response_kendall_tau')

    # If we have actual dose-response results, extract monotonic/kendall_tau from them
    if has_dose_response_data and not dose_response_monotonic:
        for dr in dose_response_results:
            if dr.get('is_monotonic'):
                dose_response_monotonic = True
                break
        # Also extract kendall_tau if not in protocol_validation
        if dose_response_kendall_tau is None:
            for dr in dose_response_results:
                if dr.get('kendall_tau') is not None:
                    dose_response_kendall_tau = dr.get('kendall_tau')
                    break

    # V4 Protocol checklist - focused on phase completion and required tests
    category_selectivity_done = protocol_validation.get('category_selectivity_done', False)
    multi_token_ablation_done = protocol_validation.get('multi_token_ablation_done', False)
    input_phase_complete = protocol_validation.get('input_phase_complete', inv.get('input_phase_complete', False))
    output_phase_complete = protocol_validation.get('output_phase_complete', inv.get('output_phase_complete', False))

    protocol_checklist = {
        'phase0_corpus_queried': protocol_validation.get('phase0_corpus_queried', False),
        'category_selectivity_done': category_selectivity_done,
        'multi_token_ablation_done': multi_token_ablation_done,
        'input_phase_complete': input_phase_complete,
        'output_phase_complete': output_phase_complete,
        # Legacy fields for backwards compatibility
        'baseline_done': protocol_validation.get('baseline_comparison_done', validation.get('baseline_comparison', False)),
        'baseline_passes': protocol_zscore is not None and protocol_zscore >= 2.0,
        'baseline_zscore': protocol_zscore,
        'dose_response_done': protocol_validation.get('dose_response_done', has_dose_response_data or validation.get('dose_response', False)),
        'dose_response_monotonic': dose_response_monotonic,
        'dose_response_kendall_tau': dose_response_kendall_tau,
        'relp_runs': protocol_validation.get('relp_runs', len(relp_results)),
        'relp_positive_control': protocol_validation.get('relp_positive_control', len(relp_results) > 0),
        'relp_negative_control': protocol_validation.get('relp_negative_control', False),
        'hypotheses_preregistered': protocol_validation.get('hypotheses_registered', len(hypotheses)) > 0,
    }

    # Determine auto-reject reasons based on V4 protocol requirements
    auto_reject_reasons = []

    # Hard gate 1: V4 requires category selectivity test
    if not category_selectivity_done:
        auto_reject_reasons.append(
            "V4 required: Category selectivity test not performed. "
            "Run run_category_selectivity_test() to validate neuron selectivity."
        )

    # Hard gate 2: V4 requires multi-token ablation for output phase
    if not multi_token_ablation_done:
        auto_reject_reasons.append(
            "V4 required: Multi-token ablation test not performed. "
            "Run ablate_and_generate() to test output effects across multiple tokens."
        )

    # Hard gate 3: Hypothesis pre-registration required
    if not protocol_checklist['hypotheses_preregistered']:
        auto_reject_reasons.append(
            "Required: No hypotheses were pre-registered. "
            "Use register_hypothesis() before running experiments."
        )

    # Hard gate 4: Both phases must be complete
    if not input_phase_complete:
        auto_reject_reasons.append(
            "Input investigation phase not complete. "
            "Call complete_input_phase() when input characterization is done."
        )

    if not output_phase_complete:
        auto_reject_reasons.append(
            "Output investigation phase not complete. "
            "Call complete_output_phase() when output characterization is done."
        )

    # Hard gate 5: Check for high-posterior hypotheses without strong evidence
    for h in hypotheses:
        posterior = h.get('posterior_probability', 50)
        h_id = h.get('hypothesis_id', '?')
        status = h.get('status', 'unknown')
        evidence = h.get('evidence', [])

        # High posterior (>80%) confirmed hypotheses need substantial evidence
        if posterior > 80 and status == 'confirmed' and len(evidence) < 2:
            auto_reject_reasons.append(
                f"Hypothesis {h_id} has high posterior ({posterior}%) but only "
                f"{len(evidence)} piece(s) of evidence. High-confidence claims need more support."
            )

    # Hard gate 6: Activation vs Output Weight Confusion Detection
    # Check if output_function claims effects on contexts where activation is LOW
    output_function = char.get('output_function', '').lower()
    if output_function:
        # Keywords that indicate claimed effects
        effect_keywords = ['suppresses', 'promotes', 'increases', 'decreases', 'boosts', 'inhibits']
        has_effect_claim = any(kw in output_function for kw in effect_keywords)

        if has_effect_claim and activating:
            # Check if any "non-activating" contexts have low activation but are mentioned in output claims
            low_activation_prompts = [
                p for p in non_activating
                if p.get('activation', 0) < 0.5
            ]

            # Also check activating prompts for suspiciously low values mentioned alongside high ones
            # (indicates the agent may be claiming effects on contexts where neuron is inactive)
            min_activating = min((p.get('activation', 0) for p in activating), default=0)
            max_activating = max((p.get('activation', 0) for p in activating), default=0)

            # Flag if there's a huge range in "activating" prompts (> 10x difference)
            # This often indicates confusion about what contexts the neuron is active on
            if max_activating > 0 and min_activating < 0.5 and max_activating > 2.0:
                auto_reject_reasons.append(
                    f"Potential activation/output confusion: 'activating' prompts range from "
                    f"{min_activating:.2f} to {max_activating:.2f}. Prompts with activation < 0.5 "
                    f"should not be considered 'activating' - the neuron's output weights have "
                    f"minimal effect when activation is near zero."
                )

            # Check if output_function mentions "negative activation" - this is often wrong
            if 'negative activation' in output_function or 'activation (-' in output_function:
                # Verify by checking actual activation values
                actual_negative = any(p.get('activation', 0) < 0 for p in activating + non_activating)
                if not actual_negative:
                    auto_reject_reasons.append(
                        "Output function claims 'negative activation' but no activation examples "
                        "show actual negative values. This may be confusing output weights with "
                        "activation values. Low positive activation (e.g., 0.07) is NOT negative."
                    )

    return {
        'neuron_id': inv.get('neuron_id', ''),
        # NOTE: Overall confidence removed in V4 - confidence is now per-hypothesis only
        # See hypotheses_tested for prior/posterior per hypothesis
        'hypothesis': char.get('final_hypothesis', ''),
        'input_function': char.get('input_function', ''),
        'output_function': char.get('output_function', ''),
        'function_type': char.get('function_type', 'unknown'),
        'evidence': {
            'activating_examples': [
                {'prompt': p.get('prompt', '')[:100], 'activation': p.get('activation', 0)}
                for p in activating
            ],
            'non_activating_examples': [
                {'prompt': p.get('prompt', '')[:100], 'activation': p.get('activation', 0)}
                for p in non_activating
            ],
            'ablation_effects': ablation_processed[:10],  # Limit to 10 effects
            'baseline_zscore': protocol_zscore,
            'relp_count': len(relp_results),
            'relp_found_count': sum(1 for r in relp_results if r.get('neuron_found')),
            'relp_corpus_count': sum(1 for r in relp_results if r.get('source') == 'corpus'),
            # ALL RelP results with FULL prompts - no truncation
            'relp_summary': [
                {
                    'prompt': r.get('prompt', ''),  # NO truncation - full prompt
                    'neuron_found': r.get('neuron_found', False),
                    'in_causal_pathway': r.get('in_causal_pathway', False),
                    'tau': r.get('tau'),
                    'relp_score': r.get('relp_score') or r.get('neuron_relp_score'),
                    'source': r.get('source', 'agent'),
                    'graph_stats': r.get('graph_stats'),  # Include graph stats
                }
                # Sort by neuron_found (True first) to show positive evidence
                for r in sorted(relp_results, key=lambda x: (not x.get('neuron_found', False), x.get('source') != 'corpus'))
                # NO LIMIT - show ALL RelP results
            ] if relp_results else [],
            'connectivity': {
                'upstream_count': len(connectivity.get('upstream_neurons', [])),
                'downstream_count': len(connectivity.get('downstream_targets', [])),
                # Use helper to select top 5 positive + top 5 negative by absolute weight
                'top_upstream': _select_top_connections(
                    connectivity.get('upstream_neurons', []),
                    n_positive=5, n_negative=5
                ),
                'top_downstream': _select_top_connections(
                    connectivity.get('downstream_targets', []),
                    n_positive=5, n_negative=5
                ),
            },
            # Include actual steering results (not just "was it done")
            # Handle both formats: individual steering and dose-response summary
            'steering_results': [
                {
                    'prompt': s.get('prompt', '')[:100],
                    # Handle both singular steering_value and plural steering_values (dose-response format)
                    'steering_value': s.get('steering_value') if s.get('steering_value') is not None else s.get('steering_values'),
                    'promotes': s.get('promotes', s.get('promoted_tokens', []))[:5],
                    'suppresses': s.get('suppresses', s.get('suppressed_tokens', []))[:5],
                    # Extract max_shift (contains the ±3 logit claim)
                    'max_shift': s.get('max_shift'),
                    # Include dose-response metadata if present
                    'pattern': s.get('pattern'),
                    'kendall_tau': s.get('kendall_tau'),
                    'logit_range': s.get('logit_range'),
                }
                for s in steering_results[:10]  # Limit to 10 results
            ] if steering_results else [],
            # Include actual dose-response data
            'dose_response_results': [
                {
                    'prompt': d.get('prompt', '')[:100],
                    'steering_values': d.get('steering_values', []),
                    'pattern': d.get('pattern'),
                    'kendall_tau': d.get('kendall_tau'),
                    'is_monotonic': d.get('is_monotonic'),
                    'responsive_tokens': d.get('responsive_tokens', [])[:5],
                    # Extract max_shift from curve (largest absolute shift across all steering values)
                    'max_shift': max(
                        (entry.get('max_shift', 0) for entry in d.get('dose_response_curve', []) if entry.get('max_shift') is not None),
                        key=abs,
                        default=None
                    ),
                    'dose_response_curve': d.get('dose_response_curve', [])[:5],  # Limit curve to 5 points
                }
                for d in dose_response_results[:5]  # Limit to 5 results
            ] if dose_response_results else [],
        },
        'hypotheses_tested': [
            {
                'id': h.get('hypothesis_id', ''),
                'hypothesis': h.get('hypothesis', '')[:200],
                'status': h.get('status', 'unknown'),
                'prior': h.get('prior_probability', 50),
                'posterior': h.get('posterior_probability', 50),
            }
            for h in hypotheses[:5]
        ],
        'key_findings': inv.get('key_findings', [])[:REVIEW_MAX_FINDINGS],
        'open_questions': inv.get('open_questions', [])[:REVIEW_MAX_QUESTIONS],
        'validation_complete': validation,
        'total_experiments': inv.get('total_experiments', 0),
        # NEW: Protocol checklist and auto-reject reasons for hard gates
        'protocol_checklist': protocol_checklist,
        'auto_reject_reasons': auto_reject_reasons,
        'missing_validation': protocol_validation.get('missing_validation', []),
    }


def parse_review_response(response_text: str) -> dict:
    """Parse GPT review response into structured result.

    Args:
        response_text: Raw text response from GPT

    Returns:
        Dict with verdict, hypothesis_assessment, gaps, feedback
    """
    result = {
        'verdict': 'REQUEST_CHANGES',  # Default to cautious
        'hypothesis_assessment': '',  # V4: per-hypothesis assessment summary
        'gaps': [],
        'feedback': '',
        'raw_response': response_text,
    }

    lines = response_text.strip().split('\n')
    current_section = None

    for line in lines:
        line_stripped = line.strip()

        # Parse verdict
        if line_stripped.upper().startswith('VERDICT:'):
            verdict_text = line_stripped.split(':', 1)[1].strip().upper()
            if 'APPROVE' in verdict_text:
                result['verdict'] = 'APPROVE'
            else:
                result['verdict'] = 'REQUEST_CHANGES'

        # Parse hypothesis assessment (V4) or legacy confidence assessment
        elif line_stripped.upper().startswith('HYPOTHESIS_ASSESSMENT:'):
            result['hypothesis_assessment'] = line_stripped.split(':', 1)[1].strip()
        elif line_stripped.upper().startswith('CONFIDENCE_ASSESSMENT:'):
            # Legacy support - map to hypothesis_assessment
            result['hypothesis_assessment'] = line_stripped.split(':', 1)[1].strip().lower()

        # Parse gaps section
        elif line_stripped.upper().startswith('GAPS:'):
            current_section = 'gaps'

        # Parse feedback section
        elif line_stripped.upper().startswith('FEEDBACK:'):
            current_section = 'feedback'

        # Collect content for current section
        elif current_section == 'gaps' and line_stripped.startswith('-'):
            result['gaps'].append(line_stripped[1:].strip())

        elif current_section == 'feedback' and line_stripped:
            if result['feedback']:
                result['feedback'] += '\n' + line_stripped
            else:
                result['feedback'] = line_stripped

    return result


def apply_hard_gates(review_result: dict, distilled: dict) -> dict:
    """Apply hard gates to enforce protocol validation, overriding lenient APPROVE verdicts.

    This function checks for auto_reject_reasons from the distilled investigation
    and overrides an APPROVE verdict to REQUEST_CHANGES if hard gates are violated.

    Args:
        review_result: The parsed review response from GPT
        distilled: The distilled investigation dict with protocol_checklist and auto_reject_reasons

    Returns:
        Updated review result with hard gates applied
    """
    auto_reject_reasons = distilled.get('auto_reject_reasons', [])

    # If there are auto-reject reasons and verdict was APPROVE, override it
    if auto_reject_reasons and review_result.get('verdict') == 'APPROVE':
        review_result['verdict'] = 'REQUEST_CHANGES'
        review_result['hard_gate_override'] = True

        # Add auto-reject reasons to gaps
        existing_gaps = review_result.get('gaps', [])
        for reason in auto_reject_reasons:
            if reason not in existing_gaps:
                existing_gaps.append(f"[HARD GATE] {reason}")
        review_result['gaps'] = existing_gaps

        # Update feedback to note the override
        override_notice = (
            "\n\n**[HARD GATE OVERRIDE]** This investigation was auto-rejected due to "
            f"{len(auto_reject_reasons)} protocol violation(s) despite passing reviewer assessment. "
            "The following issues MUST be addressed before approval:\n"
        )
        for i, reason in enumerate(auto_reject_reasons, 1):
            override_notice += f"\n{i}. {reason}"

        review_result['feedback'] = override_notice + "\n\n" + review_result.get('feedback', '')

    # Add protocol checklist to result for transparency
    review_result['protocol_checklist'] = distilled.get('protocol_checklist', {})

    return review_result
