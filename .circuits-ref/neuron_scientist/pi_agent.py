"""NeuronPI - Principal Investigator orchestrator agent.

Orchestrates the full neuron investigation pipeline:
1. Launch NeuronScientist for initial investigation
2. Send results to GPT for review (via Codex MCP)
3. Re-prompt NeuronScientist with feedback to fill gaps
4. Iterate until approved or max iterations
5. Launch Dashboard Agent to generate HTML report

This is a Claude Agent SDK agent that uses MCP tools including Codex delegation.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


def _atomic_write_json(filepath: Path, data: dict, indent: int = 2) -> None:
    """Write JSON data atomically using temp file + rename.

    This ensures that if a crash occurs during write, the original file
    (if any) is preserved. The rename operation is atomic on POSIX systems.

    Args:
        filepath: Target file path
        data: Data to serialize as JSON
        indent: JSON indentation level
    """
    # Write to temp file in same directory (important for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=filepath.stem + '_',
        dir=filepath.parent
    )
    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=indent)
        # Atomic rename (POSIX)
        os.replace(temp_path, filepath)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

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

from .agent import investigate_neuron
from .review_prompts import (
    INVESTIGATION_REVIEW_PROMPT,
    REVISION_CONTEXT_TEMPLATE,
    distill_investigation_for_review,
    parse_review_response,
)
from .schemas import NeuronInvestigation, PIResult, ReviewResult, SkepticReport
from .skeptic_agent import run_skeptic
from .transcript_summarizer import (
    summarize_gpt_review,
    summarize_scientist_transcript,
    summarize_skeptic_transcript,
)

# =============================================================================
# System Prompt for NeuronPI
# =============================================================================

PI_SYSTEM_PROMPT = """You are NeuronPI, a Principal Investigator orchestrating neuron investigations.

## Your Role

You manage the full investigation pipeline for understanding what individual neurons do. You coordinate between:
- **NeuronScientist**: An agent that runs experiments on neurons
- **NeuronSkeptic**: An adversarial agent that stress-tests hypotheses
- **GPT Code Reviewer**: A peer reviewer that evaluates investigation quality
- **Dashboard Agent**: Generates HTML reports from findings

## Available Tools

### Investigation
- `run_investigation`: Launch NeuronScientist to investigate a neuron
- `run_investigation_with_revision`: Re-run investigation with GPT feedback

### Adversarial Testing
- `run_skeptic`: Launch NeuronSkeptic to stress-test the hypothesis
  - Tries to disprove the hypothesis through alternative explanations, boundary tests, confound detection
  - Returns verdict: SUPPORTED, WEAKENED, or REFUTED with evidence

### Review
- `request_gpt_review`: Send investigation + skeptic findings to GPT for peer review
  - Returns: APPROVE or REQUEST_CHANGES with specific feedback

### Dashboard
- `generate_dashboard`: Create HTML dashboard from investigation results

### Utilities
- `load_investigation`: Load an existing investigation JSON
- `save_pi_result`: Save the final PI result with review history

## Workflow (MANDATORY SEQUENCE)

**IMPORTANT: You MUST follow this exact sequence. Do NOT skip any step.**

1. **Initial Investigation** (REQUIRED)
   - Call `run_investigation` with the neuron ID
   - This runs NeuronScientist which produces investigation.json (consolidated format)

2. **Adversarial Testing** (REQUIRED - DO NOT SKIP)
   - **YOU MUST call `run_skeptic` BEFORE requesting GPT review**
   - The Skeptic stress-tests the hypothesis through:
     - Alternative explanations (is it X or really Y?)
     - Boundary tests (edge cases that should/shouldn't activate)
     - Confound detection (position effects, co-occurrence, etc.)
   - The Skeptic's findings provide adversarial evidence for the review
   - **If you skip this step, the review will be incomplete**

3. **Peer Review Loop** (max 2 iterations)
   - Call `request_gpt_review` - includes BOTH scientist AND skeptic findings
   - If APPROVE: proceed to dashboard generation
   - If REQUEST_CHANGES: call `run_investigation_with_revision` with the feedback
   - **After revision, run the skeptic again on the updated investigation**
   - Repeat until approved or max iterations reached

4. **Dashboard Generation** (REQUIRED)
   - Call `generate_dashboard` to create HTML report

5. **Save Results** (REQUIRED)
   - Call `save_pi_result` with the final outcome

## Guidelines

- **MANDATORY**: Run the skeptic after EVERY scientist investigation (initial and revisions)
- The skeptic's job is adversarial - it's OK if it finds weaknesses
- If skeptic says REFUTED, consider this seriously in the review
- Request GPT review only AFTER running both scientist AND skeptic
- If GPT requests changes, pass the FULL feedback to the revision run
- After 2 iterations, accept the best result and note remaining issues
- Always generate a dashboard at the end

## Output Format

Report your progress as you go:
- "Starting investigation for {neuron_id}..."
- "Investigation complete. Running adversarial testing..."
- "Skeptic verdict: {SUPPORTED/WEAKENED/REFUTED}. {summary}"
- "Requesting GPT review with combined evidence..."
- "GPT verdict: {APPROVE/REQUEST_CHANGES}. {summary}"
- "Running revision to address: {gaps}"
- "Generating dashboard..."
- "Pipeline complete. Final verdict: {verdict}"
"""


# =============================================================================
# NeuronPI Agent Class
# =============================================================================

class NeuronPI:
    """Principal Investigator agent for orchestrating neuron investigations."""

    # Default labels path
    DEFAULT_LABELS_PATH = Path("data/neuron_labels_combined.json")
    # Default output directories (consolidated location)
    DEFAULT_OUTPUT_DIR = Path("neuron_reports/json")
    DEFAULT_HTML_DIR = Path("neuron_reports/html")

    def __init__(
        self,
        neuron_id: str,
        initial_label: str = "",
        initial_hypothesis: str = "",
        edge_stats_path: Path | None = None,
        labels_path: Path | None = None,
        output_dir: Path = Path("neuron_reports/json"),
        html_output_dir: Path | None = None,  # HTML dashboard output directory
        model: str = "sonnet",  # Sonnet is good for orchestration
        scientist_model: str = "opus",  # Opus for actual investigation
        max_review_iterations: int = 3,
        skip_review: bool = False,
        skip_dashboard: bool = False,
        codex_available: bool = True,  # Whether Codex MCP is available
        existing_investigation_path: Path | None = None,  # For review-only mode
        polarity_mode: str = "positive",  # "positive" or "negative" firing investigation
        gpu_server_url: str | None = None,  # URL for GPU inference server
    ):
        """Initialize NeuronPI.

        Args:
            neuron_id: Target neuron (e.g., "L15/N7890")
            initial_label: Initial label from batch labeling
            initial_hypothesis: Starting hypothesis to test
            edge_stats_path: Path to edge statistics JSON
            labels_path: Path to neuron labels JSON
            output_dir: Directory to save outputs
            html_output_dir: Directory for HTML dashboards (defaults to neuron_reports/html)
            model: Claude model for PI agent orchestration
            scientist_model: Claude model for NeuronScientist investigations
            max_review_iterations: Max review cycles before forcing completion
            skip_review: Skip GPT review (investigation + dashboard only)
            skip_dashboard: Skip dashboard generation
            codex_available: Whether Codex MCP is available for GPT review
            existing_investigation_path: Path to existing investigation JSON (review-only mode)
            polarity_mode: "positive" or "negative" - which firing direction to investigate
            gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")
        """
        self.neuron_id = neuron_id
        self.initial_label = initial_label
        self.initial_hypothesis = initial_hypothesis
        self.edge_stats_path = edge_stats_path
        self.labels_path = labels_path or self.DEFAULT_LABELS_PATH
        self.output_dir = Path(output_dir)
        self.html_output_dir = Path(html_output_dir) if html_output_dir else Path("neuron_reports/html")
        self.model = model
        self.scientist_model = scientist_model
        self.max_review_iterations = max_review_iterations
        self.skip_review = skip_review
        self.skip_dashboard = skip_dashboard
        self.codex_available = codex_available
        self.existing_investigation_path = existing_investigation_path
        self.polarity_mode = polarity_mode
        self.gpu_server_url = gpu_server_url

        # Results tracking
        self.current_investigation: NeuronInvestigation | None = None
        self.current_skeptic_report: SkepticReport | None = None
        self.review_history: list[ReviewResult] = []
        self.iterations = 0

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_output_dir.mkdir(parents=True, exist_ok=True)

    def _create_mcp_tools(self):
        """Create MCP tools for the PI agent."""

        neuron_id = self.neuron_id
        initial_label = self.initial_label
        initial_hypothesis = self.initial_hypothesis
        edge_stats_path = self.edge_stats_path
        labels_path = self.labels_path
        output_dir = self.output_dir
        scientist_model = self.scientist_model
        pi = self  # Reference for tracking

        @tool(
            "run_investigation",
            "Run NeuronScientist to investigate the neuron. Returns investigation summary. IMPORTANT: Use max_experiments=100 for thorough investigation (default). Only use lower values for quick tests.",
            {"max_experiments": int}
        )
        async def run_investigation_tool(args):
            max_experiments = args.get("max_experiments", 100)
            print(f"\n[PI] Launching NeuronScientist for {neuron_id} (max_experiments={max_experiments})...")

            try:
                investigation, agent_messages = await investigate_neuron(
                    neuron_id=neuron_id,
                    initial_label=initial_label,
                    initial_hypothesis=initial_hypothesis,
                    edge_stats_path=edge_stats_path,
                    labels_path=labels_path,
                    output_dir=output_dir,
                    max_experiments=max_experiments,
                    model=scientist_model,
                    return_transcript=True,
                    polarity_mode=pi.polarity_mode,
                    gpu_server_url=pi.gpu_server_url,
                )

                pi.current_investigation = investigation
                pi.iterations += 1

                # Summarize the scientist transcript
                print("[PI] Summarizing scientist transcript...")
                try:
                    scientist_summary = await summarize_scientist_transcript(
                        neuron_id=neuron_id,
                        agent_messages=agent_messages,
                        iteration=pi.iterations,
                    )
                    investigation.transcript_summaries.append(scientist_summary)
                    print(f"[PI] Scientist summary: {scientist_summary.get('summary', '')[:100]}...")

                    # Re-save investigation with transcript summary
                    safe_id = neuron_id.replace("/", "_")
                    polarity_suffix = "_negative" if pi.polarity_mode == "negative" else ""
                    investigation_path = output_dir / f"{safe_id}{polarity_suffix}_investigation.json"
                    _atomic_write_json(investigation_path, investigation.to_dict())
                except Exception as e:
                    print(f"[PI] Warning: Failed to summarize scientist transcript: {e}")

                # Build per-hypothesis summary
                hypothesis_summary = []
                for h in investigation.hypotheses_tested:
                    h_id = h.get("hypothesis_id", "?")
                    status = h.get("status", "pending")
                    posterior = h.get("posterior_probability", 50)
                    hypothesis_summary.append(f"{h_id}: {status} ({posterior}%)")

                # Return summary (V4: per-hypothesis confidence instead of overall)
                polarity_suffix = "_negative" if pi.polarity_mode == "negative" else ""
                result = {
                    "status": "success",
                    "neuron_id": investigation.neuron_id,
                    "total_experiments": investigation.total_experiments,
                    "input_function": investigation.input_function,
                    "output_function": investigation.output_function,
                    "hypotheses": hypothesis_summary,
                    "key_findings": investigation.key_findings[:5],
                    "investigation_path": str(output_dir / f"{neuron_id.replace('/', '_')}{polarity_suffix}_investigation.json"),
                }
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

            except Exception as e:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": str(e),
                }, indent=2)}]}

        @tool(
            "run_investigation_with_revision",
            "Re-run investigation with GPT feedback to address gaps. Builds on prior investigation data (additive). Use max_experiments=50 for revisions (default).",
            {"revision_context": str, "max_experiments": int}
        )
        async def run_investigation_with_revision_tool(args):
            revision_context = args.get("revision_context", "")
            max_experiments = args.get("max_experiments", 50)  # Fewer for revision
            print(f"\n[PI] Running revision investigation for {neuron_id} (max_experiments={max_experiments})...")

            # Track prior experiments for logging
            prior_experiments = pi.current_investigation.total_experiments if pi.current_investigation else 0

            try:
                investigation, agent_messages = await investigate_neuron(
                    neuron_id=neuron_id,
                    initial_label=initial_label,
                    initial_hypothesis=initial_hypothesis,
                    edge_stats_path=edge_stats_path,
                    labels_path=labels_path,
                    output_dir=output_dir,
                    max_experiments=max_experiments,
                    model=scientist_model,
                    revision_context=revision_context,
                    prior_investigation=pi.current_investigation,  # Pass prior for additive investigation
                    return_transcript=True,
                    polarity_mode=pi.polarity_mode,
                    gpu_server_url=pi.gpu_server_url,
                )

                pi.current_investigation = investigation
                pi.iterations += 1

                # Summarize the revision transcript
                print("[PI] Summarizing revision transcript...")
                try:
                    scientist_summary = await summarize_scientist_transcript(
                        neuron_id=neuron_id,
                        agent_messages=agent_messages,
                        iteration=pi.iterations,
                    )
                    investigation.transcript_summaries.append(scientist_summary)
                    print(f"[PI] Revision summary: {scientist_summary.get('summary', '')[:100]}...")
                except Exception as e:
                    print(f"[PI] Warning: Failed to summarize revision transcript: {e}")

                # Analyze which gaps were addressed by the revision
                gaps_addressed = []
                if investigation.revision_history:
                    last_revision = investigation.revision_history[-1]
                    gaps_to_check = last_revision.get("gaps_remaining", [])

                    # Simple heuristic: gap is addressed if new experiments were run
                    # and new evidence matches gap keywords
                    new_experiments_count = investigation.total_experiments - prior_experiments
                    new_findings = investigation.key_findings[-5:] if len(investigation.key_findings) > 5 else investigation.key_findings

                    for gap in gaps_to_check:
                        gap_lower = gap.lower()
                        # Check if any new finding seems to address this gap
                        for finding in new_findings:
                            finding_lower = finding.lower()
                            # Simple keyword overlap check
                            if any(word in finding_lower for word in gap_lower.split() if len(word) > 3):
                                gaps_addressed.append(gap)
                                break

                    # Update revision history with addressed gaps
                    last_revision["gaps_addressed"] = gaps_addressed
                    last_revision["gaps_remaining"] = [g for g in gaps_to_check if g not in gaps_addressed]

                    if gaps_addressed:
                        print(f"[PI] Revision addressed {len(gaps_addressed)}/{len(gaps_to_check)} gaps")
                    if last_revision["gaps_remaining"]:
                        print(f"[PI] Gaps still remaining: {len(last_revision['gaps_remaining'])}")

                # Build per-hypothesis summary
                hypothesis_summary = []
                for h in investigation.hypotheses_tested:
                    h_id = h.get("hypothesis_id", "?")
                    status = h.get("status", "pending")
                    posterior = h.get("posterior_probability", 50)
                    hypothesis_summary.append(f"{h_id}: {status} ({posterior}%)")

                result = {
                    "status": "success",
                    "iteration": pi.iterations,
                    "neuron_id": investigation.neuron_id,
                    "total_experiments": investigation.total_experiments,
                    "prior_experiments": prior_experiments,
                    "new_experiments": investigation.total_experiments - prior_experiments,
                    "input_function": investigation.input_function,
                    "output_function": investigation.output_function,
                    "hypotheses": hypothesis_summary,
                    "key_findings": investigation.key_findings[:5],
                    "gaps_addressed": gaps_addressed,
                    "gaps_remaining": investigation.revision_history[-1].get("gaps_remaining", []) if investigation.revision_history else [],
                    "note": "Additive investigation - built on prior data",
                }
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

            except Exception as e:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": str(e),
                }, indent=2)}]}

        @tool(
            "run_skeptic",
            "Run NeuronSkeptic to adversarially test the hypothesis. Should be called after investigation, before review.",
            {}
        )
        async def run_skeptic_tool(args):
            if pi.current_investigation is None:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "No investigation to test. Run investigation first.",
                }, indent=2)}]}

            # Use the same model as the scientist for consistency
            skeptic_model = scientist_model
            print(f"\n[PI] Launching NeuronSkeptic for {neuron_id}...")
            print(f"[PI] Target hypothesis: {pi.current_investigation.final_hypothesis[:80]}...")

            try:
                skeptic_report, skeptic_messages = await run_skeptic(
                    neuron_id=neuron_id,
                    investigation=pi.current_investigation,
                    edge_stats_path=edge_stats_path,
                    labels_path=labels_path,
                    model=skeptic_model,
                    return_transcript=True,
                    gpu_server_url=pi.gpu_server_url,
                )

                pi.current_skeptic_report = skeptic_report

                # Embed skeptic report in investigation for dashboard access
                pi.current_investigation.skeptic_report = skeptic_report.to_dict()

                # Summarize the skeptic transcript
                print("[PI] Summarizing skeptic transcript...")
                try:
                    # Count existing skeptic summaries to determine iteration
                    skeptic_iteration = sum(1 for s in pi.current_investigation.transcript_summaries if s.get("agent") == "skeptic") + 1
                    skeptic_summary = await summarize_skeptic_transcript(
                        neuron_id=neuron_id,
                        hypothesis=pi.current_investigation.final_hypothesis,
                        agent_messages=skeptic_messages,
                        iteration=skeptic_iteration,
                    )
                    pi.current_investigation.transcript_summaries.append(skeptic_summary)
                    print(f"[PI] Skeptic summary: {skeptic_summary.get('summary', '')[:100]}...")
                except Exception as e:
                    print(f"[PI] Warning: Failed to summarize skeptic transcript: {e}")

                # Apply individual hypothesis challenges to investigation's hypotheses_tested
                # This is the NEW mechanism - updates come from challenge_hypothesis calls
                hypothesis_updates_applied = 0
                for challenge in skeptic_report.hypothesis_challenges:
                    h_id = challenge.get("hypothesis_id")
                    for h in pi.current_investigation.hypotheses_tested:
                        if h.get("hypothesis_id") == h_id:
                            # Update posterior probability (normalize to 0-100 scale)
                            new_post = challenge.get("new_posterior", h.get("posterior_probability", 50))
                            if isinstance(new_post, (int, float)) and 0 < new_post <= 1.0:
                                new_post = int(round(new_post * 100))
                            h["posterior_probability"] = new_post
                            # Update status
                            if challenge.get("result") == "refuted":
                                h["status"] = "refuted"
                            elif challenge.get("result") == "weakened":
                                h["status"] = "weakened"
                            # Note: evidence_against was already updated in challenge_hypothesis tool
                            hypothesis_updates_applied += 1
                            break

                if hypothesis_updates_applied > 0:
                    print(f"[PI] Applied {hypothesis_updates_applied} individual hypothesis updates from skeptic")

                # Auto-promote skeptic alternative hypotheses that were found to be better/supported
                auto_promoted = 0
                existing_texts = {h.get("hypothesis", "").lower().strip() for h in pi.current_investigation.hypotheses_tested}
                for alt in skeptic_report.alternative_hypotheses:
                    if alt.verdict not in ("alternative_better", "alternative_stronger", "supported"):
                        continue
                    # Skip if skeptic already registered this via register_hypothesis
                    if alt.alternative.lower().strip() in existing_texts:
                        continue
                    # Generate next H{n} ID
                    existing_ids = [h.get("hypothesis_id", "") for h in pi.current_investigation.hypotheses_tested]
                    max_n = 0
                    for hid in existing_ids:
                        if hid.startswith("H") and hid[1:].isdigit():
                            max_n = max(max_n, int(hid[1:]))
                    new_id = f"H{max_n + 1}"

                    prior = 55 if alt.verdict == "alternative_better" else 45
                    new_hypothesis = {
                        "hypothesis_id": new_id,
                        "hypothesis": alt.alternative,
                        "hypothesis_type": "unknown",
                        "confirmation_criteria": "",
                        "refutation_criteria": "",
                        "prior_probability": prior,
                        "posterior_probability": prior,
                        "status": "registered",
                        "source": "skeptic_auto_promoted",
                        "evidence_for": [{"evidence": alt.evidence, "source": "skeptic_alternative_test"}],
                        "evidence_against": [],
                    }
                    pi.current_investigation.hypotheses_tested.append(new_hypothesis)
                    existing_texts.add(alt.alternative.lower().strip())
                    auto_promoted += 1

                if auto_promoted > 0:
                    print(f"[PI] Auto-promoted {auto_promoted} skeptic alternative hypotheses")

                # NOTE: V4 - confidence is now per-hypothesis, not overall
                # The skeptic updates individual hypothesis posteriors via hypothesis_challenges
                # Overall confidence adjustment is deprecated

                # Build per-hypothesis summary after skeptic updates
                hypothesis_summary = []
                for h in pi.current_investigation.hypotheses_tested:
                    h_id = h.get("hypothesis_id", "?")
                    status = h.get("status", "pending")
                    posterior = h.get("posterior_probability", 50)
                    hypothesis_summary.append(f"{h_id}: {status} ({posterior}%)")

                # Build summary with per-hypothesis data
                result = {
                    "status": "success",
                    "verdict": skeptic_report.verdict,
                    "hypothesis_challenges": len(skeptic_report.hypothesis_challenges),
                    "hypothesis_updates_applied": hypothesis_updates_applied,
                    "hypotheses_after_skeptic": hypothesis_summary,
                    "alternatives_tested": len(skeptic_report.alternative_hypotheses),
                    "boundary_tests": len(skeptic_report.boundary_tests),
                    "boundary_pass_rate": skeptic_report.selectivity_score,
                    "confounds_found": len(skeptic_report.confounds),
                    "false_positive_rate": skeptic_report.false_positive_rate,
                    "false_negative_rate": skeptic_report.false_negative_rate,
                    "key_challenges": skeptic_report.key_challenges[:5],
                    "recommendations": skeptic_report.recommendations[:3],
                    "revised_hypothesis": skeptic_report.revised_hypothesis,
                }

                # Re-save investigation JSON with embedded skeptic report (atomic)
                safe_id = neuron_id.replace("/", "_")
                polarity_suffix = "_negative" if pi.polarity_mode == "negative" else ""
                investigation_path = output_dir / f"{safe_id}{polarity_suffix}_investigation.json"
                _atomic_write_json(investigation_path, pi.current_investigation.to_dict())
                result["investigation_path"] = str(investigation_path)

                # Also save standalone skeptic report for separate access (atomic)
                skeptic_path = output_dir / f"{safe_id}{polarity_suffix}_skeptic_report.json"
                _atomic_write_json(skeptic_path, skeptic_report.to_dict())
                result["skeptic_report_path"] = str(skeptic_path)

                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": str(e),
                }, indent=2)}]}

        @tool(
            "request_gpt_review",
            "Send investigation + skeptic findings to GPT Code Reviewer for peer review. Returns APPROVE or REQUEST_CHANGES.",
            {}
        )
        async def request_gpt_review_tool(args):
            if pi.current_investigation is None:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "No investigation to review. Run investigation first.",
                }, indent=2)}]}

            print("\n[PI] Requesting GPT review...")

            # Distill investigation for review
            distilled = distill_investigation_for_review(pi.current_investigation.to_dict())

            # Add skeptic findings if available
            skeptic_section = ""
            if pi.current_skeptic_report:
                skeptic = pi.current_skeptic_report

                # Build hypothesis challenges summary
                hyp_challenges_text = ""
                if skeptic.hypothesis_challenges:
                    hyp_challenges_text = "\n**Individual Hypothesis Challenges:**\n"
                    for hc in skeptic.hypothesis_challenges[:5]:
                        hyp_challenges_text += f"- {hc.get('hypothesis_id')}: {hc.get('result')} ({hc.get('confidence_delta', 0):+.2f}) - {hc.get('challenge', '')[:100]}\n"

                skeptic_section = f"""

## ADVERSARIAL TESTING (NeuronSkeptic)

The NeuronSkeptic agent attempted to DISPROVE the hypothesis through adversarial testing.

**Skeptic Verdict: {skeptic.verdict}**
- Hypotheses challenged: {len(skeptic.hypothesis_challenges)}
- Overall confidence adjustment: {skeptic.confidence_adjustment:+.2f}
- Selectivity score: {skeptic.selectivity_score:.2f}
- False positive rate: {skeptic.false_positive_rate:.2f}
- False negative rate: {skeptic.false_negative_rate:.2f}
{hyp_challenges_text}
**Alternative Hypotheses Tested ({len(skeptic.alternative_hypotheses)}):**
{chr(10).join(f'- {alt.alternative}: {alt.verdict}' for alt in skeptic.alternative_hypotheses[:5])}

**Boundary Tests ({len(skeptic.boundary_tests)} total, {sum(1 for t in skeptic.boundary_tests if t.passed)} passed):**
{chr(10).join(f'- {t.description}: {"PASS" if t.passed else "FAIL"} (activation={t.actual_activation:.2f})' for t in skeptic.boundary_tests[:8])}

**Confounds Detected ({len(skeptic.confounds)}):**
{chr(10).join(f'- [{c.severity}] {c.factor}: {c.description}' for c in skeptic.confounds[:5])}

**Key Challenges:**
{chr(10).join(f'- {c}' for c in skeptic.key_challenges[:5])}

**Skeptic Recommendations:**
{chr(10).join(f'- {r}' for r in skeptic.recommendations[:3])}

{f'**Revised Hypothesis (from Skeptic):** {skeptic.revised_hypothesis}' if skeptic.revised_hypothesis else ''}
"""

            # Build review prompt with skeptic findings
            # NOTE: V4 - confidence is now per-hypothesis, not overall
            review_prompt = INVESTIGATION_REVIEW_PROMPT.format(
                neuron_id=pi.current_investigation.neuron_id,
                distilled_json=json.dumps(distilled, indent=2),
            ) + skeptic_section

            # Call GPT via Codex CLI directly
            import tempfile

            # Write prompt to temp file to avoid shell escaping issues
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(review_prompt)
                prompt_file = f.name

            try:
                # Call codex CLI using exec subcommand
                codex_path = "/mnt/polished-lake/home/ctigges/.npm-global/bin/codex"
                cmd = [
                    codex_path, "exec",
                    "-m", "gpt-5.2-codex",
                    "-s", "read-only",
                    "--skip-git-repo-check",
                    "-"  # Read prompt from stdin
                ]

                # Read prompt to pass via stdin
                with open(prompt_file) as f:
                    prompt_text = f.read()

                print("[PI] Calling GPT via Codex exec...")
                result = subprocess.run(
                    cmd,
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=Path.cwd(),
                )

                gpt_response = result.stdout.strip()
                if result.returncode != 0:
                    print(f"[PI] Codex returned non-zero: {result.returncode}")
                    if result.stderr:
                        print(f"[PI] stderr: {result.stderr[:200]}")
                if not gpt_response and result.stderr:
                    gpt_response = f"Error: {result.stderr}"

                print(f"[PI] GPT review received ({len(gpt_response)} chars)")

                # Parse the response
                parsed = parse_review_response(gpt_response)

                # Record the review
                review_result = ReviewResult(
                    verdict=parsed["verdict"],
                    hypothesis_assessment=parsed.get("hypothesis_assessment", ""),
                    gaps=parsed["gaps"],
                    feedback=parsed["feedback"],
                    raw_response=gpt_response,
                    iteration=pi.iterations,
                )
                pi.review_history.append(review_result)

                # Add GPT review summary to transcript summaries
                if pi.current_investigation:
                    gpt_review_iteration = sum(1 for s in pi.current_investigation.transcript_summaries if s.get("agent") == "gpt_reviewer") + 1
                    gpt_summary = summarize_gpt_review(
                        review_content=parsed,
                        iteration=gpt_review_iteration,
                    )
                    pi.current_investigation.transcript_summaries.append(gpt_summary)
                    print(f"[PI] GPT review summary: {gpt_summary.get('verdict', '')} - {gpt_summary.get('summary', '')[:80]}...")

                # Track gaps in revision history
                if parsed["gaps"] and pi.current_investigation:
                    # Get previously unaddressed gaps
                    prior_gaps = []
                    if pi.current_investigation.revision_history:
                        last_revision = pi.current_investigation.revision_history[-1]
                        prior_gaps = last_revision.get("gaps_remaining", [])

                    # New gaps from this review
                    new_gaps = parsed["gaps"]

                    # Record this review iteration
                    revision_entry = {
                        "iteration": pi.iterations,
                        "gaps_identified": new_gaps,
                        "prior_gaps_remaining": prior_gaps,
                        "gaps_addressed": [],  # Will be filled after revision runs
                        "gaps_remaining": new_gaps + [g for g in prior_gaps if g not in new_gaps],
                    }
                    pi.current_investigation.revision_history.append(revision_entry)

                # Build revision context if needed
                revision_context = ""
                if parsed["verdict"] == "REQUEST_CHANGES":
                    # Include both new gaps and any prior unaddressed gaps
                    all_gaps = parsed["gaps"]
                    if pi.current_investigation and pi.current_investigation.revision_history:
                        last_revision = pi.current_investigation.revision_history[-1]
                        prior_remaining = last_revision.get("prior_gaps_remaining", [])
                        # Add prior gaps that weren't already in new gaps
                        all_gaps = all_gaps + [g for g in prior_remaining if g not in all_gaps]

                    gaps_text = "\n".join(f"- {g}" for g in all_gaps)
                    revision_context = REVISION_CONTEXT_TEMPLATE.format(
                        verdict=parsed["verdict"],
                        confidence_assessment=parsed.get("hypothesis_assessment", ""),  # V4: hypothesis_assessment replaces confidence_assessment
                        feedback=parsed["feedback"],
                        gaps=gaps_text,
                    )

                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "success",
                    "verdict": parsed["verdict"],
                    "hypothesis_assessment": parsed.get("hypothesis_assessment", ""),
                    "gaps": parsed["gaps"],
                    "feedback": parsed["feedback"],
                    "revision_context": revision_context if parsed["verdict"] == "REQUEST_CHANGES" else None,
                    "iterations_so_far": pi.iterations,
                    "max_iterations": pi.max_review_iterations,
                }, indent=2)}]}

            except subprocess.TimeoutExpired:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "GPT review timed out after 180 seconds",
                }, indent=2)}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": f"GPT review failed: {str(e)}",
                }, indent=2)}]}
            finally:
                # Cleanup temp file
                import os
                try:
                    os.unlink(prompt_file)
                except:
                    pass

        @tool(
            "parse_gpt_review",
            "Parse GPT review response and record result.",
            {"review_response": str}
        )
        async def parse_gpt_review_tool(args):
            review_response = args.get("review_response", "")

            parsed = parse_review_response(review_response)

            review_result = ReviewResult(
                verdict=parsed["verdict"],
                hypothesis_assessment=parsed.get("hypothesis_assessment", ""),
                gaps=parsed["gaps"],
                feedback=parsed["feedback"],
                raw_response=parsed["raw_response"],
                iteration=pi.iterations,
            )
            pi.review_history.append(review_result)

            # If REQUEST_CHANGES, build revision context
            revision_context = ""
            if parsed["verdict"] == "REQUEST_CHANGES":
                gaps_text = "\n".join(f"- {g}" for g in parsed["gaps"])
                revision_context = REVISION_CONTEXT_TEMPLATE.format(
                    verdict=parsed["verdict"],
                    confidence_assessment=parsed.get("hypothesis_assessment", ""),  # Legacy template field
                    feedback=parsed["feedback"],
                    gaps=gaps_text,
                )

            result = {
                "verdict": parsed["verdict"],
                "hypothesis_assessment": parsed.get("hypothesis_assessment", ""),
                "gaps": parsed["gaps"],
                "feedback": parsed["feedback"],
                "revision_context": revision_context if parsed["verdict"] == "REQUEST_CHANGES" else None,
                "iterations_so_far": pi.iterations,
                "max_iterations": pi.max_review_iterations,
            }
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        @tool(
            "generate_dashboard",
            "Generate HTML dashboard from investigation results.",
            {}
        )
        async def generate_dashboard_tool(args):
            if pi.current_investigation is None:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "No investigation available. Run investigation first.",
                }, indent=2)}]}

            print("\n[PI] Generating dashboard...")

            # Get investigation JSON path (consolidated format)
            safe_id = neuron_id.replace("/", "_")
            polarity_suffix = "_negative" if pi.polarity_mode == "negative" else ""
            investigation_json_path = output_dir / f"{safe_id}{polarity_suffix}_investigation.json"

            if not investigation_json_path.exists():
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": f"Investigation JSON not found: {investigation_json_path}",
                }, indent=2)}]}

            try:
                # Run dashboard agent via subprocess - use configured html_output_dir
                html_output_dir = pi.html_output_dir
                cmd = [
                    sys.executable,
                    "scripts/generate_html_report.py",
                    str(investigation_json_path),
                    "--v2",
                    "--model", "opus",
                    "-o", str(html_output_dir),
                ]
                # Timeout after 10 minutes (opus can be slow but produces highest quality)
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), timeout=600)

                if result.returncode == 0:
                    html_path = html_output_dir / f"{safe_id}.html"
                    # Validate file was actually created
                    if html_path.exists():
                        return {"content": [{"type": "text", "text": json.dumps({
                            "status": "success",
                            "html_path": str(html_path),
                            "dashboard_generated": True,
                            "stdout": result.stdout[-500:] if result.stdout else "",
                        }, indent=2)}]}
                    else:
                        return {"content": [{"type": "text", "text": json.dumps({
                            "status": "error",
                            "html_path": None,
                            "dashboard_generated": False,
                            "dashboard_error": f"Dashboard script succeeded but file not created at {html_path}",
                            "stdout": result.stdout[-500:] if result.stdout else "",
                        }, indent=2)}]}
                else:
                    return {"content": [{"type": "text", "text": json.dumps({
                        "status": "error",
                        "error": result.stderr[-500:] if result.stderr else "Unknown error",
                    }, indent=2)}]}

            except subprocess.TimeoutExpired:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "Dashboard generation timed out after 10 minutes. The investigation is saved - you can generate the dashboard manually later.",
                }, indent=2)}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": str(e),
                }, indent=2)}]}

        @tool(
            "save_pi_result",
            "Save the final PI result with review history.",
            {"final_verdict": str, "dashboard_path": str}
        )
        async def save_pi_result_tool(args):
            final_verdict = args.get("final_verdict", "UNKNOWN")
            dashboard_path = args.get("dashboard_path", "")

            safe_id = neuron_id.replace("/", "_")
            polarity_suffix = "_negative" if pi.polarity_mode == "negative" else ""

            pi_result = PIResult(
                neuron_id=neuron_id,
                investigation=pi.current_investigation,
                skeptic_report=pi.current_skeptic_report,
                review_history=pi.review_history,
                iterations=pi.iterations,
                final_verdict=final_verdict,
                dashboard_path=dashboard_path,
                investigation_path=str(output_dir / f"{safe_id}{polarity_suffix}_investigation.json"),
                dashboard_json_path="",  # Deprecated - now consolidated in investigation.json
                timestamp=datetime.now().isoformat(),
            )

            # Save PI result (atomic)
            pi_result_path = output_dir / f"{safe_id}{polarity_suffix}_pi_result.json"
            _atomic_write_json(pi_result_path, pi_result.to_dict())

            print(f"\n[PI] Results saved to {pi_result_path}")

            return {"content": [{"type": "text", "text": json.dumps({
                "status": "success",
                "pi_result_path": str(pi_result_path),
                "final_verdict": final_verdict,
                "iterations": pi.iterations,
                "review_count": len(pi.review_history),
            }, indent=2)}]}

        @tool(
            "get_status",
            "Get current pipeline status.",
            {}
        )
        async def get_status_tool(args):
            return {"content": [{"type": "text", "text": json.dumps({
                "neuron_id": neuron_id,
                "iterations": pi.iterations,
                "max_iterations": pi.max_review_iterations,
                "has_investigation": pi.current_investigation is not None,
                "has_skeptic_report": pi.current_skeptic_report is not None,
                "skeptic_verdict": pi.current_skeptic_report.verdict if pi.current_skeptic_report else None,
                "review_count": len(pi.review_history),
                "last_verdict": pi.review_history[-1].verdict if pi.review_history else None,
                "skip_review": pi.skip_review,
                "skip_dashboard": pi.skip_dashboard,
            }, indent=2)}]}

        @tool(
            "load_existing_investigation",
            "Load an existing investigation JSON file. Use this in review-only mode.",
            {
                "path": {
                    "type": "string",
                    "description": "Path to the investigation JSON file",
                }
            }
        )
        async def load_existing_investigation_tool(args):
            path = args.get("path", "")
            if not path:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": "Path is required",
                }, indent=2)}]}

            try:
                investigation_path = Path(path)
                if not investigation_path.exists():
                    return {"content": [{"type": "text", "text": json.dumps({
                        "status": "error",
                        "error": f"File not found: {path}",
                    }, indent=2)}]}

                with open(investigation_path) as f:
                    data = json.load(f)

                # Use from_dict() to reconstruct NeuronInvestigation
                pi.current_investigation = NeuronInvestigation.from_dict(data)

                # Build per-hypothesis summary
                hypothesis_summary = []
                for h in pi.current_investigation.hypotheses_tested:
                    h_id = h.get("hypothesis_id", "?")
                    status = h.get("status", "pending")
                    posterior = h.get("posterior_probability", 50)
                    hypothesis_summary.append(f"{h_id}: {status} ({posterior}%)")

                print(f"[PI] Loaded existing investigation: {pi.current_investigation.neuron_id}")
                print(f"[PI]   Hypotheses: {len(pi.current_investigation.hypotheses_tested)}")
                print(f"[PI]   Experiments: {pi.current_investigation.total_experiments}")
                print(f"[PI]   Activating prompts: {len(pi.current_investigation.activating_prompts)}")
                print(f"[PI]   RelP results: {len(pi.current_investigation.relp_results)}")

                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "success",
                    "neuron_id": pi.current_investigation.neuron_id,
                    "hypotheses": hypothesis_summary,
                    "total_experiments": pi.current_investigation.total_experiments,
                    "activating_prompts": len(pi.current_investigation.activating_prompts),
                    "relp_results": len(pi.current_investigation.relp_results),
                    "input_function": pi.current_investigation.input_function[:200] + "..." if len(pi.current_investigation.input_function) > 200 else pi.current_investigation.input_function,
                    "key_findings_count": len(pi.current_investigation.key_findings),
                }, indent=2)}]}

            except Exception as e:
                return {"content": [{"type": "text", "text": json.dumps({
                    "status": "error",
                    "error": f"Failed to load investigation: {str(e)}",
                }, indent=2)}]}

        return [
            run_investigation_tool,
            run_investigation_with_revision_tool,
            run_skeptic_tool,
            request_gpt_review_tool,
            parse_gpt_review_tool,
            generate_dashboard_tool,
            save_pi_result_tool,
            get_status_tool,
            load_existing_investigation_tool,
        ]

    async def run(self) -> PIResult:
        """Run the full PI pipeline."""
        review_only = self.existing_investigation_path is not None

        print(f"\n{'='*60}")
        print(f"NeuronPI: Starting pipeline for {self.neuron_id}")
        if review_only:
            print("MODE: Review-only (using existing investigation)")
            print(f"Investigation: {self.existing_investigation_path}")
        print(f"{'='*60}\n")

        # Create MCP tools
        tools = self._create_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="pi_tools",
            version="1.0.0",
            tools=tools,
        )

        # Build initial prompt based on mode
        if review_only:
            initial_prompt = f"""Review and improve investigation for neuron {self.neuron_id}.

MODE: Review-only
- Existing investigation: {self.existing_investigation_path}
- Max review iterations: {self.max_review_iterations}
- Skip dashboard: {self.skip_dashboard}

WORKFLOW:
1. First, call `load_existing_investigation` with path "{self.existing_investigation_path}"
2. Call `request_gpt_review` to get peer review feedback
3. If GPT returns REQUEST_CHANGES, call `run_investigation_with_revision` with the revision context to address the gaps
4. Repeat review/revision cycle until APPROVE or max iterations
5. Generate dashboard when done

Start by loading the existing investigation."""
        else:
            initial_prompt = f"""Investigate neuron {self.neuron_id}.

Configuration:
- Max review iterations: {self.max_review_iterations}
- Skip GPT review: {self.skip_review}
- Skip dashboard: {self.skip_dashboard}
- Initial label: {self.initial_label or 'none'}
- Initial hypothesis: {self.initial_hypothesis or 'none'}

Start by running the initial investigation."""

        # Configure allowed tools
        allowed_tools = [
            "mcp__pi_tools__run_investigation",
            "mcp__pi_tools__run_investigation_with_revision",
            "mcp__pi_tools__run_skeptic",  # REQUIRED: adversarial testing
            "mcp__pi_tools__request_gpt_review",
            "mcp__pi_tools__parse_gpt_review",
            "mcp__pi_tools__generate_dashboard",
            "mcp__pi_tools__save_pi_result",
            "mcp__pi_tools__get_status",
            "mcp__pi_tools__load_existing_investigation",
        ]

        # Add Codex MCP if available (not needed, we call it directly now)
        # if self.codex_available and not self.skip_review:
        #     allowed_tools.append("mcp__codex__codex")

        # Configure agent options
        # Store transcripts in separate directory to avoid cluttering main project
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "neuron_reports" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        options = ClaudeAgentOptions(
            system_prompt=PI_SYSTEM_PROMPT,
            max_turns=30,  # Enough for multiple review iterations
            model=self.model,
            mcp_servers={"pi_tools": mcp_server},
            cwd=transcripts_dir,
            add_dirs=[project_root],  # Allow access to main project files
            allowed_tools=allowed_tools,
        )

        # Run agent
        print(f"Running PI agent (model: {self.model})...")

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                _pi_last_msg = time.time()
                _pi_msg_count = 0
                async for message in client.receive_response():
                    _pi_now = time.time()
                    _pi_gap = _pi_now - _pi_last_msg
                    _pi_last_msg = _pi_now
                    _pi_msg_count += 1

                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"\n[PI Agent]: {block.text}")
                            elif isinstance(block, ToolUseBlock):
                                print(f"\n[PI Tool]: {block.name}", flush=True)
                                if _pi_gap > 30:
                                    print(f"  [PI TIMING] {_pi_gap:.0f}s since last message (msg #{_pi_msg_count})", flush=True)

                    elif isinstance(message, ResultMessage):
                        print(f"\n[PI Result]: {message.subtype} (gap={_pi_gap:.0f}s, msg #{_pi_msg_count})", flush=True)

                print(f"[PI TIMING] Loop ended after {_pi_msg_count} messages", flush=True)

        except Exception as e:
            print(f"\n[PI Error]: {e}", flush=True)
            return PIResult(
                neuron_id=self.neuron_id,
                investigation=self.current_investigation,
                review_history=self.review_history,
                iterations=self.iterations,
                final_verdict="ERROR",
                error=str(e),
                timestamp=datetime.now().isoformat(),
            )

        # Return final result
        final_verdict = "APPROVE" if self.review_history and self.review_history[-1].verdict == "APPROVE" else "MAX_ITERATIONS"
        if self.skip_review:
            final_verdict = "SKIPPED_REVIEW"

        return PIResult(
            neuron_id=self.neuron_id,
            investigation=self.current_investigation,
            review_history=self.review_history,
            iterations=self.iterations,
            final_verdict=final_verdict,
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# Convenience Function
# =============================================================================

async def run_neuron_pi(
    neuron_id: str,
    initial_label: str = "",
    initial_hypothesis: str = "",
    edge_stats_path: Path | None = None,
    labels_path: Path | None = None,
    output_dir: Path = Path("neuron_reports/json"),
    html_output_dir: Path | None = None,
    model: str = "sonnet",
    scientist_model: str = "opus",
    max_review_iterations: int = 3,
    skip_review: bool = False,
    skip_dashboard: bool = False,
    existing_investigation_path: Path | None = None,
    polarity_mode: str = "positive",
    gpu_server_url: str | None = None,
) -> PIResult:
    """Convenience function to run the full NeuronPI pipeline.

    Args:
        neuron_id: Target neuron (e.g., "L15/N7890")
        initial_label: Initial label from batch labeling
        initial_hypothesis: Starting hypothesis
        edge_stats_path: Path to edge statistics
        labels_path: Path to neuron labels JSON
        output_dir: Directory for outputs
        html_output_dir: Directory for HTML dashboards (defaults to neuron_reports/html)
        model: Model for PI orchestration
        scientist_model: Model for NeuronScientist
        max_review_iterations: Max review cycles
        skip_review: Skip GPT review
        skip_dashboard: Skip dashboard generation
        existing_investigation_path: Path to existing investigation (review-only mode)
        polarity_mode: "positive" or "negative" - which firing direction to investigate
        gpu_server_url: URL for GPU inference server (e.g., "http://localhost:8477")

    Returns:
        PIResult with full pipeline results
    """
    # Set up GPU client once for the entire PI pipeline
    gpu_client = None
    if gpu_server_url:
        from neuron_scientist.gpu_client import GPUClient
        from neuron_scientist.tools import set_gpu_client
        parts = neuron_id.split("/")
        layer = int(parts[0][1:])
        neuron_idx = int(parts[1][1:])
        gpu_client = GPUClient(gpu_server_url, agent_id=f"pi-L{layer}-N{neuron_idx}")
        await gpu_client.wait_for_server()
        set_gpu_client(gpu_client)

    try:
        pi = NeuronPI(
            neuron_id=neuron_id,
            initial_label=initial_label,
            initial_hypothesis=initial_hypothesis,
            edge_stats_path=edge_stats_path,
            labels_path=labels_path,
            output_dir=output_dir,
            html_output_dir=html_output_dir,
            model=model,
            scientist_model=scientist_model,
            max_review_iterations=max_review_iterations,
            skip_review=skip_review,
            skip_dashboard=skip_dashboard,
            existing_investigation_path=existing_investigation_path,
            polarity_mode=polarity_mode,
            gpu_server_url=gpu_server_url,
        )

        return await pi.run()
    finally:
        # Clean up GPU client
        if gpu_client:
            from neuron_scientist.tools import set_gpu_client
            set_gpu_client(None)
            await gpu_client.close()
