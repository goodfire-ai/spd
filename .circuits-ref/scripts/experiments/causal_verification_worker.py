#!/usr/bin/env python3
"""
Causal Verification Worker

Processes a slice of circuits through the full verification pipeline:
1. Load circuit and cluster it (infomap)
2. Send to LLM for analysis and experiment proposals
3. Run proposed experiments (batched)
4. Send results back to LLM for verification
5. Iterate if needed until confident
6. Output final verified analysis

Usage:
    # Process a single circuit
    python causal_verification_worker.py circuit.json --output results/

    # Process multiple circuits (slice)
    python causal_verification_worker.py circuits/*.json --output results/

    # With specific LLM provider
    python causal_verification_worker.py circuit.json --llm-provider anthropic --llm-model claude-sonnet-4-20250514
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompts.circuit_analysis import (
    CIRCUIT_ANALYSIS_PROMPT,
    CIRCUIT_ANALYSIS_SYSTEM,
    VERIFICATION_PROMPT,
    VERIFICATION_SYSTEM,
    extract_modules_for_prompt,
    format_experiment_results,
    format_modules_description,
    format_original_hypotheses,
)
from schemas.experiments import BatchExperimentInput


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 32768
) -> str:
    """Call LLM API and return response text."""

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        # Use streaming to handle long requests
        response_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text
        return response_text

    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        # Newer models (gpt-5.x, o1, etc.) use max_completion_tokens
        # Older models (gpt-4o, gpt-4, etc.) use max_tokens
        use_new_api = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
        token_param = "max_completion_tokens" if use_new_api else "max_tokens"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **{token_param: max_tokens}
        )
        # Debug: print response structure for newer models
        if use_new_api:
            print(f"[DEBUG] Response finish_reason: {response.choices[0].finish_reason}", file=sys.stderr)
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"[DEBUG] Refusal: {response.choices[0].message.refusal}", file=sys.stderr)
        content = response.choices[0].message.content
        if not content and hasattr(response.choices[0].message, 'reasoning_content'):
            # Some models put reasoning in a separate field
            content = response.choices[0].message.reasoning_content
        return content or ""

    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response (may be wrapped in markdown code blocks)."""
    import re

    if not text:
        raise ValueError("Empty response from LLM")

    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Assume the whole response is JSON
            json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON. Response was:\n{text[:1000]}...", file=sys.stderr)
        raise


def create_verified_analysis(
    graph_file: Path,
    clusters_file: Path,
    initial_analysis: dict,
    final_analysis: dict,
    final_verification: dict,
    all_results: list[dict],
    output_dir: Path,
    circuit_id: str
) -> Path:
    """
    Create a flow-viewer-compatible analysis file with verified hypotheses.

    This merges the causal verification results back into the original analysis
    format so it can be viewed in the flow_viewer.

    Args:
        initial_analysis: The INITIAL LLM analysis (with all modules)
        final_analysis: The FINAL iteration's analysis (may have fewer modules)
        final_verification: The final verification response
        all_results: All experiment results across iterations
    """
    # Try to find and load the original flow-viewer analysis file
    original_flow_analysis = None
    analysis_patterns = [
        graph_file.with_name(graph_file.name.replace("-graph.json", "-analysis.json")),
        graph_file.with_name(graph_file.stem + "-analysis.json"),
        graph_file.parent / f"{circuit_id}-analysis.json",
    ]

    for pattern in analysis_patterns:
        if pattern.exists():
            with open(pattern) as f:
                original_flow_analysis = json.load(f)
            # Check if this is a flow-viewer format (has module_summaries) vs raw LLM format
            if "module_summaries" in original_flow_analysis:
                break
            else:
                original_flow_analysis = None

    # If no original flow analysis, create a minimal structure
    if original_flow_analysis is None:
        with open(clusters_file) as f:
            clusters_data = json.load(f)
        with open(graph_file) as f:
            graph_data = json.load(f)

        method_data = None
        for m in clusters_data.get("methods", []):
            if m.get("method") == "infomap":
                method_data = m
                break

        modules = method_data.get("clusters", []) if method_data else []

        original_flow_analysis = {
            "method": "infomap",
            "n_modules": len(modules),
            "prompt": graph_data.get("metadata", {}).get("prompt", ""),
            "module_summaries": [
                {"cluster_id": c["cluster_id"], "size": len(c["members"])}
                for c in modules
            ],
            "llm_synthesis": "",
            "flow_matrix": [],
            "edge_count_matrix": [],
        }

    # Deep copy to avoid modifying original
    import copy
    verified_analysis = copy.deepcopy(original_flow_analysis)

    # Build lookup of ALL hypotheses from initial analysis (has all modules)
    all_hypotheses = {}
    for hyp in initial_analysis.get("module_analyses", []):
        mod_id = hyp.get("module_id")
        if mod_id is not None:
            all_hypotheses[mod_id] = hyp

    # Build lookup of FINAL/REVISED hypotheses (may only have subset of modules)
    revised_hypotheses = {}
    for hyp in final_analysis.get("module_analyses", []):
        mod_id = hyp.get("module_id")
        if mod_id is not None:
            revised_hypotheses[mod_id] = hyp

    # Build experiment results by module
    # Note: experiments may target parent module IDs (e.g., 9) while summaries
    # have submodule IDs (e.g., '9.0', '9.1'). We need to match both.
    results_by_module = {}
    for result in all_results:
        mod_id = result.get("module_id")
        if mod_id is not None:
            # Store under both the original ID and string version
            for key in [mod_id, str(mod_id)]:
                if key not in results_by_module:
                    results_by_module[key] = []
                if result not in results_by_module[key]:
                    results_by_module[key].append(result)

    def get_module_results(cluster_id):
        """Get results for a module, handling submodule IDs like '9.0' -> 9."""
        # Try exact match first
        results = results_by_module.get(cluster_id, [])
        if results:
            return results
        # Try string version
        results = results_by_module.get(str(cluster_id), [])
        if results:
            return results
        # For submodule IDs like '9.0', try parent ID
        if isinstance(cluster_id, str) and '.' in cluster_id:
            parent_id = int(cluster_id.split('.')[0])
            results = results_by_module.get(parent_id, [])
            if results:
                return results
            results = results_by_module.get(str(parent_id), [])
        return results

    # Update module_summaries with hypotheses and verification status
    if "module_summaries" in verified_analysis:
        for ms in verified_analysis["module_summaries"]:
            mod_id = ms.get("cluster_id")

            # Get initial hypothesis (try both exact and parent ID)
            initial_hyp = all_hypotheses.get(mod_id, {})
            if not initial_hyp and isinstance(mod_id, str) and '.' in mod_id:
                parent_id = int(mod_id.split('.')[0])
                initial_hyp = all_hypotheses.get(parent_id, {})

            revised_hyp = revised_hypotheses.get(mod_id, {})
            if not revised_hyp and isinstance(mod_id, str) and '.' in mod_id:
                parent_id = int(mod_id.split('.')[0])
                revised_hyp = revised_hypotheses.get(parent_id, {})

            module_results = get_module_results(mod_id)

            # Determine if this module was tested and if hypothesis was revised
            was_tested = len(module_results) > 0
            was_revised = mod_id in revised_hypotheses

            # Use revised hypothesis if available, otherwise initial
            final_hyp = revised_hyp if was_revised else initial_hyp

            if final_hyp:
                # Update the visible name/function with verified info
                if was_revised and revised_hyp.get("hypothesis"):
                    # Mark as verified in the name
                    original_name = ms.get("name", f"Module {mod_id}")
                    if not original_name.startswith("✓"):
                        ms["name"] = f"✓ {original_name}"
                    # Update function with revised hypothesis
                    ms["function"] = revised_hyp.get("hypothesis", ms.get("function", ""))

                # Store verification metadata
                ms["verified"] = was_tested
                ms["revised"] = was_revised
                ms["initial_hypothesis"] = initial_hyp.get("hypothesis", "")
                ms["verified_hypothesis"] = final_hyp.get("hypothesis", "")
                ms["verified_evidence"] = final_hyp.get("evidence", "")
                ms["verified_importance"] = final_hyp.get("importance", 0)
                ms["experiment_count"] = len(module_results)

    # Build comprehensive LLM synthesis narrative
    original_synthesis = original_flow_analysis.get("llm_synthesis", "")
    verified_narrative_parts = []

    # Header with verification status
    verified_narrative_parts.append("# Causally Verified Circuit Analysis\n\n")

    confidence = final_verification.get("confidence_level", 0) if final_verification else 0
    total_experiments = len(all_results)
    verified_narrative_parts.append(f"**Verification Status:** {total_experiments} experiments run\n")
    verified_narrative_parts.append(f"**Final Confidence:** {confidence:.0%}\n\n")

    # Overall summary from verification
    verified_narrative_parts.append("## Verification Summary\n\n")
    verified_narrative_parts.append(final_analysis.get("summary", "No summary available."))
    verified_narrative_parts.append("\n\n")

    # Divider
    verified_narrative_parts.append("---\n\n")

    # Keep original module descriptions, annotated with verification results
    verified_narrative_parts.append("## Module Descriptions\n\n")

    def get_hypothesis(hypotheses: dict, mod_id):
        """Get hypothesis for a module, handling submodule IDs like '9.0' -> 9."""
        hyp = hypotheses.get(mod_id, {})
        if hyp:
            return hyp
        hyp = hypotheses.get(str(mod_id), {})
        if hyp:
            return hyp
        if isinstance(mod_id, str) and '.' in mod_id:
            parent_id = int(mod_id.split('.')[0])
            hyp = hypotheses.get(parent_id, {})
            if hyp:
                return hyp
            hyp = hypotheses.get(str(parent_id), {})
        return hyp

    for ms in verified_analysis.get("module_summaries", []):
        mod_id = ms.get("cluster_id")
        name = ms.get("name", f"Module {mod_id}")
        function = ms.get("function", "No description")

        initial_hyp = get_hypothesis(all_hypotheses, mod_id)
        revised_hyp = get_hypothesis(revised_hypotheses, mod_id)
        module_results = get_module_results(mod_id)

        verified_narrative_parts.append(f"### {name}\n")
        verified_narrative_parts.append(f"**Function:** {function}\n\n")

        # Show initial hypothesis from LLM
        if initial_hyp.get("hypothesis"):
            verified_narrative_parts.append(f"**Initial Hypothesis:** {initial_hyp['hypothesis']}\n\n")

        # Show experiment results for this module with detailed interpretation
        if module_results:
            verified_narrative_parts.append("**Causal Experiments:**\n\n")
            for r in module_results:
                if r.get("error") or r.get("skipped"):
                    verified_narrative_parts.append(f"- ⚠️ {r.get('experiment_id', 'Unknown')}: Skipped ({r.get('validation_error', 'error')})\n")
                    continue

                exp_id = r.get("experiment_id", "exp")
                exp_type = r.get("experiment_type", "unknown")
                baseline_top = r.get("baseline_top_token", "?")
                result_top = r.get("result_top_token", "?")
                target = r.get("target_token", baseline_top)  # Use baseline top if no explicit target
                delta = r.get("logprob_delta", 0)
                baseline_prob = r.get("baseline_top_prob", 0)
                result_prob = r.get("result_top_prob", 0)
                changed = r.get("top_token_changed", False)
                hypothesis = r.get("hypothesis", "")
                steer_scale = r.get("steer_scale")

                # Build experiment description
                if exp_type == "zero_ablate":
                    exp_desc = "**Zero ablation** (set module activations to 0)"
                elif exp_type == "mean_ablate":
                    exp_desc = "**Mean ablation** (replace with dataset mean)"
                elif exp_type == "steer":
                    scale_str = f"×{steer_scale}" if steer_scale else ""
                    exp_desc = f"**Steering{scale_str}** (scale module activations)"
                elif exp_type == "patch":
                    source_prompt = r.get("source_prompt", "")
                    exp_desc = "**Activation patching**"
                else:
                    exp_desc = f"**{exp_type}**"

                verified_narrative_parts.append(f"- {exp_desc}\n")

                # For patch experiments, show full source prompt as counterfactual
                if exp_type == "patch" and r.get("source_prompt"):
                    verified_narrative_parts.append(f"  - *Counterfactual prompt:* \"{r['source_prompt']}\"\n")

                # Show hypothesis if available
                if hypothesis:
                    verified_narrative_parts.append(f"  - *Hypothesis:* {hypothesis[:150]}{'...' if len(hypothesis) > 150 else ''}\n")

                # Show results
                prob_change = result_prob - baseline_prob
                if abs(delta) < 0.01:
                    effect = "negligible effect"
                elif abs(delta) < 0.1:
                    effect = "small effect"
                elif abs(delta) < 0.5:
                    effect = "moderate effect"
                else:
                    effect = "**large effect**"

                direction = "increased" if delta > 0 else "decreased"
                verified_narrative_parts.append(
                    f"  - *Result:* P({repr(target)}) {direction} by {abs(delta):.3f} logprobs ({effect})\n"
                )

                if baseline_prob > 0 and result_prob > 0:
                    verified_narrative_parts.append(
                        f"  - *Probability:* {baseline_prob:.1%} → {result_prob:.1%}\n"
                    )

                if changed:
                    verified_narrative_parts.append(
                        f"  - ⚡ **Top token changed:** {repr(baseline_top)} → {repr(result_top)}\n"
                    )

                # Show top-k distribution changes
                baseline_top_k = r.get("baseline_top_k", [])
                result_top_k = r.get("result_top_k", [])
                if baseline_top_k and result_top_k:
                    verified_narrative_parts.append("  - *Distribution shift:*\n")

                    # Build lookup for result probabilities
                    result_probs = {e["token"]: e["prob"] for e in result_top_k}
                    baseline_probs = {e["token"]: e["prob"] for e in baseline_top_k}

                    # Find biggest gainers and losers
                    all_tokens = set(baseline_probs.keys()) | set(result_probs.keys())
                    changes = []
                    for tok in all_tokens:
                        b_prob = baseline_probs.get(tok, 0)
                        r_prob = result_probs.get(tok, 0)
                        delta_prob = r_prob - b_prob
                        if abs(delta_prob) > 0.01:  # Only show meaningful changes
                            changes.append((tok, b_prob, r_prob, delta_prob))

                    # Sort by absolute change
                    changes.sort(key=lambda x: -abs(x[3]))

                    # Show top 5 changes
                    for tok, b_prob, r_prob, delta_prob in changes[:5]:
                        arrow = "↑" if delta_prob > 0 else "↓"
                        verified_narrative_parts.append(
                            f"    - {repr(tok)}: {b_prob:.1%} → {r_prob:.1%} ({arrow}{abs(delta_prob):.1%})\n"
                        )

                # Interpretation
                if exp_type == "zero_ablate":
                    if delta < -0.3:
                        interp = "Module is **causally important** for this prediction"
                    elif delta > 0.1:
                        interp = "Module was **suppressing** this token"
                    else:
                        interp = "Module has minimal causal effect"
                elif exp_type == "steer" and steer_scale:
                    if steer_scale > 1 and delta > 0.1:
                        interp = "Amplifying module **increases** target probability"
                    elif steer_scale > 1 and delta < -0.1:
                        interp = "Amplifying module **decreases** target (non-monotonic)"
                    elif steer_scale < 1 and delta < -0.1:
                        interp = "Reducing module **decreases** target probability"
                    else:
                        interp = "Steering has limited effect"
                else:
                    interp = None

                if interp:
                    verified_narrative_parts.append(f"  - *Interpretation:* {interp}\n")

                verified_narrative_parts.append("\n")

        # Show revised hypothesis if different
        if revised_hyp.get("hypothesis") and revised_hyp.get("hypothesis") != initial_hyp.get("hypothesis"):
            verified_narrative_parts.append(f"**Revised Hypothesis:** {revised_hyp['hypothesis']}\n")
            if revised_hyp.get("evidence"):
                verified_narrative_parts.append(f"**Evidence:** {revised_hyp['evidence']}\n")
            verified_narrative_parts.append("\n")

    # Divider
    verified_narrative_parts.append("---\n\n")

    # Prediction accuracy summary
    if final_verification and final_verification.get("prediction_scores"):
        scores = final_verification["prediction_scores"]
        correct = sum(1 for s in scores if s.get("overall_correct"))
        total = len(scores)
        verified_narrative_parts.append("## Prediction Accuracy\n\n")
        verified_narrative_parts.append(f"**Correct:** {correct}/{total} ({100*correct/total:.1f}%)\n\n")

    # Original analysis for reference
    if original_synthesis:
        verified_narrative_parts.append("---\n\n")
        verified_narrative_parts.append("## Original Analysis (Pre-Verification)\n\n")
        verified_narrative_parts.append(original_synthesis)

    verified_analysis["llm_synthesis"] = "".join(verified_narrative_parts)

    # Add verification metadata
    verified_analysis["verification"] = {
        "verified": True,
        "total_experiments": len(all_results),
        "confidence": confidence,
        "experiment_results": all_results,
        "initial_hypotheses": initial_analysis.get("module_analyses", []),
        "final_hypotheses": final_analysis.get("module_analyses", []),
    }

    # Save verified analysis
    verified_file = output_dir / f"{circuit_id}-analysis-verified.json"
    with open(verified_file, "w") as f:
        json.dump(verified_analysis, f, indent=2)

    return verified_file


def load_graph_and_clusters(graph_file: Path, clusters_file: Path) -> tuple[dict, dict]:
    """Load graph and clusters data."""
    with open(graph_file) as f:
        graph_data = json.load(f)

    with open(clusters_file) as f:
        clusters_data = json.load(f)

    return graph_data, clusters_data


def get_top_logits(graph_data: dict) -> list[dict]:
    """Extract top logit information from graph."""
    logits = []
    for node in graph_data.get("nodes", []):
        if node.get("isLogit") or node.get("node_id", "").startswith("L_"):
            clerp = node.get("clerp", "")
            # Parse token and prob from clerp like " yes (p=0.4668)"
            import re
            match = re.match(r'(.+?) \(p=([\d.]+)\)', clerp)
            if match:
                token, prob = match.groups()
                logits.append({"token": token, "prob": float(prob)})

    logits.sort(key=lambda x: -x["prob"])
    return logits[:10]


def extract_answer_prefix(graph_data: dict) -> str:
    """Extract the answer_prefix from graph metadata's prompt_tokens.

    The answer_prefix is any text after the assistant header that was used
    during graph generation. This is needed to ensure experiments use the
    same prompt format.
    """
    metadata = graph_data.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", [])

    if not prompt_tokens:
        return ""

    # Find assistant header sequence
    try:
        # Look for pattern: <|start_header_id|>, assistant, <|end_header_id|>, \n\n
        for i, token in enumerate(prompt_tokens):
            if token == "assistant" and i > 0 and prompt_tokens[i-1] == "<|start_header_id|>":
                # Found assistant header, skip to after <|end_header_id|> and newlines
                if i + 2 < len(prompt_tokens) and prompt_tokens[i+1] == "<|end_header_id|>":
                    # Token after header end and newline is start of prefix
                    # \u010a\u010a is the newline representation
                    prefix_start = i + 3  # Skip header_end and newline
                    if prefix_start < len(prompt_tokens):
                        # Collect remaining tokens as answer_prefix
                        # Convert special chars: \u0120 = leading space
                        prefix_tokens = prompt_tokens[prefix_start:]
                        prefix = ""
                        for tok in prefix_tokens:
                            if tok.startswith("\u0120"):
                                prefix += " " + tok[1:]
                            elif tok == "\u010a":
                                prefix += "\n"
                            else:
                                prefix += tok
                        return prefix if prefix else ""
    except (IndexError, ValueError):
        pass

    return ""


def validate_experiments(experiments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate experiments and return (valid, invalid) lists.

    Checks:
    - patch experiments have source_prompt
    - steer experiments have steer_scale
    """
    valid = []
    invalid = []

    for exp in experiments:
        exp_type = exp.get("experiment_type", "")
        exp_id = exp.get("experiment_id", "unknown")

        if exp_type == "patch" and not exp.get("source_prompt"):
            invalid.append({
                **exp,
                "validation_error": "patch experiment requires source_prompt"
            })
        elif exp_type == "steer" and exp.get("steer_scale") is None:
            # Default steer_scale to 2.0 if not provided
            exp["steer_scale"] = 2.0
            valid.append(exp)
        else:
            valid.append(exp)

    return valid, invalid


def run_experiments(
    experiments: list[dict],
    clusters_file: Path,
    circuit_id: str,
    cluster_method: str = "infomap",
    answer_prefix: str = ""
) -> list[dict]:
    """Run experiments using the batched experiment runner.

    Args:
        experiments: List of experiment specs
        clusters_file: Path to clusters JSON file
        circuit_id: Circuit identifier
        cluster_method: Clustering method used
        answer_prefix: Answer prefix to inject into experiments (e.g., " Answer:")
    """
    from scripts.batched_experiments import BatchedExperimentRunner

    # Inject answer_prefix into all experiments if not already set
    if answer_prefix:
        for exp in experiments:
            if not exp.get("answer_prefix"):
                exp["answer_prefix"] = answer_prefix

    # Validate experiments
    valid_experiments, invalid_experiments = validate_experiments(experiments)

    # Report invalid experiments
    results = []
    for inv in invalid_experiments:
        print(f"  SKIPPED {inv['experiment_id']}: {inv['validation_error']}", file=sys.stderr)
        results.append({
            "experiment_id": inv["experiment_id"],
            "circuit_id": inv.get("circuit_id", circuit_id),
            "module_id": inv.get("module_id", -1),
            "experiment_type": inv.get("experiment_type", "unknown"),
            "error": inv["validation_error"],
            "skipped": True
        })

    if not valid_experiments:
        print("  No valid experiments to run", file=sys.stderr)
        return results

    batch_input = BatchExperimentInput(
        circuit_id=circuit_id,
        clusters_file=str(clusters_file),
        experiments=valid_experiments
    )

    runner = BatchedExperimentRunner()
    output = runner.run_batch(
        batch_input,
        clusters_file,
        cluster_method=cluster_method,
        verbose=True
    )

    # Combine skipped and executed results
    results.extend(output.results)
    return results


def run_screening(
    modules: list[dict],
    clusters_file: Path,
    circuit_id: str,
    prompt: str,
    answer_prefix: str = "",
    cluster_method: str = "infomap",
    effect_threshold: float = 0.1
) -> dict:
    """Run quick zero-ablation screening on all modules to identify causally important ones.

    Args:
        modules: List of module dicts from extract_modules_for_prompt
        clusters_file: Path to clusters file
        circuit_id: Circuit identifier
        prompt: The prompt to test
        answer_prefix: Answer prefix for the prompt
        cluster_method: Clustering method
        effect_threshold: Minimum |logprob_delta| to consider significant

    Returns:
        Dict with:
        - screening_results: List of {module_id, delta, baseline_top, result_top, significant}
        - significant_modules: List of module IDs with significant effects
        - summary: Human-readable summary
    """
    from scripts.batched_experiments import BatchedExperimentRunner

    print(f"\n[Screening] Running zero-ablation on {len(modules)} modules...")

    # Create zero-ablation experiments for all modules
    experiments = []
    for mod in modules:
        mod_id = mod["module_id"]
        experiments.append({
            "experiment_id": f"screen_mod{mod_id}_zero",
            "circuit_id": circuit_id,
            "module_id": mod_id,
            "experiment_type": "zero_ablate",
            "target_prompt": prompt,
            "hypothesis": "Screening ablation",
            "target_token": "",  # Will use top token
            "expected_direction": "unknown",
            "expected_magnitude": "unknown",
            "confidence": 0.5,
            "priority": 1,
            "answer_prefix": answer_prefix
        })

    # Run screening experiments
    batch_input = BatchExperimentInput(
        circuit_id=circuit_id,
        clusters_file=str(clusters_file),
        experiments=experiments
    )

    runner = BatchedExperimentRunner()
    output = runner.run_batch(
        batch_input,
        clusters_file,
        cluster_method=cluster_method,
        verbose=False  # Quiet during screening
    )

    # Analyze results
    screening_results = []
    significant_modules = []

    for result in output.results:
        if "error" in result:
            continue

        mod_id = result["module_id"]
        delta = result.get("logprob_delta", 0)
        baseline_top = result.get("baseline_top_token", "?")
        result_top = result.get("result_top_token", "?")
        changed = result.get("top_token_changed", False)

        # Consider significant if delta is large OR top token changed
        significant = abs(delta) >= effect_threshold or changed

        screening_results.append({
            "module_id": mod_id,
            "delta": delta,
            "baseline_top": baseline_top,
            "result_top": result_top,
            "top_changed": changed,
            "significant": significant
        })

        if significant:
            significant_modules.append(mod_id)

    # Sort by absolute delta
    screening_results.sort(key=lambda x: -abs(x["delta"]))

    # Build summary
    summary_lines = ["**Screening Results (zero-ablation):**"]
    for sr in screening_results:
        marker = "⚡" if sr["significant"] else "  "
        change_note = f" → {repr(sr['result_top'])}" if sr["top_changed"] else ""
        summary_lines.append(
            f"{marker} Module {sr['module_id']}: Δ={sr['delta']:+.3f}{change_note}"
        )

    print(f"[Screening] Found {len(significant_modules)} significant modules: {significant_modules}")

    return {
        "screening_results": screening_results,
        "significant_modules": significant_modules,
        "summary": "\n".join(summary_lines)
    }


def process_circuit(
    graph_file: Path,
    clusters_file: Path,
    output_dir: Path,
    llm_provider: str = "anthropic",
    llm_model: str = "claude-sonnet-4-20250514",
    cluster_method: str = "infomap",
    max_iterations: int = 3,
    max_tokens: int = 32768,
    use_llm_split: bool = False,
    functional_split: bool = False
) -> dict:
    """
    Process a single circuit through the verification pipeline.

    Args:
        use_llm_split: Use LLM to split modules by competing alternatives
        functional_split: Apply position/layer/semantic splitting

    Returns final verification result.
    """
    circuit_id = graph_file.stem.replace("-graph", "")
    print(f"\n{'='*60}")
    print(f"Processing circuit: {circuit_id}")
    print(f"{'='*60}")

    # Load data
    graph_data, clusters_data = load_graph_and_clusters(graph_file, clusters_file)

    # Extract prompt from metadata
    prompt = graph_data.get("metadata", {}).get("prompt", "")
    if not prompt:
        prompt = clusters_data.get("prompt", "Unknown prompt")

    # Extract answer_prefix from graph metadata (e.g., " Answer:")
    answer_prefix = extract_answer_prefix(graph_data)
    if answer_prefix:
        print(f"  Answer prefix detected: {repr(answer_prefix)}")

    # Get top logits
    top_logits = get_top_logits(graph_data)
    top_logits_str = "\n".join([f"- {repr(l['token'])}: p={l['prob']:.4f}" for l in top_logits])

    # Extract modules for prompt
    modules = extract_modules_for_prompt(str(clusters_file), str(graph_file), cluster_method)

    # Optionally apply functional splitting
    if functional_split or use_llm_split:
        from circuits.analysis import apply_functional_split

        # Convert modules to format expected by apply_functional_split
        # Note: apply_functional_split expects 'top_neurons' and 'size' keys
        module_summaries = []
        for mod in modules:
            module_summaries.append({
                'cluster_id': mod['module_id'],
                'name': f"Module {mod['module_id']}",
                'top_neurons': mod['neurons'],  # apply_functional_split looks for 'top_neurons'
                'size': len(mod['neurons']),    # explicit size
                'edges_to_logits': mod.get('edges_to_logits', [])
            })

        print(f"  Applying functional split (llm_split={use_llm_split})...")
        # Extract token strings for LLM split context
        top_logit_tokens = [l['token'] for l in top_logits]
        split_modules, split_info = apply_functional_split(
            module_summaries,
            min_module_size=4,  # Lower threshold to split more modules
            use_position_split=functional_split,
            use_layer_split=functional_split,
            use_semantic_split=functional_split,
            use_llm_split=use_llm_split,
            llm_model=llm_model,
            llm_provider=llm_provider,
            top_logits=top_logit_tokens  # Provide output alternatives for competing alternatives splitting
        )

        # Convert back to modules format
        # Note: split modules use 'top_neurons' key from apply_functional_split
        modules = []
        for sm in split_modules:
            modules.append({
                'module_id': sm['cluster_id'],
                'neurons': sm.get('top_neurons', sm.get('neurons', [])),  # Handle both key names
                'edges_to_logits': sm.get('edges_to_logits', [])
            })

        n_original = split_info.get('original_n_modules', len(module_summaries))
        n_new = len(modules)
        if n_new != n_original:
            print(f"  Functional split: {n_original} modules → {n_new} modules")

            # Create updated clusters file with split modules for experiment runner
            import copy
            split_clusters_data = copy.deepcopy(clusters_data)

            # Update the infomap clusters with split modules
            for method_data in split_clusters_data.get('methods', []):
                if method_data.get('method') == cluster_method:
                    new_clusters = []
                    for mod in modules:
                        # Convert neurons back to cluster member format
                        members = []
                        for n in mod['neurons']:
                            members.append({
                                'layer': n.get('layer'),
                                'neuron': n.get('neuron'),
                                'position': n.get('position'),
                                'label': n.get('label', '')
                            })
                        new_clusters.append({
                            'cluster_id': mod['module_id'],
                            'members': members
                        })
                    method_data['clusters'] = new_clusters
                    break

            # Save to output directory
            split_clusters_file = output_dir / f"{circuit_id}-clusters-split.json"
            with open(split_clusters_file, 'w') as f:
                json.dump(split_clusters_data, f, indent=2)
            print(f"  Split clusters saved to: {split_clusters_file}")
            clusters_file = split_clusters_file  # Use split clusters for experiments

    modules_desc = format_modules_description(modules, graph_data)

    # --- Step 0: Screening (quick zero-ablation on all modules) ---
    screening = run_screening(
        modules=modules,
        clusters_file=clusters_file,
        circuit_id=circuit_id,
        prompt=prompt,
        answer_prefix=answer_prefix,
        cluster_method=cluster_method,
        effect_threshold=0.1
    )

    # Initialize conversation log
    conversation_log = {
        "circuit_id": circuit_id,
        "prompt": prompt,
        "answer_prefix": answer_prefix,
        "top_logits": top_logits,
        "num_modules": len(modules),
        "screening": screening,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "timestamp_start": datetime.now().isoformat(),
        "initial_analysis": None,
        "iterations": []
    }

    # --- Step 1: Initial LLM Analysis ---
    print("\n[1/4] Requesting LLM analysis...")

    # Include screening results in prompt
    screening_info = screening["summary"] if screening["significant_modules"] else ""

    analysis_prompt = CIRCUIT_ANALYSIS_PROMPT.format(
        prompt=prompt,
        top_logits=top_logits_str,
        modules_description=modules_desc,
        circuit_id=circuit_id,
        num_modules=len(modules),
        screening_results=screening_info
    )

    analysis_response_raw = call_llm(
        CIRCUIT_ANALYSIS_SYSTEM,
        analysis_prompt,
        provider=llm_provider,
        model=llm_model,
        max_tokens=max_tokens
    )

    analysis = extract_json_from_response(analysis_response_raw)
    print(f"  - Summary: {analysis.get('summary', 'N/A')[:100]}...")
    print(f"  - Modules analyzed: {len(analysis.get('module_analyses', []))}")
    print(f"  - Experiments proposed: {len(analysis.get('proposed_experiments', []))}")

    # Log initial analysis
    conversation_log["initial_analysis"] = {
        "system_prompt": CIRCUIT_ANALYSIS_SYSTEM,
        "user_prompt": analysis_prompt,
        "llm_response_raw": analysis_response_raw,
        "llm_response_parsed": analysis,
        "timestamp": datetime.now().isoformat()
    }

    # Save initial analysis
    analysis_file = output_dir / f"{circuit_id}-analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    # --- Iteration Loop ---
    iteration = 0
    initial_analysis = analysis  # Keep reference to full initial analysis
    current_analysis = analysis
    all_results = []

    while iteration < max_iterations:
        iteration += 1
        experiments = current_analysis.get("proposed_experiments", [])

        # Initialize iteration log
        iteration_log = {
            "iteration": iteration,
            "experiments_proposed": experiments,
            "experiment_results": None,
            "verification": None,
            "timestamp_start": datetime.now().isoformat()
        }

        if not experiments:
            print(f"\n[Iteration {iteration}] No experiments to run, stopping.")
            iteration_log["status"] = "no_experiments"
            conversation_log["iterations"].append(iteration_log)
            break

        # --- Step 2: Run Experiments ---
        print(f"\n[Iteration {iteration}] Running {len(experiments)} experiments...")

        results = run_experiments(
            experiments,
            clusters_file,
            circuit_id,
            cluster_method,
            answer_prefix=answer_prefix
        )
        all_results.extend(results)
        iteration_log["experiment_results"] = results

        # Save experiment results
        results_file = output_dir / f"{circuit_id}-results-iter{iteration}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # --- Step 3: LLM Verification ---
        print(f"\n[Iteration {iteration}] Requesting LLM verification...")

        results_str = format_experiment_results(results)
        hypotheses_str = format_original_hypotheses(current_analysis)

        verification_prompt = VERIFICATION_PROMPT.format(
            circuit_id=circuit_id,
            prompt=prompt,
            original_hypotheses=hypotheses_str,
            experiment_results=results_str
        )

        verification_response_raw = call_llm(
            VERIFICATION_SYSTEM,
            verification_prompt,
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens
        )

        verification = extract_json_from_response(verification_response_raw)

        # Log verification
        iteration_log["verification"] = {
            "system_prompt": VERIFICATION_SYSTEM,
            "user_prompt": verification_prompt,
            "llm_response_raw": verification_response_raw,
            "llm_response_parsed": verification,
            "timestamp": datetime.now().isoformat()
        }

        # Score predictions
        scores = verification.get("prediction_scores", [])
        correct = sum(1 for s in scores if s.get("overall_correct"))
        print(f"  - Predictions correct: {correct}/{len(scores)}")
        print(f"  - Confidence: {verification.get('confidence_level', 0):.2f}")
        print(f"  - Done: {verification.get('done', False)}")

        iteration_log["predictions_correct"] = correct
        iteration_log["predictions_total"] = len(scores)
        iteration_log["confidence"] = verification.get('confidence_level', 0)
        iteration_log["timestamp_end"] = datetime.now().isoformat()

        # Save verification
        verif_file = output_dir / f"{circuit_id}-verification-iter{iteration}.json"
        with open(verif_file, 'w') as f:
            json.dump(verification, f, indent=2)

        # Add iteration to log
        conversation_log["iterations"].append(iteration_log)

        # Check if done
        if verification.get("done", False):
            print(f"\n[Iteration {iteration}] LLM confident, stopping iteration.")
            iteration_log["status"] = "done"
            break

        # Prepare for next iteration
        follow_ups = verification.get("follow_up_experiments", [])
        if not follow_ups:
            print(f"\n[Iteration {iteration}] No follow-up experiments, stopping.")
            iteration_log["status"] = "no_followups"
            break

        # Update analysis with follow-up experiments
        current_analysis = {
            "summary": verification.get("final_summary", ""),
            "module_analyses": verification.get("revised_hypotheses", []),
            "proposed_experiments": follow_ups
        }
        iteration_log["status"] = "continue"

        print(f"  - Follow-up experiments: {len(follow_ups)}")

    # --- Final Output ---
    conversation_log["timestamp_end"] = datetime.now().isoformat()
    conversation_log["total_iterations"] = iteration
    conversation_log["total_experiments"] = len(all_results)
    conversation_log["final_confidence"] = verification.get('confidence_level', 0) if 'verification' in dir() else None
    conversation_log["final_summary"] = verification.get('final_summary', '') if 'verification' in dir() else None

    # Save conversation log
    log_file = output_dir / f"{circuit_id}-conversation-log.json"
    with open(log_file, 'w') as f:
        json.dump(conversation_log, f, indent=2)

    final_result = {
        "circuit_id": circuit_id,
        "prompt": prompt,
        "iterations": iteration,
        "final_analysis": current_analysis,
        "final_verification": verification if 'verification' in dir() else None,
        "all_experiment_results": all_results,
        "timestamp": datetime.now().isoformat()
    }

    final_file = output_dir / f"{circuit_id}-final.json"
    with open(final_file, 'w') as f:
        json.dump(final_result, f, indent=2)

    # Create flow-viewer-compatible verified analysis
    verified_file = create_verified_analysis(
        graph_file=graph_file,
        clusters_file=clusters_file,
        initial_analysis=initial_analysis,
        final_analysis=current_analysis,
        final_verification=verification if 'verification' in dir() else None,
        all_results=all_results,
        output_dir=output_dir,
        circuit_id=circuit_id
    )

    print(f"\nConversation log saved to: {log_file}")
    print(f"Final results saved to: {final_file}")
    print(f"Verified analysis (flow_viewer compatible): {verified_file}")
    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Causal verification worker for attribution circuits",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graphs", nargs="+", help="Graph JSON files to process")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--llm-provider", default="anthropic",
                        choices=["anthropic", "openai"], help="LLM provider")
    parser.add_argument("--llm-model", default="claude-sonnet-4-20250514", help="LLM model name")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Maximum tokens for LLM responses")
    parser.add_argument("--cluster-method", default="infomap", help="Clustering method")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum verification iterations per circuit")
    parser.add_argument("--use-llm-split", action="store_true",
                        help="Use LLM to split modules by competing alternatives (e.g., dopamine vs serotonin neurons)")
    parser.add_argument("--functional-split", action="store_true",
                        help="Apply functional splitting (position, layer, semantic) to modules")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for graph_path in args.graphs:
        graph_file = Path(graph_path)

        # Find corresponding clusters file
        clusters_file = graph_file.with_name(
            graph_file.name.replace("-graph.json", "-clusters.json")
        )
        if not clusters_file.exists():
            # Try alternate naming
            clusters_file = graph_file.with_name(
                graph_file.stem + "-clusters.json"
            )

        if not clusters_file.exists():
            print(f"Warning: No clusters file found for {graph_file}, skipping")
            continue

        try:
            result = process_circuit(
                graph_file,
                clusters_file,
                output_dir,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                cluster_method=args.cluster_method,
                max_iterations=args.max_iterations,
                max_tokens=args.max_tokens,
                use_llm_split=args.use_llm_split,
                functional_split=args.functional_split
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_circuits": len(results),
            "circuits": [r["circuit_id"] for r in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Processed {len(results)} circuits")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
