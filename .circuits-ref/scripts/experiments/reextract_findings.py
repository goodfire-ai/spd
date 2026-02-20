#!/usr/bin/env python3
"""Re-extract structured findings from existing investigation files.

This script reads existing investigation JSON files and uses an LLM to
extract structured findings (input_function, output_function, key_findings, etc.)
from the agent reasoning, then regenerates the dashboard files.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from openai import OpenAI


def extract_structured_findings_regex(agent_reasoning: str) -> dict:
    """Extract structured findings using regex patterns (fallback when API unavailable)."""
    result = {
        "input_function": "",
        "output_function": "",
        "function_type": "",
        "final_hypothesis": "",
        "key_findings": [],
        "open_questions": [],
        "confidence_assessment": "",
        "summary": "",
    }

    text = agent_reasoning

    # Look for input function patterns
    input_patterns = [
        r"activates?\s+(?:when|on|for|in response to)\s+([^.]+\.)",
        r"input\s+function[:\s]+([^.]+\.)",
        r"neuron\s+activates?\s+([^.]+\.)",
        r"triggers?\s+(?:when|on)\s+([^.]+\.)",
    ]
    for pattern in input_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["input_function"] = match.group(1).strip()
            break

    # Look for output function patterns
    output_patterns = [
        r"(?:promotes?|boosts?|increases?)\s+(?:the\s+)?(?:probability\s+of\s+)?([^.]+\.)",
        r"output\s+function[:\s]+([^.]+\.)",
        r"(?:suppresses?|reduces?|decreases?)\s+([^.]+\.)",
    ]
    for pattern in output_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["output_function"] = match.group(1).strip()
            break

    # Look for function type
    if re.search(r"semantic", text, re.IGNORECASE):
        result["function_type"] = "semantic"
    elif re.search(r"syntactic", text, re.IGNORECASE):
        result["function_type"] = "syntactic"
    elif re.search(r"routing", text, re.IGNORECASE):
        result["function_type"] = "routing"
    elif re.search(r"formatting", text, re.IGNORECASE):
        result["function_type"] = "formatting"

    # Look for hypothesis/conclusion patterns
    hypothesis_patterns = [
        r"(?:final\s+)?hypothesis[:\s]+([^.]+\.)",
        r"(?:in\s+)?conclusion[:\s]+([^.]+\.)",
        r"this\s+neuron\s+(?:appears?\s+to\s+)?(?:be|function\s+as)\s+([^.]+\.)",
        r"the\s+neuron\s+(?:is|appears?\s+to\s+be)\s+([^.]+\.)",
    ]
    for pattern in hypothesis_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["final_hypothesis"] = match.group(1).strip()
            break

    # Look for key findings (lines starting with - or * or numbered)
    findings = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
            # Clean up the line
            finding = re.sub(r"^[-*\d.]+\s*", "", line).strip()
            if len(finding) > 20 and len(finding) < 200:
                findings.append(finding)
    result["key_findings"] = findings[:5]  # Limit to 5

    # Look for confidence
    if re.search(r"high\s+confidence", text, re.IGNORECASE):
        result["confidence_assessment"] = "high"
    elif re.search(r"medium\s+confidence", text, re.IGNORECASE):
        result["confidence_assessment"] = "medium"
    elif re.search(r"low\s+confidence", text, re.IGNORECASE):
        result["confidence_assessment"] = "low"
    elif re.search(r"confident", text, re.IGNORECASE):
        result["confidence_assessment"] = "medium"

    # Generate summary from hypothesis or first substantial finding
    if result["final_hypothesis"]:
        result["summary"] = result["final_hypothesis"][:150]
    elif result["input_function"]:
        result["summary"] = f"Activates on {result['input_function'][:100]}"

    return result


async def extract_structured_findings_llm(agent_reasoning: str) -> dict:
    """Use LLM to extract structured findings from agent reasoning."""
    # Truncate if too long (keep last 15000 chars which contain conclusions)
    if len(agent_reasoning) > 20000:
        agent_reasoning = "...[earlier reasoning truncated]...\n\n" + agent_reasoning[-15000:]

    extraction_prompt = f"""Analyze this neuron investigation and extract structured findings.

## Investigation Reasoning:
{agent_reasoning}

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
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1500,
            messages=[{"role": "user", "content": extraction_prompt}],
        )

        # Parse JSON from response
        response_text = response.choices[0].message.content.strip()
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        return json.loads(response_text)
    except Exception as e:
        print(f"  LLM extraction failed: {e}")
        print("  Falling back to regex-based extraction...")
        return extract_structured_findings_regex(agent_reasoning)


async def extract_structured_findings(agent_reasoning: str) -> dict:
    """Extract structured findings, trying LLM first then falling back to regex."""
    return await extract_structured_findings_llm(agent_reasoning)


def regenerate_dashboard(investigation: dict, structured: dict) -> dict:
    """Regenerate dashboard data with structured findings."""
    # Start with basic dashboard structure
    neuron_id = investigation.get("neuron_id", "")
    layer = investigation.get("layer", 0)
    neuron_idx = investigation.get("neuron_idx", 0)

    # Get characterization data
    char = investigation.get("characterization", {})

    # Use structured findings if available, fall back to characterization
    input_function = structured.get("input_function", "") or char.get("input_function", "")
    output_function = structured.get("output_function", "") or char.get("output_function", "")
    function_type = structured.get("function_type", "") or char.get("function_type", "")
    final_hypothesis = structured.get("final_hypothesis", "") or char.get("final_hypothesis", "")

    # Get evidence
    evidence = investigation.get("evidence", {})

    # Positive examples
    positive_examples = []
    for ex in evidence.get("activating_prompts", [])[:20]:
        positive_examples.append({
            "prompt": ex.get("prompt", "")[:200],
            "activation": ex.get("activation", 0),
            "position": ex.get("position", -1),
            "token": ex.get("token", ""),
            "is_positive": True,
        })

    # Negative examples
    negative_examples = []
    for ex in evidence.get("non_activating_prompts", [])[:10]:
        negative_examples.append({
            "prompt": ex.get("prompt", "")[:200],
            "activation": ex.get("activation", 0),
            "position": ex.get("position", -1),
            "token": ex.get("token", ""),
            "is_positive": False,
        })

    # Ablation effects
    ablation_effects = []
    consistent_promotes = []
    consistent_suppresses = []
    for effect in evidence.get("ablation_effects", []):
        if isinstance(effect, dict):
            if "promotes" in effect:
                for item in effect.get("promotes", []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        token, shift = item[0], item[1]
                        ablation_effects.append({
                            "token": token,
                            "shift": shift,
                            "direction": "promotes",
                            "consistency": "high",
                        })
                        consistent_promotes.append(token)
            if "suppresses" in effect:
                for item in effect.get("suppresses", []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        token, shift = item[0], item[1]
                        ablation_effects.append({
                            "token": token,
                            "shift": shift,
                            "direction": "suppresses",
                            "consistency": "high",
                        })
                        consistent_suppresses.append(token)

    # Hypotheses
    hypotheses = []
    for i, h in enumerate(investigation.get("hypotheses_tested", [])):
        hypotheses.append({
            "hypothesis": h.get("hypothesis", ""),
            "formed_at_experiment": i * 10,
            "evidence_for": [],
            "evidence_against": [],
            "confidence": h.get("confidence", 0.5),
            "status": h.get("status", "tested"),
        })

    # Connectivity
    connectivity = evidence.get("connectivity", {})
    upstream_nodes = []
    downstream_nodes = []
    for u in connectivity.get("upstream_neurons", []):
        upstream_nodes.append({
            "neuron_id": u.get("neuron", ""),
            "label": u.get("label", ""),
            "weight": u.get("weight", 0),
            "direction": "upstream",
            "is_logit": False,
        })
    for d in connectivity.get("downstream_targets", []):
        is_logit = "LOGIT" in d.get("target", d.get("neuron", ""))
        downstream_nodes.append({
            "neuron_id": d.get("neuron", d.get("target", "")),
            "label": d.get("label", ""),
            "weight": d.get("weight", 0),
            "direction": "downstream",
            "is_logit": is_logit,
        })

    # Confidence
    conf_str = structured.get("confidence_assessment", "").lower()
    if "high" in conf_str:
        confidence = 0.85
    elif "medium" in conf_str:
        confidence = 0.65
    elif "low" in conf_str:
        confidence = 0.40
    else:
        confidence = investigation.get("confidence", 0.5)

    # Build dashboard
    dashboard = {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "summary_card": {
            "summary": structured.get("summary", "") or final_hypothesis or f"Investigation of {neuron_id}",
            "input_function": input_function,
            "output_function": output_function,
            "function_type": function_type,
            "confidence": confidence,
            "total_experiments": investigation.get("total_experiments", 0),
            "initial_label": investigation.get("initial_label", ""),
            "transluce_positive": connectivity.get("transluce_label_positive", ""),
            "transluce_negative": connectivity.get("transluce_label_negative", ""),
        },
        "stats": {
            "activating_count": len(evidence.get("activating_prompts", [])),
            "non_activating_count": len(evidence.get("non_activating_prompts", [])),
            "ablation_count": len(evidence.get("ablation_effects", [])),
            "hypotheses_count": len(investigation.get("hypotheses_tested", [])),
        },
        "activation_patterns": {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "minimal_triggers": [],
        },
        "ablation_effects": {
            "effects": ablation_effects,
            "consistent_promotes": consistent_promotes,
            "consistent_suppresses": consistent_suppresses,
        },
        "hypothesis_timeline": {
            "hypotheses": hypotheses,
            "final_hypothesis": final_hypothesis,
        },
        "connectivity": {
            "upstream": upstream_nodes,
            "downstream": downstream_nodes,
        },
        "findings": {
            "key_findings": structured.get("key_findings", []) or investigation.get("key_findings", []),
            "open_questions": structured.get("open_questions", []) or investigation.get("open_questions", []),
        },
        "agent_reasoning": investigation.get("agent_reasoning", ""),
        "metadata": {
            "timestamp": investigation.get("timestamp", ""),
            "investigation_duration_sec": 0,
        },
    }

    return dashboard


async def process_investigation(investigation_path: Path) -> bool:
    """Process a single investigation file."""
    print(f"\nProcessing: {investigation_path.name}")

    # Load investigation
    with open(investigation_path) as f:
        investigation = json.load(f)

    # Get agent reasoning
    agent_reasoning = investigation.get("agent_reasoning", "")
    if not agent_reasoning:
        print("  No agent_reasoning found, skipping")
        return False

    print(f"  Extracting structured findings from {len(agent_reasoning)} chars of reasoning...")

    # Extract structured findings
    structured = await extract_structured_findings(agent_reasoning)

    if structured:
        print(f"  Found: {structured.get('function_type', 'unknown')} neuron")
        print(f"  Input: {structured.get('input_function', '')[:60]}...")
        print(f"  Output: {structured.get('output_function', '')[:60]}...")
        print(f"  Findings: {len(structured.get('key_findings', []))} key findings")
    else:
        print("  Warning: Could not extract structured findings")

    # Regenerate dashboard
    dashboard = regenerate_dashboard(investigation, structured)

    # Save updated dashboard
    dashboard_path = investigation_path.parent / investigation_path.name.replace("_investigation.json", "_dashboard.json")
    with open(dashboard_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"  Saved: {dashboard_path.name}")
    return True


async def main():
    """Main entry point."""
    # Find all investigation files
    investigations_dir = Path("outputs/investigations")
    if not investigations_dir.exists():
        print(f"Error: {investigations_dir} not found")
        sys.exit(1)

    investigation_files = sorted(investigations_dir.glob("*_investigation.json"))
    print(f"Found {len(investigation_files)} investigation files")

    # Process each
    successes = 0
    for inv_path in investigation_files:
        try:
            if await process_investigation(inv_path):
                successes += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n\nSummary: {successes}/{len(investigation_files)} investigations processed successfully")


if __name__ == "__main__":
    asyncio.run(main())
