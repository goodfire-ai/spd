#!/usr/bin/env python3
"""Run LLM extraction as a Claude Agent SDK task (has API access)."""

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)


async def run_extraction_task():
    """Run extraction using Claude Agent SDK which has API access."""

    investigations_dir = Path("outputs/investigations")
    investigation_files = sorted(investigations_dir.glob("*_investigation.json"))

    # Load all investigations
    investigations = {}
    for path in investigation_files:
        neuron_id = path.stem.replace("_investigation", "")
        with open(path) as f:
            data = json.load(f)
        reasoning = data.get("agent_reasoning", "")
        if reasoning and len(reasoning) > 500:  # Only process if substantial reasoning
            investigations[neuron_id] = {
                "path": str(path),
                "reasoning": reasoning[-15000:] if len(reasoning) > 15000 else reasoning,
                "data": data,
            }

    print(f"Found {len(investigations)} investigations with reasoning to extract")

    # Create tool for saving extractions
    @tool("save_extraction", "Save extracted findings for a neuron", {"neuron_id": str, "extraction_json": str})
    async def save_extraction(args):
        neuron_id = args["neuron_id"]
        try:
            extraction = json.loads(args["extraction_json"])
        except:
            return {"content": [{"type": "text", "text": f"Invalid JSON for {neuron_id}"}]}

        # Find the investigation
        if neuron_id not in investigations:
            return {"content": [{"type": "text", "text": f"Unknown neuron: {neuron_id}"}]}

        inv = investigations[neuron_id]["data"]

        # Update characterization
        inv["characterization"] = {
            "final_hypothesis": extraction.get("final_hypothesis", ""),
            "input_function": extraction.get("input_function", ""),
            "output_function": extraction.get("output_function", ""),
            "function_type": extraction.get("function_type", ""),
        }
        inv["key_findings"] = extraction.get("key_findings", [])
        inv["open_questions"] = extraction.get("open_questions", [])

        # Update confidence
        conf_str = extraction.get("confidence_assessment", "").lower()
        if "high" in conf_str:
            inv["confidence"] = 0.85
        elif "medium" in conf_str:
            inv["confidence"] = 0.65
        elif "low" in conf_str:
            inv["confidence"] = 0.40

        # Save updated investigation
        inv_path = Path(investigations[neuron_id]["path"])
        with open(inv_path, "w") as f:
            json.dump(inv, f, indent=2)

        # Regenerate dashboard
        dashboard = regenerate_dashboard(inv, extraction)
        dashboard_path = inv_path.parent / inv_path.name.replace("_investigation.json", "_dashboard.json")
        with open(dashboard_path, "w") as f:
            json.dump(dashboard, f, indent=2)

        return {"content": [{"type": "text", "text": f"Saved {neuron_id}"}]}

    # Build prompt with all investigations
    prompt_parts = ["# Neuron Extraction Task\n\nFor each neuron below, extract structured findings and save them.\n\n"]

    for neuron_id, inv in list(investigations.items())[:5]:  # Start with 5
        prompt_parts.append(f"## {neuron_id}\n\n")
        prompt_parts.append("### Agent Reasoning:\n")
        prompt_parts.append(inv["reasoning"][:5000] + "..." if len(inv["reasoning"]) > 5000 else inv["reasoning"])
        prompt_parts.append("\n\n")

    prompt_parts.append("""
For each neuron above, call save_extraction with a JSON object containing:
{
  "input_function": "What activates this neuron",
  "output_function": "What this neuron promotes/suppresses",
  "function_type": "semantic/syntactic/routing/formatting/hybrid",
  "final_hypothesis": "What the neuron does",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "open_questions": ["Question 1", ...],
  "confidence_assessment": "high/medium/low with justification",
  "summary": "One sentence summary"
}

Process each neuron and save the extraction.
""")

    prompt = "\n".join(prompt_parts)

    mcp_server = create_sdk_mcp_server(
        name="extraction_tools",
        version="1.0.0",
        tools=[save_extraction],
    )

    options = ClaudeAgentOptions(
        system_prompt="You are an assistant that extracts structured findings from neuron investigations. For each neuron, analyze the reasoning and extract key information.",
        max_turns=20,
        model="sonnet",
        mcp_servers={"extraction_tools": mcp_server},
        allowed_tools=["mcp__extraction_tools__save_extraction"],
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            pass  # Process messages

    print("Extraction complete!")


def regenerate_dashboard(investigation: dict, structured: dict) -> dict:
    """Regenerate dashboard data with structured findings."""
    neuron_id = investigation.get("neuron_id", "")
    layer = investigation.get("layer", 0)
    neuron_idx = investigation.get("neuron_idx", 0)

    char = investigation.get("characterization", {})
    input_function = structured.get("input_function", "") or char.get("input_function", "")
    output_function = structured.get("output_function", "") or char.get("output_function", "")
    function_type = structured.get("function_type", "") or char.get("function_type", "")
    final_hypothesis = structured.get("final_hypothesis", "") or char.get("final_hypothesis", "")

    evidence = investigation.get("evidence", {})

    positive_examples = [
        {"prompt": ex.get("prompt", "")[:200], "activation": ex.get("activation", 0),
         "position": ex.get("position", -1), "token": ex.get("token", ""), "is_positive": True}
        for ex in evidence.get("activating_prompts", [])[:20]
    ]

    negative_examples = [
        {"prompt": ex.get("prompt", "")[:200], "activation": ex.get("activation", 0),
         "position": ex.get("position", -1), "token": ex.get("token", ""), "is_positive": False}
        for ex in evidence.get("non_activating_prompts", [])[:10]
    ]

    conf_str = structured.get("confidence_assessment", "").lower()
    confidence = 0.85 if "high" in conf_str else 0.65 if "medium" in conf_str else 0.40 if "low" in conf_str else investigation.get("confidence", 0.5)

    return {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "summary_card": {
            "summary": structured.get("summary", "") or final_hypothesis,
            "input_function": input_function,
            "output_function": output_function,
            "function_type": function_type,
            "confidence": confidence,
            "total_experiments": investigation.get("total_experiments", 0),
            "initial_label": investigation.get("initial_label", ""),
            "transluce_positive": "",
            "transluce_negative": "",
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
        "ablation_effects": {"effects": [], "consistent_promotes": [], "consistent_suppresses": []},
        "hypothesis_timeline": {"hypotheses": [], "final_hypothesis": final_hypothesis},
        "connectivity": {"upstream": [], "downstream": []},
        "findings": {
            "key_findings": structured.get("key_findings", []),
            "open_questions": structured.get("open_questions", []),
        },
        "agent_reasoning": investigation.get("agent_reasoning", ""),
        "metadata": {"timestamp": investigation.get("timestamp", ""), "investigation_duration_sec": 0},
    }


if __name__ == "__main__":
    asyncio.run(run_extraction_task())
