#!/usr/bin/env python3
"""
Relabel neurons using bidirectional approach:
- Pass 1 (Backwards L31→L25): What does this neuron DO? (output logits)
- Pass 2 (Forwards L0→L10): What ACTIVATES this neuron? (input patterns)
- Pass 3 (Middle + Consolidate): Combine context with function
"""

import asyncio
import json

from anthropic import AsyncAnthropic

# Load data
with open("data/neuron_labeling_data.json") as f:
    neuron_data = json.load(f)

with open("frontend/clusters/neuron_graph_viewer.json") as f:
    graph = json.load(f)

client = AsyncAnthropic()

# Store labels as we generate them
labels = {}


def format_logit_outputs(logit_outputs):
    """Format logit outputs for the prompt."""
    if not logit_outputs:
        return "No direct logit connections"

    # Dedupe by token
    by_token = {}
    for lo in logit_outputs:
        token = lo["token"]
        if token not in by_token or abs(lo["weight"]) > abs(by_token[token]["weight"]):
            by_token[token] = lo

    lines = []
    for token, lo in sorted(by_token.items(), key=lambda x: -abs(x[1]["weight"])):
        sign = "promotes" if lo["weight"] > 0 else "inhibits"
        lines.append(f"  {sign} token {repr(token)} (weight {lo['weight']:.3f}, appears {lo['frequency']*100:.0f}% of activations)")
    return "\n".join(lines[:5])


def format_connections(connections, direction="input"):
    """Format connection list with any existing labels."""
    if not connections:
        return f"No {direction}s in graph"

    lines = []
    for node_id, weight in connections[:5]:
        label_info = ""
        if node_id in labels:
            label_info = f" [{labels[node_id]['short']}]"
        sign = "+" if weight > 0 else ""
        lines.append(f"  {node_id}{label_info}: {sign}{weight:.4f}")
    return "\n".join(lines)


async def label_late_layer_neuron(node_id: str) -> dict:
    """Label a late-layer neuron based on what it DOES (output logits)."""
    data = neuron_data[node_id]

    prompt = f"""You are labeling neuron {node_id} in a language model.

ACTIVATION PATTERN (when this neuron fires):
{data['activation_pattern'] or 'No activation data available'}

OUTPUT LOGITS (what tokens this neuron promotes/inhibits):
{format_logit_outputs(data['logit_outputs'])}

TOP INPUTS (strongest connections feeding into this neuron):
{format_connections(data['top_inputs'], 'input')}

Based on the OUTPUT LOGITS, describe what this neuron DOES - what tokens does it promote or suppress?

Provide:
1. SHORT_LABEL: 3-6 words describing its function (e.g., "promotes token 'The'", "suppresses sentence starters")
2. DESCRIPTION: 1-2 sentences describing what it does and when, referencing the actual strongest inputs.

Format:
SHORT_LABEL: <label>
DESCRIPTION: <description>"""

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    short = ""
    desc = ""

    for line in text.split("\n"):
        if line.startswith("SHORT_LABEL:"):
            short = line.replace("SHORT_LABEL:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            desc = line.replace("DESCRIPTION:", "").strip()

    # Continue reading description if it spans multiple lines
    if "DESCRIPTION:" in text:
        desc_start = text.find("DESCRIPTION:") + len("DESCRIPTION:")
        desc = text[desc_start:].strip()

    return {"short": short, "detailed": desc}


async def label_early_layer_neuron(node_id: str) -> dict:
    """Label an early-layer neuron based on what ACTIVATES it."""
    data = neuron_data[node_id]

    # Get downstream targets with their labels
    downstream_info = []
    for tgt, weight in data["top_outputs"][:5]:
        if tgt in labels:
            downstream_info.append(f"{tgt} [{labels[tgt]['short']}]: {weight:+.4f}")
        else:
            downstream_info.append(f"{tgt}: {weight:+.4f}")

    prompt = f"""You are labeling neuron {node_id} in a language model.

ACTIVATION PATTERN (what input tokens/contexts trigger this neuron):
{data['activation_pattern'] or 'No activation data available'}

TOP OUTPUTS (what this neuron feeds into - some already labeled):
{chr(10).join(downstream_info) if downstream_info else 'No outputs in graph'}

Based on the ACTIVATION PATTERN, describe what ACTIVATES this neuron - what input patterns does it detect?

Provide:
1. SHORT_LABEL: 3-6 words describing what it detects (e.g., "URL token detector", "sentence boundary detector")
2. DESCRIPTION: 1-2 sentences describing what activates it and what downstream effect this has.

Format:
SHORT_LABEL: <label>
DESCRIPTION: <description>"""

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    short = ""
    desc = ""

    for line in text.split("\n"):
        if line.startswith("SHORT_LABEL:"):
            short = line.replace("SHORT_LABEL:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            desc = line.replace("DESCRIPTION:", "").strip()

    if "DESCRIPTION:" in text:
        desc_start = text.find("DESCRIPTION:") + len("DESCRIPTION:")
        desc = text[desc_start:].strip()

    return {"short": short, "detailed": desc}


async def label_mid_layer_neuron(node_id: str) -> dict:
    """Label a mid-layer neuron by combining upstream context with downstream function."""
    data = neuron_data[node_id]

    # Format inputs with labels
    input_info = []
    for src, weight in data["top_inputs"][:5]:
        if src in labels:
            input_info.append(f"{src} [{labels[src]['short']}]: {weight:+.4f}")
        else:
            input_info.append(f"{src}: {weight:+.4f}")

    # Format outputs with labels
    output_info = []
    for tgt, weight in data["top_outputs"][:5]:
        if tgt in labels:
            output_info.append(f"{tgt} [{labels[tgt]['short']}]: {weight:+.4f}")
        else:
            output_info.append(f"{tgt}: {weight:+.4f}")

    prompt = f"""You are labeling neuron {node_id} in a language model.

ACTIVATION PATTERN:
{data['activation_pattern'] or 'No activation data available'}

TOP INPUTS (upstream neurons with their functions):
{chr(10).join(input_info) if input_info else 'No inputs in graph'}

TOP OUTPUTS (downstream neurons with their functions):
{chr(10).join(output_info) if output_info else 'No outputs in graph'}

OUTPUT LOGITS (if any direct token effects):
{format_logit_outputs(data['logit_outputs'])}

Describe this neuron's role in the circuit. Consider:
- What upstream context activates it?
- What downstream effect does it have?
- How does it transform input signals into output signals?

Provide:
1. SHORT_LABEL: 3-6 words (e.g., "URL-to-sentence-start gate", "technical context integrator")
2. DESCRIPTION: 1-2 sentences in this format: "This neuron [function] when [context]. It receives [inputs] and [output effect]."

Format:
SHORT_LABEL: <label>
DESCRIPTION: <description>"""

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    short = ""
    desc = ""

    for line in text.split("\n"):
        if line.startswith("SHORT_LABEL:"):
            short = line.replace("SHORT_LABEL:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            desc = line.replace("DESCRIPTION:", "").strip()

    if "DESCRIPTION:" in text:
        desc_start = text.find("DESCRIPTION:") + len("DESCRIPTION:")
        desc = text[desc_start:].strip()

    return {"short": short, "detailed": desc}


async def label_batch(node_ids: list, label_func, desc: str):
    """Label a batch of neurons concurrently."""
    print(f"\n{desc} ({len(node_ids)} neurons)...")

    tasks = [label_func(nid) for nid in node_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for nid, result in zip(node_ids, results):
        if isinstance(result, Exception):
            print(f"  Error labeling {nid}: {result}")
            labels[nid] = {"short": "ERROR", "detailed": str(result)}
        else:
            labels[nid] = result
            print(f"  {nid}: {result['short']}")


async def main():
    # Group neurons by layer
    by_layer = {}
    for node_id in neuron_data:
        layer = neuron_data[node_id]["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(node_id)

    print(f"Total neurons: {len(neuron_data)}")
    print(f"Layers: {sorted(by_layer.keys())}")

    # Pass 1: Late layers (L31 → L25) - what they DO
    print("\n" + "="*60)
    print("PASS 1: Late layers (L31→L25) - What does each neuron DO?")
    print("="*60)

    for layer in range(31, 24, -1):
        if layer in by_layer:
            await label_batch(
                by_layer[layer],
                label_late_layer_neuron,
                f"Layer {layer}"
            )

    # Pass 2: Early layers (L0 → L10) - what ACTIVATES them
    print("\n" + "="*60)
    print("PASS 2: Early layers (L0→L10) - What ACTIVATES each neuron?")
    print("="*60)

    for layer in range(0, 11):
        if layer in by_layer:
            await label_batch(
                by_layer[layer],
                label_early_layer_neuron,
                f"Layer {layer}"
            )

    # Pass 3: Mid layers (L11 → L24) - combine context + function
    print("\n" + "="*60)
    print("PASS 3: Mid layers (L11→L24) - Combining context + function")
    print("="*60)

    for layer in range(11, 25):
        if layer in by_layer:
            await label_batch(
                by_layer[layer],
                label_mid_layer_neuron,
                f"Layer {layer}"
            )

    # Update graph with new labels
    print("\n" + "="*60)
    print("Updating graph with new labels...")
    print("="*60)

    for node_id in graph["nodes"]:
        if node_id in labels:
            graph["nodes"][node_id]["functional_label"] = labels[node_id]["short"]
            graph["nodes"][node_id]["detailed_description"] = labels[node_id]["detailed"]

    # Save updated graph
    with open("frontend/clusters/neuron_graph_viewer.json", "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved {len(labels)} labels to graph")

    # Also save labels separately
    with open("data/neuron_labels_v2.json", "w") as f:
        json.dump(labels, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
