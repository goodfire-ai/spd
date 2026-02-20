#!/usr/bin/env python3
"""Generate HTML dashboards from neuron investigation JSON files.

Usage:
    # Generate dashboard for a single investigation
    python scripts/generate_neuron_dashboard.py \
        --input outputs/investigations/L15_N7890_dashboard.json \
        --output frontend/dashboards/L15_N7890.html

    # Generate dashboards for all investigations in a directory
    python scripts/generate_neuron_dashboard.py \
        --input-dir outputs/investigations \
        --output-dir frontend/dashboards

    # Also generate an index page
    python scripts/generate_neuron_dashboard.py \
        --input-dir outputs/investigations \
        --output-dir frontend/dashboards \
        --generate-index
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any


def load_template() -> str:
    """Load the HTML template."""
    template_path = Path(__file__).parent.parent / "frontend" / "templates" / "neuron_dashboard.html"
    with open(template_path) as f:
        return f.read()


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text))


def format_activation(value: float) -> str:
    """Format activation value with color class."""
    css_class = "activation-high" if value > 0.5 else "activation-low"
    return f'<span class="example-activation {css_class}">{value:.3f}</span>'


def generate_activating_examples(examples: list[dict]) -> str:
    """Generate HTML for activating examples."""
    if not examples:
        return '<div class="example-item"><span class="example-prompt">No activating examples found</span></div>'

    items = []
    for ex in examples[:15]:  # Limit to 15
        prompt = escape_html(ex.get("prompt", "")[:150])
        if len(ex.get("prompt", "")) > 150:
            prompt += "..."
        activation = ex.get("activation", 0)
        token = escape_html(ex.get("token", ""))

        items.append(f'''
        <div class="example-item">
            <span class="example-prompt">{prompt}</span>
            {format_activation(activation)}
        </div>
        ''')

    return "\n".join(items)


def generate_non_activating_examples(examples: list[dict]) -> str:
    """Generate HTML for non-activating examples."""
    if not examples:
        return '<div class="example-item"><span class="example-prompt">No examples tested</span></div>'

    items = []
    for ex in examples[:10]:  # Limit to 10
        prompt = escape_html(ex.get("prompt", "")[:150])
        if len(ex.get("prompt", "")) > 150:
            prompt += "..."
        activation = ex.get("activation", 0)

        items.append(f'''
        <div class="example-item">
            <span class="example-prompt">{prompt}</span>
            {format_activation(activation)}
        </div>
        ''')

    return "\n".join(items)


def generate_ablation_bar(token: str, shift: float, max_shift: float) -> str:
    """Generate a single ablation bar."""
    if max_shift == 0:
        max_shift = 1.0

    width_pct = min(abs(shift) / max_shift * 50, 50)  # Max 50% width
    css_class = "positive" if shift > 0 else "negative"

    return f'''
    <div class="ablation-bar-container">
        <span class="ablation-token">{escape_html(token)}</span>
        <div class="ablation-bar-wrapper">
            <div class="ablation-bar {css_class}" style="width: {width_pct}%;"></div>
        </div>
        <span class="ablation-value">{shift:+.3f}</span>
    </div>
    '''


def generate_ablation_chart(effects: list, direction: str) -> str:
    """Generate ablation chart for promotes or suppresses."""
    filtered = [e for e in effects if e.get("direction") == direction]

    if not filtered:
        return '<div style="color: var(--text-secondary); font-style: italic;">No consistent effects found</div>'

    # Find max for scaling
    max_shift = max(abs(e.get("shift", 0)) for e in filtered) if filtered else 1.0

    bars = []
    for effect in sorted(filtered, key=lambda x: -abs(x.get("shift", 0)))[:8]:
        token = effect.get("token", "?")
        shift = effect.get("shift", 0)
        bars.append(generate_ablation_bar(token, shift, max_shift))

    return "\n".join(bars)


def generate_connectivity_nodes(nodes: list[dict], direction: str) -> str:
    """Generate connectivity node HTML."""
    if not nodes:
        return f'<div class="connectivity-node {direction}">None found</div>'

    items = []
    for node in nodes[:5]:  # Limit to 5
        neuron_id = node.get("neuron_id", "?")
        weight = node.get("weight", 0)
        label = node.get("label", "")
        is_logit = node.get("is_logit", False)

        css_class = "logit" if is_logit else direction
        display = f"{neuron_id}"
        if label:
            display = f"{neuron_id}: {label[:20]}"

        items.append(f'''
        <div class="connectivity-node {css_class}" title="Weight: {weight:.3f}">
            {escape_html(display)}
        </div>
        ''')

    return "\n".join(items)


def generate_hypothesis_timeline(hypotheses: list[dict]) -> str:
    """Generate hypothesis timeline HTML."""
    if not hypotheses:
        return '<div class="timeline-item"><div class="timeline-hypothesis">No hypotheses tested</div></div>'

    items = []
    for h in hypotheses[:10]:
        hypothesis = escape_html(h.get("hypothesis", "")[:200])
        status = h.get("status", "testing")
        confidence = h.get("confidence", 0.5)

        evidence_for = h.get("evidence_for", [])
        evidence_against = h.get("evidence_against", [])

        evidence_html = ""
        if evidence_for:
            evidence_html += f'<span style="color: var(--accent-green);">+{len(evidence_for)} for</span> '
        if evidence_against:
            evidence_html += f'<span style="color: var(--accent-red);">-{len(evidence_against)} against</span>'

        items.append(f'''
        <div class="timeline-item {status}">
            <div class="timeline-hypothesis">{hypothesis}</div>
            <div class="timeline-evidence">
                Confidence: {confidence:.0%} | {evidence_html}
            </div>
        </div>
        ''')

    return "\n".join(items)


def generate_findings_list(findings: list[str]) -> str:
    """Generate key findings or open questions list."""
    if not findings:
        return '<li>None documented</li>'

    return "\n".join(f'<li>{escape_html(f)}</li>' for f in findings[:10])


def get_function_type_badge(function_type: str) -> str:
    """Get badge HTML for function type."""
    badge_colors = {
        "semantic": "badge-green",
        "routing": "badge-blue",
        "formatting": "badge-yellow",
        "lexical": "badge-purple",
        "unknown": "badge-red",
    }
    color = badge_colors.get(function_type.lower(), "badge-blue")
    return f'<span class="badge {color}">{escape_html(function_type or "Unknown")}</span>'


def generate_dashboard(data: dict[str, Any], template: str) -> str:
    """Generate HTML dashboard from investigation data."""

    # Handle both full investigation and dashboard-only formats
    if "dashboard" in data:
        dashboard = data["dashboard"]
    elif "summary_card" in data:
        dashboard = data
    else:
        # Convert old format
        dashboard = {
            "neuron_id": data.get("neuron_id", ""),
            "layer": data.get("layer", 0),
            "neuron_idx": data.get("neuron_idx", 0),
            "summary_card": {
                "summary": data.get("characterization", {}).get("final_hypothesis", ""),
                "input_function": data.get("characterization", {}).get("input_function", ""),
                "output_function": data.get("characterization", {}).get("output_function", ""),
                "function_type": data.get("characterization", {}).get("function_type", ""),
                "confidence": data.get("confidence", 0),
                "total_experiments": data.get("total_experiments", 0),
                "initial_label": data.get("initial_label", ""),
            },
            "stats": {
                "activating_count": len(data.get("evidence", {}).get("activating_prompts", [])),
                "non_activating_count": len(data.get("evidence", {}).get("non_activating_prompts", [])),
                "ablation_count": len(data.get("evidence", {}).get("ablation_effects", [])),
                "hypotheses_count": len(data.get("hypotheses_tested", [])),
            },
            "activation_patterns": {
                "positive_examples": [{"prompt": p.get("prompt", ""), "activation": p.get("activation", 0)}
                                     for p in data.get("evidence", {}).get("activating_prompts", [])],
                "negative_examples": [{"prompt": p.get("prompt", ""), "activation": p.get("activation", 0)}
                                     for p in data.get("evidence", {}).get("non_activating_prompts", [])],
            },
            "ablation_effects": {"effects": [], "consistent_promotes": [], "consistent_suppresses": []},
            "hypothesis_timeline": {
                "hypotheses": data.get("hypotheses_tested", []),
                "final_hypothesis": data.get("characterization", {}).get("final_hypothesis", ""),
            },
            "connectivity": {"upstream": [], "downstream": []},
            "findings": {
                "key_findings": data.get("key_findings", []),
                "open_questions": data.get("open_questions", []),
            },
            "agent_reasoning": data.get("agent_reasoning", ""),
            "metadata": {"timestamp": data.get("timestamp", "")},
        }

    summary_card = dashboard.get("summary_card", {})
    stats = dashboard.get("stats", {})
    activation_patterns = dashboard.get("activation_patterns", {})
    ablation_effects = dashboard.get("ablation_effects", {})
    hypothesis_timeline = dashboard.get("hypothesis_timeline", {})
    connectivity = dashboard.get("connectivity", {})
    findings = dashboard.get("findings", {})
    metadata = dashboard.get("metadata", {})

    # Calculate values
    confidence_pct = int(summary_card.get("confidence", 0) * 100)

    # Perform replacements
    result = template

    replacements = {
        "{{NEURON_ID}}": escape_html(dashboard.get("neuron_id", "")),
        "{{LAYER}}": str(dashboard.get("layer", 0)),
        "{{NEURON_IDX}}": str(dashboard.get("neuron_idx", 0)),
        "{{TIMESTAMP}}": escape_html(metadata.get("timestamp", "")),
        "{{SUMMARY}}": escape_html(summary_card.get("summary", "Neuron Investigation")),
        "{{INPUT_FUNCTION}}": escape_html(summary_card.get("input_function", "Not determined")),
        "{{OUTPUT_FUNCTION}}": escape_html(summary_card.get("output_function", "Not determined")),
        "{{FUNCTION_TYPE_BADGE}}": get_function_type_badge(summary_card.get("function_type", "")),
        "{{CONFIDENCE_PCT}}": str(confidence_pct),
        "{{TOTAL_EXPERIMENTS}}": str(summary_card.get("total_experiments", 0)),
        "{{INITIAL_LABEL}}": escape_html(summary_card.get("initial_label", "")),
        "{{ACTIVATING_COUNT}}": str(stats.get("activating_count", 0)),
        "{{NON_ACTIVATING_COUNT}}": str(stats.get("non_activating_count", 0)),
        "{{ABLATION_COUNT}}": str(stats.get("ablation_count", 0)),
        "{{HYPOTHESES_COUNT}}": str(stats.get("hypotheses_count", 0)),
        "{{ACTIVATING_EXAMPLES}}": generate_activating_examples(
            activation_patterns.get("positive_examples", [])
        ),
        "{{NON_ACTIVATING_EXAMPLES}}": generate_non_activating_examples(
            activation_patterns.get("negative_examples", [])
        ),
        "{{PROMOTES_CHART}}": generate_ablation_chart(
            ablation_effects.get("effects", []), "promotes"
        ),
        "{{SUPPRESSES_CHART}}": generate_ablation_chart(
            ablation_effects.get("effects", []), "suppresses"
        ),
        "{{UPSTREAM_NODES}}": generate_connectivity_nodes(
            connectivity.get("upstream", []), "upstream"
        ),
        "{{DOWNSTREAM_NODES}}": generate_connectivity_nodes(
            connectivity.get("downstream", []), "downstream"
        ),
        "{{HYPOTHESIS_TIMELINE}}": generate_hypothesis_timeline(
            hypothesis_timeline.get("hypotheses", [])
        ),
        "{{FINAL_HYPOTHESIS}}": escape_html(
            hypothesis_timeline.get("final_hypothesis", "No final hypothesis determined")
        ),
        "{{KEY_FINDINGS}}": generate_findings_list(findings.get("key_findings", [])),
        "{{OPEN_QUESTIONS}}": generate_findings_list(findings.get("open_questions", [])),
        "{{AGENT_REASONING}}": escape_html(dashboard.get("agent_reasoning", "No reasoning recorded")),
        "{{DASHBOARD_JSON}}": json.dumps(dashboard),
    }

    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


def generate_index_page(dashboards: list[dict[str, Any]], output_dir: Path) -> str:
    """Generate an index page linking to all dashboards."""
    index_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuron Investigation Index</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-card: #1f2937;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --border-color: #374151;
        }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { margin-bottom: 2rem; }
        .neuron-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }
        .neuron-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            text-decoration: none;
            color: var(--text-primary);
            transition: transform 0.2s, border-color 0.2s;
        }
        .neuron-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-blue);
        }
        .neuron-id {
            font-family: 'Fira Code', monospace;
            color: var(--accent-blue);
            font-size: 1.125rem;
            margin-bottom: 0.5rem;
        }
        .neuron-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }
        .neuron-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        .stat {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        .confidence-bar {
            width: 100%;
            height: 4px;
            background: rgba(0,0,0,0.3);
            border-radius: 2px;
            margin-top: 0.5rem;
        }
        .confidence-fill {
            height: 100%;
            background: var(--accent-green);
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Neuron Investigation Index</h1>
        <p style="color: var(--text-secondary); margin-bottom: 2rem;">
            {{TOTAL_COUNT}} neurons investigated
        </p>
        <div class="neuron-grid">
            {{NEURON_CARDS}}
        </div>
    </div>
</body>
</html>'''

    cards = []
    for d in sorted(dashboards, key=lambda x: -x.get("summary_card", {}).get("confidence", 0)):
        neuron_id = d.get("neuron_id", "Unknown")
        safe_id = neuron_id.replace("/", "_")
        summary_card = d.get("summary_card", {})
        stats = d.get("stats", {})

        confidence = int(summary_card.get("confidence", 0) * 100)
        label = summary_card.get("initial_label", "") or summary_card.get("output_function", "")[:50]
        experiments = summary_card.get("total_experiments", 0)

        cards.append(f'''
        <a href="{safe_id}.html" class="neuron-card">
            <div class="neuron-id">{escape_html(neuron_id)}</div>
            <div class="neuron-label">{escape_html(label)}</div>
            <div class="neuron-stats">
                <span class="stat">{experiments} experiments</span>
                <span class="stat">{stats.get("activating_count", 0)} activating</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%"></div>
            </div>
        </a>
        ''')

    result = index_template.replace("{{TOTAL_COUNT}}", str(len(dashboards)))
    result = result.replace("{{NEURON_CARDS}}", "\n".join(cards))

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate neuron dashboards")

    parser.add_argument("--input", type=Path,
                        help="Path to single investigation JSON")
    parser.add_argument("--input-dir", type=Path,
                        help="Directory containing investigation JSONs")
    parser.add_argument("--output", type=Path,
                        help="Output path for single dashboard HTML")
    parser.add_argument("--output-dir", type=Path, default=Path("frontend/dashboards"),
                        help="Output directory for dashboards")
    parser.add_argument("--generate-index", action="store_true",
                        help="Generate index.html linking all dashboards")

    args = parser.parse_args()

    template = load_template()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dashboards = []

    if args.input:
        # Single file mode
        with open(args.input) as f:
            data = json.load(f)

        html_content = generate_dashboard(data, template)

        if args.output:
            output_path = args.output
        else:
            safe_id = data.get("neuron_id", "unknown").replace("/", "_")
            output_path = args.output_dir / f"{safe_id}.html"

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Generated: {output_path}")

        if "dashboard" in data:
            dashboards.append(data["dashboard"])
        elif "summary_card" in data:
            dashboards.append(data)

    elif args.input_dir:
        # Directory mode
        json_files = list(args.input_dir.glob("*_dashboard.json"))
        if not json_files:
            json_files = list(args.input_dir.glob("*_investigation.json"))

        print(f"Found {len(json_files)} investigation files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                html_content = generate_dashboard(data, template)

                safe_id = data.get("neuron_id", json_file.stem).replace("/", "_")
                output_path = args.output_dir / f"{safe_id}.html"

                with open(output_path, "w") as f:
                    f.write(html_content)

                print(f"Generated: {output_path}")

                if "dashboard" in data:
                    dashboards.append(data["dashboard"])
                elif "summary_card" in data:
                    dashboards.append(data)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    else:
        print("Please specify --input or --input-dir")
        return

    # Generate index if requested
    if args.generate_index and dashboards:
        index_html = generate_index_page(dashboards, args.output_dir)
        index_path = args.output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write(index_html)
        print(f"Generated index: {index_path}")


if __name__ == "__main__":
    main()
