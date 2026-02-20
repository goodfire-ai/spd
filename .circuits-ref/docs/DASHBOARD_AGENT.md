# Dashboard HTML Generator Agent

Transforms neuron_scientist investigation JSON into beautiful Distill.pub-style HTML pages.

## Overview

The Dashboard Agent uses the Claude Agent SDK to generate compelling, science-communication-style HTML reports from raw neuron investigation data. It takes structured JSON output from the neuron_scientist and creates visually appealing, interactive dashboards.

## Usage

```bash
# Single dashboard
python scripts/generate_html_report.py outputs/investigations/L4_N10555_dashboard.json

# Batch mode (directory)
python scripts/generate_html_report.py --batch outputs/investigations/ -o frontend/reports/

# Custom model
python scripts/generate_html_report.py L4_N10555_dashboard.json --model opus
```

## System Prompt

```
You are a science communicator creating beautiful, accessible explanations of neural network internals.

Your task is to transform raw neuron investigation data into compelling, Distill.pub-style HTML pages.

## Your Process

1. First, call `get_dashboard_data` to see the full investigation data
2. Analyze the data and craft:
   - A creative 2-4 word **title** (e.g., "Monoamine Neurotransmitter Gate")
   - A **lead paragraph** (one sentence starting with "This neuron...")
   - A **body paragraph** (2-3 sentences elaborating on the neuron's behavior)
   - A **key finding** (the most surprising discovery, 2-3 sentences)
   - **Selectivity groups** as JSON for the circuit diagram
3. Call `write_html` with all your generated content

## Writing Guidelines

**Title**: 2-4 words, conceptual and memorable, not technical jargon

**Lead paragraph**:
- ONE compelling sentence
- Start with "This neuron..."
- Use <strong> tags for key concepts

**Body paragraph**:
- 2-3 sentences elaborating on the neuron's behavior
- Highlight interesting patterns: what it responds to, what it ignores, surprising exceptions
- Only mention selectivity if it's genuinely notable or surprising
- Use <em> for emphasis
- Vary your sentence structure - don't start every body with the same phrase

**Key finding**:
- The SINGLE most surprising or important discovery
- Often involves a refuted hypothesis
- Use <strong> for key terms

**Selectivity groups** (JSON format):
{
  "fires": [
    {"label": "Fires on [category]", "examples": [{"text": "example with <mark>key</mark> word", "activation": 2.78}]}
  ],
  "ignores": [
    {"label": "Ignores [category]", "examples": [{"text": "example text", "activation": 0.08}]}
  ]
}

Focus on telling a STORY about what makes this neuron interesting.
```

## Tools

### `get_dashboard_data`

Returns curated investigation data for the agent to analyze:

| Field | Description |
|-------|-------------|
| `neuron_id` | e.g., "L15/N7414" |
| `summary` | Brief description |
| `input_function` | What inputs activate this neuron |
| `output_function` | What this neuron promotes in output |
| `confidence` | 0-1 confidence score |
| `total_experiments` | Number of experiments run |
| `positive_examples` | High-activation prompts |
| `negative_examples` | Low-activation prompts |
| `key_findings` | Top discoveries |
| `open_questions` | Unresolved questions |
| `hypotheses` | Tested hypotheses with status |
| `upstream_neurons` | Neurons feeding into this one |
| `downstream_neurons` | Neurons this one feeds |

### `write_html`

Assembles content and writes the HTML file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | string | Creative 2-4 word title |
| `narrative_lead` | string | One sentence starting with "This neuron..." |
| `narrative_body` | string | 2-3 sentences elaborating on behavior |
| `key_finding` | string | Most surprising discovery |
| `selectivity_json` | string | JSON with "fires" and "ignores" groups |

## Task Prompt Template

```
Generate a beautiful HTML dashboard for neuron {neuron_id}.

First, call `get_dashboard_data` to see the investigation results.

Then craft compelling content and call `write_html` with:
- title: A creative 2-4 word title
- narrative_lead: One sentence starting with "This neuron..."
- narrative_body: 2-3 sentences elaborating on behavior (vary your opening!)
- key_finding: The most surprising discovery (2-3 sentences)
- selectivity_json: JSON with "fires" and "ignores" groups

Make it tell a story about what makes this neuron interesting! Don't use repetitive phrasing.
```

## HTML Features

The generated HTML includes:

1. **Header**: Neuron ID, creative title, confidence badge
2. **Narrative**: Lead paragraph + body paragraph
3. **Circuit Diagram**: Upstream neurons → This neuron → Downstream neurons
4. **Selectivity Groups**: What it fires on vs ignores
5. **Steering Results**: Effects of amplifying/suppressing the neuron
6. **Activation Examples**: High and low activation prompts
7. **Key Finding**: Highlighted discovery box
8. **Hypotheses**: Tested hypotheses with confirm/refute status
9. **Expandable Experiments**: Detailed ablation, steering, and RelP results
10. **Open Questions**: Remaining unknowns

## Expandable Experiment Types

### Ablation
Shows what tokens are promoted/suppressed when the neuron is removed.

### Steering
Shows effects of artificially amplifying (+) or suppressing (-) the neuron.

### RelP (Relevance Propagation)
Shows:
- Full prompt and target tokens
- Graph statistics (τ, nodes, time)
- Whether neuron is in causal pathway
- Upstream neurons feeding this neuron
- Downstream logit connections

## Input Format

The agent expects dashboard JSON with this structure:

```json
{
  "neuron_id": "L15/N7414",
  "summary_card": {
    "summary": "...",
    "input_function": "...",
    "output_function": "...",
    "confidence": 0.85,
    "total_experiments": 25
  },
  "activation_patterns": {
    "positive_examples": [...],
    "negative_examples": [...]
  },
  "connectivity": {
    "upstream": [...],
    "downstream": [...]
  },
  "hypothesis_timeline": {
    "hypotheses": [...]
  },
  "findings": {
    "key_findings": [...],
    "open_questions": [...]
  },
  "detailed_experiments": {
    "ablation": [...],
    "steering": [...]
  },
  "relp_analysis": {
    "results": [...]
  }
}
```

## Architecture

```
neuron_scientist/
├── dashboard_agent.py    # Agent class with tools and prompts
├── html_template.py      # CSS and HTML generation
└── __init__.py          # Exports

scripts/
└── generate_html_report.py  # CLI entry point
```

## Styling

The HTML uses a Distill.pub-inspired design with:
- Clean typography (Inter font)
- Monospace code (JetBrains Mono)
- Warm accent colors
- Card-based layout
- Expandable `<details>` sections
- Clickable neuron ID links (L3/N9778 → L3_N9778.html)
