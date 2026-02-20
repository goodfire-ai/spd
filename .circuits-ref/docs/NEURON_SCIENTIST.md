# Neuron Scientist Agent

Automated neuron investigation using an LLM agent with MCP tools. The agent performs hypothesis-driven experiments to understand what each neuron does.

## Overview

The full pipeline includes:
1. **NeuronScientist**: Forms and tests hypotheses about neuron function
2. **NeuronSkeptic**: Adversarially stress-tests the hypothesis
3. **GPT Reviewer**: Peer reviews the combined evidence
4. **Dashboard Agent**: Generates HTML reports

The Neuron Scientist agent:
1. Starts with a neuron ID (e.g., `L31/N6452`)
2. Runs activation tests, RelP experiments, and connectivity analysis
3. Forms and tests hypotheses about the neuron's function
4. Produces a structured investigation report + dashboard JSON

## Quick Start

### Single Neuron Investigation

```bash
# Basic investigation
.venv/bin/python scripts/run_neuron_scientist.py --neuron L15/N7890

# With initial hypothesis
.venv/bin/python scripts/run_neuron_scientist.py \
    --neuron L15/N7890 \
    --hypothesis "This neuron responds to medical terminology"

# With edge stats for connectivity analysis
.venv/bin/python scripts/run_neuron_scientist.py \
    --neuron L15/N7890 \
    --edge-stats data/medical_edge_stats_v6_enriched.json
```

### Batch Investigation (Local)

Run multiple investigations on the current GPU:

```bash
# Investigate all neurons in a circuit graph
.venv/bin/python scripts/batch_local_investigate.py \
    --graph graphs/aspirin-cox-target.json \
    --parallel 4

# Limit to first N neurons
.venv/bin/python scripts/batch_local_investigate.py \
    --graph graphs/aspirin-cox-target.json \
    --parallel 4 \
    --limit 20
```

### Batch Investigation (SLURM - Multi-GPU)

Distribute investigations across multiple GPUs via SLURM:

```bash
# Dry run to see job distribution
.venv/bin/python scripts/batch_slurm_investigate.py \
    --graph graphs/aspirin-cox-target.json \
    --agents-per-gpu 4 \
    --max-gpus 24 \
    --dry-run

# Submit jobs
.venv/bin/python scripts/batch_slurm_investigate.py \
    --graph graphs/aspirin-cox-target.json \
    --agents-per-gpu 4 \
    --max-gpus 24 \
    --edge-stats data/medical_edge_stats_v6_enriched.json

# Monitor progress
squeue -u $USER | grep neuron
grep "Completed" outputs/investigations/slurm_*/slurm_*_*.out
```

**Requirements:**
- ANTHROPIC_API_KEY must be set in `.env` file
- SLURM partition with GPU access (default: `h200-reserved`)

## What the Agent Does

The agent uses Claude (opus model) with MCP tools to investigate neurons:

### Phase 1: Exploration
- `batch_activation_test`: Test 10+ prompts to find what activates the neuron
- `analyze_connectivity`: Check upstream sources and downstream targets
- `get_output_projections`: See what tokens the neuron promotes/suppresses

### Phase 2: Hypothesis Testing
- `test_activation`: Test specific prompts with activation measurement
- `run_relp`: Run RelP attribution to see neuron's role in circuits
- `steer_dose_response`: Test causal effects by steering the neuron

### Phase 3: Verification
- `adaptive_relp`: Find prompts where the neuron has high influence
- `run_baseline_comparison`: Compare against random neurons for significance

## NeuronSkeptic (Adversarial Testing)

After the Scientist produces a hypothesis, the NeuronSkeptic tries to **disprove** it through adversarial testing. This produces evidence that goes to the GPT reviewer alongside the Scientist's findings.

### What the Skeptic Tests

1. **Alternative Hypotheses**: "Is it really 'medical terms' or just 'Latin-derived words'?"
2. **Boundary Cases**: Edge cases that should/shouldn't activate
3. **Confounds**: Position effects, length effects, co-occurrence
4. **Selectivity**: False positive/negative rates

### Running the Skeptic

```bash
# Run on an existing investigation
.venv/bin/python scripts/run_skeptic.py \
    --investigation neuron_reports/json/L15_N7890_investigation.json

# With specific model (sonnet is default, good for adversarial thinking)
.venv/bin/python scripts/run_skeptic.py \
    --investigation neuron_reports/json/L15_N7890_investigation.json \
    --model opus
```

### Skeptic Report

The skeptic produces a report with:
- **Verdict**: SUPPORTED, WEAKENED, or REFUTED
- **Confidence adjustment**: How much to adjust the scientist's confidence
- **Alternative hypotheses tested**: With verdicts (distinguished, indistinguishable)
- **Boundary tests**: Pass/fail for edge cases
- **Confounds detected**: With severity ratings
- **Revised hypothesis**: If the skeptic has a better explanation

### Integration with NeuronPI

When using the full NeuronPI pipeline, the Skeptic runs automatically:

```
NeuronScientist → NeuronSkeptic → GPT Review → (iterate if needed) → Dashboard
```

The GPT reviewer sees BOTH the Scientist's findings AND the Skeptic's adversarial evidence, leading to more robust verdicts.

## Output Files

Each investigation produces two files in `outputs/investigations/`:

- `L{layer}_N{neuron}_investigation.json` - Full experiment log
- `L{layer}_N{neuron}_dashboard.json` - Structured data for visualization

### Dashboard JSON Structure

```json
{
  "neuron_id": "L31/N6452",
  "layer": 31,
  "neuron": 6452,
  "headline": "Short description of function",
  "summary": "Detailed explanation...",
  "confidence": 0.85,
  "selectivity": {
    "activating": ["prompt1", "prompt2"],
    "non_activating": ["prompt3", "prompt4"]
  },
  "experiments": {
    "activation_tests": [...],
    "relp_experiments": [...],
    "ablation_effects": [...]
  },
  "key_findings": ["finding1", "finding2"],
  "hypotheses_tested": [...]
}
```

## Generating HTML Reports

Convert dashboard JSON to interactive HTML:

```bash
# Single dashboard
.venv/bin/python scripts/generate_html_report.py \
    outputs/investigations/L15_N7414_dashboard.json \
    -o frontend/reports/

# Batch mode
.venv/bin/python scripts/generate_html_report.py \
    --batch outputs/investigations/ \
    -o frontend/reports/
```

View reports at: `http://localhost:8888/` (if running `python -m http.server 8888` in `frontend/reports/`)

## Circuit-Based Investigation

To investigate all neurons in a specific circuit:

### 1. Generate a Circuit Graph

```bash
# Trace a prompt to see what neurons are involved
.venv/bin/python scripts/generate_graph.py \
    "The enzyme inhibited by aspirin that reduces inflammation is" \
    --slug aspirin-cox \
    --target-tokens " cyclo" \
    --no-labels
```

### 2. Check Overlap with Existing Data

```python
import json

# Load graph neurons
with open('graphs/aspirin-cox.json') as f:
    graph = json.load(f)
neurons = [(int(n['layer']), n['feature'])
           for n in graph['nodes']
           if n.get('feature_type') == 'mlp_neuron']

# Load v6 neurons
with open('data/medical_edge_stats_v6_enriched.json') as f:
    v6 = json.load(f)
v6_neurons = set((p['layer'], p['neuron']) for p in v6['profiles'])

# Check overlap
overlap = set(neurons) & v6_neurons
print(f"{len(overlap)}/{len(neurons)} neurons in v6")
```

### 3. Run Batch Investigation

```bash
.venv/bin/python scripts/batch_slurm_investigate.py \
    --graph graphs/aspirin-cox.json \
    --agents-per-gpu 4 \
    --max-gpus 24
```

## Tips

- **4 agents per GPU** works well on H200s (143GB VRAM)
- Investigations take **2-5 minutes** each
- Use `--v6-only` (default) to focus on neurons with existing profiles
- Late layers (L25-L31) are most interpretable
- Check SLURM logs in `outputs/investigations/slurm_*/` for debugging

## Related Docs

- [NEURON_ANALYSIS_GUIDE.md](NEURON_ANALYSIS_GUIDE.md) - Manual analysis techniques
- [AUTOINTERP_PIPELINE.md](AUTOINTERP_PIPELINE.md) - Full interpretation pipeline
