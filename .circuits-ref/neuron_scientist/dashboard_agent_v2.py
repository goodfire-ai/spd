"""Dashboard HTML Generator Agent V2 - Freeform Evidence.

Consumes investigation.json to generate Distill.pub-style HTML reports
with model-selected figures. Dashboard data is derived dynamically from
the investigation results.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from .figure_tools import (
    escape_html_preserve_tags,
    generate_ablation_matrix,
    generate_activation_grid,
    generate_alternative_hypothesis_cards,
    generate_anomaly_box,
    generate_batch_ablation_summary,
    generate_batch_steering_summary,
    generate_boundary_test_cards,
    generate_category_selectivity_chart,
    generate_custom_visualization,
    generate_downstream_dependency_table,
    generate_downstream_steering_slope_table,
    generate_downstream_wiring_table,
    generate_evidence_card,
    generate_homograph_comparison,
    generate_hypothesis_timeline,
    generate_output_projections,
    generate_patching_comparison,
    generate_selectivity_gallery,
    generate_stacked_density_chart,
    generate_steering_curves,
    generate_steering_downstream_table,
    generate_upstream_dependency_table,
    generate_upstream_steering_table,
    generate_wiring_polarity_table,
    linkify_neuron_ids,
)
from .html_builder import (
    assemble_page,
    build_fixed_sections,
    render_collapsible,
    render_hypothesis_testing_section,
    render_input_stimuli_section,
    render_open_questions,
    render_open_questions_section,
    render_output_function_section,
)
from .tools import get_model_config, merge_selectivity_runs


def _resolve_selectivity_data(investigation: dict[str, Any]) -> dict[str, Any]:
    """Resolve category_selectivity_data from investigation, handling list or dict format.

    Returns a single merged dict suitable for chart generation.
    """
    raw = investigation.get("category_selectivity_results") or investigation.get("category_selectivity_data")
    if not raw:
        return {}
    if isinstance(raw, list):
        return merge_selectivity_runs(raw)
    if isinstance(raw, dict):
        return raw
    return {}


# =============================================================================
# NEURONDB LOOKUP
# =============================================================================

def _parse_neuron_id(nid: str) -> tuple[int, int] | None:
    """Parse neuron ID like 'L12/N8459' into (layer, neuron) tuple."""
    match = re.match(r'L(\d+)/N(\d+)', nid)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _get_labels_from_file(neuron_ids: list[str]) -> dict[str, str]:
    """Fallback: load neuron labels from local JSON file.

    Args:
        neuron_ids: List of neuron IDs like ["L12/N8459", "L21/N6856"]

    Returns:
        Dict mapping neuron_id to label
    """
    labels_path = Path(__file__).parent.parent / "data" / "neuron_labels_combined.json"
    if not labels_path.exists():
        labels_path = Path("data/neuron_labels_combined.json")

    if not labels_path.exists():
        return {}

    try:
        with open(labels_path) as f:
            data = json.load(f)

        neurons = data.get("neurons", {})
        results = {}
        # Labels to skip (not meaningful)
        skip_labels = {"uninterpretable-routing", "uninterpretable", "unknown", ""}

        for nid in neuron_ids:
            if nid in neurons:
                neuron_data = neurons[nid]
                # Try multiple label fields, skipping uninformative ones
                label = None
                for field in ["function_label", "input_label", "label"]:
                    candidate = neuron_data.get(field, "")
                    if candidate and candidate.lower() not in skip_labels:
                        label = candidate
                        break
                if label:
                    results[nid] = label
        return results
    except Exception as e:
        print(f"Labels file lookup error: {e}")
        return {}


# Cache for CSV labels (loaded once per process)
_CSV_LABEL_CACHE: dict[str, str] | None = None


def _get_labels_from_csv(neuron_ids: list[str]) -> dict[str, str]:
    """Load neuron labels from pre-extracted NeuronDB CSV export.

    Uses data/neurondb_labels.csv which has ~99.9% coverage and doesn't
    require the PostgreSQL server to be running.

    Args:
        neuron_ids: List of neuron IDs like ["L12/N8459", "L21/N6856"]

    Returns:
        Dict mapping neuron_id to label
    """
    global _CSV_LABEL_CACHE

    if _CSV_LABEL_CACHE is None:
        csv_path = Path(__file__).parent.parent / "data" / "neurondb_labels.csv"
        if not csv_path.exists():
            csv_path = Path("data/neurondb_labels.csv")

        if not csv_path.exists():
            _CSV_LABEL_CACHE = {}
        else:
            try:
                import csv
                labels = {}
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        layer = int(row['layer'])
                        neuron = int(row['neuron'])
                        desc = row.get('description', '')
                        neuron_id = f"L{layer}/N{neuron}"
                        if neuron_id not in labels and desc:
                            labels[neuron_id] = desc
                _CSV_LABEL_CACHE = labels
            except Exception as e:
                print(f"NeuronDB CSV lookup error: {e}")
                _CSV_LABEL_CACHE = {}

    return {nid: _CSV_LABEL_CACHE[nid] for nid in neuron_ids if nid in _CSV_LABEL_CACHE}


def get_neuron_labels(neuron_ids: list[str]) -> dict[str, str]:
    """Get neuron labels from available sources.

    Priority order:
    1. Local labels file (neuron_labels_combined.json) - primary source
    2. NeuronDB CSV export (data/neurondb_labels.csv) - 99.9% coverage, no DB needed
    3. NeuronDB database - fallback for neurons not in file or CSV
    4. "Unknown" - final fallback

    Args:
        neuron_ids: List of neuron IDs like ["L12/N8459", "L21/N6856"]

    Returns:
        Dict mapping neuron_id to label
    """
    if not neuron_ids:
        return {}

    # Step 1: Try labels file first (primary source)
    results = _get_labels_from_file(neuron_ids)

    # Step 2: For any missing neurons, try CSV export (fast, no DB required)
    missing = [nid for nid in neuron_ids if nid not in results]
    if missing:
        csv_labels = _get_labels_from_csv(missing)
        results.update(csv_labels)

    # Step 3: For still-missing neurons, try NeuronDB
    missing = [nid for nid in neuron_ids if nid not in results]
    if missing:
        db_labels = _get_neurondb_descriptions_only(missing)
        results.update(db_labels)

    return results


def _get_neurondb_descriptions_only(neuron_ids: list[str]) -> dict[str, str]:
    """Fetch neuron descriptions from the PostgreSQL database only.

    This is the internal function that only queries NeuronDB.
    Use get_neuron_labels() for the full lookup with file fallback.

    Args:
        neuron_ids: List of neuron IDs like ["L12/N8459", "L21/N6856"]

    Returns:
        Dict mapping neuron_id to description
    """
    if not neuron_ids:
        return {}

    # Parse neuron IDs into (layer, neuron) tuples
    layer_neuron_tuples = []
    id_map = {}  # (layer, neuron) -> original_id

    for nid in neuron_ids:
        parsed = _parse_neuron_id(nid)
        if parsed:
            layer_neuron_tuples.append(parsed)
            id_map[parsed] = nid

    if not layer_neuron_tuples:
        return {}

    try:
        # Import neurondb (requires observatory_repo setup)
        repo_path = Path(__file__).parent.parent / "observatory_repo"
        if not repo_path.exists():
            repo_path = Path("observatory_repo")

        if not repo_path.exists():
            return {}

        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        try:
            os.chdir(repo_path)

            # Load .env file and set environment variables BEFORE importing neurondb
            # This is critical because the ENV object is created on first import
            from dotenv import dotenv_values
            env_values = dotenv_values()
            for key, value in env_values.items():
                if value is not None:
                    os.environ[key] = value

            # Add lib subdirectories to path
            lib_path = repo_path / "lib"
            if lib_path.exists():
                for subdir in lib_path.iterdir():
                    if subdir.is_dir():
                        sys.path.insert(0, str(subdir))

            # Remove cached imports to force reload with new env vars
            modules_to_remove = [m for m in sys.modules if m.startswith(('neurondb', 'util.env'))]
            for m in modules_to_remove:
                del sys.modules[m]

            from neurondb.postgres import DBManager
            from neurondb.schemas import SQLANeuron, SQLANeuronDescription

            # Clear the instances cache to force new connection
            DBManager.instances.clear()

            db = DBManager.get_instance()

            results = db.get(
                [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
                joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
                layer_neuron_tuples=layer_neuron_tuples,
                timeout_ms=30000
            )

            # Build result dict
            descriptions = {}
            for row in results:
                layer, neuron, desc = row
                original_id = id_map.get((layer, neuron))
                if original_id and desc:
                    # If we already have a description for this ID, concatenate
                    if original_id in descriptions:
                        descriptions[original_id] = descriptions[original_id] + " | " + desc
                    else:
                        descriptions[original_id] = desc

            return descriptions

        finally:
            os.chdir(original_cwd)
            sys.path = original_path

    except Exception as e:
        print(f"NeuronDB lookup error: {e}")
        return {}


# Generic labels that should trigger database lookup
GENERIC_LABELS = {
    "tech-router",
    "router",
    "tech",
    "generic",
    "unknown",
    "unlabeled",
    "mlp neuron",
    "neuron",
}

# Patterns that indicate a formulaic/generic label (these are auto-generated labels
# that don't provide meaningful semantic information)
GENERIC_LABEL_PATTERNS = [
    r'tech[-\s]?identifier',
    r'tech[-\s]?delimiter',
    r'identifier[-\s/]?tech',
    r'delimiter[-\s/]?tech',
    r'routing\s*(gate|enabler|trigger|initiator|hub)',
    r'(gate|enabler|trigger|initiator)\s*routing',
    r'topic[-\s]?to[-\s]?tech',
    r'^[\w\s/-]*(router|routing)[\w\s/-]*$',  # Any label that's primarily about routing
]


def _is_generic_label(label: str) -> bool:
    """Check if a label is generic and should be replaced with database description."""
    if not label:
        return True

    normalized = label.lower().strip()

    # Check against known generic patterns
    if normalized in GENERIC_LABELS:
        return True

    # Check if it's just the neuron ID format
    if re.match(r'^l\d+/n\d+$', normalized):
        return True

    # Check against formulaic label patterns (tech-identifier, routing gate, etc.)
    for pattern in GENERIC_LABEL_PATTERNS:
        if re.search(pattern, normalized):
            return True

    # Check if it's very short and non-descriptive
    if len(normalized) < 5 and normalized.isalpha():
        return True

    return False


def _extract_wiring_summary(wiring_data: dict | None) -> dict | None:
    """Extract key information from wiring analysis for prose writing.

    The wiring analysis shows weight-based predictions of which upstream neurons
    would excite vs inhibit this neuron. This is STATIC (based on weights) and
    complements the dynamic RelP-based connectivity.

    Args:
        wiring_data: From analyze_wiring tool, contains top_excitatory, top_inhibitory, stats

    Returns:
        Summarized dict for the dashboard agent to write prose about
    """
    if not wiring_data:
        return None

    stats = wiring_data.get("stats", {})
    top_excitatory = wiring_data.get("top_excitatory", [])
    top_inhibitory = wiring_data.get("top_inhibitory", [])
    coverage = wiring_data.get("label_coverage_pct", 0)

    if not top_excitatory and not top_inhibitory:
        return None

    return {
        "summary": f"{len(top_excitatory)} excitatory, {len(top_inhibitory)} inhibitory connections analyzed ({coverage:.0f}% labeled)",
        "top_excitatory": [
            {
                "neuron_id": c.get("neuron_id"),
                "label": c.get("label", "")[:100],
                "c_combined": c.get("c_combined", 0),
                "polarity_confidence": c.get("polarity_confidence", 0),
            }
            for c in top_excitatory[:5]  # Top 5 for prose
        ],
        "top_inhibitory": [
            {
                "neuron_id": c.get("neuron_id"),
                "label": c.get("label", "")[:100],
                "c_combined": c.get("c_combined", 0),
                "polarity_confidence": c.get("polarity_confidence", 0),
            }
            for c in top_inhibitory[:5]  # Top 5 for prose
        ],
        "interpretation_note": (
            "Weight-based wiring predicts which upstream neurons would INCREASE (excitatory) "
            "or DECREASE (inhibitory) this neuron's activation. Compare with RelP connectivity "
            "which shows actual influence in specific prompts."
        ),
        "operating_regime": stats.get("target_regime"),
        "regime_correction_applied": stats.get("regime_correction_applied", False),
    }


def _extract_skeptic_summary(skeptic_report: dict | None) -> dict | None:
    """Extract key information from skeptic report for dashboard display.

    Args:
        skeptic_report: Full skeptic report dict or None

    Returns:
        Summarized dict with key adversarial findings, or None if not available
    """
    if not skeptic_report:
        return None

    # Extract metrics
    metrics = skeptic_report.get("metrics", {})

    # Extract boundary test summary
    boundary_tests = skeptic_report.get("boundary_tests", [])
    passed = sum(1 for t in boundary_tests if t.get("passed", False))
    failed = len(boundary_tests) - passed

    # Get notable failures (false positives/negatives)
    notable_failures = [
        {
            "description": t.get("description", ""),
            "prompt": t.get("prompt", "")[:100],
            "expected": t.get("expected_behavior", ""),
            "actual": t.get("actual_activation", 0),
        }
        for t in boundary_tests
        if not t.get("passed", False)
    ][:5]  # Limit to 5 most notable

    # Extract alternative hypotheses
    alternatives = skeptic_report.get("alternative_hypotheses", [])
    alternatives_summary = [
        {
            "alternative": a.get("alternative", ""),
            "verdict": a.get("verdict", ""),
            "evidence": a.get("evidence", "")[:200],
        }
        for a in alternatives
    ][:5]

    # Extract confounds
    confounds = skeptic_report.get("confounds", [])
    confounds_summary = [
        {
            "factor": c.get("factor", ""),
            "description": c.get("description", ""),
            "severity": c.get("severity", ""),
        }
        for c in confounds
    ]

    return {
        "verdict": skeptic_report.get("verdict", ""),
        "confidence_adjustment": skeptic_report.get("confidence_adjustment", 0),
        "revised_hypothesis": skeptic_report.get("revised_hypothesis"),
        "metrics": {
            "selectivity_score": metrics.get("selectivity_score", 0),
            "false_positive_rate": metrics.get("false_positive_rate", 0),
            "false_negative_rate": metrics.get("false_negative_rate", 0),
        },
        "boundary_tests": {
            "total": len(boundary_tests),
            "passed": passed,
            "failed": failed,
            "notable_failures": notable_failures,
        },
        "alternatives_tested": alternatives_summary,
        "confounds": confounds_summary,
        "key_challenges": skeptic_report.get("key_challenges", [])[:5],
        "recommendations": skeptic_report.get("recommendations", [])[:3],
        "total_tests": skeptic_report.get("total_tests", 0),
    }


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a science writer creating Distill.pub-style HTML reports for neural network interpretability research.

## Context

You're documenting investigations of **MLP neurons** in **{model_name}** ({num_layers} layers, {neurons_per_layer} neurons/layer).

Each neuron has:
- **Input function**: What patterns trigger activation?
- **Output function**: What does activation promote/suppress?

## Your Task

Write a compelling scientific article about this neuron. You have **complete creative freedom** over the prose—write naturally, tell the story of the investigation, and explain what makes this neuron interesting.

## CRITICAL: Use the Scientist's Characterization

The `get_full_data` tool returns a `characterization` object with `final_hypothesis`, `input_function`, `output_function`, and `key_findings`. These represent the **authoritative conclusions from a multi-hour investigation** by a neuron scientist agent who ran dozens of experiments. Your job is to **explain and narrate these findings**, not to re-derive your own interpretation from raw data. The scientist's characterization is the ground truth for this report. If the raw selectivity z-scores or output projections seem to conflict with the characterization, trust the characterization — the scientist had access to far more context including steering experiments, ablation results, and upstream/downstream causal analysis.

## CRITICAL: Data Source

**Use ONLY the `get_full_data` MCP tool for investigation data.** Do NOT use Read, Glob, or Bash to browse the filesystem for other investigation files, neuron reports, or JSON data. All the data you need is provided through the MCP tools. Reading other files risks contaminating this report with data from a different neuron.

## CRITICAL: Page Structure

The dashboard has a FIXED STRUCTURE with auto-generated figures at specific locations. You can insert prose BEFORE or AFTER any auto-generated figure. Plan your commentary around the data tables.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HEADER (auto-generated: neuron ID, title, confidence badge)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  NARRATIVE LEAD + BODY (your title and intro prose)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  CIRCUIT DIAGRAM (auto-generated: upstream → neuron → downstream)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ INPUT FUNCTION SECTION ─────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ═══ PART 1: BEHAVIORAL TRIGGERS ═══                                  │   │
│  │                                                                       │   │
│  │  [prose_before_selectivity] ← optional intro                          │   │
│  │  Category Selectivity Chart ← AUTO (title: "Category Selectivity")    │   │
│  │  [prose_after_selectivity] ← explain what triggers this neuron        │   │
│  │  [custom figures via <FIGURE_N>] ← optional                           │   │
│  │  [prose_after_other_figures] ← optional                               │   │
│  │                                                                       │   │
│  │  ═══ PART 2: UPSTREAM CIRCUIT ARCHITECTURE ═══                        │   │
│  │                                                                       │   │
│  │  [prose_before_wiring / prose_part2] ← intro to circuit architecture  │   │
│  │  Upstream Wiring Table ← AUTO (title: "Upstream Wiring")              │   │
│  │  [prose_after_wiring] ← interpret wiring predictions                  │   │
│  │  [prose_before_ablation] ← optional                                   │   │
│  │  Upstream Ablation Table ← AUTO (title: "Upstream Ablation Effects")  │   │
│  │  [prose_after_ablation] ← interpret ablation results                  │   │
│  │  [prose_before_steering] ← optional                                   │   │
│  │  Upstream Steering Table ← AUTO (if data exists)                      │   │
│  │  [prose_after_steering] ← interpret steering results                  │   │
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ OUTPUT FUNCTION SECTION ────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ═══ PART 1: DIRECT TOKEN EFFECTS ═══                                 │   │
│  │                                                                       │   │
│  │  [prose_before_projections] ← optional intro                          │   │
│  │  Output Projections ← AUTO (title: "Output Projections")              │   │
│  │  [prose_after_projections / prose_part1] ← interpret projections      │   │
│  │  [prose_before_ablation] ← optional                                   │   │
│  │  Ablation Completions ← AUTO (title: "Ablation Effects on...")        │   │
│  │  [prose_after_ablation] ← interpret ablation behavioral effects       │   │
│  │  [prose_before_steering] ← optional                                   │   │
│  │  Steering Completions ← AUTO (title: "Intelligent Steering...")       │   │
│  │  [prose_after_steering] ← interpret steering behavioral effects       │   │
│  │                                                                       │   │
│  │  ═══ PART 2: DOWNSTREAM CIRCUIT EFFECTS ═══                           │   │
│  │                                                                       │   │
│  │  [prose_before_downstream_wiring / prose_part2] ← intro to circuit    │   │
│  │  Downstream Wiring Table ← AUTO (title: "Downstream Wiring")          │   │
│  │  [prose_after_downstream_wiring] ← interpret wiring predictions       │   │
│  │  [prose_before_downstream_ablation] ← optional                        │   │
│  │  Downstream Ablation Effects ← AUTO (title: "Downstream Ablation...")  │   │
│  │  [prose_after_downstream_ablation] ← interpret downstream ablation    │   │
│  │  [prose_before_downstream_steering] ← optional                        │   │
│  │  Downstream Steering Response ← AUTO (slope + R² per downstream)      │   │
│  │  [prose_after_downstream_steering] ← interpret downstream steering    │   │
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ HYPOTHESIS TESTING SECTION ─────────────────────────────────────────┐   │
│  │  Hypothesis Timeline ← AUTO-GENERATED                                 │   │
│  │  [hypothesis_testing.prose] ← YOUR PROSE                              │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ OPEN QUESTIONS SECTION ─────────────────────────────────────────────┐   │
│  │  [open_questions.prose] ← YOUR PROSE                                  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CRITICAL: Prose Placement Strategy

You have FULL FLEXIBILITY to place prose before or after ANY auto-generated figure. The table titles above (in quotes) are auto-generated—DO NOT duplicate these titles in your prose.

**Strategy:**
1. Review `preview_visualizations` to see what data will appear in each table
2. Plan where to insert commentary—before tables to set context, after tables to interpret
3. You can write prose for ANY slot—use as many or as few as makes sense for the story
4. At minimum, write: `prose` or `prose_after_selectivity` (Part 1), `prose_part2` (Part 2 for both input/output)

**DO NOT:**
- Duplicate auto-generated titles (e.g., don't write "## Upstream Wiring" before the wiring table)
- Embed figures INSIDE your prose text
- Mix topics across sections

## Input Function: BOTH Parts Required

### Part 1: Behavioral Triggers (what activates this neuron)
- `prose_after_selectivity` — Interpret the category selectivity chart. What contexts/tokens trigger this neuron?
  **IMPORTANT**: Your prose MUST be grounded in the selectivity chart data shown above.
  Reference specific category z-scores from the chart. Do NOT describe examples that aren't
  visible in the chart. If you reference prompts from outside the selectivity data (e.g.,
  from the scientist's manual test_activation probes), explicitly note their source and
  generate an additional figure to show them.

### Part 2: Upstream Circuit Architecture (where does input come from)
- `prose_part2` or `prose_before_wiring` — Introduce the circuit analysis
- `prose_after_wiring` — Interpret the wiring predictions
- `prose_after_ablation` — Interpret the ablation results (which upstream neurons actually matter?)

## Output Function: BOTH Parts Required

### Part 1: Direct Token Effects (behavioral impact)
- `prose_part1` or `prose_after_projections` — What tokens does this neuron promote/suppress?
- `prose_after_ablation` — How does ablating change completions?
- `prose_after_steering` — How does steering change completions?

### Part 2: Downstream Circuit Effects (circuit routing role)
- `prose_part2` or `prose_before_downstream_wiring` — Introduce the downstream circuit role
- `prose_after_downstream_wiring` — Interpret wiring predictions
- `prose_after_downstream_ablation` — How does ablating THIS neuron affect OTHER neurons?

**IMPORTANT:** The auto-generated tables show DATA. Your prose provides INTERPRETATION. Explain what the data means, why certain patterns emerge, and what it tells us about the neuron's function.

## CRITICAL: Two-Pass Workflow

**ALWAYS start by calling `preview_visualizations`** before writing any prose. This tool:
1. Pre-generates all auto-visualizations and tells you what will appear
2. Returns summaries so you know what each figure shows
3. Lets you write prose that references the correct figures

**Workflow:**
1. Call `get_full_data` to load the investigation data
2. Call `preview_visualizations` to see all auto-generated tables/charts
3. Read the summaries to understand what visuals will appear WHERE
4. Write your prose for each section, referencing those visualizations
5. Call `write_dashboard` with section_content

## Auto-Generated Figures (DO NOT RECREATE)

These figures are AUTOMATICALLY included at their designated positions:

**Input Function Section:**
- Category selectivity chart
- Upstream wiring polarity table (excitatory/inhibitory predictions)
- Upstream ablation dependency table

**Output Function Section:**
- Output projections (promotes/suppresses)
- Ablation changed completions (before/after examples)
- Intelligent steering gallery (if intelligent_steering_analysis was run)
- Downstream wiring polarity table
- Downstream ablation effects table
- Downstream steering effects table

**Hypothesis Testing Section:**
- Hypothesis timeline (evolution of hypotheses)

**You do NOT need to generate these figures—they appear automatically.**

## Custom Figures (Optional)

You can ADD extra figures using `<FIGURE_N>` placeholders in your prose:

- `generate_selectivity_gallery` — Show category-specific examples
- `generate_evidence_card` — Highlight key findings
- `generate_anomaly_box` — Call out surprises (check `anomaly_investigation` in data)
- `generate_custom_visualization` — Custom charts/tables

**IMPORTANT: Anomaly Investigation Data**
If `data_availability.has_anomaly_investigation` is true, the investigation includes anomalies the scientist identified and investigated. Use `generate_anomaly_box` to highlight the most interesting ones! The `anomaly_investigation` object contains:
- `anomalies_identified`: List of all anomalies found
- `anomalies_investigated`: Details on each investigation with `anomaly`, `explanation`, `experiments_run`, and `confidence`

Generate the figure first, then place `<FIGURE_0>` etc. in your prose where you want it.

## Formatting Your Prose

Use markdown (it will be converted to HTML):

- **## Headings** for major topics
- **### Subheadings** for sub-topics
- **Blank lines** between paragraphs
- **Bold** with `**text**`
- **Bullet lists** with `-` or `*`

**GOOD structure:**
```
## What Activates This Neuron

The neuron shows remarkable selectivity for NSAID medications. The category
selectivity chart above reveals strong activation (z > 2) on COX inhibitor
contexts...

## Upstream Circuit

The wiring analysis shows L5/N5772 as the primary excitatory driver...
```

**BAD (mixing topics, embedding figures in prose):**
```
The neuron shows selectivity and here's a figure [FIGURE_0] and the
downstream effects show X and the ablation results Y...
```

## Technical Reference

**Wiring Analysis**: Weight-based connectivity showing what COULD influence this neuron.
- **Excitatory**: Would increase target activation
- **Inhibitory**: Would decrease target activation
- If `operating_regime` is "inverted", the neuron operates in an unusual SwiGLU regime where gate and up channels are both negative at peak activation. Polarity labels have been regime-corrected in this case.
- If `regime_correction_applied` is true, mention this in the prose to explain any sign-agreement improvements.

**Ablation**: Zero out a neuron, measure effect on outputs.

**Steering**: Amplify/suppress a neuron's activation, measure effect.

**Direct Effect Ratio**: Fraction of influence from direct logit projections vs. downstream routing.

## write_dashboard Parameters

```
section_content = {{
    "input_function": {{
        "prose_after_selectivity": "Part 1 prose: what triggers this neuron",
        "prose_part2": "Part 2 intro: upstream circuit architecture",
        "prose_after_wiring": "Interpret the wiring predictions",
        "prose_after_ablation": "Interpret the upstream ablation results",
        "prose_after_steering": "Interpret the upstream steering dose-response",
        "prose_part3": "(BIPOLAR ONLY) Part 3: Negative Firing Triggers — what makes this neuron fire negatively"
    }},
    "output_function": {{
        "prose_part1": "Part 1 prose: direct token effects and behavioral impact",
        "prose_after_ablation": "Interpret ablation behavioral effects",
        "prose_after_steering": "Interpret steering behavioral effects",
        "prose_part2": "Part 2 intro: downstream circuit effects",
        "prose_after_downstream_ablation": "Interpret downstream ablation results",
        "prose_after_downstream_steering": "Interpret downstream steering dose-response",
        "prose_part3": "(BIPOLAR ONLY) Part 3: Negative Firing Effects — what happens when this neuron fires negatively"
    }},
    "hypothesis_testing": {{"prose": "Your hypothesis testing prose here"}},
    "open_questions": {{"prose": "Your open questions prose here"}}
}}
```

**Simplified usage:** You can use just `prose` and `prose_part2` for each section, or use the granular slots for finer control.

**Key points:**
1. Write SEPARATE prose for each section
2. For output_function, write BOTH `prose` (Part 1: behavioral) AND `prose_part2` (Part 2: circuit routing)
3. DO NOT include figures in the prose dict—use `figures: [0, 1]` only for custom figures you generated
4. Auto-generated figures (selectivity, wiring, ablation, steering) appear automatically

Make it a compelling scientific read.
"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class DashboardAgentV2:
    """Agent that generates HTML dashboards from investigation.json."""

    def __init__(
        self,
        investigation_path: Path,
        output_dir: Path = Path("frontend/reports"),
        model: str = "sonnet",
        # Legacy parameter for backward compatibility
        dashboard_path: Path | None = None,
        negative_investigation_path: Path | None = None,
    ):
        """Initialize the dashboard generator.

        Args:
            investigation_path: Path to investigation JSON file (primary input)
            output_dir: Directory to write output HTML
            model: Claude model to use ("opus", "sonnet", or "haiku")
            dashboard_path: DEPRECATED - ignored, kept for backward compatibility
            negative_investigation_path: Optional path to negative-polarity investigation JSON.
                When provided, the dashboard will include both positive and negative sections.
        """
        self.investigation_path = Path(investigation_path)
        self.output_dir = Path(output_dir)
        self.model = model
        self.negative_investigation_path = Path(negative_investigation_path) if negative_investigation_path else None

        # Legacy compatibility: if dashboard_path passed as first positional arg
        if dashboard_path is not None:
            print("  Note: dashboard_path is deprecated, using investigation_path")

        self.investigation_data = None
        self.negative_investigation_data = None  # Loaded when negative_investigation_path is set
        self.generated_figures = []
        self.auto_figures = {}  # Auto-generated figures by type
        self.preview_figures = {}  # Pre-generated visualizations from Pass 1
        self.preview_summaries = {}  # Textual summaries for agent visibility
        self.preview_done = False  # Track if preview has been run
        self.enriched_connectivity = None  # Set by get_full_data

    def add_auto_figure(self, figure_type: str, html: str) -> None:
        """Add an auto-generated figure that will be included in the dashboard.

        Args:
            figure_type: Type identifier for the figure (e.g., 'wiring_polarity_table')
            html: The HTML content for the figure
        """
        self.auto_figures[figure_type] = html

    def _load_data(self) -> None:
        """Load investigation JSON and derive dashboard data."""
        with open(self.investigation_path) as f:
            self.investigation_data = json.load(f)

        # Load negative polarity investigation if provided
        if self.negative_investigation_path and self.negative_investigation_path.exists():
            with open(self.negative_investigation_path) as f:
                self.negative_investigation_data = json.load(f)
            print(f"  Loaded negative investigation: {self.negative_investigation_path.name}")

    def _create_mcp_tools(self):
        """Create MCP tools for the agent."""
        investigation = self.investigation_data
        output_dir = self.output_dir
        agent = self

        # =================================================================
        # DATA TOOL
        # =================================================================

        @tool(
            "get_full_data",
            "Get investigation data for the neuron. Automatically enriches connectivity labels from NeuronDB and extracts connectivity from RelP when not available.",
            {}
        )
        async def tool_get_full_data(args: dict[str, Any]) -> dict[str, Any]:
            """Return data from investigation.json."""
            # Extract from investigation (consolidated format)
            characterization = investigation.get("characterization", {})
            evidence = investigation.get("evidence", {})

            # Output projections (now stored directly in investigation)
            output_projections = investigation.get("output_projections", {"promote": [], "suppress": []})

            # Connectivity from evidence
            connectivity_raw = evidence.get("connectivity", investigation.get("connectivity", {}))

            # RelP results
            relp_results = investigation.get("relp_results", evidence.get("relp_results", []))

            # Get source neuron's layer for filtering downstream
            source_layer = investigation.get("layer", 0)

            # If connectivity is empty, extract from RelP analysis
            upstream_raw = connectivity_raw.get("upstream_neurons", connectivity_raw.get("upstream", []))
            downstream_raw = connectivity_raw.get("downstream_targets", connectivity_raw.get("downstream", []))

            # Filter existing downstream to only include neurons from later layers
            if downstream_raw and source_layer > 0:
                filtered_downstream = []
                for n in downstream_raw:
                    nid = n.get("neuron_id", "")
                    if nid.startswith("L") and "/" in nid:
                        try:
                            target_layer = int(nid.split("/")[0][1:])
                            if target_layer > source_layer:
                                filtered_downstream.append(n)
                        except ValueError:
                            filtered_downstream.append(n)  # Keep if can't parse
                downstream_raw = filtered_downstream

            if not upstream_raw or not downstream_raw:
                # Extract from RelP results
                if relp_results:
                    # Aggregate across all RelP runs
                    upstream_map = {}  # neuron_id -> max weight
                    downstream_map = {}

                    for result in relp_results:
                        for edge in result.get("upstream_edges", []):
                            source_info = edge.get("source_info", {})
                            if source_info.get("type") == "mlp_neuron":
                                nid = f"L{source_info.get('layer')}/N{source_info.get('feature')}"
                                weight = edge.get("weight", 0)
                                if nid not in upstream_map or weight > upstream_map[nid]:
                                    upstream_map[nid] = weight

                        for edge in result.get("downstream_edges", []):
                            target_info = edge.get("target_info", {})
                            if target_info.get("type") == "mlp_neuron":
                                target_layer = target_info.get("layer", 0)
                                # Only include downstream neurons from LATER layers
                                if target_layer > source_layer:
                                    nid = f"L{target_layer}/N{target_info.get('feature')}"
                                    weight = edge.get("weight", 0)
                                    if nid not in downstream_map or weight > downstream_map[nid]:
                                        downstream_map[nid] = weight

                    # Convert to lists: select top 8 positive + top 8 negative by absolute weight
                    def select_top_by_sign(items_map, n_pos=8, n_neg=8):
                        """Select top positive and negative items by absolute weight."""
                        items = [(nid, w) for nid, w in items_map.items()]
                        positive = [(nid, w) for nid, w in items if w > 0]
                        negative = [(nid, w) for nid, w in items if w < 0]
                        # Sort by absolute weight descending
                        positive_top = sorted(positive, key=lambda x: abs(x[1]), reverse=True)[:n_pos]
                        negative_top = sorted(negative, key=lambda x: abs(x[1]), reverse=True)[:n_neg]
                        return [
                            {"neuron_id": nid, "label": "", "weight": w}
                            for nid, w in positive_top + negative_top
                        ]

                    if not upstream_raw:
                        upstream_raw = select_top_by_sign(upstream_map, n_pos=8, n_neg=8)
                    if not downstream_raw:
                        downstream_raw = select_top_by_sign(downstream_map, n_pos=8, n_neg=8)

            # Auto-enrich connectivity labels from NeuronDB
            neurons_to_lookup = []
            for n in upstream_raw + downstream_raw:
                label = n.get("label", "")
                if _is_generic_label(label):
                    nid = n.get("neuron_id", "")
                    if nid:
                        neurons_to_lookup.append(nid)

            # Fetch descriptions for neurons with generic labels
            enriched_labels = {}
            if neurons_to_lookup:
                enriched_labels = get_neuron_labels(neurons_to_lookup)

            # Store enriched connectivity for use by write_dashboard
            agent.enriched_connectivity = {
                "upstream": [],
                "downstream": [],
            }

            # Process up to 16 (8 positive + 8 negative) for each direction
            for n in upstream_raw[:16]:
                nid = n.get("neuron_id", "")
                original_label = n.get("label", "Unknown")
                # Use enriched label if original is generic
                if _is_generic_label(original_label) and nid in enriched_labels:
                    label = enriched_labels[nid]
                    # Truncate long descriptions
                    if len(label) > 80:
                        label = label[:77] + "..."
                else:
                    label = original_label if original_label else "Unknown"
                agent.enriched_connectivity["upstream"].append({
                    "id": nid,
                    "label": label,
                    "weight": n.get("weight", 0),
                })

            for n in downstream_raw[:16]:
                nid = n.get("neuron_id", "")
                original_label = n.get("label", "Unknown")
                # Use enriched label if original is generic
                if _is_generic_label(original_label) and nid in enriched_labels:
                    label = enriched_labels[nid]
                    # Truncate long descriptions
                    if len(label) > 80:
                        label = label[:77] + "..."
                else:
                    label = original_label if original_label else "Unknown"
                agent.enriched_connectivity["downstream"].append({
                    "id": nid,
                    "label": label,
                    "weight": n.get("weight", 0),
                })

            # Get activation examples from evidence
            hypotheses_tested = investigation.get("hypotheses_tested", [])
            positive = evidence.get("activating_prompts", [])
            negative = evidence.get("non_activating_prompts", [])

            # Get prior claims (seed hypotheses) if available
            prior_claims = investigation.get("prior_claims", {})

            # Key findings from the scientist (list of strings)
            key_findings = investigation.get("key_findings", [])
            key_findings_text = "\n".join(f"- {f}" for f in key_findings[:5]) if key_findings else ""

            data = {
                # ============================================================
                # SCIENTIST'S CONCLUSIONS (use these as the basis for your article)
                # ============================================================
                "SCIENTIST_CONCLUSION": characterization.get("final_hypothesis", ""),
                "SCIENTIST_INPUT_FUNCTION": characterization.get("input_function", ""),
                "SCIENTIST_OUTPUT_FUNCTION": characterization.get("output_function", ""),
                "SCIENTIST_KEY_FINDINGS": key_findings_text,

                "neuron_id": investigation.get("neuron_id", ""),
                "layer": investigation.get("layer", 0),

                # Characterization (from investigation)
                "characterization": {
                    "input_function": characterization.get("input_function", ""),
                    "output_function": characterization.get("output_function", ""),
                    "function_type": characterization.get("function_type", ""),
                    "final_hypothesis": characterization.get("final_hypothesis", ""),
                },

                # Prior claims (seed hypotheses from LLM labels)
                "prior_claims": prior_claims,

                # Summary stats
                "summary": {
                    "confidence": investigation.get("confidence", 0),
                    "total_experiments": investigation.get("total_experiments", 0),
                },

                # Output projections
                "output_projections": {
                    "promote": output_projections.get("promote", [])[:15],
                    "suppress": output_projections.get("suppress", [])[:15],
                },

                # Activation examples
                "activation_examples": {
                    "positive": [
                        {"prompt": ex.get("prompt", ""), "activation": ex.get("activation", 0), "token": ex.get("token", "")}
                        for ex in positive[:15]
                    ],
                    "negative": [
                        {"prompt": ex.get("prompt", ""), "activation": ex.get("activation", 0)}
                        for ex in negative[:10]
                    ],
                },

                # Hypotheses with full evidence
                "hypotheses": [
                    {
                        "id": h.get("hypothesis_id", f"H{i}"),
                        "text": h.get("hypothesis", "")[:200],
                        "status": h.get("status", "testing"),
                        "prior": h.get("prior_probability", 50),
                        "posterior": h.get("posterior_probability", 50),
                        "evidence": h.get("evidence", [])[:3],
                        "type": h.get("hypothesis_type", ""),
                    }
                    for i, h in enumerate(hypotheses_tested)
                ],

                # Experiments
                # NOTE: Prefer multi_token_ablation_results (batch data) over evidence.ablation_effects (legacy)
                "experiments": {
                    "ablation": evidence.get("ablation_effects", [])[:10],
                    "steering": investigation.get("dose_response_results") or investigation.get("steering_results") or evidence.get("steering_results", [])[:10],
                },

                # Batch ablation summary (multi_token_ablation_results) - this is the PRIMARY ablation data
                "batch_ablation_summary": investigation.get("multi_token_ablation_results", []),

                # RelP analysis - summarize corpus vs live counts (handle malformed data)
                "relp_analysis": {
                    "corpus_relp_count": sum(1 for r in relp_results if isinstance(r, dict) and r.get("source") == "corpus"),
                    "live_relp_count": sum(1 for r in relp_results if isinstance(r, dict) and r.get("source") != "corpus"),
                    "total_found": sum(1 for r in relp_results if isinstance(r, dict) and r.get("neuron_found")),
                    "sample_results": relp_results[:5],
                    "note": "corpus_relp_count shows how many pre-computed graphs contain this neuron (strong causal evidence)"
                },

                # Connectivity (with enriched labels from NeuronDB)
                "connectivity": agent.enriched_connectivity,

                # Wiring analysis - weight-based upstream excitatory/inhibitory neurons
                # This is DIFFERENT from connectivity - wiring shows predicted polarity based on weights
                "wiring_analysis": _extract_wiring_summary(investigation.get("wiring_analysis", {})),

                # Findings
                "findings": {
                    "key": investigation.get("key_findings", [])[:8],
                    "open_questions": investigation.get("open_questions", [])[:5],
                },

                # Skeptic adversarial testing (if available)
                "skeptic_report": _extract_skeptic_summary(investigation.get("skeptic_report")),

                # Visualization data (for specialized figures)
                "categorized_prompts": investigation.get("categorized_prompts", {}),
                "homograph_tests": investigation.get("homograph_tests", []),

                # Protocol validation metrics (for confidence display)
                "protocol_validation": investigation.get("protocol_validation", {}),

                # Category selectivity data (for chart) - merged from all runs
                "category_selectivity_data": _resolve_selectivity_data(investigation),

                # Dependency results (for upstream/downstream tables)
                "upstream_dependency_results": investigation.get("upstream_dependency_results", []),
                "downstream_dependency_results": investigation.get("downstream_dependency_results", []),

                # Multi-token results
                "multi_token_ablation_results": investigation.get("multi_token_ablation_results", []),
                "multi_token_steering_results": investigation.get("multi_token_steering_results", []),

                # Dose-response results (for steering curves)
                "dose_response_results": investigation.get("dose_response_results", []),

                # Transcript summaries - narrative summaries from each agent (scientist, skeptic, gpt_reviewer)
                # Use these to understand the STORY of the investigation
                "transcript_summaries": investigation.get("transcript_summaries", []),

                # Anomaly investigation - anomalies identified and investigated during V5 phase
                # Use generate_anomaly_box to highlight these findings
                "anomaly_investigation": investigation.get("anomaly_investigation", {}),

                # Data availability summary (helps agent know what figures to generate)
                "data_availability": {
                    "has_transcript_summaries": bool(investigation.get("transcript_summaries")),
                    "has_category_selectivity": bool(_resolve_selectivity_data(investigation).get("categories")),
                    "has_upstream_dependency": bool(investigation.get("upstream_dependency_results")),
                    "has_downstream_dependency": bool(investigation.get("downstream_dependency_results")),
                    "has_steering_downstream": any(
                        r.get("downstream_effects") for r in investigation.get("multi_token_steering_results", []) if isinstance(r, dict)
                    ),
                    "has_dose_response": bool(investigation.get("dose_response_results")),
                    "has_multi_token_ablation": bool(investigation.get("multi_token_ablation_results")),
                    "has_skeptic_report": investigation.get("skeptic_report") is not None,
                    "has_boundary_tests": bool(investigation.get("skeptic_report", {}).get("boundary_tests")),
                    "has_alternative_hypotheses": bool(investigation.get("skeptic_report", {}).get("alternative_hypotheses")),
                    "has_open_questions": bool(investigation.get("open_questions")),
                    "has_anomaly_investigation": bool(investigation.get("anomaly_investigation", {}).get("anomalies_investigated")),
                    "has_wiring_analysis": bool(investigation.get("wiring_analysis", {}).get("top_excitatory")),
                    # RelP corpus validation - used for corroborating weight-based predictions
                    "has_relp_corpus": any(
                        r.get("source") == "corpus" for r in relp_results if isinstance(r, dict)
                    ),
                    "relp_corpus_count": sum(
                        1 for r in relp_results if isinstance(r, dict) and r.get("source") == "corpus"
                    ),
                    "n_activating_prompts": len(evidence.get("activating_prompts", [])),
                    "n_hypotheses_tested": len(investigation.get("hypotheses_tested", [])),
                },
            }

            return {
                "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
            }

        # =================================================================
        # PREVIEW VISUALIZATIONS TOOL (Two-Pass Generation)
        # =================================================================

        def _generate_preview_figures(investigation: dict[str, Any]) -> dict[str, str]:
            """Internal helper to generate preview figures. Returns dict of figure name -> HTML.

            This is the core logic used by both preview_visualizations tool and write_dashboard.
            """
            evidence = investigation.get("evidence", {})
            neuron_id = investigation.get("neuron_id", "L0/N0")
            figures = {}

            # =====================================================
            # OUTPUT FUNCTION SECTION - Ablation Changed Completions
            # =====================================================
            ablation_results = investigation.get("multi_token_ablation_results", [])
            batch_ablation_runs = [r for r in ablation_results if isinstance(r, dict) and r.get("type") == "batch"]
            if batch_ablation_runs:
                try:
                    from neuron_scientist.figure_tools import generate_ablation_cards
                    html = generate_ablation_cards(
                        ablation_results,
                        title="Ablation Effects on Completions",
                    )
                    if html:
                        figures["ablation_completions"] = html
                except Exception as e:
                    print(f"  [Preview helper] ablation completions failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Intelligent Steering Gallery
            # =====================================================
            steering_results = investigation.get("multi_token_steering_results", [])
            intelligent_steering_runs = [r for r in steering_results if isinstance(r, dict) and r.get("type") == "intelligent_steering"]
            if intelligent_steering_runs:
                try:
                    from neuron_scientist.figure_tools import generate_intelligent_steering_gallery
                    html = generate_intelligent_steering_gallery(
                        steering_results,
                        title="Intelligent Steering Analysis",
                    )
                    if html:
                        figures["steering_completions"] = html
                except Exception as e:
                    print(f"  [Preview helper] steering completions failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Ablation Effects Table
            # =====================================================
            if batch_ablation_runs and any(r.get("dependency_summary") for r in batch_ablation_runs):
                try:
                    from neuron_scientist.figure_tools import generate_downstream_ablation_effects
                    # Collect neuron IDs from dependency_summary and fetch labels
                    downstream_neuron_ids = []
                    for run in batch_ablation_runs:
                        if run.get("dependency_summary"):
                            downstream_neuron_ids.extend(run["dependency_summary"].keys())
                    neuron_labels = get_neuron_labels(list(set(downstream_neuron_ids))) if downstream_neuron_ids else {}

                    html = generate_downstream_ablation_effects(
                        ablation_results,
                        neuron_labels=neuron_labels,
                        title="Downstream Ablation Effects",
                    )
                    if html:
                        figures["downstream_ablation_effects"] = html
                except Exception as e:
                    print(f"  [Preview helper] downstream ablation effects failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Steering Response (Slope Table)
            # =====================================================
            steering_with_slopes = [r for r in intelligent_steering_runs if r.get("downstream_steering_slopes")]
            if steering_with_slopes:
                try:
                    from neuron_scientist.figure_tools import (
                        generate_downstream_steering_slope_table,
                    )
                    sr = steering_with_slopes[0]
                    slopes_data = sr["downstream_steering_slopes"]
                    # Collect neuron IDs and fetch labels
                    ds_neuron_ids = list(slopes_data.keys())
                    steering_labels = get_neuron_labels(ds_neuron_ids) if ds_neuron_ids else {}

                    sv_tested = sr.get("steering_values", [])
                    n_prompts = sr.get("total_prompts", 0)
                    html = generate_downstream_steering_slope_table(
                        slopes_data,
                        neuron_labels=steering_labels,
                        title="Downstream Steering Response",
                        steering_values_tested=[sv for sv in sv_tested if sv != 0],
                        n_prompts=n_prompts,
                    )
                    if html:
                        figures["downstream_steering_slopes"] = html
                except Exception as e:
                    print(f"  [Preview helper] downstream steering slopes failed: {e}")

            return figures

        @tool(
            "preview_visualizations",
            "REQUIRED FIRST STEP: Generate all auto-visualizations and get summaries before writing prose. "
            "This shows you what tables/charts will appear in each section so you can write prose that references them.",
            {}
        )
        async def tool_preview_visualizations(args: dict[str, Any]) -> dict[str, Any]:
            """Generate all auto-visualizations and return summaries for agent visibility."""
            investigation = agent.investigation_data or {}
            evidence = investigation.get("evidence", {})
            neuron_id = investigation.get("neuron_id", "L0/N0")
            summaries = {}

            # =====================================================
            # INPUT STIMULI SECTION - Category Selectivity Chart
            # =====================================================
            selectivity_data = _resolve_selectivity_data(investigation)
            if selectivity_data:
                try:
                    html = generate_category_selectivity_chart(
                        selectivity_data,
                        neuron_id,
                        title="Category Selectivity Analysis",
                        caption="Activation levels across semantic categories"
                    )
                    agent.preview_figures["category_selectivity_chart"] = html

                    # Build rich summary with full category list and both positive/negative z-scores
                    categories = selectivity_data.get("categories", {})
                    if isinstance(categories, dict):
                        cat_items = list(categories.items())
                    else:
                        cat_items = [(c.get("category", "?"), c) for c in categories]

                    # Sort by positive z-mean (descending) for top activating
                    sorted_by_pos = sorted(cat_items, key=lambda x: x[1].get("z_mean", 0) if isinstance(x[1], dict) else 0, reverse=True)
                    # Sort by negative z-mean (ascending = most negative first)
                    sorted_by_neg = sorted(cat_items, key=lambda x: x[1].get("neg_z_mean", 0) if isinstance(x[1], dict) else 0)

                    # Build per-category stats table for the agent
                    cat_lines = []
                    cat_lines.append("POSITIVE z-score ranking (what triggers positive firing):")
                    for name, data in sorted_by_pos[:10]:
                        if isinstance(data, dict):
                            z = data.get("z_mean", 0)
                            mean = data.get("mean", 0)
                            cat_type = data.get("type", "?")
                            cat_lines.append(f"  {z:+.2f}σ  mean={mean:.3f}  {name} ({cat_type})")

                    cat_lines.append("")
                    cat_lines.append("NEGATIVE z-score ranking (what triggers negative firing):")
                    for name, data in sorted_by_neg[:10]:
                        if isinstance(data, dict):
                            neg_z = data.get("neg_z_mean", 0)
                            neg_mean = data.get("neg_mean", 0)
                            cat_type = data.get("type", "?")
                            cat_lines.append(f"  {neg_z:+.2f}σ  neg_mean={neg_mean:.3f}  {name} ({cat_type})")

                    global_mean = selectivity_data.get("global_mean", 0)
                    global_std = selectivity_data.get("global_std", 0)
                    neg_global_mean = selectivity_data.get("neg_global_mean", 0)
                    polarity_summary = selectivity_data.get("polarity_summary", "")

                    summary_content = (
                        f"Category selectivity across {len(cat_items)} categories. "
                        f"Global: mean={global_mean:.3f}, std={global_std:.3f}. "
                        f"Neg global mean={neg_global_mean:.3f}.\n"
                        f"{polarity_summary}\n\n"
                        + "\n".join(cat_lines)
                        + "\n\nIMPORTANT: When writing prose about this chart, use the EXACT category "
                        "names listed above. Do NOT invent category names that are not in this list."
                    )

                    summaries["category_selectivity_chart"] = {
                        "section": "Input Stimuli",
                        "type": "Stacked area chart with scatter overlay showing category selectivity (positive and negative)",
                        "content": summary_content,
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: category selectivity failed: {e}")

            # =====================================================
            # INPUT STIMULI SECTION - Upstream Wiring Table
            # =====================================================
            # Read directly from wiring_analysis (required by protocol)
            wiring_data = investigation.get("wiring_analysis", {})

            if wiring_data.get("top_excitatory") or wiring_data.get("top_inhibitory"):
                try:
                    wiring_caption = "Predicted upstream neurons that excite/inhibit this neuron based on model weights (c_up + c_gate from SwiGLU)"
                    wiring_stats = wiring_data.get("stats", {})
                    if wiring_stats.get("regime_correction_applied"):
                        wiring_caption += (
                            ". <strong>Polarity labels have been regime-corrected</strong>"
                            " (target operates in inverted SwiGLU regime where gate and up channels are both negative)."
                        )
                    elif wiring_stats.get("regime_warning"):
                        wiring_caption += f". <em>Warning: {wiring_stats['regime_warning']}</em>"
                    html = generate_wiring_polarity_table(
                        wiring_data,
                        title="Upstream Wiring (Weight-Based Polarity)",
                        caption=wiring_caption,
                    )
                    agent.preview_figures["wiring_polarity_table"] = html

                    n_exc = len(wiring_data.get("top_excitatory", []))
                    n_inh = len(wiring_data.get("top_inhibitory", []))
                    top_exc = wiring_data.get("top_excitatory", [])[:3]
                    top_inh = wiring_data.get("top_inhibitory", [])[:3]
                    summaries["wiring_polarity_table"] = {
                        "section": "Input Stimuli",
                        "type": "Dual-column table showing excitatory vs inhibitory upstream neurons",
                        "content": f"Shows {n_exc} excitatory and {n_inh} inhibitory upstream connections. "
                                   f"Top excitatory: {', '.join(n.get('label', n.get('neuron_id', '?'))[:30] for n in top_exc)}. "
                                   f"Top inhibitory: {', '.join(n.get('label', n.get('neuron_id', '?'))[:30] for n in top_inh)}.",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: wiring table failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Output Projections
            # =====================================================
            logit_data = evidence.get("logit_attribution")
            if logit_data:
                try:
                    html = generate_output_projections(
                        logit_data,
                        title="Output Projections (Logit Attribution)",
                        caption="Top tokens this neuron promotes/suppresses in the vocabulary"
                    )
                    agent.preview_figures["output_projections"] = html

                    promotes = logit_data.get("promotes", [])[:5]
                    suppresses = logit_data.get("suppresses", [])[:5]
                    summaries["output_projections"] = {
                        "section": "Output Function",
                        "type": "Dual-column table showing promoted vs suppressed tokens",
                        "content": f"Shows top tokens this neuron projects to. "
                                   f"Promotes: {', '.join(repr(t.get('token', '?')) for t in promotes)}. "
                                   f"Suppresses: {', '.join(repr(t.get('token', '?')) for t in suppresses)}.",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: output projections failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Wiring Table
            # =====================================================
            # Read directly from output_wiring_analysis (required by protocol)
            output_wiring_data = investigation.get("output_wiring_analysis", {})

            if output_wiring_data.get("top_excitatory") or output_wiring_data.get("top_inhibitory"):
                try:
                    html = generate_downstream_wiring_table(
                        output_wiring_data,
                        title="Downstream Wiring (Weight-Based Polarity)",
                        caption="Predicted downstream neurons this neuron excites/inhibits based on model weights"
                    )
                    agent.preview_figures["downstream_wiring_table"] = html

                    n_exc = len(output_wiring_data.get("top_excitatory", []))
                    n_inh = len(output_wiring_data.get("top_inhibitory", []))
                    top_exc = output_wiring_data.get("top_excitatory", [])[:3]
                    top_inh = output_wiring_data.get("top_inhibitory", [])[:3]
                    summaries["downstream_wiring_table"] = {
                        "section": "Output Function",
                        "type": "Dual-column table showing neurons this neuron excites vs inhibits",
                        "content": f"Shows {n_exc} excited and {n_inh} inhibited downstream neurons. "
                                   f"Top excited: {', '.join(n.get('label', n.get('neuron_id', '?'))[:30] for n in top_exc)}. "
                                   f"Top inhibited: {', '.join(n.get('label', n.get('neuron_id', '?'))[:30] for n in top_inh)}.",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: downstream wiring failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Upstream Dependency Table
            # =====================================================
            upstream_dep = investigation.get("upstream_dependency_results")
            if upstream_dep:
                try:
                    # Handle both list and dict formats
                    dep_data = upstream_dep[0] if isinstance(upstream_dep, list) else upstream_dep
                    html = generate_upstream_dependency_table(
                        dep_data,
                        title="Upstream Causal Dependencies",
                        caption="How ablating upstream neurons affects this neuron's activation"
                    )
                    agent.preview_figures["upstream_dependency_table"] = html

                    deps = dep_data.get("individual_ablation", dep_data.get("dependencies", []))
                    strong_deps = [d for d in deps if abs(d.get("effect", 0)) > 0.5]
                    summaries["upstream_dependency_table"] = {
                        "section": "Output Function",
                        "type": "Table showing causal impact of ablating upstream neurons",
                        "content": f"Shows effects of ablating {len(deps)} upstream neurons. "
                                   f"{len(strong_deps)} have strong effects (>50% activation change).",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: upstream dependency failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Dependency Table
            # =====================================================
            downstream_dep = investigation.get("downstream_dependency_results")
            if downstream_dep:
                try:
                    # Handle both list and dict formats
                    dep_data = downstream_dep[0] if isinstance(downstream_dep, list) else downstream_dep
                    html = generate_downstream_dependency_table(
                        dep_data,
                        title="Downstream Causal Dependencies",
                        caption="How this neuron affects downstream neurons when ablated"
                    )
                    agent.preview_figures["downstream_dependency_table"] = html

                    deps = dep_data.get("individual_ablation", dep_data.get("dependencies", []))
                    strong_deps = [d for d in deps if abs(d.get("effect", 0)) > 0.5]
                    summaries["downstream_dependency_table"] = {
                        "section": "Output Function",
                        "type": "Table showing how ablating this neuron affects downstream neurons",
                        "content": f"Shows effects on {len(deps)} downstream neurons. "
                                   f"{len(strong_deps)} are strongly affected (>50% change).",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: downstream dependency failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Steering Downstream Effects
            # =====================================================
            steering_results = investigation.get("multi_token_steering_results", [])
            steering_with_downstream = [r for r in steering_results if isinstance(r, dict) and r.get("downstream_effects")]
            if steering_with_downstream:
                try:
                    html = generate_steering_downstream_table(
                        steering_with_downstream,
                        title="Steering Effects on Downstream Neurons",
                        caption="How steering this neuron affects downstream neuron activations"
                    )
                    agent.preview_figures["steering_downstream_table"] = html

                    n_experiments = len(steering_with_downstream)
                    summaries["steering_downstream_table"] = {
                        "section": "Output Function",
                        "type": "Table showing downstream effects of steering this neuron",
                        "content": f"Shows downstream effects from {n_experiments} steering experiments.",
                        "auto_generated": True
                    }
                except Exception as e:
                    print(f"  Preview: steering downstream failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Ablation Changed Completions
            # =====================================================
            ablation_results = investigation.get("multi_token_ablation_results", [])
            batch_ablation_runs = [r for r in ablation_results if isinstance(r, dict) and r.get("type") == "batch"]
            if batch_ablation_runs:
                try:
                    from neuron_scientist.figure_tools import generate_ablation_cards
                    html = generate_ablation_cards(
                        ablation_results,
                        title="Ablation Effects on Completions",
                    )
                    if html:
                        agent.preview_figures["ablation_completions"] = html

                        total_prompts = sum(r.get("total_prompts", 0) for r in batch_ablation_runs)
                        total_changed = sum(r.get("total_changed", 0) for r in batch_ablation_runs)
                        change_rate = (total_changed / total_prompts * 100) if total_prompts > 0 else 0
                        summaries["ablation_completions"] = {
                            "section": "Output Function",
                            "type": "Card gallery showing how ablation changes model completions",
                            "content": f"Shows {total_changed} of {total_prompts} completions that changed when neuron was ablated ({change_rate:.1f}%).",
                            "auto_generated": True
                        }
                except Exception as e:
                    print(f"  Preview: ablation completions failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Intelligent Steering Gallery
            # =====================================================
            intelligent_steering_runs = [r for r in steering_results if isinstance(r, dict) and r.get("type") == "intelligent_steering"]
            if intelligent_steering_runs:
                try:
                    from neuron_scientist.figure_tools import generate_intelligent_steering_gallery
                    html = generate_intelligent_steering_gallery(
                        steering_results,
                        title="Intelligent Steering Analysis",
                    )
                    if html:
                        agent.preview_figures["steering_completions"] = html

                        total_prompts = sum(r.get("n_prompts", 0) for r in intelligent_steering_runs)
                        total_examples = sum(len(r.get("illustrative_examples", [])) for r in intelligent_steering_runs)
                        # Build rich summary so the dashboard agent can write accurate prose
                        run_details = []
                        for r in intelligent_steering_runs:
                            hs = r.get("hypothesis_supported")
                            hs_str = str(hs).lower() if hs is not None else "inconclusive"
                            # Get max change rate across steering values
                            # Field is "rate" (not "change_rate") in stats_by_steering_value
                            stats = r.get("stats_by_steering_value", {})
                            max_cr = max(
                                (s.get("rate") or s.get("change_rate") or 0 for s in stats.values() if isinstance(s, dict)),
                                default=0
                            )
                            cr_str = f", max change rate {max_cr:.0%}" if max_cr else ""
                            summary_text = (r.get("analysis_summary") or "")[:200]
                            run_details.append(
                                f"Run (hypothesis_supported={hs_str}, {r.get('n_prompts', 0)} prompts{cr_str}): {summary_text}"
                            )
                        summaries["steering_completions"] = {
                            "section": "Output Function",
                            "type": "Gallery showing steering effects on completions with illustrative examples",
                            "content": f"Shows {len(intelligent_steering_runs)} steering runs with {total_prompts} total prompts and {total_examples} illustrative examples. " + " | ".join(run_details),
                            "auto_generated": True
                        }
                except Exception as e:
                    print(f"  Preview: steering completions failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Ablation Effects Table
            # =====================================================
            if batch_ablation_runs and any(r.get("dependency_summary") for r in batch_ablation_runs):
                try:
                    from neuron_scientist.figure_tools import generate_downstream_ablation_effects
                    # Collect neuron IDs from dependency_summary and fetch labels
                    downstream_neuron_ids = []
                    for run in batch_ablation_runs:
                        if run.get("dependency_summary"):
                            downstream_neuron_ids.extend(run["dependency_summary"].keys())
                    neuron_labels = get_neuron_labels(list(set(downstream_neuron_ids))) if downstream_neuron_ids else {}

                    html = generate_downstream_ablation_effects(
                        ablation_results,
                        neuron_labels=neuron_labels,
                        title="Downstream Ablation Effects",
                    )
                    if html:
                        agent.preview_figures["downstream_ablation_effects"] = html

                        # Count neurons with strong effects (support both key names)
                        all_deps = {}
                        for r in batch_ablation_runs:
                            all_deps.update(r.get("dependency_summary", {}))
                        strong_effects = [d for d in all_deps.values() if abs(d.get("mean_change", d.get("mean_change_percent", 0))) > 20]
                        summaries["downstream_ablation_effects"] = {
                            "section": "Output Function",
                            "type": "Table showing how ablating this neuron affects downstream neuron activations",
                            "content": f"Shows effects on {len(all_deps)} downstream neurons. {len(strong_effects)} have >20% change.",
                            "auto_generated": True
                        }
                except Exception as e:
                    print(f"  Preview: downstream ablation effects failed: {e}")

            # =====================================================
            # OUTPUT FUNCTION SECTION - Downstream Steering Response (Slope Table)
            # =====================================================
            steering_with_slopes = [r for r in intelligent_steering_runs if r.get("downstream_steering_slopes")]
            if steering_with_slopes:
                try:
                    from neuron_scientist.figure_tools import (
                        generate_downstream_steering_slope_table,
                    )
                    sr = steering_with_slopes[0]
                    slopes_data = sr["downstream_steering_slopes"]
                    ds_neuron_ids = list(slopes_data.keys())
                    steering_labels = get_neuron_labels(ds_neuron_ids) if ds_neuron_ids else {}

                    sv_tested = sr.get("steering_values", [])
                    n_prompts = sr.get("total_prompts", 0)
                    html = generate_downstream_steering_slope_table(
                        slopes_data,
                        neuron_labels=steering_labels,
                        title="Downstream Steering Response",
                        steering_values_tested=[sv for sv in sv_tested if sv != 0],
                        n_prompts=n_prompts,
                    )
                    if html:
                        agent.preview_figures["downstream_steering_slopes"] = html

                        # Build summary for agent visibility
                        n_neurons = len(slopes_data)
                        strong = [s for s in slopes_data.values() if abs(s.get("slope", 0) or 0) > 5]
                        reliable = [s for s in slopes_data.values() if (s.get("r_squared") or 0) > 0.5]
                        summaries["downstream_steering_response"] = {
                            "section": "Output Function",
                            "type": "Table showing causal slope (% change per unit steering) and R² per downstream neuron",
                            "content": (
                                f"Shows dose-response for {n_neurons} downstream neurons. "
                                f"{len(strong)} have |slope| > 5, {len(reliable)} have R² > 0.5. "
                                f"Includes wiring weight agreement check."
                            ),
                            "auto_generated": True
                        }
                except Exception as e:
                    print(f"  Preview: downstream steering slopes failed: {e}")

            # Mark preview as done
            agent.preview_done = True
            agent.preview_summaries = summaries

            # Build response
            response = {
                "success": True,
                "visualizations_generated": len(agent.preview_figures),
                "sections": {
                    "input_stimuli": [k for k, v in summaries.items() if v.get("section") == "Input Stimuli"],
                    "output_function": [k for k, v in summaries.items() if v.get("section") == "Output Function"],
                },
                "summaries": summaries,
                "instructions": (
                    "These visualizations will be auto-inserted into the dashboard. "
                    "You can now write prose that references and explains these tables/charts. "
                    "Use the write_* tools to add your prose to each section. "
                    "You may also generate additional custom figures using the figure generation tools."
                )
            }

            return {
                "content": [{"type": "text", "text": json.dumps(response, indent=2)}]
            }

        # =================================================================
        # NEURONDB LOOKUP TOOL
        # =================================================================

        @tool(
            "lookup_neuron_descriptions",
            "Look up neuron descriptions from the NeuronDB database. Use this to get meaningful labels for neurons that have missing or generic labels (like 'tech-router', 'router', or just the neuron ID).",
            {
                "neuron_ids": str,  # JSON array of neuron IDs like ["L12/N8459", "L21/N6856"]
            }
        )
        async def tool_lookup_neuron_descriptions(args: dict[str, Any]) -> dict[str, Any]:
            """Look up descriptions for neurons from the database."""
            neuron_ids = json.loads(args.get("neuron_ids", "[]"))

            if not neuron_ids:
                return {
                    "content": [{"type": "text", "text": json.dumps({
                        "success": False,
                        "error": "No neuron IDs provided",
                        "descriptions": {}
                    })}]
                }

            descriptions = get_neuron_labels(neuron_ids)

            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "found": len(descriptions),
                    "requested": len(neuron_ids),
                    "descriptions": descriptions
                }, indent=2)}]
            }

        # =================================================================
        # FIGURE TOOLS
        # =================================================================

        @tool(
            "generate_activation_grid",
            "Create side-by-side comparison of high vs low activation examples",
            {
                "high_examples": str,
                "low_examples": str,
                "title": str,
                "caption": str,
            }
        )
        async def tool_activation_grid(args: dict[str, Any]) -> dict[str, Any]:
            high = json.loads(args.get("high_examples", "[]"))
            low = json.loads(args.get("low_examples", "[]"))
            html = generate_activation_grid(
                high, low,
                title=args.get("title", "Activation Comparison"),
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "activation_grid"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_hypothesis_timeline",
            "Create visual of hypothesis prior->posterior evolution. If hypotheses arg is empty or '[]', auto-uses investigation data.",
            {
                "hypotheses": str,
                "title": str,
                "caption": str,
            }
        )
        async def tool_hypothesis_timeline(args: dict[str, Any]) -> dict[str, Any]:
            hypotheses_arg = args.get("hypotheses", "[]")
            hypotheses = json.loads(hypotheses_arg) if hypotheses_arg and hypotheses_arg != "[]" else []

            # If no hypotheses provided or missing probability data, use investigation data
            if not hypotheses:
                # Auto-populate from investigation
                hypotheses_tested = investigation.get("hypotheses_tested", [])
                hypotheses = [
                    {
                        "id": h.get("hypothesis_id", f"H{i}"),
                        "text": h.get("hypothesis", "")[:150],
                        "status": h.get("status", "testing"),
                        "prior": h.get("prior_probability", 50),
                        "posterior": h.get("posterior_probability", 50),
                    }
                    for i, h in enumerate(hypotheses_tested)
                ]
            else:
                # Enrich provided hypotheses with missing probability data from investigation
                hypotheses_tested = {h.get("hypothesis_id", ""): h for h in investigation.get("hypotheses_tested", [])}
                for h in hypotheses:
                    hid = h.get("id", h.get("hypothesis_id", ""))
                    if hid in hypotheses_tested:
                        source = hypotheses_tested[hid]
                        # Fill in missing probability data
                        if h.get("prior", 50) == 50 and h.get("posterior", 50) == 50:
                            h["prior"] = source.get("prior_probability", 50)
                            h["posterior"] = source.get("posterior_probability", 50)
                        if not h.get("status"):
                            h["status"] = source.get("status", "testing")

            html = generate_hypothesis_timeline(
                hypotheses,
                title=args.get("title", "Hypothesis Evolution"),
                caption=args.get("caption", "")
            )
            # Check if hypothesis_timeline already exists - update instead of adding duplicate
            existing_idx = next((i for i, f in enumerate(agent.generated_figures) if f.get("type") == "hypothesis_timeline"), None)
            if existing_idx is not None:
                agent.generated_figures[existing_idx] = {"html": html, "type": "hypothesis_timeline"}
                return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": existing_idx, "note": "Updated existing hypothesis_timeline"})}]}
            else:
                agent.generated_figures.append({"html": html, "type": "hypothesis_timeline"})
                return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        # NOTE: generate_circuit_diagram removed - the three-panel circuit block in
        # write_dashboard already shows upstream/downstream connectivity. The circuit
        # diagram was redundant and added visual noise.

        @tool(
            "generate_selectivity_gallery",
            "Create grid of examples organized by category",
            {
                "categories": str,
                "title": str,
                "caption": str,
            }
        )
        async def tool_selectivity_gallery(args: dict[str, Any]) -> dict[str, Any]:
            categories = json.loads(args.get("categories", "[]"))
            html = generate_selectivity_gallery(
                categories,
                title=args.get("title", "Selectivity Patterns"),
                caption=args.get("caption", "")
            )
            # Check if selectivity_gallery already exists - update instead of adding duplicate
            existing_idx = next((i for i, f in enumerate(agent.generated_figures) if f.get("type") == "selectivity_gallery"), None)
            if existing_idx is not None:
                agent.generated_figures[existing_idx] = {"html": html, "type": "selectivity_gallery"}
                return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": existing_idx, "note": "Updated existing selectivity_gallery"})}]}
            else:
                agent.generated_figures.append({"html": html, "type": "selectivity_gallery"})
                return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_ablation_matrix",
            """Create table of token effects across prompts.

            IMPORTANT: Pass the ablation data directly from experiments.ablation, which has format:
            [{"prompt": "...", "promotes": [["token", value], ...], "suppresses": [["token", value], ...]}, ...]

            If experiments is empty or "[]", will auto-inject from investigation context.""",
            {
                "experiments": str,
                "title": str,
                "caption": str,
            }
        )
        async def tool_ablation_matrix(args: dict[str, Any]) -> dict[str, Any]:
            experiments = json.loads(args.get("experiments", "[]"))
            # Auto-inject from context if empty - prefer steering_results (has prompts)
            if not experiments:
                steering_results = investigation.get("steering_results", [])
                inv_evidence = investigation.get("evidence", {})
                ablation_effects = inv_evidence.get("ablation_effects", [])
                # Use steering_results if available and has prompts
                experiments = steering_results if steering_results and any(d.get("prompt") for d in steering_results) else ablation_effects
                experiments = experiments[:10]
            html = generate_ablation_matrix(
                experiments,
                title=args.get("title", "Ablation Effects"),
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "ablation_matrix"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_steering_curves",
            """Create table of effects at different steering values.

            IMPORTANT: Pass the steering data directly from experiments.steering, which has format:
            [{"steering_value": 10, "promotes": [["token", value], ...], "suppresses": [["token", value], ...]}, ...]

            If dose_response_data is empty or "[]", will auto-inject from investigation context.
            The highlight_tokens are optional token names to emphasize.""",
            {
                "dose_response_data": str,
                "highlight_tokens": str,
                "title": str,
                "caption": str,
            }
        )
        async def tool_steering_curves(args: dict[str, Any]) -> dict[str, Any]:
            data = json.loads(args.get("dose_response_data", "[]"))
            # Auto-inject from context if empty
            # Note: steering data is stored in dose_response_results/steering_results at top level
            if not data:
                inv_evidence = investigation.get("evidence", {})
                data = investigation.get("dose_response_results") or investigation.get("steering_results") or inv_evidence.get("steering_effects", [])
                data = data[:10] if data else []
            tokens = json.loads(args.get("highlight_tokens", "[]"))
            html = generate_steering_curves(
                data, tokens,
                title=args.get("title", "Steering Response"),
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "steering_curves"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_evidence_card",
            "Create key finding highlight card",
            {
                "finding": str,
                "evidence_type": str,
                "supporting_data": str,
            }
        )
        async def tool_evidence_card(args: dict[str, Any]) -> dict[str, Any]:
            finding = args.get("finding", "")

            # Check for duplicate evidence cards with the same finding
            for i, fig in enumerate(agent.generated_figures):
                if fig.get("type") == "evidence_card" and fig.get("finding") == finding:
                    return {"content": [{"type": "text", "text": json.dumps({
                        "success": True,
                        "figure_index": i,
                        "note": "Reusing existing evidence card with same finding"
                    })}]}

            html = generate_evidence_card(
                finding=finding,
                evidence_type=args.get("evidence_type", "confirmation"),
                supporting_data=args.get("supporting_data", "")
            )
            agent.generated_figures.append({"html": html, "type": "evidence_card", "finding": finding})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_anomaly_box",
            "Create callout for surprising findings",
            {
                "anomaly_description": str,
                "expected_behavior": str,
                "observed_behavior": str,
                "possible_explanations": str,
            }
        )
        async def tool_anomaly_box(args: dict[str, Any]) -> dict[str, Any]:
            description = args.get("anomaly_description", "")

            # Check for duplicate anomaly boxes with the same description
            for i, fig in enumerate(agent.generated_figures):
                if fig.get("type") == "anomaly_box" and fig.get("description") == description:
                    return {"content": [{"type": "text", "text": json.dumps({
                        "success": True,
                        "figure_index": i,
                        "note": "Reusing existing anomaly box with same description"
                    })}]}

            explanations = json.loads(args.get("possible_explanations", "[]"))
            html = generate_anomaly_box(
                anomaly_description=description,
                expected_behavior=args.get("expected_behavior", ""),
                observed_behavior=args.get("observed_behavior", ""),
                possible_explanations=explanations
            )
            agent.generated_figures.append({"html": html, "type": "anomaly_box", "description": description})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_homograph_comparison",
            """Create side-by-side comparison showing how a neuron discriminates between different meanings of the same word.

            Use this when the neuron shows different activation levels for ambiguous words in different contexts.

            IMPORTANT: The `pairs` parameter must be a JSON string with this EXACT structure:
            [
              {
                "word": "virus",
                "contexts": [
                  {"label": "Malware", "example": "infected computers", "activation": 4.75, "category": "malware"},
                  {"label": "Biological", "example": "infected people", "activation": 0.64, "category": "biological"}
                ]
              },
              {
                "word": "worm",
                "contexts": [
                  {"label": "Malware", "example": "exploited Windows", "activation": 5.06, "category": "malware"},
                  {"label": "Animal", "example": "bird ate a worm", "activation": 0.06, "category": "animal"}
                ]
              }
            ]

            Each pair MUST have:
            - "word": the ambiguous word
            - "contexts": array of EXACTLY 2 context objects, each with:
              - "label": category name (e.g., "Malware", "Biological")
              - "example": short phrase showing usage (in quotes in display)
              - "activation": the neuron's activation value
              - "category": one of: malware, biological, animal, mythology, neutral""",
            {
                "pairs": str,
                "title": str,
                "caption": str,
                "explanation": str,
            }
        )
        async def tool_homograph_comparison(args: dict[str, Any]) -> dict[str, Any]:
            pairs = json.loads(args.get("pairs", "[]"))
            html = generate_homograph_comparison(
                pairs,
                title=args.get("title", "Homograph Discrimination"),
                caption=args.get("caption", ""),
                explanation=args.get("explanation", "")
            )
            agent.generated_figures.append({"html": html, "type": "homograph_comparison"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_stacked_density_chart",
            """Create a stacked bar chart showing category distributions at different activation z-scores.

            Useful for showing how the proportion of different categories changes as activation increases. For example, showing that at high activations, malware contexts dominate while biological contexts are rare.""",
            {
                "bin_data": str,  # JSON array: [{"zMid": -2.0, "malware": 5, "biological": 45, "neutral": 20}]
                "categories": str,  # JSON array: [{"name": "malware", "color": "#dc2626", "description": "Technical malware context"}]
                "title": str,
                "caption": str,
                "explanation": str,  # Required narrative explaining the significance
            }
        )
        async def tool_stacked_density_chart(args: dict[str, Any]) -> dict[str, Any]:
            bin_data = json.loads(args.get("bin_data", "[]"))
            categories = json.loads(args.get("categories", "[]"))
            html = generate_stacked_density_chart(
                bin_data,
                categories,
                title=args.get("title", "Activation Distribution by Category"),
                caption=args.get("caption", ""),
                explanation=args.get("explanation", "")
            )
            agent.generated_figures.append({"html": html, "type": "stacked_density_chart"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_patching_comparison",
            """Create visualization for counterfactual activation patching experiments.

            Shows what happens when activation from a high-activation source prompt is patched into a low-activation target prompt. Useful for demonstrating causal effects.

            IMPORTANT: The `experiments` parameter must be a JSON string with this structure:
            [
              {
                "source_prompt": "The malware infected the server",
                "target_prompt": "The patient was infected by a virus",
                "source_activation": 4.75,
                "target_activation": 0.64,
                "promoted_tokens": [["malware", 2.3], ["virus", 1.1]],
                "suppressed_tokens": [["patient", -1.5]],
                "max_shift": 2.3
              }
            ]

            If experiments is empty or "[]", will auto-inject from investigation's patching_experiments.""",
            {
                "experiments": str,  # JSON array of patching experiments
                "title": str,
                "caption": str,
                "explanation": str,  # Narrative explaining the significance
            }
        )
        async def tool_patching_comparison(args: dict[str, Any]) -> dict[str, Any]:
            experiments_arg = args.get("experiments", "[]")
            experiments = json.loads(experiments_arg) if experiments_arg and experiments_arg != "[]" else []

            # Auto-inject from investigation if not provided
            if not experiments:
                experiments = investigation.get("patching_experiments", [])

            html = generate_patching_comparison(
                experiments,
                title=args.get("title", "Counterfactual Patching"),
                caption=args.get("caption", ""),
                explanation=args.get("explanation", "")
            )
            agent.generated_figures.append({"html": html, "type": "patching_comparison"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_category_selectivity_chart",
            """Create interactive category selectivity visualization (stacked area chart with hoverable dots).

            Shows the conditional probability of each semantic category at different activation z-score levels.
            Individual prompts are shown as dots with tooltips showing the prompt text and activation.

            This visualization demonstrates whether the neuron is truly selective for certain semantic domains
            vs just having high baseline activation.

            The tool auto-loads category_selectivity_data from the investigation if available.
            If not, pass category_data as JSON with structure:
            {
                "global_mean": 0.5, "global_std": 0.3,
                "categories": {
                    "category_name": {
                        "type": "target"|"control"|"inhibitory"|"unrelated",
                        "prompts": [{"prompt": "...", "activation": 1.5, "z_score": 2.1}, ...],
                        "z_mean": 2.1
                    }
                },
                "selectivity_summary": "HIGHLY SELECTIVE: ..."
            }""",
            {
                "category_data": str,  # JSON or empty to auto-load
                "title": str,
                "caption": str,
                "explanation": str,
            }
        )
        async def tool_category_selectivity_chart(args: dict[str, Any]) -> dict[str, Any]:
            category_data_arg = args.get("category_data", "")
            category_data = json.loads(category_data_arg) if category_data_arg and category_data_arg != "{}" else {}

            # Auto-inject from investigation if not provided (merge all runs)
            if not category_data:
                category_data = _resolve_selectivity_data(investigation)

            if not category_data or "categories" not in category_data:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No category_selectivity_data available. Run category selectivity test first."})}]}

            html = generate_category_selectivity_chart(
                category_data,
                neuron_id=investigation.get("neuron_id", "Unknown"),
                title=args.get("title", ""),
                caption=args.get("caption", ""),
                explanation=args.get("explanation", "")
            )
            agent.generated_figures.append({"html": html, "type": "category_selectivity_chart"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        # =================================================================
        # DEPENDENCY VISUALIZATION TOOLS
        # =================================================================

        @tool(
            "generate_upstream_dependency_table",
            """Create table showing how ablating upstream neurons affects this neuron.

            Auto-loads from investigation's upstream_dependency_results if available.
            Shows which upstream neurons this neuron depends on for its activation.""",
            {
                "title": str,
                "caption": str,
            }
        )
        async def tool_upstream_dependency_table(args: dict[str, Any]) -> dict[str, Any]:
            dep_data = investigation.get("upstream_dependency_results", [])
            if not dep_data:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No upstream_dependency_results available"})}]}

            # Use first result if multiple
            data = dep_data[0] if isinstance(dep_data, list) else dep_data

            # Collect neuron IDs and fetch labels
            neuron_ids = data.get("upstream_neurons", []) + list(data.get("individual_ablation", {}).keys())
            neuron_labels = get_neuron_labels(list(set(neuron_ids))) if neuron_ids else {}

            # Get wiring weights from connectivity for agreement comparison
            wiring_weights = {}
            evidence = investigation.get("evidence", {})
            connectivity = evidence.get("connectivity", {}) if isinstance(evidence, dict) else {}
            for u in connectivity.get("upstream_neurons", []):
                if isinstance(u, dict) and u.get("neuron_id"):
                    wiring_weights[u["neuron_id"]] = u.get("weight", 0)

            html = generate_upstream_dependency_table(
                data,
                title=args.get("title", "Upstream Dependencies"),
                caption=args.get("caption", ""),
                neuron_labels=neuron_labels,
                wiring_weights=wiring_weights
            )
            agent.generated_figures.append({"html": html, "type": "upstream_dependency_table"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_downstream_dependency_table",
            """Create table showing how ablating this neuron affects downstream neurons.

            Auto-loads from investigation's downstream_dependency_results if available.
            Shows which downstream neurons depend on this neuron's activation.""",
            {
                "title": str,
                "caption": str,
            }
        )
        async def tool_downstream_dependency_table(args: dict[str, Any]) -> dict[str, Any]:
            dep_data = investigation.get("downstream_dependency_results", [])
            if not dep_data:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No downstream_dependency_results available"})}]}

            # Use first result if multiple
            data = dep_data[0] if isinstance(dep_data, list) else dep_data

            # Collect neuron IDs and fetch labels
            neuron_ids = data.get("downstream_neurons", [])
            neuron_labels = get_neuron_labels(list(set(neuron_ids))) if neuron_ids else {}

            # Get wiring weights from connectivity for comparison
            wiring_weights = {}
            inv_evidence = investigation.get("evidence", {})
            connectivity = inv_evidence.get("connectivity", {}) if isinstance(inv_evidence, dict) else {}
            for d in connectivity.get("downstream_neurons", connectivity.get("downstream_targets", [])):
                if isinstance(d, dict) and d.get("neuron_id"):
                    wiring_weights[d["neuron_id"]] = d.get("weight", 0)
                elif isinstance(d, dict) and d.get("id"):
                    wiring_weights[d["id"]] = d.get("weight", 0)

            html = generate_downstream_dependency_table(
                data,
                title=args.get("title", "Downstream Ablation Effects"),
                caption=args.get("caption", ""),
                neuron_labels=neuron_labels,
                wiring_weights=wiring_weights
            )
            agent.generated_figures.append({"html": html, "type": "downstream_dependency_table"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_steering_downstream_table",
            """Create table showing how steering this neuron affects downstream neurons.

            Shows circuit propagation: when this neuron is boosted/suppressed, how do connected
            downstream neurons respond? Useful for understanding causal flow through the circuit.
            Auto-loads from investigation's multi_token_steering_results if available.""",
            {
                "title": str,
                "caption": str,
            }
        )
        async def tool_steering_downstream_table(args: dict[str, Any]) -> dict[str, Any]:
            steering_data = investigation.get("multi_token_steering_results", [])
            if not steering_data:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No multi_token_steering_results available"})}]}

            # Filter to only results with downstream_effects (handle malformed data)
            results_with_downstream = [r for r in steering_data if isinstance(r, dict) and r.get("downstream_effects")]
            if not results_with_downstream:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No downstream_effects in steering results"})}]}

            html = generate_steering_downstream_table(
                results_with_downstream,
                title=args.get("title", "Steering Propagation"),
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "steering_downstream_table"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        # =================================================================
        # SKEPTIC VISUALIZATION TOOLS
        # =================================================================

        @tool(
            "generate_boundary_test_cards",
            """Create cards showing skeptic boundary test results.

            Auto-loads from investigation's skeptic_report.boundary_tests if available.
            Shows which tests passed/failed and why.""",
            {
                "show_only_failures": str,  # "true" or "false"
                "title": str,
                "caption": str,
            }
        )
        async def tool_boundary_test_cards(args: dict[str, Any]) -> dict[str, Any]:
            skeptic = investigation.get("skeptic_report", {})
            boundary_tests = skeptic.get("boundary_tests", []) if skeptic else []

            if not boundary_tests:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No boundary_tests available in skeptic_report"})}]}

            show_failures = args.get("show_only_failures", "false").lower() == "true"

            html = generate_boundary_test_cards(
                boundary_tests,
                title=args.get("title", "Boundary Tests"),
                show_only_failures=show_failures,
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "boundary_test_cards"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        @tool(
            "generate_alternative_hypothesis_cards",
            """Create cards showing skeptic alternative hypothesis test results.

            Auto-loads from investigation's skeptic_report.alternative_hypotheses if available.
            Shows which alternative explanations were tested and their outcomes.""",
            {
                "title": str,
                "caption": str,
            }
        )
        async def tool_alternative_hypothesis_cards(args: dict[str, Any]) -> dict[str, Any]:
            skeptic = investigation.get("skeptic_report", {})
            alternatives = skeptic.get("alternative_hypotheses", []) if skeptic else []

            if not alternatives:
                return {"content": [{"type": "text", "text": json.dumps({"error": "No alternative_hypotheses available in skeptic_report"})}]}

            html = generate_alternative_hypothesis_cards(
                alternatives,
                title=args.get("title", "Alternative Hypotheses Tested"),
                caption=args.get("caption", "")
            )
            agent.generated_figures.append({"html": html, "type": "alternative_hypothesis_cards"})
            return {"content": [{"type": "text", "text": json.dumps({"success": True, "figure_index": len(agent.generated_figures) - 1})}]}

        # =================================================================
        # OUTPUT TOOL
        # =================================================================

        @tool(
            "write_dashboard",
            """Assemble and write the final HTML dashboard.

            Use section_content for the 4-section structure:
            {
                "input_function": {"prose": "...", "figures": [0, 1]},
                "output_function": {"prose": "...", "figures": [2, 3]},
                "hypothesis_testing": {"prose": "...", "figures": [4, 5]},
                "open_questions": {"prose": "...", "figures": []}
            }

            ALTERNATIVE: Use prose_sections for legacy freeform structure (backward compatible).

            NOTE: Selectivity examples (Fires On) are auto-injected from investigation data.""",
            {
                "title": str,
                "narrative_lead": str,
                "narrative_body": str,
                "section_content": str,  # JSON with input_function, output_function, hypothesis_testing, open_questions
                "prose_sections": str,  # LEGACY: JSON array of {heading, content}
            }
        )
        async def tool_write_dashboard(args: dict[str, Any]) -> dict[str, Any]:
            # Debug: log what we received
            print(f"[DEBUG write_dashboard] args type: {type(args)}")
            print(f"[DEBUG write_dashboard] args keys: {list(args.keys()) if isinstance(args, dict) else 'NOT A DICT'}")
            for k, v in (args.items() if isinstance(args, dict) else []):
                print(f"[DEBUG write_dashboard] {k}: type={type(v).__name__}, value={str(v)[:100]}")
            try:
                return await _do_write_dashboard(args)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[DEBUG write_dashboard] ERROR: {error_details}")
                return {
                    "content": [{"type": "text", "text": json.dumps({
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": error_details[-1000:],  # Last 1000 chars of traceback
                    })}]
                }

        async def _do_write_dashboard(args: dict[str, Any]) -> dict[str, Any]:
            neuron_id = investigation.get("neuron_id", "L0/N0")

            # Auto-generate preview figures if not already done
            # This ensures ablation/steering completion cards are available even if
            # the agent didn't explicitly call preview_visualizations
            if not agent.preview_done:
                new_figures = _generate_preview_figures(investigation)
                agent.preview_figures.update(new_figures)

            # Defensive: ensure output_projections is a dict
            op_raw = investigation.get("output_projections", {"promote": [], "suppress": []})
            output_projections = op_raw if isinstance(op_raw, dict) else {"promote": [], "suppress": []}

            # Defensive: ensure findings is a dict
            findings_raw = investigation.get("findings")
            findings = findings_raw if isinstance(findings_raw, dict) else {}

            # SINGLE SOURCE OF TRUTH for upstream/downstream neurons:
            # Use WIRING ANALYSIS (weight-based) - required by protocol
            # Wiring shows which neurons COULD influence this neuron based on model weights
            # Ablation/steering experiments then TEST these neurons for actual causal effects
            upstream_neurons = []
            downstream_neurons = []

            # Read directly from wiring_analysis and output_wiring_analysis (required by protocol)
            wiring_data = investigation.get("wiring_analysis", {})
            output_wiring_data = investigation.get("output_wiring_analysis", {})

            # Build upstream neurons from wiring analysis (excitatory + inhibitory, sorted by |weight|)
            if wiring_data.get("top_excitatory") or wiring_data.get("top_inhibitory"):
                all_upstream = wiring_data.get("top_excitatory", []) + wiring_data.get("top_inhibitory", [])
                # Sort by effective_strength (primary field from SwiGLU analysis)
                sorted_upstream = sorted(all_upstream, key=lambda x: abs(x.get("effective_strength", x.get("weight", 0))), reverse=True)[:6]
                upstream_neurons = [
                    {
                        "id": n.get("neuron_id", ""),
                        "label": n.get("label", "Unknown"),
                        "weight": n.get("effective_strength", n.get("weight", 0)),
                        "polarity": "excitatory" if n in wiring_data.get("top_excitatory", []) else "inhibitory"
                    }
                    for n in sorted_upstream
                    if n.get("neuron_id")
                ]

            # Build downstream neurons from output wiring analysis
            if output_wiring_data.get("top_excitatory") or output_wiring_data.get("top_inhibitory"):
                all_downstream = output_wiring_data.get("top_excitatory", []) + output_wiring_data.get("top_inhibitory", [])
                # Sort by effective_strength (primary field from SwiGLU analysis)
                sorted_downstream = sorted(all_downstream, key=lambda x: abs(x.get("effective_strength", x.get("weight", 0))), reverse=True)[:6]
                downstream_neurons = [
                    {
                        "id": n.get("neuron_id", ""),
                        "label": n.get("label", "Unknown"),
                        "weight": n.get("effective_strength", n.get("weight", 0)),
                        "polarity": "excitatory" if n in output_wiring_data.get("top_excitatory", []) else "inhibitory"
                    }
                    for n in sorted_downstream
                    if n.get("neuron_id")
                ]

            # Enrich any "Unknown" labels from NeuronDB
            all_neuron_ids = [n["id"] for n in upstream_neurons + downstream_neurons if n.get("id")]
            unknown_ids = [n["id"] for n in upstream_neurons + downstream_neurons if n.get("label") == "Unknown"]
            if unknown_ids:
                neuron_labels = get_neuron_labels(list(set(unknown_ids)))
                for n in upstream_neurons:
                    if n["id"] in neuron_labels and n.get("label") == "Unknown":
                        n["label"] = neuron_labels[n["id"]]
                for n in downstream_neurons:
                    if n["id"] in neuron_labels and n.get("label") == "Unknown":
                        n["label"] = neuron_labels[n["id"]]

            # Parse prose sections - handle empty strings and already-parsed lists
            prose_raw = args.get("prose_sections", "[]")
            if isinstance(prose_raw, list):
                prose_sections = prose_raw
            elif isinstance(prose_raw, str) and prose_raw.strip():
                try:
                    prose_sections = json.loads(prose_raw)
                except json.JSONDecodeError:
                    prose_sections = []
            else:
                prose_sections = []

            # AUTO-INJECT selectivity from investigation data (ignore agent parameters)
            # Priority: category_selectivity_data > activating_prompts
            evidence = investigation.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}

            selectivity_fires = []
            selectivity_ignores = []  # Not displayed, but kept for API compatibility

            # Try category_selectivity_data first (has richer category information)
            cat_sel_data = _resolve_selectivity_data(investigation)
            if isinstance(cat_sel_data, dict):
                # Build a lookup from prompt text to token/position from categories
                # (top_activating may be missing token info in older investigations)
                prompt_to_token = {}
                categories = cat_sel_data.get("categories", {})
                if isinstance(categories, dict):
                    for cat_name, cat_data in categories.items():
                        if isinstance(cat_data, dict):
                            for p in cat_data.get("prompts", []):
                                if isinstance(p, dict) and p.get("prompt"):
                                    prompt_to_token[p["prompt"]] = {
                                        "token": p.get("token", ""),
                                        "position": p.get("position"),
                                    }

                # Use top_activating (already sorted by activation)
                top_activating = cat_sel_data.get("top_activating", [])
                if top_activating and isinstance(top_activating, list):
                    examples = []
                    for ex in top_activating[:3]:
                        if not isinstance(ex, dict):
                            continue
                        prompt_text = ex.get("prompt", "")
                        # Get token from top_activating, or look it up from categories
                        token = ex.get("token", "")
                        if not token and prompt_text in prompt_to_token:
                            token = prompt_to_token[prompt_text].get("token", "")
                        examples.append({
                            "text": prompt_text,
                            "activation": ex.get("activation", 0),
                            "token": token,
                        })
                    if examples:
                        selectivity_fires = [{
                            "label": "Top Activating",
                            "type": "fires",
                            "examples": examples
                        }]
                # Fallback: extract directly from categories (sorted by activation)
                elif categories:
                    all_examples = []
                    for cat_name, cat_data in categories.items():
                        if isinstance(cat_data, dict):
                            for ex in cat_data.get("prompts", []):
                                if isinstance(ex, dict):
                                    all_examples.append({
                                        "text": ex.get("prompt", ""),
                                        "activation": ex.get("activation", 0),
                                        "token": ex.get("token", ""),
                                    })
                    # Sort by activation and take top 3
                    all_examples.sort(key=lambda x: x.get("activation", 0), reverse=True)
                    if all_examples:
                        selectivity_fires = [{
                            "label": "Top Activating",
                            "type": "fires",
                            "examples": all_examples[:3]
                        }]

            # Fallback to activating_prompts if no category selectivity data
            if not selectivity_fires:
                activating = evidence.get("activating_prompts", [])
                if isinstance(activating, list) and activating:
                    selectivity_fires = [{
                        "label": "Activating Examples",
                        "type": "fires",
                        "examples": [
                            {
                                "text": ex.get("prompt", "") if isinstance(ex, dict) else "",  # Full text
                                "activation": ex.get("activation", 0) if isinstance(ex, dict) else 0,
                                "token": ex.get("token", "") if isinstance(ex, dict) else "",
                            }
                            for ex in activating[:3]
                            if isinstance(ex, dict)
                        ]
                    }]

            # Extract confidence adjustment information (kept for compatibility)
            confidence_downgraded = investigation.get("confidence_downgraded", False)
            pre_skeptic_confidence = investigation.get("pre_skeptic_confidence")
            skeptic_adjustment = investigation.get("skeptic_confidence_adjustment", 0)

            # Compute confidence from hypotheses if available
            # This gives hypothesis-level confidence rather than an arbitrary aggregate
            hypotheses = investigation.get("hypotheses_tested", [])
            if hypotheses:
                posteriors = []
                for h in hypotheses:
                    post = h.get("posterior_probability") or h.get("posterior") or h.get("confidence") or 0.5
                    # Convert percentage to decimal if needed
                    if isinstance(post, (int, float)) and post > 1:
                        post = post / 100.0
                    elif not isinstance(post, (int, float)):
                        post = 0.5
                    posteriors.append(post)
                # Use average of hypothesis posteriors as the displayed confidence
                display_confidence = sum(posteriors) / len(posteriors) if posteriors else 0.5
            else:
                display_confidence = investigation.get("confidence", 0.5)

            # Extract Variant 5 data
            characterization = investigation.get("characterization", {})

            # Full descriptions for executive summary
            input_function = characterization.get("input_function", "")
            output_function = characterization.get("output_function", "")
            final_hypothesis = characterization.get("final_hypothesis", "")
            function_type = characterization.get("function_type", "")

            # Use the full description for the summary (prefer final_hypothesis, then output_function)
            function_description = final_hypothesis or output_function

            # Extract steering downstream effects for amplification section
            steering_downstream = investigation.get("multi_token_steering_results", [])
            # Filter out non-dict items (data might be malformed)
            steering_with_downstream = [r for r in steering_downstream if isinstance(r, dict) and r.get("downstream_effects")]

            # Extract selectivity metrics (merged from all runs)
            cat_sel_data = _resolve_selectivity_data(investigation)
            selectivity_zscore = None
            if cat_sel_data:
                categories = cat_sel_data.get("categories", {})
                # Categories can be a dict (keyed by category name) or a list
                if isinstance(categories, dict):
                    # Use z_mean for each category
                    zscores = [c.get("z_mean", c.get("zscore", 0)) for c in categories.values() if isinstance(c, dict)]
                elif isinstance(categories, list):
                    zscores = [c.get("z_mean", c.get("zscore", 0)) for c in categories if isinstance(c, dict)]
                else:
                    zscores = []
                # Compute selectivity gap (max - min z-score)
                zscores = [z for z in zscores if z is not None]
                if zscores:
                    selectivity_zscore = max(zscores) - min(zscores) if len(zscores) > 1 else abs(max(zscores))

            # Get peak activation from activating prompts
            peak_activation = None
            activating = evidence.get("activating_prompts", [])
            if activating:
                activations = [p.get("activation", 0) for p in activating]
                if activations:
                    peak_activation = max(activations)

            # NOTE: executive_summary removed - now using agent's narrative_lead/narrative_body

            # Build fixed sections (Variant 5 style)
            # Debug: ensure selectivity_fires contains dicts, not strings
            safe_selectivity_fires = []
            for item in selectivity_fires:
                if isinstance(item, dict):
                    safe_selectivity_fires.append(item)
                # Skip strings or other non-dict items

            safe_selectivity_ignores = []
            for item in selectivity_ignores:
                if isinstance(item, dict):
                    safe_selectivity_ignores.append(item)

            fixed_sections = build_fixed_sections(
                neuron_id=neuron_id,
                title=args.get("title", "Neuron Investigation"),
                confidence=None,  # Removed - confidence is hypothesis-specific now
                total_experiments=investigation.get("total_experiments", 0),
                narrative_lead=args.get("narrative_lead", ""),
                narrative_body=args.get("narrative_body", ""),
                upstream_neurons=upstream_neurons,
                downstream_neurons=downstream_neurons,
                selectivity_fires=safe_selectivity_fires,
                selectivity_ignores=safe_selectivity_ignores,
                output_promote=output_projections.get("promote", []),
                output_suppress=output_projections.get("suppress", []),
                open_questions=investigation.get("open_questions", findings.get("open_questions", [])),
                stats=[],  # Removed metrics badges - now using metrics_row
                executive_summary="",  # Removed - now using agent's narrative
                confidence_downgraded=False,
                pre_skeptic_confidence=None,
                skeptic_adjustment=0,
                hypothesis_count=len(hypotheses),
                # New Variant 5 parameters
                function_description=function_description or "",
                steering_downstream=steering_with_downstream,
                selectivity_zscore=selectivity_zscore,
                peak_activation=peak_activation,
            )

            # Data grounding check: flag if characterization text overstates activation values
            data_grounding_note = ""
            if peak_activation is not None and input_function:
                # Find numbers in input_function text (e.g., "activations ≥1.0", "max 2.5")
                numbers_in_text = re.findall(r'[\d]+\.[\d]+|[\d]+', input_function)
                for num_str in numbers_in_text:
                    try:
                        num_val = float(num_str)
                        # Flag if claimed value is >1.5x the observed peak (and looks like an activation value)
                        if num_val > peak_activation * 1.5 and num_val > 0.5:
                            data_grounding_note = (
                                f'<div class="data-grounding-note" style="background:#fef3c7;border:1px solid #f59e0b;'
                                f'border-radius:6px;padding:8px 12px;margin:8px 0;font-size:12px;color:#92400e;">'
                                f'⚠️ <strong>Data grounding note:</strong> The characterization mentions a value of {num_val}, '
                                f'but the observed peak activation is {peak_activation:.3f}. '
                                f'The reported value may be overstated.</div>'
                            )
                            break
                    except ValueError:
                        continue

            # Auto-generate ablation/steering figures if data exists and agent didn't generate them
            # Defensive: only process dict items in generated_figures
            figure_types = {f.get("type") for f in agent.generated_figures if isinstance(f, dict)}
            inv_evidence = investigation.get("evidence", {}) if isinstance(investigation.get("evidence"), dict) else {}

            # Auto-add ablation matrix if data exists
            # Prefer steering_results (has prompts) over ablation_effects (simplified format)
            steering_results = investigation.get("steering_results", [])
            ablation_data = inv_evidence.get("ablation_effects", [])
            # Use steering_results if available and has prompts, otherwise fall back to ablation_effects
            ablation_for_matrix = steering_results if steering_results and any(d.get("prompt") for d in steering_results) else ablation_data
            if ablation_for_matrix and "ablation_matrix" not in figure_types:
                ablation_html = generate_ablation_matrix(
                    ablation_for_matrix[:8],
                    title="Ablation Effects",
                    caption=""
                )
                agent.generated_figures.append({"html": ablation_html, "type": "ablation_matrix"})

            # Auto-add steering curves if data exists AND has usable STRUCTURED content
            # Note: steering data is stored in dose_response_results/steering_results at top level
            steering_data = investigation.get("dose_response_results") or investigation.get("steering_results") or inv_evidence.get("steering_effects", [])
            # Check if steering data has actual structured results (lists, not strings)
            # Handle multiple field naming conventions
            has_usable_steering = steering_data and any(
                d.get("results") or d.get("promotes") or d.get("suppresses") or
                d.get("promoted_tokens") or d.get("suppressed_tokens") or
                d.get("dose_response_curve") or  # Nested format with curve data
                (isinstance(d.get("effects"), list) and d.get("effects"))
                for d in steering_data
            )
            if has_usable_steering and "steering_curves" not in figure_types:
                steering_html = generate_steering_curves(
                    steering_data[:8],
                    highlight_tokens=[],
                    title="Steering Dose-Response",
                    caption=""
                )
                agent.generated_figures.append({"html": steering_html, "type": "steering_curves"})

            # Filter out stat_cards from figures (they're now in fixed section)
            evidence_figures = [f for f in agent.generated_figures if f.get("type") != "stat_cards"]

            # Remove agent-generated hypothesis_timeline from evidence_figures since
            # we always auto-generate one from canonical data (prevents duplicates)
            evidence_figures = [f for f in evidence_figures if f.get("type") != "hypothesis_timeline"]

            # Parse section_content if provided - handle various input formats
            section_content_raw = args.get("section_content", "")
            if isinstance(section_content_raw, dict):
                section_content = section_content_raw
            elif isinstance(section_content_raw, str) and section_content_raw.strip():
                try:
                    section_content = json.loads(section_content_raw)
                except json.JSONDecodeError:
                    section_content = {}
            else:
                section_content = {}

            # ALWAYS BUILD FOUR-SECTION STRUCTURE
            # (Use section_content for prose if provided, otherwise extract from prose_sections)

            # Helper to get figures HTML for a section
            def get_section_figures_html(figure_indices):
                if not figure_indices:
                    return ""
                return "".join(
                    evidence_figures[i].get("html", "")
                    for i in figure_indices
                    if i < len(evidence_figures)
                )

            # Auto-generate category selectivity chart for input_stimuli if data available (only if not already generated)
            category_selectivity_html = ""
            cat_sel_data = _resolve_selectivity_data(investigation)
            if cat_sel_data and cat_sel_data.get("categories") and "category_selectivity_chart" not in figure_types:
                # Use preview figure if available (from two-pass workflow)
                if "category_selectivity_chart" in agent.preview_figures:
                    category_selectivity_html = agent.preview_figures["category_selectivity_chart"]
                else:
                    category_selectivity_html = generate_category_selectivity_chart(
                        cat_sel_data,
                        neuron_id,
                        title="Category Selectivity",
                    )

            # Get output projections HTML (compact version)
            # Use preview figure if available (from two-pass workflow)
            if "output_projections" in agent.preview_figures:
                output_projections_html = agent.preview_figures["output_projections"]
            else:
                output_projections_html = generate_output_projections(
                    output_projections.get("promote", []),
                    output_projections.get("suppress", [])
                )

            # Auto-generate ablation/steering tables for output_function (only if not already in evidence_figures)
            # SKIP if batch summary exists - batch_ablation_summary supersedes ablation_matrix
            ablation_table_html = ""
            steering_table_html = ""
            ablation_data = inv_evidence.get("ablation_effects", [])
            steering_data = investigation.get("steering_results", inv_evidence.get("steering_effects", []))

            # Generate ablation matrix HTML if data exists AND no batch summary already exists
            # (batch_ablation_summary provides better visualization than ablation_matrix)
            if ablation_data and "ablation_matrix" not in figure_types and "batch_ablation_summary" not in figure_types:
                ablation_table_html = generate_ablation_matrix(
                    ablation_data[:8],
                    title="Ablation Effects",
                    caption=""
                )

            # Generate steering curves HTML if data exists AND no batch summary already exists
            if steering_data and "steering_curves" not in figure_types and "batch_steering_summary" not in figure_types:
                steering_table_html = generate_steering_curves(
                    steering_data[:8],
                    highlight_tokens=[],
                    title="Steering Effects",
                    caption=""
                )

            # NOTE: Circuit diagram removed - the three-panel circuit block already shows
            # upstream/downstream neurons with the same data (now using dependency_results
            # as single source of truth). The tool is still available for agent to call
            # if they want a separate diagram, but we don't auto-generate it.

            # Build hypothesis timeline for investigation history (ALWAYS generated from canonical data)
            # Even if the agent generated one via tool, we prefer the auto-generated version
            # because it uses the final investigation data (agent's version may be stale)
            hypothesis_timeline_html = ""
            hypotheses_raw = investigation.get("hypotheses_tested", [])
            try:
                if hypotheses_raw:
                    # Filter out hypotheses that were registered but never tested
                    # (both status and posterior_probability are null)
                    tested_hypotheses = [
                        h for h in hypotheses_raw
                        if h.get("status") is not None or h.get("posterior_probability") is not None
                    ]
                    hypothesis_data = [
                        {
                            "id": h.get("hypothesis_id", f"H{i}"),
                            "text": h.get("hypothesis", ""),
                            "status": h.get("status"),  # None passed through; inferred in renderer
                            "prior": h.get("prior_probability", 50),
                            "posterior": h.get("posterior_probability") or h.get("prior_probability", 50),
                            "evidence_for": h.get("evidence_for", []),
                            "evidence_against": h.get("evidence_against", []),
                        }
                        for i, h in enumerate(tested_hypotheses)
                    ]
                    hypothesis_timeline_html = generate_hypothesis_timeline(
                        hypothesis_data,
                        title="Hypothesis Evolution",
                        caption=""
                    )
                    print(f"  [write_dashboard] Auto-generated hypothesis timeline: {len(hypothesis_timeline_html)} chars, {len(hypothesis_data)} hypotheses")
            except Exception as e:
                print(f"  [write_dashboard] WARNING: hypothesis timeline generation failed: {e}")
                hypothesis_timeline_html = ""

            # Build open questions HTML
            open_questions = investigation.get("open_questions", findings.get("open_questions", []))
            open_questions_html = render_open_questions(open_questions) if open_questions else ""

            # Get section-specific content from section_content if provided
            # Support both new names and legacy names
            input_section = section_content.get("input_function", section_content.get("input_stimuli", {}))
            output_section = section_content.get("output_function", {})
            hypothesis_section = section_content.get("hypothesis_testing", section_content.get("investigation_history", {}))
            open_questions_section = section_content.get("open_questions", {})

            # Map user-friendly prose keys to internal slot names for input_function
            # prose -> prose_after_selectivity (Part 1: behavioral triggers)
            # prose_part2 -> prose_before_wiring (Part 2: upstream circuit architecture)
            if "prose" in input_section and "prose_after_selectivity" not in input_section:
                input_section["prose_after_selectivity"] = input_section["prose"]
            if "prose_part2" in input_section and "prose_before_wiring" not in input_section:
                input_section["prose_before_wiring"] = input_section["prose_part2"]

            # Map user-friendly prose keys to internal slot names for output_function
            # prose / prose_part1 -> prose_after_projections (Part 1: direct token effects)
            # prose_part2 -> prose_before_downstream_wiring (Part 2: downstream circuit effects)
            if "prose" in output_section and "prose_after_projections" not in output_section:
                output_section["prose_after_projections"] = output_section["prose"]
            if "prose_part1" in output_section and "prose_after_projections" not in output_section:
                output_section["prose_after_projections"] = output_section["prose_part1"]
            if "prose_part2" in output_section and "prose_before_downstream_wiring" not in output_section:
                output_section["prose_before_downstream_wiring"] = output_section["prose_part2"]

            # If section_content is empty but prose_sections provided, distribute prose by keyword
            if not section_content and prose_sections:
                input_keywords = ["input", "stimulus", "activat", "selectiv", "fires", "respond", "trigger", "category", "upstream"]
                output_keywords = ["output", "project", "ablat", "steer", "effect", "promot", "suppress", "function", "downstream", "circuit"]
                hypothesis_keywords = ["hypothes", "history", "evolution", "test", "investig", "skeptic", "review", "boundar", "confound"]

                def match_section(heading, content):
                    h_lower = (heading or "").lower()
                    c_lower = (content or "").lower()[:200]
                    text = h_lower + " " + c_lower

                    scores = {
                        "input": sum(1 for k in input_keywords if k in text),
                        "output": sum(1 for k in output_keywords if k in text),
                        "hypothesis": sum(1 for k in hypothesis_keywords if k in text),
                    }
                    best = max(scores, key=scores.get)
                    return best if scores[best] > 0 else "input"  # Default to input

                # Distribute prose_sections to appropriate sections
                section_prose = {"input": [], "output": [], "hypothesis": []}
                for ps in prose_sections:
                    heading = ps.get("heading", "")
                    content = ps.get("content", "")
                    section_type = match_section(heading, content)
                    prose_text = f"<h4>{escape_html(heading)}</h4>" if heading else ""
                    if content:
                        # Process content for markdown and figure placeholders
                        # NOTE: Don't wrap in <p> - convert_markdown_to_html handles that
                        processed = linkify_neuron_ids(escape_html_preserve_tags(content))
                        prose_text += processed
                    section_prose[section_type].append(prose_text)

                # Assign distributed prose
                input_section = {"prose_html": "".join(section_prose["input"])}
                output_section = {"prose_html": "".join(section_prose["output"])}
                hypothesis_section = {"prose_html": "".join(section_prose["hypothesis"])}

            # Track which figure indices have been embedded via prose placeholders
            embedded_figure_indices = set()

            # Storage for figures to insert after markdown conversion
            figure_insertions = {}  # marker_id -> figure_html

            # Helper to convert markdown to HTML and mark FIGURE placeholders
            def process_prose(text):
                if not text:
                    return ""
                # Convert markdown bold/italic to HTML (re imported at module level)
                # Bold: **text** -> <strong>text</strong>
                text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
                # Italic: *text* -> <em>text</em> (but not inside words)
                text = re.sub(r'(?<![a-zA-Z])\*([^*]+)\*(?![a-zA-Z])', r'<em>\1</em>', text)

                # Replace FIGURE placeholders with markers that will survive markdown conversion
                # The actual figure HTML will be inserted AFTER markdown processing
                def mark_figure(match):
                    fig_idx_str = match.group(1) or match.group(2)
                    fig_idx = int(fig_idx_str)
                    if fig_idx < len(evidence_figures):
                        embedded_figure_indices.add(fig_idx)  # Track this figure as embedded
                        fig_html = evidence_figures[fig_idx].get("html", "")
                        fig_type = evidence_figures[fig_idx].get("type", "")
                        # Wrap ablation/steering figures in collapsibles (collapsed by default)
                        if fig_type == "ablation_matrix":
                            fig_html = render_collapsible("Ablation Effects", fig_html, expanded=False)
                        elif fig_type == "steering_curves":
                            fig_html = render_collapsible("Steering Effects", fig_html, expanded=False)
                        # Store figure for later insertion
                        # Use a marker that won't be corrupted by markdown (no underscores or asterisks)
                        marker_id = f"FIGPLACEHOLDER{fig_idx}ENDFIG"
                        figure_insertions[marker_id] = fig_html
                        # Return marker on its own line (so markdown treats it as a block)
                        return f"\n\n{marker_id}\n\n"
                    return ""

                text = re.sub(r'<FIGURE_(\d+)>|&lt;FIGURE_(\d+)&gt;', mark_figure, text)
                return text

            # Helper to replace figure markers with actual HTML (after markdown conversion)
            def insert_figure_html(html_text):
                # First, replace pre-processed markers (from process_prose)
                for marker_id, fig_html in figure_insertions.items():
                    # The marker might be wrapped in <p> tags - remove them and insert figure
                    html_text = html_text.replace(f'<p class="prose">{marker_id}</p>', fig_html)
                    html_text = html_text.replace(marker_id, fig_html)  # Fallback

                # Also handle raw <FIGURE_N> patterns that weren't pre-processed
                # (e.g., in individual prose slots that bypassed process_prose)
                def replace_raw_figure(match):
                    fig_idx_str = match.group(1) or match.group(2)
                    fig_idx = int(fig_idx_str)
                    if fig_idx < len(evidence_figures):
                        fig_html = evidence_figures[fig_idx].get("html", "")
                        fig_type = evidence_figures[fig_idx].get("type", "")
                        # Wrap ablation/steering figures in collapsibles
                        if fig_type == "ablation_matrix":
                            fig_html = render_collapsible("Ablation Effects", fig_html, expanded=False)
                        elif fig_type == "steering_curves":
                            fig_html = render_collapsible("Steering Effects", fig_html, expanded=False)
                        return fig_html
                    return ""  # Remove invalid figure references

                # Match both <FIGURE_N> and HTML-escaped versions, including when wrapped in <p> tags
                html_text = re.sub(
                    r'<p class="prose"><FIGURE_(\d+)></p>|<p class="prose">&lt;FIGURE_(\d+)&gt;</p>',
                    replace_raw_figure, html_text
                )
                html_text = re.sub(
                    r'<FIGURE_(\d+)>|&lt;FIGURE_(\d+)&gt;',
                    replace_raw_figure, html_text
                )
                return html_text

            # Helper for prose HTML
            def get_prose_html(section_dict, key="prose"):
                # Check for pre-built prose_html first (from legacy distribution)
                if section_dict.get("prose_html"):
                    return process_prose(section_dict["prose_html"])
                # Otherwise build from prose text
                # NOTE: Don't wrap in <p> here - convert_markdown_to_html in html_builder handles that
                prose = section_dict.get(key, "")
                if prose:
                    processed = process_prose(linkify_neuron_ids(escape_html_preserve_tags(prose)))
                    return processed
                return ""

            # Helper to filter figures by type (exclude certain types already rendered separately)
            # Also excludes figures already embedded via FIGURE placeholders in prose
            def get_filtered_figures_html(figure_indices, exclude_types):
                if not figure_indices:
                    return ""
                filtered = [
                    evidence_figures[i].get("html", "")
                    for i in figure_indices
                    if i < len(evidence_figures)
                    and evidence_figures[i].get("type") not in exclude_types
                    and i not in embedded_figure_indices
                ]
                return "".join(filtered)

            # Auto-generate upstream/downstream dependency tables if data exists
            upstream_dep_html = ""
            downstream_dep_html = ""
            upstream_dep_data = investigation.get("upstream_dependency_results", [])
            downstream_dep_data = investigation.get("downstream_dependency_results", [])

            # Collect all neuron IDs that need labels
            dep_neuron_ids = []
            if upstream_dep_data:
                data = upstream_dep_data[0] if isinstance(upstream_dep_data, list) else upstream_dep_data
                dep_neuron_ids.extend(data.get("upstream_neurons", []))
                dep_neuron_ids.extend(list(data.get("individual_ablation", {}).keys()))
            if downstream_dep_data:
                data = downstream_dep_data[0] if isinstance(downstream_dep_data, list) else downstream_dep_data
                dep_neuron_ids.extend(data.get("downstream_neurons", []))

            # Build neuron labels dict - start with enriched connectivity
            neuron_labels = {}
            if hasattr(agent, 'enriched_connectivity') and agent.enriched_connectivity:
                for n in agent.enriched_connectivity.get("upstream", []):
                    nid = n.get("id", "")
                    label = n.get("label", "")
                    if nid and label and label != "Unknown":
                        neuron_labels[nid] = label
                for n in agent.enriched_connectivity.get("downstream", []):
                    nid = n.get("id", "")
                    label = n.get("label", "")
                    if nid and label and label != "Unknown":
                        neuron_labels[nid] = label

            # Fetch labels for any dependency neurons not in enriched connectivity
            missing_ids = [nid for nid in dep_neuron_ids if nid and nid not in neuron_labels]
            if missing_ids:
                extra_labels = get_neuron_labels(list(set(missing_ids)))
                neuron_labels.update(extra_labels)

            # Get wiring weights from connectivity for ablation agreement comparison
            wiring_weights = {}
            inv_evidence = investigation.get("evidence", {})
            connectivity = inv_evidence.get("connectivity", {}) if isinstance(inv_evidence, dict) else {}
            for u in connectivity.get("upstream_neurons", []):
                if isinstance(u, dict) and u.get("neuron_id"):
                    wiring_weights[u["neuron_id"]] = u.get("weight", 0)

            # Auto-add wiring polarity table (weight-based upstream connectivity)
            wiring_data = investigation.get("wiring_analysis", {})
            if (wiring_data.get("top_excitatory") or wiring_data.get("top_inhibitory")) and "wiring_polarity_table" not in figure_types:
                # Use preview figure if available (from two-pass workflow)
                if "wiring_polarity_table" in agent.preview_figures:
                    wiring_html = agent.preview_figures["wiring_polarity_table"]
                else:
                    fallback_caption = (
                        "Predicted excitatory/inhibitory inputs based on weight connectivity (c_up, c_gate). "
                        "Compare with RelP-based connectivity which shows actual influence in specific contexts."
                    )
                    fb_stats = wiring_data.get("stats", {})
                    if fb_stats.get("regime_correction_applied"):
                        fallback_caption += (
                            " <strong>Polarity labels have been regime-corrected</strong>"
                            " (target operates in inverted SwiGLU regime)."
                        )
                    elif fb_stats.get("regime_warning"):
                        fallback_caption += f" <em>Warning: {fb_stats['regime_warning']}</em>"
                    wiring_html = generate_wiring_polarity_table(
                        wiring_data,
                        title="Upstream Wiring (Weight-Based Polarity)",
                        caption=fallback_caption,
                    )
                if wiring_html:
                    agent.add_auto_figure("wiring_polarity_table", wiring_html)

            if upstream_dep_data and "upstream_dependency_table" not in figure_types:
                # Use preview figure if available (from two-pass workflow)
                if "upstream_dependency_table" in agent.preview_figures:
                    upstream_dep_html = agent.preview_figures["upstream_dependency_table"]
                else:
                    data = upstream_dep_data[0] if isinstance(upstream_dep_data, list) else upstream_dep_data
                    upstream_dep_html = generate_upstream_dependency_table(
                        data, title="Upstream Dependencies", neuron_labels=neuron_labels,
                        wiring_weights=wiring_weights
                    )

            # Auto-add upstream steering table
            upstream_steering_html = ""
            upstream_steering_data = investigation.get("upstream_steering_results", [])
            if upstream_steering_data:
                steer_data = upstream_steering_data[0] if isinstance(upstream_steering_data, list) else upstream_steering_data
                if steer_data.get("upstream_results"):
                    upstream_steering_html = generate_upstream_steering_table(
                        steer_data, neuron_labels=neuron_labels,
                        title="Upstream Steering Response",
                    )

            # Get wiring weights for downstream neurons from connectivity
            downstream_wiring_weights = {}
            for d in connectivity.get("downstream_neurons", connectivity.get("downstream_targets", [])):
                if isinstance(d, dict) and d.get("neuron_id"):
                    downstream_wiring_weights[d["neuron_id"]] = d.get("weight", 0)
                elif isinstance(d, dict) and d.get("id"):
                    downstream_wiring_weights[d["id"]] = d.get("weight", 0)

            if downstream_dep_data and "downstream_dependency_table" not in figure_types:
                # Use preview figure if available (from two-pass workflow)
                if "downstream_dependency_table" in agent.preview_figures:
                    downstream_dep_html = agent.preview_figures["downstream_dependency_table"]
                else:
                    data = downstream_dep_data[0] if isinstance(downstream_dep_data, list) else downstream_dep_data
                    downstream_dep_html = generate_downstream_dependency_table(
                        data, title="Downstream Ablation Effects", neuron_labels=neuron_labels,
                        wiring_weights=downstream_wiring_weights
                    )

            # Auto-add downstream wiring polarity table (weight-based downstream connectivity)
            output_wiring_data = investigation.get("output_wiring_analysis", {})
            downstream_wiring_html = ""
            if (output_wiring_data.get("top_excitatory") or output_wiring_data.get("top_inhibitory")) and "downstream_wiring_table" not in figure_types:
                # Use preview figure if available (from two-pass workflow)
                if "downstream_wiring_table" in agent.preview_figures:
                    downstream_wiring_html = agent.preview_figures["downstream_wiring_table"]
                else:
                    downstream_wiring_html = generate_downstream_wiring_table(
                        output_wiring_data,
                        title="Downstream Wiring (Weight-Based Polarity)",
                        caption="Predicted downstream neurons this neuron excites/inhibits based on weight connectivity. "
                                "Compare with ablation effects which show actual influence in specific contexts."
                    )
                if downstream_wiring_html:
                    agent.add_auto_figure("downstream_wiring_table", downstream_wiring_html)

            # Auto-generate downstream steering slope table (shows slope + R² per downstream neuron)
            steering_data = investigation.get("multi_token_steering_results", [])
            downstream_steering_slope_html = ""
            # Use preview figure if available (from two-pass workflow)
            if "downstream_steering_slopes" in agent.preview_figures:
                downstream_steering_slope_html = agent.preview_figures["downstream_steering_slopes"]
            else:
                for sr in steering_data:
                    if isinstance(sr, dict) and sr.get("downstream_steering_slopes"):
                        slopes_data = sr["downstream_steering_slopes"]
                        sv_tested = sr.get("steering_values", [])
                        n_prompts = sr.get("total_prompts", 0)
                        downstream_steering_slope_html = generate_downstream_steering_slope_table(
                            slopes_data,
                            neuron_labels=neuron_labels,
                            wiring_weights=downstream_wiring_weights,
                            title="Downstream Steering Response",
                            steering_values_tested=[sv for sv in sv_tested if sv != 0],
                            n_prompts=n_prompts,
                        )
                        break  # Use first result with slopes

            # Auto-generate boundary test cards and alternative hypothesis cards
            boundary_test_html = ""
            alternative_hypo_html = ""
            skeptic = investigation.get("skeptic_report", {})
            if skeptic:
                if skeptic.get("boundary_tests") and "boundary_test_cards" not in figure_types:
                    boundary_test_html = generate_boundary_test_cards(
                        skeptic["boundary_tests"],
                        title="Boundary Tests",
                        show_only_failures=False
                    )
                if skeptic.get("alternative_hypotheses") and "alternative_hypothesis_cards" not in figure_types:
                    alternative_hypo_html = generate_alternative_hypothesis_cards(
                        skeptic["alternative_hypotheses"],
                        title="Alternative Hypotheses Tested"
                    )

            # Process all prose first to populate embedded_figure_indices
            # This must happen BEFORE get_filtered_figures_html to avoid duplicates
            input_prose_html = get_prose_html(input_section)
            output_prose_html = get_prose_html(output_section)
            hypothesis_prose_html = get_prose_html(hypothesis_section)
            open_questions_prose_html = get_prose_html(open_questions_section)

            # Get wiring polarity tables from auto_figures if available
            wiring_polarity_html = agent.auto_figures.get("wiring_polarity_table", "")
            downstream_wiring_html = agent.auto_figures.get("downstream_wiring_table", "")

            # Get new completion-focused figures from preview
            ablation_completions_html = agent.preview_figures.get("ablation_completions", "")
            steering_completions_html = agent.preview_figures.get("steering_completions", "")
            downstream_ablation_effects_html = agent.preview_figures.get("downstream_ablation_effects", "")

            # Build the four main sections
            main_sections = {
                # Input Function section - includes upstream dependencies
                # Split into Part 1 (Behavioral Triggers) and Part 2 (Upstream Circuit Architecture)
                "input_function": render_input_stimuli_section(
                    category_selectivity_html=category_selectivity_html,
                    activation_selectivity_html=get_filtered_figures_html(
                        input_section.get("figures", []),
                        exclude_types={"category_selectivity_chart", "selectivity_gallery", "upstream_dependency_table", "wiring_polarity_table"}
                    ),
                    upstream_dep_html=upstream_dep_html,
                    wiring_polarity_html=wiring_polarity_html,
                    upstream_steering_html=upstream_steering_html,
                    prose_html=input_prose_html,
                    # Part 1 prose slots (Behavioral Triggers)
                    prose_before_selectivity=input_section.get("prose_before_selectivity", ""),
                    prose_after_selectivity=input_section.get("prose_after_selectivity", ""),
                    prose_after_other_figures=input_section.get("prose_after_other_figures", ""),
                    # Part 2 prose slots (Upstream Circuit Architecture)
                    prose_before_wiring=input_section.get("prose_before_wiring", ""),
                    prose_after_wiring=input_section.get("prose_after_wiring", ""),
                    prose_before_ablation=input_section.get("prose_before_ablation", ""),
                    prose_after_ablation=input_section.get("prose_after_ablation", ""),
                    prose_before_steering=input_section.get("prose_before_steering", ""),
                    prose_after_steering=input_section.get("prose_after_steering", ""),
                    prose_part2=input_section.get("prose_part2", ""),  # Simplified Part 2 slot
                    # Part 3: Negative polarity triggers (bipolar only)
                    prose_part3=input_section.get("prose_part3", ""),
                ),
                # Output Function section - includes downstream dependencies and steering propagation
                # Split into Part 1 (Direct Token Effects) and Part 2 (Downstream Circuit Effects)
                "output_function": render_output_function_section(
                    output_projections_html=output_projections_html,
                    ablation_completions_html=ablation_completions_html,
                    steering_completions_html=steering_completions_html,
                    downstream_wiring_html=downstream_wiring_html,
                    downstream_ablation_effects_html=downstream_ablation_effects_html,
                    downstream_steering_slopes_html=downstream_steering_slope_html,
                    # Part 1 prose slots (Direct Token Effects)
                    prose_before_projections=output_section.get("prose_before_projections", ""),
                    prose_after_projections=output_section.get("prose_after_projections", ""),
                    prose_before_ablation=output_section.get("prose_before_ablation", ""),
                    prose_after_ablation=output_section.get("prose_after_ablation", ""),
                    prose_before_steering=output_section.get("prose_before_steering", ""),
                    prose_after_steering=output_section.get("prose_after_steering", ""),
                    # Part 2 prose slots (Downstream Circuit Effects)
                    prose_before_downstream_wiring=output_section.get("prose_before_downstream_wiring", ""),
                    prose_after_downstream_wiring=output_section.get("prose_after_downstream_wiring", ""),
                    prose_before_downstream_ablation=output_section.get("prose_before_downstream_ablation", ""),
                    prose_after_downstream_ablation=output_section.get("prose_after_downstream_ablation", ""),
                    prose_before_downstream_steering=output_section.get("prose_before_downstream_steering", ""),
                    prose_after_downstream_steering=output_section.get("prose_after_downstream_steering", ""),
                    # Simplified prose slots
                    prose_part1=output_section.get("prose_part1", ""),
                    prose_part2=output_section.get("prose_part2", ""),
                    # Part 3: Negative polarity effects (bipolar only)
                    prose_part3=output_section.get("prose_part3", ""),
                    # Legacy parameters for backward compatibility
                    ablation_table_html=ablation_table_html,
                    steering_table_html=steering_table_html,
                    downstream_dep_html=downstream_dep_html,
                    prose_html=output_prose_html
                ),
                # Hypothesis Testing section
                "hypothesis_testing": render_hypothesis_testing_section(
                    boundary_test_html=boundary_test_html,
                    alternative_hypo_html=alternative_hypo_html,
                    hypothesis_timeline_html=hypothesis_timeline_html + get_filtered_figures_html(
                        hypothesis_section.get("figures", []),
                        exclude_types={"hypothesis_timeline", "boundary_test_cards", "alternative_hypothesis_cards"}
                    ),
                    prose_html=data_grounding_note + hypothesis_prose_html
                ),
                # Open Questions section
                "open_questions": render_open_questions_section(
                    open_questions_html=open_questions_html,
                    prose_html=open_questions_prose_html
                ),
            }

            # Freeform HTML is no longer used (four-section structure is always used)
            freeform_html = ""

            # Assemble page (pass main_sections if using new structure)
            html = assemble_page(
                neuron_id=neuron_id,
                title=args.get("title", "Neuron Investigation"),
                fixed_sections=fixed_sections,
                freeform_html=freeform_html,
                main_sections=main_sections
            )

            # Replace figure markers with actual figure HTML
            # This must happen AFTER markdown conversion (which happens in render_ functions)
            html = insert_figure_html(html)

            # SAFETY NET: If hypothesis timeline is missing from final HTML but data exists,
            # inject it into the Hypothesis Testing section. This guards against agent
            # nondeterminism or intermediate errors that silently drop the timeline.
            if 'class="timeline-item' not in html and hypotheses_raw:
                try:
                    tested_hypotheses = [
                        h for h in hypotheses_raw
                        if h.get("status") is not None or h.get("posterior_probability") is not None
                    ]
                    if tested_hypotheses:
                        fallback_data = [
                            {
                                "id": h.get("hypothesis_id", f"H{i}"),
                                "text": h.get("hypothesis", ""),
                                "status": h.get("status"),
                                "prior": h.get("prior_probability", 50),
                                "posterior": h.get("posterior_probability") or h.get("prior_probability", 50),
                            }
                            for i, h in enumerate(tested_hypotheses)
                        ]
                        fallback_html = generate_hypothesis_timeline(
                            fallback_data,
                            title="Hypothesis Evolution",
                            caption=""
                        )
                        if fallback_html:
                            # Insert right after <div class="main-section-content"> inside Hypothesis Testing
                            # The section structure is: ...Hypothesis Testing</div>\n        </div>\n        <div class="main-section-content">\n            {content}
                            # We look for the main-section-content div that follows the Hypothesis Testing title
                            # Note: re is imported at module level — do NOT re-import here (breaks nested function closures)
                            pattern = r'(Hypothesis Testing</div>\s*</div>\s*<div class="main-section-content">\s*)'
                            match = re.search(pattern, html)
                            if match:
                                insert_pos = match.end()
                                html = html[:insert_pos] + fallback_html + '\n' + html[insert_pos:]
                                print(f"  [write_dashboard] SAFETY NET: Injected hypothesis timeline ({len(fallback_data)} hypotheses)")
                            else:
                                print("  [write_dashboard] SAFETY NET: Could not find injection point in HTML")
                except Exception as e:
                    print(f"  [write_dashboard] SAFETY NET failed: {e}")

            # Write file
            safe_id = neuron_id.replace("/", "_")
            output_path = output_dir / f"{safe_id}.html"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "output_path": str(output_path),
                    "neuron_id": neuron_id,
                    "title": args.get("title", ""),
                    "figures_generated": len(agent.generated_figures),
                })}]
            }

        # =================================================================
        # AUTO-GENERATE FIGURES TOOL (Convenience - auto-injects data)
        # =================================================================

        @tool(
            "auto_generate_figures",
            """Generate SUPPLEMENTARY figures only. Most figures are AUTO-INSERTED—don't duplicate them!

            AUTO-INSERTED (DO NOT GENERATE—they appear automatically):
            - Category selectivity chart (Input Function)
            - Upstream/downstream wiring tables (Input/Output Function)
            - Ablation completions gallery (Output Function Part 1)
            - Intelligent steering gallery (Output Function Part 1)
            - Downstream ablation/steering effects tables (Output Function Part 2)
            - Hypothesis timeline (Hypothesis Testing)

            THIS TOOL GENERATES (supplementary figures only):
            - activation_grid: Activation examples grid
            - steering_curves: Dose-response curves
            - batch_ablation_summary: Ablation statistics summary
            - batch_steering_summary: Steering statistics summary
            - skeptic_card: Skeptic verdict card

            Use <FIGURE_N> in your prose to place these supplementary figures.
            Returns a mapping of figure_type -> figure_index.""",
            {}
        )
        async def tool_auto_generate_figures(args: dict[str, Any]) -> dict[str, Any]:
            """Auto-generate figures from investigation data."""
            evidence = investigation.get("evidence", {})
            characterization = investigation.get("characterization", {})
            generated = {}

            # 1. Activation Grid - from positive/negative examples
            positive = evidence.get("activating_prompts", [])
            negative = evidence.get("non_activating_prompts", [])
            if positive and len(positive) >= 3:
                high_examples = [
                    {"text": ex.get("prompt", "")[:150], "activation": ex.get("activation", 0), "token": ex.get("token", "")}
                    for ex in positive[:6]
                ]
                low_examples = [
                    {"text": ex.get("prompt", "")[:150], "activation": ex.get("activation", 0)}
                    for ex in negative[:6]
                ] if negative else []

                html = generate_activation_grid(
                    high_examples, low_examples,
                    title="Activation Selectivity",
                    caption="Examples showing when the neuron fires (left) vs remains inactive (right)"
                )
                agent.generated_figures.append({"html": html, "type": "activation_grid"})
                generated["activation_grid"] = len(agent.generated_figures) - 1

            # 2. Hypothesis Timeline - SKIPPED (generated by write_dashboard in Investigation History section)
            # This prevents duplicate hypothesis timeline figures

            # 3. Token Bar Chart - REMOVED (duplicates Output Projections visualization)
            # The compact output-projections display shows the same data more cleanly

            # 4. Steering Curves - from dose-response data
            # Pass original data directly - generate_steering_curves handles multiple formats
            dose_response = investigation.get("dose_response_results", [])
            if not dose_response:
                dose_response = investigation.get("steering_results", evidence.get("steering_results", []))

            if dose_response and len(dose_response) >= 1:
                # Pass original data - generate_steering_curves handles:
                # - dose_response_curve nested format
                # - promoted_tokens/suppressed_tokens
                # - promotes/suppresses
                html = generate_steering_curves(
                    dose_response[:6], [],
                    title="Steering Response",
                    caption="How neuron activation level affects model outputs"
                )
                # Only add if actual content was generated (not "Insufficient" message)
                if "Insufficient" not in html:
                    agent.generated_figures.append({"html": html, "type": "steering_curves"})
                    generated["steering_curves"] = len(agent.generated_figures) - 1

            # 5. Batch Ablation Results - REQUIRED for output phase
            batch_ablation = investigation.get("multi_token_ablation_results", [])
            legacy_ablation = evidence.get("ablation_effects", [])

            # Use batch ablation summary if available (shows total_prompts, change_rate)
            if batch_ablation:
                html = generate_batch_ablation_summary(
                    batch_ablation,
                    title="Batch Ablation Results",
                    caption="Effect of ablating this neuron across multiple prompts"
                )
                if html:
                    agent.generated_figures.append({"html": html, "type": "batch_ablation_summary"})
                    generated["batch_ablation_summary"] = len(agent.generated_figures) - 1

            # 6. Batch Steering Results - REQUIRED for output phase
            batch_steering = investigation.get("multi_token_steering_results", [])
            if batch_steering:
                html = generate_batch_steering_summary(
                    batch_steering,
                    title="Batch Steering Results",
                    caption="Effect of steering this neuron across multiple prompts"
                )
                if html:
                    agent.generated_figures.append({"html": html, "type": "batch_steering_summary"})
                    generated["batch_steering_summary"] = len(agent.generated_figures) - 1

            # NOTE: The following figures are AUTO-INSERTED at fixed positions and should NOT
            # be generated here to avoid duplicates:
            # - ablation_completions (auto-inserted in Output Function Part 1)
            # - steering_completions / intelligent_steering_gallery (auto-inserted in Output Function Part 1)
            # - downstream_ablation_effects (auto-inserted in Output Function Part 2)
            # - downstream_steering_slopes (auto-inserted in Output Function Part 2)
            # - ablation_matrix (DEPRECATED - removed entirely)

            # Skeptic Evidence Card - if skeptic report available
            skeptic_report = investigation.get("skeptic_report")
            if skeptic_report and skeptic_report.get("verdict"):
                verdict = skeptic_report.get("verdict", "UNKNOWN")
                challenges = skeptic_report.get("key_challenges", [])
                selectivity = skeptic_report.get("metrics", {}).get("selectivity_score", 0)

                finding = f"Skeptic Verdict: {verdict}"
                evidence_text = f"Selectivity: {selectivity:.2f}\n"
                if challenges:
                    evidence_text += "Challenges:\n" + "\n".join(f"• {c}" for c in challenges[:3])

                html = generate_evidence_card(
                    finding=finding,
                    evidence_type="confirmation" if verdict == "SUPPORTED" else "challenge",
                    supporting_data=evidence_text
                )
                agent.generated_figures.append({"html": html, "type": "skeptic_card"})
                generated["skeptic_card"] = len(agent.generated_figures) - 1

            # 7. Circuit Diagram - SKIPPED (generated by write_dashboard in Connectivity section)
            # This prevents duplicate circuit diagrams

            # 8. Selectivity Gallery - SKIPPED (redundant with category_selectivity_chart)
            # The stacked area chart provides better visualization of category selectivity

            # 9. Homograph Comparison - from homograph_tests
            homograph_tests = investigation.get("homograph_tests", [])
            if homograph_tests and len(homograph_tests) >= 1:
                pairs = []
                for test in homograph_tests[:4]:
                    word = test.get("word", "")
                    contexts = test.get("contexts", [])
                    if word and len(contexts) >= 2:
                        pairs.append({
                            "word": word,
                            "contexts": [
                                {
                                    "label": ctx.get("label", ""),
                                    "example": ctx.get("example", "")[:50],
                                    "activation": ctx.get("activation", 0),
                                    "category": ctx.get("category", "neutral")
                                }
                                for ctx in contexts[:2]
                            ]
                        })

                if pairs:
                    html = generate_homograph_comparison(
                        pairs,
                        title="Homograph Discrimination",
                        caption="How the neuron distinguishes different meanings of the same word"
                    )
                    agent.generated_figures.append({"html": html, "type": "homograph_comparison"})
                    generated["homograph_comparison"] = len(agent.generated_figures) - 1

            # 10. Patching Comparison - from patching_experiments
            patching_experiments = investigation.get("patching_experiments", [])
            if patching_experiments and len(patching_experiments) >= 1:
                experiments = []
                for exp in patching_experiments[:6]:
                    experiments.append({
                        "source_prompt": exp.get("source_prompt", "")[:60],
                        "target_prompt": exp.get("target_prompt", "")[:60],
                        "source_activation": exp.get("source_activation", 0),
                        "target_activation": exp.get("target_activation", 0),
                        "promoted_tokens": exp.get("promoted_tokens", [])[:3],
                        "suppressed_tokens": exp.get("suppressed_tokens", [])[:3],
                        "max_shift": exp.get("max_shift", 0),
                    })

                if experiments:
                    html = generate_patching_comparison(
                        experiments,
                        title="Counterfactual Patching",
                        caption="What happens when activation from high-activation prompts is transferred to low-activation contexts"
                    )
                    agent.generated_figures.append({"html": html, "type": "patching_comparison"})
                    generated["patching_comparison"] = len(agent.generated_figures) - 1

            # 11. Category Selectivity Chart - SKIPPED (generated by write_dashboard in Input Stimuli section)
            # This prevents duplicate category selectivity charts

            # Summary
            omitted = []
            if "activation_grid" not in generated:
                omitted.append("activation_grid (need >= 3 activating examples)")
            if "steering_curves" not in generated:
                omitted.append("steering_curves (need >= 1 dose-response data point)")
            if "ablation_matrix" not in generated:
                omitted.append("ablation_matrix (need >= 2 ablation experiments)")
            if "skeptic_card" not in generated and investigation.get("skeptic_report"):
                omitted.append("skeptic_card (skeptic report missing verdict)")
            if "homograph_comparison" not in generated:
                omitted.append("homograph_comparison (need >= 1 homograph test)")
            if "patching_comparison" not in generated:
                omitted.append("patching_comparison (need >= 1 patching experiment)")
            # Note: hypothesis_timeline, circuit_diagram, selectivity_gallery, and category_selectivity_chart
            # are generated by write_dashboard, not auto_generate_figures

            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "figures_generated": generated,
                    "total_figures": len(generated),
                    "omitted_due_to_missing_data": omitted,
                    "usage": "Reference figures in prose using <FIGURE_N> where N is the figure_index",
                }, indent=2)}]
            }

        # =================================================================
        # CUSTOM VISUALIZATION TOOL (Creative freedom)
        # =================================================================

        @tool(
            "generate_custom_visualization",
            """Create a custom visualization with your own HTML/CSS.

            This tool gives you CREATIVE FREEDOM to design novel visualizations
            that aren't covered by the pre-built figure types. Use this when you
            want to create something unique for this specific neuron's story.

            **Encouraged uses:**
            - RelP vs Steering agreement comparisons (which predictions matched?)
            - Dose-response scatter plots or trends
            - Upstream/downstream signal flow animations
            - Category activation heatmaps
            - Causal pathway diagrams
            - Any novel visualization that tells the neuron's story

            **HTML capabilities:**
            - SVG for custom graphics (lines, circles, paths, text)
            - CSS Grid/Flexbox for complex layouts
            - Inline styles or custom classes (prefix with 'cv-')
            - Basic JavaScript for interactivity

            **Parameters:**
            - title: Visualization title
            - html_content: Your HTML (can include SVG, divs, etc.)
            - css: Custom CSS styles (will be auto-scoped)
            - caption: Explanation of what the visualization shows

            **Example - RelP vs Steering Agreement:**
            ```
            html_content: '''
            <div class="cv-grid">
              <div class="cv-row cv-agree">
                <span class="cv-neuron">L5/N5772</span>
                <span class="cv-check">✓</span>
                <span class="cv-desc">Both excitatory (RelP: +0.26, Steering: +2252)</span>
              </div>
              <div class="cv-row cv-disagree">
                <span class="cv-neuron">L3/N305</span>
                <span class="cv-x">✗</span>
                <span class="cv-desc">RelP excitatory but Steering inhibitory</span>
              </div>
            </div>
            '''
            css: '''
            .cv-grid { display: flex; flex-direction: column; gap: 8px; }
            .cv-row { display: flex; align-items: center; gap: 12px; padding: 12px; border-radius: 8px; }
            .cv-agree { background: #dcfce7; border-left: 4px solid #22c55e; }
            .cv-disagree { background: #fef2f2; border-left: 4px solid #ef4444; }
            .cv-neuron { font-family: monospace; font-weight: 600; }
            .cv-check { color: #22c55e; font-size: 18px; }
            .cv-x { color: #ef4444; font-size: 18px; }
            '''
            ```""",
            {
                "title": str,
                "html_content": str,
                "css": str,
                "caption": str,
            }
        )
        async def tool_custom_visualization(args: dict[str, Any]) -> dict[str, Any]:
            html = generate_custom_visualization(
                title=args.get("title", "Custom Visualization"),
                html_content=args.get("html_content", ""),
                css=args.get("css", ""),
                caption=args.get("caption", ""),
            )
            agent.generated_figures.append({"html": html, "type": "custom_visualization"})
            return {"content": [{"type": "text", "text": json.dumps({
                "success": True,
                "figure_index": len(agent.generated_figures) - 1,
                "note": "Custom visualization created - be creative!"
            })}]}

        # Tool to access negative polarity investigation data (for bipolar reports)
        @tool(
            "get_negative_data",
            "Get the negative-polarity investigation data for bipolar neuron reports. "
            "Returns characterization, evidence, and selectivity data from the negative firing investigation. "
            "Only available when a negative investigation was provided.",
            {}
        )
        async def tool_get_negative_data(args: dict[str, Any]) -> dict[str, Any]:
            neg = agent.negative_investigation_data
            if neg is None:
                return {"content": [{"type": "text", "text": json.dumps({
                    "error": "No negative investigation available. This is a single-polarity report."
                })}]}

            neg_char = neg.get("characterization", {})
            neg_evidence = neg.get("evidence", {})

            summary = {
                "polarity_mode": neg.get("polarity_mode", "negative"),
                "neuron_id": neg.get("neuron_id", ""),
                "final_hypothesis": neg_char.get("final_hypothesis", ""),
                "input_function": neg_char.get("input_function", ""),
                "output_function": neg_char.get("output_function", ""),
                "function_type": neg_char.get("function_type", ""),
                "confidence": neg.get("confidence", 0),
                "key_findings": neg.get("key_findings", []),
                "open_questions": neg.get("open_questions", []),
                "activating_prompts": neg_evidence.get("activating_prompts", [])[:10],
                "ablation_effects": neg_evidence.get("ablation_effects", [])[:5],
                "steering_results": neg.get("steering_results", [])[:5],
                "output_projections": neg.get("output_projections", {}),
                "hypotheses_tested": neg.get("hypotheses_tested", []),
                "category_selectivity_data": neg.get("category_selectivity_data", []),
                "skeptic_report": neg.get("skeptic_report"),
            }

            return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

        tools_list = [
            tool_get_full_data,
            tool_auto_generate_figures,  # Convenience tool for auto-populating figures
            tool_custom_visualization,   # Creative freedom for novel visualizations
            tool_lookup_neuron_descriptions,
            tool_activation_grid,
            tool_hypothesis_timeline,
            # NOTE: circuit_diagram removed - redundant with three-panel circuit block
            tool_selectivity_gallery,
            tool_ablation_matrix,
            tool_steering_curves,
            tool_evidence_card,
            tool_anomaly_box,
            tool_homograph_comparison,
            tool_stacked_density_chart,
            tool_patching_comparison,
            tool_write_dashboard,
        ]

        # Only include get_negative_data tool when negative investigation exists
        if agent.negative_investigation_data is not None:
            tools_list.insert(1, tool_get_negative_data)  # After get_full_data

        return tools_list

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agent."""
        neuron_id = self.investigation_data.get("neuron_id", "")

        return f"""Write a scientific article about neuron {neuron_id}.

## Process (Two-Pass Workflow)

**CRITICAL: Follow this order exactly:**

1. `get_full_data` → Load the investigation data
2. `preview_visualizations` → **REQUIRED** See all auto-generated tables/charts BEFORE writing
3. `auto_generate_figures` → Generate intelligent steering gallery and other figures
4. `generate_hypothesis_timeline` → Create COMPLETE hypothesis evolution (ALL hypotheses tested)
5. `generate_batch_ablation_summary` and `generate_batch_steering_summary` → Batch experiment results
6. (Optional) `generate_custom_visualization` → Custom figures for uncovered data
7. `write_dashboard` → Assemble the HTML (REQUIRED)

**Why preview_visualizations?** This tool generates category selectivity charts, wiring tables, dependency tables, and output projections, then returns SUMMARIES of each. This lets you SEE what visualizations will appear in the dashboard before writing prose, so you can reference and explain them coherently.

## Your Freedom

Write naturally. You decide:
- What to say and how to say it
- Where to place prose relative to figures
- What aspects to emphasize
- How to structure your narrative

The investigation data contains rich evidence about this neuron's input triggers, output effects, upstream/downstream dependencies, and hypothesis testing. Examine it carefully and write a compelling scientific article.

## Required Visualizations

These MUST appear (preview_visualizations auto-generates most, or call the tool):
- Category selectivity chart (auto-generated)
- Upstream wiring table (auto-generated if wiring_analysis exists)
- Upstream dependency table (auto-generated if data exists)
- Output projections (auto-generated)
- Downstream wiring table (auto-generated if output_wiring_analysis exists)
- Batch ablation summary
- Batch steering summary
- Downstream dependency table (auto-generated if data exists)
- Hypothesis evolution timeline (with ALL hypotheses)

## write_dashboard Parameters

```
title: "Creative 2-4 word title"
narrative_lead: "Opening sentence about this neuron"
narrative_body: "Additional context"
section_content: JSON with input_function, output_function, hypothesis_testing, open_questions
```

Tell the story of this investigation. What makes this neuron interesting? What did the experiments reveal?
""" + self._bipolar_prompt_section()

    def _bipolar_prompt_section(self) -> str:
        """Generate additional prompt section for bipolar (merged) dashboards."""
        if self.negative_investigation_data is None:
            return ""

        neg = self.negative_investigation_data
        neg_char = neg.get("characterization", {})
        neg_hypothesis = neg_char.get("final_hypothesis", "unknown")
        neg_input = neg_char.get("input_function", "unknown")
        neg_output = neg_char.get("output_function", "unknown")

        return f"""

## BIPOLAR NEURON — MERGED REPORT

This neuron has been investigated for BOTH positive and negative firing. You must write
a report that covers both polarities. Use `get_negative_data` to load the negative investigation.

**Negative firing function:**
- Hypothesis: {neg_hypothesis}
- Input function: {neg_input}
- Output function: {neg_output}

### How to include negative polarity in write_dashboard

The Input Function and Output Function sections each have a **Part 3 slot** for negative polarity.
Use these `prose_part3` fields in your `section_content`:

```
"input_function": {{
    "prose_after_selectivity": "... positive firing triggers ...",
    "prose_part2": "... upstream circuit ...",
    "prose_part3": "Write 2-3 paragraphs about what triggers NEGATIVE firing. "
                   "Include: what categories/prompts cause the most negative activation, "
                   "how strong the negative signal is vs positive, and key differences "
                   "in what triggers each polarity."
}},
"output_function": {{
    "prose_part1": "... positive output effects ...",
    "prose_part2": "... downstream circuit ...",
    "prose_part3": "Write 2-3 paragraphs about what happens when the neuron fires NEGATIVELY. "
                   "Include: how output projections flip (promoted become suppressed), "
                   "steering effects in the negative direction, and how the two polarities "
                   "relate (opposite poles of an axis? disjoint features?)."
}}
```

These Part 3 slots render as "Negative Firing Triggers" and "Negative Firing Effects" subsections.
They appear AFTER the upstream/downstream circuit parts, giving the reader a complete picture:
  Part 1: What triggers positive firing / What positive firing does
  Part 2: Upstream circuit / Downstream circuit
  Part 3: What triggers negative firing / What negative firing does

**IMPORTANT**: You MUST populate both `prose_part3` fields when negative data is available.
"""

    async def generate(self) -> Path:
        """Generate HTML dashboard using Claude Agent SDK.

        Returns:
            Path to generated HTML file
        """
        print(f"Loading data from {self.investigation_path}")
        self._load_data()

        neuron_id = self.investigation_data.get("neuron_id", "unknown")

        print(f"Generating HTML dashboard for {neuron_id}")
        start_time = time.time()

        # Reset generated figures
        self.generated_figures = []

        # Create MCP tools and server
        tools = self._create_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="dashboard_tools_v2",
            version="2.0.0",
            tools=tools,
        )

        # Build initial prompt
        initial_prompt = self._build_initial_prompt()

        # Configure options
        # Generate model-aware system prompt
        model_config = get_model_config()
        total_neurons = model_config.num_layers * model_config.neurons_per_layer
        if total_neurons >= 1_000_000_000:
            total_str = f"{total_neurons / 1_000_000_000:.1f} billion"
        else:
            total_str = f"{total_neurons / 1_000_000:.0f} million"
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            model_name=model_config.name,
            num_layers=model_config.num_layers,
            max_layer=model_config.num_layers - 1,
            neurons_per_layer=f"{model_config.neurons_per_layer:,}",
            total_neurons=total_str,
        )

        # Inject scientist's conclusions into system prompt so the agent
        # anchors its narrative on the investigation findings
        characterization = self.investigation_data.get("characterization", {})
        key_findings = self.investigation_data.get("key_findings", [])
        scientist_block = ""
        final_hyp = characterization.get("final_hypothesis", "")
        input_fn = characterization.get("input_function", "")
        output_fn = characterization.get("output_function", "")
        if final_hyp or input_fn:
            findings_text = "\n".join(f"  - {f}" for f in key_findings[:5]) if key_findings else "  (none recorded)"
            scientist_block = f"""

## SCIENTIST'S VERIFIED CONCLUSIONS (MANDATORY — base your article on these)

The neuron scientist ran {self.investigation_data.get('total_experiments', 'many')} experiments and reached these conclusions. Your article MUST be consistent with these findings:

**Final Hypothesis:** {final_hyp}

**Input Function:** {input_fn}

**Output Function:** {output_fn}

**Key Findings:**
{findings_text}

Do NOT contradict these conclusions. If the raw data (e.g., output projections, selectivity z-scores) seems to conflict, explain the discrepancy rather than overriding the scientist's interpretation — the scientist had access to causal experiments (steering, ablation) that are more reliable than correlational data.
"""
            system_prompt += scientist_block

        # Store transcripts in separate directory to avoid cluttering main project
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "neuron_reports" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        allowed = [
            "mcp__dashboard_tools_v2__get_full_data",
            "mcp__dashboard_tools_v2__auto_generate_figures",  # Convenience tool
            "mcp__dashboard_tools_v2__lookup_neuron_descriptions",
            "mcp__dashboard_tools_v2__generate_activation_grid",
            "mcp__dashboard_tools_v2__generate_hypothesis_timeline",
            # NOTE: circuit_diagram removed - redundant with three-panel circuit block
            "mcp__dashboard_tools_v2__generate_selectivity_gallery",
            "mcp__dashboard_tools_v2__generate_ablation_matrix",
            "mcp__dashboard_tools_v2__generate_steering_curves",
            "mcp__dashboard_tools_v2__generate_evidence_card",
            "mcp__dashboard_tools_v2__generate_anomaly_box",
            "mcp__dashboard_tools_v2__generate_homograph_comparison",
            "mcp__dashboard_tools_v2__generate_stacked_density_chart",
            "mcp__dashboard_tools_v2__generate_patching_comparison",
            "mcp__dashboard_tools_v2__generate_category_selectivity_chart",
            "mcp__dashboard_tools_v2__write_dashboard",
        ]
        if self.negative_investigation_data is not None:
            allowed.insert(1, "mcp__dashboard_tools_v2__get_negative_data")

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=50,  # Allow sufficient turns - Opus tends to explore extensively
            model=self.model,
            mcp_servers={"dashboard_tools_v2": mcp_server},
            cwd=transcripts_dir,
            allowed_tools=allowed,
        )

        output_path = None

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                preview = block.text[:100].replace("\n", " ")
                                print(f"Agent: {preview}...")
                            elif isinstance(block, ToolUseBlock):
                                print(f"Tool: {block.name}")

                    elif isinstance(message, ToolResultBlock):
                        try:
                            result = json.loads(message.content)
                            if result.get("success") and result.get("output_path"):
                                output_path = Path(result["output_path"])
                                print(f"Generated: {output_path}")
                                print(f"  Figures: {result.get('figures_generated', 0)}")
                        except (json.JSONDecodeError, TypeError):
                            pass

                    elif isinstance(message, ResultMessage):
                        if message.subtype == "error":
                            print(f"Error: {message}")

        except Exception as e:
            print(f"Agent error: {e}")
            import traceback
            traceback.print_exc()

        duration = time.time() - start_time
        print(f"Dashboard generation complete in {duration:.1f}s")

        if output_path and output_path.exists():
            return output_path
        else:
            # Fall back to expected path
            safe_id = neuron_id.replace("/", "_")
            return self.output_dir / f"{safe_id}.html"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_dashboard_v2(
    investigation_path: Path,
    output_dir: Path = Path("frontend/reports"),
    model: str = "sonnet",
    # Legacy parameter for backward compatibility
    dashboard_path: Path | None = None,
    negative_investigation_path: Path | None = None,
) -> Path:
    """Generate HTML dashboard from investigation JSON.

    Args:
        investigation_path: Path to investigation JSON (primary input)
        output_dir: Output directory
        model: Model to use
        dashboard_path: DEPRECATED - ignored, kept for backward compatibility
        negative_investigation_path: Optional path to negative-polarity investigation.
            When provided, generates a merged bipolar dashboard with both sections.

    Returns:
        Path to generated HTML
    """
    agent = DashboardAgentV2(
        investigation_path=investigation_path,
        output_dir=output_dir,
        model=model,
        negative_investigation_path=negative_investigation_path,
    )
    return await agent.generate()


def generate_dashboard_v2_sync(
    investigation_path: Path,
    output_dir: Path = Path("frontend/reports"),
    model: str = "sonnet",
    # Legacy parameter for backward compatibility
    dashboard_path: Path | None = None,
    negative_investigation_path: Path | None = None,
) -> Path:
    """Synchronous wrapper for generate_dashboard_v2."""
    return asyncio.run(generate_dashboard_v2(
        investigation_path, output_dir, model,
        negative_investigation_path=negative_investigation_path,
    ))
