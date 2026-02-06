"""Generate a static HTML report showcasing autointerp results for a run.

Includes example prompts for every LLM call type (interpretation, detection, fuzzing, intruder)
so you can sense-check tokenization and formatting.

Usage:
    python -m spd.autointerp.scripts.generate_report <wandb_path>
"""

import json
import random
from pathlib import Path
from typing import Any

import markdown
import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup
from spd.autointerp.eval.intruder import (
    N_REAL,
    DensityIndex,
    _build_prompt,
    _sample_intruder,
)
from spd.autointerp.interpret import get_architecture_info
from spd.autointerp.loaders import find_latest_results_path
from spd.autointerp.schemas import get_autointerp_dir
from spd.autointerp.scoring.detection import (
    N_ACTIVATING,
    N_NON_ACTIVATING,
    _build_detection_prompt,
    _format_example_with_center_token,
    _sample_activating_examples,
    _sample_non_activating_examples,
)
from spd.autointerp.scoring.fuzzing import (
    N_CORRECT,
    N_INCORRECT,
    _build_fuzzing_prompt,
    _delimit_high_ci_tokens,
    _delimit_random_low_ci_tokens,
)
from spd.harvest.loaders import load_all_components, load_harvest_ci_threshold
from spd.harvest.schemas import ComponentData
from spd.harvest.storage import TokenStatsStorage
from spd.utils.wandb_utils import parse_wandb_run_path


def _pick_component_with_label(
    components: list[ComponentData],
    labels: dict[str, str],
    rng: random.Random,
    min_examples: int = 10,
) -> ComponentData:
    """Pick a random component that has a label and enough examples."""
    eligible = [
        c
        for c in components
        if c.component_key in labels and len(c.activation_examples) >= min_examples
    ]
    assert eligible, "No components with labels and enough examples"
    return rng.choice(eligible)


def _build_example_detection_prompt(
    component: ComponentData,
    all_components: list[ComponentData],
    lookup: dict[int, str],
    label: str,
    rng: random.Random,
) -> str:
    activating = _sample_activating_examples(component, N_ACTIVATING, rng)
    non_activating = _sample_non_activating_examples(
        component, all_components, N_NON_ACTIVATING, rng
    )

    formatted: list[tuple[str, bool]] = []
    for ex in activating:
        formatted.append((_format_example_with_center_token(ex, lookup), True))
    for ex in non_activating:
        formatted.append((_format_example_with_center_token(ex, lookup), False))
    rng.shuffle(formatted)
    return _build_detection_prompt(label, formatted)


def _build_example_fuzzing_prompt(
    component: ComponentData,
    lookup: dict[int, str],
    label: str,
    rng: random.Random,
    ci_threshold: float,
) -> str:
    sampled = rng.sample(component.activation_examples, N_CORRECT + N_INCORRECT)
    correct_examples = sampled[:N_CORRECT]
    incorrect_examples = sampled[N_CORRECT:]

    formatted: list[tuple[str, bool]] = []
    for ex in correct_examples:
        text, _ = _delimit_high_ci_tokens(ex, lookup, ci_threshold)
        formatted.append((text, True))
    for ex in incorrect_examples:
        _, n_delimited = _delimit_high_ci_tokens(ex, lookup, ci_threshold)
        text = _delimit_random_low_ci_tokens(ex, lookup, max(n_delimited, 1), rng, ci_threshold)
        formatted.append((text, False))
    rng.shuffle(formatted)
    return _build_fuzzing_prompt(label, formatted)


def _build_example_intruder_prompt(
    component: ComponentData,
    all_components: list[ComponentData],
    lookup: dict[int, str],
    rng: random.Random,
    ci_threshold: float,
) -> str:
    density_index = DensityIndex(all_components, min_examples=N_REAL + 1, ci_threshold=ci_threshold)
    real_examples = rng.sample(component.activation_examples, N_REAL)
    intruder = _sample_intruder(component, density_index, rng)
    intruder_pos = rng.randint(0, N_REAL)
    return _build_prompt(real_examples, intruder, intruder_pos, lookup, ci_threshold)


def generate_report(wandb_path: str, output_path: Path | None = None) -> Path:
    _, _, run_id = parse_wandb_run_path(wandb_path)
    rng = random.Random(42)

    autointerp_dir = get_autointerp_dir(run_id)

    # Load interpretation results
    results_path = find_latest_results_path(run_id)
    assert results_path is not None, f"No interpretation results found in {autointerp_dir}"
    interp_results: list[dict[str, str]] = []
    with open(results_path) as f:
        for line in f:
            interp_results.append(json.loads(line))

    high = [r for r in interp_results if r["confidence"] == "high"]
    med = [r for r in interp_results if r["confidence"] == "medium"]
    low = [r for r in interp_results if r["confidence"] == "low"]

    # Load components and tokenizer for prompt examples
    arch = get_architecture_info(wandb_path)
    tokenizer = AutoTokenizer.from_pretrained(arch.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    lookup = build_token_lookup(tokenizer, arch.tokenizer_name)

    components = load_all_components(run_id)
    ci_threshold = load_harvest_ci_threshold(run_id)

    labels = {r["component_key"]: r["label"] for r in interp_results}

    # Load token stats for interpretation prompt example
    from spd.autointerp.config import CompactSkepticalConfig
    from spd.autointerp.strategies.dispatch import format_prompt
    from spd.harvest.analysis import get_input_token_stats, get_output_token_stats
    from spd.harvest.schemas import get_correlations_dir

    correlations_dir = get_correlations_dir(run_id)
    token_stats_path = correlations_dir / "token_stats.pt"
    has_token_stats = token_stats_path.exists()
    token_stats = TokenStatsStorage.load(token_stats_path) if has_token_stats else None

    # Try loading intruder eval results
    intruder_dir = autointerp_dir / "eval" / "intruder"
    if not intruder_dir.exists():
        intruder_dir = autointerp_dir / "scoring" / "intruder"
    intruder_results: list[dict[str, Any]] = []
    intruder_files = sorted(intruder_dir.glob("results_*.jsonl")) if intruder_dir.exists() else []
    if intruder_files:
        with open(intruder_files[-1]) as f:
            for line in f:
                intruder_results.append(json.loads(line))

    # Try loading detection/fuzzing results
    detection_results: list[dict[str, Any]] = []
    for scoring_dir in [autointerp_dir / "scoring" / "detection"]:
        if scoring_dir.exists():
            files = sorted(scoring_dir.glob("results_*.jsonl"))
            if files:
                with open(files[-1]) as f:
                    for line in f:
                        detection_results.append(json.loads(line))

    fuzzing_results: list[dict[str, Any]] = []
    for scoring_dir in [autointerp_dir / "scoring" / "fuzzing"]:
        if scoring_dir.exists():
            files = sorted(scoring_dir.glob("results_*.jsonl"))
            if files:
                with open(files[-1]) as f:
                    for line in f:
                        fuzzing_results.append(json.loads(line))

    # Pick example component
    example_component = _pick_component_with_label(components, labels, rng)
    example_label = labels[example_component.component_key]

    # Build markdown
    md = f"""
# SPD Autointerp Report

**Run:** `{run_id}`
**Components interpreted:** {len(interp_results):,}

---

## Interpretation Confidence Breakdown

| Confidence | Count | Percentage |
|---|---|---|
| High | {len(high):,} | {len(high) / len(interp_results) * 100:.0f}% |
| Medium | {len(med):,} | {len(med) / len(interp_results) * 100:.0f}% |
| Low | {len(low):,} | {len(low) / len(interp_results) * 100:.0f}% |

---

## High-Confidence Examples

"""
    for r in rng.sample(high, min(8, len(high))):
        md += f"**`{r['component_key']}`** -- {r['label']}\n\n"
        md += f"> {r['reasoning']}\n\n"

    md += "## Medium-Confidence Examples\n\n"
    for r in rng.sample(med, min(4, len(med))):
        md += f"**`{r['component_key']}`** -- {r['label']}\n\n"
        md += f"> {r['reasoning']}\n\n"

    md += "## Low-Confidence Examples\n\n"
    for r in rng.sample(low, min(3, len(low))):
        md += f"**`{r['component_key']}`** -- {r['label']}\n\n"
        md += f"> {r['reasoning']}\n\n"

    # === Prompt Examples Section ===
    # Ordered by pipeline dependency: intruder (label-free) → interpretation → scoring (label-dependent)
    md += """---

## Prompt Examples

One example of every LLM prompt template used in the pipeline, rendered with real data.
For sense-checking tokenization and formatting.

"""

    # 1. Intruder prompt (label-free, runs first)
    md += "### 1. Intruder Eval Prompt (label-free)\n\n"
    md += f"**Component:** `{example_component.component_key}`\n\n"
    intruder_prompt = _build_example_intruder_prompt(
        example_component,
        components,
        lookup,
        rng,
        ci_threshold,
    )
    md += f"```\n{intruder_prompt}\n```\n\n"

    # 2. Interpretation prompt
    md += "### 2. Interpretation Prompt\n\n"
    md += f"**Component:** `{example_component.component_key}` (label: *{example_label}*)\n\n"

    if token_stats is not None:
        interp_config = CompactSkepticalConfig(
            model="google/gemini-3-flash-preview",
            reasoning_effort=None,
        )
        input_stats = get_input_token_stats(
            token_stats,
            example_component.component_key,
            tokenizer,
            top_k=20,
        )
        output_stats = get_output_token_stats(
            token_stats,
            example_component.component_key,
            tokenizer,
            top_k=50,
        )
        if input_stats is not None and output_stats is not None:
            interp_prompt = format_prompt(
                interp_config,
                example_component,
                arch,
                tokenizer,
                input_stats,
                output_stats,
                ci_threshold,
            )
            md += f"```\n{interp_prompt}\n```\n\n"
        else:
            md += "*Token stats not available for this component.*\n\n"
    else:
        # Fall back to stored prompt from results
        prompt_example = rng.choice(high) if high else rng.choice(interp_results)
        md += f"*(Rendered from stored result for `{prompt_example['component_key']}`)*\n\n"
        md += f"```\n{prompt_example['prompt'][:5000]}\n```\n\n"

    # 3. Detection prompt (label-dependent)
    md += "### 3. Detection Scoring Prompt (label-dependent)\n\n"
    md += f"**Component:** `{example_component.component_key}` (label: *{example_label}*)\n\n"
    detection_prompt = _build_example_detection_prompt(
        example_component,
        components,
        lookup,
        example_label,
        rng,
    )
    md += f"```\n{detection_prompt}\n```\n\n"

    # 4. Fuzzing prompt (label-dependent)
    md += "### 4. Fuzzing Scoring Prompt (label-dependent)\n\n"
    md += f"**Component:** `{example_component.component_key}` (label: *{example_label}*)\n\n"
    fuzzing_prompt = _build_example_fuzzing_prompt(
        example_component,
        lookup,
        example_label,
        rng,
        ci_threshold,
    )
    md += f"```\n{fuzzing_prompt}\n```\n\n"

    # === Scoring Results ===
    if intruder_results:
        scores = [float(r["score"]) for r in intruder_results if int(r["n_trials"]) > 0]
        md += f"""---

## Intruder Detection Eval

Tests component coherence: 4 real examples + 1 intruder from a different component.
Random baseline = 20%.

**{len(scores)} components scored**

| Metric | Value |
|---|---|
| Mean accuracy | {np.mean(scores) * 100:.1f}% |
| Median accuracy | {np.median(scores) * 100:.1f}% |
| Random baseline | 20.0% |

### Score Distribution

| Coherence Level | Count |
|---|---|
"""
        bins = [
            (0, 0.2, "Incoherent (<20%)"),
            (0.2, 0.4, "Weak (20-40%)"),
            (0.4, 0.6, "Moderate (40-60%)"),
            (0.6, 0.8, "Good (60-80%)"),
            (0.8, 1.01, "Excellent (80-100%)"),
        ]
        for lo, hi, label in bins:
            count = sum(1 for s in scores if lo <= s < hi)
            md += f"| {label} | {count} |\n"

    if detection_results:
        scores = [float(r["score"]) for r in detection_results]
        md += f"""
---

## Detection Scoring

Tests whether labels predict activations. Random baseline = 50%.

**{len(scores)} components scored**

| Metric | Value |
|---|---|
| Mean balanced accuracy | {np.mean(scores) * 100:.1f}% |
| Median balanced accuracy | {np.median(scores) * 100:.1f}% |

"""

    if fuzzing_results:
        scores = [float(r["score"]) for r in fuzzing_results]
        md += f"""
---

## Fuzzing Scoring

Tests label specificity via correct vs. random highlighting. Random baseline = 50%.

**{len(scores)} components scored**

| Metric | Value |
|---|---|
| Mean balanced accuracy | {np.mean(scores) * 100:.1f}% |
| Median balanced accuracy | {np.median(scores) * 100:.1f}% |

"""

    md += "\n\n---\n\n*Generated by `spd.autointerp.scripts.generate_report`*\n"

    html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html_full = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autointerp Report — {run_id}</title>
<style>
  :root {{ --bg: #fff; --fg: #1a1a1a; --muted: #555; --border: #ddd;
           --accent: #3498db; --code-bg: #f4f4f4; --pre-bg: #f8f8f8;
           --table-stripe: #fafafa; --heading: #2c3e50; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
         font-size: 15px; line-height: 1.6; color: var(--fg); background: var(--bg);
         max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }}
  h1 {{ font-size: 1.8rem; border-bottom: 2px solid var(--fg); padding-bottom: 0.5rem; }}
  h2 {{ font-size: 1.4rem; color: var(--heading); border-bottom: 1px solid var(--border);
        padding-bottom: 0.3rem; margin-top: 2rem; }}
  h3 {{ font-size: 1.1rem; color: var(--heading); margin-top: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: var(--code-bg); font-weight: 600; }}
  tr:nth-child(even) {{ background: var(--table-stripe); }}
  code {{ background: var(--code-bg); padding: 0.15rem 0.4rem; border-radius: 3px;
         font-size: 0.85em; }}
  pre {{ background: var(--pre-bg); padding: 1rem; border-radius: 6px; overflow-x: auto;
         font-size: 0.8rem; line-height: 1.5; border: 1px solid var(--border); }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 3px solid var(--accent); margin: 0.75rem 0; padding: 0.5rem 1rem;
               color: var(--muted); font-size: 0.9rem; background: var(--pre-bg);
               border-radius: 0 4px 4px 0; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}
  a {{ color: var(--accent); }}
</style>
</head><body>
{html_body}
</body></html>"""

    if output_path is None:
        output_path = Path(f"autointerp_report_{run_id}.html")
    output_path.write_text(html_full)
    print(f"Report written to {output_path}")
    return output_path


if __name__ == "__main__":
    import fire

    fire.Fire(generate_report)
