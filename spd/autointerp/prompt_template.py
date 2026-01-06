"""Prompt templates for component auto-interpretation."""

import json

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup
from spd.autointerp.schemas import ArchitectureInfo
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def _parse_layer_description(layer: str, n_blocks: int) -> str:
    """Parse layer name into human-readable description.

    e.g. "h.2.mlp.c_fc" -> "MLP up-projection in layer 3 of 4"
    """
    parts = layer.split(".")
    assert parts[0] == "h", f"unexpected layer format: {layer}"
    layer_idx = int(parts[1])
    layer_num = layer_idx + 1

    sublayer = ".".join(parts[2:])
    sublayer_desc = {
        "mlp.c_fc": "MLP up-projection",
        "mlp.c_proj": "MLP down-projection",
        "attn.q_proj": "attention Q projection",
        "attn.k_proj": "attention K projection",
        "attn.v_proj": "attention V projection",
        "attn.o_proj": "attention output projection",
    }.get(sublayer, sublayer)

    return f"{sublayer_desc} in layer {layer_num} of {n_blocks}"


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "3-10 word label describing what the component detects/represents",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How clear-cut the interpretation is",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the evidence and ambiguities",
        },
    },
    "required": ["label", "confidence", "reasoning"],
}

INTERPRETATION_SCHEMA_JSON_STR = json.dumps(INTERPRETATION_SCHEMA, indent=2)


DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": """\
SimpleStories is a dataset of 2M+ short stories (200-350 words each) at a grade 1-8 reading level.
The stories cover diverse themes (friendship, courage, loss, discovery) and settings (magical lands,
schools, forests, space). The vocabulary is simple, everyday English. Stories feature common narrative
elements: characters with names, emotions, dialogue, and simple plot arcs with resolutions.""",
}

SPD_THEORETICAL_CONTEXT = """\
SPD (Stochastic Parameter Decomposition) decomposes a neural network's weight matrices into rank-1
"subcomponents". Each subcomponent has a causal importance (CI) value predicted *per sequence position*
by a small auxiliary neural network. CI indicates how necessary the component is for the model's output
at that position: high CI (close to 1) means the component is essential and cannot be ablated; low CI
(close to 0) means it can be removed without affecting output. The training objective encourages
sparsity: as few components as possible should have high CI for any given input."""


def format_prompt_template(
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
    max_examples: int,
) -> str:
    lookup = build_token_lookup(tokenizer, tokenizer.name_or_path)

    PADDING_SENTINEL = -1

    examples_str = ""
    examples = component.activation_examples[:max_examples]
    for example_idx, example in enumerate(examples):
        # Filter out padding sentinel (-1) for decoding
        valid_token_ids: list[int] = []
        for tid in example.token_ids:
            if tid == PADDING_SENTINEL:
                continue
            assert tid >= 0, (
                f"Unexpected token_id {tid} (expected valid token or {PADDING_SENTINEL})"
            )
            valid_token_ids.append(tid)

        full_text = tokenizer.decode(valid_token_ids) if valid_token_ids else ""
        full_text_escaped_str = full_text.replace('"', '\\"')

        def token_str(token_id: int) -> str:
            if token_id == PADDING_SENTINEL:
                return "<pad>"
            return lookup[token_id]

        token_activation_pairs_str = ", ".join(
            [
                f'("{token_str(token_id)}", {ci:.2f})'
                for token_id, ci in zip(example.token_ids, example.ci_values, strict=True)
            ]
        )

        this_example_str = f'''\
**Example {example_idx + 1}**

Full text: "{full_text_escaped_str}"

Token activation pairs: {token_activation_pairs_str}'''

        examples_str += this_example_str

    top_input_tokens_by_pmi_str = "\n".join(
        [f'- "{lookup[token_id]}" ({pmi:.2f})' for token_id, pmi in component.input_token_pmi.top]
    )
    top_output_tokens_by_pmi_str = "\n".join(
        [f'- "{lookup[token_id]}" ({pmi:.2f})' for token_id, pmi in component.output_token_pmi.top]
    )
    bottom_output_tokens_by_pmi_str = "\n".join(
        [
            f'- "{lookup[token_id]}" ({pmi:.2f})'
            for token_id, pmi in component.output_token_pmi.bottom
        ]
    )

    dataset_description = DATASET_DESCRIPTIONS[arch.dataset_name]

    return f"""\
Hi Claude,

I'm working on interpretability research and could use your help labeling a component from a neural
network. We've decomposed a language model into sparse components using SPD (Stochastic Parameter
Decomposition), and I'd like to understand what this particular component does. To give some
background, in spd, we learn decompositions of a model's parameter weight matrices, in terms of
rank-1 components. We train the decomposition such that very few of these components need to be
present in order to recreate the behaviour of the orginal model on any given prompt. This means that
locally, the model becomes extremely low rank and more inherently interpretable. These components 
are then treated as the basic atoms of computation.


## Context

**Target model (the model we're decomposing)**: {arch.model_class} ({arch.n_blocks} layers),

**Training data**: {arch.dataset_name} — {dataset_description}

This component is from the {_parse_layer_description(component.layer, arch.n_blocks)}.

Mean causal importance (how densely this component is active in the training data): {component.mean_ci * 100:.4f}%

---

## Component-Token Correlations:

**Top Input Tokens by PMI** - Tokens on which this component has a higher than expected probability of firing:
{top_input_tokens_by_pmi_str}

**Top Output Tokens by PMI** - Tokens which have a higher than expected probability of being predicted when this component fires:
{top_output_tokens_by_pmi_str}

**Bottom Output Tokens by PMI** - Tokens which have a lower than expected probability of being predicted when this component fires:
{bottom_output_tokens_by_pmi_str}

*Note: Bottom input tokens by PMI are not really meaningful because many tokens never co-occur with many components (-inf PMI)*

--- 

## Activation Examples

These are contexts where this component fires strongly. The texts are shown twice. One tokenized as
normal, and once as (token, ci) pairs.

{examples_str}

---

Keep in mind:
- Earlier layers often capture local/syntactic patterns; later layers capture semantics

---

## Response Format

Your response should be in JSON format, matching this schema:
```json
{INTERPRETATION_SCHEMA_JSON_STR}
```

Please directly output the JSON object, without any other text or comments. Thank you!
"""


def format_prompt_template_v2(
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
    input_token_stats: TokenPRLift | None,
    output_token_stats: TokenPRLift | None,
    max_examples: int,
    ci_display_threshold: float = 0.3,
    output_precision_top_k: int = 40,
) -> str:
    """Improved prompt template using recall/precision/PMI.

    Key improvements over v1:
    - Uses recall AND precision for input tokens
    - Uses precision AND PMI for output tokens
    - Only shows high-CI tokens in examples (reduces noise)
    - Includes inline metric definitions
    - Better dataset descriptions to avoid vacuous interpretations
    - Filters noisy PMI entries (short subword tokens)
    """
    lookup = build_token_lookup(tokenizer, tokenizer.name_or_path)
    PADDING_SENTINEL = -1

    # Convert PMI from ComponentData to decoded tokens
    input_pmi = (
        [(lookup[tid], pmi) for tid, pmi in component.input_token_pmi.top]
        if component.input_token_pmi.top
        else None
    )
    output_pmi = (
        [(lookup[tid], pmi) for tid, pmi in component.output_token_pmi.top]
        if component.output_token_pmi.top
        else None
    )

    # Build input token section using recall, precision, and PMI
    input_section = _build_input_token_section(input_token_stats, input_pmi)

    # Build output token section using precision and PMI
    output_section = _build_output_token_section(
        output_token_stats, output_pmi, output_precision_top_k
    )

    # Build examples showing only high-CI tokens
    examples_section = _build_examples_section(
        component, tokenizer, lookup, max_examples, ci_display_threshold, PADDING_SENTINEL
    )

    # Get dataset description
    dataset_description = DATASET_DESCRIPTIONS.get(
        arch.dataset_name, f"Dataset: {arch.dataset_name}"
    )

    # Calculate firing rate context
    firing_rate_context = ""
    if component.mean_ci > 0:
        tokens_per_firing = int(1 / component.mean_ci)
        firing_rate_context = f" (fires on ~1 in {tokens_per_firing} tokens)"

    layer_desc = _parse_layer_description(component.layer, arch.n_blocks)

    return f"""\
Label this neural network component from a sparse decomposition.

## Background

{SPD_THEORETICAL_CONTEXT}

## Context

**Model**: {arch.model_class} ({arch.n_blocks} layers)
**Dataset**: {dataset_description}
**Component location**: {layer_desc}
**Firing density**: {component.mean_ci * 100:.2f}%{firing_rate_context}

---

{input_section}

---

{output_section}

---

{examples_section}

---

## Task

Based on the output tokens this component helps predict, what concept or pattern does it represent?
Focus on what the component *does* (what tokens it helps predict) rather than just what triggers it.

Return JSON:
```json
{INTERPRETATION_SCHEMA_JSON_STR}
```
"""


def _build_input_token_section(
    input_stats: TokenPRLift | None,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    """Build input token analysis section using recall, precision, and PMI."""
    section = """\
## Input Token Analysis

"""
    if not input_stats:
        section += "  (No input token data available)\n"
        return section

    # Recall section
    if input_stats.top_recall:
        section += '_**Recall** = "What % of this component\'s firings occurred on token X?"_\n'
        for token, recall in input_stats.top_recall[:10]:
            pct = recall * 100
            if pct >= 1:
                section += f"  {repr(token)}: {pct:.0f}%\n"
            elif pct >= 0.1:
                section += f"  {repr(token)}: {pct:.1f}%\n"
        section += "\n"

    # Precision section - very important for detecting deterministic triggers
    if input_stats.top_precision:
        section += '_**Precision** = "When token X appears, what % of the time does this component fire?"_\n'
        # Filter to tokens with meaningful precision (>20%) to avoid noise
        high_prec = [(t, p) for t, p in input_stats.top_precision[:15] if p > 0.20]
        if high_prec:
            for token, prec in high_prec[:10]:
                section += f"  {repr(token)}: {prec * 100:.0f}%\n"
        else:
            section += "  (No tokens with >20% precision)\n"
        section += "\n"

    # PMI section - shows surprising associations
    if input_pmi:
        section += (
            "_**PMI** = Tokens with higher-than-expected co-occurrence (surprisal measure)_\n"
        )
        # Filter out very short tokens (likely subword noise) and show top PMI
        filtered_pmi = [
            (t, p)
            for t, p in input_pmi
            if len(t.strip()) > 1 or t.strip() in {".", ",", '"', "'", "!", "?"}
        ]
        for token, pmi in filtered_pmi[:8]:
            section += f"  {repr(token)}: {pmi:.2f}\n"
        section += "\n"

    # Add interpretive note
    top_recall = input_stats.top_recall[0][1] if input_stats.top_recall else 0
    if top_recall > 0.5:
        section += "_This component fires predominantly on a specific token/pattern._"
    elif top_recall < 0.15:
        section += "_No single input token dominates - fires across many tokens._"

    return section


def _build_output_token_section(
    output_stats: TokenPRLift | None,
    output_pmi: list[tuple[str, float]] | None,
    top_k: int,
) -> str:
    """Build output token analysis section using precision and PMI."""
    section = """\
## Output Token Analysis

_These are tokens the model predicts when this component is active._

"""
    if not output_stats and not output_pmi:
        section += "  (No output token data available)\n"
        return section

    # Precision section
    if output_stats and output_stats.top_precision:
        section += '_**Precision** = "When the model predicts token X, what % of the time is this component active?"_\n\n'
        # Group by precision ranges
        very_high = [(t, p) for t, p in output_stats.top_precision[:top_k] if p > 0.90]
        high = [(t, p) for t, p in output_stats.top_precision[:top_k] if 0.70 <= p <= 0.90]
        medium = [(t, p) for t, p in output_stats.top_precision[:top_k] if 0.50 <= p < 0.70]

        if very_high:
            tokens = [repr(t) for t, _ in very_high[:30]]
            section += f"**Very high (>90%)**: {', '.join(tokens)}\n\n"

        if high:
            tokens = [repr(t) for t, _ in high[:20]]
            section += f"**High (70-90%)**: {', '.join(tokens)}\n\n"

        if medium:
            tokens = [repr(t) for t, _ in medium[:15]]
            section += f"**Medium (50-70%)**: {', '.join(tokens)}\n\n"

        if not very_high and not high and not medium:
            tokens_with_prec = [
                f"{repr(t)} ({p * 100:.0f}%)" for t, p in output_stats.top_precision[:15]
            ]
            section += f"Top by precision: {', '.join(tokens_with_prec)}\n\n"

    # PMI section for output tokens
    if output_pmi:
        section += (
            "_**PMI** = Tokens with higher-than-expected probability when this component fires_\n"
        )
        filtered_pmi = [
            (t, p)
            for t, p in output_pmi
            if len(t.strip()) > 1 or t.strip() in {".", ",", '"', "'", "!", "?"}
        ]
        for token, pmi in filtered_pmi[:10]:
            section += f"  {repr(token)}: {pmi:.2f}\n"

    return section


def _build_examples_section(
    component: ComponentData,
    tokenizer: PreTrainedTokenizerBase,
    lookup: dict[int, str],
    max_examples: int,
    ci_threshold: float,
    padding_sentinel: int,
) -> str:
    """Build examples section showing only high-CI tokens."""
    section = f"""\
## Activation Examples

_Showing tokens where CI > {ci_threshold} (component is active)_

"""
    examples = component.activation_examples[:max_examples]
    for i, example in enumerate(examples):
        # Decode full text
        valid_tokens = [t for t in example.token_ids if t != padding_sentinel and t >= 0]
        full_text = tokenizer.decode(valid_tokens) if valid_tokens else ""
        # Truncate for display
        display_text = full_text[:70] + "..." if len(full_text) > 70 else full_text
        display_text = display_text.replace("\n", " ")

        # Get high-CI tokens
        active_tokens = []
        for tid, ci in zip(example.token_ids, example.ci_values, strict=True):
            if ci > ci_threshold and tid != padding_sentinel and tid >= 0:
                tok = lookup[tid].strip()
                active_tokens.append(f'"{tok}"')

        active_str = ", ".join(active_tokens[:8])
        if len(active_tokens) > 8:
            active_str += f" (+{len(active_tokens) - 8} more)"

        section += f'Ex {i + 1}: "{display_text}"\n'
        section += f"  Active tokens: {active_str}\n\n"

    return section
