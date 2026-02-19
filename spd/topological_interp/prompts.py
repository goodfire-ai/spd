"""Prompt formatters for topological interpretation.

Three prompts:
1. Output pass (late→early): "What does this component DO?"
2. Input pass (early→late): "What TRIGGERS this component?"
3. Unification: Synthesize output + input labels into unified label.

Output and input passes are independent — neither depends on the other's labels.
The unification step combines them.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_fires_on_examples,
    build_input_section,
    build_output_section,
    build_says_examples,
    density_note,
    human_layer_desc,
    layer_position_note,
)
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.topological_interp.neighbors import NeighborContext
from spd.topological_interp.schemas import LabelResult

LABEL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}

_FORBIDDEN = "FORBIDDEN vague words: narrative, story, character, theme, descriptive, content, transition, scene."


def _build_context_block(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    max_examples: int,
) -> str:
    """Shared context block used by both output and input prompts."""
    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = layer_position_note(canonical, model_metadata.n_blocks)
    dens_note = density_note(component.firing_density)

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    dataset_desc = DATASET_DESCRIPTIONS.get(
        model_metadata.dataset_name, model_metadata.dataset_name
    )

    input_pmi = (
        [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.input_token_pmi.top]
        if component.input_token_pmi.top
        else None
    )
    output_pmi = (
        [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.output_token_pmi.top]
        if component.output_token_pmi.top
        else None
    )

    output_section = build_output_section(output_token_stats, output_pmi)
    input_section = build_input_section(input_token_stats, input_pmi)
    fires_on = build_fires_on_examples(component, app_tok, max_examples)
    says = build_says_examples(component, app_tok, max_examples)

    context_notes = " ".join(filter(None, [position_note, dens_note]))

    return f"""\
## Context
- Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks), dataset: {dataset_desc}
- Component: {layer_desc} (component {component.component_idx})
- Firing rate: {component.firing_density * 100:.2f}% ({rate_str})
{context_notes}

## Output tokens (what the model produces when this component fires)
{output_section}
## Input tokens (what causes this component to fire)
{input_section}
## Activation examples — where the component fires
{fires_on}
## Activation examples — what the model produces
{says}"""


def format_output_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    downstream_neighbors: list[NeighborContext],
    label_max_words: int,
    max_examples: int,
) -> str:
    context = _build_context_block(
        component, model_metadata, app_tok, input_token_stats, output_token_stats, max_examples
    )
    downstream_table = _format_neighbor_table(downstream_neighbors)

    return f"""\
You are analyzing a component in a neural network to understand its OUTPUT FUNCTION — what it does when it fires.

{context}
## Downstream components (what this component influences)
These components in later layers are most influenced by this component (by gradient attribution):
{downstream_table}
## Task
Give a {label_max_words}-word-or-fewer label describing this component's OUTPUT FUNCTION — what it does when it fires.

Examples of good labels:
- "word stem completion (stems → suffixes)"
- "closes dialogue with quotation marks"
- "object pronouns after verbs"
- "aquatic scene vocabulary (frog, river, pond)"

{_FORBIDDEN} Lowercase only. Say "unclear" if the evidence is too weak.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def format_input_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    upstream_neighbors: list[NeighborContext],
    cofiring_neighbors: list[NeighborContext],
    label_max_words: int,
    max_examples: int,
) -> str:
    context = _build_context_block(
        component, model_metadata, app_tok, input_token_stats, output_token_stats, max_examples
    )
    upstream_table = _format_neighbor_table(upstream_neighbors)
    cofiring_table = _format_cofiring_table(cofiring_neighbors)

    return f"""\
You are analyzing a component in a neural network to understand its INPUT FUNCTION — what triggers it to fire.

{context}
## Upstream components (what feeds into this component)
These components in earlier layers most strongly attribute to this component:
{upstream_table}
## Co-firing components
Components that frequently fire together with this one:
{cofiring_table}
## Task
Give a {label_max_words}-word-or-fewer label describing this component's INPUT FUNCTION — what conditions trigger it to fire.

Examples of good labels:
- "periods and sentence boundaries"
- "prepositions before noun phrases"
- "tokens following proper nouns"
- "positions requiring verb conjugation"

{_FORBIDDEN} Lowercase only. Say "unclear" if the evidence is too weak.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def format_unification_prompt(
    output_label: LabelResult,
    input_label: LabelResult,
    label_max_words: int,
) -> str:
    return f"""\
A neural network component has been analyzed from two perspectives:

OUTPUT FUNCTION: "{output_label.label}" (confidence: {output_label.confidence})
  Reasoning: {output_label.reasoning}

INPUT FUNCTION: "{input_label.label}" (confidence: {input_label.confidence})
  Reasoning: {input_label.reasoning}

Synthesize these into a single unified label (max {label_max_words} words) that captures the component's complete role. If input and output suggest the same concept, unify them. If they describe genuinely different aspects (e.g. fires on X, produces Y), combine both. Lowercase only.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def _format_neighbor_table(neighbors: list[NeighborContext]) -> str:
    if not neighbors:
        return "(no attributed neighbors found)\n"

    lines: list[str] = []
    for n in neighbors:
        parts = [f"  {n.component_key} (attribution: {n.attribution:.4f}"]
        if n.jaccard is not None:
            parts.append(f", co-firing Jaccard: {n.jaccard:.3f}")
        parts.append(")")

        line = "".join(parts)
        if n.label is not None:
            line += f'\n    label: "{n.label}" (confidence: {n.confidence})'
        lines.append(line)

    return "\n".join(lines) + "\n"


def _format_cofiring_table(neighbors: list[NeighborContext]) -> str:
    if not neighbors:
        return "(no co-firing components found)\n"

    lines: list[str] = []
    for n in neighbors:
        parts = [f"  {n.component_key}"]
        if n.jaccard is not None:
            parts.append(f" (Jaccard: {n.jaccard:.3f}")
            if n.pmi is not None:
                parts.append(f", PMI: {n.pmi:.2f}")
            parts.append(")")
        line = "".join(parts)
        if n.label is not None:
            line += f'\n    label: "{n.label}" (confidence: {n.confidence})'
        lines.append(line)

    return "\n".join(lines) + "\n"
