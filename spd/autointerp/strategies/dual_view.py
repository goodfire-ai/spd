"""Dual-view interpretation strategy.

Key differences from compact_skeptical:
- Output token data presented first
- Two example sections: "fires on" (current token) and "produces" (next token)
- Human-readable layer descriptions with position context
- Task framing asks for functional description, not detection label
"""

import re

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import DualViewConfig
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "A concise description of what this component does (~3-10 words)",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "low = speculative/unclear; medium = plausible but noisy; high = clear pattern with strong evidence",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the evidence and what the component appears to be doing",
        },
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}


DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": (
        "SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. "
        "Simple vocabulary, common narrative elements."
    ),
}

WEIGHT_NAMES: dict[str, str] = {
    "attn.q": "attention query projection",
    "attn.k": "attention key projection",
    "attn.v": "attention value projection",
    "attn.o": "attention output projection",
    "mlp.up": "MLP up-projection",
    "mlp.down": "MLP down-projection",
    "glu.up": "GLU up-projection",
    "glu.down": "GLU down-projection",
    "glu.gate": "GLU gate projection",
}

_ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def _ordinal(n: int) -> str:
    if 1 <= n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    return f"{n}th"


def _human_layer_desc(canonical: str, n_blocks: int) -> str:
    """Convert canonical layer string to human-readable description.

    '0.mlp.up' -> 'MLP up-projection in the 1st of 4 blocks'
    '1.attn.q' -> 'attention query projection in the 2nd of 4 blocks'
    """
    m = re.match(r"(\d+)\.(.*)", canonical)
    if not m:
        return canonical
    layer_idx = int(m.group(1))
    weight_key = m.group(2)
    weight_name = WEIGHT_NAMES.get(weight_key, weight_key)
    return f"{weight_name} in the {_ordinal(layer_idx + 1)} of {n_blocks} blocks"


def _layer_position_note(canonical: str, n_blocks: int) -> str:
    """Brief note about what layer position means for interpretation."""
    m = re.match(r"(\d+)\.", canonical)
    if not m:
        return ""
    layer_idx = int(m.group(1))
    if layer_idx == n_blocks - 1:
        return "This is in the final block, so its output directly influences token predictions."
    remaining = n_blocks - 1 - layer_idx
    return (
        f"This is {remaining} block{'s' if remaining > 1 else ''} from the output, "
        f"so its effect on token predictions is indirect — filtered through later layers."
    )


def _density_note(firing_density: float) -> str:
    if firing_density > 0.15:
        return (
            "This is a high-density component (fires frequently). "
            "High-density components often act as broad biases rather than selective features."
        )
    if firing_density < 0.005:
        return "This is a very sparse component, likely highly specific."
    return ""


def format_prompt(
    config: DualViewConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> str:
    input_pmi: list[tuple[str, float]] | None = None
    output_pmi: list[tuple[str, float]] | None = None

    if config.include_pmi:
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

    output_section = _build_output_section(output_token_stats, output_pmi)
    input_section = _build_input_section(input_token_stats, input_pmi)
    fires_on_examples = _build_fires_on_examples(component, app_tok, config.max_examples)
    says_examples = _build_says_examples(component, app_tok, config.max_examples)

    if component.firing_density > 0.0:
        rate_str = f"~1 in {int(1 / component.firing_density)} tokens"
    else:
        rate_str = "extremely rare"

    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = _human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = _layer_position_note(canonical, model_metadata.n_blocks)
    density_note = _density_note(component.firing_density)

    context_notes = " ".join(filter(None, [position_note, density_note]))

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS.get(
            model_metadata.dataset_name, model_metadata.dataset_name
        )
        dataset_line = f", dataset: {dataset_desc}"

    forbidden_sentence = (
        "FORBIDDEN vague words: " + ", ".join(config.forbidden_words) + ". "
        if config.forbidden_words
        else ""
    )

    return f"""\
Describe what this neural network component does.

Each component is a learned linear transformation inside a weight matrix. It has an input function (what causes it to fire) and an output function (what tokens it causes the model to produce). These are often different — a component might fire on periods but produce sentence-opening words, or fire on prepositions but produce abstract nouns.

Consider all of the evidence below critically. Token statistics can be noisy, especially for high-density components. The activation examples are sampled and may not be representative. Look for patterns that are consistent across multiple sources of evidence.

## Context
- Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks){dataset_line}
- Component location: {layer_desc}
- Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})

{context_notes}

## Output tokens (what the model produces when this component fires)

{output_section}
## Input tokens (what causes this component to fire)

{input_section}
## Activation examples — where the component fires

<<delimiters>> mark tokens where this component is active.

{fires_on_examples}
## Activation examples — what the model produces

Same examples with <<delimiters>> shifted right by one — showing the token that follows each firing position.

{says_examples}

## Task

Give a {config.label_max_words}-word-or-fewer label describing this component's function. The label should read like a short description of the job this component does in the network. Use both the input and output evidence.

Examples of good labels across different component types:
- "word stem completion (stems → suffixes)"
- "closes dialogue with quotation marks"
- "object pronouns after verbs"
- "story-ending moral resolution vocabulary"
- "aquatic scene vocabulary (frog, river, pond)"
- "'of course' and abstract nouns after prepositions"

Say "unclear" if the evidence is too weak or diffuse. {forbidden_sentence}Lowercase only.
"""


def _build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if output_pmi:
        section += (
            "**Output PMI (pointwise mutual information, in nats: how much more likely "
            "a token is to be produced when this component fires, vs its base rate. "
            "0 = no association, 1 = ~3x more likely, 2 = ~7x, 3 = ~20x):**\n"
        )
        for tok, pmi in output_pmi[:10]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    if output_stats.top_precision:
        section += "\n**Output precision — of all probability mass for token X, what fraction is at positions where this component fires?**\n"
        for tok, prec in output_stats.top_precision[:10]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    return section


def _build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if input_pmi:
        section += "**Input PMI (same metric as above, for input tokens):**\n"
        for tok, pmi in input_pmi[:6]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    if input_stats.top_recall:
        section += "\n**Input recall — most common tokens when the component fires:**\n"
        for tok, recall in input_stats.top_recall[:8]:
            section += f"- {repr(tok)}: {recall * 100:.0f}%\n"

    if input_stats.top_precision:
        section += "\n**Input precision — probability the component fires given the current token is X:**\n"
        for tok, prec in input_stats.top_precision[:8]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    return section


def _build_fires_on_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        if any(ex.firings):
            spans = app_tok.get_spans(ex.token_ids)
            tokens = list(zip(spans, ex.firings, strict=True))
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section


def _build_says_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        if any(ex.firings):
            spans = app_tok.get_spans(ex.token_ids)
            shifted_firings = [False] + ex.firings[:-1]
            tokens = list(zip(spans, shifted_firings, strict=True))
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section
