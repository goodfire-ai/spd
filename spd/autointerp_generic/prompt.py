"""Generic prompt builder for component interpretation.

No PMI, no token stats, no ArchitectureInfo — just activation examples and
caller-provided context strings.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp_generic.types import (
    ActivatingExample,
    ComponentAutointerpData,
    InterpretConfig,
)

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "2-5 word label for what this component detects",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "low = speculative/unclear; medium = plausible but noisy; "
                "high = clear specific pattern with strong evidence"
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentences explaining the evidence",
        },
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}


def format_example(example: ActivatingExample, app_tok: AppTokenizer) -> str:
    spans = app_tok.get_spans(example.tokens)
    assert len(spans) == len(example.bold)
    tokens = list(zip(spans, example.bold, strict=True))
    return delimit_tokens(tokens)


def format_interpret_prompt(
    config: InterpretConfig,
    component: ComponentAutointerpData,
    decomposition_explanation: str,
    app_tok: AppTokenizer,
) -> str:
    examples_section = ""
    n_shown = 0
    for ex in component.activating_examples[: config.max_examples]:
        formatted = format_example(ex, app_tok)
        if any(ex.bold):
            n_shown += 1
            examples_section += f"{n_shown}. {formatted}\n"

    forbidden = ", ".join(config.forbidden_words) if config.forbidden_words else "(none)"

    return f"""\
Label this neural network component.

{decomposition_explanation}

## Context
- {component.component_explanation}

## Activation examples (active tokens in <<delimiters>>)

{examples_section}

## Task

Give a 2-{config.label_max_words} word label for what this component detects.

Be SKEPTICAL. If you can't identify specific tokens or a tight grammatical pattern, say "unclear".

Rules:
1. Good labels name SPECIFIC tokens: "'the'", "##ing suffix", "she/her pronouns"
2. Say "unclear" if: tokens are too varied, pattern is abstract, or evidence is weak
3. FORBIDDEN words (too vague): {forbidden}
4. Lowercase only
5. Confidence: "high" = clear, specific pattern with strong evidence; "medium" = plausible but noisy; "low" = speculative

GOOD: "##ed suffix", "'and' conjunction", "she/her/hers", "period then capital", "unclear"
BAD: "various words and punctuation", "verbs and adjectives", "tokens near commas"
"""
