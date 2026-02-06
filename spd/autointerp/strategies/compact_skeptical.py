"""Compact skeptical interpretation strategy.

Short labels (2-5 words), skeptical tone, structured JSON output.
Extracted from the original prompt_template.py.
"""

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup, delimit_tokens
from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.schemas import ArchitectureInfo
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

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
            "description": "low = speculative/unclear; medium = plausible but noisy; high = clear specific pattern with strong evidence",
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentences explaining the evidence",
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

SPD_CONTEXT = (
    "Each component has a causal importance (CI) value per token position. "
    "High CI (near 1) = essential, cannot be ablated. Low CI (near 0) = ablatable."
)


def _parse_layer_description(layer: str, n_blocks: int) -> str:
    parts = layer.split(".")
    assert parts[0] == "h", f"unexpected layer format: {layer}"
    layer_idx = int(parts[1])
    layer_num = layer_idx + 1

    sublayer = ".".join(parts[2:])
    sublayer_desc = {
        "mlp.c_fc": "MLP up-projection",
        "mlp.c_proj": "MLP down-projection",
        "mlp.down_proj": "MLP down-projection",
        "attn.q_proj": "attention Q projection",
        "attn.k_proj": "attention K projection",
        "attn.v_proj": "attention V projection",
        "attn.o_proj": "attention output projection",
    }.get(sublayer, sublayer)

    return f"{sublayer_desc} in layer {layer_num} of {n_blocks}"


def format_prompt(
    config: CompactSkepticalConfig,
    component: ComponentData,
    arch: ArchitectureInfo,
    tokenizer: PreTrainedTokenizerBase,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> str:
    lookup = build_token_lookup(tokenizer, tokenizer.name_or_path)

    input_pmi: list[tuple[str, float]] | None = None
    output_pmi: list[tuple[str, float]] | None = None

    if config.include_pmi:
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

    input_section = _build_input_section(input_token_stats, input_pmi)
    output_section = _build_output_section(
        output_token_stats, output_pmi, config.output_precision_top_k
    )
    examples_section = _build_examples_section(
        component,
        lookup,
        config.ci_display_threshold,
        config.max_examples,
    )

    if component.mean_ci > 0:
        rate_str = f"~1 in {int(1 / component.mean_ci)} tokens"
    else:
        rate_str = "extremely rare"

    layer_desc = _parse_layer_description(component.layer, arch.n_blocks)

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS.get(arch.dataset_name, arch.dataset_name)
        dataset_line = f", dataset: {dataset_desc}"

    spd_context_block = f"\n{SPD_CONTEXT}\n" if config.include_spd_context else ""

    forbidden = ", ".join(config.forbidden_words) if config.forbidden_words else "(none)"

    return f"""\
Label this neural network component.
{spd_context_block}
## Context
- Model: {arch.model_class} ({arch.n_blocks} layers){dataset_line}
- Location: {layer_desc}
- Activation rate: {component.mean_ci * 100:.2f}% ({rate_str})

## Token correlations

{input_section}
{output_section}

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


def _build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if input_stats.top_recall:
        section += "**Input recall — when active, which tokens appear?**\n"
        for tok, recall in input_stats.top_recall[:8]:
            section += f"  {repr(tok)}: {recall * 100:.0f}%\n"

    if input_stats.top_precision:
        section += "\n**Input precision — which tokens predict this component?**\n"
        for tok, prec in input_stats.top_precision[:8]:
            section += f"  {repr(tok)}: {prec * 100:.0f}%\n"

    if input_pmi:
        section += "\n**Input PMI — surprising associations:**\n"
        for tok, pmi in input_pmi[:6]:
            section += f"  {repr(tok)}: {pmi:.2f}\n"

    return section


def _build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
    top_k: int,
) -> str:
    section = ""

    if output_stats.top_precision:
        high_prec = [(t, p) for t, p in output_stats.top_precision[:top_k] if p > 0.5]
        if high_prec:
            tokens = [repr(t) for t, _ in high_prec[:15]]
            section += (
                f"**Output precision — tokens predicted when active (>50%):** {', '.join(tokens)}\n"
            )
        else:
            top_few = output_stats.top_precision[:10]
            tokens = [f"{repr(t)} ({p * 100:.0f}%)" for t, p in top_few]
            section += f"**Output precision — top predicted tokens:** {', '.join(tokens)}\n"

    if output_pmi:
        section += "\n**Output PMI — surprising predictions:**\n"
        for tok, pmi in output_pmi[:6]:
            section += f"  {repr(tok)}: {pmi:.2f}\n"

    return section


def _build_examples_section(
    component: ComponentData,
    lookup: dict[int, str],
    ci_threshold: float,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        tokens = [
            (lookup[tid], ci > ci_threshold)
            for tid, ci in zip(ex.token_ids, ex.ci_values, strict=True)
            if tid >= 0
        ]
        has_active = any(active for _, active in tokens)
        if has_active:
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section
