"""Jinja2 template for component auto-interpretation prompts."""

import json
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup
from spd.autointerp.schemas import (
    ArchitectureInfo,
    ComponentData,
    ComponentTokenPMI,
)


@dataclass
class StringifiedTokensAndCI:
    full_text: str
    token_activation_pairs: list[tuple[str, float]]


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


def format_prompt_template(
    component: ComponentData, arch: ArchitectureInfo, tokenizer: PreTrainedTokenizerBase
) -> str:
    lookup = build_token_lookup(tokenizer, tokenizer.name_or_path)

    examples_str = ""
    for example_idx, example in enumerate(component.activation_examples):
        full_text = tokenizer.decode(example.token_ids)
        full_text_escaped_str = full_text.replace('"', '\\"')

        token_activation_pairs_str = ", ".join(
            [
                f'("{lookup[token_id]}", {ci:.2f})'
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

**Training data**: {arch.dataset_name} — {arch.dataset_description}

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


if __name__ == "__main__":
    from spd.autointerp.schemas import ActivationExample

    component = ComponentData(
        component_key="h.2.mlp.c_fc",
        layer="h.2.mlp.c_fc",
        component_idx=0,
        mean_ci=0.2830273,
        activation_examples=[
            ActivationExample(
                token_ids=[100, 102, 103, 104, 105],
                ci_values=[0.1, 0.0, 0.9, 0.0, 0.0],
            )
        ],
        input_token_pmi=ComponentTokenPMI(
            top=[(100, 0.5), (102, 0.3)],
            bottom=[(100, 0.5), (102, 0.3)],
        ),
        output_token_pmi=ComponentTokenPMI(
            top=[(100, 0.5), (102, 0.3)],
            bottom=[(100, 0.5), (102, 0.3)],
        ),
    )
    arch = ArchitectureInfo(
        model_class="GPT-2",
        n_blocks=12,
        dataset_name="OpenWebText",
        dataset_description="OpenWebText is a dataset of web text",
        c=256,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    print(format_prompt_template(component, arch, tokenizer))  # pyright: ignore[reportArgumentType]
