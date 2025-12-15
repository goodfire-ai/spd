"""Jinja2 template for component auto-interpretation prompts."""

import jinja2

from spd.autointerp.schemas import ArchitectureInfo, ComponentData


def format_prompt(component: ComponentData, arch: ArchitectureInfo) -> str:
    return TEMPLATE.render(
        c=component,
        arch=arch,
        layer_info=_parse_layer(component.layer, arch.n_layers),
        interpretation_schema=INTERPRETATION_SCHEMA,
    )


def _parse_layer(layer: str, n_layers: int) -> str:
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

    return f"{sublayer_desc} in layer {layer_num} of {n_layers}"


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
    "additionalProperties": False,
}


TEMPLATE = jinja2.Template(
    """\
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

**Target model (the model we're decomposing)**: {{ arch.model_class }} ({{ arch.n_layers }} layers, {{ arch.c }} components per layer)

**Training data**: {{ arch.dataset_name }} — {{ arch.dataset_description }}

This component is from {{ layer_info }}.

Mean causal importance (how densely this component is active in the training data): {{ "%.4f"|format(c.mean_ci) }}

---

## Activation Examples

These are contexts where this component fires strongly. The activating token is marked with `>>token<<`.

{% for ex in c.activation_examples %}
**Example {{ loop.index }}** (CI={{ "%.3f"|format(ex.active_ci) }}):
{% for i in range(ex.tokens|length) %}{% if i == ex.active_pos %}`>>{{ ex.tokens[i] }}<<`{% else %}`{{ ex.tokens[i] }}`{% endif %}{% if not loop.last %} {% endif %}{% endfor %}
{% endfor %}

---

## Token Statistics

From {{ arch.dataset_name }}:

**Input tokens** (what precedes/triggers this component):
- Precision: {% for tok, s in c.input_token_stats.top_precision %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- Recall: {% for tok, s in c.input_token_stats.top_recall %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- PMI: {% for tok, s in c.input_token_stats.top_pmi %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}

**Output tokens** (what this component helps predict):
- Precision: {% for tok, s in c.output_token_stats.top_precision %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- Recall: {% for tok, s in c.output_token_stats.top_recall %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- PMI: {% for tok, s in c.output_token_stats.top_pmi %}"{{ tok }}"({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}

---

## Co-occurring Components

- High precision: {% for key, s in c.correlations.precision %}{{ key }}({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- High recall: {% for key, s in c.correlations.recall %}{{ key }}({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- High PMI: {% for key, s in c.correlations.pmi %}{{ key }}({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}
- Low PMI: {% for key, s in c.correlations.bottom_pmi %}{{ key }}({{ "%.2f"|format(s) }}){% if not loop.last %} {% endif %}{% endfor %}

---

## What I'm Looking For

- **Label** (3-10 words): What does this component detect or represent? Be specific.
- **Confidence** (low/medium/high): How clear-cut is this?
- **Reasoning** (2-4 sentences): What evidence supports this? What's ambiguous?

Keep in mind:
- Earlier layers often capture local/syntactic patterns; later layers capture semantics
- MLP layers represent features; attention layers move information
- The training data is narrow — interpret components in that context

---

## Response Format

Your response should be in JSON format, matching this schema:
```json
{{ interpretation_schema_json }}
```

Please directly output the JSON object, without any other text or comments. Thank you!
"""
)
