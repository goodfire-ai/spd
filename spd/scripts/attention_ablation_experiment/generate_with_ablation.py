"""Generate text completions with and without ablation at a single position.

Produces an HTML comparison table with token-level alignment and color-coding.
Each sample picks a random position t, truncates the prompt to t+1 tokens,
then generates greedily. All ablations apply on the first forward pass only
(one predicted token). Subsequent tokens are generated without ablation but
conditioned on the (potentially different) first token.

Usage:
    python -m spd.scripts.attention_ablation_experiment.generate_with_ablation \
        wandb:goodfire/spd/runs/s-275c8f21 \
        --comp_sets '{"2c": "h.1.attn.q_proj:279,h.1.attn.k_proj:177"}' \
        --heads L1H1 --restrict_to_heads L1H1 \
        --n_samples 40 --prompt_len 16 --gen_len 24
"""

import html
import json
import random
from pathlib import Path
from typing import Any

import fire
import torch
from jaxtyping import Int
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.attention_ablation_experiment.attention_ablation_experiment import (
    _build_component_head_ablations,
    _build_deterministic_masks_multi_pos,
    _build_prev_token_component_positions,
    parse_components,
    parse_heads,
    patched_attention_forward,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent

CRAFTED_PROMPTS = [
    ("The cat sat on the mat. The cat sat on the", "Repetition"),
    ("The United States of", "Bigram"),
    ('def hello_world():\n    print("Hello', "Code"),
    ("Once upon a", "Phrase"),
    ("1, 2, 3, 4, 5, 6, 7, 8, 9,", "Counting"),
    ("<div><p>Hello</p></", "HTML"),
    ("Thank you very", "Phrase"),
    ("the the the the the the the the", "Repetition"),
    ("Dear Sir or", "Phrase"),
    ('{"name": "John", "age":', "JSON"),
    ("The quick brown fox jumps over the", "Phrase"),
    ("2 + 2 =", "Math"),
    ("import numpy as", "Code"),
    ("red, green, blue, yellow,", "List"),
    ("What is your", "Question"),
    (
        "The president of the United States gave a speech about the economy"
        " and said that the country needs to invest in",
        "Long",
    ),
    (
        "In a shocking turn of events, the company announced that it would be"
        " laying off thousands of workers due to the recent",
        "Long",
    ),
    (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-",
        "Code",
    ),
    (
        "The recipe calls for 2 cups of flour, 1 cup of sugar, 3 eggs, and a pinch of",
        "Long",
    ),
    (
        "She walked into the room and saw that everyone was staring at her."
        " She felt embarrassed because she had forgotten to",
        "Narrative",
    ),
    ("To install the package, run the following command:\n\npip install", "Docs"),
    (
        "The temperature today is expected to reach a high of 95 degrees"
        " Fahrenheit, which is about 35 degrees",
        "Conversion",
    ),
    ("A B C D E F G H I J K L M N O P Q R S T U V W X Y", "Alphabet"),
    (
        "The patient was diagnosed with a severe case of pneumonia"
        " and was immediately admitted to the",
        "Medical",
    ),
    (
        "<!DOCTYPE html>\n<html>\n<head>\n<title>My Page</title>\n</head>\n"
        "<body>\n<h1>Welcome</h1>\n<p>This is my",
        "HTML",
    ),
    ("for i in range(10):\n    for j in range(10):\n        if i ==", "Code"),
    ("Mon Tue Wed Thu Fri Sat", "Days"),
    (
        "January February March April May June July August September October November",
        "Months",
    ),
    ("The cat chased the mouse. The dog chased the cat. The lion chased the", "Pattern"),
    ("SELECT * FROM users WHERE name =", "SQL"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────


def _build_baseline_mask_infos(
    spd_model: ComponentModel,
    device: torch.device,
) -> dict[str, ComponentsMaskInfo]:
    """All-ones masks so the SPD model uses component reconstruction (not target passthrough)."""
    masks = {name: torch.ones(1, c, device=device) for name, c in spd_model.module_to_c.items()}
    return make_mask_infos(masks)


def _predict_next_token(
    target_model: LlamaSimpleMLP,
    prompt_ids: Int[Tensor, "1 prompt_len"],
    **ablation_kwargs: Any,
) -> int:
    """Run one forward pass with ablation and return the greedy next token."""
    spd_model: ComponentModel | None = ablation_kwargs.pop("spd_model", None)
    mask_infos: dict[str, ComponentsMaskInfo] | None = ablation_kwargs.pop("mask_infos", None)

    with patched_attention_forward(target_model, **ablation_kwargs):
        if spd_model is not None:
            baseline = _build_baseline_mask_infos(spd_model, prompt_ids.device)
            out = spd_model(prompt_ids, mask_infos=mask_infos or baseline)
            assert isinstance(out, Tensor)
            logits = out
        else:
            logits, _ = target_model(prompt_ids)
            assert logits is not None

    return int(logits[0, -1].argmax().item())


# ──────────────────────────────────────────────────────────────────────────────
# Condition definitions
# ──────────────────────────────────────────────────────────────────────────────


def _head_label(heads: list[tuple[int, int]]) -> str:
    return ",".join(f"L{ly}H{hd}" for ly, hd in heads)


def _build_conditions(
    target_model: LlamaSimpleMLP,
    spd_model: ComponentModel,
    prompt_ids: Int[Tensor, "1 seq_len"],
    t: int,
    parsed_heads: list[tuple[int, int]],
    comp_sets: dict[str, list[tuple[str, int]]],
    parsed_restrict_heads: list[tuple[int, int]],
    n_layers: int,
) -> list[tuple[str, int]]:
    """Run all conditions and return (name, predicted_token) pairs in display order."""
    assert t >= 2, f"t must be >= 2 for value ablation conditions, got {t}"
    seq_len = prompt_ids.shape[1]
    conditions: list[tuple[str, int]] = []

    def predict(**kwargs: Any) -> int:
        return _predict_next_token(target_model, prompt_ids, **kwargs)

    # --- Baselines ---
    conditions.append(("Target model", predict()))
    conditions.append(("SPD baseline", predict(spd_model=spd_model)))

    # --- Head ablation: zero head output at t ---
    if parsed_heads:
        head_abl = [(layer, head, t) for layer, head in parsed_heads]
        conditions.append(
            (
                f"Head ablated ({_head_label(parsed_heads)})",
                predict(head_pos_ablations=head_abl),
            )
        )

    # --- Value ablations: zero values at specific positions ---
    # Layer derived from parsed_heads (tests whether the head's layer uses values from t-1).
    if parsed_heads:
        val_layer = parsed_heads[0][0]
        hl = _head_label(parsed_heads)

        conditions.append(
            (
                f"Vals @t-1 (all heads, L{val_layer})",
                predict(value_pos_ablations=[(val_layer, t - 1)]),
            )
        )
        conditions.append(
            (
                f"Vals @t-1 ({hl})",
                predict(value_head_pos_ablations=[(ly, hd, t - 1) for ly, hd in parsed_heads]),
            )
        )
        conditions.append(
            (
                f"Vals @t-1,t-2 (all heads, L{val_layer})",
                predict(value_pos_ablations=[(val_layer, t - 1), (val_layer, t - 2)]),
            )
        )
        conditions.append(
            (
                f"Vals @all prev (all heads, L{val_layer})",
                predict(value_pos_ablations=[(val_layer, p) for p in range(seq_len)]),
            )
        )

    conditions.append(
        (
            "Vals @all prev (ALL layers)",
            predict(
                value_pos_ablations=[(ly, p) for ly in range(n_layers) for p in range(seq_len)]
            ),
        )
    )

    # --- Component ablations ---
    # Full: zero component masks at t (q) / t-1 (k), affects all heads
    # Per-head: subtract component contribution from restrict_heads' rows only
    for set_name, comps in comp_sets.items():
        cp = _build_prev_token_component_positions(comps, t)
        bs = (prompt_ids.shape[0], prompt_ids.shape[1])
        _, ablated_masks = _build_deterministic_masks_multi_pos(
            spd_model, cp, bs, prompt_ids.device
        )
        conditions.append(
            (
                f"Full comp ({set_name})",
                predict(spd_model=spd_model, mask_infos=ablated_masks),
            )
        )
        if parsed_restrict_heads:
            cha = _build_component_head_ablations(spd_model, comps, parsed_restrict_heads, t)
            conditions.append(
                (
                    f"Per-head {_head_label(parsed_restrict_heads)} ({set_name})",
                    predict(spd_model=spd_model, component_head_ablations=cha),
                )
            )

    return conditions


# ──────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────────────────────────

HTML_HEADER = """\
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{font-family:'Menlo','Consolas',monospace;font-size:13px;max-width:1400px;margin:40px auto;background:#fafafa}
h1{font-family:sans-serif}
h2{font-family:sans-serif;border-top:2px solid #333;padding-top:16px;margin-top:40px;font-size:15px}
.sample{margin-bottom:40px}
.prompt{background:#e8e8e8;padding:8px 12px;border-radius:4px;white-space:pre-wrap;word-break:break-all;margin:8px 0}
table{border-collapse:collapse;margin:12px 0}
td,th{padding:3px 5px;text-align:center;border:1px solid #ccc;min-width:28px;font-size:11px}
th{background:#f0f0f0;font-weight:600}
.match{background:#e8f5e9}
.diff{background:#ffcdd2;font-weight:bold}
.tok{white-space:pre}
.label{text-align:left;font-weight:600;padding-right:12px;background:#f5f5f5;min-width:230px;font-size:10px}
.info{font-family:sans-serif;font-size:13px;color:#555;margin:4px 0}
</style></head><body>
"""


def _render_sample_html(
    prompt_tokens: list[str],
    conditions: list[tuple[str, int]],
    t: int,
    label: str,
    decode_tok: Any,
) -> str:
    ref_token_id = conditions[0][1]
    ref_tok = decode_tok([ref_token_id])
    h: list[str] = []
    h.append(f'<div class="sample"><h2>{html.escape(label)} | ablation at t={t}</h2>')
    ablated_tok = html.escape(prompt_tokens[t]) if t < len(prompt_tokens) else "?"
    prev_tok = html.escape(prompt_tokens[t - 1]) if t >= 1 else "?"
    h.append(
        f'<div class="info">Prompt ({len(prompt_tokens)} tok).'
        f' t={t}: "<b>{ablated_tok}</b>", t-1: "<b>{prev_tok}</b>"</div>'
    )
    h.append(f'<div class="prompt">{html.escape("".join(prompt_tokens))}</div>')
    h.append('<div style="overflow-x:auto"><table><tr><th></th><th>predicted token</th></tr>')

    for name, token_id in conditions:
        tok = decode_tok([token_id])
        escaped = html.escape(tok).replace("\n", "\\n").replace(" ", "&nbsp;")
        css = "match" if tok == ref_tok else "diff"
        h.append(
            f'<tr><td class="label">{html.escape(name)}</td>'
            f'<td class="tok {css}">{escaped}</td></tr>'
        )

    h.append("</table></div></div>")
    return "\n".join(h)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def generate_with_ablation(
    wandb_path: ModelPath,
    comp_sets: str | dict[str, str] | None = None,
    heads: str | None = None,
    restrict_to_heads: str | None = None,
    n_samples: int = 40,
    prompt_len: int = 16,
    include_crafted: bool = True,
    seed: int = 42,
) -> None:
    """Generate comparison HTML with multiple ablation conditions.

    Args:
        comp_sets: JSON dict mapping set names to component specs, e.g.
            '{"2c": "h.1.attn.q_proj:279,h.1.attn.k_proj:177"}'
        heads: Head spec for head ablation, e.g. "L1H1"
        restrict_to_heads: Head spec for per-head component ablation
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parsed_comp_sets: dict[str, list[tuple[str, int]]] = {}
    if comp_sets is not None:
        raw: dict[str, str] = json.loads(comp_sets) if isinstance(comp_sets, str) else comp_sets
        for name, spec in raw.items():
            parsed_comp_sets[name] = parse_components(spec)

    parsed_heads = parse_heads(heads) if heads else []
    parsed_restrict_heads = parse_heads(restrict_to_heads) if restrict_to_heads else []

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    spd_model = spd_model.to(device)
    target_model = spd_model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for block in target_model._h:
        block.attn.flash_attention = False
    n_layers = len(target_model._h)

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=1, buffer_size=1000
    )
    encode = tokenizer.encode
    decode_tok = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]

    out_dir = SCRIPT_DIR / "out" / run_id / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables: list[str] = []

    def run_sample(prompt_ids: Tensor, t: int, label: str) -> None:
        prompt_tokens = [decode_tok([tid]) for tid in prompt_ids[0].tolist()]
        conditions = _build_conditions(
            target_model,
            spd_model,
            prompt_ids,
            t,
            parsed_heads,
            parsed_comp_sets,
            parsed_restrict_heads,
            n_layers,
        )
        all_tables.append(_render_sample_html(prompt_tokens, conditions, t, label, decode_tok))

    with torch.no_grad():
        # Dataset samples: take first prompt_len tokens, pick random t, truncate to t+1
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break
            input_ids: Int[Tensor, "1 seq"] = batch_data[task_config.column_name][
                :, :prompt_len
            ].to(device)
            rng = random.Random(i)
            t = rng.randint(2, min(input_ids.shape[1], prompt_len) - 1)
            run_sample(input_ids[:, : t + 1], t, f"Dataset sample {i}")
            if (i + 1) % 10 == 0:
                logger.info(f"Dataset: {i + 1}/{n_samples}")

        # Crafted prompts: use full text, ablate at last token
        if include_crafted:
            for idx, (text, desc) in enumerate(CRAFTED_PROMPTS):
                token_ids = encode(text)
                ids_list: list[int] = (
                    token_ids if isinstance(token_ids, list) else token_ids.ids  # pyright: ignore[reportAttributeAccessIssue]
                )
                ids_tensor = torch.tensor([ids_list], device=device)
                run_sample(ids_tensor, ids_tensor.shape[1] - 1, f"Crafted: {desc}")
                if (idx + 1) % 10 == 0:
                    logger.info(f"Crafted: {idx + 1}/{len(CRAFTED_PROMPTS)}")

    # Write HTML
    comp_desc = ", ".join(
        f"<b>{name}</b> ({len(comps)})" for name, comps in parsed_comp_sets.items()
    )
    html_parts = [
        HTML_HEADER,
        "<h1>Generation Comparison: Ablation Effects</h1>",
        f'<p class="info">Model: {run_id} | {n_layers} layers</p>',
        f'<p class="info">Component sets: {comp_desc}</p>' if comp_desc else "",
        '<p class="info">All ablations apply on the first generated token only. '
        "Subsequent tokens generated normally. Green = matches target. Red = differs.</p>",
        *all_tables,
        "</body></html>",
    ]
    html_path = out_dir / "comparison.html"
    html_path.write_text("\n".join(html_parts))
    logger.info(f"Saved {html_path} ({len(all_tables)} samples)")


if __name__ == "__main__":
    fire.Fire(generate_with_ablation)
