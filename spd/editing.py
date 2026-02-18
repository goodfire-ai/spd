"""Component-level model editing for SPD decompositions.

Utilities for searching components by interpretation labels, building edited
forward functions (ablate/boost/CI-guided), generating text with edits, and
measuring intervention effects.

Usage:
    from spd.editing import search_interpretations, make_edit_fn, generate, measure_kl

    matches = search_interpretations(harvest, interp, r"male pronoun")
    edits = {m.key: 0.0 for m in matches[:3]}
    edit_fn = make_edit_fn(model, edits)

    text = generate(edit_fn, tokens, tokenizer)
    effect = measure_kl(model, edit_fn, token_seqs)
"""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import orjson
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizer

from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

ForwardFn = Callable[[Int[Tensor, "1 seq"]], Float[Tensor, "1 seq vocab"]]


# -- Component key utilities ---------------------------------------------------


def parse_component_key(key: str) -> tuple[str, int]:
    """'h.1.mlp.c_fc:802' -> ('h.1.mlp.c_fc', 802)."""
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


# -- Search --------------------------------------------------------------------


@dataclass
class ComponentMatch:
    key: str
    label: str
    confidence: str
    firing_density: float
    mean_activations: dict[str, float]


def search_interpretations(
    harvest: HarvestRepo,
    interp: InterpRepo,
    pattern: str,
    min_firing_density: float = 0.0,
) -> list[ComponentMatch]:
    """Search component interpretations by regex on label. Sorted by firing density desc."""
    all_interps = interp.get_all_interpretations()
    summary = harvest.get_summary()

    matches = []
    for key, result in all_interps.items():
        if key not in summary:
            continue
        if not re.search(pattern, result.label, re.IGNORECASE):
            continue
        s = summary[key]
        if s.firing_density < min_firing_density:
            continue
        matches.append(
            ComponentMatch(
                key=key,
                label=result.label,
                confidence=result.confidence,
                firing_density=s.firing_density,
                mean_activations=s.mean_activations,
            )
        )

    matches.sort(key=lambda m: -m.firing_density)
    return matches


@dataclass
class TokenPMIMatch:
    key: str
    pmi: float
    firing_density: float


def search_by_token_pmi(
    harvest: HarvestRepo,
    token_ids: list[int],
    side: str,
    min_pmi: float = 0.5,
    min_firing_density: float = 0.01,
    top_k: int = 20,
) -> list[TokenPMIMatch]:
    """Find components by input or output token PMI.

    side="output" finds components that PREDICT the given tokens.
    side="input" finds components that RESPOND TO (fire on) the given tokens.

    For ablation, you almost always want side="output" — ablating output-side
    components suppresses token production with far less collateral damage than
    ablating input-side components.
    """
    assert side in ("input", "output")
    column = "output_token_pmi" if side == "output" else "input_token_pmi"
    target_set = set(token_ids)
    summary = harvest.get_summary()

    db_path = harvest._dir / "harvest.db"
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)

    results = []
    for row in conn.execute(f"SELECT component_key, {column} FROM components"):
        key: str = row[0]
        if key not in summary or summary[key].firing_density < min_firing_density:
            continue
        pmi_data: dict[str, list[list[float]]] = orjson.loads(row[1])
        max_pmi = 0.0
        for tok_id, pmi in pmi_data.get("top", []):
            if int(tok_id) in target_set and pmi > max_pmi:
                max_pmi = pmi
        if max_pmi >= min_pmi:
            results.append(
                TokenPMIMatch(
                    key=key,
                    pmi=max_pmi,
                    firing_density=summary[key].firing_density,
                )
            )

    conn.close()
    results.sort(key=lambda r: -r.pmi)
    return results[:top_k]


def inspect_component(
    harvest: HarvestRepo,
    interp: InterpRepo,
    key: str,
    tokenizer: PreTrainedTokenizer,
    n_examples: int = 5,
    n_pmi_tokens: int = 10,
) -> ComponentData:
    """Print a detailed inspection of a component and return its data."""
    comp = harvest.get_component(key)
    assert comp is not None, f"No harvest data for {key}"
    interp_result = interp.get_interpretation(key)

    ci = comp.mean_activations.get("causal_importance", None)
    ci_str = f", ci={ci:.4f}" if ci is not None else ""
    print(f"{'=' * 70}")
    print(f"{key}  (density={comp.firing_density:.4f}{ci_str})")
    if interp_result:
        print(f"Label: [{interp_result.confidence}] {interp_result.label}")
    print()

    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]

    print("INPUT tokens (what makes it fire):")
    for tok_id, pmi in comp.input_token_pmi.top[:n_pmi_tokens]:
        print(f"  {decode([tok_id]):15s} PMI={pmi:.2f}")

    print("\nOUTPUT tokens (what it predicts):")
    for tok_id, pmi in comp.output_token_pmi.top[:n_pmi_tokens]:
        print(f"  {decode([tok_id]):15s} PMI={pmi:.2f}")

    print(f"\nActivation examples ({n_examples}):")
    for ex in comp.activation_examples[:n_examples]:
        parts = []
        for tid, firing in zip(ex.token_ids, ex.firings, strict=True):
            tok_str = decode([tid])
            parts.append(f">>>{tok_str}<<<" if firing else tok_str)
        act_vals = ex.activations.get("causal_importance", ex.activations.get("activation", []))
        max_act = max(act_vals) if act_vals else 0
        print(f"  [max_act={max_act:.3f}] {''.join(parts)}")
    print()

    return comp


# -- Permanent weight editing --------------------------------------------------


def make_weight_edited_model(
    model: ComponentModel,
    ablate_keys: list[str],
) -> ComponentModel:
    """Create a deep copy of the model with components permanently removed from weights.

    Subtracts rank-1 contributions (V[:, c] @ U[c, :]) from the target model's
    weight matrices. The returned model is a standard transformer — no CI function
    or mask_infos needed at inference.

    The component forward is: out = (x @ V) @ U, so the effective weight in the
    target model's nn.Linear convention (W where out = x @ W.T) is: W = (V @ U).T.
    Removing component c subtracts (V[:, c:c+1] @ U[c:c+1, :]).T from W.
    """
    import copy

    edited = copy.deepcopy(model)

    by_layer: dict[str, list[int]] = {}
    for key in ablate_keys:
        layer, idx = parse_component_key(key)
        by_layer.setdefault(layer, []).append(idx)

    for layer_name, indices in by_layer.items():
        components = edited.components[layer_name]
        # Navigate to the target module (e.g. "h.1.mlp.c_fc" -> model.h[1].mlp.c_fc)
        target_module = edited.target_model
        for part in layer_name.split("."):
            target_module = getattr(target_module, part)

        for idx in indices:
            # Component contribution to W.T = V[:, c:c+1] @ U[c:c+1, :]
            # Transpose to get contribution to W
            contribution = (components.V[:, idx : idx + 1] @ components.U[idx : idx + 1, :]).T
            target_module.weight.data -= contribution  # pyright: ignore[reportOperatorIssue]

    return edited


# -- Forward functions with edits (mask-based, runtime) ------------------------


def edited_forward(
    model: ComponentModel,
    tokens: Int[Tensor, "1 seq"],
    edits: dict[str, float],
) -> Float[Tensor, "1 seq vocab"]:
    """Forward pass with component mask edits applied uniformly across all positions.

    edits maps component_key -> mask value: 0.0 = ablate, 1.0 = identity, >1 = boost.
    """
    seq_len = tokens.shape[1]
    device = tokens.device

    component_masks = {
        layer: torch.ones(1, seq_len, C, device=device) for layer, C in model.module_to_c.items()
    }
    for key, value in edits.items():
        layer, idx = parse_component_key(key)
        assert layer in component_masks, f"Unknown layer: {layer}"
        component_masks[layer][0, :, idx] = value

    mask_infos = make_mask_infos(component_masks, routing_masks="all")
    return model(tokens, mask_infos=mask_infos)


def ci_guided_forward(
    model: ComponentModel,
    tokens: Int[Tensor, "1 seq"],
    edits: dict[str, float],
    ci_threshold: float,
) -> Float[Tensor, "1 seq vocab"]:
    """Forward with edits applied only where component CI exceeds threshold.

    More surgical than edited_forward: only modifies positions where the component
    is actually active, reducing collateral damage.
    """
    seq_len = tokens.shape[1]
    device = tokens.device

    output_with_cache = model(tokens, cache_type="input")
    ci_outputs = model.calc_causal_importances(
        pre_weight_acts=output_with_cache.cache,
        sampling="continuous",
        detach_inputs=False,
    )
    ci_vals = ci_outputs.lower_leaky

    component_masks = {
        layer: torch.ones(1, seq_len, C, device=device) for layer, C in model.module_to_c.items()
    }
    for key, value in edits.items():
        layer, idx = parse_component_key(key)
        assert layer in component_masks, f"Unknown layer: {layer}"
        high_ci = ci_vals[layer][0, :, idx] > ci_threshold
        component_masks[layer][0, high_ci, idx] = value

    mask_infos = make_mask_infos(component_masks, routing_masks="all")
    return model(tokens, mask_infos=mask_infos)


def make_edit_fn(
    model: ComponentModel,
    edits: dict[str, float],
    ci_threshold: float | None = None,
) -> ForwardFn:
    """Create a reusable tokens -> logits function with edits baked in.

    If ci_threshold is provided, edits are applied only at positions where CI
    exceeds the threshold (more surgical). Otherwise, applied uniformly.
    """
    if ci_threshold is not None:
        return lambda tokens: ci_guided_forward(model, tokens, edits, ci_threshold)
    return lambda tokens: edited_forward(model, tokens, edits)


# -- Generation ----------------------------------------------------------------


def generate(
    forward_fn: ForwardFn,
    tokens: Int[Tensor, "1 seq"],
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
) -> str:
    """Greedy (temperature=0) or sampled generation from an arbitrary forward function."""
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        logits = forward_fn(generated)
        next_logits = logits[0, -1]
        if temperature == 0:
            next_id = next_logits.argmax()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:  # pyright: ignore[reportAttributeAccessIssue]
            break
    return tokenizer.decode(generated[0].tolist())  # pyright: ignore[reportAttributeAccessIssue]


# -- Measurement ---------------------------------------------------------------


@dataclass
class AblationEffect:
    mean_kl: float
    baseline_ppl: float
    edited_ppl: float
    n_tokens: int

    @property
    def ppl_increase_pct(self) -> float:
        return (self.edited_ppl / self.baseline_ppl - 1) * 100


def measure_kl(
    model: ComponentModel,
    edit_fn: ForwardFn,
    token_seqs: list[Int[Tensor, "1 seq"]],
) -> AblationEffect:
    """Measure KL divergence and perplexity shift between baseline and edited model."""
    total_kl = 0.0
    total_baseline_nll = 0.0
    total_edited_nll = 0.0
    total_tokens = 0

    for tokens in token_seqs:
        assert tokens.shape[0] == 1
        if tokens.shape[1] < 3:
            continue

        with torch.no_grad():
            baseline_logits = model(tokens)
            edited_logits = edit_fn(tokens)

        baseline_lp = F.log_softmax(baseline_logits[0, :-1], dim=-1)
        edited_lp = F.log_softmax(edited_logits[0, :-1], dim=-1)

        kl = F.kl_div(edited_lp, baseline_lp.exp(), reduction="sum", log_target=False)

        targets = tokens[0, 1:]
        baseline_nll = -baseline_lp[range(len(targets)), targets].sum()
        edited_nll = -edited_lp[range(len(targets)), targets].sum()

        total_kl += kl.item()
        total_baseline_nll += baseline_nll.item()
        total_edited_nll += edited_nll.item()
        total_tokens += len(targets)

    assert total_tokens > 0, "No tokens to evaluate"
    return AblationEffect(
        mean_kl=total_kl / total_tokens,
        baseline_ppl=torch.exp(torch.tensor(total_baseline_nll / total_tokens)).item(),
        edited_ppl=torch.exp(torch.tensor(total_edited_nll / total_tokens)).item(),
        n_tokens=total_tokens,
    )


@dataclass
class TokenGroupShift:
    group_name: str
    baseline_mean_prob: float
    edited_mean_prob: float
    n_positions: int

    @property
    def change_pct(self) -> float:
        if self.baseline_mean_prob == 0:
            return float("inf") if self.edited_mean_prob > 0 else 0.0
        return (self.edited_mean_prob / self.baseline_mean_prob - 1) * 100


def measure_token_probs(
    model: ComponentModel,
    edit_fn: ForwardFn,
    token_seqs: list[Int[Tensor, "1 seq"]],
    token_groups: dict[str, list[int]],
) -> dict[str, TokenGroupShift]:
    """Measure probability shift for named groups of token IDs.

    token_groups maps group names to token ID lists, e.g.:
        {"male": [he_id, him_id, his_id], "female": [she_id, her_id]}
    """
    baseline_sums: dict[str, float] = {name: 0.0 for name in token_groups}
    edited_sums: dict[str, float] = {name: 0.0 for name in token_groups}
    total_positions = 0

    for tokens in token_seqs:
        assert tokens.shape[0] == 1

        with torch.no_grad():
            baseline_logits = model(tokens)
            edited_logits = edit_fn(tokens)

        bp = F.softmax(baseline_logits[0], dim=-1)
        ep = F.softmax(edited_logits[0], dim=-1)

        for name, ids in token_groups.items():
            baseline_sums[name] += bp[:, ids].sum().item()
            edited_sums[name] += ep[:, ids].sum().item()
        total_positions += bp.shape[0]

    assert total_positions > 0
    return {
        name: TokenGroupShift(
            group_name=name,
            baseline_mean_prob=baseline_sums[name] / total_positions,
            edited_mean_prob=edited_sums[name] / total_positions,
            n_positions=total_positions,
        )
        for name in token_groups
    }
