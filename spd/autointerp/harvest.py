"""Harvest correlations and activation contexts in a single pass."""

import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.autointerp.schemas import (
    AUTOINTERP_DATA_DIR,
    ActivationExample,
    ComponentCorrelations,
    ComponentData,
    TokenStats,
)
from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


@dataclass
class HarvestConfig:
    wandb_path: str
    n_batches: int
    batch_size: int
    context_length: int
    ci_threshold: float
    n_examples: int
    context_window: int  # tokens on each side


@dataclass
class HarvestResult:
    components: list[ComponentData]
    config: HarvestConfig


def harvest(
    config: HarvestConfig,
    model: ComponentModel,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader[Int[Tensor, "B S"]],
    spd_config: Config,
) -> HarvestResult:
    """Single-pass harvest of correlations, token stats, and activation contexts."""
    device = next(model.parameters()).device
    layer_names = list(model.target_module_paths)
    C = model.C
    n_components = len(layer_names) * C
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    # Accumulators
    count_i = torch.zeros(n_components, device=device)
    count_ij = torch.zeros(n_components, n_components, device=device)
    ci_sum = torch.zeros(n_components, device=device)
    input_counts = torch.zeros(n_components, vocab_size, device=device)
    input_totals = torch.zeros(vocab_size, device=device)
    output_counts = torch.zeros(n_components, vocab_size, device=device)
    output_totals = torch.zeros(vocab_size, device=device)

    # Top-k activation examples per component
    examples: list[list[tuple[float, list[int], list[float], int]]] = [
        [] for _ in range(n_components)
    ]  # (ci, tokens, ci_values, active_pos)

    n_tokens = 0
    pad_id = tokenizer.pad_token_id
    assert isinstance(pad_id, int)

    train_iter = iter(train_loader)
    for _ in tqdm.tqdm(range(config.n_batches), desc="Harvesting"):
        batch = extract_batch_data(next(train_iter)).to(device)
        B, S = batch.shape
        n_tokens += B * S

        with torch.no_grad():
            out = model(batch, cache_type="input")
            logits = out.output
            probs = torch.softmax(logits, dim=-1)

            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=spd_config.sampling,
            ).lower_leaky

            # Stack CI: (B, S, n_layers, C) -> (B, S, n_components)
            ci_stacked = torch.stack([ci_dict[layer] for layer in layer_names], dim=2)
            ci_flat: Float[Tensor, "B S n_comp"] = ci_stacked.view(B, S, n_components)

            # Binary activations
            binary = (ci_flat > config.ci_threshold).float()
            flat_binary = binary.view(B * S, n_components)

            # Accumulate
            count_i += binary.sum(dim=(0, 1))
            count_ij += flat_binary.T @ flat_binary
            ci_sum += ci_flat.sum(dim=(0, 1))

            # Input token stats
            batch_flat = batch.view(B * S)
            input_onehot = torch.zeros(B * S, vocab_size, device=device)
            input_onehot.scatter_(1, batch_flat.unsqueeze(1), 1.0)
            input_counts += flat_binary.T @ input_onehot
            input_totals += input_onehot.sum(dim=0)

            # Output token stats
            probs_flat = probs.view(B * S, vocab_size)
            output_counts += flat_binary.T @ probs_flat
            output_totals += probs_flat.sum(dim=0)

            # Collect activation examples
            _collect_examples(
                batch,
                ci_flat,
                config.ci_threshold,
                config.n_examples,
                config.context_window,
                pad_id,
                examples,
            )

    # Build component data
    mean_ci = (ci_sum / n_tokens).cpu()
    components = _build_components(
        layer_names,
        C,
        mean_ci,
        config.ci_threshold,
        count_i.cpu(),
        count_ij.cpu(),
        n_tokens,
        input_counts.cpu(),
        input_totals.cpu(),
        output_counts.cpu(),
        output_totals.cpu(),
        examples,
        tokenizer,
        config.n_examples,
    )

    return HarvestResult(components=components, config=config)


def _collect_examples(
    batch: Int[Tensor, "B S"],
    ci: Float[Tensor, "B S n_comp"],
    threshold: float,
    n_examples: int,
    context_window: int,
    pad_id: int,
    examples: list[list[tuple[float, list[int], list[float], int]]],
) -> None:
    """Collect top-k activation examples per component."""
    device = batch.device

    # Pad for window extraction
    batch_padded = torch.nn.functional.pad(batch, (context_window, context_window), value=pad_id)

    # Find firings
    mask = ci > threshold
    b_idx, s_idx, c_idx = torch.where(mask)
    if len(b_idx) == 0:
        return

    ci_vals = ci[b_idx, s_idx, c_idx]

    # Extract windows
    window_size = 2 * context_window + 1
    offsets = torch.arange(-context_window, context_window + 1, device=device)
    s_padded = s_idx + context_window
    window_indices = s_padded.unsqueeze(1) + offsets
    b_expanded = b_idx.unsqueeze(1).expand(-1, window_size)

    tokens_window = batch_padded[b_expanded, window_indices]

    # Get CI for entire window (for this component)
    # ci shape: (B, S, n_comp), need to extract window for each firing
    ci_padded = torch.nn.functional.pad(ci, (0, 0, context_window, context_window), value=0.0)
    c_expanded = c_idx.unsqueeze(1).expand(-1, window_size)
    ci_window = ci_padded[b_expanded, window_indices, c_expanded]

    # Move to CPU for heap operations
    c_idx_cpu = c_idx.cpu().tolist()
    ci_vals_cpu = ci_vals.cpu().tolist()
    tokens_cpu = tokens_window.cpu().tolist()
    ci_window_cpu = ci_window.cpu().tolist()

    for i, comp in enumerate(c_idx_cpu):
        ci_val = ci_vals_cpu[i]
        heap = examples[comp]

        if len(heap) < n_examples:
            heapq.heappush(heap, (ci_val, tokens_cpu[i], ci_window_cpu[i], context_window))
        elif ci_val > heap[0][0]:
            heapq.heapreplace(heap, (ci_val, tokens_cpu[i], ci_window_cpu[i], context_window))


def _build_components(
    layer_names: list[str],
    C: int,
    mean_ci: Tensor,
    ci_threshold: float,
    count_i: Tensor,
    count_ij: Tensor,
    n_tokens: int,
    input_counts: Tensor,
    input_totals: Tensor,
    output_counts: Tensor,
    output_totals: Tensor,
    examples: list[list[tuple[float, list[int], list[float], int]]],
    tokenizer: PreTrainedTokenizerBase,
    n_examples: int,
) -> list[ComponentData]:
    """Build ComponentData for each live component."""
    components = []

    for layer_idx, layer in enumerate(layer_names):
        for c_idx in range(C):
            flat_idx = layer_idx * C + c_idx
            m_ci = float(mean_ci[flat_idx])

            # Skip dead components
            if m_ci < ci_threshold:
                continue

            component_key = f"{layer}:{c_idx}"

            # Activation examples
            heap = examples[flat_idx]
            sorted_examples = sorted(heap, key=lambda x: x[0], reverse=True)
            act_examples = [
                ActivationExample(
                    tokens=[tokenizer.decode([t]) for t in toks],
                    ci_values=ci_vals,
                    active_pos=pos,
                    active_ci=ci_val,
                )
                for ci_val, toks, ci_vals, pos in sorted_examples[:n_examples]
            ]

            # Token stats
            input_stats = _compute_token_stats(
                input_counts[flat_idx], input_totals, count_i[flat_idx], n_tokens, tokenizer
            )
            output_stats = _compute_token_stats(
                output_counts[flat_idx], output_totals, count_i[flat_idx], n_tokens, tokenizer
            )

            # Correlations
            correlations = _compute_correlations(
                flat_idx, count_i, count_ij, n_tokens, layer_names, C
            )

            components.append(
                ComponentData(
                    component_key=component_key,
                    layer=layer,
                    component_idx=c_idx,
                    mean_ci=m_ci,
                    activation_examples=act_examples,
                    input_token_stats=input_stats,
                    output_token_stats=output_stats,
                    correlations=correlations,
                )
            )

    return components


def _compute_token_stats(
    counts: Tensor,
    totals: Tensor,
    firing_count: Tensor,
    n_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = 10,
) -> TokenStats:
    """Compute precision/recall/PMI for tokens."""
    fc = float(firing_count)
    assert fc > 0

    valid = (counts > 0) & (totals > 0)
    recall = counts / fc
    precision = torch.where(totals > 0, counts / totals, torch.zeros_like(counts))
    pmi = torch.log(counts * n_tokens / (fc * totals + 1e-10))
    pmi = torch.where(valid, pmi, torch.full_like(pmi, float("-inf")))

    def top(vals: Tensor) -> list[tuple[str, float]]:
        masked = torch.where(valid, vals, torch.full_like(vals, float("-inf")))
        top_vals, top_idx = torch.topk(masked, min(top_k, int(valid.sum())))
        return [
            (tokenizer.decode([int(idx)]), round(float(v), 3))
            for idx, v in zip(top_idx.tolist(), top_vals.tolist(), strict=True)
            if v > float("-inf")
        ]

    return TokenStats(
        top_precision=top(precision),
        top_recall=top(recall),
        top_pmi=top(pmi),
    )


def _compute_correlations(
    idx: int,
    count_i: Tensor,
    count_ij: Tensor,
    n_tokens: int,
    layer_names: list[str],
    C: int,
    top_k: int = 5,
) -> ComponentCorrelations:
    """Compute correlation metrics with other components."""
    n_comp = len(count_i)
    ci = float(count_i[idx])
    assert ci > 0

    cooc = count_ij[idx]
    precision = cooc / ci
    recall = cooc / (count_i + 1e-10)
    pmi = torch.log(cooc * n_tokens / (ci * count_i + 1e-10))

    # Mask self and zeros
    mask = torch.ones(n_comp, dtype=torch.bool)
    mask[idx] = False
    mask &= count_i > 0
    mask &= cooc > 0

    def idx_to_key(i: int) -> str:
        layer_idx = i // C
        comp_idx = i % C
        return f"{layer_names[layer_idx]}:{comp_idx}"

    def top(vals: Tensor, largest: bool = True) -> list[tuple[str, float]]:
        masked = torch.where(
            mask, vals, torch.full_like(vals, float("-inf") if largest else float("inf"))
        )
        top_vals, top_idx = torch.topk(masked, min(top_k, int(mask.sum())), largest=largest)
        return [
            (idx_to_key(int(i)), round(float(v), 3))
            for i, v in zip(top_idx.tolist(), top_vals.tolist(), strict=True)
            if (v > float("-inf") if largest else v < float("inf"))
        ]

    return ComponentCorrelations(
        precision=top(precision),
        recall=top(recall),
        pmi=top(pmi, largest=True),
        bottom_pmi=top(pmi, largest=False),
    )


def save_harvest(result: HarvestResult, run_id: str) -> Path:
    """Save harvest result to disk."""
    out_dir = AUTOINTERP_DATA_DIR / run_id / "harvest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(result.config), indent=2))

    # Components as JSONL
    components_path = out_dir / "components.jsonl"
    with open(components_path, "w") as f:
        for comp in result.components:
            f.write(json.dumps(asdict(comp)) + "\n")

    return out_dir


def load_harvest(run_id: str) -> HarvestResult:
    """Load harvest result from disk."""
    harvest_dir = AUTOINTERP_DATA_DIR / run_id / "harvest"
    assert harvest_dir.exists(), f"No harvest found for {run_id}"

    config_path = harvest_dir / "config.json"
    config_data = json.loads(config_path.read_text())
    config = HarvestConfig(**config_data)

    components_path = harvest_dir / "components.jsonl"
    components = []
    with open(components_path) as f:
        for line in f:
            data = json.loads(line)
            # Reconstruct nested dataclasses
            data["activation_examples"] = [
                ActivationExample(**ex) for ex in data["activation_examples"]
            ]
            data["input_token_stats"] = TokenStats(**data["input_token_stats"])
            data["output_token_stats"] = TokenStats(**data["output_token_stats"])
            data["correlations"] = ComponentCorrelations(**data["correlations"])
            components.append(ComponentData(**data))

    return HarvestResult(components=components, config=config)
