"""Harvest correlations and activation contexts in a single pass."""

import json
import random
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

# Sentinel for padding token windows at sequence boundaries.
# Using -1 ensures it's never a valid token ID.
WINDOW_PAD_SENTINEL = -1


# Entry: (ci_value, token_ids, ci_values_in_window, active_pos_in_window)
ActivationExampleTuple = tuple[float, list[int], list[float], int]


class ReservoirSampler:
    """Uniform random sampling from a stream via reservoir sampling."""

    def __init__(self, k: int):
        self.k = k
        self.samples: list[ActivationExampleTuple] = []
        self.n_seen = 0

    def add(self, item: ActivationExampleTuple) -> None:
        self.n_seen += 1
        if len(self.samples) < self.k:
            self.samples.append(item)
        elif random.randint(1, self.n_seen) <= self.k:
            self.samples[random.randrange(self.k)] = item


@dataclass
class HarvestConfig:
    wandb_path: str
    n_batches: int
    batch_size: int
    context_length: int
    ci_threshold: float
    activation_examples_per_component: int
    activation_context_tokens_per_side: int


@dataclass
class HarvestResult:
    components: list[ComponentData]
    config: HarvestConfig


class Harvester:
    """Accumulates correlations, token stats, and activation examples in a single pass."""

    def __init__(
        self,
        layer_names: list[str],
        components_per_layer: int,
        vocab_size: int,
        ci_threshold: float,
        max_examples_per_component: int,
        context_tokens_per_side: int,
        device: torch.device,
    ):
        self.layer_names = layer_names
        self.components_per_layer = components_per_layer
        self.vocab_size = vocab_size
        self.ci_threshold = ci_threshold
        self.max_examples_per_component = max_examples_per_component
        self.context_tokens_per_side = context_tokens_per_side
        self.device = device

        n_components = len(layer_names) * components_per_layer

        # Correlation accumulators
        self.firing_counts = torch.zeros(n_components, device=device)
        self.cooccurrence_counts = torch.zeros(n_components, n_components, device=device)
        self.ci_sums = torch.zeros(n_components, device=device)

        # Token stat accumulators
        # Input: actual counts of token co-occurrences with component firings
        self.input_token_counts = torch.zeros(n_components, vocab_size, device=device)
        self.input_token_totals = torch.zeros(vocab_size, device=device)
        # Output: accumulated probability mass (soft counts) weighted by firing
        self.output_token_prob_mass = torch.zeros(n_components, vocab_size, device=device)
        self.output_token_prob_totals = torch.zeros(vocab_size, device=device)

        # Reservoir samplers for activation examples per component
        self.activation_example_samplers = [
            ReservoirSampler(k=max_examples_per_component) for _ in range(n_components)
        ]

        self.total_tokens_processed = 0

    def process_batch(
        self,
        batch: Int[Tensor, "B S"],
        ci_flat: Float[Tensor, "B S n_comp"],
        output_probs: Float[Tensor, "B S V"],
    ) -> None:
        """Accumulate stats from a single batch."""
        B, S, n_components = ci_flat.shape
        self.total_tokens_processed += B * S

        # Binary firing indicators for correlations
        is_firing = (ci_flat > self.ci_threshold).float()
        is_firing_flat = is_firing.view(B * S, n_components)

        # Correlation accumulators
        self.firing_counts += is_firing.sum(dim=(0, 1))
        self.cooccurrence_counts += is_firing_flat.T @ is_firing_flat
        self.ci_sums += ci_flat.sum(dim=(0, 1))

        # Input token stats: which tokens co-occur with component firings
        batch_flat = batch.view(B * S)
        input_onehot = torch.zeros(B * S, self.vocab_size, device=self.device)
        input_onehot.scatter_(1, batch_flat.unsqueeze(1), 1.0)
        self.input_token_counts += is_firing_flat.T @ input_onehot
        self.input_token_totals += input_onehot.sum(dim=0)

        # Output token stats: probability mass on each token when component fires
        output_probs_flat = output_probs.view(B * S, self.vocab_size)
        self.output_token_prob_mass += is_firing_flat.T @ output_probs_flat
        self.output_token_prob_totals += output_probs_flat.sum(dim=0)

        self._collect_activation_examples(batch, ci_flat)

    def _collect_activation_examples(
        self,
        batch: Int[Tensor, "B S"],
        ci: Float[Tensor, "B S n_comp"],
    ) -> None:
        """Reservoir sample activation examples from high-CI firings."""
        is_firing = ci > self.ci_threshold
        batch_idx, seq_idx, component_idx = torch.where(is_firing)
        if len(batch_idx) == 0:
            return

        # Pad for window extraction
        batch_padded = torch.nn.functional.pad(
            batch,
            (self.context_tokens_per_side, self.context_tokens_per_side),
            value=WINDOW_PAD_SENTINEL,
        )
        ci_padded = torch.nn.functional.pad(
            ci, (0, 0, self.context_tokens_per_side, self.context_tokens_per_side), value=0.0
        )

        # Extract windows vectorized
        window_size = 2 * self.context_tokens_per_side + 1
        offsets = torch.arange(
            -self.context_tokens_per_side, self.context_tokens_per_side + 1, device=self.device
        )
        seq_idx_padded = seq_idx + self.context_tokens_per_side
        window_seq_indices = seq_idx_padded.unsqueeze(1) + offsets
        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
        component_idx_expanded = component_idx.unsqueeze(1).expand(-1, window_size)

        token_windows = batch_padded[batch_idx_expanded, window_seq_indices]
        ci_windows = ci_padded[batch_idx_expanded, window_seq_indices, component_idx_expanded]
        ci_at_firing = ci[batch_idx, seq_idx, component_idx]

        # Move to CPU and add to reservoirs
        component_idx_list = component_idx.cpu().tolist()
        ci_at_firing_list = ci_at_firing.cpu().tolist()
        token_windows_list = token_windows.cpu().tolist()
        ci_windows_list = ci_windows.cpu().tolist()

        for i, comp_idx in enumerate(component_idx_list):
            self.activation_example_samplers[comp_idx].add(
                (
                    ci_at_firing_list[i],
                    token_windows_list[i],
                    ci_windows_list[i],
                    self.context_tokens_per_side,
                )
            )

    def build_results(self, tokenizer: PreTrainedTokenizerBase) -> list[ComponentData]:
        """Convert accumulated state into ComponentData objects."""
        print("  Moving tensors to CPU...")
        mean_ci_per_component = (self.ci_sums / self.total_tokens_processed).cpu()
        firing_counts = self.firing_counts.cpu()
        cooccurrence_counts = self.cooccurrence_counts.cpu()
        input_token_counts = self.input_token_counts.cpu()
        input_token_totals = self.input_token_totals.cpu()
        output_token_prob_mass = self.output_token_prob_mass.cpu()
        output_token_prob_totals = self.output_token_prob_totals.cpu()

        n_total = len(self.layer_names) * self.components_per_layer
        print(
            f"  Computing stats for {n_total} components across {len(self.layer_names)} layers..."
        )
        components = []
        for layer_idx, layer_name in enumerate(self.layer_names):
            for component_idx in range(self.components_per_layer):
                flat_idx = layer_idx * self.components_per_layer + component_idx
                mean_ci = float(mean_ci_per_component[flat_idx])

                # Skip components that never fired above threshold
                component_firing_count = float(firing_counts[flat_idx])
                if component_firing_count == 0:
                    continue

                # Build activation examples from reservoir (sort by CI for display)
                sampler = self.activation_example_samplers[flat_idx]
                sorted_samples = sorted(sampler.samples, key=lambda x: x[0], reverse=True)

                def decode_token(token_id: int) -> str:
                    if token_id == WINDOW_PAD_SENTINEL:
                        return "<pad>"
                    return tokenizer.decode([token_id])

                activation_examples = [
                    ActivationExample(
                        tokens=[decode_token(t) for t in token_ids],
                        ci_values=ci_vals_in_window,
                        active_pos=active_pos_in_window,
                        active_ci=ci_at_active,
                    )
                    for ci_at_active, token_ids, ci_vals_in_window, active_pos_in_window in sorted_samples
                ]

                input_stats = _compute_token_stats(
                    input_token_counts[flat_idx],
                    input_token_totals,
                    firing_counts[flat_idx],
                    self.total_tokens_processed,
                    tokenizer,
                )
                output_stats = _compute_token_stats(
                    output_token_prob_mass[flat_idx],
                    output_token_prob_totals,
                    firing_counts[flat_idx],
                    self.total_tokens_processed,
                    tokenizer,
                )

                correlations = _compute_correlations(
                    flat_idx,
                    firing_counts,
                    cooccurrence_counts,
                    self.total_tokens_processed,
                    self.layer_names,
                    self.components_per_layer,
                )

                components.append(
                    ComponentData(
                        component_key=f"{layer_name}:{component_idx}",
                        layer=layer_name,
                        component_idx=component_idx,
                        mean_ci=mean_ci,
                        activation_examples=activation_examples,
                        input_token_stats=input_stats,
                        output_token_stats=output_stats,
                        correlations=correlations,
                    )
                )

        return components


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
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    harvester = Harvester(
        layer_names=layer_names,
        components_per_layer=model.C,
        vocab_size=vocab_size,
        ci_threshold=config.ci_threshold,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        device=device,
    )

    train_iter = iter(train_loader)
    for _ in tqdm.tqdm(range(config.n_batches), desc="Harvesting"):
        batch = extract_batch_data(next(train_iter)).to(device)

        with torch.no_grad():
            out = model(batch, cache_type="input")
            probs = torch.softmax(out.output, dim=-1)

            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=spd_config.sampling,
            ).lower_leaky

            # Stack CI: (B, S, n_layers, C) -> (B, S, n_components)
            ci_stacked = torch.stack([ci_dict[layer] for layer in layer_names], dim=2)
            B, S, _, _ = ci_stacked.shape
            ci_flat: Float[Tensor, "B S n_comp"] = ci_stacked.view(B, S, -1)

            harvester.process_batch(batch, ci_flat, probs)

    print(f"Batch processing complete. Total tokens: {harvester.total_tokens_processed:,}")
    print("Building component results...")
    components = harvester.build_results(tokenizer)
    print(f"Built {len(components)} components (skipped components with no firings)")
    return HarvestResult(components=components, config=config)


# Pure functions for computing stats from accumulated tensors


def _compute_token_stats(
    token_mass_for_component: Tensor,
    token_mass_totals: Tensor,
    component_firing_count: Tensor,
    total_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = 10,
) -> TokenStats:
    """Compute precision/recall/PMI for tokens associated with a component.

    Works with either hard counts (input tokens) or soft probability mass (output tokens).
    """
    firing_count = float(component_firing_count)
    assert firing_count > 0

    has_cooccurrence = (token_mass_for_component > 0) & (token_mass_totals > 0)
    recall = token_mass_for_component / firing_count
    precision = torch.where(
        token_mass_totals > 0,
        token_mass_for_component / token_mass_totals,
        torch.zeros_like(token_mass_totals),
    )
    pmi = torch.log(
        token_mass_for_component * total_tokens / (firing_count * token_mass_totals + 1e-10)
    )
    pmi = torch.where(has_cooccurrence, pmi, torch.full_like(pmi, float("-inf")))

    def get_top_k(values: Tensor) -> list[tuple[str, float]]:
        masked = torch.where(has_cooccurrence, values, torch.full_like(values, float("-inf")))
        top_values, top_indices = torch.topk(masked, min(top_k, int(has_cooccurrence.sum())))
        return [
            (tokenizer.decode([int(token_id)]), round(float(value), 3))
            for token_id, value in zip(top_indices.tolist(), top_values.tolist(), strict=True)
            if value > float("-inf")
        ]

    return TokenStats(
        top_precision=get_top_k(precision),
        top_recall=get_top_k(recall),
        top_pmi=get_top_k(pmi),
    )


def _compute_correlations(
    component_flat_idx: int,
    firing_counts: Tensor,
    cooccurrence_counts: Tensor,
    total_tokens: int,
    layer_names: list[str],
    components_per_layer: int,
    top_k: int = 5,
) -> ComponentCorrelations:
    """Compute correlation metrics between this component and all others."""
    n_components = len(firing_counts)
    this_firing_count = float(firing_counts[component_flat_idx])
    assert this_firing_count > 0

    cooccurrences_with_this = cooccurrence_counts[component_flat_idx]
    precision = cooccurrences_with_this / this_firing_count
    recall = cooccurrences_with_this / (firing_counts + 1e-10)
    pmi = torch.log(
        cooccurrences_with_this * total_tokens / (this_firing_count * firing_counts + 1e-10)
    )

    # Exclude self and components with no co-occurrences
    is_valid = torch.ones(n_components, dtype=torch.bool)
    is_valid[component_flat_idx] = False
    is_valid &= firing_counts > 0
    is_valid &= cooccurrences_with_this > 0

    def flat_idx_to_key(flat_idx: int) -> str:
        layer_idx = flat_idx // components_per_layer
        component_idx = flat_idx % components_per_layer
        return f"{layer_names[layer_idx]}:{component_idx}"

    def get_top_k(values: Tensor, largest: bool = True) -> list[tuple[str, float]]:
        sentinel = float("-inf") if largest else float("inf")
        masked = torch.where(is_valid, values, torch.full_like(values, sentinel))
        top_values, top_indices = torch.topk(
            masked, min(top_k, int(is_valid.sum())), largest=largest
        )
        return [
            (flat_idx_to_key(int(flat_idx)), round(float(value), 3))
            for flat_idx, value in zip(top_indices.tolist(), top_values.tolist(), strict=True)
            if (value > float("-inf") if largest else value < float("inf"))
        ]

    return ComponentCorrelations(
        precision=get_top_k(precision),
        recall=get_top_k(recall),
        pmi=get_top_k(pmi, largest=True),
        bottom_pmi=get_top_k(pmi, largest=False),
    )


# I/O


def save_harvest(result: HarvestResult, run_id: str) -> Path:
    """Save harvest result to disk."""
    out_dir = AUTOINTERP_DATA_DIR / run_id / "harvest"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(result.config), indent=2))

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
            data["activation_examples"] = [
                ActivationExample(**ex) for ex in data["activation_examples"]
            ]
            data["input_token_stats"] = TokenStats(**data["input_token_stats"])
            data["output_token_stats"] = TokenStats(**data["output_token_stats"])
            data["correlations"] = ComponentCorrelations(**data["correlations"])
            components.append(ComponentData(**data))

    return HarvestResult(components=components, config=config)
