"""Harvest correlations and activation contexts.

note for simplestories:
n_toks_total = 600,000,000
2.3 seconds per batch of size 256 * 512 toks each

((n_toks_total / (256 * 512)) * 2.3) seconds in hours ≈ 3 hours to harvest the whole dataset

"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.autointerp.schemas import (
    AUTOINTERP_DATA_DIR,
    ActivationExample,
    ComponentCorrelations,
    ComponentData,
    TokenStats,
)
from spd.utils.general_utils import extract_batch_data

# Sentinel for padding token windows at sequence boundaries.
# Using -1 ensures it's never a valid token ID.
WINDOW_PAD_SENTINEL = -1


# Entry: (ci_value, token_ids, ci_values_in_window, active_pos_in_window)
ActivationExampleTuple = tuple[float, list[int], list[float], int]


@dataclass
class ReservoirState:
    """Serializable state of a ReservoirSampler."""

    k: int
    samples: list[ActivationExampleTuple]
    n_seen: int


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

    def get_state(self) -> ReservoirState:
        return ReservoirState(k=self.k, samples=list(self.samples), n_seen=self.n_seen)

    @staticmethod
    def from_state(state: ReservoirState) -> "ReservoirSampler":
        sampler = ReservoirSampler(k=state.k)
        sampler.samples = list(state.samples)
        sampler.n_seen = state.n_seen
        return sampler

    @staticmethod
    def merge(states: list[ReservoirState]) -> "ReservoirSampler":
        """Merge multiple reservoir states into one.

        Each sample from reservoir i has probability n_i / sum(n_j) of being in the final reservoir.
        We achieve this by weighted random sampling from the combined pool.
        """
        assert len(states) > 0
        k = states[0].k
        assert all(s.k == k for s in states)

        total_seen = sum(s.n_seen for s in states)
        if total_seen == 0:
            return ReservoirSampler(k)

        # Build weighted pool: (sample, weight) where weight = n_seen for that reservoir
        weighted_samples: list[tuple[ActivationExampleTuple, int]] = []
        for state in states:
            for sample in state.samples:
                weighted_samples.append((sample, state.n_seen))

        # Sample k items with probability proportional to weight
        result = ReservoirSampler(k)
        result.n_seen = total_seen

        if len(weighted_samples) <= k:
            result.samples = [s for s, _ in weighted_samples]
        else:
            # Weighted random sampling without replacement
            weights = [w for _, w in weighted_samples]
            indices = []
            remaining_weights = list(weights)
            remaining_indices = list(range(len(weighted_samples)))

            for _ in range(k):
                r = random.random() * sum(remaining_weights)
                cumsum = 0.0
                for i, (idx, w) in enumerate(
                    zip(remaining_indices, remaining_weights, strict=True)
                ):
                    cumsum += w
                    if r <= cumsum:
                        indices.append(idx)
                        remaining_indices.pop(i)
                        remaining_weights.pop(i)
                        break

            result.samples = [weighted_samples[i][0] for i in indices]

        return result


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


@dataclass
class HarvesterState:
    """Serializable state of a Harvester for parallel merging."""

    layer_names: list[str]
    components_per_layer: int
    vocab_size: int
    ci_threshold: float
    max_examples_per_component: int
    context_tokens_per_side: int

    # Tensor accumulators (on CPU)
    firing_counts: Tensor
    cooccurrence_counts: Tensor
    ci_sums: Tensor
    input_token_counts: Tensor
    input_token_totals: Tensor
    output_token_prob_mass: Tensor
    output_token_prob_totals: Tensor
    total_tokens_processed: int

    # Reservoir states
    reservoir_states: list[ReservoirState]


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

        # Subsample if too many firings - we only need enough to feed the reservoirs
        MAX_FIRINGS_PER_BATCH = 10_000
        if len(batch_idx) > MAX_FIRINGS_PER_BATCH:
            keep = torch.randperm(len(batch_idx), device=self.device)[:MAX_FIRINGS_PER_BATCH]
            batch_idx = batch_idx[keep]
            seq_idx = seq_idx[keep]
            component_idx = component_idx[keep]

        batch_padded = torch.nn.functional.pad(
            batch,
            (self.context_tokens_per_side, self.context_tokens_per_side),
            value=WINDOW_PAD_SENTINEL,
        )
        ci_padded = torch.nn.functional.pad(
            ci, (0, 0, self.context_tokens_per_side, self.context_tokens_per_side), value=0.0
        )

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

        component_idx_list: list[int] = component_idx.cpu().tolist()
        ci_at_firing_list: list[float] = ci_at_firing.cpu().tolist()
        token_windows_list: list[list[int]] = token_windows.cpu().tolist()
        ci_windows_list: list[list[float]] = ci_windows.cpu().tolist()

        for i, comp_idx in enumerate(component_idx_list):
            self.activation_example_samplers[comp_idx].add(
                (
                    ci_at_firing_list[i],
                    token_windows_list[i],
                    ci_windows_list[i],
                    self.context_tokens_per_side,
                )
            )

    def get_state(self) -> HarvesterState:
        """Extract serializable state for parallel merging. Moves tensors to CPU."""
        return HarvesterState(
            layer_names=self.layer_names,
            components_per_layer=self.components_per_layer,
            vocab_size=self.vocab_size,
            ci_threshold=self.ci_threshold,
            max_examples_per_component=self.max_examples_per_component,
            context_tokens_per_side=self.context_tokens_per_side,
            firing_counts=self.firing_counts.cpu(),
            cooccurrence_counts=self.cooccurrence_counts.cpu(),
            ci_sums=self.ci_sums.cpu(),
            input_token_counts=self.input_token_counts.cpu(),
            input_token_totals=self.input_token_totals.cpu(),
            output_token_prob_mass=self.output_token_prob_mass.cpu(),
            output_token_prob_totals=self.output_token_prob_totals.cpu(),
            total_tokens_processed=self.total_tokens_processed,
            reservoir_states=[s.get_state() for s in self.activation_example_samplers],
        )

    @staticmethod
    def from_state(state: HarvesterState, device: torch.device) -> "Harvester":
        """Reconstruct Harvester from state. Used after merging."""
        harvester = Harvester(
            layer_names=state.layer_names,
            components_per_layer=state.components_per_layer,
            vocab_size=state.vocab_size,
            ci_threshold=state.ci_threshold,
            max_examples_per_component=state.max_examples_per_component,
            context_tokens_per_side=state.context_tokens_per_side,
            device=device,
        )
        harvester.firing_counts = state.firing_counts.to(device)
        harvester.cooccurrence_counts = state.cooccurrence_counts.to(device)
        harvester.ci_sums = state.ci_sums.to(device)
        harvester.input_token_counts = state.input_token_counts.to(device)
        harvester.input_token_totals = state.input_token_totals.to(device)
        harvester.output_token_prob_mass = state.output_token_prob_mass.to(device)
        harvester.output_token_prob_totals = state.output_token_prob_totals.to(device)
        harvester.total_tokens_processed = state.total_tokens_processed
        harvester.activation_example_samplers = [
            ReservoirSampler.from_state(s) for s in state.reservoir_states
        ]
        return harvester

    def _log_base_rate_summary(self, firing_counts: Tensor, input_token_totals: Tensor) -> None:
        """Log summary statistics about base rates for sanity checking."""
        # Component firing distribution
        active_counts = firing_counts[firing_counts > 0]
        if len(active_counts) == 0:
            print("  WARNING: No components fired above threshold!")
            return

        sorted_counts = active_counts.sort().values
        n_active = len(active_counts)
        print("\n  === Base Rate Summary ===")
        print(f"  Components with firings: {n_active} / {len(firing_counts)}")
        print(
            f"  Firing counts - min: {int(sorted_counts[0])}, "
            f"median: {int(sorted_counts[n_active // 2])}, "
            f"max: {int(sorted_counts[-1])}"
        )

        # Flag sparse components
        LOW_FIRING_THRESHOLD = 100
        n_sparse = int((active_counts < LOW_FIRING_THRESHOLD).sum())
        if n_sparse > 0:
            print(
                f"  WARNING: {n_sparse} components have <{LOW_FIRING_THRESHOLD} firings "
                f"(stats may be noisy)"
            )

        # Token occurrence distribution
        active_tokens = input_token_totals[input_token_totals > 0]
        sorted_token_counts = active_tokens.sort().values
        n_tokens = len(active_tokens)
        print(
            f"  Tokens seen: {n_tokens} unique, "
            f"occurrences - min: {int(sorted_token_counts[0])}, "
            f"median: {int(sorted_token_counts[n_tokens // 2])}, "
            f"max: {int(sorted_token_counts[-1])}"
        )

        # Flag rare tokens
        RARE_TOKEN_THRESHOLD = 10
        n_rare = int((active_tokens < RARE_TOKEN_THRESHOLD).sum())
        if n_rare > 0:
            print(
                f"  Note: {n_rare} tokens have <{RARE_TOKEN_THRESHOLD} occurrences "
                f"(high precision/recall with these may be spurious)"
            )
        print()

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

        self._log_base_rate_summary(firing_counts, input_token_totals)

        n_total = len(self.layer_names) * self.components_per_layer
        print(
            f"  Computing stats for {n_total} components across {len(self.layer_names)} layers..."
        )
        components = []
        for layer_idx, layer_name in tqdm.tqdm(
            enumerate(self.layer_names), total=len(self.layer_names), desc="Building components"
        ):
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


def merge_harvester_states(states: list[HarvesterState]) -> HarvesterState:
    """Merge multiple HarvesterStates from parallel workers into one."""
    assert len(states) > 0
    first = states[0]

    # Verify all states have compatible configs
    for s in states[1:]:
        assert s.layer_names == first.layer_names
        assert s.components_per_layer == first.components_per_layer
        assert s.vocab_size == first.vocab_size
        assert s.ci_threshold == first.ci_threshold

    # Sum tensor accumulators
    firing_counts = torch.stack([s.firing_counts for s in states]).sum(dim=0)
    cooccurrence_counts = torch.stack([s.cooccurrence_counts for s in states]).sum(dim=0)
    ci_sums = torch.stack([s.ci_sums for s in states]).sum(dim=0)
    input_token_counts = torch.stack([s.input_token_counts for s in states]).sum(dim=0)
    input_token_totals = torch.stack([s.input_token_totals for s in states]).sum(dim=0)
    output_token_prob_mass = torch.stack([s.output_token_prob_mass for s in states]).sum(dim=0)
    output_token_prob_totals = torch.stack([s.output_token_prob_totals for s in states]).sum(dim=0)
    total_tokens_processed = sum(s.total_tokens_processed for s in states)

    # Merge reservoir samplers
    n_components = len(first.reservoir_states)
    merged_reservoirs = []
    for comp_idx in range(n_components):
        component_reservoir_states = [s.reservoir_states[comp_idx] for s in states]
        merged = ReservoirSampler.merge(component_reservoir_states)
        merged_reservoirs.append(merged.get_state())

    return HarvesterState(
        layer_names=first.layer_names,
        components_per_layer=first.components_per_layer,
        vocab_size=first.vocab_size,
        ci_threshold=first.ci_threshold,
        max_examples_per_component=first.max_examples_per_component,
        context_tokens_per_side=first.context_tokens_per_side,
        firing_counts=firing_counts,
        cooccurrence_counts=cooccurrence_counts,
        ci_sums=ci_sums,
        input_token_counts=input_token_counts,
        input_token_totals=input_token_totals,
        output_token_prob_mass=output_token_prob_mass,
        output_token_prob_totals=output_token_prob_totals,
        total_tokens_processed=total_tokens_processed,
        reservoir_states=merged_reservoirs,
    )


def harvest(config: HarvestConfig) -> HarvestResult:
    """Single-pass harvest of correlations, token stats, and activation contexts."""
    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import ComponentModel, SPDRunInfo
    from spd.utils.distributed_utils import get_device

    device = torch.device(get_device())
    print(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(config.wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(
        spd_config, config.context_length, config.batch_size
    )

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


# Parallel harvesting


def _harvest_worker(
    rank: int,
    world_size: int,
    wandb_path: str,
    n_batches: int,
    batch_size: int,
    context_length: int,
    ci_threshold: float,
    activation_examples_per_component: int,
    activation_context_tokens_per_side: int,
    result_queue: "torch.multiprocessing.Queue[HarvesterState]",
) -> None:
    """Worker function for parallel harvesting. Runs in subprocess."""
    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import ComponentModel, SPDRunInfo

    device = torch.device(f"cuda:{rank}")
    print(f"[Worker {rank}] Starting on {device}")

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, tokenizer = train_loader_and_tokenizer(spd_config, context_length, batch_size)

    layer_names = list(model.target_module_paths)
    vocab_size = tokenizer.vocab_size
    assert isinstance(vocab_size, int)

    harvester = Harvester(
        layer_names=layer_names,
        components_per_layer=model.C,
        vocab_size=vocab_size,
        ci_threshold=ci_threshold,
        max_examples_per_component=activation_examples_per_component,
        context_tokens_per_side=activation_context_tokens_per_side,
        device=device,
    )

    # Each worker processes every world_size-th batch
    train_iter = iter(train_loader)
    batches_processed = 0
    for batch_idx in range(n_batches):
        batch_data = extract_batch_data(next(train_iter))
        if batch_idx % world_size != rank:
            continue

        batch = batch_data.to(device)
        with torch.no_grad():
            out = model(batch, cache_type="input")
            probs = torch.softmax(out.output, dim=-1)
            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=spd_config.sampling,
            ).lower_leaky

            ci_stacked = torch.stack([ci_dict[layer] for layer in layer_names], dim=2)
            B, S, _, _ = ci_stacked.shape
            ci_flat: Float[Tensor, "B S n_comp"] = ci_stacked.view(B, S, -1)
            harvester.process_batch(batch, ci_flat, probs)

        batches_processed += 1
        if batches_processed % 100 == 0:
            print(f"[Worker {rank}] Processed {batches_processed} batches")

    print(
        f"[Worker {rank}] Done. Processed {batches_processed} batches, {harvester.total_tokens_processed:,} tokens"
    )
    state = harvester.get_state()
    result_queue.put(state)


def harvest_parallel(
    config: HarvestConfig,
    n_gpus: int = 8,
) -> HarvestResult:
    """Parallel harvest across multiple GPUs using multiprocessing."""
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue[HarvesterState] = mp.Queue()

    processes = []
    for rank in range(n_gpus):
        p = mp.Process(
            target=_harvest_worker,
            args=(
                rank,
                n_gpus,
                config.wandb_path,
                config.n_batches,
                config.batch_size,
                config.context_length,
                config.ci_threshold,
                config.activation_examples_per_component,
                config.activation_context_tokens_per_side,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    print(f"Launched {n_gpus} workers. Waiting for results...")
    states = [result_queue.get() for _ in range(n_gpus)]

    for p in processes:
        p.join()

    print("All workers finished. Merging states...")
    merged_state = merge_harvester_states(states)
    print(f"Merged. Total tokens: {merged_state.total_tokens_processed:,}")

    # Load tokenizer for final build_results
    from spd.data import train_loader_and_tokenizer
    from spd.models.component_model import SPDRunInfo

    run_info = SPDRunInfo.from_path(config.wandb_path)
    _, tokenizer = train_loader_and_tokenizer(
        run_info.config, config.context_length, config.batch_size
    )

    # Build final results from merged state
    harvester = Harvester.from_state(merged_state, torch.device("cpu"))
    print("Building component results...")
    components = harvester.build_results(tokenizer)
    print(f"Built {len(components)} components")

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
