"""Harvester for collecting component statistics in a single pass."""

from dataclasses import dataclass
from typing import cast

import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor

from spd.autointerp.lib.reservior_sampler import ReservoirSampler, ReservoirState
from spd.autointerp.schemas import ActivationExample, ComponentData, ComponentTokenPMI

# Sentinel for padding token windows at sequence boundaries.
WINDOW_PAD_SENTINEL = -1

# Entry: (token_ids, ci_values_in_window)
ActivationExampleTuple = tuple[list[int], list[float]]


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
    ci_sums: Tensor
    input_token_counts: Tensor
    input_token_totals: Tensor
    output_token_prob_mass: Tensor
    output_token_prob_totals: Tensor
    total_tokens_processed: int

    # Reservoir states
    reservoir_states: list[ReservoirState[ActivationExampleTuple]]

    @staticmethod
    def merge(states: list["HarvesterState"]) -> "HarvesterState":
        """Merge multiple HarvesterStates from parallel workers into one."""
        assert len(states) > 0
        first = states[0]

        for s in states[1:]:
            assert s.layer_names == first.layer_names
            assert s.components_per_layer == first.components_per_layer
            assert s.vocab_size == first.vocab_size
            assert s.ci_threshold == first.ci_threshold

        # Sum tensor accumulators
        firing_counts = torch.stack([s.firing_counts for s in states]).sum(dim=0)
        ci_sums = torch.stack([s.ci_sums for s in states]).sum(dim=0)
        input_token_counts = torch.stack([s.input_token_counts for s in states]).sum(dim=0)
        input_token_totals = torch.stack([s.input_token_totals for s in states]).sum(dim=0)
        output_token_prob_mass = torch.stack([s.output_token_prob_mass for s in states]).sum(dim=0)
        output_token_prob_totals = torch.stack([s.output_token_prob_totals for s in states]).sum(
            dim=0
        )
        total_tokens_processed = sum(s.total_tokens_processed for s in states)

        # Merge reservoir states
        n_components = len(first.reservoir_states)
        merged_reservoirs = [
            ReservoirState.merge([s.reservoir_states[i] for s in states])
            for i in range(n_components)
        ]

        return HarvesterState(
            layer_names=first.layer_names,
            components_per_layer=first.components_per_layer,
            vocab_size=first.vocab_size,
            ci_threshold=first.ci_threshold,
            max_examples_per_component=first.max_examples_per_component,
            context_tokens_per_side=first.context_tokens_per_side,
            firing_counts=firing_counts,
            ci_sums=ci_sums,
            input_token_counts=input_token_counts,
            input_token_totals=input_token_totals,
            output_token_prob_mass=output_token_prob_mass,
            output_token_prob_totals=output_token_prob_totals,
            total_tokens_processed=total_tokens_processed,
            reservoir_states=merged_reservoirs,
        )


class Harvester:
    """Accumulates component statistics in a single pass over data."""

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
        self.ci_sums = torch.zeros(n_components, device=device)

        # Token stat accumulators
        self.input_token_counts: Int[Tensor, "n_components vocab"] = torch.zeros(
            n_components, vocab_size, device=device, dtype=torch.long
        )
        self.input_token_totals: Int[Tensor, " vocab"] = torch.zeros(
            vocab_size, device=device, dtype=torch.long
        )
        self.output_token_prob_mass: Float[Tensor, "n_components vocab"] = torch.zeros(
            n_components, vocab_size, device=device
        )
        self.output_token_prob_totals: Float[Tensor, " vocab"] = torch.zeros(
            vocab_size, device=device
        )

        # Reservoir samplers for activation examples
        self.activation_example_samplers = [
            ReservoirSampler[ActivationExampleTuple](k=max_examples_per_component)
            for _ in range(n_components)
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

        is_firing = (ci_flat > self.ci_threshold).float()
        is_firing_flat = is_firing.view(B * S, n_components)
        batch_flat = batch.view(B * S)
        output_probs_flat = output_probs.view(B * S, self.vocab_size)

        self._accumulate_firing_stats(ci_flat, is_firing)
        self._accumulate_input_token_stats(batch_flat, is_firing_flat, n_components)
        self._accumulate_output_token_stats(output_probs_flat, is_firing_flat)
        self._collect_activation_examples(batch, ci_flat)

    def _accumulate_firing_stats(
        self,
        ci_flat: Float[Tensor, "B S n_comp"],
        is_firing: Float[Tensor, "B S n_comp"],
    ) -> None:
        self.firing_counts += is_firing.sum(dim=(0, 1))
        self.ci_sums += ci_flat.sum(dim=(0, 1))

    def _accumulate_input_token_stats(
        self, batch_flat: Tensor, is_firing_flat: Tensor, n_components: int
    ) -> None:
        token_indices = batch_flat.unsqueeze(0).expand(n_components, -1)
        self.input_token_counts.scatter_add_(
            dim=1, index=token_indices, src=is_firing_flat.T.long()
        )
        self.input_token_totals.scatter_add_(
            dim=0,
            index=batch_flat,
            src=torch.ones(batch_flat.shape[0], device=self.device, dtype=torch.long),
        )

    def _accumulate_output_token_stats(
        self, output_probs_flat: Tensor, is_firing_flat: Tensor
    ) -> None:
        self.output_token_prob_mass += is_firing_flat.T @ output_probs_flat
        self.output_token_prob_totals += output_probs_flat.sum(dim=0)

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
        
        print(f"got {len(batch_idx)} firings")

        # Subsample if too many firings
        MAX_FIRINGS_PER_BATCH = 10_000
        if len(batch_idx) > MAX_FIRINGS_PER_BATCH:
            keep = torch.randperm(len(batch_idx), device=self.device)[:MAX_FIRINGS_PER_BATCH]
            batch_idx = batch_idx[keep]
            seq_idx = seq_idx[keep]
            component_idx = component_idx[keep]

        # Pad for context window extraction
        batch_padded = torch.nn.functional.pad(
            batch,
            (self.context_tokens_per_side, self.context_tokens_per_side),
            value=WINDOW_PAD_SENTINEL,
        )
        ci_padded = torch.nn.functional.pad(
            ci, (0, 0, self.context_tokens_per_side, self.context_tokens_per_side), value=0.0
        )

        # Extract token windows around each firing
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

        # Add to reservoir samplers
        for comp_idx, tokens, ci_vals in zip(
            cast(list[int], component_idx.cpu().tolist()),
            cast(list[list[int]], token_windows.cpu().tolist()),
            cast(list[list[float]], ci_windows.cpu().tolist()),
            strict=True,
        ):
            self.activation_example_samplers[comp_idx].add((tokens, ci_vals))

    def get_state(self) -> HarvesterState:
        """Extract serializable state for parallel merging."""
        return HarvesterState(
            layer_names=self.layer_names,
            components_per_layer=self.components_per_layer,
            vocab_size=self.vocab_size,
            ci_threshold=self.ci_threshold,
            max_examples_per_component=self.max_examples_per_component,
            context_tokens_per_side=self.context_tokens_per_side,
            firing_counts=self.firing_counts.cpu(),
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
        """Reconstruct Harvester from state."""
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

    def build_results(self, pmi_top_k_tokens: int) -> list[ComponentData]:
        """Convert accumulated state into ComponentData objects."""
        print("  Moving tensors to CPU...")
        mean_ci_per_component = (self.ci_sums / self.total_tokens_processed).cpu()
        firing_counts = self.firing_counts.cpu()
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

                component_firing_count = float(firing_counts[flat_idx])
                if component_firing_count == 0:
                    continue

                # Build activation examples from reservoir
                sampler = self.activation_example_samplers[flat_idx]
                sorted_samples = sorted(sampler.samples, key=lambda x: x[0], reverse=True)
                activation_examples = [
                    ActivationExample(token_ids=token_ids, ci_values=ci_values)
                    for token_ids, ci_values in sorted_samples
                ]

                input_token_pmi = _compute_token_pmi(
                    token_mass_for_component=input_token_counts[flat_idx],
                    token_mass_totals=input_token_totals,
                    component_firing_count=component_firing_count,
                    total_tokens=self.total_tokens_processed,
                    top_k=pmi_top_k_tokens,
                )

                output_token_pmi = _compute_token_pmi(
                    token_mass_for_component=output_token_prob_mass[flat_idx],
                    token_mass_totals=output_token_prob_totals,
                    component_firing_count=component_firing_count,
                    total_tokens=self.total_tokens_processed,
                    top_k=pmi_top_k_tokens,
                )

                components.append(
                    ComponentData(
                        component_key=f"{layer_name}:{component_idx}",
                        layer=layer_name,
                        component_idx=component_idx,
                        mean_ci=mean_ci,
                        activation_examples=activation_examples,
                        input_token_pmi=input_token_pmi,
                        output_token_pmi=output_token_pmi,
                    )
                )

        return components

    def _log_base_rate_summary(self, firing_counts: Tensor, input_token_totals: Tensor) -> None:
        """Log summary statistics about base rates."""
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

        LOW_FIRING_THRESHOLD = 100
        n_sparse = int((active_counts < LOW_FIRING_THRESHOLD).sum())
        if n_sparse > 0:
            print(
                f"  WARNING: {n_sparse} components have <{LOW_FIRING_THRESHOLD} firings "
                f"(stats may be noisy)"
            )

        active_tokens = input_token_totals[input_token_totals > 0]
        sorted_token_counts = active_tokens.sort().values
        n_tokens = len(active_tokens)
        print(
            f"  Tokens seen: {n_tokens} unique, "
            f"occurrences - min: {int(sorted_token_counts[0])}, "
            f"median: {int(sorted_token_counts[n_tokens // 2])}, "
            f"max: {int(sorted_token_counts[-1])}"
        )

        RARE_TOKEN_THRESHOLD = 10
        n_rare = int((active_tokens < RARE_TOKEN_THRESHOLD).sum())
        if n_rare > 0:
            print(
                f"  Note: {n_rare} tokens have <{RARE_TOKEN_THRESHOLD} occurrences "
                f"(high precision/recall with these may be spurious)"
            )
        print()


def _compute_token_pmi(
    token_mass_for_component: Tensor,
    token_mass_totals: Tensor,
    component_firing_count: float,
    total_tokens: int,
    top_k: int,
) -> ComponentTokenPMI:
    """Compute PMI for tokens associated with a component."""
    has_cooccurrence = (token_mass_for_component > 0) & (token_mass_totals > 0)

    pmi = torch.log(
        token_mass_for_component
        * total_tokens
        / (component_firing_count * token_mass_totals + 1e-10)
    )

    pmi = torch.where(has_cooccurrence, pmi, torch.full_like(pmi, float("-inf")))

    top = torch.topk(pmi, min(top_k, int(has_cooccurrence.sum())))
    bottom = torch.topk(pmi, min(top_k, int(has_cooccurrence.sum())), largest=False)

    top_pmi = [
        (int(token_id), float(value))
        for token_id, value in zip(top.indices.tolist(), top.values.tolist(), strict=True)
    ]
    bottom_pmi = [
        (int(token_id), float(value))
        for token_id, value in zip(bottom.indices.tolist(), bottom.values.tolist(), strict=True)
    ]

    return ComponentTokenPMI(top=top_pmi, bottom=bottom_pmi)
