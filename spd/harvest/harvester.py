"""Harvester for collecting component statistics in a single pass."""

from dataclasses import dataclass
from typing import cast

import torch
import tqdm
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor

from spd.harvest.reservoir_sampler import ReservoirSampler, ReservoirState
from spd.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

# Sentinel for padding token windows at sequence boundaries.
WINDOW_PAD_SENTINEL = -1

# Entry: (token_ids, ci_values_in_window, component_acts_in_window)
ActivationExampleTuple = tuple[list[int], list[float], list[float]]


@dataclass
class HarvesterState:
    """Serializable state of a Harvester for parallel merging."""

    layer_names: list[str]
    c_per_layer: dict[str, int]  # Maps layer name -> number of components
    vocab_size: int
    ci_threshold: float
    max_examples_per_component: int
    context_tokens_per_side: int

    # Tensor accumulators (on CPU)
    firing_counts: Tensor
    ci_sums: Tensor
    count_ij: Tensor  # Component co-occurrence matrix
    input_token_counts: Tensor
    input_token_totals: Tensor
    output_token_prob_mass: Tensor
    output_token_prob_totals: Tensor
    total_tokens_processed: int

    # Reservoir states
    reservoir_states: list[ReservoirState[ActivationExampleTuple]]

    def merge_into(self, other: "HarvesterState") -> None:
        """Merge another HarvesterState into this one (in-place accumulation).

        This is the streaming merge primitive used to avoid OOM when merging many workers.
        """
        assert other.layer_names == self.layer_names
        assert other.c_per_layer == self.c_per_layer
        assert other.vocab_size == self.vocab_size
        assert other.ci_threshold == self.ci_threshold
        assert other.max_examples_per_component == self.max_examples_per_component
        assert other.context_tokens_per_side == self.context_tokens_per_side
        assert len(other.reservoir_states) == len(self.reservoir_states)

        # Accumulate tensor stats
        self.firing_counts += other.firing_counts
        self.ci_sums += other.ci_sums
        self.count_ij += other.count_ij
        self.input_token_counts += other.input_token_counts
        self.input_token_totals += other.input_token_totals
        self.output_token_prob_mass += other.output_token_prob_mass
        self.output_token_prob_totals += other.output_token_prob_totals
        self.total_tokens_processed += other.total_tokens_processed

        # Merge reservoir states pairwise
        for i in range(len(self.reservoir_states)):
            merged = ReservoirState.merge([self.reservoir_states[i], other.reservoir_states[i]])
            self.reservoir_states[i] = merged


class Harvester:
    """Accumulates component statistics in a single pass over data."""

    def __init__(
        self,
        layer_names: list[str],
        c_per_layer: dict[str, int],
        vocab_size: int,
        ci_threshold: float,
        max_examples_per_component: int,
        context_tokens_per_side: int,
        device: torch.device,
    ):
        self.layer_names = layer_names
        self.c_per_layer = c_per_layer
        self.vocab_size = vocab_size
        self.ci_threshold = ci_threshold
        self.max_examples_per_component = max_examples_per_component
        self.context_tokens_per_side = context_tokens_per_side
        self.device = device

        # Precompute layer offsets for flat indexing
        # layer_offsets[layer_name] gives the starting flat index for that layer's components
        self.layer_offsets: dict[str, int] = {}
        offset = 0
        for layer in layer_names:
            self.layer_offsets[layer] = offset
            offset += c_per_layer[layer]

        n_components = sum(c_per_layer[layer] for layer in layer_names)

        # Correlation accumulators
        self.firing_counts = torch.zeros(n_components, device=device)
        self.ci_sums = torch.zeros(n_components, device=device)
        self.count_ij = torch.zeros(n_components, n_components, device=device, dtype=torch.float32)

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
        ci: Float[Tensor, "B S n_comp"],
        output_probs: Float[Tensor, "B S V"],
        subcomp_acts: Float[Tensor, "B S n_comp"],
    ) -> None:
        """Accumulate stats from a single batch.

        Args:
            batch: Token IDs
            ci: Causal importance values per component
            output_probs: Output probabilities
            subcomp_acts: Normalized subcomponent activations: (v_i^T @ a) * ||u_i||.
        """
        self.total_tokens_processed += batch.numel()

        firing = (ci > self.ci_threshold).float()

        firing_flat = rearrange(firing, "b s c -> (b s) c")
        batch_flat = rearrange(batch, "b s -> (b s)")
        output_probs_flat = rearrange(output_probs, "b s v -> (b s) v")

        self._accumulate_firing_stats(ci, firing)
        self._accumulate_cooccurrence_stats(firing_flat)
        self._accumulate_input_token_stats(batch_flat, firing_flat)
        self._accumulate_output_token_stats(output_probs_flat, firing_flat)
        self._collect_activation_examples(batch, ci, subcomp_acts)

    def _accumulate_firing_stats(
        self,
        ci: Float[Tensor, "B S n_comp"],
        firing: Float[Tensor, "B S n_comp"],
    ) -> None:
        self.firing_counts += reduce(firing, "b s c -> c", "sum")
        self.ci_sums += reduce(ci, "b s c -> c", "sum")

    def _accumulate_cooccurrence_stats(self, firing_flat: Float[Tensor, "pos n_comp"]) -> None:
        """Accumulate component-component co-occurrence counts."""
        self.count_ij += einsum(firing_flat, firing_flat, "pos c1, pos c2 -> c1 c2")

    def _accumulate_input_token_stats(
        self,
        batch_flat: Int[Tensor, " pos"],
        firing_flat: Float[Tensor, "pos n_comp"],
    ) -> None:
        """Accumulate which input tokens caused each component to fire.

        Uses scatter_add_ to efficiently accumulate counts into a sparse [n_comp, vocab] matrix.
        For each position, we add the firing indicator (0 or 1) to the count for that token.

        Equivalent to: for each pos, for each component c:
            input_token_counts[c, batch_flat[pos]] += firing_flat[pos, c]
        """
        n_components = firing_flat.shape[1]
        # Broadcast token_ids to [n_comp, pos] so scatter_add_ can index into vocab dim
        token_indices = batch_flat.unsqueeze(0).expand(n_components, -1)
        # input_token_counts[c, token_indices[c, pos]] += firing_flat.T[c, pos]
        self.input_token_counts.scatter_add_(
            dim=1, index=token_indices, src=rearrange(firing_flat, "pos c -> c pos").long()
        )
        # Count total occurrences of each token (denominator for precision)
        self.input_token_totals.scatter_add_(
            dim=0,
            index=batch_flat,
            src=torch.ones(batch_flat.shape[0], device=self.device, dtype=torch.long),
        )

    def _accumulate_output_token_stats(
        self,
        output_probs_flat: Float[Tensor, "pos vocab"],
        firing_flat: Float[Tensor, "pos n_comp"],
    ) -> None:
        """Accumulate which output tokens each component predicts.

        Unlike input tokens (hard counts), we accumulate probability mass.
        When component c fires, we add the full output probability distribution,
        weighted by the firing indicator.
        """
        # Sum of P(token | pos) for positions where component c fired
        self.output_token_prob_mass += einsum(firing_flat, output_probs_flat, "pos c, pos v -> c v")
        # Sum of P(token | pos) across all positions (for normalization)
        self.output_token_prob_totals += reduce(output_probs_flat, "pos v -> v", "sum")

    def _collect_activation_examples(
        self,
        batch: Int[Tensor, "B S"],
        ci: Float[Tensor, "B S n_comp"],
        subcomp_acts: Float[Tensor, "B S n_comp"],
    ) -> None:
        """Reservoir sample activation examples from high-CI firings."""
        firing = ci > self.ci_threshold
        batch_idx, seq_idx, component_idx = torch.where(firing)
        if len(batch_idx) == 0:
            return

        # Cap firings per component to ensure rare components get examples.
        # With ~3000 batches and topk=1000 examples, we only need ~1 per component per batch.
        MAX_FIRINGS_PER_COMPONENT = 5
        keep_mask = sample_at_most_n_per_group(component_idx, MAX_FIRINGS_PER_COMPONENT)
        batch_idx = batch_idx[keep_mask]
        seq_idx = seq_idx[keep_mask]
        component_idx = component_idx[keep_mask]

        # Pad sequences so we can extract windows at boundaries without going out of bounds.
        # E.g. if context_tokens_per_side=3, a firing at seq_idx=0 needs tokens at [-3, -2, -1, 0, 1, 2, 3]
        # Padding with sentinel allows uniform window extraction; sentinels are filtered in display.
        batch_padded = torch.nn.functional.pad(
            batch,
            (self.context_tokens_per_side, self.context_tokens_per_side),
            value=WINDOW_PAD_SENTINEL,
        )
        ci_padded = torch.nn.functional.pad(
            ci, (0, 0, self.context_tokens_per_side, self.context_tokens_per_side), value=0.0
        )
        subcomp_acts_padded = torch.nn.functional.pad(
            subcomp_acts,
            (0, 0, self.context_tokens_per_side, self.context_tokens_per_side),
            value=0.0,
        )

        # Build indices to extract [n_firings, window_size] windows via advanced indexing.
        # For each firing, we want tokens at [seq_idx - k, ..., seq_idx, ..., seq_idx + k]
        window_size = 2 * self.context_tokens_per_side + 1
        offsets = torch.arange(
            -self.context_tokens_per_side, self.context_tokens_per_side + 1, device=self.device
        )
        seq_idx_padded = seq_idx + self.context_tokens_per_side  # Adjust for padding
        window_seq_indices = seq_idx_padded.unsqueeze(1) + offsets  # [n_firings, window_size]
        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
        component_idx_expanded = component_idx.unsqueeze(1).expand(-1, window_size)

        # Advanced indexing: token_windows[i, j] = batch_padded[batch_idx[i], window_seq_indices[i, j]]
        token_windows = batch_padded[batch_idx_expanded, window_seq_indices]
        ci_windows = ci_padded[batch_idx_expanded, window_seq_indices, component_idx_expanded]
        component_act_windows = subcomp_acts_padded[
            batch_idx_expanded, window_seq_indices, component_idx_expanded
        ]

        # Add to reservoir samplers
        for comp_idx, tokens, ci_vals, component_acts in zip(
            cast(list[int], component_idx.cpu().tolist()),
            cast(list[list[int]], token_windows.cpu().tolist()),
            cast(list[list[float]], ci_windows.cpu().tolist()),
            cast(list[list[float]], component_act_windows.cpu().tolist()),
            strict=True,
        ):
            self.activation_example_samplers[comp_idx].add((tokens, ci_vals, component_acts))

    def get_state(self) -> HarvesterState:
        """Extract serializable state for parallel merging."""
        return HarvesterState(
            layer_names=self.layer_names,
            c_per_layer=self.c_per_layer,
            vocab_size=self.vocab_size,
            ci_threshold=self.ci_threshold,
            max_examples_per_component=self.max_examples_per_component,
            context_tokens_per_side=self.context_tokens_per_side,
            firing_counts=self.firing_counts.cpu(),
            ci_sums=self.ci_sums.cpu(),
            count_ij=self.count_ij.cpu(),
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
            c_per_layer=state.c_per_layer,
            vocab_size=state.vocab_size,
            ci_threshold=state.ci_threshold,
            max_examples_per_component=state.max_examples_per_component,
            context_tokens_per_side=state.context_tokens_per_side,
            device=device,
        )
        harvester.firing_counts = state.firing_counts.to(device)
        harvester.ci_sums = state.ci_sums.to(device)
        harvester.count_ij = state.count_ij.to(device)
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

        n_total = sum(self.c_per_layer[layer] for layer in self.layer_names)
        print(
            f"  Computing stats for {n_total} components across {len(self.layer_names)} layers..."
        )
        components = []
        for layer_name in tqdm.tqdm(self.layer_names, desc="Building components"):
            layer_offset = self.layer_offsets[layer_name]
            layer_c = self.c_per_layer[layer_name]

            for component_idx in range(layer_c):
                flat_idx = layer_offset + component_idx
                mean_ci = float(mean_ci_per_component[flat_idx])

                component_firing_count = float(firing_counts[flat_idx])
                if component_firing_count == 0:
                    continue

                # Build activation examples from reservoir (uniform random sample)
                sampler = self.activation_example_samplers[flat_idx]
                activation_examples = [
                    ActivationExample(
                        token_ids=token_ids, ci_values=ci_values, component_acts=component_acts
                    )
                    for token_ids, ci_values, component_acts in sampler.samples
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
    top, bottom = top_k_pmi(
        cooccurrence_counts=token_mass_for_component,
        marginal_counts=token_mass_totals,
        target_count=component_firing_count,
        total_count=total_tokens,
        top_k=top_k,
    )
    return ComponentTokenPMI(top=top, bottom=bottom)
