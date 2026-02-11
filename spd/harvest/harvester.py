"""Harvester for collecting component statistics in a single pass."""

from dataclasses import dataclass
from typing import cast

import torch
import tqdm
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor

from spd.harvest.reservoir_sampler import ReservoirSampler
from spd.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

# Sentinel for padding token windows at sequence boundaries.
WINDOW_PAD_SENTINEL = -1

# Entry: (token_ids, ci_values_in_window, component_acts_in_window)
ActivationExampleTuple = tuple[list[int], list[float], list[float]]


@dataclass
class HarvesterState:
    """Serializable state of a Harvester for parallel merging.

    Reservoir data is stored as dense tensors rather than Python lists to avoid
    massive Python object overhead (~170 GB as lists vs ~26 GB as tensors for
    40K components × 1K examples × 41 window).
    """

    layer_names: list[str]
    c_per_layer: dict[str, int]
    vocab_size: int
    ci_threshold: float
    max_examples_per_component: int
    context_tokens_per_side: int

    # Tensor accumulators
    firing_counts: Tensor
    ci_sums: Tensor
    count_ij: Tensor
    input_token_counts: Tensor
    input_token_totals: Tensor
    output_token_prob_mass: Tensor
    output_token_prob_totals: Tensor
    total_tokens_processed: int

    # Reservoir data as tensors: [n_components, k, window_size]
    reservoir_tokens: Int[Tensor, "n_comp k window"]
    reservoir_ci: Float[Tensor, "n_comp k window"]
    reservoir_acts: Float[Tensor, "n_comp k window"]
    reservoir_n_items: Int[Tensor, " n_comp"]  # actual items per component (0..k)
    reservoir_n_seen: Int[Tensor, " n_comp"]  # total items seen (for merge weighting)

    def merge_into(self, other: "HarvesterState") -> None:
        """Merge another HarvesterState into this one (in-place).

        Tensor stats: simple +=.
        Reservoirs: Efraimidis-Spirakis weighted merge, vectorized over components.
        """
        assert other.layer_names == self.layer_names
        assert other.c_per_layer == self.c_per_layer

        self.firing_counts += other.firing_counts
        self.ci_sums += other.ci_sums
        self.count_ij += other.count_ij
        self.input_token_counts += other.input_token_counts
        self.input_token_totals += other.input_token_totals
        self.output_token_prob_mass += other.output_token_prob_mass
        self.output_token_prob_totals += other.output_token_prob_totals
        self.total_tokens_processed += other.total_tokens_processed

        _merge_reservoirs_inplace(self, other)


def _merge_reservoirs_inplace(dst: HarvesterState, src: HarvesterState) -> None:
    """Merge src reservoir tensors into dst, vectorized over all components.

    Uses Efraimidis-Spirakis: key = random()^(1/weight), take top-k.
    Avoids concatenating the big [n_comp, k, window] tensors — computes
    selection indices on small [n_comp, 2k] tensors, then gathers from
    dst/src separately based on whether each selected index came from dst or src.
    """
    k = dst.max_examples_per_component
    device = dst.reservoir_tokens.device
    n_comp = dst.reservoir_n_items.shape[0]

    # Validity mask over virtual [n_comp, 2k] index space: [0..k) = dst, [k..2k) = src
    idx = torch.arange(k, device=device).unsqueeze(0)
    valid_dst = idx < dst.reservoir_n_items.unsqueeze(1)
    valid_src = idx < src.reservoir_n_items.unsqueeze(1)
    valid = torch.cat([valid_dst, valid_src], dim=1)  # [n_comp, 2k]

    # Efraimidis-Spirakis weighted sampling keys
    weights = torch.zeros(n_comp, 2 * k, device=device)
    weights[:, :k] = dst.reservoir_n_seen.unsqueeze(1).float()
    weights[:, k:] = src.reservoir_n_seen.unsqueeze(1).float()
    weights[~valid] = 0.0

    rand = torch.rand(n_comp, 2 * k, device=device).clamp(min=1e-30)
    keys = rand.pow(1.0 / weights.clamp(min=1.0))
    keys[~valid] = -1.0

    _, top_indices = keys.topk(k, dim=1)  # [n_comp, k] — indices into [0..2k)

    # Split selected indices into dst-sourced and src-sourced
    from_dst = top_indices < k
    dst_indices = top_indices.clamp(max=k - 1)  # indices into dst's k dim
    src_indices = (top_indices - k).clamp(min=0)  # indices into src's k dim

    window = dst.reservoir_tokens.shape[2]
    dst_idx_exp = dst_indices.unsqueeze(-1).expand(-1, -1, window)
    src_idx_exp = src_indices.unsqueeze(-1).expand(-1, -1, window)
    from_dst_exp = from_dst.unsqueeze(-1).expand(-1, -1, window)

    dst.reservoir_tokens = torch.where(
        from_dst_exp,
        dst.reservoir_tokens.gather(1, dst_idx_exp),
        src.reservoir_tokens.gather(1, src_idx_exp),
    )
    dst.reservoir_ci = torch.where(
        from_dst_exp,
        dst.reservoir_ci.gather(1, dst_idx_exp),
        src.reservoir_ci.gather(1, src_idx_exp),
    )
    dst.reservoir_acts = torch.where(
        from_dst_exp,
        dst.reservoir_acts.gather(1, dst_idx_exp),
        src.reservoir_acts.gather(1, src_idx_exp),
    )

    total_valid = valid.sum(dim=1)
    dst.reservoir_n_items = total_valid.clamp(max=k)
    dst.reservoir_n_seen = dst.reservoir_n_seen + src.reservoir_n_seen


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

        # Reservoir samplers for activation examples (Python-side during harvesting)
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
        """Accumulate stats from a single batch."""
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
        self.count_ij += einsum(firing_flat, firing_flat, "pos c1, pos c2 -> c1 c2")

    def _accumulate_input_token_stats(
        self,
        batch_flat: Int[Tensor, " pos"],
        firing_flat: Float[Tensor, "pos n_comp"],
    ) -> None:
        n_components = firing_flat.shape[1]
        token_indices = batch_flat.unsqueeze(0).expand(n_components, -1)
        self.input_token_counts.scatter_add_(
            dim=1, index=token_indices, src=rearrange(firing_flat, "pos c -> c pos").long()
        )
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
        self.output_token_prob_mass += einsum(firing_flat, output_probs_flat, "pos c, pos v -> c v")
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

        MAX_FIRINGS_PER_COMPONENT = 5
        keep_mask = sample_at_most_n_per_group(component_idx, MAX_FIRINGS_PER_COMPONENT)
        batch_idx = batch_idx[keep_mask]
        seq_idx = seq_idx[keep_mask]
        component_idx = component_idx[keep_mask]

        S = batch.shape[1]
        window_size = 2 * self.context_tokens_per_side + 1
        offsets = torch.arange(
            -self.context_tokens_per_side, self.context_tokens_per_side + 1, device=self.device
        )
        window_seq_indices = seq_idx.unsqueeze(1) + offsets  # [n_firings, window_size]
        valid_mask = (window_seq_indices >= 0) & (window_seq_indices < S)
        clamped_indices = window_seq_indices.clamp(0, S - 1)

        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
        component_idx_expanded = component_idx.unsqueeze(1).expand(-1, window_size)

        token_windows = batch[batch_idx_expanded, clamped_indices]
        token_windows[~valid_mask] = WINDOW_PAD_SENTINEL

        ci_windows = ci[batch_idx_expanded, clamped_indices, component_idx_expanded]
        ci_windows[~valid_mask] = 0.0

        component_act_windows = subcomp_acts[
            batch_idx_expanded, clamped_indices, component_idx_expanded
        ]
        component_act_windows[~valid_mask] = 0.0

        for comp_idx, tokens, ci_vals, component_acts in zip(
            cast(list[int], component_idx.cpu().tolist()),
            cast(list[list[int]], token_windows.cpu().tolist()),
            cast(list[list[float]], ci_windows.cpu().tolist()),
            cast(list[list[float]], component_act_windows.cpu().tolist()),
            strict=True,
        ):
            self.activation_example_samplers[comp_idx].add((tokens, ci_vals, component_acts))

    def get_state(self) -> HarvesterState:
        """Extract serializable state for parallel merging.

        Packs reservoir sampler contents into dense tensors to avoid Python object
        overhead during serialization and merge (~26 GB as tensors vs ~170 GB as
        Python lists for 40K components).
        """
        n_components = sum(self.c_per_layer[layer] for layer in self.layer_names)
        k = self.max_examples_per_component
        window_size = 2 * self.context_tokens_per_side + 1

        reservoir_tokens = torch.full(
            (n_components, k, window_size), WINDOW_PAD_SENTINEL, dtype=torch.long
        )
        reservoir_ci = torch.zeros(n_components, k, window_size)
        reservoir_acts = torch.zeros(n_components, k, window_size)
        reservoir_n_items = torch.zeros(n_components, dtype=torch.long)
        reservoir_n_seen = torch.zeros(n_components, dtype=torch.long)

        for i, sampler in enumerate(self.activation_example_samplers):
            n = len(sampler.samples)
            reservoir_n_items[i] = n
            reservoir_n_seen[i] = sampler.n_seen
            for j, (tokens, ci_vals, acts) in enumerate(sampler.samples):
                reservoir_tokens[i, j] = torch.tensor(tokens, dtype=torch.long)
                reservoir_ci[i, j] = torch.tensor(ci_vals)
                reservoir_acts[i, j] = torch.tensor(acts)

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
            reservoir_tokens=reservoir_tokens,
            reservoir_ci=reservoir_ci,
            reservoir_acts=reservoir_acts,
            reservoir_n_items=reservoir_n_items,
            reservoir_n_seen=reservoir_n_seen,
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

        # Unpack tensor reservoirs back into Python samplers for build_results
        n_components = state.reservoir_n_items.shape[0]
        reservoir_tokens = state.reservoir_tokens.cpu()
        reservoir_ci = state.reservoir_ci.cpu()
        reservoir_acts = state.reservoir_acts.cpu()
        reservoir_n_items = state.reservoir_n_items.cpu()
        reservoir_n_seen = state.reservoir_n_seen.cpu()

        samplers: list[ReservoirSampler[ActivationExampleTuple]] = []
        for i in range(n_components):
            sampler: ReservoirSampler[ActivationExampleTuple] = ReservoirSampler(
                k=state.max_examples_per_component
            )
            sampler.n_seen = int(reservoir_n_seen[i])
            n = int(reservoir_n_items[i])
            sampler.samples = [
                (
                    reservoir_tokens[i, j].tolist(),
                    reservoir_ci[i, j].tolist(),
                    reservoir_acts[i, j].tolist(),
                )
                for j in range(n)
            ]
            samplers.append(sampler)
        harvester.activation_example_samplers = samplers

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

                sampler = self.activation_example_samplers[flat_idx]
                activation_examples = [
                    ActivationExample(
                        token_ids=[t for t in token_ids if t != WINDOW_PAD_SENTINEL],
                        ci_values=[
                            c
                            for t, c in zip(token_ids, ci_values, strict=True)
                            if t != WINDOW_PAD_SENTINEL
                        ],
                        component_acts=[
                            a
                            for t, a in zip(token_ids, component_acts, strict=True)
                            if t != WINDOW_PAD_SENTINEL
                        ],
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
    top, bottom = top_k_pmi(
        cooccurrence_counts=token_mass_for_component,
        marginal_counts=token_mass_totals,
        target_count=component_firing_count,
        total_count=total_tokens,
        top_k=top_k,
    )
    return ComponentTokenPMI(top=top, bottom=bottom)
