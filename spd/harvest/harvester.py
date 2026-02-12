"""Harvester for collecting component statistics in a single pass.

All accumulator state lives as tensors on `device` (GPU during harvesting, CPU during merge).
Reservoir sampling uses Algorithm R directly on tensor storage — no Python list intermediary.
"""

import random
from collections.abc import Iterator
from pathlib import Path

import torch
import tqdm
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor

from spd.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI
from spd.log import logger

WINDOW_PAD_SENTINEL = -1


class Harvester:
    """Accumulates component statistics in a single pass over data.

    All mutable state is stored as tensors on `device`. Workers on GPU accumulate
    into GPU tensors; the merge job reconstructs on CPU. No Python-object reservoirs.
    """

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

        self.layer_offsets: dict[str, int] = {}
        offset = 0
        for layer in layer_names:
            self.layer_offsets[layer] = offset
            offset += c_per_layer[layer]

        n = sum(c_per_layer[layer] for layer in layer_names)
        w = 2 * context_tokens_per_side + 1
        k = max_examples_per_component

        # Correlation accumulators
        self.firing_counts = torch.zeros(n, device=device)
        self.ci_sums = torch.zeros(n, device=device)
        self.count_ij = torch.zeros(n, n, device=device, dtype=torch.float32)

        # Token stat accumulators
        self.input_token_counts: Int[Tensor, "n vocab"] = torch.zeros(
            n, vocab_size, device=device, dtype=torch.long
        )
        self.input_token_totals: Int[Tensor, " vocab"] = torch.zeros(
            vocab_size, device=device, dtype=torch.long
        )
        self.output_token_prob_mass: Float[Tensor, "n vocab"] = torch.zeros(
            n, vocab_size, device=device
        )
        self.output_token_prob_totals: Float[Tensor, " vocab"] = torch.zeros(
            vocab_size, device=device
        )

        # Activation example reservoir (tensor-backed, Algorithm R)
        self.reservoir_tokens: Int[Tensor, "n k w"] = torch.full(
            (n, k, w), WINDOW_PAD_SENTINEL, dtype=torch.long, device=device
        )
        self.reservoir_ci: Float[Tensor, "n k w"] = torch.zeros(n, k, w, device=device)
        self.reservoir_acts: Float[Tensor, "n k w"] = torch.zeros(n, k, w, device=device)
        self.reservoir_n_items: Int[Tensor, " n"] = torch.zeros(n, dtype=torch.long, device=device)
        self.reservoir_n_seen: Int[Tensor, " n"] = torch.zeros(n, dtype=torch.long, device=device)

        self.total_tokens_processed = 0

    # -- Batch processing --------------------------------------------------

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
        w = 2 * self.context_tokens_per_side + 1
        offsets = torch.arange(
            -self.context_tokens_per_side, self.context_tokens_per_side + 1, device=self.device
        )
        window_positions = seq_idx.unsqueeze(1) + offsets  # [N, w]
        valid = (window_positions >= 0) & (window_positions < S)
        clamped = window_positions.clamp(0, S - 1)

        bi = batch_idx.unsqueeze(1).expand(-1, w)
        ci_idx = component_idx.unsqueeze(1).expand(-1, w)

        token_windows = batch[bi, clamped]
        token_windows[~valid] = WINDOW_PAD_SENTINEL

        ci_windows = ci[bi, clamped, ci_idx]
        ci_windows[~valid] = 0.0

        act_windows = subcomp_acts[bi, clamped, ci_idx]
        act_windows[~valid] = 0.0

        self._reservoir_add(component_idx, token_windows, ci_windows, act_windows)

    def _reservoir_add(
        self,
        comp_idx: Int[Tensor, " N"],
        token_windows: Int[Tensor, "N W"],
        ci_windows: Float[Tensor, "N W"],
        act_windows: Float[Tensor, "N W"],
    ) -> None:
        """Add firing windows to the reservoir via Algorithm R.

        Bookkeeping on CPU (cheap integer ops), then batch-write to device.
        """
        k = self.max_examples_per_component
        device = comp_idx.device
        comps = comp_idx.cpu().tolist()
        items_cpu = self.reservoir_n_items.cpu()
        seen_cpu = self.reservoir_n_seen.cpu()

        write_comps: list[int] = []
        write_slots: list[int] = []
        write_srcs: list[int] = []

        for i, c in enumerate(comps):
            n = int(seen_cpu[c])
            if items_cpu[c] < k:
                write_comps.append(c)
                write_slots.append(int(items_cpu[c]))
                write_srcs.append(i)
                items_cpu[c] += 1
            else:
                j = random.randint(0, n)
                if j < k:
                    write_comps.append(c)
                    write_slots.append(j)
                    write_srcs.append(i)
            seen_cpu[c] += 1

        self.reservoir_n_items.copy_(items_cpu)
        self.reservoir_n_seen.copy_(seen_cpu)

        if write_comps:
            c_t = torch.tensor(write_comps, dtype=torch.long, device=device)
            s_t = torch.tensor(write_slots, dtype=torch.long, device=device)
            f_t = torch.tensor(write_srcs, dtype=torch.long, device=device)
            self.reservoir_tokens[c_t, s_t] = token_windows[f_t]
            self.reservoir_ci[c_t, s_t] = ci_windows[f_t]
            self.reservoir_acts[c_t, s_t] = act_windows[f_t]

    # -- Serialization & merge ---------------------------------------------

    _TENSOR_FIELDS = [
        "firing_counts", "ci_sums", "count_ij",
        "input_token_counts", "input_token_totals",
        "output_token_prob_mass", "output_token_prob_totals",
        "reservoir_tokens", "reservoir_ci", "reservoir_acts",
        "reservoir_n_items", "reservoir_n_seen",
    ]  # fmt: skip

    def save(self, path: Path) -> None:
        """Serialize all state to disk (tensors moved to CPU)."""
        data: dict[str, object] = {
            "layer_names": self.layer_names,
            "c_per_layer": self.c_per_layer,
            "vocab_size": self.vocab_size,
            "ci_threshold": self.ci_threshold,
            "max_examples_per_component": self.max_examples_per_component,
            "context_tokens_per_side": self.context_tokens_per_side,
            "total_tokens_processed": self.total_tokens_processed,
        }
        for f in self._TENSOR_FIELDS:
            data[f] = getattr(self, f).cpu()
        torch.save(data, path)

    _CPU = torch.device("cpu")

    @staticmethod
    def load(path: Path, device: torch.device = _CPU) -> "Harvester":
        """Load from disk."""
        from typing import Any

        d: dict[str, Any] = torch.load(path, weights_only=False)
        h = Harvester.__new__(Harvester)
        h.layer_names = d["layer_names"]
        h.c_per_layer = d["c_per_layer"]
        h.vocab_size = d["vocab_size"]
        h.ci_threshold = d["ci_threshold"]
        h.max_examples_per_component = d["max_examples_per_component"]
        h.context_tokens_per_side = d["context_tokens_per_side"]
        h.total_tokens_processed = d["total_tokens_processed"]
        h.device = device

        h.layer_offsets = {}
        offset = 0
        for layer in h.layer_names:
            h.layer_offsets[layer] = offset
            offset += h.c_per_layer[layer]

        for f in Harvester._TENSOR_FIELDS:
            setattr(h, f, d[f].to(device))

        return h

    def merge(self, other: "Harvester") -> None:
        """Merge another Harvester into this one (in-place)."""
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

        self._merge_reservoirs(other)

    def _merge_reservoirs(self, other: "Harvester") -> None:
        """Merge other's reservoir into self, vectorized over all components.

        Uses Efraimidis-Spirakis: key = random()^(1/weight), take top-k.
        Computes selection indices on small [n_comp, 2k] tensors, then gathers
        from self/other based on whether each selected index came from self or other.
        """
        k = self.max_examples_per_component
        device = self.reservoir_tokens.device
        n_comp = self.reservoir_n_items.shape[0]

        idx = torch.arange(k, device=device).unsqueeze(0)
        valid_self = idx < self.reservoir_n_items.unsqueeze(1)
        valid_other = idx < other.reservoir_n_items.unsqueeze(1)
        valid = torch.cat([valid_self, valid_other], dim=1)  # [n_comp, 2k]

        weights = torch.zeros(n_comp, 2 * k, device=device)
        weights[:, :k] = self.reservoir_n_seen.unsqueeze(1).float()
        weights[:, k:] = other.reservoir_n_seen.unsqueeze(1).float()
        weights[~valid] = 0.0

        rand = torch.rand(n_comp, 2 * k, device=device).clamp(min=1e-30)
        keys = rand.pow(1.0 / weights.clamp(min=1.0))
        keys[~valid] = -1.0

        _, top_indices = keys.topk(k, dim=1)  # [n_comp, k]

        from_self = top_indices < k
        self_indices = top_indices.clamp(max=k - 1)
        other_indices = (top_indices - k).clamp(min=0)

        window = self.reservoir_tokens.shape[2]
        si = self_indices.unsqueeze(-1).expand(-1, -1, window)
        oi = other_indices.unsqueeze(-1).expand(-1, -1, window)
        mask = from_self.unsqueeze(-1).expand(-1, -1, window)

        self.reservoir_tokens = torch.where(
            mask, self.reservoir_tokens.gather(1, si), other.reservoir_tokens.gather(1, oi)
        )
        self.reservoir_ci = torch.where(
            mask, self.reservoir_ci.gather(1, si), other.reservoir_ci.gather(1, oi)
        )
        self.reservoir_acts = torch.where(
            mask, self.reservoir_acts.gather(1, si), other.reservoir_acts.gather(1, oi)
        )

        self.reservoir_n_items = valid.sum(dim=1).clamp(max=k)
        self.reservoir_n_seen = self.reservoir_n_seen + other.reservoir_n_seen

    # -- Result building ---------------------------------------------------

    def build_results(self, pmi_top_k_tokens: int) -> Iterator[ComponentData]:
        """Yield ComponentData objects one at a time (constant memory)."""
        logger.info("  Moving tensors to CPU...")
        mean_ci = (self.ci_sums / self.total_tokens_processed).cpu()
        firing_counts = self.firing_counts.cpu()
        input_token_counts = self.input_token_counts.cpu()
        input_token_totals = self.input_token_totals.cpu()
        output_token_prob_mass = self.output_token_prob_mass.cpu()
        output_token_prob_totals = self.output_token_prob_totals.cpu()

        res_tokens = self.reservoir_tokens.cpu()
        res_ci = self.reservoir_ci.cpu()
        res_acts = self.reservoir_acts.cpu()
        res_n_items = self.reservoir_n_items.cpu()

        _log_base_rate_summary(firing_counts, input_token_totals)

        n_total = sum(self.c_per_layer[layer] for layer in self.layer_names)
        logger.info(
            f"  Computing stats for {n_total} components across {len(self.layer_names)} layers..."
        )
        for layer_name in tqdm.tqdm(self.layer_names, desc="Building components"):
            layer_offset = self.layer_offsets[layer_name]
            layer_c = self.c_per_layer[layer_name]

            for cidx in range(layer_c):
                flat = layer_offset + cidx

                n_firings = float(firing_counts[flat])
                if n_firings == 0:
                    continue

                n = int(res_n_items[flat])
                examples = []
                for j in range(n):
                    toks = res_tokens[flat, j]
                    mask = toks != WINDOW_PAD_SENTINEL
                    examples.append(
                        ActivationExample(
                            token_ids=toks[mask].tolist(),
                            ci_values=res_ci[flat, j][mask].tolist(),
                            component_acts=res_acts[flat, j][mask].tolist(),
                        )
                    )

                yield ComponentData(
                    component_key=f"{layer_name}:{cidx}",
                    layer=layer_name,
                    component_idx=cidx,
                    mean_ci=float(mean_ci[flat]),
                    activation_examples=examples,
                    input_token_pmi=_compute_token_pmi(
                        input_token_counts[flat],
                        input_token_totals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    ),
                    output_token_pmi=_compute_token_pmi(
                        output_token_prob_mass[flat],
                        output_token_prob_totals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    ),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_base_rate_summary(firing_counts: Tensor, input_token_totals: Tensor) -> None:
    active_counts = firing_counts[firing_counts > 0]
    if len(active_counts) == 0:
        logger.info("  WARNING: No components fired above threshold!")
        return

    sorted_counts = active_counts.sort().values
    n_active = len(active_counts)
    logger.info("\n  === Base Rate Summary ===")
    logger.info(f"  Components with firings: {n_active} / {len(firing_counts)}")
    logger.info(
        f"  Firing counts - min: {int(sorted_counts[0])}, "
        f"median: {int(sorted_counts[n_active // 2])}, "
        f"max: {int(sorted_counts[-1])}"
    )

    LOW_FIRING_THRESHOLD = 100
    n_sparse = int((active_counts < LOW_FIRING_THRESHOLD).sum())
    if n_sparse > 0:
        logger.info(
            f"  WARNING: {n_sparse} components have <{LOW_FIRING_THRESHOLD} firings "
            f"(stats may be noisy)"
        )

    active_tokens = input_token_totals[input_token_totals > 0]
    sorted_token_counts = active_tokens.sort().values
    n_tokens = len(active_tokens)
    logger.info(
        f"  Tokens seen: {n_tokens} unique, "
        f"occurrences - min: {int(sorted_token_counts[0])}, "
        f"median: {int(sorted_token_counts[n_tokens // 2])}, "
        f"max: {int(sorted_token_counts[-1])}"
    )

    RARE_TOKEN_THRESHOLD = 10
    n_rare = int((active_tokens < RARE_TOKEN_THRESHOLD).sum())
    if n_rare > 0:
        logger.info(
            f"  Note: {n_rare} tokens have <{RARE_TOKEN_THRESHOLD} occurrences "
            f"(high precision/recall with these may be spurious)"
        )
    logger.info("")


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
