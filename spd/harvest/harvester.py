"""Harvester for collecting component statistics in a single pass.

All accumulator state lives as tensors on `device` (GPU during harvesting, CPU during merge).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import tqdm
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor

from spd.harvest.reservoir import ActivationExamplesReservoir
from spd.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI
from spd.log import logger


class Harvester:
    """Accumulates component statistics in a single pass over data.

    All mutable state is stored as tensors on `device`. Workers on GPU accumulate
    into GPU tensors; the merge job reconstructs on CPU.
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

        self.reservoir = ActivationExamplesReservoir(n, max_examples_per_component, w, device)
        self.total_tokens_processed = 0

    # -- Batch processing --------------------------------------------------

    def process_batch(
        self,
        batch: Int[Tensor, "B S"],
        ci: Float[Tensor, "B S n_comp"],
        output_probs: Float[Tensor, "B S V"],
        subcomp_acts: Float[Tensor, "B S n_comp"],
    ) -> None:
        self.total_tokens_processed += batch.numel()

        firing = (ci > self.ci_threshold).float()
        firing_flat = rearrange(firing, "b s c -> (b s) c")
        batch_flat = rearrange(batch, "b s -> (b s)")
        output_probs_flat = rearrange(output_probs, "b s v -> (b s) v")

        self.firing_counts += reduce(firing, "b s c -> c", "sum")
        self.ci_sums += reduce(ci, "b s c -> c", "sum")
        self.count_ij += einsum(firing_flat, firing_flat, "pos c1, pos c2 -> c1 c2")
        self._accumulate_token_stats(batch_flat, output_probs_flat, firing_flat)
        self._collect_activation_examples(batch, ci, subcomp_acts)

    def _accumulate_token_stats(
        self,
        batch_flat: Int[Tensor, " pos"],
        output_probs_flat: Float[Tensor, "pos vocab"],
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
        self.output_token_prob_mass += einsum(firing_flat, output_probs_flat, "pos c, pos v -> c v")
        self.output_token_prob_totals += reduce(output_probs_flat, "pos v -> v", "sum")

    def _collect_activation_examples(
        self,
        batch: Int[Tensor, "B S"],
        ci: Float[Tensor, "B S n_comp"],
        subcomp_acts: Float[Tensor, "B S n_comp"],
    ) -> None:
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
        window_positions = seq_idx.unsqueeze(1) + offsets
        valid = (window_positions >= 0) & (window_positions < S)
        clamped = window_positions.clamp(0, S - 1)

        bi = batch_idx.unsqueeze(1).expand(-1, w)
        ci_idx = component_idx.unsqueeze(1).expand(-1, w)

        from spd.harvest.reservoir import WINDOW_PAD_SENTINEL

        token_windows = batch[bi, clamped]
        token_windows[~valid] = WINDOW_PAD_SENTINEL

        ci_windows = ci[bi, clamped, ci_idx]
        ci_windows[~valid] = 0.0

        act_windows = subcomp_acts[bi, clamped, ci_idx]
        act_windows[~valid] = 0.0

        self.reservoir.add(component_idx, token_windows, ci_windows, act_windows)

    # -- Serialization & merge ---------------------------------------------

    _STAT_FIELDS = [
        "firing_counts", "ci_sums", "count_ij",
        "input_token_counts", "input_token_totals",
        "output_token_prob_mass", "output_token_prob_totals",
    ]  # fmt: skip

    def save(self, path: Path) -> None:
        data: dict[str, object] = {
            "layer_names": self.layer_names,
            "c_per_layer": self.c_per_layer,
            "vocab_size": self.vocab_size,
            "ci_threshold": self.ci_threshold,
            "max_examples_per_component": self.max_examples_per_component,
            "context_tokens_per_side": self.context_tokens_per_side,
            "total_tokens_processed": self.total_tokens_processed,
            "reservoir": self.reservoir.state_dict(),
        }
        for f in self._STAT_FIELDS:
            data[f] = getattr(self, f).cpu()
        torch.save(data, path)

    _CPU = torch.device("cpu")

    @staticmethod
    def load(path: Path, device: torch.device = _CPU) -> "Harvester":
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

        for f in Harvester._STAT_FIELDS:
            setattr(h, f, d[f].to(device))

        h.reservoir = ActivationExamplesReservoir.from_state_dict(d["reservoir"], device)
        return h

    def merge(self, other: "Harvester") -> None:
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

        self.reservoir.merge(other.reservoir)

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

        reservoir_cpu = ActivationExamplesReservoir.from_state_dict(
            self.reservoir.state_dict(), torch.device("cpu")
        )

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

                examples = [
                    ActivationExample(
                        token_ids=toks.tolist(),
                        ci_values=ci_vals.tolist(),
                        component_acts=acts.tolist(),
                    )
                    for toks, ci_vals, acts in reservoir_cpu.examples(flat)
                ]

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
