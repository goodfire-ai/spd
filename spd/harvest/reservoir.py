"""Activation examples reservoir backed by dense tensors.

Stores [n_components, k, window] activation example windows using Algorithm R
for sampling and Efraimidis-Spirakis for merging parallel reservoirs.
"""

import random
from collections.abc import Iterator
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor

WINDOW_PAD_SENTINEL = -1


class ActivationExamplesReservoir:
    """Fixed-capacity reservoir of activation example windows per component.

    Each component slot holds up to `k` windows of size `w`, where each window
    contains (token_ids, ci_values, component_acts) aligned by position.
    """

    def __init__(self, n_components: int, k: int, window: int, device: torch.device):
        self.k = k
        self.window = window
        self.tokens: Int[Tensor, "C k w"] = torch.full(
            (n_components, k, window), WINDOW_PAD_SENTINEL, dtype=torch.long, device=device
        )
        self.ci: Float[Tensor, "C k w"] = torch.zeros(n_components, k, window, device=device)
        self.acts: Float[Tensor, "C k w"] = torch.zeros(n_components, k, window, device=device)
        self.n_items: Int[Tensor, " C"] = torch.zeros(n_components, dtype=torch.long, device=device)
        self.n_seen: Int[Tensor, " C"] = torch.zeros(n_components, dtype=torch.long, device=device)

    @property
    def n_components(self) -> int:
        return self.n_items.shape[0]

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    def add(
        self,
        comp_idx: Int[Tensor, " N"],
        token_windows: Int[Tensor, "N W"],
        ci_windows: Float[Tensor, "N W"],
        act_windows: Float[Tensor, "N W"],
    ) -> None:
        """Add firing windows via Algorithm R.

        Bookkeeping on CPU (cheap integer ops), then batch-write to device.
        """
        device = comp_idx.device
        comps = comp_idx.cpu().tolist()
        items_cpu = self.n_items.cpu()
        seen_cpu = self.n_seen.cpu()

        write_comps: list[int] = []
        write_slots: list[int] = []
        write_srcs: list[int] = []

        for i, c in enumerate(comps):
            n = int(seen_cpu[c])
            if items_cpu[c] < self.k:
                write_comps.append(c)
                write_slots.append(int(items_cpu[c]))
                write_srcs.append(i)
                items_cpu[c] += 1
            else:
                j = random.randint(0, n)
                if j < self.k:
                    write_comps.append(c)
                    write_slots.append(j)
                    write_srcs.append(i)
            seen_cpu[c] += 1

        self.n_items.copy_(items_cpu)
        self.n_seen.copy_(seen_cpu)

        if write_comps:
            c_t = torch.tensor(write_comps, dtype=torch.long, device=device)
            s_t = torch.tensor(write_slots, dtype=torch.long, device=device)
            f_t = torch.tensor(write_srcs, dtype=torch.long, device=device)
            self.tokens[c_t, s_t] = token_windows[f_t]
            self.ci[c_t, s_t] = ci_windows[f_t]
            self.acts[c_t, s_t] = act_windows[f_t]

    def merge(self, other: "ActivationExamplesReservoir") -> None:
        """Merge other's reservoir into self via Efraimidis-Spirakis.

        Computes selection indices on small [C, 2k] tensors, then gathers
        from self/other based on whether each selected index came from self or other.
        """
        assert other.n_components == self.n_components
        assert other.k == self.k
        device = self.device
        n_comp = self.n_components

        idx = torch.arange(self.k, device=device).unsqueeze(0)
        valid_self = idx < self.n_items.unsqueeze(1)
        valid_other = idx < other.n_items.unsqueeze(1)
        valid = torch.cat([valid_self, valid_other], dim=1)

        weights = torch.zeros(n_comp, 2 * self.k, device=device)
        weights[:, : self.k] = self.n_seen.unsqueeze(1).float()
        weights[:, self.k :] = other.n_seen.unsqueeze(1).float()
        weights[~valid] = 0.0

        rand = torch.rand(n_comp, 2 * self.k, device=device).clamp(min=1e-30)
        keys = rand.pow(1.0 / weights.clamp(min=1.0))
        keys[~valid] = -1.0

        _, top_indices = keys.topk(self.k, dim=1)

        from_self = top_indices < self.k
        self_indices = top_indices.clamp(max=self.k - 1)
        other_indices = (top_indices - self.k).clamp(min=0)

        si = self_indices.unsqueeze(-1).expand(-1, -1, self.window)
        oi = other_indices.unsqueeze(-1).expand(-1, -1, self.window)
        mask = from_self.unsqueeze(-1).expand(-1, -1, self.window)

        self.tokens = torch.where(mask, self.tokens.gather(1, si), other.tokens.gather(1, oi))
        self.ci = torch.where(mask, self.ci.gather(1, si), other.ci.gather(1, oi))
        self.acts = torch.where(mask, self.acts.gather(1, si), other.acts.gather(1, oi))

        self.n_items = valid.sum(dim=1).clamp(max=self.k)
        self.n_seen = self.n_seen + other.n_seen

    def examples(self, component: int) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Yield (token_ids, ci_values, component_acts) for each stored example, sentinel-filtered."""
        n = int(self.n_items[component])
        for j in range(n):
            toks = self.tokens[component, j]
            mask = toks != WINDOW_PAD_SENTINEL
            yield toks[mask], self.ci[component, j][mask], self.acts[component, j][mask]

    def state_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "window": self.window,
            "tokens": self.tokens.cpu(),
            "ci": self.ci.cpu(),
            "acts": self.acts.cpu(),
            "n_items": self.n_items.cpu(),
            "n_seen": self.n_seen.cpu(),
        }

    @staticmethod
    def from_state_dict(d: dict[str, Any], device: torch.device) -> "ActivationExamplesReservoir":
        r = ActivationExamplesReservoir.__new__(ActivationExamplesReservoir)
        r.k = d["k"]
        r.window = d["window"]
        r.tokens = d["tokens"].to(device)
        r.ci = d["ci"].to(device)
        r.acts = d["acts"].to(device)
        r.n_items = d["n_items"].to(device)
        r.n_seen = d["n_seen"].to(device)
        return r
