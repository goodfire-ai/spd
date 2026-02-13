"""Activation examples reservoir backed by dense tensors.

Stores [n_components, k, window] activation example windows using Algorithm R
for sampling and Efraimidis-Spirakis for merging parallel reservoirs.
"""

import random
from collections.abc import Iterator

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from spd.utils.general_utils import runtime_cast

WINDOW_PAD_SENTINEL = -1


class ActivationExamplesReservoir:
    """Fixed-capacity reservoir of activation example windows per component.

    Each component slot holds up to `k` windows of size `w`, where each window
    contains (token_ids, ci_values, component_acts) aligned by position.

    Use create() for fresh allocation, from_state_dict() for deserialization.
    """

    def __init__(
        self,
        n_components: int,
        k: int,
        window: int,
        device: torch.device,
        tokens: Int[Tensor, "C k w"],
        ci: Float[Tensor, "C k w"],
        acts: Float[Tensor, "C k w"],
        n_items: Int[Tensor, " C"],
        n_seen: Int[Tensor, " C"],
    ):
        self.n_components = n_components
        self.k = k
        self.window = window
        self.device = device
        self.tokens = tokens
        self.ci = ci
        self.acts = acts
        self.n_items = n_items
        self.n_seen = n_seen

    @classmethod
    def create(
        cls, n_components: int, k: int, window: int, device: torch.device
    ) -> "ActivationExamplesReservoir":
        return cls(
            n_components=n_components,
            k=k,
            window=window,
            device=device,
            tokens=torch.full(
                (n_components, k, window), WINDOW_PAD_SENTINEL, dtype=torch.long, device=device
            ),
            ci=torch.zeros(n_components, k, window, device=device),
            acts=torch.zeros(n_components, k, window, device=device),
            n_items=torch.zeros(n_components, dtype=torch.long, device=device),
            n_seen=torch.zeros(n_components, dtype=torch.long, device=device),
        )

    @classmethod
    def from_state_dict(
        cls, d: dict[str, object], device: torch.device
    ) -> "ActivationExamplesReservoir":
        tokens = runtime_cast(Tensor, d["tokens"])
        return cls(
            n_components=tokens.shape[0],
            k=runtime_cast(int, d["k"]),
            window=runtime_cast(int, d["window"]),
            device=device,
            tokens=tokens.to(device),
            ci=runtime_cast(Tensor, d["ci"]).to(device),
            acts=runtime_cast(Tensor, d["acts"]).to(device),
            n_items=runtime_cast(Tensor, d["n_items"]).to(device),
            n_seen=runtime_cast(Tensor, d["n_seen"]).to(device),
        )

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

        idx = rearrange(torch.arange(self.k, device=device), "k -> 1 k")
        valid_self = idx < rearrange(self.n_items, "c -> c 1")
        valid_other = idx < rearrange(other.n_items, "c -> c 1")
        valid = torch.cat([valid_self, valid_other], dim=1)

        weights = torch.zeros(n_comp, 2 * self.k, device=device)
        weights[:, : self.k] = rearrange(self.n_seen.float(), "c -> c 1")
        weights[:, self.k :] = rearrange(other.n_seen.float(), "c -> c 1")
        weights[~valid] = 0.0

        rand = torch.rand(n_comp, 2 * self.k, device=device).clamp(min=1e-30)
        keys = rand.pow(1.0 / weights.clamp(min=1.0))
        keys[~valid] = -1.0

        _, top_indices = keys.topk(self.k, dim=1)

        from_self = top_indices < self.k
        self_indices = top_indices.clamp(max=self.k - 1)
        other_indices = (top_indices - self.k).clamp(min=0)

        si = repeat(self_indices, "c k -> c k w", w=self.window)
        oi = repeat(other_indices, "c k -> c k w", w=self.window)
        mask = repeat(from_self, "c k -> c k w", w=self.window)

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

    def to(self, device: torch.device) -> "ActivationExamplesReservoir":
        return ActivationExamplesReservoir(
            n_components=self.n_components,
            k=self.k,
            window=self.window,
            device=device,
            tokens=self.tokens.to(device),
            ci=self.ci.to(device),
            acts=self.acts.to(device),
            n_items=self.n_items.to(device),
            n_seen=self.n_seen.to(device),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "window": self.window,
            "tokens": self.tokens.cpu(),
            "ci": self.ci.cpu(),
            "acts": self.acts.cpu(),
            "n_items": self.n_items.cpu(),
            "n_seen": self.n_seen.cpu(),
        }
