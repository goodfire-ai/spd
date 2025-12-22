import random
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ReservoirState(Generic[T]):  # noqa: UP046 - PEP 695 syntax breaks pickling
    """Serializable state of a ReservoirSampler."""

    k: int
    samples: list[T]
    n_seen: int

    @staticmethod
    def merge(states: list["ReservoirState[T]"]) -> "ReservoirState[T]":
        """Merge multiple reservoir states via weighted random sampling.

        Each sample from reservoir i has probability n_i / sum(n_j) of being selected.
        """
        assert len(states) > 0
        k = states[0].k
        assert all(s.k == k for s in states)

        total_seen = sum(s.n_seen for s in states)
        if total_seen == 0:
            return ReservoirState(k=k, samples=[], n_seen=0)

        # Build weighted pool: (sample, weight) where weight = n_seen for that reservoir
        weighted_samples: list[tuple[T, int]] = []
        for state in states:
            for sample in state.samples:
                weighted_samples.append((sample, state.n_seen))

        if len(weighted_samples) <= k:
            merged_samples = [s for s, _ in weighted_samples]
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

            merged_samples = [weighted_samples[i][0] for i in indices]

        return ReservoirState(k=k, samples=merged_samples, n_seen=total_seen)


class ReservoirSampler(Generic[T]):  # noqa: UP046 - PEP 695 syntax breaks pickling
    """Uniform random sampling from a stream via reservoir sampling."""

    def __init__(self, k: int):
        self.k = k
        self.samples: list[T] = []
        self.n_seen = 0

    def add(self, item: T) -> None:
        self.n_seen += 1
        if len(self.samples) < self.k:
            self.samples.append(item)
        elif random.randint(1, self.n_seen) <= self.k:
            self.samples[random.randrange(self.k)] = item

    def get_state(self) -> ReservoirState[T]:
        return ReservoirState(k=self.k, samples=list(self.samples), n_seen=self.n_seen)

    @staticmethod
    def from_state(state: ReservoirState[T]) -> "ReservoirSampler[T]":
        sampler: ReservoirSampler[T] = ReservoirSampler(k=state.k)
        sampler.samples = list(state.samples)
        sampler.n_seen = state.n_seen
        return sampler
