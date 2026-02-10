import heapq
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

        Uses Efraimidis-Spirakis algorithm: each sample gets key = random()^(1/weight),
        take k largest. O(n + k log n) vs O(k*n) for naive weighted sampling.
        """
        assert len(states) > 0
        k = states[0].k
        assert all(s.k == k for s in states)

        total_seen = sum(s.n_seen for s in states)
        if total_seen == 0:
            return ReservoirState(k=k, samples=[], n_seen=0)

        # Build weighted pool: each sample weighted by its reservoir's n_seen
        weighted_samples: list[tuple[T, int]] = []
        for state in states:
            for sample in state.samples:
                weighted_samples.append((sample, state.n_seen))

        if len(weighted_samples) <= k:
            merged_samples = [s for s, _ in weighted_samples]
        else:
            # Efraimidis-Spirakis: key = random()^(1/weight), take k largest
            keys_and_samples = [(random.random() ** (1.0 / w), s) for s, w in weighted_samples]
            top_k = heapq.nlargest(k, keys_and_samples, key=lambda x: x[0])
            merged_samples = [s for _, s in top_k]

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
