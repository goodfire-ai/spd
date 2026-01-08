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
        """Merge multiple reservoir states via uniform random sampling."""
        assert len(states) > 0
        k = states[0].k
        assert all(s.k == k for s in states)
        assert len(set(s.n_seen for s in states)) == 1, "all states must have same n_seen"

        total_seen = sum(s.n_seen for s in states)
        if total_seen == 0:
            return ReservoirState(k=k, samples=[], n_seen=0)

        all_samples = [s for state in states for s in state.samples]
        merged_samples = random.sample(all_samples, k) if len(all_samples) > k else all_samples

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
