from itertools import permutations

import numpy as np
import pytest

from spd.clustering.math.perm_invariant_hamming import perm_invariant_hamming

# pyright complains about the types when calling perm_invariant_hamming
# pyright: reportCallIssue=false


def brute_force_min_hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Exhaustive check for small k."""
    k = int(max(a.max(), b.max()) + 1)
    best = len(a)
    for perm in permutations(range(k)):
        mapping = np.array(perm)
        best = min(best, int((mapping[a] != b).sum()))
    return best


def test_identity() -> None:
    """a == b should give distance 0."""
    a = np.array([0, 1, 2, 1, 0])
    b = a.copy()
    d, _ = perm_invariant_hamming(a, b)
    assert d == 0


def test_all_one_group() -> None:
    """All rows belong to one group in both arrays (possibly different labels)."""
    a = np.zeros(10, dtype=int)
    b = np.ones(10, dtype=int)  # different label but identical grouping
    d, _ = perm_invariant_hamming(a, b)
    assert d == 0


def test_permuted_labels() -> None:
    a = np.array([0, 2, 1, 1, 0])
    b = np.array([1, 0, 0, 2, 1])
    d, _ = perm_invariant_hamming(a, b)
    assert d == 1


def test_swap_two_labels() -> None:
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    d, _ = perm_invariant_hamming(a, b)
    assert d == 0


def test_random_small_bruteforce() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 7
        k = 3
        a = rng.integers(0, k, size=n)
        b = rng.integers(0, k, size=n)
        d_alg, _ = perm_invariant_hamming(a, b)
        d_true = brute_force_min_hamming(a, b)
        assert d_alg == d_true


def test_shape_mismatch() -> None:
    a = np.array([0, 1, 2])
    b = np.array([0, 1])
    with pytest.raises(AssertionError):
        perm_invariant_hamming(a, b)


def test_return_mapping() -> None:
    """Verify the returned mapping is correct."""
    a = np.array([0, 0, 1, 1])
    b = np.array([2, 2, 3, 3])
    d, mapping = perm_invariant_hamming(a, b, return_mapping=True)
    assert d == 0
    assert mapping[0] == 2
    assert mapping[1] == 3


def test_return_mapping_false() -> None:
    """Test return_mapping=False."""
    a = np.array([0, 1, 0])
    b = np.array([1, 0, 1])
    d, mapping = perm_invariant_hamming(a, b, return_mapping=False)
    assert d == 0
    assert mapping is None


def test_unused_labels() -> None:
    """Test when arrays don't use all labels 0..k-1."""
    a = np.array([0, 0, 3, 3])  # skips 1, 2
    b = np.array([1, 1, 2, 2])
    d, _ = perm_invariant_hamming(a, b)
    assert d == 0
