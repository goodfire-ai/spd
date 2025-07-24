import numpy as np
import pytest

from spd.clustering.compute_rank import compute_rank_of_sum as rank_of_sum  # noqa: E402


def random_orth(
    d: int,
    r: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a d x r column-orthonormal matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((d, r)), mode="reduced")
    return q[:, :r]


def brute_rank(m: np.ndarray, tol: float = 1.0e-12) -> int:
    """
    Ground-truth numerical rank via full SVD.

    # Parameters:
     - `m : np.ndarray`
        matrix
     - `tol : float`
        singular values ≤ `tol` are treated as zero

    # Returns:
     - `int` : numerical rank
    """
    return int(np.sum(np.linalg.svd(m, compute_uv=False) > tol))


@pytest.mark.parametrize("d,r1,r2", [(8, 0, 0), (8, 0, 5), (8, 5, 0), (32, 3, 3)])
def test_edge_zero_ranks(
    d: int,
    r1: int,
    r2: int,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Edge-cases with zero-rank summands (all-zero matrices).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    u1 = np.empty((d, 0))
    v1 = np.empty((d, 0))
    s1 = np.empty((0,))

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    p1 = np.zeros((d, d))
    p2 = u2 @ np.diag(s2) @ v2.T

    expected = brute_rank(p1 + p2)
    got = rank_of_sum(u1, s1, v1, u2, s2, v2)

    assert expected == got


@pytest.mark.parametrize(
    "d,r1,r2",
    [
        (32, 5, 4),
        (64, 3, 3),
        (50, 10, 1),
        (40, 6, 6),
        (128, 15, 15),
    ],
)
def test_random_cases(
    d: int,
    r1: int,
    r2: int,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Random low-rank matrices with independent subspaces.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    u1 = random_orth(d, r1, rng)
    v1 = random_orth(d, r1, rng)
    s1 = rng.random(r1) + 0.1

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    p1 = u1 @ np.diag(s1) @ v1.T
    p2 = u2 @ np.diag(s2) @ v2.T

    expected = brute_rank(p1 + p2)
    got = rank_of_sum(u1, s1, v1, u2, s2, v2)

    assert expected == got


@pytest.mark.parametrize("overlap_dim", [1, 2, 3])
def test_overlapping_subspaces(
    overlap_dim: int,
    d: int = 32,
    r_base: int = 6,
    rng: np.random.Generator | None = None,
) -> None:
    """
    P1 and P2 share `overlap_dim` identical left singular vectors,
    so rank(P1+P2) < rank(P1)+rank(P2).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # Build common orthonormal basis
    u_common = random_orth(d, overlap_dim, rng)

    # Unique parts
    u1_unique = random_orth(d, r_base - overlap_dim, rng)
    u2_unique = random_orth(d, r_base - overlap_dim, rng)

    # Orthonormalise w.r.t common part
    u1 = np.linalg.qr(np.concatenate((u_common, u1_unique), axis=1))[0]
    u2 = np.linalg.qr(np.concatenate((u_common, u2_unique), axis=1))[0]

    v1 = random_orth(d, r_base, rng)
    v2 = random_orth(d, r_base, rng)

    s1 = rng.random(r_base) + 0.1
    s2 = rng.random(r_base) + 0.1

    p1 = u1 @ np.diag(s1) @ v1.T
    p2 = u2 @ np.diag(s2) @ v2.T

    expected = brute_rank(p1 + p2)
    got = rank_of_sum(u1, s1, v1, u2, s2, v2)

    assert expected == got
    assert expected < (2 * r_base)  # sanity: reduced rank due to overlap


def test_small_singular_values(
    d: int = 20,
    r1: int = 4,
    r2: int = 4,
    tiny: float = 1.0e-14,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Verify tolerance handling: one matrix has almost-zero singular values.
    """
    if rng is None:
        rng = np.random.default_rng(999)

    u1 = random_orth(d, r1, rng)
    v1 = random_orth(d, r1, rng)
    s1 = np.full(r1, tiny)

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    p1 = u1 @ np.diag(s1) @ v1.T
    p2 = u2 @ np.diag(s2) @ v2.T

    expected = brute_rank(p1 + p2)
    got = rank_of_sum(u1, s1, v1, u2, s2, v2)

    assert expected == got


@pytest.mark.parametrize("seed", list(range(10)))
def test_many_random_seeds(
    seed: int,
    d: int = 48,
    r1: int = 5,
    r2: int = 7,
) -> None:
    """
    Monte-carlo sweep across seeds to catch hidden edge cases.
    """
    rng = np.random.default_rng(seed)
    u1 = random_orth(d, r1, rng)
    v1 = random_orth(d, r1, rng)
    s1 = rng.random(r1) + 0.1

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    expected = brute_rank(u1 @ np.diag(s1) @ v1.T + u2 @ np.diag(s2) @ v2.T)
    got = rank_of_sum(u1, s1, v1, u2, s2, v2)

    assert expected == got


def test_argument_order_symmetry(
    d: int = 40,
    r1: int = 6,
    r2: int = 4,
    rng: np.random.Generator | None = None,
) -> None:
    """
    rank(P1+P2) must be symmetric w.r.t argument order.
    """
    if rng is None:
        rng = np.random.default_rng(2025)

    u1 = random_orth(d, r1, rng)
    v1 = random_orth(d, r1, rng)
    s1 = rng.random(r1) + 0.1

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    rank12 = rank_of_sum(u1, s1, v1, u2, s2, v2)
    rank21 = rank_of_sum(u2, s2, v2, u1, s1, v1)

    assert rank12 == rank21


def test_large_dimension_performance(
    d: int = 256,
    r1: int = 20,
    r2: int = 25,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Sanity test on a moderately large `d`; ensures code runs in <~0.1 s.
    """
    if rng is None:
        rng = np.random.default_rng(7)

    u1 = random_orth(d, r1, rng)
    v1 = random_orth(d, r1, rng)
    s1 = rng.random(r1) + 0.1

    u2 = random_orth(d, r2, rng)
    v2 = random_orth(d, r2, rng)
    s2 = rng.random(r2) + 0.1

    got = rank_of_sum(u1, s1, v1, u2, s2, v2)
    assert got <= (r1 + r2)  # trivial upper bound
