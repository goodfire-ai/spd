#%%
# Vectorize, then verify equivalence against the original on many randomized cases.
# Prints "ALL TESTS PASSED" on success and shows a small example.

from typing import Any
import time
import torch
from torch import Tensor

from jaxtyping import Bool, Float, Int


def matching_dist_orig(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    """original loop version"""
    s_ensemble, _n_components = X.shape
    matches: Bool[Tensor, "s n n"] = X[:, :, None] == X[:, None, :]
    dists: Float[Tensor, "s s"] = torch.full((s_ensemble, s_ensemble), torch.nan, device=X.device)
    for i in range(s_ensemble):
        for j in range(i + 1, s_ensemble):
            _largest_grp_size_j: int = int(matches[j].sum(dim=1).max().item())  # unused, preserved
            dist_mat = matches[i].float() - matches[j].float()
            dists[i, j] = torch.tril(dist_mat, diagonal=-1).abs().sum()
    return dists


def matching_dist(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    """vectorized version"""
    s_ensemble: int
    n_components: int
    s_ensemble, n_components = X.shape
    matches: Bool[Tensor, "s n n"] = (X[:, :, None] == X[:, None, :])
    diffs: Bool[Tensor, "s s n n"] = matches[:, None, :, :] ^ matches[None, :, :, :]
    tril_mask: Bool[Tensor, "n n"] = torch.tril(
        torch.ones((n_components, n_components), dtype=torch.bool, device=X.device),
        diagonal=-1,
    )
    masked_diffs: Bool[Tensor, "s s n n"] = diffs & tril_mask
    pair_counts: Tensor = masked_diffs.sum(dim=(-1, -2))
    out: Tensor = torch.full((s_ensemble, s_ensemble), float("nan"), device=X.device)
    upper_mask: Tensor = torch.triu(
        torch.ones((s_ensemble, s_ensemble), dtype=torch.bool, device=X.device),
        diagonal=1,
    )
    out[upper_mask] = pair_counts.to(torch.float32)[upper_mask]
    return out


def assert_allclose_nansame(a: Tensor, b: Tensor, rtol: float = 0.0, atol: float = 0.0) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        raise AssertionError("NaN pattern mismatch")
    mask = ~torch.isnan(a)
    if mask.any():
        if not torch.allclose(a[mask], b[mask], rtol=rtol, atol=atol):
            raise AssertionError("non-NaN elements differ")


def run_tests() -> None:
    torch.manual_seed(0)

    # small exact case
    X_small: Tensor = torch.tensor([[0,1,2,1],[0,2,2,1],[3,3,3,3]], dtype=torch.long)
    a = matching_dist_orig(X_small)
    b = matching_dist(X_small)
    assert_allclose_nansame(a, b)

    # randomized coverage
    for s in [1, 2, 3, 5, 8, 12]:
        for n in [1, 2, 4, 7, 10, 16]:
            for vocab in [2, 3, 5, 11]:
                X: Tensor = torch.randint(low=0, high=vocab, size=(s, n), dtype=torch.long)
                a = matching_dist_orig(X)
                b = matching_dist(X)
                assert_allclose_nansame(a, b)

    # rough speed sanity
    s, n, vocab = 32, 64, 16
    X_perf: Tensor = torch.randint(0, vocab, (s, n), dtype=torch.long)
    t0 = time.time(); _ = matching_dist_orig(X_perf); t1 = time.time()
    _ = matching_dist(X_perf); t2 = time.time()
    speedup = (t1 - t0) / max(t2 - t1, 1e-9)

    print("ALL TESTS PASSED")
    print(f"Approx speedup (vectorized vs orig): {speedup:.2f}x")

    # show example output for the small case
    print("Example input X_small:")
    print(X_small)
    print("matching_dist_orig(X_small):")
    print(a)
    print("matching_dist(X_small):")
    print(b)


run_tests()
