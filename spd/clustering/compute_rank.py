import numpy as np
from jaxtyping import Float


def compute_rank_of_sum(  # noqa: D401 - imperative summary is intentional
    U1: Float[np.ndarray, "d r1"],
    S1: Float[np.ndarray, " r1"],
    V1: Float[np.ndarray, "d r1"],
    U2: Float[np.ndarray, "d r2"],
    S2: Float[np.ndarray, " r2"],
    V2: Float[np.ndarray, "d r2"],
    *,
    tol: float = 1e-10,
) -> int:
    """Compute ``rank(P₁ + P₂)`` in **O(d (r₁+r₂)²)** time.

    Let ``P₁ = U₁ diag(S₁) V₁ᵀ`` and ``P₂ = U₂ diag(S₂) V₂ᵀ`` be two matrices whose
    thin SVD factors are already known.  
    By concatenating the factors and forming a *small* ``(r₁+r₂) x (r₁+r₂)``
    eigen-problem, the numerical rank of the sum can be found far faster than
    recomputing a full SVD of ``P₁ + P₂``.

    # Parameters:
     - `U1 : Float[np.ndarray, "d r1"]`  
       Left singular vectors of ``P₁``.
     - `S1 : Float[np.ndarray, "r1"]`  
       Singular values of ``P₁`` (1-D array).
     - `V1 : Float[np.ndarray, "d r1"]`  
       Right singular vectors of ``P₁``.
     - `U2 : Float[np.ndarray, "d r2"]`  
       Left singular vectors of ``P₂``.
     - `S2 : Float[np.ndarray, "r2"]`  
       Singular values of ``P₂``.
     - `V2 : Float[np.ndarray, "d r2"]`  
       Right singular vectors of ``P₂``.
     - `tol : float`  
       Eigenvalues ≤ `tol` are treated as zero  
       (defaults to `1e-10`).

    # Returns:
     - `int`  
       Numerical rank of ``P₁ + P₂``.

    # Usage:
    ```python
    >>> d, r1, r2 = 50, 5, 4
    >>> rng = np.random.default_rng(0)
    >>> def rand_orth(d_: int, r_: int) -> np.ndarray:
    ...     q, _ = np.linalg.qr(rng.standard_normal((d_, r_)))
    ...     return q[:, :r_]
    ...
    >>> U1, V1 = rand_orth(d, r1), rand_orth(d, r1)
    >>> U2, V2 = rand_orth(d, r2), rand_orth(d, r2)
    >>> S1, S2 = rng.random(r1) + 0.1, rng.random(r2) + 0.1
    >>> compute_rank_of_sum(U1, S1, V1, U2, S2, V2)
    9
    ```

    # Raises:
     - `ValueError` - if input shapes are inconsistent.
    """
    # ---- shape checks -------------------------------------------------------
    d: int = U1.shape[0]
    if (
        U1.shape != (d, S1.size)
        or V1.shape != (d, S1.size)
        or U2.shape[0] != d
        or V2.shape[0] != d
        or U2.shape[1] != S2.size
        or V2.shape[1] != S2.size
    ):
        raise ValueError("Inconsistent SVD factor shapes")

    # ---- concatenate factors ------------------------------------------------
    U: Float[np.ndarray, "d r"] = np.concatenate((U1, U2), axis=1)
    V: Float[np.ndarray, "d r"] = np.concatenate((V1, V2), axis=1)
    Sigma: Float[np.ndarray, "r r"] = np.diag(np.concatenate((S1, S2)))

    # ---- small eigen-problem: K_L = Σ (VᵀV) Σ --------------------------------
    G_R: Float[np.ndarray, "r r"] = V.T @ V                # Gram matrix
    K_L: Float[np.ndarray, "r r"] = Sigma @ G_R @ Sigma    # r x r

    eigvals: Float[np.ndarray, " r"] = np.linalg.eigvalsh(K_L)
    rank: int = int(np.sum(eigvals > tol))

    return rank
