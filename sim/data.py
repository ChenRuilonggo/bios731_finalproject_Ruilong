"""Data generation for the high-dimensional regression simulation."""

import numpy as np


def make_covariance(p: int, rho: float, structure: str) -> np.ndarray:
    """Build a p×p covariance matrix.

    Parameters
    ----------
    p         : number of predictors
    rho       : correlation strength (0 = independent)
    structure : 'independent' | 'ar1' | 'block'
    """
    if structure == "independent" or rho == 0:
        return np.eye(p)
    if structure == "ar1":
        idx = np.arange(p)
        return rho ** np.abs(idx[:, None] - idx[None, :])
    if structure == "block":
        block_size = 10
        Sigma = np.eye(p)
        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            sz  = end - start
            block = np.full((sz, sz), rho)
            np.fill_diagonal(block, 1.0)
            Sigma[start:end, start:end] = block
        return Sigma
    raise ValueError(f"Unknown covariance structure: {structure!r}")


def generate_data(
    n: int,
    p: int,
    beta_true: np.ndarray,
    rho: float,
    structure: str,
    error_dist: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) from a high-dimensional linear model.

    Parameters
    ----------
    n          : sample size
    p          : number of predictors
    beta_true  : true coefficient vector, shape (p,)
    rho        : predictor correlation strength
    structure  : covariance structure ('independent' | 'ar1' | 'block')
    error_dist : 'normal' or 't3' (t with 3 df, scaled to unit variance)
    seed       : RNG seed for reproducibility
    """
    rng   = np.random.default_rng(seed)
    Sigma = make_covariance(p, rho, structure)
    L     = np.linalg.cholesky(Sigma + 1e-8 * np.eye(p))  # jitter for stability
    X     = rng.standard_normal((n, p)) @ L.T
    X    /= X.std(axis=0, keepdims=True)                   # standardise columns

    if error_dist == "normal":
        eps = rng.standard_normal(n)
    else:  # t3, scaled to variance 1
        eps = rng.standard_t(df=3, size=n) / np.sqrt(3)

    y = X @ beta_true + eps
    return X, y
