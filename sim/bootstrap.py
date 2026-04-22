"""Pairs bootstrap inference: percentile CI and bootstrap-t CI."""

import numpy as np
from .estimators import fit_lasso_fixed


def bootstrap_coefs(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    lam: float,
    B: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Draw B pairs-bootstrap resamples and return beta*_j for each.

    Uses a fixed lambda (the one chosen by CV on the original data) to
    avoid running cross-validation inside every bootstrap iteration.

    Returns
    -------
    beta_stars : shape (B,); NaN entries indicate failed fits
    """
    n = X.shape[0]
    rng        = np.random.default_rng(seed)
    beta_stars = np.full(B, np.nan)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            beta_stars[b] = fit_lasso_fixed(X[idx], y[idx], lam)[j]
        except Exception:
            pass
    return beta_stars


def percentile_ci(
    beta_stars: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Basic percentile bootstrap CI.

    Returns (nan, nan) when fewer than 10 valid resamples are available.
    """
    valid = beta_stars[np.isfinite(beta_stars)]
    if len(valid) < 10:
        return np.nan, np.nan
    return float(np.quantile(valid, alpha / 2)), float(np.quantile(valid, 1 - alpha / 2))


def bootstrap_t_ci(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat_j: float,
    beta_stars: np.ndarray,
    j: int,
    lam: float,
    B_inner: int = 30,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap-t CI using an inner bootstrap to estimate SE for each resample.

    To keep computation affordable, only the first 50 valid outer resamples
    are used to build the t* distribution.

    Returns
    -------
    (ci_lo, ci_hi) or (nan, nan) if too few valid resamples.
    """
    n         = X.shape[0]
    rng       = np.random.default_rng(seed + 77777)
    valid_idx = np.where(np.isfinite(beta_stars))[0]

    if len(valid_idx) < 10:
        return np.nan, np.nan

    t_stars = []
    subset  = valid_idx[:min(len(valid_idx), 50)]

    for b in subset:
        beta_b = beta_stars[b]
        inner  = np.full(B_inner, np.nan)
        for k in range(B_inner):
            idx_k = rng.integers(0, n, size=n)
            try:
                inner[k] = fit_lasso_fixed(X[idx_k], y[idx_k], lam)[j]
            except Exception:
                pass
        s_star = np.nanstd(inner)
        if s_star > 1e-10:
            t_stars.append((beta_b - beta_hat_j) / s_star)

    if len(t_stars) < 10:
        return np.nan, np.nan

    t_arr  = np.array(t_stars)
    se_all = np.nanstd(beta_stars)
    q_lo   = np.quantile(t_arr, alpha / 2)
    q_hi   = np.quantile(t_arr, 1 - alpha / 2)
    return beta_hat_j - q_hi * se_all, beta_hat_j - q_lo * se_all
