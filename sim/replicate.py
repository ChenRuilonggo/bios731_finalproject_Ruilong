"""Single simulation replicate: generates data and evaluates all three methods."""

import time
import numpy as np

from .data       import generate_data
from .estimators import fit_lasso_cv
from .debiased   import debiased_ci
from .bootstrap  import bootstrap_coefs, percentile_ci, bootstrap_t_ci


def one_replicate(
    n: int,
    p: int,
    rho: float,
    structure: str,
    error_dist: str,
    s: int = 10,
    beta_val: float = 0.5,
    B: int = 150,
    B_inner: int = 30,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run one simulation replicate and return a flat result dictionary.

    Evaluates three inference methods (debiased Lasso, bootstrap percentile,
    bootstrap-t) on two target coefficients:
      j_signal : a nonzero coefficient (true value = beta_val)
      j_null   : a zero coefficient  (used for false-positive rate)

    Parameters
    ----------
    n, p        : sample size and number of predictors
    rho         : predictor correlation strength
    structure   : covariance structure ('independent' | 'ar1' | 'block')
    error_dist  : error distribution ('normal' | 't3')
    s           : number of nonzero true coefficients
    beta_val    : common value of nonzero coefficients
    B, B_inner  : outer / inner bootstrap resamples
    alpha       : nominal error rate (CI level = 1 - alpha)
    seed        : RNG seed

    Returns
    -------
    dict with keys: n, p, rho, structure, error_dist, seed, and per-method
    columns for bias, cover, width, fpr, time.
    """
    beta_true       = np.zeros(p)
    beta_true[:s]   = beta_val
    j_signal        = 0   # nonzero coefficient
    j_null          = s   # null coefficient

    X, y = generate_data(n, p, beta_true, rho, structure, error_dist, seed)

    out = dict(n=n, p=p, rho=rho, structure=structure,
               error_dist=error_dist, seed=seed)

    # ── Lasso (CV) ──────────────────────────────────────────────
    t0 = time.perf_counter()
    beta_lasso, lam, lasso_resid = fit_lasso_cv(X, y)
    out["time_lasso"] = time.perf_counter() - t0
    out["bias_lasso"] = beta_lasso[j_signal] - beta_val

    # ── Debiased Lasso ───────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        b_tilde, lo_db, hi_db = debiased_ci(
            X, y, j_signal, beta_lasso, lasso_resid, alpha)
        _, lo_db_z, hi_db_z = debiased_ci(
            X, y, j_null,   beta_lasso, lasso_resid, alpha)
        out["debiased_bias"]  = b_tilde - beta_val
        out["debiased_cover"] = int(lo_db <= beta_val <= hi_db)
        out["debiased_width"] = hi_db - lo_db
        out["debiased_fpr"]   = int(not (lo_db_z <= 0.0 <= hi_db_z))
    except Exception:
        out.update(debiased_bias=np.nan, debiased_cover=np.nan,
                   debiased_width=np.nan, debiased_fpr=np.nan)
    out["time_debiased"] = time.perf_counter() - t0

    # ── Bootstrap (shared outer resamples) ───────────────────────
    t0 = time.perf_counter()
    stars_sig  = bootstrap_coefs(X, y, j_signal, lam, B=B, seed=seed)
    stars_null = bootstrap_coefs(X, y, j_null,   lam, B=B, seed=seed + 11111)
    t_outer    = time.perf_counter() - t0

    # Percentile CI
    lo_pct,   hi_pct   = percentile_ci(stars_sig,  alpha)
    lo_pct_z, hi_pct_z = percentile_ci(stars_null, alpha)
    out["pct_bias"]  = np.nanmean(stars_sig) - beta_val
    out["pct_cover"] = int(lo_pct <= beta_val <= hi_pct)   if np.isfinite(lo_pct)   else np.nan
    out["pct_width"] = hi_pct - lo_pct                     if np.isfinite(lo_pct)   else np.nan
    out["pct_fpr"]   = int(not (lo_pct_z <= 0.0 <= hi_pct_z)) if np.isfinite(lo_pct_z) else np.nan
    out["time_pct"]  = t_outer

    # Bootstrap-t CI
    t0 = time.perf_counter()
    lo_bt,   hi_bt   = bootstrap_t_ci(
        X, y, beta_lasso[j_signal], stars_sig,  j_signal, lam, B_inner, alpha, seed)
    lo_bt_z, hi_bt_z = bootstrap_t_ci(
        X, y, beta_lasso[j_null],   stars_null, j_null,   lam, B_inner, alpha, seed + 22222)
    out["boot_t_bias"]  = np.nanmean(stars_sig) - beta_val
    out["boot_t_cover"] = int(lo_bt <= beta_val <= hi_bt)       if np.isfinite(lo_bt)   else np.nan
    out["boot_t_width"] = hi_bt - lo_bt                         if np.isfinite(lo_bt)   else np.nan
    out["boot_t_fpr"]   = int(not (lo_bt_z <= 0.0 <= hi_bt_z)) if np.isfinite(lo_bt_z) else np.nan
    out["time_boot_t"]  = t_outer + (time.perf_counter() - t0)

    return out
