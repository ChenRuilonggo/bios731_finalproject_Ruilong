"""Debiased (desparsified) Lasso inference via nodewise regression."""

import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso


def nodewise_lasso_row(
    X: np.ndarray,
    j: int,
    mu: float | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate row j of the precision matrix via nodewise Lasso.

    Regresses X_j on X_{-j} with Lasso penalty mu = sqrt(log p / n),
    then constructs the rescaled precision-matrix row Theta_j.

    Parameters
    ----------
    X  : design matrix, shape (n, p)
    j  : column index of interest
    mu : Lasso penalty; defaults to sqrt(log p / n)

    Returns
    -------
    Theta_j  : estimated j-th row of Theta = Sigma^{-1}, shape (p,)
    tau2_hat : estimated noise variance for the nodewise regression
    """
    n, p = X.shape
    if mu is None:
        mu = np.sqrt(np.log(p) / n)

    X_j   = X[:, j]
    mask  = np.ones(p, dtype=bool)
    mask[j] = False
    X_neg = X[:, mask]

    m = Lasso(alpha=mu, max_iter=3000, fit_intercept=False)
    m.fit(X_neg, X_j)
    gamma_hat = m.coef_

    resid    = X_j - X_neg @ gamma_hat
    tau2_hat = np.dot(resid, resid) / n + mu * np.sum(np.abs(gamma_hat))
    tau2_hat = max(tau2_hat, 1e-8)

    Theta_j        = np.zeros(p)
    Theta_j[mask]  = -gamma_hat
    Theta_j[j]     = 1.0
    Theta_j       /= tau2_hat
    return Theta_j, tau2_hat


def debiased_ci(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    beta_lasso: np.ndarray,
    lasso_resid: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Construct a debiased Lasso confidence interval for coefficient j.

    Reuses pre-computed beta_lasso and lasso_resid to avoid redundant fits.

    Returns
    -------
    beta_tilde_j : debiased point estimate
    ci_lo        : lower CI bound
    ci_hi        : upper CI bound
    """
    n, _ = X.shape
    Theta_j, _ = nodewise_lasso_row(X, j)

    # First-order bias correction
    correction   = (Theta_j @ X.T @ lasso_resid) / n
    beta_tilde_j = beta_lasso[j] + correction

    # Asymptotic sandwich variance: sigma^2 * (Theta_j X^T X Theta_j) / n^2
    sigma2 = np.var(lasso_resid)
    ThetaX = Theta_j @ X.T          # shape (n,)
    avar   = sigma2 * np.dot(ThetaX, ThetaX) / (n ** 2)
    se     = np.sqrt(max(avar, 1e-12))

    z     = stats.norm.ppf(1 - alpha / 2)
    ci_lo = beta_tilde_j - z * se
    ci_hi = beta_tilde_j + z * se
    return beta_tilde_j, ci_lo, ci_hi
