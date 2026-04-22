"""Lasso fitting utilities (CV-tuned and fixed-lambda)."""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV


def fit_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 3,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit Lasso with cross-validated lambda.

    Returns
    -------
    beta_hat : coefficient vector, shape (p,)
    lam      : chosen regularisation parameter
    resid    : in-sample residuals y - X @ beta_hat
    """
    m = LassoCV(cv=cv, max_iter=3000, alphas=30, fit_intercept=False, n_jobs=1)
    m.fit(X, y)
    resid = y - m.predict(X)
    return m.coef_, m.alpha_, resid


def fit_lasso_fixed(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Fit Lasso with a pre-specified lambda (no CV, fast).

    Returns
    -------
    beta_hat : coefficient vector, shape (p,)
    """
    m = Lasso(alpha=lam, max_iter=3000, fit_intercept=False)
    m.fit(X, y)
    return m.coef_
