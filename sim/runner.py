"""Scenario grid construction, parallel execution, and result summarisation."""

import itertools
import pandas as pd
from joblib import Parallel, delayed

from .replicate import one_replicate


# ── Scenario grids ────────────────────────────────────────────────────────────

def build_scenarios() -> list[dict]:
    """Full factorial design matching the proposal.

    Crosses n, p, rho, covariance structure, and error distribution while
    skipping logically inconsistent combinations (rho > 0 with 'independent',
    rho = 0 with a structured covariance).
    """
    grid = []
    for n, p, rho, structure, error_dist in itertools.product(
        [100, 200],
        [200, 500],
        [0.0, 0.3, 0.6, 0.9],
        ["independent", "block", "ar1"],
        ["normal", "t3"],
    ):
        if rho == 0 and structure != "independent":
            continue
        if rho > 0 and structure == "independent":
            continue
        grid.append(dict(n=n, p=p, rho=rho,
                         structure=structure, error_dist=error_dist))
    return grid


def build_quick_scenarios() -> list[dict]:
    """Minimal 3-scenario grid for smoke-testing (completes in seconds)."""
    return [
        dict(n=100, p=200, rho=0.0, structure="independent", error_dist="normal"),
        dict(n=100, p=200, rho=0.6, structure="ar1",         error_dist="normal"),
        dict(n=100, p=200, rho=0.6, structure="block",       error_dist="t3"),
    ]


# ── Parallel scenario runner ──────────────────────────────────────────────────

def run_scenario(
    scenario: dict,
    n_sim: int = 200,
    B: int = 150,
    B_inner: int = 30,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run n_sim replicates for one scenario in parallel.

    Seeds are set to 0, 1, …, n_sim-1 so results are fully reproducible.

    Returns
    -------
    DataFrame with one row per replicate.
    """
    n, p, rho, structure, error_dist = (
        scenario["n"], scenario["p"], scenario["rho"],
        scenario["structure"], scenario["error_dist"],
    )
    label = f"n={n},p={p},rho={rho},{structure},{error_dist}"
    print(f"  {label}  ({n_sim} reps)")

    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(one_replicate)(
            n=n, p=p, rho=rho, structure=structure, error_dist=error_dist,
            B=B, B_inner=B_inner, seed=s,
        )
        for s in range(n_sim)
    )
    return pd.DataFrame(rows)


# ── Result summary ────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one scenario's replicate-level results into a summary table."""
    rows = []
    for method, pref in [
        ("Debiased Lasso",       "debiased"),
        ("Bootstrap Percentile", "pct"),
        ("Bootstrap-t",          "boot_t"),
    ]:
        rows.append({
            "method":   method,
            "coverage": df[f"{pref}_cover"].mean(),
            "ci_width": df[f"{pref}_width"].mean(),
            "bias":     df[f"{pref}_bias"].mean(),
            "fpr":      df[f"{pref}_fpr"].mean(),
            "time":     df[f"time_{pref}"].mean(),
        })
    return pd.DataFrame(rows).set_index("method")
