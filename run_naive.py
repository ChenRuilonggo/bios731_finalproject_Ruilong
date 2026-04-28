"""
run_naive.py — Compute naive CI for all scenarios and merge into all_results.csv.

Naive CI: beta_lasso[j] +/- 1.96 * sqrt(var(resid) / n)

Usage
-----
  python run_naive.py               # runs all 56 scenarios, n_jobs=-1
  python run_naive.py --n_jobs 4
"""

import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sim.data       import generate_data
from sim.estimators import fit_lasso_cv
from sim.runner     import build_scenarios


S        = 10       # number of nonzero coefficients
BETA_VAL = 0.5
ALPHA    = 0.05
Z_VAL    = 1.96


def naive_replicate(n, p, rho, structure, error_dist, seed):
    beta_true      = np.zeros(p)
    beta_true[:S]  = BETA_VAL
    j_signal       = 0
    j_null         = S

    X, y = generate_data(n, p, beta_true, rho, structure, error_dist, seed)
    beta_lasso, _, resid = fit_lasso_cv(X, y)

    se = np.sqrt(np.var(resid) / n)

    lo_sig = beta_lasso[j_signal] - Z_VAL * se
    hi_sig = beta_lasso[j_signal] + Z_VAL * se
    lo_nul = beta_lasso[j_null]   - Z_VAL * se
    hi_nul = beta_lasso[j_null]   + Z_VAL * se

    return dict(
        n=n, p=p, rho=rho, structure=structure, error_dist=error_dist, seed=seed,
        naive_bias  = beta_lasso[j_signal] - BETA_VAL,
        naive_cover = int(lo_sig <= BETA_VAL <= hi_sig),
        naive_width = hi_sig - lo_sig,
        naive_fpr   = int(not (lo_nul <= 0.0 <= hi_nul)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",  type=int, default=-1)
    parser.add_argument("--n_sim",   type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    scenarios = build_scenarios()
    print(f"Running naive CI for {len(scenarios)} scenarios x {args.n_sim} reps ...")

    all_rows = []
    for i, sc in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] n={sc['n']} p={sc['p']} "
              f"rho={sc['rho']} {sc['structure']} {sc['error_dist']}")
        rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(naive_replicate)(
                sc["n"], sc["p"], sc["rho"], sc["structure"], sc["error_dist"], s
            )
            for s in range(args.n_sim)
        )
        all_rows.extend(rows)

    naive_df = pd.DataFrame(all_rows)

    # Merge with existing all_results.csv
    existing = pd.read_csv(f"{args.out_dir}/all_results.csv")
    key_cols = ["n", "p", "rho", "structure", "error_dist", "seed"]
    merged   = existing.merge(naive_df, on=key_cols, how="left")
    merged.to_csv(f"{args.out_dir}/all_results.csv", index=False)
    print(f"Done. Merged naive CI columns into {args.out_dir}/all_results.csv")
    print(f"  naive_cover mean: {merged['naive_cover'].mean():.3f}")


if __name__ == "__main__":
    main()
