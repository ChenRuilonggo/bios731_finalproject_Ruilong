"""
run_simulation.py — CLI entry point for the BIOS 731 simulation study.

Usage
-----
  python run_simulation.py                          # full run
  python run_simulation.py --quick                  # smoke test (3 scenarios, 10 reps)
  python run_simulation.py --n_sim 50 --n_jobs 4
  python run_simulation.py --out_dir my_results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from sim.runner import build_scenarios, build_quick_scenarios, run_scenario, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIOS 731: inference for high-dimensional regression")
    p.add_argument("--quick",    action="store_true",
                   help="Smoke test: 3 scenarios × 10 reps")
    p.add_argument("--n_sim",   type=int, default=200,
                   help="Replicates per scenario (default 200)")
    p.add_argument("--B",       type=int, default=150,
                   help="Outer bootstrap resamples (default 150)")
    p.add_argument("--B_inner", type=int, default=30,
                   help="Inner bootstrap resamples for bootstrap-t (default 30)")
    p.add_argument("--n_jobs",  type=int, default=-1,
                   help="Parallel workers; -1 = all cores (default -1)")
    p.add_argument("--out_dir", type=str, default="results",
                   help="Output directory (default: results/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_sim   = 10 if args.quick else args.n_sim
    B       = 20 if args.quick else args.B
    B_inner = 5  if args.quick else args.B_inner

    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = build_quick_scenarios() if args.quick else build_scenarios()

    # Persist run configuration for reproducibility
    config = dict(
        n_sim=n_sim, B=B, B_inner=B_inner,
        n_jobs=args.n_jobs, quick=args.quick,
        n_scenarios=len(scenarios),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Total scenarios : {len(scenarios)}")
    print(f"Replicates/scen : {n_sim}")
    print(f"Output dir      : {args.out_dir}/\n")

    all_dfs = []
    for i, sc in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}]", end=" ")
        df = run_scenario(sc, n_sim=n_sim, B=B, B_inner=B_inner, n_jobs=args.n_jobs)

        fname = os.path.join(
            args.out_dir,
            "n{n}_p{p}_rho{rho}_{structure}_{error_dist}.csv".format(**sc),
        )
        df.to_csv(fname, index=False)
        print(summarize(df).round(3).to_string())
        print()
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(args.out_dir, "all_results.csv"), index=False)
    print(f"Done. Saved {args.out_dir}/all_results.csv")


if __name__ == "__main__":
    main()
