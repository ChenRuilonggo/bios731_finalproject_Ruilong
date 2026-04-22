"""
run_simulation.py — CLI entry point for the BIOS 731 simulation study.

Usage
-----
  # Local / interactive
  python run_simulation.py --quick
  python run_simulation.py --n_sim 200 --n_jobs 8

  # SLURM array job (one task per scenario)
  python run_simulation.py --scenario_idx 3 --n_jobs 8

  # After all array tasks finish, merge per-scenario CSVs:
  python run_simulation.py --merge
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from sim.runner import build_scenarios, build_quick_scenarios, run_scenario, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIOS 731: inference for high-dimensional regression")

    # ── modes ──────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Smoke test: 3 scenarios × 10 reps")
    mode.add_argument("--scenario_idx", type=int, metavar="N",
                      help="Run only scenario N (0-based); used by SLURM array jobs")
    mode.add_argument("--merge", action="store_true",
                      help="Merge per-scenario CSVs into all_results.csv (run after array job)")

    # ── tuning ─────────────────────────────────────────────────
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


# ── helpers ───────────────────────────────────────────────────────────────────

def scenario_csv(sc: dict, out_dir: str) -> str:
    return os.path.join(
        out_dir,
        "n{n}_p{p}_rho{rho}_{structure}_{error_dist}.csv".format(**sc),
    )


def save_config(args, n_sim, B, B_inner, n_scenarios, out_dir):
    config = dict(
        n_sim=n_sim, B=B, B_inner=B_inner,
        n_jobs=args.n_jobs, n_scenarios=n_scenarios,
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


# ── run modes ─────────────────────────────────────────────────────────────────

def run_single(args) -> None:
    """Run one scenario by index — called by each SLURM array task."""
    scenarios = build_scenarios()
    idx = args.scenario_idx
    if idx < 0 or idx >= len(scenarios):
        raise SystemExit(f"--scenario_idx must be in [0, {len(scenarios) - 1}]")

    sc = scenarios[idx]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scenario {idx}/{len(scenarios) - 1}: "
          f"n={sc['n']} p={sc['p']} rho={sc['rho']} "
          f"{sc['structure']} {sc['error_dist']}")

    df = run_scenario(sc, n_sim=args.n_sim, B=args.B,
                      B_inner=args.B_inner, n_jobs=args.n_jobs)
    df.to_csv(scenario_csv(sc, args.out_dir), index=False)
    print(summarize(df).round(3).to_string())


def run_merge(args) -> None:
    """Concatenate all per-scenario CSVs produced by the array job."""
    pattern = os.path.join(args.out_dir, "n*_p*_rho*_*_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No per-scenario CSVs found in {args.out_dir}/")

    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out    = os.path.join(args.out_dir, "all_results.csv")
    df_all.to_csv(out, index=False)
    print(f"Merged {len(files)} files → {out}  ({len(df_all)} rows)")

    scenarios = build_scenarios()
    save_config(args, args.n_sim, args.B, args.B_inner, len(scenarios), args.out_dir)


def run_all(args, quick: bool = False) -> None:
    """Run all scenarios sequentially (local / interactive use)."""
    n_sim   = 10 if quick else args.n_sim
    B       = 20 if quick else args.B
    B_inner = 5  if quick else args.B_inner

    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = build_quick_scenarios() if quick else build_scenarios()
    save_config(args, n_sim, B, B_inner, len(scenarios), args.out_dir)

    print(f"Total scenarios : {len(scenarios)}")
    print(f"Replicates/scen : {n_sim}")
    print(f"Output dir      : {args.out_dir}/\n")

    all_dfs = []
    for i, sc in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}]", end=" ")
        df = run_scenario(sc, n_sim=n_sim, B=B, B_inner=B_inner, n_jobs=args.n_jobs)
        df.to_csv(scenario_csv(sc, args.out_dir), index=False)
        print(summarize(df).round(3).to_string())
        print()
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(args.out_dir, "all_results.csv"), index=False)
    print(f"Done. Saved {args.out_dir}/all_results.csv")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.scenario_idx is not None:
        run_single(args)
    elif args.merge:
        run_merge(args)
    elif args.quick:
        run_all(args, quick=True)
    else:
        run_all(args, quick=False)


if __name__ == "__main__":
    main()
