# BIOS 731 Final Project — Ruilong Chen

Simulation study on inference methods for high-dimensional linear regression.  
Compares Lasso, debiased Lasso, bootstrap, and naive OLS across a range of covariance structures and error distributions.

## Repository structure

```
sim/                  # Core simulation modules (Python)
  data.py             # Data generation
  estimators.py       # Lasso, debiased Lasso
  debiased.py         # Debiasing procedure
  bootstrap.py        # Bootstrap inference
  replicate.py        # Single-replicate logic
  runner.py           # Scenario builder and parallel runner
run_simulation.py     # Main simulation entry point
run_naive.py          # Naive OLS baseline
plot_results.R        # Result figures (report)
plot_presentation.R   # Result figures (slides)
submit.sh             # SLURM array job script
Makefile              # Convenience targets
requirements.txt      # Python dependencies
731_project.pdf       # Project description
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run a quick local test (small n_sim)
python run_simulation.py --quick

# Full run with parallel workers
python run_simulation.py --n_sim 200 --n_jobs 8

# Merge per-scenario CSVs after SLURM array job finishes
python run_simulation.py --merge

# Plot results
Rscript plot_results.R
```

## SLURM

```bash
sbatch submit.sh
```

Each array task runs one scenario. After all tasks complete, run `--merge` to combine results.
