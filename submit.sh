#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM array job — BIOS 731 simulation
#
# Each array task runs one scenario; tasks run in parallel across
# the cluster.  After all tasks finish, run:
#   make merge       → merge CSVs into results/all_results.csv
#   make figures     → draw R figures
#
# Submit:
#   make sbatch
# or manually:
#   sbatch --array=0-55 submit.sh
# ─────────────────────────────────────────────────────────────

#SBATCH --job-name=bios731
#SBATCH --partition=yanglab,week-long-cpu,day-long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

# ── environment ───────────────────────────────────────────────
CONDA_ROOT=/projects/YangLabData/Ruilong/tools/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate base

WORKDIR=/projects/YangLabData/Ruilong/731/final
cd "$WORKDIR"

mkdir -p logs results

echo "=== Task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_ARRAY_JOB_ID} ==="
echo "Host    : $(hostname)"
echo "CPUs    : ${SLURM_CPUS_PER_TASK}"
echo "Started : $(date)"

# ── run one scenario ──────────────────────────────────────────
python run_simulation.py \
    --scenario_idx "${SLURM_ARRAY_TASK_ID}" \
    --n_sim  200 \
    --B      150 \
    --B_inner 30 \
    --n_jobs "${SLURM_CPUS_PER_TASK}" \
    --out_dir results

echo "Finished: $(date)"
