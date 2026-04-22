# ─────────────────────────────────────────────────────────────
# BIOS 731 Final Project — Reproducible Workflow
#
# Local targets:
#   make          → full sequential run + figures (single machine)
#   make quick    → smoke-test (3 scenarios × 10 reps) + figures
#   make figures  → draw R figures from existing results/
#   make clean    → remove generated results and figures
#
# SLURM targets (cluster):
#   make sbatch   → submit 56-task array job to SLURM
#   make merge    → merge per-scenario CSVs after array job finishes
#   make cleanlog → remove SLURM log files
# ─────────────────────────────────────────────────────────────

PYTHON   := python3
RSCRIPT  := Rscript

N_SCENARIOS := $(shell $(PYTHON) -c \
    "from sim.runner import build_scenarios; print(len(build_scenarios()))")
ARRAY_END   := $(shell echo $$(($(N_SCENARIOS) - 1)))

RESULTS  := results/all_results.csv
FIGURES  := figures/fig_coverage.pdf figures/fig_width.pdf \
            figures/fig_fpr.pdf      figures/fig_time.pdf  \
            figures/fig_summary_tile.pdf

SIM_SOURCES := run_simulation.py sim/data.py sim/estimators.py sim/debiased.py \
               sim/bootstrap.py sim/replicate.py sim/runner.py

.PHONY: all simulate figures quick merge sbatch clean cleanlog

# ── Default ────────────────────────────────────────────────────
all: figures

# ── Local sequential run ───────────────────────────────────────
simulate: $(RESULTS)

$(RESULTS): $(SIM_SOURCES)
	$(PYTHON) run_simulation.py \
	    --n_sim   200 \
	    --B       150 \
	    --B_inner  30 \
	    --n_jobs   -1 \
	    --out_dir results

# ── R figures ──────────────────────────────────────────────────
figures: $(FIGURES)

$(FIGURES): plot_results.R $(RESULTS)
	$(RSCRIPT) plot_results.R

# ── Smoke test ─────────────────────────────────────────────────
quick:
	$(PYTHON) run_simulation.py --quick --out_dir results
	$(RSCRIPT) plot_results.R

# ── SLURM: submit array job ────────────────────────────────────
sbatch:
	@echo "Submitting array job: 0-$(ARRAY_END) ($(N_SCENARIOS) scenarios)"
	@mkdir -p logs results
	sbatch --array=0-$(ARRAY_END) submit.sh

# ── SLURM: merge results after array job ───────────────────────
merge:
	$(PYTHON) run_simulation.py --merge --out_dir results

# ── Clean ──────────────────────────────────────────────────────
clean:
	rm -rf results/ figures/

cleanlog:
	rm -f logs/sim_*.out logs/sim_*.err
