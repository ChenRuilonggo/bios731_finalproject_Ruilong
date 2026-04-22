# ─────────────────────────────────────────────────────────────
# BIOS 731 Final Project — Reproducible Workflow
#
# Targets:
#   make          → full run: simulate + figures
#   make simulate → run Python simulation (200 reps, all scenarios)
#   make figures  → draw figures from existing results/
#   make quick    → smoke-test with 10 reps on a small grid
#   make clean    → remove generated results and figures
# ─────────────────────────────────────────────────────────────

PYTHON   := python3
RSCRIPT  := Rscript

RESULTS  := results/all_results.csv
FIGURES  := figures/fig_coverage.pdf figures/fig_width.pdf \
            figures/fig_fpr.pdf      figures/fig_time.pdf  \
            figures/fig_summary_tile.pdf

.PHONY: all simulate figures quick clean

# Default: full pipeline
all: figures

# ── Python simulation ──────────────────────────────────────────
simulate: $(RESULTS)

$(RESULTS): run_simulation.py sim/data.py sim/estimators.py sim/debiased.py \
            sim/bootstrap.py sim/replicate.py sim/runner.py
	$(PYTHON) run_simulation.py \
	    --n_sim  200 \
	    --B      150 \
	    --B_inner 30 \
	    --n_jobs  -1 \
	    --out_dir results

# ── R figures ─────────────────────────────────────────────────
figures: $(FIGURES)

$(FIGURES): plot_results.R $(RESULTS)
	$(RSCRIPT) plot_results.R

# ── Smoke test ────────────────────────────────────────────────
quick:
	$(PYTHON) run_simulation.py --quick --out_dir results
	$(RSCRIPT) plot_results.R

# ── Clean ─────────────────────────────────────────────────────
clean:
	rm -rf results/ figures/
