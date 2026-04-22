"""sim — BIOS 731 simulation package.

Public API
----------
from sim.data       import make_covariance, generate_data
from sim.estimators import fit_lasso_cv, fit_lasso_fixed
from sim.debiased   import nodewise_lasso_row, debiased_ci
from sim.bootstrap  import bootstrap_coefs, percentile_ci, bootstrap_t_ci
from sim.replicate  import one_replicate
from sim.runner     import build_scenarios, build_quick_scenarios, run_scenario, summarize
"""
