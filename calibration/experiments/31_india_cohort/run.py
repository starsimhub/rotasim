"""exp31 — India Vellore cohort HM: launcher reference.
All runs were executed on covaguest (akraay@52.137.65.109) via tmux.
Checkpoint outputs are in outputs/hm/, outputs/hm_neoprime/, outputs/hm_irall/.
Trajectory selection outputs are in outputs/ts_hm/, outputs/ts_neoprime/, outputs/ts_irall/.

── HM run commands (covaguest, MALED_SITE=india, rota-hm conda env) ──────────

# hm: symptomatic-only targets, no neonatal priming (baseline)
MALED_SITE=india HM_WORKERS=120 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python hm_calibrate.py --model infnum --fix-titer-shape --all-targets \
  --early-stop --max-iter 3 --out-dir experiments/31_india_cohort/outputs/hm

# hm_neoprime: symptomatic-only targets + neonatal priming (NEO_PRIME=1)
MALED_SITE=india NEO_PRIME=1 HM_WORKERS=120 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python hm_calibrate.py --model infnum --fix-titer-shape --all-targets \
  --early-stop --max-iter 3 --out-dir experiments/31_india_cohort/outputs/hm_neoprime

# hm_irall: adds all-infection IR targets (USE_IR_ALL auto-on for india), no priming
MALED_SITE=india HM_WORKERS=120 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python hm_calibrate.py --model infnum --fix-titer-shape --all-targets \
  --early-stop --max-iter 3 --out-dir experiments/31_india_cohort/outputs/hm_irall

── Trajectory selection commands ─────────────────────────────────────────────

MALED_SITE=india HM_WORKERS=40 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python trajectory_select.py --model infnum --fix-titer-shape --early-stop --n 3000 \
  --hm-dir experiments/31_india_cohort/outputs/hm/maled_infnum_titer_fixedshape \
  --out-dir experiments/31_india_cohort/outputs/ts_hm

MALED_SITE=india NEO_PRIME=1 HM_WORKERS=40 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python trajectory_select.py --model infnum --fix-titer-shape --early-stop --n 3000 \
  --hm-dir experiments/31_india_cohort/outputs/hm_neoprime/maled_infnum_titer_fixedshape \
  --out-dir experiments/31_india_cohort/outputs/ts_neoprime

MALED_SITE=india HM_WORKERS=40 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python trajectory_select.py --model infnum --fix-titer-shape --early-stop --n 3000 \
  --hm-dir experiments/31_india_cohort/outputs/hm_irall/maled_infnum_titer_fixedshape \
  --out-dir experiments/31_india_cohort/outputs/ts_irall

── Figures ───────────────────────────────────────────────────────────────────

python experiments/31_india_cohort/fig_bestfit_comparison.py
"""
