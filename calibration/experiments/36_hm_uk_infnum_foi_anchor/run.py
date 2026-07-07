"""Exp 36 — UK HM (infnum) with first-infection timing target.

Re-calibrates UK infnum posterior including a first-infection age constraint
(Hasso-Agopsowicz et al. PMC6736387: HIC median 65 wk / 15 mo, IQR 40-107 wk).
Replaces exp 28 (shape-only) as the canonical UK posterior for VE prediction.

Run on zebra (akraay@20.14.72.192) — requires zebra to be free.

── Pre-run: rsync code changes ──────────────────────────────────────────────
rsync -avz -e "ssh -i ~/Downloads/zebra_akraay.pem" \\
  /path/to/rotasim/rotasim/rotasim/analyzers.py \\
  /path/to/rotasim/rotasim/calibration/calibrate_maled.py \\
  /path/to/rotasim/rotasim/calibration/hm_calibrate_uk.py \\
  akraay@20.14.72.192:/home/akraay/rotasim/rotasim/rotasim/
# (analyzers.py and calibrate_maled.py go to rotasim/; hm_calibrate_uk.py stays in calibration/)

── HM run ────────────────────────────────────────────────────────────────────
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=158 \\
  /home/akraay/rotasim/rotasim/.venv/bin/python hm_calibrate_uk.py \\
  --model infnum --fix-titer-shape --all-targets \\
  --max-iter 3 \\
  --out-dir experiments/36_hm_uk_infnum_foi_anchor/outputs/hm

── Trajectory selection ──────────────────────────────────────────────────────
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=80 \\
  /home/akraay/rotasim/rotasim/.venv/bin/python trajectory_select.py \\
  --model infnum --fix-titer-shape --site uk --n 3000 \\
  --hm-dir experiments/36_hm_uk_infnum_foi_anchor/outputs/hm/uk_infnum_titer_fixedshape \\
  --out-dir experiments/36_hm_uk_infnum_foi_anchor/outputs/ts

── Expected outputs ──────────────────────────────────────────────────────────
outputs/hm/uk_infnum_titer_fixedshape/wave1/nroy_samples.csv  (new NROY, first_inf ~9-24m)
outputs/ts/nroy_draw.csv + sir_results.jsonl  (trajectory selection)
"""
