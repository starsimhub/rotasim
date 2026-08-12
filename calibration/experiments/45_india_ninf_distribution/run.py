"""Exp 45 -- India / Vellore: per-agent infection-count distribution from exp39.

Diagnostic (not an HM run): re-runs 8 draws from exp39's resampled posterior
(age_binned, fix_age_psymp, titer maternal) at 40k agents each and pools the
per-child true cumulative infection count (MALEDCohort.n_inf) across the
non-extinct replicates. Answers: does the model show one exposure-risk pool,
or a low/high-exposure bimodal split, among enrolled children? See README.md
/ SUMMARY.md for the result.

Requires the `n_inf` field added to calibrate_maled._run_one_replicate's
cohort-observation output (harmless/additive; ignored by the GOF/HM path).

Run locally (no HM/zebra needed -- each replicate ~30-60s at 40k agents):
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MALED_SITE=india NEO_PRIME=1 \
    python3 experiments/45_india_ninf_distribution/run.py
"""
import sys, os, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
os.chdir(CALIB)   # posterior.csv path below is relative to calibration/

import numpy as np, pandas as pd
import hm_calibrate as H
import calibrate_maled as cm

N_REPS = 8
OUT = HERE / 'outputs' / 'ninf_per_rep.jsonl'

post = pd.read_csv('experiments/39_india_age_binned_fixed/outputs/ts/posterior.csv')
uniq = post.drop_duplicates().reset_index(drop=True)

with open(OUT, 'w') as fout:
    for i in range(N_REPS):
        row = uniq.iloc[i % len(uniq)]
        sp = H.untransform(row, 'age_binned', 'titer', fix_age_psymp=True)
        cfg = H.build_sim_config('age_binned', 40000, 'titer')
        mo = cm._run_one_replicate((cfg, sp, 1000 + i, H.CAL_WINDOW))
        ninf = mo['n_inf']
        fout.write(json.dumps(ninf) + '\n')
        print(f"rep {i}: n_enrolled={len(ninf)}, mean={np.mean(ninf):.3f}, "
              f"extinct={sum(ninf) == 0}")
