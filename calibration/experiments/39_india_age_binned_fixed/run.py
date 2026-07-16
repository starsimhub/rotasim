"""Exp 39 — India / Vellore: age_binned model with biweekly-fixed p_symp per age bin.

Structural rationale
--------------------
infnum forces p_symp_1 = maximum symptomatic rate, but Vellore biweekly data shows
<6m (0.381) < 6-11m (0.407) — very young first infections are LESS symptomatic.
age_binned assigns p_symp directly per age bin, so it can match this pattern.
We fix the three p_symp bins at their biweekly-calibrated values (FIXED_AGE_PSYMP)
to reduce free parameters: only FOI + susceptibility + maternal titer are searched.

Env flags required:
  MALED_SITE=india     — India demographics, data files, FIRST_INF_QUANT=q25, USE_IR_ALL=True
  NEO_PRIME=1          — NeonatalPriming(p_neo=0.5, age_weeks=2.0, sus_effect=0.0)
  EXT_PENALTY=1        — log_symp_ir_sum extinction penalty placed first in OBS_COLS

Run on zebra (160 cores, non-spot):
  tmux new-session -s india39 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --fix-age-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 \\
       --out-dir experiments/39_india_age_binned_fixed/outputs/hm \\
     2>&1 | tee experiments/39_india_age_binned_fixed/india39.log"
"""
import subprocess, pathlib, os, sys

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]

cmd = [
    sys.executable, str(CALIB / 'hm_calibrate.py'),
    '--model', 'age_binned',
    '--maternal', 'titer',
    '--fix-age-psymp',
    '--all-targets',
    '--n-samples', '1500',
    '--max-iter', '6',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
