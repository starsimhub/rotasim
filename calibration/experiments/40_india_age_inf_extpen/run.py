"""Exp 40 — India / Vellore: age_and_infection model with extinction penalty + 6 waves.

Structural rationale
--------------------
age_and_infection combines per-infection-number symptom modifiers (infnum) with
an age-binned offset, so it can match <6m < 6-11m while keeping the infection-order
structure. Exp 33 used only 3 waves and no extinction penalty (ESS=1 in TS).
This run adds EXT_PENALTY=1 (places log_symp_ir_sum first in CYCLE, cutting the
low-beta extinction zone early) and runs 6 waves to give the emulator more room
to tighten. Analogous to what the extinction penalty did for exp35 infnum.

Env flags required:
  MALED_SITE=india     — India demographics, data files, FIRST_INF_QUANT=q25, USE_IR_ALL=True
  NEO_PRIME=1          — NeonatalPriming(p_neo=0.5, age_weeks=2.0, sus_effect=0.0)
  EXT_PENALTY=1        — log_symp_ir_sum extinction penalty placed first in OBS_COLS

Run on zebra (160 cores, non-spot):
  tmux new-session -s india40 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_and_infection --maternal titer \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 \\
       --out-dir experiments/40_india_age_inf_extpen/outputs/hm \\
     2>&1 | tee experiments/40_india_age_inf_extpen/india40.log"
"""
import subprocess, pathlib, os, sys

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]

cmd = [
    sys.executable, str(CALIB / 'hm_calibrate.py'),
    '--model', 'age_and_infection',
    '--maternal', 'titer',
    '--all-targets',
    '--n-samples', '1500',
    '--max-iter', '6',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
