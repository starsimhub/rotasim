"""Exp 48 -- India / Vellore: exp47's design (age_binned, free p_symp bounded
to the slum<->MAL-ED bracket) plus a smoothed extinction-probability
classifier instead of the raw single-seed sentinel-mixed regression target.
See README.md.

Staged to run AFTER exp47 finishes (same machine, sequential). Launch
pattern used: a waiter tmux session polls for exp47's TS completion, then
runs this. Manual/standalone launch on zebra:
  tmux new-session -s india48 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 EXT_CLASSIFIER=1 AGE_PSYMP_INTERP=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/48_india_ext_classifier/outputs/hm \\
     2>&1 | tee experiments/48_india_ext_classifier/india48.log"
"""
import subprocess, pathlib, os, sys

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]

cmd = [
    sys.executable, str(CALIB / 'hm_calibrate.py'),
    '--model', 'age_binned',
    '--maternal', 'titer',
    '--all-targets',
    '--n-samples', '1500',
    '--max-iter', '6',
    '--early-stop',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1',
            'EXT_CLASSIFIER': '1', 'AGE_PSYMP_INTERP': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
