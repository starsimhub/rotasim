"""Exp 43 — India / Vellore: age_binned refit with neonatal priming as a real,
detectable (but never symptomatic) early infection -- see README.md for the full
rationale. Same HM configuration as exp39; the only change is the NeonatalPriming/
MALEDCohort mechanics behind NEO_PRIME=1 (rotasim/analyzers.py + calibrate_maled.py).

Run on zebra (160 cores, non-spot) -- check `who` / `ps` for other users first:
  tmux new-session -s india43 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --fix-age-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/43_india_neonatal_detected/outputs/hm \\
     2>&1 | tee experiments/43_india_neonatal_detected/india43.log"
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
    '--early-stop',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
