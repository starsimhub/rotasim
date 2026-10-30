"""Exp 47 -- India / Vellore: age_binned with p_symp free, bounded to the
slum<->MAL-ED bracket (AGE_PSYMP_INTERP=1) instead of fixed at either
endpoint. See README.md.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  tmux new-session -s india47 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 AGE_PSYMP_INTERP=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/47_india_age_psymp_interp/outputs/hm \\
     2>&1 | tee experiments/47_india_age_psymp_interp/india47.log"
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
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1', 'AGE_PSYMP_INTERP': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
