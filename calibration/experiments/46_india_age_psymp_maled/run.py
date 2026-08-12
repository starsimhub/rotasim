"""Exp 46 -- India / Vellore: age_binned refit with MAL-ED-derived (ascertainment-
corrected) p_symp instead of the slum-cohort-derived FIXED_AGE_PSYMP exp39 used.
See README.md.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  tmux new-session -s india46 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 AGE_PSYMP_SOURCE=maled \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --fix-age-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/46_india_age_psymp_maled/outputs/hm \\
     2>&1 | tee experiments/46_india_age_psymp_maled/india46.log"
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
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1', 'AGE_PSYMP_SOURCE': 'maled'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
