"""Exp 49 -- India / Vellore: exp47's design with the 6-11m p_symp bracket
widened to (0.40, 0.60) to cover the corrected slum anchor (0.593, found
2026-08-14 -- FIXED_AGE_PSYMP's original 0.407 had a calculation error).
See README.md. STAGED, not auto-launched -- run manually once exp48's
result is in and a decision is made on whether to build on exp47 or exp48.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  tmux new-session -s india49 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 AGE_PSYMP_INTERP=1 AGE_PSYMP_INTERP_V=2 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/49_india_age_psymp_interp_wide/outputs/hm \\
     2>&1 | tee experiments/49_india_age_psymp_interp_wide/india49.log"
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
            'AGE_PSYMP_INTERP': '1', 'AGE_PSYMP_INTERP_V': '2'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
