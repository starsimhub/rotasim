"""Exp 52 -- India / Vellore: re-run exp39's age_binned HM config with the new
multi-seed survival vote (SURVIVAL_VOTE=1) replacing the old single-seed
log_symp_ir_sum sentinel (EXT_PENALTY=1). See README.md.

Run on zebra (160 cores, non-spot):
  tmux new-session -d -s india52 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 SURVIVAL_VOTE=1 HM_WORKERS=150 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --fix-age-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 \\
       --out-dir experiments/52_india_survival_vote/outputs/hm \\
     2>&1 | tee experiments/52_india_survival_vote/india52.log"
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
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'SURVIVAL_VOTE': '1'})
env.setdefault('HM_WORKERS', '150')
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
