"""Exp 52 -- India / Vellore: re-run exp39's age_binned HM config with the new
multi-seed survival vote (SURVIVAL_VOTE=1) replacing the old single-seed
log_symp_ir_sum sentinel (EXT_PENALTY=1). See README.md.

First launch (48 min in, HM_WORKERS=130, no --early-stop) was killed and
restarted: robyn's headroom wasn't guaranteed to clear, and with 5 seeds/draw
now paying the extinct-sim cost 5x over, --early-stop's ~3.5x speedup on the
~75-80%% of India sims that go extinct matters far more than the 48 min of
sunk progress.

Run on zebra (160 cores, non-spot; robyn's session measured at <1%% CPU, so
155 workers leaves comfortable headroom):
  tmux new-session -d -s india52 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 SURVIVAL_VOTE=1 HM_WORKERS=155 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_binned --maternal titer \\
       --fix-age-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 \\
       --early-stop \\
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
    '--early-stop',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'SURVIVAL_VOTE': '1'})
env.setdefault('HM_WORKERS', '155')
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
