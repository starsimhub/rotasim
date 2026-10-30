"""Exp 44 (age_and_infection sibling) -- India / Vellore: age_and_infection
(now bug-fixed in MALEDCohort, see README.md) + fractional neonatal
order-crediting (neonatal_order_effect, FITTED). All of beta0-3 free (no
--fix mechanism exists yet for this model) -> 14 free params, watch ESS.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  tmux new-session -s india44ageinf \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model age_and_infection --maternal titer \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/44_india_order_effect_age_inf/outputs/hm \\
     2>&1 | tee experiments/44_india_order_effect_age_inf/india44ageinf.log"
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
    '--early-stop',
    '--out-dir', str(HERE / 'outputs' / 'hm'),
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1', 'EXT_PENALTY': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
