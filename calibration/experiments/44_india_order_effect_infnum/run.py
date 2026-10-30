"""Exp 44 (infnum sibling) -- India / Vellore: infnum + fractional neonatal
order-crediting (neonatal_order_effect, FITTED). See README.md.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  tmux new-session -s india44infnum \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 \\
     ~/ukvenv/bin/python hm_calibrate.py \\
       --model infnum --maternal titer \\
       --fix-psymp \\
       --all-targets \\
       --n-samples 1500 --max-iter 6 --early-stop \\
       --out-dir experiments/44_india_order_effect_infnum/outputs/hm \\
     2>&1 | tee experiments/44_india_order_effect_infnum/india44infnum.log"
"""
import subprocess, pathlib, os, sys

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]

cmd = [
    sys.executable, str(CALIB / 'hm_calibrate.py'),
    '--model', 'infnum',
    '--maternal', 'titer',
    '--fix-psymp',
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
