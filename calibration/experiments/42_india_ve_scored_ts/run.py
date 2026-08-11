"""Exp 42 — India / Vellore: trajectory selection on exp39 NROY with VE scoring.

Re-scores the exp39 (age_binned + fixed psymp) NROY draws adding a Gaussian
VE logL term centred at 0.50 (sigma=0.07).  This breaks the beta/sus_after
degeneracy that the cohort-only calibration cannot resolve: high-FOI +
weak immunity fits the cohort data equally well as lower-FOI + stronger
immunity, but the two solutions produce very different VE predictions.
Bangladesh infnum posterior gave ~50% population-impact VE; India exp39 gave
33-34%.  Adding VE as a scoring target steers the posterior toward parameter
regions that simultaneously fit the Vellore cohort data AND reproduce
plausible vaccine dynamics.

VE target: 0.50 (population-impact at 90% coverage, 6-11m age bin)
  - Nair et al. Nature Medicine 2025 individual test-negative VE = 54% (47-68%)
  - User ecologic estimate at 6-11m: ~52%
  - Bangladesh model (infnum, similar cohort design): ~50% population VE
  Sigma = 0.07 (±14 pp at 2σ)

Each NROY draw runs 3 sims:
  1. Cohort (novax): Poisson IR + Binomial repeat + survival logL
  2. Surveillance novax: reference IR at 6-11m for VE denominator
  3. Surveillance vax: Rotavac 3-dose (6/10/14 wk), 90% coverage, take=0.74

Output: experiments/39_india_age_binned_fixed/outputs/ts/sir_results_ve50.jsonl
        experiments/39_india_age_binned_fixed/outputs/ts/posterior_ve50.csv
        experiments/39_india_age_binned_fixed/outputs/ts/ts_stats_ve50.json

Reuses exp39 HM checkpoint (no new HM needed).
Existing exp39 sir_results.jsonl and posterior.csv are NOT overwritten.

Run on zebra:
  tmux new-session -s india42 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 \\
     ~/ukvenv/bin/python experiments/42_india_ve_scored_ts/run.py \\
     2>&1 | tee experiments/42_india_ve_scored_ts/india42.log"
"""
import subprocess, pathlib, os, sys

HERE  = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]

EXP39_HM  = CALIB / 'experiments' / '39_india_age_binned_fixed' / 'outputs' / 'hm' \
             / 'maled_age_binned_titer_fixedagepsymp'
EXP39_TS  = CALIB / 'experiments' / '39_india_age_binned_fixed' / 'outputs' / 'ts'

cmd = [
    sys.executable, str(CALIB / 'trajectory_select.py'),
    '--model',          'age_binned',
    '--maternal',       'titer',
    '--fix-age-psymp',
    '--hm-dir',         str(EXP39_HM),
    '--out-dir',        str(EXP39_TS),
    '--n',              '3000',
    '--ve-target',      '0.50',
    '--ve-sigma',       '0.07',
    '--ve-take',        '0.74',
]
env = os.environ.copy()
env.update({'MALED_SITE': 'india', 'NEO_PRIME': '1'})
print('Running:', ' '.join(cmd))
subprocess.run(cmd, env=env, check=True)
