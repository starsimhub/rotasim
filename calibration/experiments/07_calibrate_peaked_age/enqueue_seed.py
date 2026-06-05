"""
Enqueue the exp-05 peaked age-symptom configuration as a starting trial for the
exp-07 calibration (see README.md). This is a *seed*, not a constraint: Optuna
evaluates it first, then is free to move away from it. The point is to keep TPE
from settling into exp-02's monotone-declining basin without ever trying a peak.

Run from the calibration dir (so the sqlite path matches calibrate_maled.py):
  cd /path/to/calibration && python experiments/07_calibrate_peaked_age/enqueue_seed.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_erlang6'        # age_only + erlang6 + joint fit-target
DB = 'rota_maled_bangladesh_erlang6.db'

# exp-05 'sharp_peak_9mo' config that reproduced the MAL-ED shape.
SEED = dict(
    base_beta=0.20,
    sus_after_3plus=0.25, sus_after_2=0.40, sus_after_1=0.60,
    maternal_immunity_efficacy=0.95,
    maternal_immunity_mean_duration_days=200.0,
    beta0=-1.30, beta1=-0.06, beta2=-0.01,
)

study = optuna.create_study(
    study_name=STUDY, direction='minimize',
    storage=f'sqlite:///{DB}', load_if_exists=True,
)
study.enqueue_trial(SEED, skip_if_exists=True)
print(f'Enqueued peaked seed into {STUDY}. Existing trials: {len(study.trials)}')
