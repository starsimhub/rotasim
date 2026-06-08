"""
Enqueue the peaked-shape seed ladder for the exp-10 (corrected-denominator) re-run of
the peaked-age Poisson calibration. Same seeds as exp 09, into a FRESH study DB
(*_ptfix) so old- and new-denominator trials never mix. See README.md.

Run from the calibration dir:
  cd /path/to/calibration && python experiments/10_denominator_rerun/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_erlang6_poisson'           # same name, fresh DB file
DB = 'rota_maled_bangladesh_erlang6_poisson_ptfix.db'

BASE_BETA_LADDER = [0.20, 0.12, 0.08, 0.06]
SHAPE = dict(
    sus_after_3plus=0.25, sus_after_2=0.40, sus_after_1=0.60,
    maternal_immunity_efficacy=0.95,
    maternal_immunity_mean_duration_days=200.0,
    beta0=-1.30, beta1=-0.06, beta2=-0.01,
)

study = optuna.create_study(
    study_name=STUDY, direction='minimize',
    storage=f'sqlite:///{DB}', load_if_exists=True,
)
for bb in BASE_BETA_LADDER:
    study.enqueue_trial(dict(base_beta=bb, **SHAPE), skip_if_exists=True)
print(f'Enqueued {len(BASE_BETA_LADDER)} peaked seeds into {DB}::{STUDY}. '
      f'Existing trials: {len(study.trials)}')
