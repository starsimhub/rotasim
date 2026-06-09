"""
Seed the exp-12 peaked-age + titer-maternal calibration with the exp-09/10 peaked
betas + DK-titer config, so the peaked basin has a foothold. See README.md.

  cd /path/to/calibration && python experiments/12_age_titer/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_titer_poisson'
DB = 'rota_maled_bangladesh_titer_poisson.db'

SEED = dict(
    base_beta=0.12,
    sus_after_3plus=0.25, sus_after_2=0.40, sus_after_1=0.60,
    maternal_immunity_efficacy=0.95,
    maternal_titer_median=41.0, maternal_titer_gsd=2.6,
    maternal_titer_half_life_days=68.0, maternal_hill_slope=6.2,
    beta0=-1.30, beta1=-0.06, beta2=-0.01,
)

study = optuna.create_study(
    study_name=STUDY, direction='minimize',
    storage=f'sqlite:///{DB}', load_if_exists=True,
)
study.enqueue_trial(SEED, skip_if_exists=True)
print(f'Enqueued peaked+titer seed into {DB}::{STUDY}. Existing trials: {len(study.trials)}')
