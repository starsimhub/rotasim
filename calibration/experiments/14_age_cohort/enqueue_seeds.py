"""
Seed the exp-14 AGE-symptom + titer-maternal COHORT calibration. To give age+titer its
best shot (exp 12 showed strong titer crushes the age model's already-mild <6m), seed
with a WEAK titer (low efficacy / short half-life / gentle Hill -> minimal extra <6m
suppression) and the exp-10 peaked age curve -- i.e. let the age curve do the work and
titer be near-off. If the cohort+repeat objective can hold a good fit here, exp 13 + exp 14
form a clean same-maternal-structure (titer) matched pair. See ../14_age_cohort/README.md.

  cd /path/to/calibration && python experiments/14_age_cohort/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_titer_cohort'
DB = 'rota_maled_bangladesh_titer_cohort.db'

SEED = dict(
    base_beta=0.10,
    sus_after_3plus=0.50, sus_after_2=0.80, sus_after_1=0.90,
    maternal_immunity_efficacy=0.50,        # weak
    maternal_titer_median=4.0, maternal_titer_gsd=1.4,
    maternal_titer_half_life_days=25.0, maternal_hill_slope=1.5,   # gentle -> minimal <6m crush
    beta0=-1.52, beta1=-0.12, beta2=-0.036,                        # exp-10 peaked age curve
)

study = optuna.create_study(
    study_name=STUDY, direction='minimize',
    storage=f'sqlite:///{DB}', load_if_exists=True,
)
study.enqueue_trial(SEED, skip_if_exists=True)
print(f'Enqueued weak-titer age seed into {DB}::{STUDY}. Existing trials: {len(study.trials)}')
