"""
Seed the exp-13 infection-number + titer-maternal COHORT calibration with a reasonable
foothold (titer shape near DK exp-06; base_beta on the low side since the repeat-fraction
constraint penalizes the hyperendemic regime). See README.md / BUILD_PLAN.md.

  cd /path/to/calibration && python experiments/13_cohort_emulation/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_infnum_titer_cohort'
DB = 'rota_maled_bangladesh_infnum_titer_cohort.db'

SEED = dict(
    base_beta=0.15,
    sus_after_3plus=0.50, sus_after_2=0.80, sus_after_1=0.90,
    maternal_immunity_efficacy=0.95,
    maternal_titer_median=41.0, maternal_titer_gsd=2.6,
    maternal_titer_half_life_days=68.0, maternal_hill_slope=6.2,
    p_symp_1=0.80, p_symp_2=0.30, p_symp_3plus=0.10,
)

study = optuna.create_study(
    study_name=STUDY, direction='minimize',
    storage=f'sqlite:///{DB}', load_if_exists=True,
)
study.enqueue_trial(SEED, skip_if_exists=True)
print(f'Enqueued cohort seed into {DB}::{STUDY}. Existing trials: {len(study.trials)}')
