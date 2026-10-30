"""
Seed the exp-11 infection-number + titer-maternal calibration near D. Klein's exp-06
titer best, so the sharp-protection basin has a foothold under homogeneous mixing
(his values were found with reservoir mixing, so base_beta/sus may shift -- this just
hands TPE the titer *shape*). See README.md.

Run from the calibration dir:
  cd /path/to/calibration && python experiments/11_titer_maternal_infnum/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_infnum_titer_poisson'
DB = 'rota_maled_bangladesh_infnum_titer_poisson.db'

SEED = dict(
    base_beta=0.22,
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
print(f'Enqueued DK-titer seed into {DB}::{STUDY}. Existing trials: {len(study.trials)}')
