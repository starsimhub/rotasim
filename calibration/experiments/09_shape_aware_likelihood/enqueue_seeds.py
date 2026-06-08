"""
Enqueue starting trials for the exp-09 peaked-age Poisson calibration (see README.md).

Under the Poisson objective, level matters linearly (not on a log scale), so exp-07's
peaked seed at base_beta=0.20 -- which overshoots incidence ~3x -- is now a *bad*
starting point (its total-count deviance is large). We still want to hand TPE the
peaked SHAPE, so we enqueue the peaked age-betas at a ladder of lower base_beta values
to bracket the level. TPE then refines shape+level from a good shape foothold.

Run from the calibration dir:
  cd /path/to/calibration && python experiments/09_shape_aware_likelihood/enqueue_seeds.py
"""
import optuna

STUDY = 'rota_maled_bangladesh_erlang6_poisson'     # age_only + erlang6 + poisson fit-target
DB = 'rota_maled_bangladesh_erlang6_poisson.db'

# Peaked shape (exp-05 'sharp_peak_9mo'), seeded across a base_beta ladder so the
# optimizer starts with the right shape and brackets the level under Poisson.
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
print(f'Enqueued {len(BASE_BETA_LADDER)} peaked seeds (base_beta ladder {BASE_BETA_LADDER}) '
      f'into {STUDY}. Existing trials: {len(study.trials)}')
