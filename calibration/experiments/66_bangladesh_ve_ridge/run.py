"""Exp 66 -- Bangladesh direct-VE ridge analysis, both models. Mirrors
India's exp60 exactly (same vaccine mechanism, same Rotavac schedule, same
take sweep, same coverage=1.0 direct-effect isolation) but using exp64's
age_binned and exp65's infnum 6-seed pools for Bangladesh. See README.md.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))

from ode_model import ODEParams
from ode_model_age import simulate_age, N_CONTACTS, IDX_IS, IDX_IA, N_STATE, N_AGE_BINS
import age_binned_cohort_model as age_vax
import infnum_cohort_model as inf_vax

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

BIRTH_RATE, DEATH_RATE = 19, 6   # Bangladesh demographics
DOSE_AGES_DAYS = [42.0, 70.0, 98.0]  # Rotavac: 6, 10, 14 weeks -- India's real schedule, reused per AK
TAKE_VALUES = [0.6, 0.74, 0.9]


def foi_eq_for(base_beta, sus_after_1, sus_r2, sus_r3, log_titer_median, titer_gsd,
               titer_half_life_days, hill_slope, maternal_efficacy):
    p = ODEParams(base_beta=base_beta, sus_after_1=sus_after_1, sus_r2=sus_r2, sus_r3=sus_r3,
                  maternal_titer_median=np.exp(log_titer_median), maternal_titer_gsd=titer_gsd,
                  maternal_titer_half_life_days=titer_half_life_days, maternal_hill_slope=hill_slope,
                  maternal_immunity_efficacy=maternal_efficacy,
                  birth_rate_per_1000=BIRTH_RATE, death_rate_per_1000=DEATH_RATE)
    sol = simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    Y = sol.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum(); IA_total = Y[:, IDX_IA].sum()
    foi_eq = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total
    return foi_eq, p


def rate_6_11m_take(cohort_mod, cp, dose, take):
    sol = cohort_mod.simulate_cohort(cp, max_age_months=14, n_eval=400,
                                      dose_ages_days=DOSE_AGES_DAYS if dose else None,
                                      coverage=1.0 if dose else 0.0, take=take if dose else 0.0)
    age_m, cum_symp = cohort_mod.cum_symp_curve(sol)
    c6 = np.interp(6.0, age_m, cum_symp)
    c12 = np.interp(12.0, age_m, cum_symp)
    return c12 - c6


results = []
for model, exp_dir, cohort_mod in [
    ('age_binned', '64_bangladesh_age_binned_ode', age_vax),
    ('infnum', '65_bangladesh_infnum_ode', inf_vax),
]:
    seeds = [json.loads(l) for l in open(CALIB / f'experiments/{exp_dir}/outputs/seed_runs.jsonl')]
    seeds = sorted(seeds, key=lambda r: -r['best_logL'])
    print(f"\n=== {model} ===")
    for row in seeds:
        bp = row['best_params']
        foi_eq, p = foi_eq_for(np.exp(bp['log_base_beta']), bp['sus_after_1'], bp['sus_r2'], bp['sus_r3'],
                                 bp['log_titer_median'], bp['titer_gsd'], bp['titer_half_life_days'],
                                 bp['hill_slope'], bp['maternal_efficacy'])
        if model == 'age_binned':
            p_symp = {'<6m': bp['p_symp_age_0_6'], '6-11m': bp['p_symp_age_6_11'], '12plus': bp['p_symp_age_12plus']}
            cp = cohort_mod.CohortParams(foi_eq=foi_eq, sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                                          p_symp_age=p_symp)
        else:
            p_symp = [bp['p_symp_order1'], bp['p_symp_order2'], bp['p_symp_order3plus']]
            cp = cohort_mod.CohortParams(foi_eq=foi_eq, sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                                          p_symp_order=p_symp)

        unvax_rate = rate_6_11m_take(cohort_mod, cp, dose=False, take=0.0)
        print(f"  seed={row['seed']:>10} logL={row['best_logL']:.2f} base_beta={p.base_beta:.4f} "
              f"unvax_rate={unvax_rate:.5f}")
        for take in TAKE_VALUES:
            vax_rate = rate_6_11m_take(cohort_mod, cp, dose=True, take=take)
            ve = 1.0 - vax_rate / unvax_rate if unvax_rate > 0 else float('nan')
            results.append(dict(model=model, seed=row['seed'], logL=row['best_logL'], base_beta=p.base_beta,
                                 sus_r2=bp['sus_r2'], sus_r3=bp['sus_r3'], take=take,
                                 unvax_rate=unvax_rate, vax_rate=vax_rate, direct_ve=ve))
            print(f"    take={take}: vax_rate={vax_rate:.5f}  direct_VE={ve:.3f}")

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 've_by_draw.csv', index=False)

print("\n=== Direct VE (6-11m) spread across the 6 ridge draws, by model x take ===")
summary = df.groupby(['model', 'take'])['direct_ve'].agg(['median', 'min', 'max'])
summary['spread_pts'] = (summary['max'] - summary['min']) * 100
print(summary.to_string())
summary.to_csv(OUT_DIR / 've_summary.csv')

# ---- Figure: direct VE spread across draws, by model, by take ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(TAKE_VALUES)))
for ax, model in zip(axes, ['age_binned', 'infnum']):
    sub_model = df[df.model == model]
    for take, c in zip(TAKE_VALUES, colors):
        sub = sub_model[sub_model['take'] == take].sort_values('logL', ascending=False)  # .take collides with DataFrame.take()
        ax.plot(range(1, len(sub) + 1), sub['direct_ve'] * 100, 'o-', color=c, label=f'take={take}')
    ax.set_xlabel('ridge draw (ranked by logL)')
    ax.set_title(f'{model}')
axes[0].set_ylabel('Direct VE, 6-11m (%)')
axes[-1].legend(fontsize=8)
plt.suptitle("Exp 66: Bangladesh direct VE across each model's 6 ridge draws")
plt.tight_layout()
plt.savefig(FIG_DIR / 've_ridge_both_models.png', dpi=140)
print("\nSaved figures/ve_ridge_both_models.png")
