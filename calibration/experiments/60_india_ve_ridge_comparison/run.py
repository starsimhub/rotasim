"""Exp 60 -- direct-VE sensitivity across exp58's fitted ridge (age_binned, the
confirmed winning structure per exp59). For each of exp58's top-10 pooled
high-likelihood draws, run the birth-cohort ODE (this dir's cohort_model.py,
extended with the cum_symp accumulator + dose-based vaccination) twice --
unvaccinated and with Rotavac's 3-dose schedule (6/10/14 weeks) at low
coverage (0.05, isolating the direct/individual effect, matching exp41 and
Nair et al.'s test-negative estimand) -- across a small sweep of `take`
values, and compare achieved direct VE in the 6-11m window (Nair et al.
52-59%) across the ridge.
"""
import sys, pathlib, json, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP58 = HERE.parents[0] / '58_india_ode_seed_stability'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))

from ode_model_age import simulate_age, N_CONTACTS, IDX_IS, IDX_IA, N_STATE, N_AGE_BINS
from cohort_model import CohortParams, simulate_cohort, cum_symp_curve

# exp57's params_from_vec (age_binned parameterization) -- load by explicit
# path since ode_model_age.py's own sys.path.insert would otherwise shadow it
# (both 57's and 58's dirs are irrelevant here, but the same collision risk
# applies to any dir sharing a filename with 54/55's inserted paths).
EXP57 = HERE.parents[0] / '57_india_ode_direct_fit'
_spec = importlib.util.spec_from_file_location("exp57_run", str(EXP57 / "run.py"))
exp57 = importlib.util.module_from_spec(_spec)
sys.modules["exp57_run"] = exp57
_spec.loader.exec_module(exp57)

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

N_DRAWS = 10
DOSE_AGES_DAYS = [42.0, 70.0, 98.0]  # Rotavac: 6, 10, 14 weeks
COVERAGE = 0.05                       # low, isolates direct effect (exp41 design)
TAKE_VALUES = [0.6, 0.74, 0.9]

pool = [json.loads(l) for l in open(EXP58 / 'outputs' / 'pooled_candidates.jsonl')]
df_pool = pd.DataFrame(pool)
top = df_pool.sort_values('logL', ascending=False).head(N_DRAWS).reset_index(drop=True)
print(f"Top {N_DRAWS} pooled age_binned candidates (across {df_pool['seed'].nunique()} seeds):")
print(top[['seed', 'rank', 'logL']].to_string(index=False))


def foi_and_params(params_dict):
    x = [params_dict[name] for name in exp57.PARAM_NAMES]
    p, p_symp_age = exp57.params_from_vec(x)
    sol_age = simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    Y = sol_age.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum(); IA_total = Y[:, IDX_IA].sum()
    foi_eq = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total
    return foi_eq, p, p_symp_age


def symp_rate_6_11m(cp: CohortParams, dose=False, take=0.0):
    # coverage's role is to justify using the UNVACCINATED population's foi_eq
    # for both arms (low enough that herd effects are negligible, exp41's
    # direct-effect isolation design) -- it must NOT dilute the vaccinated
    # arm's own dose-transfer fraction, which would mix 95% "never vaccinated"
    # mass into what's supposed to be a pure vaccinated-individual comparison.
    # Within the simulated vaccinated arm, coverage=1.0 (everyone in this arm
    # received the dose by construction); `take` alone sets per-dose success.
    sol = simulate_cohort(cp, max_age_months=14, n_eval=400,
                           dose_ages_days=DOSE_AGES_DAYS if dose else None,
                           coverage=1.0 if dose else 0.0, take=take)
    age_m, cum_symp = cum_symp_curve(sol)
    c6 = np.interp(6.0, age_m, cum_symp)
    c12 = np.interp(12.0, age_m, cum_symp)
    return c12 - c6  # expected symptomatic-detected infections per unit cohort mass, ages 6-12mo


results = []
for i, row in top.iterrows():
    foi_eq, p, p_symp_age = foi_and_params(row['params'])
    cp = CohortParams(foi_eq=foi_eq, sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_age=p_symp_age)
    unvax_rate = symp_rate_6_11m(cp, dose=False)
    print(f"[{i+1}/{N_DRAWS}] seed={row['seed']} logL={row['logL']:.2f} "
          f"base_beta={p.base_beta:.4f} unvax_6-11m_rate={unvax_rate:.5f}")
    for take in TAKE_VALUES:
        vax_rate = symp_rate_6_11m(cp, dose=True, take=take)
        ve = 1.0 - vax_rate / unvax_rate if unvax_rate > 0 else float('nan')
        results.append(dict(seed=row['seed'], rank=row['rank'], logL=row['logL'],
                             base_beta=p.base_beta, sus_r2=row['params']['sus_r2'],
                             sus_r3=row['params']['sus_r3'],
                             take=take, unvax_rate=unvax_rate, vax_rate=vax_rate, direct_ve=ve))
        print(f"    take={take}: vax_rate={vax_rate:.5f}  direct_VE={ve:.3f}")

res = pd.DataFrame(results)
res.to_csv(OUT_DIR / 've_by_draw.csv', index=False)

print("\n=== Direct VE (6-11m) spread across the 10 ridge draws, by take ===")
summary = res.groupby('take')['direct_ve'].agg(['median', 'min', 'max', 'std'])
print(summary.to_string())
summary.to_csv(OUT_DIR / 've_summary_by_take.csv')

# ---- Figure: direct VE spread across draws, by take, vs Nair et al. anchor ----
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(TAKE_VALUES)))
for take, c in zip(TAKE_VALUES, colors):
    sub = res[res['take'] == take].sort_values('logL', ascending=False)  # res.take collides with DataFrame.take()
    ax.plot(range(1, len(sub) + 1), sub['direct_ve'] * 100, 'o-', color=c, label=f'take={take}')
ax.axhspan(52, 59, color='#888888', alpha=0.25, label='Nair et al. 6-11m target (52-59%)')
ax.set_xlabel('ridge draw (ranked by logL, all within ~0.5 log-units of each other)')
ax.set_ylabel('Direct VE, 6-11m (%)')
ax.set_title('exp60: direct VE across age_binned\'s fitted ridge\n(10 draws, all equally good natural-history fits)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 've_ridge_sensitivity.png', dpi=140)
print("\nSaved figures/ve_ridge_sensitivity.png")
