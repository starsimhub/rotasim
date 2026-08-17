"""Exp 56 -- India Vellore: age-structured ODE equilibrium across 10 high-likelihood
parameter draws, for a confidence band on the equilibrium immune/IR-by-age state
rather than a single point (exp55 used only the single MLE). Runs entirely locally
via the ODE (fast -- no ABM sims), while exp52's real HM/TS sims continue on zebra.

Point selection: top 10 by logL from exp47's ALREADY-COMPLETE trajectory-selection
scoring (experiments/47_india_age_psymp_interp/outputs/ts/sir_results.jsonl, 3000
draws each already scored against the real composite likelihood) -- not the
collapsed posterior.csv (ESS=1.02, only 6 unique rows), which has too little
diversity for a spread. exp52's own TS run (survival-vote-weighted) is still in
progress; this uses the best complete dataset available now, per AK.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
from ode_model import ODEParams
from ode_model_age import simulate_age, summarize_by_age, AGE_BIN_LABELS, IDX_S, IDX_IS, IDX_IA, N_CONTACTS
import process_incidence_maled as P

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
N_POINTS = 10

EXP47 = CALIB / 'experiments/47_india_age_psymp_interp/outputs/ts'
recs = [json.loads(l) for l in open(EXP47 / 'sir_results.jsonl')]
sir = pd.DataFrame(recs)
nroy = pd.read_csv(EXP47 / 'nroy_draw.csv')
nroy['base_beta'] = np.exp(nroy['log_base_beta'])

top = sir.sort_values('logL', ascending=False).head(N_POINTS)
print(f"Top {N_POINTS} by logL from exp47's TS scoring:")
print(top[['idx', 'logL']].to_string(index=False))


def params_from_row(row):
    return ODEParams(
        base_beta=float(row['base_beta']),
        sus_after_1=float(row['sus_after_1']), sus_r2=float(row['sus_r2']), sus_r3=float(row['sus_r3']),
        maternal_titer_median=float(np.exp(row['log_titer_median'])),
        maternal_titer_gsd=float(row['titer_gsd']),
        maternal_titer_half_life_days=float(row['titer_half_life_days']),
        maternal_hill_slope=float(row['hill_slope']),
        maternal_immunity_efficacy=float(row['maternal_efficacy']),
        birth_rate_per_1000=16.0, death_rate_per_1000=7.0,
    )


DET_SYMP = 0.80 * 0.85
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85
DET_ASYMP = {'<6m': (SHED_DAYS / MONTHLY_D) * EIA, '6-11m': (SHED_DAYS / MONTHLY_D) * EIA,
             '12-23m': (SHED_DAYS / QUARTERLY_D) * EIA, '24-35m': (SHED_DAYS / QUARTERLY_D) * EIA}

all_rows = []
for rank, (_, sir_row) in enumerate(top.iterrows()):
    idx = int(sir_row['idx'])
    row = nroy.loc[idx]
    p = params_from_row(row)
    sol = simulate_age(p, n_agents=40_000, years=60, n_eval=800)
    rows = summarize_by_age(sol, p)
    psymp_age = {'<6m': float(row['p_symp_age_0_6']), '6-11m': float(row['p_symp_age_6_11']),
                 '12-23m': float(row['p_symp_age_12plus']), '24-35m': float(row['p_symp_age_12plus'])}
    for r in rows:
        b = r['age_bin']
        if b != '36m+':
            det_prob = psymp_age[b] * DET_SYMP + (1 - psymp_age[b]) * DET_ASYMP[b]
            r['ir_all_detected_per_100pm'] = r['ir_all_per_100pm'] * det_prob
        else:
            r['ir_all_detected_per_100pm'] = float('nan')
        r['idx'] = idx; r['rank'] = rank; r['logL'] = float(sir_row['logL'])
        r['base_beta'] = p.base_beta
    all_rows.extend(rows)
    print(f"  [{rank+1}/{N_POINTS}] idx={idx} base_beta={p.base_beta:.4f} logL={sir_row['logL']:.1f} done")

df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / 'ode_by_draw.csv', index=False)

# ---- Summary: median + range across the 10 draws, per age bin ----
metrics = ['pct_maternal', 'pct_susceptible_order0', 'pct_susceptible_order1',
           'pct_susceptible_order2', 'pct_susceptible_order3plus', 'pct_currently_infected',
           'pct_recently_immune', 'mean_prior_infections', 'ir_all_detected_per_100pm']
summary = df.groupby('age_bin')[metrics].agg(['median', 'min', 'max'])
summary.to_csv(OUT_DIR / 'summary_by_age.csv')
pd.set_option('display.width', 200)
print("\n=== Median [min, max] across 10 high-likelihood draws ===")
print(summary.to_string())

# ---- Figure 1: IR-by-age with model spread (median/range across 10 draws) vs
# target with Poisson CI ----
def poisson_exact_ci(cases, pt, scale=100.0, alpha=0.05):
    lo = 0.5 * chi2.ppf(alpha / 2, 2 * cases) if cases > 0 else 0.0
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (cases + 1))
    return lo / pt * scale, hi / pt * scale

target = P.load_ir_all_targets('india')
age_order = ['<6m', '6-11m', '12-23m', '24-35m']
tgt_labels_ordered = ['<6 m', '6-11 m', '12-23 m', '24-35 m']

model_med, model_lo, model_hi = [], [], []
for b in age_order:
    vals = df[df.age_bin == b]['ir_all_detected_per_100pm']
    model_med.append(vals.median()); model_lo.append(vals.min()); model_hi.append(vals.max())

tgt_vals, tgt_err_lo, tgt_err_hi = [], [], []
for t in tgt_labels_ordered:
    cases = int(target.loc[t, 'cases']); pt = float(target.loc[t, 'PT']); ir = target.loc[t, 'IR']
    lo, hi = poisson_exact_ci(cases, pt)
    tgt_vals.append(ir); tgt_err_lo.append(ir - lo); tgt_err_hi.append(hi - ir)

x = np.arange(4)
fig, ax = plt.subplots(figsize=(7.5, 5))
model_err = [np.array(model_med) - np.array(model_lo), np.array(model_hi) - np.array(model_med)]
ax.bar(x - 0.175, model_med, width=0.35, label='ODE, detection-adjusted (median, range across 10 draws)',
       color='#55A868', yerr=model_err, capsize=4, ecolor='black', error_kw=dict(elinewidth=1.2))
ax.bar(x + 0.175, tgt_vals, width=0.35, label='real MAL-ED target (detected, 95% exact Poisson CI)',
       color='#DD8452', yerr=[tgt_err_lo, tgt_err_hi], capsize=4, ecolor='black', error_kw=dict(elinewidth=1.2))
ax.set_xticks(x); ax.set_xticklabels(tgt_labels_ordered)
ax.set_ylabel('all-infection IR (per 100 person-months)')
ax.set_title(f'Detection-adjusted ODE (10 high-likelihood draws) vs real target\n(24-35m: only {int(target.loc["24-35 m","PT"])} person-months -- low power)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'ir_by_age_ci.png', dpi=150)
print("\nSaved figures/ir_by_age_ci.png")

# ---- Figure 2: mean prior infections by age, median + range across draws ----
fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
meds = df.groupby('age_bin')['mean_prior_infections'].median().reindex(AGE_BIN_LABELS)
los = df.groupby('age_bin')['mean_prior_infections'].min().reindex(AGE_BIN_LABELS)
his = df.groupby('age_bin')['mean_prior_infections'].max().reindex(AGE_BIN_LABELS)
ax2.plot(AGE_BIN_LABELS, meds, 'o-', color='#55A868', label='median (10 draws)')
ax2.fill_between(AGE_BIN_LABELS, los, his, color='#55A868', alpha=0.25, label='range (10 draws)')
ax2.set_ylabel('mean # prior infections')
ax2.set_title('Equilibrium mean infection count by age\n(median + range across 10 high-likelihood draws)')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / 'mean_prior_infections_ci.png', dpi=150)
print("Saved figures/mean_prior_infections_ci.png")
