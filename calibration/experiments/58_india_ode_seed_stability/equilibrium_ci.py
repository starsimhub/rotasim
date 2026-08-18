"""Exp 58 part 2 -- redo exp56's equilibrium-uncertainty analysis (age-structured
compartment fractions across 10 high-likelihood draws), but sourcing the draws
from THIS experiment's directly-optimized ODE MLE pool
(pooled_candidates.jsonl, top-K final-population members from 6
differential_evolution seeds) instead of exp56's ABM-HM-posterior draws
(exp47's trajectory-selection scoring). Question: does the equilibrium CI
narrow using directly-optimized draws vs the ABM-HM posterior?

Reuses exp57's params_from_vec (via explicit file-path import, single-process
here so no multiprocessing-pickling concern) to build ODEParams from each
pooled candidate's parameter dict, exactly as exp56 built them from exp47's
NROY rows.
"""
import sys, pathlib, json, importlib.util
import numpy as np, pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP57 = HERE.parents[0] / '57_india_ode_direct_fit'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(EXP57))

_spec = importlib.util.spec_from_file_location("exp57_run", str(EXP57 / "run.py"))
exp57 = importlib.util.module_from_spec(_spec)
sys.modules["exp57_run"] = exp57
_spec.loader.exec_module(exp57)

from ode_model_age import simulate_age, summarize_by_age, AGE_BIN_LABELS
import process_incidence_maled as P

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
FIG_DIR.mkdir(exist_ok=True)
N_POINTS = 10

recs = [json.loads(l) for l in open(OUT_DIR / 'pooled_candidates.jsonl')]
df_pool = pd.DataFrame(recs)
top = df_pool.sort_values('logL', ascending=False).head(N_POINTS).reset_index(drop=True)
print(f"Top {N_POINTS} pooled candidates (across {df_pool['seed'].nunique()} seeds):")
print(top[['seed', 'rank', 'logL']].to_string(index=False))

DET_SYMP = 0.80 * 0.85
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85
DET_ASYMP = {'<6m': (SHED_DAYS / MONTHLY_D) * EIA, '6-11m': (SHED_DAYS / MONTHLY_D) * EIA,
             '12-23m': (SHED_DAYS / QUARTERLY_D) * EIA, '24-35m': (SHED_DAYS / QUARTERLY_D) * EIA}

all_rows = []
for rank, row in top.iterrows():
    x = [row['params'][name] for name in exp57.PARAM_NAMES]
    p, p_symp_age3 = exp57.params_from_vec(x)
    psymp_age = {'<6m': p_symp_age3['<6m'], '6-11m': p_symp_age3['6-11m'],
                 '12-23m': p_symp_age3['12plus'], '24-35m': p_symp_age3['12plus']}
    sol = simulate_age(p, n_agents=40_000, years=60, n_eval=800)
    rows = summarize_by_age(sol, p)
    for r in rows:
        b = r['age_bin']
        if b != '36m+':
            det_prob = psymp_age[b] * DET_SYMP + (1 - psymp_age[b]) * DET_ASYMP[b]
            r['ir_all_detected_per_100pm'] = r['ir_all_per_100pm'] * det_prob
        else:
            r['ir_all_detected_per_100pm'] = float('nan')
        r['rank'] = rank; r['seed'] = row['seed']; r['logL'] = row['logL']; r['base_beta'] = p.base_beta
    all_rows.extend(rows)
    print(f"  [{rank+1}/{N_POINTS}] seed={row['seed']} base_beta={p.base_beta:.4f} logL={row['logL']:.2f} done")

df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / 'ode_by_draw.csv', index=False)

metrics = ['pct_maternal', 'pct_susceptible_order0', 'pct_susceptible_order1',
           'pct_susceptible_order2', 'pct_susceptible_order3plus', 'pct_currently_infected',
           'pct_recently_immune', 'mean_prior_infections', 'ir_all_detected_per_100pm']
summary = df.groupby('age_bin')[metrics].agg(['median', 'min', 'max'])
summary.to_csv(OUT_DIR / 'summary_by_age.csv')
pd.set_option('display.width', 200)
print("\n=== Median [min, max] across 10 directly-optimized draws (exp58) ===")
print(summary.to_string())

comp_cols = ['pct_maternal', 'pct_susceptible_order0', 'pct_susceptible_order1',
             'pct_susceptible_order2', 'pct_susceptible_order3plus',
             'pct_currently_infected', 'pct_recently_immune']
comp_labels = ['maternally\nprotected', 'susceptible\n(naive)', 'susceptible\n(1 prior inf)',
               'susceptible\n(2 prior inf)', 'susceptible\n(3+ prior inf)',
               'currently\ninfected', 'recently immune\n(post-recovery)']
comp_colors = plt.cm.tab10(np.linspace(0, 1, len(comp_cols)))

fig0, axes0 = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for ax, age_bin in zip(axes0, AGE_BIN_LABELS):
    sub = df[df.age_bin == age_bin]
    meds = [sub[c].median() for c in comp_cols]
    los = [meds[i] - sub[c].min() for i, c in enumerate(comp_cols)]
    his = [sub[c].max() - meds[i] for i, c in enumerate(comp_cols)]
    ax.bar(range(len(comp_cols)), meds, yerr=[los, his], color=comp_colors,
           capsize=4, ecolor='black', error_kw=dict(elinewidth=1.2))
    ax.set_xticks(range(len(comp_cols))); ax.set_xticklabels(comp_labels, rotation=60, ha='right', fontsize=8)
    ax.set_title(age_bin)
    ax.set_ylim(0, 100)
axes0[0].set_ylabel('% of age-bin population\n(median, range across 10 directly-optimized draws)')
fig0.suptitle('Exp58: equilibrium compartment composition by age -- directly-optimized ODE draws', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'compartment_fractions_ci.png', dpi=150, bbox_inches='tight')
print("Saved figures/compartment_fractions_ci.png")

# ---- width comparison vs exp56's original (same metrics, same age bins) ----
exp56_summary_path = HERE.parents[0] / '56_india_ode_posterior_ci' / 'outputs' / 'summary_by_age.csv'
if exp56_summary_path.exists():
    exp56_summary = pd.read_csv(exp56_summary_path, header=[0, 1], index_col=0)
    print("\n=== Range width comparison: exp56 (HM-posterior draws) vs exp58 (directly-optimized draws) ===")
    for age_bin in AGE_BIN_LABELS:
        if age_bin not in exp56_summary.index or age_bin not in summary.index:
            continue
        for m in ['pct_susceptible_order2', 'pct_susceptible_order3plus']:
            try:
                w56 = exp56_summary.loc[age_bin, (m, 'max')] - exp56_summary.loc[age_bin, (m, 'min')]
                w58 = summary.loc[age_bin, (m, 'max')] - summary.loc[age_bin, (m, 'min')]
                print(f"  {age_bin:8s} {m:28s} exp56 width={w56:6.2f}  exp58 width={w58:6.2f}")
            except KeyError:
                pass
else:
    print(f"\n(exp56 summary not found at {exp56_summary_path}, skipping width comparison)")
