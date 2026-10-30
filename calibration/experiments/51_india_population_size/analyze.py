"""Exp 51 -- combine 40k (exp50) + 100k/200k/400k (this exp's two runs) results
and produce the CCS-rescue summary figure + stats table.

Run on zebra after both run.py and run_1072.py have completed:
  ~/ukvenv/bin/python experiments/51_india_population_size/analyze.py
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXP50_PATH = CALIB / 'experiments/50_india_extinction_seed_dependence/outputs/extinction_seed_results.jsonl'
BETAS = {2320: 0.050, 1720: 0.076, 1072: 0.107}  # for labeling only

# Corrected R0: n_contacts=7 (RandomNet, resampled daily -- hm_calibrate.py/
# calibrate_maled.py), transmission ~90% suppressed during the ~8-day
# asymptomatic phase (rel_trans=0.1, rotasim/rotavirus.py:238,242), full
# strength during the ~5-day symptomatic phase. See SUMMARY.md obs 2.
N_CONTACTS, DUR_SYMP, DUR_ASYMP = 7, 5.0, 8.0
def r0_of(beta):
    return N_CONTACTS * ((1 - np.exp(-beta)) * DUR_SYMP + (1 - np.exp(-0.1 * beta)) * DUR_ASYMP)
R0S = {idx: r0_of(b) for idx, b in BETAS.items()}

rows = []
exp50 = pd.DataFrame([json.loads(l) for l in open(EXP50_PATH)])
exp50 = exp50[exp50['orig_idx'].isin([2320, 1720])].copy()
exp50['n_agents'] = 40_000
rows.append(exp50[['orig_idx', 'n_agents', 'seed', 'extinct', 'last_active_year']])

for fname in ['popsize_results.jsonl', 'popsize_results_1072.jsonl']:
    p = OUT_DIR / fname
    if p.exists():
        df = pd.DataFrame([json.loads(l) for l in open(p)])
        rows.append(df[['orig_idx', 'n_agents', 'seed', 'extinct', 'last_active_year']])

all_df = pd.concat(rows, ignore_index=True)
all_df['base_beta'] = all_df['orig_idx'].map(BETAS)
all_df.to_csv(OUT_DIR / 'combined_results.csv', index=False)

summary = all_df.groupby(['orig_idx', 'base_beta', 'n_agents']).agg(
    n=('extinct', 'size'), n_extinct=('extinct', 'sum'),
    median_last_active_year=('last_active_year', 'median'),
    max_last_active_year=('last_active_year', 'max'),
).reset_index()
summary['frac_extinct'] = summary['n_extinct'] / summary['n']
summary = summary.sort_values(['base_beta', 'n_agents'])
summary.to_csv(OUT_DIR / 'combined_summary.csv', index=False)
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = {2320: '#4C72B0', 1720: '#DD8452', 1072: '#55A868'}
for orig_idx, g in summary.groupby('orig_idx'):
    g = g.sort_values('n_agents')
    label = f"orig_idx={orig_idx} (beta={BETAS[orig_idx]:.3f}, R0={R0S[orig_idx]:.1f})"
    axes[0].plot(g['n_agents'], g['frac_extinct'], 'o-', color=colors[orig_idx], label=label)
    axes[1].plot(g['n_agents'], g['median_last_active_year'], 'o-', color=colors[orig_idx], label=label)

axes[0].set_xscale('log')
axes[0].set_xlabel('N agents')
axes[0].set_ylabel('fraction extinct (of 10 seeds)')
axes[0].set_ylim(-0.05, 1.05)
axes[0].set_title('Extinction rate vs population size')
axes[0].legend(fontsize=8)

axes[1].set_xscale('log')
axes[1].set_xlabel('N agents')
axes[1].set_ylabel('median last-active year')
axes[1].axhline(10.0, color='gray', ls='--', lw=1, label='full 10y window')
axes[1].set_title('Time-to-extinction vs population size')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'ccs_rescue_by_beta.png', dpi=150)
print(f"\nSaved figures/ccs_rescue_by_beta.png")

with open(OUT_DIR / 'stats_summary.txt', 'w') as f:
    f.write(summary.to_string(index=False))
print("Saved outputs/stats_summary.txt")
