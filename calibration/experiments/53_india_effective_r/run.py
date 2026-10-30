"""Exp 53 -- India Vellore: effective R after the initial wave, for the 3 parameter
points already simulated in exp50/51 (no new simulations -- reuses their peak_frac
and R0 already established). See README.md / SUMMARY.md.

R0 (naive, fully susceptible): n_contacts=7, transmission ~90% suppressed during the
~8-day asymptomatic phase, full strength during the ~5-day symptomatic phase (see
exp51 SUMMARY.md obs 2; rotasim/rotavirus.py:238,242).

Re_after_wave (mean-field, first-order): right after a single fast synchronized wave,
almost everyone who was infected has exactly ONE prior infection (order=1) and
everyone else is still fully naive (rel_sus=1). So the population-average relative
susceptibility right after the wave is
    <rel_sus> = (1 - peak_frac) * 1.0 + peak_frac * sus_after_1
and Re_after_wave = R0 * <rel_sus>.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
FIG_DIR = HERE / 'figures'
OUT_DIR = HERE / 'outputs'

N_CONTACTS, DUR_SYMP, DUR_ASYMP = 7, 5.0, 8.0
def r0_of(beta):
    return N_CONTACTS * ((1 - np.exp(-beta)) * DUR_SYMP + (1 - np.exp(-0.1 * beta)) * DUR_ASYMP)

nroy = pd.read_csv(CALIB / 'experiments/39_india_age_binned_fixed/outputs/ts/nroy_draw.csv')
nroy['base_beta'] = np.exp(nroy['log_base_beta'])

POINTS = [2320, 1720, 1072]

# peak_frac at N=400k, averaged over the 10 seeds (exp51 outputs)
import json
rows = ([json.loads(l) for l in open(CALIB / 'experiments/51_india_population_size/outputs/popsize_results.jsonl')]
        + [json.loads(l) for l in open(CALIB / 'experiments/51_india_population_size/outputs/popsize_results_1072.jsonl')])
df51 = pd.DataFrame(rows)
peak_frac_400k = df51[df51.n_agents == 400_000].groupby('orig_idx')['peak_frac'].mean()
frac_extinct_400k = df51[df51.n_agents == 400_000].groupby('orig_idx')['extinct'].mean()

results = []
for idx in POINTS:
    row = nroy.loc[idx]
    beta = float(row['base_beta'])
    s1 = float(row['sus_after_1'])
    r0 = r0_of(beta)
    pf = float(peak_frac_400k.loc[idx])
    avg_rel_sus = (1 - pf) * 1.0 + pf * s1
    re_after = r0 * avg_rel_sus
    fe = float(frac_extinct_400k.loc[idx])
    results.append(dict(orig_idx=idx, base_beta=beta, R0=r0, sus_after_1=s1,
                         peak_frac_400k=pf, avg_rel_sus_after_wave=avg_rel_sus,
                         Re_after_wave=re_after, frac_extinct_400k=fe))

res = pd.DataFrame(results)
res.to_csv(OUT_DIR / 're_after_wave.csv', index=False)
print(res.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
ax[0].bar(range(len(res)), res['R0'], width=0.35, label='R0 (naive)', color='#4C72B0')
ax[0].bar([x + 0.35 for x in range(len(res))], res['Re_after_wave'], width=0.35,
          label='Re (after 1st wave)', color='#DD8452')
ax[0].axhline(1.0, color='gray', ls='--', lw=1)
ax[0].set_xticks([x + 0.175 for x in range(len(res))])
ax[0].set_xticklabels([f"idx={r.orig_idx}\nbeta={r.base_beta:.3f}" for r in res.itertuples()])
ax[0].set_ylabel('reproduction number')
ax[0].set_title('R0 vs Re after the initial wave')
ax[0].legend(fontsize=8)

ax[1].scatter(res['Re_after_wave'], res['frac_extinct_400k'], s=80, color='#55A868')
for r in res.itertuples():
    ax[1].annotate(f"idx={r.orig_idx}", (r.Re_after_wave, r.frac_extinct_400k),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[1].set_xlabel('Re after initial wave (mean-field)')
ax[1].set_ylabel('empirical fraction extinct, N=400k')
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_title('Mean-field Re (all >1) vs actual extinction risk')

plt.tight_layout()
plt.savefig(FIG_DIR / 're_after_wave.png', dpi=150)
print(f"\nSaved figures/re_after_wave.png")
