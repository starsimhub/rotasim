"""Plot corrected FOI sweep (all-unique draws) vs original (n=60 sampled draws).
Two rows: top = infection-blocking, bottom = symptom-blocking (if available).
Left col = original foi_sweep.csv; right col = corrected foi_sweep_alluniq.csv.
Both models on each panel; conditional VE (<=0.95) shown as filled marker.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)

MODELS = {'age_binned': ('Age (binned)', '#2c7fb8'),
          'infnum':     ('Infection-number', '#c0392b')}
ELIM = 0.95


def summarise(df, cond=False):
    rows = []
    for fac, sub in df.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if cond:
            alive = alive[alive.ve_overall <= ELIM]
        if len(alive) == 0:
            continue
        rows.append((alive.age_of_inf.median(), alive.ve_overall.median(),
                     len(alive), len(sub), fac))
    return sorted(rows)


PANELS = [
    ('_foi_sweep.csv',        '_foi_sweep_alluniq.csv',
     'Original (n=60 weighted draws)', 'Corrected (all unique draws)'),
]

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (orig_suf, corr_suf, orig_title, corr_title) in zip(axs, PANELS):
    pass  # replaced below

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
titles = ['Original (n=60 weighted)', 'Corrected (all unique draws, n≈270–370)']

for col, (suf, title) in enumerate(zip(['_foi_sweep.csv', '_foi_sweep_alluniq.csv'], titles)):
    ax = axs[col]
    ax.set_title(title)
    for m, (lab, col_) in MODELS.items():
        f = OUT / f'{m}{suf}'
        if not f.exists():
            ax.annotate(f'{m} missing', (0.5, 0.5), xycoords='axes fraction', ha='center')
            continue
        d = pd.read_csv(f)
        rows_all  = summarise(d, cond=False)
        rows_cond = summarise(d, cond=True)

        ages_a = [r[0] for r in rows_all];  ves_a = [r[1] for r in rows_all]
        ages_c = [r[0] for r in rows_cond]; ves_c = [r[1] for r in rows_cond]

        ax.plot(ages_a, ves_a, '-', color=col_, lw=1.5, alpha=0.35, label=f'{lab} (uncond)')
        ax.plot(ages_c, ves_c, 'o--', color=col_, lw=2.0, alpha=0.9, label=f'{lab} (cond ≤0.95)')
        for a, v, na, nt, fac in rows_cond:
            rel = na / nt
            ax.scatter([a], [v], s=60 + 200*rel, color=col_,
                       alpha=0.3 + 0.6*rel, edgecolor=col_, zorder=5)

    for x, lab in [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]:
        ax.axvline(x, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (x, 0.02), rotation=90, va='bottom', ha='right',
                    fontsize=8, color='gray')
    ax.set_xlabel('simulated median age-of-infection (months)')
    ax.set_ylim(-0.05, 1.05)

axs[0].set_ylabel('achieved VE (IB MOA, resp=0.75)')
axs[0].legend(frameon=False, fontsize=8, loc='upper left')

fig.suptitle('exp20 corrected FOI sweep — all unique posterior draws vs original n=60\n'
             'solid=uncond, dashed=cond(VE≤0.95); larger/darker=more draws survived',
             fontsize=10)
fig.tight_layout()
out = FIG / 've_vs_age_of_infection_alluniq.png'
fig.savefig(out, dpi=140)
plt.close(fig)
print('wrote', out)

# Also print the gap table
print('\n=== Conditional VE gap (infnum - age_binned) by factor ===')
print(f'{"factor":>8}  {"age_orig":>10}  {"inf_orig":>10}  {"gap_orig":>10}  '
      f'{"age_corr":>10}  {"inf_corr":>10}  {"gap_corr":>10}')
for m, suf in [('age_binned', '_foi_sweep.csv'), ('infnum', '_foi_sweep.csv'),
               ('age_binned', '_foi_sweep_alluniq.csv'), ('infnum', '_foi_sweep_alluniq.csv')]:
    pass  # done below in combined loop

data = {}
for m in ['age_binned', 'infnum']:
    for tag in ['orig', 'corr']:
        suf = '_foi_sweep.csv' if tag == 'orig' else '_foi_sweep_alluniq.csv'
        f = OUT / f'{m}{suf}'
        if f.exists():
            d = pd.read_csv(f)
            data[(m, tag)] = d

for tag in ['orig', 'corr']:
    print(f'\n--- {tag} ---')
    ab = data.get(('age_binned', tag))
    inf = data.get(('infnum', tag))
    if ab is None or inf is None:
        print('  (data missing)')
        continue
    factors = sorted(set(ab.factor.unique()) & set(inf.factor.unique()))
    for fac in factors:
        def cond_med(df, f):
            s = df[(df.factor == f) & (df.novax_ir > 0.1) & (df.ve_overall <= 0.95)]
            return s.ve_overall.median() if len(s) else float('nan')
        va, vi = cond_med(ab, fac), cond_med(inf, fac)
        print(f'  factor={fac:.2f}  age={va:.3f}  infnum={vi:.3f}  gap={vi-va:+.3f}')
