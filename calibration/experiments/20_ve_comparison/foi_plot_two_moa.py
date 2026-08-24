"""Two-MOA corrected FOI sweep comparison.
Reads the all-unique-draw corrected CSVs (foi_sweep_alluniq*.csv) for both
infection-blocking and symptom-blocking MOAs, shows both models on each panel.

Run after both SB sweeps complete on covaguest:
  python foi_plot_two_moa.py
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT  = HERE / 'outputs'
FIG  = HERE / 'figures'; FIG.mkdir(exist_ok=True)

MODELS = {
    'infnum': ('Infection-number', '#c0392b'),
}
ELIM = 0.95
REF_AGES = [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]


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
    ('_foi_sweep_alluniq.csv',    'A. Infection-blocking (corrected)'),
    ('_foi_sweep_alluniq_sb.csv', 'B. Symptom-blocking (corrected)'),
]

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, (suf, title) in zip(axs, PANELS):
    any_data = False
    for m, (lab, col) in MODELS.items():
        fpath = OUT / f'{m}{suf}'
        if not fpath.exists():
            ax.annotate(f'{m} — missing\n({fpath.name})', (0.5, 0.5),
                        xycoords='axes fraction', ha='center', va='center',
                        fontsize=9, color='gray')
            continue
        d = pd.read_csv(fpath)
        rows_all  = summarise(d, cond=False)
        rows_cond = summarise(d, cond=True)
        if not rows_all:
            continue
        any_data = True

        ages_a = [r[0] for r in rows_all];  ves_a = [r[1] for r in rows_all]
        ages_c = [r[0] for r in rows_cond]; ves_c = [r[1] for r in rows_cond]

        ax.plot(ages_a, ves_a, '-', color=col, lw=1.5, alpha=0.3)
        ax.plot(ages_c, ves_c, 'o--', color=col, lw=2.0, alpha=0.9, label=lab)
        for age, ve, na, ntot, fac in rows_cond:
            ax.scatter([age], [ve], s=60 + 200*(na/ntot),
                       color=col, alpha=0.3 + 0.6*(na/ntot),
                       edgecolor=col, zorder=5)

        # annotate gap at the LMIC reference
        if rows_cond:
            # find point closest to 8.7mo
            closest = min(rows_cond, key=lambda r: abs(r[0] - 8.7))
            ax.annotate(f'{closest[1]:.2f}', xy=(closest[0], closest[1]),
                        xytext=(closest[0] + 0.5, closest[1] + 0.04),
                        fontsize=7.5, color=col, ha='left')

    for x, lab in REF_AGES:
        ax.axvline(x, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (x, 0.03), rotation=90, va='bottom', ha='right',
                    fontsize=8, color='gray')

    ax.set_title(title)
    ax.set_xlabel('Simulated median age-of-infection (months)')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)

axs[0].set_ylabel('Achieved VE (overall symptomatic, resp=0.75)')
axs[0].legend(frameon=False, fontsize=9, loc='upper left')

# Print gap table
print('\n=== infnum VE at closest factor to LMIC (~8.7mo) — conditional VE (VE≤0.95) ===')
print(f'{"MOA":<22}  {"VE_cond":>8}  {"age (mo)":>10}  {"n_alive":>8}')
for suf, moa_label in [('_foi_sweep_alluniq.csv', 'Infection-blocking'),
                        ('_foi_sweep_alluniq_sb.csv', 'Symptom-blocking')]:
    in_path = OUT / f'infnum{suf}'
    if not in_path.exists():
        print(f'{moa_label:<22}  (data missing)'); continue
    rows = summarise(pd.read_csv(in_path), cond=True)
    if not rows:
        print(f'{moa_label:<22}  (no alive draws)'); continue
    pt = min(rows, key=lambda r: abs(r[0] - 8.7))
    print(f'{moa_label:<22}  {pt[1]:>8.3f}  {pt[0]:>10.1f}  {pt[2]:>8}')

fig.suptitle(
    'exp20 corrected FOI sweep — infection-number model, IB vs SB MOA\n'
    'all unique posterior draws (n=373); '
    'dashed=cond(VE≤0.95); larger/darker=more draws survived; resp=0.75',
    fontsize=9)
fig.tight_layout()
out = FIG / 've_vs_age_two_moa.png'
fig.savefig(out, dpi=140)
plt.close(fig)
print(f'\nwrote {out}')
