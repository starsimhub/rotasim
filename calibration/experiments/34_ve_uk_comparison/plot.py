"""exp34 plot — UK vs Bangladesh VE-FOI gradient, infnum model, by MOA.
Overlays UK draws (exp 34) on Bangladesh draws (exp 20) in the same two-panel figure.
Run after both IB and SB sweeps complete on covaguest.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
EXP20 = HERE.parents[1] / 'experiments' / '20_ve_comparison' / 'outputs'

COL_UK = '#2471a3'   # blue for UK
COL_BD = '#c0392b'   # red for Bangladesh (exp 20 colour)

PANELS = [
    ('uk_infnum_foi_sweep.csv',    'infnum_foi_sweep.csv',    'A. Infection-blocking MOA'),
    ('uk_infnum_foi_sweep_sb.csv', 'infnum_foi_sweep_sb.csv', 'B. Symptom-blocking MOA'),
]


def summarise(df):
    rows = []
    for fac, sub in df.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if len(alive) == 0:
            continue
        rows.append((alive.age_of_inf.median(), alive.ve_overall.median(),
                     len(alive), len(sub), fac))
    return sorted(rows)


fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, (uk_f, bd_f, title) in zip(axs, PANELS):
    for fpath, col, label in [(OUT / uk_f, COL_UK, 'UK (exp 28)'),
                               (EXP20 / bd_f, COL_BD, 'Bangladesh (exp 27)')]:
        if not fpath.exists():
            ax.set_title(f'{title}\n(missing: {fpath.name})')
            continue
        d = pd.read_csv(fpath)
        rows = summarise(d)
        ages = [r[0] for r in rows]; ves = [r[1] for r in rows]
        nalive = [r[2] for r in rows]; ntot = [r[3] for r in rows]
        ax.plot(ages, ves, '-', color=col, lw=1.8, alpha=0.6, label=label)
        for a, v, na, nt, fac in rows:
            rel = na / nt
            ax.scatter([a], [v], s=50 + 180 * rel, color=col,
                       alpha=0.25 + 0.65 * rel, edgecolor=col, zorder=5)

    for x_emp, lab in [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]:
        ax.axvline(x_emp, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (x_emp, 0.03), rotation=90, va='bottom', ha='right',
                    fontsize=8, color='gray')

    ax.set_xlabel('simulated median age-of-infection (months)')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)

axs[0].set_ylabel('achieved VE (overall symptomatic, resp=0.75)')
axs[0].legend(frameon=False, loc='upper left')

fig.suptitle('exp34 — UK vs Bangladesh infnum: VE vs FOI gradient\n'
             'larger/darker = more draws survived; resp=0.75',
             fontsize=10)
fig.tight_layout()
out = FIG / 've_foi_uk_vs_bd.png'
fig.savefig(out, dpi=140)
plt.close(fig)
print('wrote', out)
