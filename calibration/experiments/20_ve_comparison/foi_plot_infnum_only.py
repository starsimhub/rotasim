"""exp20 — VE vs FOI (age-of-infection gradient), infnum model only, by mechanism of action.
Two panels: infection-blocking MOA (left) and symptom-blocking MOA (right).
Per beta-factor: median age-of-infection vs median VE over surviving draws (novax IR>0.1).
Point opacity/size reflects reliability (fraction surviving at that FOI).
Vertical markers = empirical age-at-(first/severe)-infection: LMIC ~8.7mo, high-income ~15mo.
Reads infnum_foi_sweep.csv and infnum_foi_sweep_sb.csv from outputs/.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)

COL = '#c0392b'
PANELS = [
    ('infnum_foi_sweep.csv',    'A. Infection-blocking MOA'),
    ('infnum_foi_sweep_sb.csv', 'B. Symptom-blocking MOA'),
]

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, (fname, title) in zip(axs, PANELS):
    f = OUT / fname
    if not f.exists():
        ax.set_title(f'{title}\n(data not found: {fname})')
        continue
    d = pd.read_csv(f)
    rows = []
    for fac, sub in d.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if len(alive) == 0:
            continue
        rows.append((alive.age_of_inf.median(), alive.ve_overall.median(),
                     len(alive), len(sub), fac))
    rows.sort()

    ages   = [r[0] for r in rows]
    ves    = [r[1] for r in rows]
    nalive = [r[2] for r in rows]
    ntot   = [r[3] for r in rows]

    ax.plot(ages, ves, '-', color=COL, lw=1.8, alpha=0.6)
    for a, v, na, nt, fac in rows:
        rel = na / nt
        ax.scatter([a], [v], s=50 + 180 * rel, color=COL,
                   alpha=0.25 + 0.65 * rel, edgecolor=COL, zorder=5)
        ax.annotate(f'×{fac:.2f}', (a, v), textcoords='offset points',
                    xytext=(5, 3), fontsize=7.5, color=COL, alpha=0.7)

    for x_emp, lab in [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]:
        ax.axvline(x_emp, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (x_emp, 0.03), rotation=90, va='bottom', ha='right',
                    fontsize=8, color='gray')

    ax.set_xlabel('simulated median age-of-infection (months)')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    print(f"\n{title}")
    for a, v, na, nt, fac in rows:
        print(f"  factor {fac:.2f}: age-of-inf {a:.1f}mo  VE {v:.3f}  (alive {na}/{nt})")

axs[0].set_ylabel('achieved VE (overall symptomatic, resp=0.75)')
# single legend entry
axs[0].scatter([], [], s=120, color=COL, label='Infection-number model')
axs[0].legend(frameon=False, loc='upper left')

fig.suptitle('exp20 — Infection-number model: achieved VE rises as infection shifts to older ages\n'
             'larger/darker = more draws survived; ×factor = beta scaling',
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG / 've_foi_by_moa.png', dpi=140)
plt.close(fig)
print('\nwrote', FIG / 've_foi_by_moa.png')
