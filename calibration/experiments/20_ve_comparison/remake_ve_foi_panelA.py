"""Standalone, large-text remake of ve_foi_by_moa.png's Panel A (infnum model,
infection-blocking MOA) only, as a PDF for slides/print. Same data
(outputs/infnum_foi_sweep.csv) and same computation as foi_plot_infnum_only.py --
purely a presentation-quality single-panel remake, not a new analysis.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)

COL = '#c0392b'

FONT_SCALE = 2.2
plt.rcParams.update({
    'font.size': 12 * FONT_SCALE,
    'axes.titlesize': 12 * FONT_SCALE,
    'axes.labelsize': 12 * FONT_SCALE,
    'xtick.labelsize': 11 * FONT_SCALE,
    'ytick.labelsize': 11 * FONT_SCALE,
    'legend.fontsize': 10.5 * FONT_SCALE,
})

d = pd.read_csv(OUT / 'infnum_foi_sweep.csv')
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

fig, ax = plt.subplots(figsize=(13, 10))

ax.plot(ages, ves, '-', color=COL, lw=2.5, alpha=0.6)
for a, v, na, nt, fac in rows:
    rel = na / nt
    ax.scatter([a], [v], s=140 + 320 * rel, color=COL,
               alpha=0.25 + 0.65 * rel, edgecolor=COL, zorder=5)
    ax.annotate(f'×{fac:.2f}', (a, v), textcoords='offset points',
                xytext=(8, 5), color=COL, alpha=0.8)

for x_emp, lab in [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]:
    ax.axvline(x_emp, color='gray', ls=':', lw=1.5)
    ax.annotate(lab, (x_emp, 0.03), rotation=90, va='bottom', ha='right', color='gray')

ax.set_xlabel('simulated median age-of-infection (months)')
ax.set_ylabel('achieved VE (overall symptomatic, resp=0.75)')
ax.set_title('Expected VE rises as age of infection increases')
ax.set_ylim(-0.05, 1.05)
ax.scatter([], [], s=200, color=COL, label='Infection-number model')
ax.legend(frameon=False, loc='upper left')

fig.tight_layout()
fig.savefig(FIG / 've_foi_by_moa_panelA.pdf', bbox_inches='tight')
fig.savefig(FIG / 've_foi_by_moa_panelA_hires.png', dpi=200, bbox_inches='tight')
print('wrote', FIG / 've_foi_by_moa_panelA.pdf', 'and', FIG / 've_foi_by_moa_panelA_hires.png')
