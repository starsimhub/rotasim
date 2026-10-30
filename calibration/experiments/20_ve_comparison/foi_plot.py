"""Plot achieved VE vs simulated age-of-infection (FOI gradient), both models.
Per beta-factor: median age-of-infection vs median VE over surviving draws (novax IR>0.1).
Point opacity/size reflects how many draws survived (low-FOI factors are extinction-biased).
Vertical markers = empirical age-at-(first/severe)-infection: LMIC ~8.7mo, high-income ~15mo."""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
MODELS = {'age_binned': ('Age (binned)', '#2c7fb8'), 'infnum': ('Infection-number', '#c0392b')}

fig, ax = plt.subplots(figsize=(9, 6))
print("FOI gradient (median age-of-infection -> median VE, surviving draws):")
for m, (lab, col) in MODELS.items():
    f = OUT / f'{m}_foi_sweep.csv'
    if not f.exists(): continue
    d = pd.read_csv(f)
    rows = []
    for fac, sub in d.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if len(alive) == 0: continue
        rows.append((alive.age_of_inf.median(), alive.ve_overall.median(), len(alive), len(sub), fac))
    rows.sort()
    ages = [r[0] for r in rows]; ves = [r[1] for r in rows]; nalive = [r[2] for r in rows]; ntot = [r[3] for r in rows]
    ax.plot(ages, ves, '-', color=col, lw=1.5, alpha=.6, label=lab)
    for a, v, na, nt, fac in rows:
        rel = na / nt                              # fraction surviving = reliability
        ax.scatter([a], [v], s=40 + 160 * rel, color=col, alpha=0.25 + 0.65 * rel, edgecolor=col, zorder=5)
        print(f"  {lab:18s} factor {fac:4.2f}: age-of-inf {a:5.1f}mo  VE {v:.3f}  (alive {na}/{nt})")
# empirical age-of-infection markers
for x, lab in [(8.7, 'LMIC ~8.7mo'), (15.0, 'high-income ~15mo')]:
    ax.axvline(x, color='gray', ls=':', lw=1.2); ax.annotate(lab, (x, 0.02), rotation=90, va='bottom', ha='right', fontsize=8, color='gray')
ax.set_xlabel('simulated median age-of-infection (months)'); ax.set_ylabel('achieved VE (overall symptomatic, resp=0.75)')
ax.set_ylim(-0.05, 1.05)
ax.set_title('exp20 FOI gradient — achieved VE rises steeply as infection shifts older (lower FOI)\n'
             'larger/darker points = more draws survived; faint = extinction-biased low-FOI end')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(FIG / 've_vs_age_of_infection.png', dpi=140); plt.close(fig)
print("wrote", FIG / 've_vs_age_of_infection.png')
