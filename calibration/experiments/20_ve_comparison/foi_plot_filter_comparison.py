"""Compare three filtering approaches for VE vs FOI curves.

Three lines on one plot, all using infnum IB resp=0.75 g11 data:
  A. novax_ir > 0.1 only         — no upper VE cap
  B. novax_ir > 0.1, vax_ir > 0  — drops only exact-zero vax_ir (stochastic extinction)
  C. novax_ir > 0.1, ve <= 0.95  — current approach (drops all high-VE draws)

Run:
  python foi_plot_filter_comparison.py
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT  = HERE / 'outputs'
FIG  = HERE / 'figures'; FIG.mkdir(exist_ok=True)

REF_AGES = [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]

FILTERS = [
    ('novax_only',  'A. novax_ir > 0.1 only',            '#1a6b3a', '-',  'o'),
    ('vax_nonzero', 'B. + vax_ir > 0 (drop exact zeros)', '#e07b00', '--', 's'),
    ('cond_095',    'C. + ve ≤ 0.95 (current)',           '#8b1a1a', ':',  '^'),
]


def summarise(df, filter_name):
    rows = []
    for fac, sub in df.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if filter_name == 'vax_nonzero':
            alive = alive[alive.vax_ir > 0]
        elif filter_name == 'cond_095':
            alive = alive[alive.ve_overall <= 0.95]
        # novax_only: no additional filter
        if len(alive) == 0:
            continue
        rows.append((
            alive.age_of_inf.median(),
            alive.ve_overall.median(),
            len(alive),
            len(sub),
            fac,
        ))
    return sorted(rows)


fpath = OUT / 'infnum_foi_sweep_alluniq_g11.csv'
df = pd.read_csv(fpath)

fig, ax = plt.subplots(figsize=(9, 5.5))

print(f'\n{"Filter":<14}  {"age~8.7mo":>10}  {"VE@8.7":>8}  {"age~15mo":>10}  {"VE@15":>8}  {"n@15":>6}')
for fname, label, col, ls, mk in FILTERS:
    rows = summarise(df, fname)
    if not rows:
        continue
    ages = [r[0] for r in rows]
    ves  = [r[1] for r in rows]
    ns   = [r[2] for r in rows]
    tots = [r[3] for r in rows]

    ax.plot(ages, ves, ls=ls, color=col, lw=2.0, marker=mk, ms=7,
            alpha=0.85, label=label)

    # bubble size ~ fraction alive
    for age, ve, na, ntot, fac in rows:
        frac = na / ntot
        ax.scatter([age], [ve], s=40 + 180*frac,
                   color=col, alpha=0.15 + 0.5*frac, edgecolor=col, zorder=4)

    c87  = min(rows, key=lambda r: abs(r[0] - 8.7))
    c15  = min(rows, key=lambda r: abs(r[0] - 15.0))
    print(f'{fname:<14}  {c87[0]:>10.1f}  {c87[1]:>8.3f}  {c15[0]:>10.1f}  {c15[1]:>8.3f}  {c15[2]:>6}')

for x, lab in REF_AGES:
    ax.axvline(x, color='gray', ls=':', lw=1.2)
    ax.annotate(lab, (x, 0.03), rotation=90, va='bottom', ha='right',
                fontsize=8, color='gray')

# annotate VE at 15mo for each filter
for fname, label, col, ls, mk in FILTERS:
    rows = summarise(df, fname)
    if not rows:
        continue
    c15 = min(rows, key=lambda r: abs(r[0] - 15.0))
    ax.annotate(f'{c15[1]:.2f}', xy=(c15[0], c15[1]),
                xytext=(c15[0] + 0.3, c15[1] + 0.025),
                fontsize=8, color=col, ha='left')

ax.set_xlabel('Simulated median age-of-infection (months)')
ax.set_ylabel('Achieved VE (overall symptomatic, median over draws)')
ax.set_xlim(left=0)
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title(
    'exp20 — filter comparison, infnum IB resp=0.75, 11-point g11 grid\n'
    'n=373 unique draws; bubble size ∝ fraction alive at each factor',
    fontsize=10)

fig.tight_layout()
out = FIG / 've_filter_comparison_g11.png'
fig.savefig(out, dpi=140)
plt.close(fig)
print(f'\nwrote {out}')
