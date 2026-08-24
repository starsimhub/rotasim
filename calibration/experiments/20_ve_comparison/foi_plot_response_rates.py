"""FOI sweep: infnum IB model at three vaccine response rates (0.63, 0.75, 0.90).
Shows how response rate interacts with FOI to shape achievable VE.

5-point run (default):
  outputs/infnum_foi_sweep_alluniq.csv        (resp=0.75)
  outputs/infnum_foi_sweep_alluniq_r063.csv   (resp=0.63)
  outputs/infnum_foi_sweep_alluniq_r090.csv   (resp=0.90)

11-point run (--grid g11):
  outputs/infnum_foi_sweep_alluniq_g11.csv       (resp=0.75)
  outputs/infnum_foi_sweep_alluniq_g11_r063.csv  (resp=0.63)
  outputs/infnum_foi_sweep_alluniq_g11_r090.csv  (resp=0.90)

Run:
  python foi_plot_response_rates.py           # 5-point
  python foi_plot_response_rates.py --grid g11  # 11-point
"""
import argparse, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--grid', default='5pt', choices=['5pt', 'g11'],
                help='which factor grid to plot (default: 5pt)')
ap.add_argument('--size-tag', default='', dest='size_tag',
                help='extra suffix after grid tag, e.g. _50k')
args = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
OUT  = HERE / 'outputs'
FIG  = HERE / 'figures'; FIG.mkdir(exist_ok=True)

ELIM = 0.95
REF_AGES = [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]

_G11 = '_g11' if args.grid == 'g11' else ''
_SZ  = args.size_tag   # e.g. '_50k'
CONFIGS = [
    (0.63, f'{_G11}{_SZ}_r063', '#f0a500'),   # amber
    (0.75, f'{_G11}{_SZ}',      '#c0392b'),   # red
    (0.90, f'{_G11}{_SZ}_r090', '#6c0017'),   # deep crimson
]

out_tag = (f'_{args.grid}' if args.grid != '5pt' else '') + _SZ


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


fig, ax = plt.subplots(figsize=(8, 5.5))

# y-offsets so factor=0.9 annotations don't overlap
_F09_YOFF = {0.63: -0.04, 0.75: 0.0, 0.90: +0.04}
fac09_age = None   # set once from first loaded dataset

print(f'\n{"Response":>10}  {"age~8.7mo":>10}  {"VE_cond":>8}  {"n_alive":>8}')
for resp, tag, col in CONFIGS:
    fpath = OUT / f'infnum_foi_sweep_alluniq{tag}.csv'
    if not fpath.exists():
        print(f'{resp:>10.2f}  (missing: {fpath.name})')
        ax.annotate(f'resp={resp} missing', (0.5, resp),
                    xycoords=('axes fraction', 'data'), ha='center', fontsize=8,
                    color=col, alpha=0.7)
        continue

    d = pd.read_csv(fpath)
    rows = summarise(d, cond=False)   # novax_ir > 0.1 only — no ELIM cap

    ages = [r[0] for r in rows]; ves = [r[1] for r in rows]

    ax.plot(ages, ves, 'o--', color=col, lw=2.2, alpha=0.9,
            label=f'resp={resp:.2f}')
    for age, ve, na, ntot, fac in rows:
        ax.scatter([age], [ve], s=55 + 190*(na/ntot),
                   color=col, alpha=0.25 + 0.6*(na/ntot), edgecolor=col, zorder=5)

    if rows:
        closest = min(rows, key=lambda r: abs(r[0] - 8.7))
        print(f'{resp:>10.2f}  {closest[0]:>10.1f}  {closest[1]:>8.3f}  {closest[2]:>8}')
        ax.annotate(f'{closest[1]:.2f}', xy=(closest[0], closest[1]),
                    xytext=(closest[0] + 0.4, closest[1] + 0.025),
                    fontsize=8, color=col, ha='left')

        # annotate VE at factor=0.9
        r09 = next((r for r in rows if abs(r[4] - 0.9) < 0.01), None)
        if r09:
            if fac09_age is None:
                fac09_age = r09[0]
            yoff = _F09_YOFF.get(resp, 0.0)
            ax.annotate(f'{r09[1]:.2f}', xy=(r09[0], r09[1]),
                        xytext=(r09[0] + 0.4, r09[1] + yoff),
                        fontsize=8, color=col, ha='left')

for x, lab in REF_AGES:
    ax.axvline(x, color='gray', ls=':', lw=1.2)
    ax.annotate(lab, (x, 0.03), rotation=90, va='bottom', ha='right',
                fontsize=8, color='gray')

if fac09_age is not None:
    ax.axvline(fac09_age, color='steelblue', ls='--', lw=1.0, alpha=0.6)
    ax.annotate(f'factor=0.9\n~{fac09_age:.1f}mo', (fac09_age, 0.03),
                rotation=90, va='bottom', ha='right', fontsize=8, color='steelblue')

ax.set_xlabel('Simulated median age-of-infection (months)')
ax.set_ylabel('Achieved VE (overall symptomatic)')
ax.set_xlim(left=0)
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=10, loc='upper left',
          title='Vaccine response rate', title_fontsize=9)
grid_label = '11-point factor grid' if args.grid == 'g11' else '5-point factor grid'
n_agents_label = '50k agents' if '50k' in _SZ else '20k agents'
ax.set_title(
    f'exp20 — infnum model, infection-blocking MOA\n'
    f'VE vs FOI gradient at three response rates\n'
    f'({grid_label}, {n_agents_label}, n=373 draws, filter: novax_ir>0.1)',
    fontsize=10)

fig.tight_layout()
out = FIG / f've_vs_age_response_rates{out_tag}.png'
fig.savefig(out, dpi=140)
plt.close(fig)
print(f'\nwrote {out}')

# ── Summary table: VE by factor (age) × response rate ─────────────────────
all_rows = {}
for resp, tag, col in CONFIGS:
    fpath = OUT / f'infnum_foi_sweep_alluniq{tag}.csv'
    if fpath.exists():
        all_rows[resp] = {r[4]: r for r in summarise(pd.read_csv(fpath), cond=False)}

resps = [r for r, *_ in CONFIGS if r in all_rows]
all_factors = sorted({f for rr in all_rows.values() for f in rr})

header = f'\n{"factor":>8}  {"age(mo)":>8}  ' + '  '.join(f'VE@r={r:.2f}' for r in resps)
print(header)
print('-' * len(header.lstrip('\n')))
for fac in all_factors:
    age_str = ''
    ve_strs = []
    for resp in resps:
        row = all_rows[resp].get(fac)
        if row:
            age_str = f'{row[0]:>8.1f}'
            ve_strs.append(f'{row[1]:>10.3f}')
        else:
            ve_strs.append(f'{"—":>10}')
    print(f'{fac:>8.2f}  {age_str}  ' + '  '.join(ve_strs))
