"""exp20 summary plot: achieved VE by efficacy, grouped by model (age vs infnum).
box = IQR, whisker = 95% CrI, line = median -- computed on draws EXCLUDING near-elimination
(VE_overall > 0.95, reported separately as a fraction, since those are genuine
total-effect near-elimination regimes that otherwise pin the upper CrI at 1.0)."""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
MODELS = {'age_binned': ('Age (binned)', '#2c7fb8'), 'infnum': ('Infection-number', '#c0392b')}
ELIM = 0.95
dfs = {m: pd.read_csv(OUT / f'{m}_ve_draws.csv') for m in MODELS if (OUT / f'{m}_ve_draws.csv').exists()}
responses = sorted(set().union(*[set(d['response'].unique()) for d in dfs.values()]))

fig, ax = plt.subplots(figsize=(9, 6)); w = 0.18
print("near-elimination fraction (VE>0.95) + conditional VE [excl. near-elim]:")
for j, (m, (lab, col)) in enumerate(MODELS.items()):
    if m not in dfs: continue
    for k, r in enumerate(responses):
        s = dfs[m][dfs[m].response == r]['ve_overall'].dropna()
        elim = float((s > ELIM).mean()); v = s[s <= ELIM]
        q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        x = k + (j - 0.5) * w * 1.25
        ax.plot([x, x], [q[0], q[4]], color=col, lw=1.5)                       # 95% CrI whisker
        ax.add_patch(Rectangle((x - w / 2, q[1]), w, q[3] - q[1], facecolor=col, alpha=.45, edgecolor=col))  # IQR box
        ax.plot([x - w / 2, x + w / 2], [q[2], q[2]], color=col, lw=2.5)       # median
        ax.annotate(f'{100*elim:.0f}% elim', (x, q[4]), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=7, color=col)
        print(f"  {lab:18s} eff {r}: near-elim {100*elim:4.0f}% | cond VE {q[2]:.3f} [{q[0]:.3f}, {q[4]:.3f}] (n={len(v)})")
    ax.plot([], [], color=col, lw=7, alpha=.5, label=lab)
ax.set_xticks(range(len(responses))); ax.set_xticklabels([str(r) for r in responses])
ax.set_xlabel('per-dose seroconversion (underlying efficacy)'); ax.set_ylabel('achieved VE (overall symptomatic)')
ax.set_title('exp20 — achieved VE by model and efficacy\nbox = IQR, whisker = 95% CrI, line = median (excl. near-elim; "% elim" = VE>0.95 fraction)')
ax.legend(frameon=False, loc='lower right'); ax.set_ylim(-0.1, 1.02); ax.axhline(0, color='gray', lw=.5)
fig.tight_layout(); fig.savefig(FIG / 've_boxplot_by_efficacy.png', dpi=140); plt.close(fig)
print("wrote", FIG / 've_boxplot_by_efficacy.png')
