"""Two-panel: achieved VE vs efficacy, age vs infnum, under each MOA.
Left = infection-blocking (the exp20 gap); right = symptom-blocking (gap collapses).
Conditional VE (excl. near-elimination VE>0.95); median + 95% CrI ribbon."""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
MODELS = {'age_binned': ('Age (binned)', '#2c7fb8'), 'infnum': ('Infection-number', '#c0392b')}
PANELS = [('', 'Infection-blocking MOA\n(age vs infnum diverge)'),
          ('_symptomblock', 'Symptom-blocking MOA\n(gap collapses)')]
ELIM = 0.95
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, (tag, title) in zip(axs, PANELS):
    for m, (lab, col) in MODELS.items():
        f = OUT / f'{m}_ve_draws{tag}.csv'
        if not f.exists(): continue
        d = pd.read_csv(f); resp = sorted(d['response'].unique())
        med, lo, hi = [], [], []
        for r in resp:
            v = d[d.response == r]['ve_overall']; v = v[v <= ELIM].dropna()
            med.append(v.median()); lo.append(v.quantile(.025)); hi.append(v.quantile(.975))
        ax.plot(resp, med, 'o-', color=col, lw=2, label=lab); ax.fill_between(resp, lo, hi, color=col, alpha=.2)
    ax.set_title(title); ax.set_xlabel('per-dose seroconversion (efficacy)'); ax.set_xticks(resp); ax.set_ylim(0, 1)
axs[0].set_ylabel('achieved VE (overall symptomatic, cond. excl. near-elim)'); axs[0].legend(frameon=False)
fig.suptitle('exp20 — the age vs infnum VE divergence is INFECTION-BLOCKING-SPECIFIC')
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / 've_moa_gap.png', dpi=140); plt.close(fig)
print("VE_overall median (cond.) by MOA x model x efficacy:")
for tag, title in PANELS:
    for m, (lab, _) in MODELS.items():
        f = OUT / f'{m}_ve_draws{tag}.csv'
        if not f.exists(): continue
        d = pd.read_csv(f); resp = sorted(d['response'].unique())
        meds = [d[(d.response == r) & (d.ve_overall <= ELIM)]['ve_overall'].median() for r in resp]
        print(f"  {title.splitlines()[0]:28s} {lab:18s}: " + " ".join(f"{x:.2f}" for x in meds))
print("wrote", FIG / 've_moa_gap.png')
