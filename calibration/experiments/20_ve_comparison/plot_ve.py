"""Overlay the two VE posterior distributions (age_binned vs infnum) from exp20.
Reads {model}_ve_draws.csv; makes (1) VE_overall vs efficacy with 95% CrI ribbons
(does the gap vary by efficacy? do the bands separate?), (2) per-efficacy overlaid
VE_overall distributions, (3) VE by age bin. Pure pandas/matplotlib -- runs anywhere."""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'outputs'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
MODELS = {'age_binned': ('Age (binned)', '#2c7fb8'), 'infnum': ('Infection-number', '#c0392b')}
BINS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']

dfs = {m: pd.read_csv(OUT / f'{m}_ve_draws.csv') for m in MODELS if (OUT / f'{m}_ve_draws.csv').exists()}
assert dfs, "no *_ve_draws.csv in outputs/"
responses = sorted(set().union(*[set(d['response'].unique()) for d in dfs.values()]))

def stat(v):
    v = v.dropna(); return v.median(), v.quantile(.025), v.quantile(.975)

# (1) VE_overall vs efficacy, median + 95% CrI ribbon
fig, ax = plt.subplots(figsize=(8, 5.5))
for m, (lab, col) in MODELS.items():
    if m not in dfs: continue
    s = [stat(dfs[m][dfs[m].response == r]['ve_overall']) for r in responses]
    med = [x[0] for x in s]; lo = [x[1] for x in s]; hi = [x[2] for x in s]
    ax.plot(responses, med, 'o-', color=col, lw=2, label=lab)
    ax.fill_between(responses, lo, hi, color=col, alpha=.2)
ax.set_xlabel('per-dose seroconversion (underlying efficacy)'); ax.set_ylabel('achieved VE (overall symptomatic)')
ax.set_xticks(responses); ax.axhline(0, color='gray', lw=.5)
ax.set_title('exp20 — achieved VE vs efficacy: age vs infection-number\n(separation = robust structural divergence; overlap = unresolved)')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / 've_overall_vs_efficacy.png', dpi=130); plt.close(fig)

# (2) per-efficacy overlaid VE_overall distributions
fig, axs = plt.subplots(1, len(responses), figsize=(5 * len(responses), 4.5), sharey=True, squeeze=False); axs = axs[0]
for ax, r in zip(axs, responses):
    for m, (lab, col) in MODELS.items():
        if m not in dfs: continue
        v = dfs[m][dfs[m].response == r]['ve_overall'].dropna()
        ax.hist(v, bins=25, density=True, color=col, alpha=.5, label=lab); ax.axvline(v.median(), color=col, ls='--')
    ax.set_title(f'efficacy {r}'); ax.set_xlabel('achieved VE')
axs[0].set_ylabel('density'); axs[0].legend(frameon=False)
fig.suptitle('exp20 — VE_overall posterior distributions by efficacy'); fig.tight_layout(); fig.savefig(FIG / 've_distributions.png', dpi=130); plt.close(fig)

# table
print("VE_overall  median [95% CrI]  by model x efficacy:")
for r in responses:
    parts = []
    for m, (lab, _) in MODELS.items():
        if m in dfs:
            md, lo, hi = stat(dfs[m][dfs[m].response == r]['ve_overall'])
            parts.append(f"{lab}: {md:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  eff {r}:  " + "   |   ".join(parts))
print("wrote", FIG / 've_overall_vs_efficacy.png', "+", FIG / 've_distributions.png')
