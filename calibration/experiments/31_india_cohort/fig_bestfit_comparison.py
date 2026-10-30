"""exp31 summary figure: best-fit IR by age and repeats across the three HM variants.
Hardcoded from trajectory selection best-fit outputs (ts_hm, ts_neoprime, ts_irall).
Run: python fig_bestfit_comparison.py
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent

# ── observed targets (MAL-ED Vellore / India) ────────────────────────────────
BINS       = ['<6 m', '6-11 m', '12-23 m']
OBS_IR     = [0.396, 1.706, 0.609]          # symptomatic IR per 100 child-years
OBS_REPEAT = 0.138

# ── best-fit outputs from trajectory selection ────────────────────────────────
RUNS = {
    'No priming\n(hm)':          {'ir': [0.575, 1.113, 0.850], 'rep': 0.058, 'logL': -288.69, 'col': '#718096'},
    'Neonatal priming\n(hm_neoprime)': {'ir': [0.976, 1.101, 0.633], 'rep': 0.125, 'logL': -287.95, 'col': '#2b6cb0'},
    'All-infection IR\n(hm_irall)':    {'ir': [0.574, 1.150, 0.844], 'rep': 0.067, 'logL': -288.72, 'col': '#9f7aea'},
}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A — IR by age bin
x = np.arange(len(BINS)); w = 0.18
# observed
axA.bar(x - 1.5*w, OBS_IR, w, label='Observed (MAL-ED)', color='#2d3748', zorder=3)
for i, (lab, d) in enumerate(RUNS.items()):
    offset = (i - 0.5) * w
    axA.bar(x + offset, d['ir'], w, label=lab, color=d['col'], alpha=0.85, zorder=3)

axA.set_xticks(x); axA.set_xticklabels(BINS)
axA.set_ylabel('Symptomatic IR (per 100 child-years)')
axA.set_title('A. Best-fit symptomatic IR by age bin')
axA.legend(fontsize=8.5, frameon=False)
axA.set_ylim(0, 2.2)

# Panel B — repeats + logL
run_labels = [lab.replace('\n', ' ') for lab in RUNS]
rep_vals   = [d['rep'] for d in RUNS.values()]
cols       = [d['col'] for d in RUNS.values()]
logL_vals  = [d['logL'] for d in RUNS.values()]

xb = np.arange(len(run_labels))
bars = axB.bar(xb, rep_vals, 0.45, color=cols, alpha=0.85, zorder=3)
axB.axhline(OBS_REPEAT, color='#2d3748', ls='--', lw=1.6, label=f'Observed ({OBS_REPEAT})')
for xi, (rv, ll) in enumerate(zip(rep_vals, logL_vals)):
    axB.text(xi, rv + 0.003, f'{rv:.3f}', ha='center', fontsize=9)
    axB.text(xi, -0.012, f'logL={ll:.1f}', ha='center', fontsize=8, color='#555')
axB.set_xticks(xb); axB.set_xticklabels(run_labels, fontsize=8.5)
axB.set_ylabel('Repeat-detected fraction')
axB.set_title('B. Repeat detections vs observed (best-fit logL below)')
axB.legend(fontsize=9, frameon=False)
axB.set_ylim(-0.03, 0.20)

fig.suptitle('Exp 31 — India Vellore cohort: three infnum HM variants, best-fit trajectories\n'
             'Neonatal priming (blue) is best logL and closest on repeats; '
             'all three overshoot <6m or 6-11m IR', fontsize=10)
fig.tight_layout()
out = HERE / 'figures' / 'bestfit_comparison.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print('wrote', out)
