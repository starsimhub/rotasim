"""Re-render figures/icddrb_posterior_overlay.png as a larger-text, higher-resolution PDF
(figures/icddrb_posterior_overlay.pdf). Same data, same layout as the original -- no new
simulation, no new science, purely a presentation-quality remake for a talk/document. The
original plotting script was run interactively and never committed; this reconstructs it from
the already-saved outputs/posterior_overlay.json plus the same target loaders
relative_incidence_fig.py used, and was checked by eye against the original PNG.
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import process_surveillance_icddrb as IC, process_incidence_maled as P

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / 'figures'

FONT_SCALE = 2.2  # larger text throughout, for slide/print legibility
plt.rcParams.update({
    'font.size': 12 * FONT_SCALE,
    'axes.titlesize': 13 * FONT_SCALE,
    'axes.labelsize': 12 * FONT_SCALE,
    'xtick.labelsize': 11 * FONT_SCALE,
    'ytick.labelsize': 11 * FONT_SCALE,
    'legend.fontsize': 10.5 * FONT_SCALE * 1.3,
})

labels = ['<6\nmo', '6-11\nmo', '12-23\nmo', '24-59\nmo']
x = np.arange(4)

ic = IC.load_targets_icddrb()['ir_proxy']
mt = P.load_targets('bangladesh')['ir_by_age']['IR'].to_numpy()
overlay = json.load(open(HERE / 'outputs' / 'posterior_overlay.json'))

norm = lambda v: np.asarray(v, float) / np.asarray(v, float)[1]

fig, ax = plt.subplots(figsize=(21, 14))

for model, color in [('infnum', '#c0392b'), ('age_binned', '#2ca25f')]:
    d = overlay[model]
    med = norm(d['ir_med']); lo = np.asarray(d['ir_lo']) / d['ir_med'][1]; hi = np.asarray(d['ir_hi']) / d['ir_med'][1]
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, med, 'o-', color=color, lw=2.5, ms=11, label=f"model {model} (median, n={d['n']})")

ax.plot(x, norm(mt), ':', color='dimgray', lw=2.5, marker='s', ms=10,
        label='MAL-ED cohort IR (4th bin=24-35mo)')
ax.plot(x, norm(ic), '-', color='black', lw=3, marker='*', ms=22,
        label='icddr,b Dhaka observed (risk=cases/width)')

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('relative per-child risk (norm. to 6-11mo)')
ax.set_title("icddr,b validation: MAL-ED-fitted posterior vs observed risk-by-age (no refit)\n"
             "risk-adjusted all peak 6-11m; the 24-59m gap is medically-attended severity selection")
ax.legend(frameon=False, loc='upper right')
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(FIG_DIR / 'icddrb_posterior_overlay.pdf', bbox_inches='tight')
fig.savefig(FIG_DIR / 'icddrb_posterior_overlay_hires.png', dpi=300, bbox_inches='tight')
print("Saved figures/icddrb_posterior_overlay.pdf and figures/icddrb_posterior_overlay_hires.png")
