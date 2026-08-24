"""exp30 consistency figure: two-panel comparison.
Panel A — observed pre-vaccine vs model no-vaccine (calibration check).
Panel B — observed post-vaccine vs model vaccine take=0.9 (forward-prediction check).
Reads outputs/uk_vaccine_predict.json.  Run: python fig_uk_prevax_postvax_panels.py
"""
import json, pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
d = json.load(open(HERE / 'outputs' / 'uk_vaccine_predict.json'))
labels = d['bin_labels']
x = np.arange(len(labels))
w = 0.35

FONT_SCALE = 2.2  # larger text throughout, for slide/print legibility
plt.rcParams.update({
    'font.size': 12 * FONT_SCALE,
    'axes.titlesize': 12 * FONT_SCALE,
    'axes.labelsize': 12 * FONT_SCALE,
    'xtick.labelsize': 10.5 * FONT_SCALE,
    'ytick.labelsize': 11 * FONT_SCALE,
    'legend.fontsize': 9 * FONT_SCALE * 1.3,
})

fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 16), sharex=True, sharey=True)

# Panel A: pre-vaccine observed vs model no-vaccine
axA.bar(x - w/2, d['observed_pre'], w, label='Observed pre-vaccine', color='#4a5568')
axA.bar(x + w/2, d['novax']['prop_med'], w, label='Model no-vaccine', color='#90cdf4', edgecolor='#2b6cb0')
axA.set_ylabel('Share of cases')
axA.set_title('A. Pre-vaccine: observed vs model (no vaccine)')
axA.legend(frameon=False)
axA.set_ylim(0, 0.55)

# Panel B: post-vaccine observed vs model vaccine take=0.9
axB.bar(x - w/2, d['observed_post'], w, label='Observed post-vaccine', color='#2d3748')
axB.bar(x + w/2, d['vax_0.9']['prop_med'], w, label='Model vaccine (take=0.9)', color='#2b6cb0', edgecolor='#1a365d')
axB.set_xticks(x); axB.set_xticklabels(labels, rotation=20)
axB.set_ylabel('Share of cases')
axB.set_title('B. Post-vaccine: observed vs model (take=0.9)')
axB.legend(frameon=False)

fig.suptitle('UK vaccine validation — pre-vaccine calibration (A)\nand post-vaccine forward prediction (B)',
             fontsize=13 * FONT_SCALE)
fig.tight_layout()
out = HERE / 'figures' / 'uk_prevax_novax_postvax_09.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches='tight')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
print('wrote', out, 'and', out.with_suffix('.pdf'))
