"""Standalone, large-text remake of uk_vaccine_validation.png's Panel A (direct/total
2-dose VE by take vs UK test-negative and India surveillance reference lines) only,
as a PDF for slides/print. Same data/computation as fig_uk_vaccine_validation.py --
purely a presentation-quality remake, plus a more legible India reference-line color
(the original green was low-contrast against the light-blue bar/white background).
"""
import json, pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / 'figures'
d = json.load(open(HERE / 'outputs' / 'uk_vaccine_predict.json'))
EFF = d['eff']
UK_TN_VE = 0.77      # PMC6668223 test-negative 2-dose VE, lab-confirmed rotavirus
IND_VE = 0.524       # India surveillance population (total-effect) VE, 6-11mo, all-states
IND_COLOR = 'black'  # replaces the original low-contrast green

FONT_SCALE = 2.2
plt.rcParams.update({
    'font.size': 12 * FONT_SCALE,
    'axes.titlesize': 12 * FONT_SCALE,
    'axes.labelsize': 12 * FONT_SCALE,
    'xtick.labelsize': 11 * FONT_SCALE,
    'ytick.labelsize': 11 * FONT_SCALE,
    'legend.fontsize': 10 * FONT_SCALE,
})

fig, ax = plt.subplots(figsize=(13, 10))

x = np.arange(len(EFF)); w = 0.36
direct12 = [d[f'vax_{e}']['direct12']['ve'] for e in EFF]
total12 = [d[f'vax_{e}']['ve12_med'] for e in EFF]
ax.bar(x - w/2, direct12, w, label='Direct VE (2-dose vs 0-dose)', color='#2b6cb0')
ax.bar(x + w/2, total12, w, label='Total/population VE (incl. herd)', color='#90cdf4')
ax.axhline(UK_TN_VE, ls='--', c='#c53030', lw=2.2)
ax.text(len(EFF)-1.05, UK_TN_VE+0.05, 'UK test-negative ~77% (PMC6668223)', color='#c53030', ha='right')
ax.axhline(IND_VE, ls=':', c=IND_COLOR, lw=2.2)
ax.text(0.0, IND_VE-0.08, 'India surveillance ~52% (6-11mo, total effect)', color=IND_COLOR, ha='left')
for xi, (dv, tv) in enumerate(zip(direct12, total12)):
    ax.text(xi - w/2, dv + 0.015, f'{dv:.2f}', ha='center')
    ax.text(xi + w/2, tv + 0.015, f'{tv:.2f}', ha='center')
ax.set_xticks(x); ax.set_xticklabels([f'take={e}' for e in EFF])
ax.set_ylabel('VE in children <12 months'); ax.set_ylim(0, 1.08)
ax.set_title('Model VE vs observed (per-dose take = seroconversion)')
ax.legend(loc='upper left', frameon=False)

fig.tight_layout()
fig.savefig(FIG_DIR / 'uk_vaccine_validation_panelA.pdf', bbox_inches='tight')
fig.savefig(FIG_DIR / 'uk_vaccine_validation_panelA_hires.png', dpi=200, bbox_inches='tight')
print('wrote', FIG_DIR / 'uk_vaccine_validation_panelA.pdf', 'and',
      FIG_DIR / 'uk_vaccine_validation_panelA_hires.png')
