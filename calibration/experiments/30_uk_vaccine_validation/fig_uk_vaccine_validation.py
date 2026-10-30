"""exp30 figure: UK vaccine-impact validation.
Panel A - model 2-dose DIRECT VE (test-negative analog) and TOTAL (population) VE in <12mo, by
per-dose take, vs the reported UK test-negative ~77% (PMC6668223). Panel B - predicted vs observed
pre->post case age-distribution shift (the model matches VE magnitude but UNDER-shifts the
distribution => waning signature). Reads outputs/uk_vaccine_predict.json. Run: python fig_uk_vaccine_validation.py
"""
import json, pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
d = json.load(open(HERE / 'outputs' / 'uk_vaccine_predict.json'))
EFF = d['eff']; labels = d['bin_labels']
UK_TN_VE = 0.77      # PMC6668223 test-negative 2-dose VE, lab-confirmed rotavirus
IND_VE = 0.524       # India surveillance population (total-effect) VE, 6-11mo, all-states

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: VE by take
x = np.arange(len(EFF)); w = 0.36
direct12 = [d[f'vax_{e}']['direct12']['ve'] for e in EFF]
total12 = [d[f'vax_{e}']['ve12_med'] for e in EFF]
axA.bar(x - w/2, direct12, w, label='Direct VE (2-dose vs 0-dose)', color='#2b6cb0')
axA.bar(x + w/2, total12, w, label='Total/population VE (incl. herd)', color='#90cdf4')
axA.axhline(UK_TN_VE, ls='--', c='#c53030', lw=1.6)
axA.text(len(EFF)-1.05, UK_TN_VE+0.012, 'UK test-negative ~77% (PMC6668223)', color='#c53030', fontsize=9, ha='right')
axA.axhline(IND_VE, ls=':', c='#2f855a', lw=1.6)
axA.text(0.0, IND_VE-0.05, 'India surveillance ~52% (6-11mo, total effect)', color='#2f855a', fontsize=9, ha='left')
for xi, (dv, tv) in enumerate(zip(direct12, total12)):
    axA.text(xi - w/2, dv + 0.01, f'{dv:.2f}', ha='center', fontsize=9)
    axA.text(xi + w/2, tv + 0.01, f'{tv:.2f}', ha='center', fontsize=9)
axA.set_xticks(x); axA.set_xticklabels([f'take={e}' for e in EFF])
axA.set_ylabel('VE in children <12 months'); axA.set_ylim(0, 1.0)
axA.set_title('A. Model VE vs observed (per-dose take = seroconversion)')
axA.legend(loc='upper left', fontsize=8.5)

# Panel B: age-distribution shift, take=0.9
xb = np.arange(len(labels)); ww = 0.2
e_hi = max(EFF)
axB.bar(xb - 1.5*ww, d['observed_pre'], ww, label='Observed pre-vaccine', color='#718096')
axB.bar(xb - 0.5*ww, d['observed_post'], ww, label='Observed post-vaccine', color='#2d3748')
axB.bar(xb + 0.5*ww, d['novax']['prop_med'], ww, label='Model no-vaccine', color='#90cdf4')
axB.bar(xb + 1.5*ww, d[f'vax_{e_hi}']['prop_med'], ww, label=f'Model vaccine (take={e_hi})', color='#2b6cb0')
axB.set_xticks(xb); axB.set_xticklabels(labels, rotation=20)
axB.set_ylabel('Share of cases'); axB.set_title('B. Case age-distribution shift (model under-shifts = waning signature)')
axB.legend(fontsize=8.5)

fig.tight_layout()
out = HERE / 'figures' / 'uk_vaccine_validation.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches='tight')
print('wrote', out)
