"""Exp 05 — figure: the mixing-induced knife-edge + IR-by-age status."""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
def load(p):
    return [json.loads(l) for l in open(p) if json.loads(l).get('ok')]

e5 = load(HERE / 'outputs' / 'results.jsonl')
e4 = load(HERE.parent / '04_maled_cohort_emulation' / 'outputs' / 'results.jsonl')
tgt = np.array([1.91, 5.37, 2.35]); labs = ['<6m', '6-11m', '12-23m']

ev5 = np.array([r['frac_ever_infected'] for r in e5])
M5 = np.array([[r['ir_symp_<6 m'], r['ir_symp_6-11 m'], r['ir_symp_12-23 m']] for r in e5])
per5 = ev5 > 0.05
sse = np.sum((np.log(M5[per5] + .01) - np.log(tgt + .01))**2, axis=1)
best = M5[per5][np.argmin(sse)]

# exp04 smooth gradient
b4 = np.array([r['par_base_beta'] for r in e4]); rep4 = np.array([r['repeat_detected_frac'] for r in e4])
ev4 = np.array([r['frac_ever_infected'] for r in e4])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].hist(ev5, bins=np.linspace(0, 1, 21), color='tab:red', alpha=0.8)
ax[0].set_title('(a) exp05 MixingPools: ever-infected is BIMODAL\n(extinct or saturated, nothing between)')
ax[0].set_xlabel('fraction ever infected by 24mo'); ax[0].set_ylabel('# draws')

ax[1].scatter(b4, ev4, s=8, alpha=0.4, label='ever-infected')
ax[1].scatter(b4, rep4, s=8, alpha=0.4, label='repeat frac')
ax[1].axhline(0.63, color='gray', ls=':', label='data ever~0.63'); ax[1].axhline(0.2, color='k', ls=':', label='data repeat~0.2')
ax[1].set_xscale('log'); ax[1].set_xlabel('base_beta (log)'); ax[1].set_title('(b) exp04 RandomNet: SMOOTH, close at low FOI')
ax[1].legend(fontsize=7)

x = np.arange(3)
ax[2].plot(x, best, 'd-', color='tab:blue', label=f'best exp05 draw')
ax[2].plot(x, tgt, 's-', color='red', ms=9, label='MAL-ED')
ax[2].set_xticks(x); ax[2].set_xticklabels(labs); ax[2].set_ylim(0, 7)
ax[2].set_title('(c) IR-by-age: peak BIN right, MAGNITUDE undershot'); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=0.25)
plt.suptitle('Exp 05 — structured mixing induces a knife-edge; homogeneous is smooth & near-data', fontsize=12)
plt.tight_layout(); fig.savefig(HERE / 'figures' / 'mixing_knife_edge.png', dpi=150, bbox_inches='tight')
print('saved figures/mixing_knife_edge.png; best exp05 IR:', best.round(2).tolist())
