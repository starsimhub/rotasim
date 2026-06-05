"""Exp 06 — figure: the sharp IR peak achieved; reinfection the remaining gap."""
import json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
tgt = np.array([1.91, 5.37, 2.35]); labs = ['<6m', '6-11m', '12-23m']
rs = [json.loads(l) for l in open(HERE / 'outputs' / 'results.jsonl') if json.loads(l).get('ok')]
ev = np.array([r['frac_ever_infected'] for r in rs])
M = np.array([[r['ir_symp_<6 m'], r['ir_symp_6-11 m'], r['ir_symp_12-23 m']] for r in rs])
rep = np.array([r['repeat_detected_frac'] for r in rs])
per = ev > 0.05
sse = np.sum((np.log(M + .01) - np.log(tgt + .01))**2, axis=1)
sse_masked = np.where(per, sse, 1e9)
best = int(np.argmin(sse_masked))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(3)
# (a) ensemble of good-IR fits + best + target
good = np.argsort(sse_masked)[:40]
for i in good:
    ax[0].plot(x, M[i], color='tab:blue', alpha=0.12)
ax[0].plot(x, M[best], 'd-', color='k', lw=2, label=f'best (SSE={sse[best]:.2f})')
ax[0].plot(x, tgt, 's-', color='red', ms=10, label='MAL-ED')
ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylim(0, 8)
ax[0].set_title('(a) IR-by-age: SHARP 6-11mo peak achieved'); ax[0].legend(fontsize=8)

# (b) winning region: young_reservoir vs adult_contacts, colored by IR fit
yr = np.array([r['par_young_reservoir'] for r in rs]); ad = np.array([r['par_adult_contacts'] for r in rs])
top = np.argsort(sse_masked)[:60]
ax[1].scatter(yr[per], ad[per], s=6, c='lightgray', label='persistent')
sc = ax[1].scatter(yr[top], ad[top], s=30, c=sse[top], cmap='viridis_r', label='best-60 IR')
ax[1].set_xlabel('young_reservoir'); ax[1].set_ylabel('adult_contacts')
ax[1].set_title('(b) winning region: low reservoir + adult persistence'); ax[1].legend(fontsize=8)
plt.colorbar(sc, ax=ax[1], label='IR log-SSE')

# (c) the remaining gap: repeat fraction among good-IR draws
ax[2].scatter(sse_masked[per], rep[per], s=8, alpha=0.4)
ax[2].axhline(0.2, color='red', ls='--', label='data repeat ~0.2')
ax[2].scatter([sse[best]], [rep[best]], s=60, c='k', label=f'best-IR (repeat={rep[best]:.2f})')
ax[2].set_xlim(0, 3); ax[2].set_xlabel('IR log-SSE'); ax[2].set_ylabel('repeat fraction')
ax[2].set_title('(c) reinfection still ~2.5x high (the next target)'); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=0.25)
plt.suptitle('Exp 06 — titer maternal + low reservoir + adult persistence: sharp peak achieved', fontsize=12)
plt.tight_layout(); fig.savefig(HERE / 'figures' / 'peak_achieved.png', dpi=150, bbox_inches='tight')
r = rs[best]
print('best IR:', M[best].round(2).tolist(), 'SSE', round(sse[best], 3), 'repeat', round(rep[best], 2))
print('best params: yr=%.0f adult=%.1f titerMed=%.0f hl=%.0f gsd=%.1f hill=%.1f beta=%.2f sus1=%.2f' % (
    r['par_young_reservoir'], r['par_adult_contacts'], r['par_titer_median'], r['par_titer_half_life_days'],
    r['par_titer_gsd'], r['par_hill_slope'], r['par_base_beta'], r['par_sus_after_1']))
print('saved figures/peak_achieved.png')
