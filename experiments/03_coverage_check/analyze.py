"""
Exp 03 — analyze the prior predictive / coverage check.

Loads the 1000-draw ensemble and the MAL-ED Bangladesh targets, then asks:
  1. Reachability: do the observed IR-by-age and first-infection quartiles fall
     inside the model's 5-95% ensemble envelope?
  2. Endemic sanity: what distribution of prevalence does the prior produce?

Outputs: figures/coverage.png and a printed coverage verdict.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'
FIGDIR = HERE / 'figures'
REPO = HERE.parent.parent
MALED = REPO / 'calibration' / 'maled_data'

LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']

# --- load ensemble ---
rows = [json.loads(l) for l in open(OUTDIR / 'results.jsonl')]
df = pd.DataFrame([r for r in rows if r.get('ok')])
print(f'Loaded {len(df)} successful draws')

# --- load observed targets (Bangladesh, symptomatic IR + first-infection events) ---
ir_t = pd.read_csv(MALED / 'ir_by_age_symp_bangladesh.csv').set_index('age_cat').reindex(LABELS)
fi = pd.read_csv(MALED / 'first_infection_bangladesh.csv')
fi_ev = fi[fi['event_observed'] == 1]['age_event_months'].dropna()
fi_q = np.quantile(fi_ev, [0.25, 0.5, 0.75])
target_ir = ir_t['IR'].values
print(f'Target IR (symptomatic): {dict(zip(LABELS, target_ir.round(2)))}')
print(f'Target first-inf quartiles (events-only): q25={fi_q[0]:.1f} med={fi_q[1]:.1f} q75={fi_q[2]:.1f}')


def envelope(arr, lo=5, hi=95):
    return np.nanpercentile(arr, lo), np.nanpercentile(arr, hi), np.nanmedian(arr)


# --- coverage verdict per IR bin ---
print('\n=== IR-by-age coverage (5-95% ensemble envelope vs observed) ===')
print(f'{"bin":>8}  {"obs":>7}  {"ens 5%":>8}  {"ens 95%":>8}  {"median":>8}  covered?')
ir_cols = [f'ir_{b}' for b in LABELS]
covered = {}
for b in LABELS:
    lo, hi, med = envelope(df[f'ir_{b}'].values)
    o = ir_t.loc[b, 'IR']
    cov = lo <= o <= hi
    covered[b] = cov
    print(f'{b:>8}  {o:>7.2f}  {lo:>8.2f}  {hi:>8.2f}  {med:>8.2f}  {"YES" if cov else "NO"}')

# --- first-infection coverage ---
print('\n=== First-infection median coverage ===')
lo, hi, med = envelope(df['fi_median'].values)
fi_cov = lo <= fi_q[1] <= hi
print(f'  obs median={fi_q[1]:.2f}  ens 5-95%=[{lo:.2f},{hi:.2f}]  ens median={med:.2f}  '
      f'{"covered" if fi_cov else "NOT covered"}')

# --- prevalence sanity ---
pv = df['prev_mean'].values
print(f'\n=== Endemic prevalence across prior ===')
print(f'  prevalence: median={np.median(pv):.3f}, 5-95%=[{np.percentile(pv,5):.3f},{np.percentile(pv,95):.3f}], '
      f'max={pv.max():.3f}')
print(f'  draws with prev>0.2 (saturated-ish): {(pv>0.2).mean()*100:.0f}%')
print(f'  max drift |late-early| in window: {np.nanmax(np.abs(df["prev_drift"].values)):.4f} (stationarity)')

# --- JOINT coverage: can a single draw match the whole shape at once? ---
# This is the scientifically meaningful question (marginal per-bin coverage above
# is trivially passed by an over-wide prior). Mirrors Alicia's exp 02 finding.
print('\n=== JOINT coverage (single draw matching all targets simultaneously) ===')
model_ir = df[[f'ir_{b}' for b in LABELS]].values  # (n_draws, 4)
# (1) qualitative shape: 6-11m is the peak AND 24-35m below 6-11m (rotavirus pattern)
peak_at_611 = (np.argmax(model_ir, axis=1) == 1)
print(f'  draws with peak at 6-11m: {peak_at_611.mean()*100:.1f}%')
# (2) all 4 bins within a factor of 3 of target (loose joint band)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = model_ir / target_ir[None, :]
within3 = np.all((ratio >= 1/3) & (ratio <= 3), axis=1)
print(f'  draws with all 4 IR bins within 3x of target: {within3.sum()} ({within3.mean()*100:.2f}%)')
# (3) joint log-SSE GOF (same form as calibration gof_incidence); report the best
log_eps = 0.01
gof_inc = np.sum((np.log(model_ir + log_eps) - np.log(target_ir[None, :] + log_eps)) ** 2, axis=1)
best = np.argmin(gof_inc)
print(f'  best joint IR GOF = {gof_inc[best]:.3f} (draw {df.iloc[best]["draw_id"]}); '
      f'model IR = {model_ir[best].round(2).tolist()}')
print(f'  best draw also: peak_at_611={bool(peak_at_611[best])}, prev={df.iloc[best]["prev_mean"]:.3f}')
joint_ok = within3.sum() > 0

# --- figure ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) IR-by-age: ensemble envelope vs target
ax = axes[0, 0]
x = np.arange(len(LABELS))
los = [np.nanpercentile(df[f'ir_{b}'], 5) for b in LABELS]
his = [np.nanpercentile(df[f'ir_{b}'], 95) for b in LABELS]
meds = [np.nanmedian(df[f'ir_{b}']) for b in LABELS]
ax.fill_between(x, los, his, alpha=0.25, color='tab:blue', label='model 5-95%')
ax.plot(x, meds, 'o-', color='tab:blue', label='model median')
ax.plot(x, target_ir, 's-', color='red', ms=9, label='MAL-ED observed')
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel('Symptomatic IR (per 100 PM)')
ax.set_title('(a) IR-by-age: coverage')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# (b) IR-by-age zoomed to data scale
ax = axes[0, 1]
ax.fill_between(x, los, his, alpha=0.25, color='tab:blue', label='model 5-95%')
ax.plot(x, meds, 'o-', color='tab:blue')
ax.plot(x, target_ir, 's-', color='red', ms=9, label='MAL-ED observed')
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylim(0, max(target_ir.max(), 10) * 1.5)
ax.set_ylabel('Symptomatic IR (per 100 PM)')
ax.set_title('(b) Same, zoomed to data scale')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# (c) prevalence distribution
ax = axes[1, 0]
ax.hist(pv, bins=40, color='tab:purple', alpha=0.8)
ax.axvline(np.median(pv), color='k', ls='--', label=f'median {np.median(pv):.2f}')
ax.set_xlabel('Mean endemic prevalence (in window)')
ax.set_ylabel('# draws')
ax.set_title('(c) Endemic prevalence across prior')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# (d) first-infection median distribution
ax = axes[1, 1]
ax.hist(df['fi_median'].dropna(), bins=40, color='tab:green', alpha=0.8)
ax.axvline(fi_q[1], color='red', lw=2, label=f'MAL-ED median {fi_q[1]:.1f}')
ax.set_xlabel('Model first-infection median (months)')
ax.set_ylabel('# draws')
ax.set_title('(d) First-infection median: coverage')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.suptitle(f'Exp 03 — Prior Predictive Coverage (Bangladesh, {len(df)} draws)', fontsize=12)
plt.tight_layout()
fig.savefig(FIGDIR / 'coverage.png', dpi=150, bbox_inches='tight')
print(f'\nSaved figures/coverage.png')

# --- overall verdict ---
all_ir = all(covered.values())
print('\n=== VERDICT ===')
print(f'  IR bins covered: {sum(covered.values())}/4  {covered}')
print(f'  First-inf median covered: {fi_cov}')
if all_ir and fi_cov:
    print('  => COVERED: data is inside the achievable envelope; proceed to calibration.')
else:
    print('  => NOT fully covered: some targets outside the model envelope.')
    for b in LABELS:
        if not covered[b]:
            o = ir_t.loc[b, 'IR']; med = np.nanmedian(df[f'ir_{b}'])
            print(f'     {b}: obs={o:.2f} vs model median={med:.2f} '
                  f'({"model too high" if med > o else "model too low"})')
