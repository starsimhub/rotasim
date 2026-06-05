"""
Exp 04 — analyze the MAL-ED cohort emulation.

Compares the emulated birth cohort to MAL-ED Bangladesh on three fronts, all
under matched observation (surveillance detection + individual data-driven
dropout):
  1. Age-at-first-DETECTION: Kaplan-Meier (hand-rolled) of model vs data.
  2. IR-by-age (symptomatic): cohort model vs data.
  3. Repeat-infection fraction: model vs MAL-ED's reported ~10%.

Outputs figures/cohort_fit.png and a printed verdict (best-fitting draw).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'; FIGDIR = HERE / 'figures'
REPO = HERE.parent.parent
MALED = REPO / 'calibration' / 'maled_data'
LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
REPEAT_TARGET = 0.10  # MAL-ED: repeat detected infection in ~10% of children


def km(times, observed, grid):
    """Hand-rolled Kaplan-Meier survival S(t) on a grid. times/observed arrays."""
    times = np.asarray(times, float); observed = np.asarray(observed, bool)
    order = np.argsort(times); times, observed = times[order], observed[order]
    n = len(times); S = []; surv = 1.0; at_risk = n; i = 0
    event_times = np.unique(times[observed])
    et = 0
    for g in grid:
        # advance through all event times <= g
        while et < len(event_times) and event_times[et] <= g:
            t = event_times[et]
            d = int(np.sum((times == t) & observed))
            risk = int(np.sum(times >= t))
            if risk > 0:
                surv *= (1 - d / risk)
            et += 1
        S.append(surv)
    return np.array(S)


# --- load ensemble ---
rows = [json.loads(l) for l in open(OUTDIR / 'results.jsonl')]
df = pd.DataFrame([r for r in rows if r.get('ok')])
print(f'Loaded {len(df)} successful draws')

# --- data: KM of age-at-first-infection (Bangladesh), IR symptomatic ---
fi = pd.read_csv(MALED / 'first_infection_bangladesh.csv')
ir_t = pd.read_csv(MALED / 'ir_by_age_symp_bangladesh.csv').set_index('age_cat').reindex(LABELS)
target_ir = ir_t['IR'].values
grid = np.linspace(0, 24, 97)
S_data = km(fi['age_event_months'].values, fi['event_observed'].values == 1, grid)
data_frac_det_24 = 1 - S_data[-1]
print(f'Data: {len(fi)} children, fraction detected by 24mo = {data_frac_det_24:.2f}')

# --- per-draw KM + fit metrics ---
def draw_km(r):
    return km(r['km_time'], np.array(r['km_observed']) == 1, grid)

km_curves = np.array([draw_km(r) for _, r in df.iterrows()])  # (n_draws, grid)
# KM fit: integrated squared survival difference vs data
km_sse = np.mean((km_curves - S_data[None, :]) ** 2, axis=1)
# IR fit (symptomatic, log-SSE on the 3 infant bins with data signal)
ir_cols = [f'ir_symp_{b}' for b in LABELS[:3]]
model_ir = df[ir_cols].values
log_eps = 0.01
ir_sse = np.sum((np.log(model_ir + log_eps) - np.log(target_ir[:3][None, :] + log_eps)) ** 2, axis=1)
# repeat-fraction distance
rep = df['repeat_detected_frac'].values
rep_dist = np.abs(rep - REPEAT_TARGET)

# joint score (standardized): KM + IR + repeat all matter
joint = km_sse / np.median(km_sse) + ir_sse / np.median(ir_sse) + rep_dist / np.median(rep_dist)
best = int(np.argmin(joint))
br = df.iloc[best]
print(f'\n=== Best joint draw: #{br["draw_id"]} ===')
print(f'  KM frac-det-by-24mo: model={1-km_curves[best][-1]:.2f} vs data={data_frac_det_24:.2f}')
print(f'  repeat frac: model={br["repeat_detected_frac"]:.2f} vs data ~{REPEAT_TARGET:.2f}')
print(f'  ever detected: {br["frac_ever_detected"]:.2f}; ever infected (true): {br["frac_ever_infected"]:.2f}')
print(f'  IR symp [{",".join(LABELS[:3])}]: model={model_ir[best].round(2).tolist()} vs data={target_ir[:3].round(2).tolist()}')
print(f'  prevalence proxy via ever_infected; base_beta={br["par_base_beta"]:.3f}')

# --- how many draws are even in the MAL-ED ballpark? ---
plausible = (np.abs(rep - REPEAT_TARGET) < 0.10) & (np.abs(df['frac_ever_detected'].values - data_frac_det_24) < 0.15)
print(f'\nDraws with repeat~10% (+-10pp) AND frac-detected~data (+-15pp): {plausible.sum()} ({plausible.mean()*100:.1f}%)')
print(f'repeat fraction across prior: median={np.median(rep):.2f}, 5-95%=[{np.percentile(rep,5):.2f},{np.percentile(rep,95):.2f}]')

# --- figure ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
# plot a sample of model KM curves (light) + data + best
idx_sample = np.random.default_rng(0).choice(len(df), min(60, len(df)), replace=False)
for i in idx_sample:
    ax.plot(grid, km_curves[i], color='tab:blue', alpha=0.06)
ax.plot(grid, S_data, color='red', lw=2.5, label='MAL-ED data (KM)')
ax.plot(grid, km_curves[best], color='k', lw=2, label=f'best draw #{br["draw_id"]}')
ax.set_xlabel('Age (months)'); ax.set_ylabel('S(t) = P(not yet first-detected)')
ax.set_title('(a) Age-at-first-detection: KM, model ensemble vs data')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[0, 1]
x = np.arange(3)
los = [np.percentile(df[f'ir_symp_{b}'], 5) for b in LABELS[:3]]
his = [np.percentile(df[f'ir_symp_{b}'], 95) for b in LABELS[:3]]
meds = [np.median(df[f'ir_symp_{b}']) for b in LABELS[:3]]
ax.fill_between(x, los, his, alpha=0.25, color='tab:blue', label='model 5-95%')
ax.plot(x, meds, 'o-', color='tab:blue', label='model median')
ax.plot(x, model_ir[best], 'd-', color='k', label=f'best #{br["draw_id"]}')
ax.plot(x, target_ir[:3], 's-', color='red', ms=9, label='MAL-ED')
ax.set_xticks(x); ax.set_xticklabels(LABELS[:3]); ax.set_ylim(0, max(target_ir[:3].max(), 8) * 2)
ax.set_ylabel('Symptomatic IR (per 100 PM)'); ax.set_title('(b) IR-by-age (cohort)')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[1, 0]
ax.hist(rep, bins=40, color='tab:purple', alpha=0.8)
ax.axvline(REPEAT_TARGET, color='red', lw=2, label=f'MAL-ED ~{REPEAT_TARGET:.0%}')
ax.axvline(br['repeat_detected_frac'], color='k', ls='--', label=f'best #{br["draw_id"]}')
ax.set_xlabel('Repeat-infection fraction (of detected children)')
ax.set_ylabel('# draws'); ax.set_title('(c) Repeat fraction across prior vs MAL-ED ~10%')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.scatter(df['par_base_beta'], rep, s=8, alpha=0.4, c='tab:purple')
ax.axhline(REPEAT_TARGET, color='red', lw=1.5, label='MAL-ED ~10%')
ax.set_xscale('log'); ax.set_xlabel('base_beta (log)'); ax.set_ylabel('repeat fraction')
ax.set_title('(d) Repeat fraction vs transmission')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.suptitle(f'Exp 04 — MAL-ED Cohort Emulation (Bangladesh, {len(df)} draws, matched detection+dropout)', fontsize=12)
plt.tight_layout()
fig.savefig(FIGDIR / 'cohort_fit.png', dpi=150, bbox_inches='tight')
print(f'\nSaved figures/cohort_fit.png')
