"""Post-hoc analysis of exp63's already-completed raw results: population-
impact VE and required-take-for-75% broken down by individual age bin, plus
a population-weighted '<2y' aggregate (<6m + 6-11m + 12-23m), per AK's
request. No new simulations -- reuses outputs/{unvax,vax}_raw.jsonl.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AGE_LABELS = ['<6m', '6-11m', '12-23m', '24-35m', '36m+']
TARGET_VE = 0.75

unvax = {}
for line in open('outputs/unvax_raw.jsonl'):
    d = json.loads(line)
    unvax[(d['seed'], d['foi_scale'])] = (d['ir'], d['pct'])

vax = []
for line in open('outputs/vax_raw.jsonl'):
    d = json.loads(line)
    vax.append(d)

SEED_SUS = {}
for line in open('../58_india_ode_seed_stability/outputs/seed_runs.jsonl'):
    d = json.loads(line)
    SEED_SUS[d['seed']] = dict(sus_r2=d['best_params']['sus_r2'], sus_r3=d['best_params']['sus_r3'])

rows = []
for d in vax:
    seed, foi, cov, take = d['seed'], d['foi_scale'], d['coverage'], d['take']
    ir_unvax, pct_unvax = unvax[(seed, foi)]
    ir_vax, pct_vax = d['ir'], d['pct']

    # individual bins
    for i, label in enumerate(AGE_LABELS):
        ve = 1 - ir_vax[i] / ir_unvax[i] if ir_unvax[i] > 0 else float('nan')
        rows.append(dict(seed=seed, foi_scale=foi, coverage=cov, take=take, age_bin=label, ve=ve))

    # <2y aggregate: population-weighted over <6m, 6-11m, 12-23m (indices 0,1,2)
    w = np.array(pct_vax[:3]) / sum(pct_vax[:3])
    overall_unvax_2y = float(np.dot(w, ir_unvax[:3]))
    overall_vax_2y = float(np.dot(w, ir_vax[:3]))
    ve_2y = 1 - overall_vax_2y / overall_unvax_2y
    rows.append(dict(seed=seed, foi_scale=foi, coverage=cov, take=take, age_bin='<2y (pop-weighted)', ve=ve_2y))

df = pd.DataFrame(rows)
df.to_csv('outputs/by_age_bin_results.csv', index=False)
print(f"Saved outputs/by_age_bin_results.csv ({len(df)} rows)")

FOCUS_BINS = ['6-11m', '<2y (pop-weighted)']
COVERAGE_LEVELS = sorted(df.coverage.unique())
FOI_SCALES = sorted(df.foi_scale.unique(), reverse=True)
seeds_sorted = sorted(SEED_SUS.keys())

# ---- range table, printed ----
for age_bin in FOCUS_BINS:
    print(f"\n=== {age_bin}: VE median [min,max] by coverage x FOI ===")
    for cov in COVERAGE_LEVELS:
        for foi in FOI_SCALES:
            row = []
            for take in [0.6, 0.8, 0.95]:
                sub = df[(df.age_bin == age_bin) & (df.coverage == cov) & (df.foi_scale == foi)
                         & (abs(df['take'] - take) < 0.001)]
                row.append(f"{sub.ve.median()*100:.1f}% [{sub.ve.min()*100:.1f},{sub.ve.max()*100:.1f}]")
            print(f"  cov={cov:5} foi={foi}: take0.6={row[0]:22s} take0.8={row[1]:22s} take0.95={row[2]}")

# ---- required take for 75%, per age bin, per seed, per coverage x FOI ----
thresh_rows = []
for age_bin in FOCUS_BINS + AGE_LABELS:
    for cov in COVERAGE_LEVELS:
        for foi in FOI_SCALES:
            for seed in seeds_sorted:
                sub = df[(df.age_bin == age_bin) & (df.coverage == cov) & (df.foi_scale == foi)
                         & (df.seed == seed)].sort_values('take')
                ve = sub.ve.values; tk = sub['take'].values
                if len(ve) == 0:
                    continue
                if ve.max() < TARGET_VE:
                    req = float('nan'); status = 'not reached by take=0.98'
                elif ve.min() >= TARGET_VE:
                    req = float(tk.min()); status = f'already exceeded at take={tk.min()}'
                else:
                    req = float(np.interp(TARGET_VE, ve, tk)); status = 'interpolated'
                thresh_rows.append(dict(age_bin=age_bin, coverage=cov, foi_scale=foi, seed=seed,
                                         sus_r2=SEED_SUS[seed]['sus_r2'], sus_r3=SEED_SUS[seed]['sus_r3'],
                                         required_take=req, status=status))
thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv('outputs/required_take_for_75pct_by_age.csv', index=False)

print("\n=== Required take for 75%, by age bin (focus bins only) ===")
for age_bin in FOCUS_BINS:
    print(f"\n--- {age_bin} ---")
    sub = thresh_df[thresh_df.age_bin == age_bin].pivot_table(
        index=['seed', 'sus_r2', 'sus_r3'], columns=['coverage', 'foi_scale'], values='required_take')
    print(sub.to_string())

# ---- figure: VE vs take, focus bins, 3x3 grid (rows=FOI, cols=coverage), one panel-set per bin ----
colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_sorted)))
for age_bin in FOCUS_BINS:
    fig, axes = plt.subplots(len(FOI_SCALES), len(COVERAGE_LEVELS), figsize=(13, 11), sharex=True, sharey=True)
    for i, foi in enumerate(FOI_SCALES):
        for j, cov in enumerate(COVERAGE_LEVELS):
            ax = axes[i, j]
            sub = df[(df.age_bin == age_bin) & (df.foi_scale == foi) & (df.coverage == cov)]
            for seed, c in zip(seeds_sorted, colors):
                s2 = sub[sub.seed == seed].sort_values('take')
                ax.plot(s2['take'], s2.ve * 100, 'o-', color=c, markersize=3,
                        label=f'seed {seed}' if (i == 0 and j == 0) else None)
            ax.axhline(TARGET_VE * 100, color='black', ls='--', lw=1, alpha=0.6)
            if i == 0:
                ax.set_title(f'coverage={cov}')
            if j == 0:
                ax.set_ylabel(f'FOI x{foi}\nVE (%)')
            if i == len(FOI_SCALES) - 1:
                ax.set_xlabel('take')
    axes[0, 0].legend(fontsize=6, loc='lower right')
    safe_name = age_bin.split(' ')[0].replace('<', 'lt')
    plt.suptitle(f'Exp 63: {age_bin} VE vs take, by coverage x FOI (dashed = 75% target)', y=1.01)
    plt.tight_layout()
    plt.savefig(f'figures/ve_grid_{safe_name}.png', dpi=140, bbox_inches='tight')
    print(f"\nSaved figures/ve_grid_{safe_name}.png")
