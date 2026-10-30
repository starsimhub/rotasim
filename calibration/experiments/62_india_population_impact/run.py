"""Exp 62 -- population-level (herd-inclusive) vaccine impact vs take (0.6-0.95)
across exp58's 6 confirmed-stable age_binned ridge draws, at a fixed realistic
coverage (NFHS-5 Tamil Nadu dose-3 coverage, 66.4%). See README.md for the
resolved-dose-schedule mechanism (ode_model_age_vax.py) and design rationale.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))

from ode_model import ODEParams
import ode_model_age as base
import ode_model_age_vax as vax

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

COVERAGE3 = 0.664  # NFHS-5 Tamil Nadu dose-3 coverage
TAKE_VALUES = [0.6, 0.7, 0.8, 0.9, 0.95]
AGE_LABELS = ['<6m', '6-11m', '12-23m', '24-35m', '36m+']

seeds = [json.loads(l) for l in open(CALIB / 'experiments/58_india_ode_seed_stability/outputs/seed_runs.jsonl')]


def params_from_seed(row):
    return ODEParams(
        base_beta=np.exp(row['log_base_beta']), sus_after_1=row['sus_after_1'],
        sus_r2=row['sus_r2'], sus_r3=row['sus_r3'],
        maternal_titer_median=np.exp(row['log_titer_median']), maternal_titer_gsd=row['titer_gsd'],
        maternal_titer_half_life_days=row['titer_half_life_days'], maternal_hill_slope=row['hill_slope'],
        maternal_immunity_efficacy=row['maternal_efficacy'],
        birth_rate_per_1000=16, death_rate_per_1000=7,
    )


def ir_and_weight_by_age_agg(rows):
    """Collapse the 4 fine <6m sub-bins into one aggregate '<6m' entry
    (n_agents-weighted mean IR, summed population share), matching the
    unvaccinated model's 5-bin output shape for direct comparison. Returns
    (ir_by_bin, pct_of_pop_by_bin) -- the latter used to weight the
    overall/population-wide VE."""
    fine = rows[:4]; coarse = rows[4:]
    n_fine = sum(r['n_agents'] for r in fine)
    ir_fine = sum(r['ir_all_per_100pm'] * r['n_agents'] for r in fine) / n_fine
    pct_fine = sum(r['pct_of_pop'] for r in fine)
    ir = [ir_fine] + [r['ir_all_per_100pm'] for r in coarse]
    pct = [pct_fine] + [r['pct_of_pop'] for r in coarse]
    return ir, pct


results = []
for row in seeds:
    seed = row['seed']
    best_params = row['best_params']
    p = params_from_seed(best_params)

    sol_unvax = base.simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    unvax_rows = base.summarize_by_age(sol_unvax, p)
    # summarize_by_age already returns 5 rows in AGE_LABELS order for the unvaccinated model
    # (ir_and_weight_by_age_agg is only needed to collapse the vax model's finer <6m sub-bins)
    ir_unvax = [r['ir_all_per_100pm'] for r in unvax_rows]
    pct_unvax = [r['pct_of_pop'] for r in unvax_rows]  # demographics are disease-independent here
    # (uniform per-capita deaths, no disease-specific mortality), so the same population
    # weights apply to both the unvaccinated and vaccinated equilibria -- confirmed below.

    print(f"\n=== seed {seed} (logL={row['best_logL']:.2f}) ===")
    print(f"  unvax IR-by-age: {[f'{x:.3f}' for x in ir_unvax]}")

    for take in TAKE_VALUES:
        sol_vax = vax.simulate_age_vax(p, take=take, coverage3=COVERAGE3, n_agents=40_000, years=60, n_eval=400)
        ir_vax, pct_vax = ir_and_weight_by_age_agg(vax.summarize_by_age_vax(sol_vax, p))
        pop_impact_ve = [1 - v / u if u > 0 else float('nan') for v, u in zip(ir_vax, ir_unvax)]

        # ---- overall, population-wide VE: weight each bin's IR by its (vaccinated-
        # equilibrium) population share, not a simple unweighted average across bins ----
        w = np.array(pct_vax) / sum(pct_vax)
        overall_ir_unvax = float(np.dot(w, ir_unvax))
        overall_ir_vax = float(np.dot(w, ir_vax))
        overall_ve = 1 - overall_ir_vax / overall_ir_unvax

        print(f"  take={take}: IR-vax={[f'{x:.3f}' for x in ir_vax]}  pop_VE={[f'{x:.3f}' for x in pop_impact_ve]}"
              f"  OVERALL_VE={overall_ve:.3f}")
        for label, u, v, ve in zip(AGE_LABELS, ir_unvax, ir_vax, pop_impact_ve):
            results.append(dict(seed=seed, logL=row['best_logL'], take=take, coverage3=COVERAGE3,
                                 age_bin=label, ir_unvax=u, ir_vax=v, pop_impact_ve=ve))
        results.append(dict(seed=seed, logL=row['best_logL'], take=take, coverage3=COVERAGE3,
                             age_bin='ALL (population-wide)', ir_unvax=overall_ir_unvax,
                             ir_vax=overall_ir_vax, pop_impact_ve=overall_ve))

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 'population_impact.csv', index=False)
ALL_LABELS = AGE_LABELS + ['ALL (population-wide)']
print(f"\nSaved outputs/population_impact.csv ({len(df)} rows)")

# ---- Figure: population-impact VE vs take, one panel per age bin + one overall panel,
# lines = ridge draws ----
fig, axes = plt.subplots(1, 6, figsize=(24, 4.2), sharey=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
for ax, label in zip(axes, ALL_LABELS):
    sub = df[df.age_bin == label]
    for (seed, ), c in zip([(s,) for s in df['seed'].unique()], colors):
        s2 = sub[sub['seed'] == seed].sort_values('take')
        ax.plot(s2['take'], s2['pop_impact_ve'] * 100, 'o-', color=c, label=f'seed {seed}', markersize=4)
    ax.set_title(label, fontweight='bold' if 'ALL' in label else 'normal')
    ax.set_xlabel('take')
axes[0].set_ylabel('Population-impact VE (%)\n(herd-inclusive, coverage=66.4%)')
axes[-1].legend(fontsize=7, loc='lower right')
plt.suptitle('Exp 62: population-impact VE vs take, across exp58\'s 6 ridge draws', y=1.03)
plt.tight_layout()
plt.savefig(FIG_DIR / 'population_impact_vs_take.png', dpi=140, bbox_inches='tight')
print("Saved figures/population_impact_vs_take.png")

# ---- spread-across-ridge summary, for comparison to exp60's direct-VE spread ----
print("\n=== Spread across the 6 ridge draws, by age bin and take ===")
spread = df.groupby(['age_bin', 'take'])['pop_impact_ve'].agg(['median', 'min', 'max'])
spread['spread_pts'] = (spread['max'] - spread['min']) * 100
print(spread.to_string())
spread.to_csv(OUT_DIR / 'spread_summary.csv')
