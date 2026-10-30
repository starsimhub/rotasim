"""Exp 63 -- population impact vs take, across coverage levels (0.40/0.664/0.85)
and FOI scenarios (base_beta x 1.0/0.85/0.7), across exp58's 6 ridge draws.
Reuses exp62's ode_model_age_vax.py mechanism unchanged. See README.md.

Runs the full grid via one multiprocessing.Pool (not the repeated-pool-
creation pattern that caused exp58's file-descriptor bug -- this creates
the pool exactly once for the whole grid).
"""
import os
# MUST be set before numpy is imported anywhere (including transitively via
# pandas/scipy below): the vax model's ~440-state Jacobian is large enough to
# cross OpenBLAS's internal auto-threading threshold (unlike the ~110-state
# unvax model, which stays single-threaded and never hit this). A forked
# multiprocessing worker that later performs its OWN first large BLAS
# operation can deadlock initializing its per-process thread pool post-fork
# -- a well-known threaded-BLAS + fork() gotcha. Confirmed by direct
# reproduction on zebra: compute_vax hung indefinitely via Pool.imap_unordered
# (even as the first and only pool usage) without this, and completed in ~3s
# with it. Forcing single-threaded BLAS also avoids core oversubscription
# now that the OS-level Pool already parallelizes across all cores.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import sys, pathlib, json, time
import numpy as np, pandas as pd
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE.parents[0] / '62_india_population_impact'))

from ode_model import ODEParams
import ode_model_age as base
import ode_model_age_vax as vax

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

TAKE_VALUES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
COVERAGE_LEVELS = [0.40, 0.664, 0.85]
FOI_SCALES = [1.0, 0.85, 0.7]
AGE_LABELS = ['<6m', '6-11m', '12-23m', '24-35m', '36m+']
TARGET_VE = 0.75

SEEDS = [json.loads(l) for l in open(CALIB / 'experiments/58_india_ode_seed_stability/outputs/seed_runs.jsonl')]
SEED_PARAMS = {r['seed']: r['best_params'] for r in SEEDS}
SEED_LOGL = {r['seed']: r['best_logL'] for r in SEEDS}


def params_from_seed(best_params, foi_scale=1.0):
    return ODEParams(
        base_beta=np.exp(best_params['log_base_beta']) * foi_scale,
        sus_after_1=best_params['sus_after_1'], sus_r2=best_params['sus_r2'], sus_r3=best_params['sus_r3'],
        maternal_titer_median=np.exp(best_params['log_titer_median']), maternal_titer_gsd=best_params['titer_gsd'],
        maternal_titer_half_life_days=best_params['titer_half_life_days'], maternal_hill_slope=best_params['hill_slope'],
        maternal_immunity_efficacy=best_params['maternal_efficacy'],
        birth_rate_per_1000=16, death_rate_per_1000=7,
    )


def ir_agg(rows):
    fine = rows[:4]; coarse = rows[4:]
    n_fine = sum(r['n_agents'] for r in fine)
    ir_fine = sum(r['ir_all_per_100pm'] * r['n_agents'] for r in fine) / n_fine
    pct_fine = sum(r['pct_of_pop'] for r in fine)
    return [ir_fine] + [r['ir_all_per_100pm'] for r in coarse], [pct_fine] + [r['pct_of_pop'] for r in coarse]


def compute_unvax(args):
    seed, foi_scale = args
    p = params_from_seed(SEED_PARAMS[seed], foi_scale)
    sol = base.simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    rows = base.summarize_by_age(sol, p)
    ir = [r['ir_all_per_100pm'] for r in rows]; pct = [r['pct_of_pop'] for r in rows]
    return dict(seed=seed, foi_scale=foi_scale, ir=ir, pct=pct)


def compute_vax(args):
    seed, foi_scale, coverage, take = args
    p = params_from_seed(SEED_PARAMS[seed], foi_scale)
    sol = vax.simulate_age_vax(p, take=take, coverage3=coverage, n_agents=40_000, years=60, n_eval=400)
    ir, pct = ir_agg(vax.summarize_by_age_vax(sol, p))
    return dict(seed=seed, foi_scale=foi_scale, coverage=coverage, take=take, ir=ir, pct=pct)


UNVAX_RAW = OUT_DIR / 'unvax_raw.jsonl'
VAX_RAW = OUT_DIR / 'vax_raw.jsonl'


def _load_done(path, keys):
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        d = json.loads(line)
        out[tuple(d[k] for k in keys)] = d
    return out


if __name__ == '__main__':
    unvax_tasks = [(seed, foi) for seed in SEED_PARAMS for foi in FOI_SCALES]
    vax_tasks = [(seed, foi, cov, take) for seed in SEED_PARAMS for foi in FOI_SCALES
                 for cov in COVERAGE_LEVELS for take in TAKE_VALUES]

    # resumable: skip tasks already written to the raw JSONL files (so a kill
    # + relaunch, e.g. to move machines, doesn't lose completed work)
    done_unvax = _load_done(UNVAX_RAW, ['seed', 'foi_scale'])
    done_vax = _load_done(VAX_RAW, ['seed', 'foi_scale', 'coverage', 'take'])
    unvax_todo = [t for t in unvax_tasks if t not in done_unvax]
    vax_todo = [t for t in vax_tasks if t not in done_vax]
    print(f"Resuming: {len(done_unvax)}/{len(unvax_tasks)} unvax and {len(done_vax)}/{len(vax_tasks)} "
          f"vax tasks already done.")
    print(f"Running {len(unvax_todo)} unvax + {len(vax_todo)} vax sims via multiprocessing.Pool "
          f"({mp.cpu_count()} cores), writing incrementally...")
    t0 = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        with UNVAX_RAW.open('a') as f:
            for i, res in enumerate(pool.imap_unordered(compute_unvax, unvax_todo)):
                f.write(json.dumps(res) + '\n'); f.flush()
                if (i + 1) % 5 == 0 or i + 1 == len(unvax_todo):
                    print(f"  unvax {i+1}/{len(unvax_todo)}  [{time.time()-t0:.0f}s]")
        with VAX_RAW.open('a') as f:
            for i, res in enumerate(pool.imap_unordered(compute_vax, vax_todo)):
                f.write(json.dumps(res) + '\n'); f.flush()
                if (i + 1) % 20 == 0 or i + 1 == len(vax_todo):
                    print(f"  vax {i+1}/{len(vax_todo)}  [{time.time()-t0:.0f}s]")
    print(f"Done in {time.time()-t0:.0f}s")

    unvax_results = {(d['seed'], d['foi_scale']): (d['ir'], d['pct'])
                      for d in _load_done(UNVAX_RAW, ['seed', 'foi_scale']).values()}
    vax_results = {(d['seed'], d['foi_scale'], d['coverage'], d['take']): (d['ir'], d['pct'])
                    for d in _load_done(VAX_RAW, ['seed', 'foi_scale', 'coverage', 'take']).values()}

    rows_out = []
    for (seed, foi, cov, take), (ir_vax, pct_vax) in vax_results.items():
        ir_unvax, pct_unvax = unvax_results[(seed, foi)]
        pop_ve = [1 - v / u if u > 0 else float('nan') for v, u in zip(ir_vax, ir_unvax)]
        w = np.array(pct_vax) / sum(pct_vax)
        overall_unvax = float(np.dot(w, ir_unvax)); overall_vax = float(np.dot(w, ir_vax))
        overall_ve = 1 - overall_vax / overall_unvax
        sus_r3 = SEED_PARAMS[seed]['sus_r3']
        for label, ve in zip(AGE_LABELS, pop_ve):
            rows_out.append(dict(seed=seed, logL=SEED_LOGL[seed], sus_r3=sus_r3, foi_scale=foi,
                                  coverage=cov, take=take, age_bin=label, pop_impact_ve=ve))
        rows_out.append(dict(seed=seed, logL=SEED_LOGL[seed], sus_r3=sus_r3, foi_scale=foi,
                              coverage=cov, take=take, age_bin='ALL (population-wide)', pop_impact_ve=overall_ve))

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_DIR / 'sweep_results.csv', index=False)
    print(f"Saved outputs/sweep_results.csv ({len(df)} rows)")

    # ---- Figure 1: 3x3 grid (rows=FOI scale, cols=coverage), population-wide VE vs take,
    # one line per ridge draw, dashed line at TARGET_VE ----
    df_all = df[df.age_bin == 'ALL (population-wide)']
    seeds_sorted = sorted(SEED_PARAMS.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_sorted)))
    fig, axes = plt.subplots(len(FOI_SCALES), len(COVERAGE_LEVELS), figsize=(13, 11), sharex=True, sharey=True)
    for i, foi in enumerate(FOI_SCALES):
        for j, cov in enumerate(COVERAGE_LEVELS):
            ax = axes[i, j]
            sub = df_all[(df_all.foi_scale == foi) & (df_all.coverage == cov)]
            for seed, c in zip(seeds_sorted, colors):
                s2 = sub[sub.seed == seed].sort_values('take')
                ax.plot(s2['take'], s2['pop_impact_ve'] * 100, 'o-', color=c, markersize=3,
                        label=f'seed {seed}' if (i == 0 and j == 0) else None)
            ax.axhline(TARGET_VE * 100, color='black', ls='--', lw=1, alpha=0.6)
            if i == 0:
                ax.set_title(f'coverage={cov}')
            if j == 0:
                ax.set_ylabel(f'FOI x{foi}\nPop. VE (%)')
            if i == len(FOI_SCALES) - 1:
                ax.set_xlabel('take')
    axes[0, 0].legend(fontsize=6, loc='lower right')
    plt.suptitle(f'Exp 63: population-wide VE vs take, by coverage x FOI scenario\n'
                 f'(dashed line = {int(TARGET_VE*100)}% target)', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 've_grid_coverage_foi.png', dpi=140, bbox_inches='tight')
    print("Saved figures/ve_grid_coverage_foi.png")

    # ---- Threshold-take-to-hit-75% calculation, per (seed, coverage, foi_scale) ----
    thresh_rows = []
    for foi in FOI_SCALES:
        for cov in COVERAGE_LEVELS:
            for seed in seeds_sorted:
                s2 = df_all[(df_all.foi_scale == foi) & (df_all.coverage == cov) & (df_all.seed == seed)].sort_values('take')
                ve = s2['pop_impact_ve'].values; tk = s2['take'].values
                if ve.max() < TARGET_VE:
                    req_take = float('nan'); status = 'not reached by take=0.98'
                elif ve.min() >= TARGET_VE:
                    req_take = float(tk.min()); status = f'already exceeded at take={tk.min()}'
                else:
                    req_take = float(np.interp(TARGET_VE, ve, tk))  # ve is monotonic increasing in take
                    status = 'interpolated'
                thresh_rows.append(dict(foi_scale=foi, coverage=cov, seed=seed, sus_r3=SEED_PARAMS[seed]['sus_r3'],
                                         required_take=req_take, status=status))
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(OUT_DIR / 'required_take_for_75pct.csv', index=False)
    print(f"\nSaved outputs/required_take_for_75pct.csv")
    print(thresh_df.to_string(index=False))

    # ---- Figure 2: required take vs sus_r3, faceted by coverage, colored by FOI scale ----
    fig2, axes2 = plt.subplots(1, len(COVERAGE_LEVELS), figsize=(15, 4.5), sharey=True)
    foi_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(FOI_SCALES)))
    for j, cov in enumerate(COVERAGE_LEVELS):
        ax = axes2[j]
        for foi, c in zip(FOI_SCALES, foi_colors):
            sub = thresh_df[(thresh_df.coverage == cov) & (thresh_df.foi_scale == foi)].sort_values('sus_r3')
            ax.plot(sub['sus_r3'], sub['required_take'], 'o-', color=c, label=f'FOI x{foi}')
        ax.set_title(f'coverage={cov}')
        ax.set_xlabel('sus_r3')
        ax.set_ylim(0.45, 1.0)
    axes2[0].set_ylabel(f'Take required to hit {int(TARGET_VE*100)}% population VE')
    axes2[-1].legend(fontsize=8)
    plt.suptitle('Exp 63: required take to hit 75% population impact, vs sus_r3', y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'required_take_vs_sus_r3.png', dpi=140, bbox_inches='tight')
    print("Saved figures/required_take_vs_sus_r3.png")
