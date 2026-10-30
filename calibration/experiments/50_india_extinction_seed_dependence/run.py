"""Exp 50 -- India / Vellore: is extinction (or time-to-extinction) seed-dependent?

10 parameter sets from exp39's NROY pool, all classified extinct on their
original single seed, spanning nearly the full log_base_beta range explored.
For each, run 10 fresh seeds (100 sims total) and record extinction status +
timing, using the n_infected_series diagnostic field. See README.md.

Run on zebra (160 cores, non-spot) -- check `uptime`/`ps` for other users first:
  MALED_SITE=india NEO_PRIME=1 ~/ukvenv/bin/python \\
    experiments/50_india_extinction_seed_dependence/run.py \\
    > experiments/50_india_extinction_seed_dependence/outputs/run.log 2>&1
"""
import sys, os, json, pathlib
from multiprocessing import get_context

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
os.chdir(CALIB)

import numpy as np, pandas as pd
import hm_calibrate as H
import calibrate_maled as cm

OUT_DIR = HERE / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Original exp39 NROY indices, all classified extinct on their single original
# seed, spanning nearly the full log_base_beta range explored (see README.md).
SELECTED_IDX = [2320, 1720, 852, 1454, 2287, 2306, 2538, 1841, 104, 1414]
N_SEEDS = 10
SEED_BASE = 5000   # seeds 5000-5099, independent of the original NROY seeds

nroy = pd.read_csv(CALIB / 'experiments/39_india_age_binned_fixed/outputs/ts/nroy_draw.csv')


def _run_one(args):
    orig_idx, seed, sp, cfg = args
    mo = cm._run_one_replicate((cfg, sp, seed, H.CAL_WINDOW))
    n_inf = np.array(mo['n_infected_series'])
    extinct = bool(n_inf[-1] == 0)
    nz = np.nonzero(n_inf > 0)[0]
    last_active_year = float(nz[-1] / 365.25) if len(nz) else 0.0
    peak_day = int(np.argmax(n_inf))
    peak_n = int(n_inf.max())
    return dict(orig_idx=int(orig_idx), seed=int(seed), extinct=extinct,
                last_active_year=last_active_year, peak_day=peak_day, peak_n=peak_n,
                ir_sum=float(mo['ir_by_age']['IR'].sum()))


if __name__ == '__main__':
    cfg = H.build_sim_config('age_binned', 40000, 'titer')
    tasks = []
    for orig_idx in SELECTED_IDX:
        row = nroy.loc[orig_idx]
        sp = H.untransform(row, 'age_binned', 'titer', fix_age_psymp=True)
        for s in range(N_SEEDS):
            tasks.append((orig_idx, SEED_BASE + s, sp, cfg))

    n_workers = int(os.environ.get('HM_WORKERS', os.cpu_count() or 20))
    print(f"Running {len(tasks)} sims ({len(SELECTED_IDX)} param sets x {N_SEEDS} seeds) "
          f"with {n_workers} workers", flush=True)

    results = []
    out_path = OUT_DIR / 'extinction_seed_results.jsonl'
    with open(out_path, 'w') as fout:
        with get_context('spawn').Pool(processes=min(n_workers, len(tasks)), maxtasksperchild=4) as pool:
            for r in pool.imap_unordered(_run_one, tasks):
                results.append(r)
                fout.write(json.dumps(r) + '\n')
                fout.flush()
                print(f"  orig_idx={r['orig_idx']} seed={r['seed']} extinct={r['extinct']} "
                      f"last_active_year={r['last_active_year']:.2f} peak_n={r['peak_n']}", flush=True)

    print(f"\nDone. {len(results)} results written to {out_path}", flush=True)

    # Quick summary per parameter set
    df = pd.DataFrame(results)
    summary = df.groupby('orig_idx').agg(
        n=('extinct', 'size'),
        n_extinct=('extinct', 'sum'),
        median_last_active_year=('last_active_year', 'median'),
        min_last_active_year=('last_active_year', 'min'),
        max_last_active_year=('last_active_year', 'max'),
    )
    summary['frac_extinct'] = summary['n_extinct'] / summary['n']
    summary.to_csv(OUT_DIR / 'summary_by_param.csv')
    print(summary)
