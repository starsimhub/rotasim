"""Exp 51 follow-on -- India / Vellore: does the CCS-rescue effect strengthen
closer to the viable beta corridor?

orig_idx=1072 (base_beta=0.107, ~1.4x above orig_idx=1720's 0.076) was chosen
as the closest-matching sibling to 1720 on every OTHER dimension (sus_after_1,
sus_r2/r3, titer params, maternal_efficacy all within ~1 std of 1720) so that
beta is close to the only thing changing. 1720 showed partial rescue at
N=200k (2/10 extinct->survived by seed 6000 at N=100k tier onward); this
checks whether a further step towards the viable corridor rescues more
strongly, per AK's note that orig_idx=2320 (lowest beta, 0.050) was probably
the wrong pick for "most likely to be rescued" since a too-low-R0 point
shouldn't respond to population size at all. See README.md and SUMMARY.md.

Run on zebra (160 cores, non-spot):
  MALED_SITE=india NEO_PRIME=1 ~/ukvenv/bin/python \\
    experiments/51_india_population_size/run_1072.py \\
    > experiments/51_india_population_size/outputs/run_1072.log 2>&1
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

SELECTED_IDX = [1072]   # base_beta=0.107, closest sibling to 1720 (0.076) on all other dims
N_AGENTS_LIST = [100_000, 200_000, 400_000]
N_SEEDS = 10
SEED_BASE = 6100   # distinct from the 6000-6009 range used for 2320/1720

nroy = pd.read_csv(CALIB / 'experiments/39_india_age_binned_fixed/outputs/ts/nroy_draw.csv')


def _run_one(args):
    orig_idx, n_agents, seed, sp = args
    cfg = H.build_sim_config('age_binned', n_agents, 'titer')
    mo = cm._run_one_replicate((cfg, sp, seed, H.CAL_WINDOW))
    n_inf = np.array(mo['n_infected_series'])
    extinct = bool(n_inf[-1] == 0)
    nz = np.nonzero(n_inf > 0)[0]
    last_active_year = float(nz[-1] / 365.25) if len(nz) else 0.0
    peak_n = int(n_inf.max())
    return dict(orig_idx=int(orig_idx), n_agents=int(n_agents), seed=int(seed),
                extinct=extinct, last_active_year=last_active_year, peak_n=peak_n,
                peak_frac=peak_n / n_agents,
                ir_sum=float(mo['ir_by_age']['IR'].sum()))


WORKERS_BY_N = {100_000: 20, 200_000: 13, 400_000: 6}

if __name__ == '__main__':
    param_sps = {}
    for orig_idx in SELECTED_IDX:
        row = nroy.loc[orig_idx]
        param_sps[orig_idx] = H.untransform(row, 'age_binned', 'titer', fix_age_psymp=True)

    results = []
    out_path = OUT_DIR / 'popsize_results_1072.jsonl'
    with open(out_path, 'w') as fout:
        for n_agents in N_AGENTS_LIST:
            tasks = [(orig_idx, n_agents, SEED_BASE + s, param_sps[orig_idx])
                     for orig_idx in SELECTED_IDX for s in range(N_SEEDS)]
            n_workers = WORKERS_BY_N[n_agents]
            print(f"\n=== n_agents={n_agents}: {len(tasks)} sims, {n_workers} workers ===", flush=True)
            with get_context('spawn').Pool(processes=min(n_workers, len(tasks)), maxtasksperchild=2) as pool:
                for r in pool.imap_unordered(_run_one, tasks):
                    results.append(r)
                    fout.write(json.dumps(r) + '\n')
                    fout.flush()
                    print(f"  orig_idx={r['orig_idx']} n_agents={r['n_agents']} seed={r['seed']} "
                          f"extinct={r['extinct']} last_active_year={r['last_active_year']:.2f} "
                          f"peak_frac={r['peak_frac']:.3f}", flush=True)

    print(f"\nDone. {len(results)} results written to {out_path}", flush=True)
    df = pd.DataFrame(results)
    summary = df.groupby(['orig_idx', 'n_agents']).agg(
        n=('extinct', 'size'), n_extinct=('extinct', 'sum'),
        median_last_active_year=('last_active_year', 'median'),
        max_last_active_year=('last_active_year', 'max'),
        median_peak_frac=('peak_frac', 'median'),
    )
    summary['frac_extinct'] = summary['n_extinct'] / summary['n']
    summary.to_csv(OUT_DIR / 'summary_by_popsize_1072.csv')
    print(summary)
