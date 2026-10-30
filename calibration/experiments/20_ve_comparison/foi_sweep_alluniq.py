"""Corrected FOI sweep using ALL unique posterior draws (drop_duplicates first).
The original foi_sweep.py sampled n=60 from a 5000-row posterior that is only
5-7% unique; this oversampled near-MLE draws and gave noisy median estimates.
This script deduplicates first (274 age_binned, 373 infnum unique draws) and uses
them all, giving reliable gap estimates across the full posterior spread.

n_agents default reduced to 20000 (from 40000) to keep local runtime ~3.5h on
12 cores; VE medians across 274+ draws are stable even at half the population size.

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python foi_sweep_alluniq.py --model age_binned
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python foi_sweep_alluniq.py --model infnum
Then plot with foi_plot_alluniq.py (overlays corrected vs original).
"""
import sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
vt = sc.importbypath(CALIB / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')
from hm_calibrate import untransform  # noqa: E402

POST = {
    'age_binned': CALIB / 'experiments' / '25_age_binned_titer_fixedshape' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
    'infnum':     CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
    'uk_infnum':  CALIB / 'experiments' / '28_hm_uk_infnum' / 'outputs' / 'hm' / 'uk_infnum_titer_fixedshape' / 'wave1' / 'nroy_samples.csv',
}
# model -> simulation model name (for MODELS lookup in vaccine_toy)
SIM_MODEL = {'uk_infnum': 'infnum'}
# model -> demographics site
SIM_SITE  = {'uk_infnum': 'uk'}

FOI_BASE = 35000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age_binned', 'infnum', 'uk_infnum'])
    ap.add_argument('--response', type=float, default=0.75)
    ap.add_argument('--factors', type=float, nargs='+', default=[1.5, 1.25, 1.0, 0.85, 0.7])
    ap.add_argument('--n-agents', type=int, default=20000,
                    help='agents per sim (default 20000; original exp20 used 40000)')
    ap.add_argument('--n-workers', type=int, default=None,
                    help='parallel workers (default: all cores)')
    ap.add_argument('--moa', choices=['infection_blocking', 'symptom_blocking'],
                    default='infection_blocking')
    ap.add_argument('--tag', default='', help='output filename suffix')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    import multiprocessing
    n_workers = a.n_workers or multiprocessing.cpu_count()

    if a.smoke:
        a.factors, a.n_agents, n_workers = [1.0, 0.85], 8000, 4

    post = pd.read_csv(POST[a.model]).drop_duplicates().reset_index(drop=True)
    if a.smoke:
        post = post.iloc[:4]

    sim_model = SIM_MODEL.get(a.model, a.model)
    site      = SIM_SITE.get(a.model, 'bangladesh')

    print(f"FOI sweep (all-unique) {a.model}: {len(post)} unique draws x "
          f"{len(a.factors)} factors {a.factors}, n_agents={a.n_agents}, "
          f"workers={n_workers}, moa={a.moa}, site={site}", flush=True)

    pair_tasks, meta = [], []
    for i, (_, row) in enumerate(post.iterrows()):
        p = untransform(row, sim_model, 'titer', fix_titer_shape=True)
        for k, f in enumerate(a.factors):
            seed = FOI_BASE + i * 100 + k
            bb = p['base_beta'] * f
            pair_tasks.append((sim_model, p, bb, a.response, seed, a.n_agents, a.moa, 1.0, site))
            meta.append((i, f))

    with get_context('spawn').Pool(processes=min(n_workers, len(pair_tasks)),
                                   maxtasksperchild=4) as pool:
        pair_outs = pool.map(vt._build_pair, pair_tasks)

    rows = []
    for (i, f), (no, vx) in zip(meta, pair_outs):
        if vx is None:
            continue   # novax extinct — skip
        ve = (1 - vx['overall'] / no['overall']) if no['overall'] > 0 else float('nan')
        rows.append(dict(model=a.model, draw=int(i), factor=float(f),
                         age_of_inf=no['first_inf_median'],
                         novax_ir=no['overall'], vax_ir=vx['overall'], ve_overall=ve))
    df = pd.DataFrame(rows)
    suf = ('_smoke' if a.smoke else '') + a.tag
    out = HERE / 'outputs' / f'{a.model}_foi_sweep_alluniq{suf}.csv'  # a.model preserves uk_infnum tag
    df.to_csv(out, index=False)

    print(f"\nby factor (median age-of-inf, median conditional VE, ELIM=0.95):")
    if df.empty:
        print("  (no surviving pairs — all draws extinct at these factors/n_agents)")
    for f in a.factors if not df.empty else []:
        sub = df[df.factor == f]
        alive = sub[sub.novax_ir > 0.1]
        cond = alive[alive.ve_overall <= 0.95]
        print(f"  factor {f:.2f}: age {alive.age_of_inf.median():.1f}mo  "
              f"VE_all {alive.ve_overall.median():.3f}  "
              f"VE_cond {cond.ve_overall.median():.3f}  "
              f"(alive {len(alive)}, cond {len(cond)}/{len(sub)})")
    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
