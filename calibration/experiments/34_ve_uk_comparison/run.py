"""exp34 — FOI-gradient sweep using UK-fitted infnum parameters (exp 28 posterior).
Same logic as exp20 foi_sweep.py but draws from UK posterior_hmreweight.csv instead of
Bangladesh posterior. Run both MOAs on covaguest, then plot vs Bangladesh draws for comparison.

Launch commands (covaguest, rota-hm env):
  # infection-blocking
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run.py --moa infection_blocking

  # symptom-blocking
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run.py --moa symptom_blocking --tag _sb

Then plot locally:
  python plot.py
"""
import sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
vt = sc.importbypath(CALIB / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')
from hm_calibrate import untransform  # noqa: E402

UK_POST = CALIB / 'experiments' / '28_hm_uk_infnum' / 'outputs' / 'posterior_hmreweight.csv'
FOI_BASE = 34000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=60)
    ap.add_argument('--response', type=float, default=0.75)
    ap.add_argument('--factors', type=float, nargs='+', default=[1.5, 1.25, 1.0, 0.85, 0.7])
    ap.add_argument('--n-agents', type=int, default=40000)
    ap.add_argument('--n-workers', type=int, default=118)
    ap.add_argument('--moa', choices=['infection_blocking', 'symptom_blocking'], default='infection_blocking')
    ap.add_argument('--tag', default='', help='output filename suffix, e.g. _sb')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.n_draws, a.factors, a.n_agents, a.n_workers = 4, [1.0, 0.85], 8000, 12

    post = pd.read_csv(UK_POST).drop_duplicates().reset_index(drop=True)
    if len(post) > a.n_draws:
        post = post.sample(a.n_draws, random_state=0).reset_index(drop=True)
    print(f"UK FOI sweep infnum: {len(post)} draws x {len(a.factors)} beta-factors {a.factors}, "
          f"response={a.response}, moa={a.moa}", flush=True)

    tasks, meta = [], []
    for i, (_, row) in enumerate(post.iterrows()):
        p = untransform(row, 'infnum', 'titer', fix_titer_shape=True)
        for k, f in enumerate(a.factors):
            seed = FOI_BASE + i * 100 + k
            bb = p['base_beta'] * f
            tasks.append(('infnum', p, bb, 0.0, False, seed, a.n_agents, a.moa, 1.0))
            meta.append((i, f, 'novax'))
            tasks.append(('infnum', p, bb, a.response, True, seed, a.n_agents, a.moa, 1.0))
            meta.append((i, f, 'vax'))

    with get_context('spawn').Pool(processes=min(a.n_workers, len(tasks)), maxtasksperchild=4) as pool:
        outs = pool.map(vt._build_run, tasks)

    nov, vax = {}, {}
    for (i, f, kind), o in zip(meta, outs):
        (nov if kind == 'novax' else vax)[(i, f)] = o

    rows = []
    for (i, f), no in nov.items():
        vx = vax.get((i, f))
        if vx is None:
            continue
        ve = (1 - vx['overall'] / no['overall']) if no['overall'] > 0 else float('nan')
        rows.append(dict(draw=int(i), factor=float(f), age_of_inf=no['first_inf_median'],
                         novax_ir=no['overall'], vax_ir=vx['overall'], ve_overall=ve))
    df = pd.DataFrame(rows)
    suf = ('_smoke' if a.smoke else '') + a.tag
    out = HERE / 'outputs' / f'uk_infnum_foi_sweep{suf}.csv'
    df.to_csv(out, index=False)

    print("\nby beta-factor (median age-of-infection, median VE):")
    for f in a.factors:
        sub = df[df.factor == f]
        alive = sub[sub.novax_ir > 0.1]
        print(f"  factor {f:4.2f}: age-of-inf {alive.age_of_inf.median():5.1f}mo  "
              f"VE {alive.ve_overall.median():.3f}  (alive {len(alive)}/{len(sub)})")
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
