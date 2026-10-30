"""exp20 FOI-gradient sweep: map achieved VE vs simulated age-of-infection for one model.
Scales base_beta across factors (a what-if FOI axis; all other params from the posterior draw,
so NOT a re-calibration). For each draw x factor: novax + vax@response (infection-blocking),
paired seed. VE = 1 - IR(vax)/IR(novax); age-of-infection = novax median first-infection age.
Tests (1) does achieved VE rise as infection shifts older (lower FOI), (2) do age vs infnum
diverge along that gradient. Extinction caps the low-FOI end (factors that go extinct -> dropped).

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python foi_sweep.py --model age_binned
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python foi_sweep.py --model infnum
"""
import sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
HERE = pathlib.Path(__file__).resolve().parent; CALIB = HERE.parents[1]; sys.path.insert(0, str(CALIB))
vt = sc.importbypath(CALIB / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')
from hm_calibrate import untransform  # noqa: E402
POST = {'age_binned': CALIB / 'experiments' / '25_age_binned_titer_fixedshape' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
        'infnum':     CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'}
FOI_BASE = 31000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age_binned', 'infnum'])
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

    post = pd.read_csv(POST[a.model]).drop_duplicates().reset_index(drop=True)
    if len(post) > a.n_draws:
        post = post.sample(a.n_draws, random_state=0).reset_index(drop=True)
    print(f"FOI sweep {a.model}: {len(post)} draws x {len(a.factors)} beta-factors {a.factors}, response={a.response}", flush=True)

    tasks, meta = [], []
    for i, (_, row) in enumerate(post.iterrows()):
        p = untransform(row, a.model, 'titer', fix_titer_shape=True)
        for k, f in enumerate(a.factors):
            seed = FOI_BASE + i * 100 + k; bb = p['base_beta'] * f
            tasks.append((a.model, p, bb, 0.0, False, seed, a.n_agents, a.moa, 1.0)); meta.append((i, f, 'novax'))
            tasks.append((a.model, p, bb, a.response, True, seed, a.n_agents, a.moa, 1.0)); meta.append((i, f, 'vax'))

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
        rows.append(dict(model=a.model, draw=int(i), factor=float(f), age_of_inf=no['first_inf_median'],
                         novax_ir=no['overall'], vax_ir=vx['overall'], ve_overall=ve))
    df = pd.DataFrame(rows)
    suf = ('_smoke' if a.smoke else '') + a.tag
    df.to_csv(HERE / 'outputs' / f'{a.model}_foi_sweep{suf}.csv', index=False)
    print("\nby beta-factor (median age-of-infection, median VE; novax IR shows extinction):")
    for f in a.factors:
        sub = df[df.factor == f]
        alive = sub[sub.novax_ir > 0.1]
        print(f"  factor {f:4.2f}: age-of-inf {alive.age_of_inf.median():5.1f}mo  VE {alive.ve_overall.median():.3f}  "
              f"(novax IR {sub.novax_ir.median():.2f}, n_alive {len(alive)}/{len(sub)})")
    print(f"wrote {a.model}_foi_sweep{suf}.csv", flush=True)


if __name__ == '__main__':
    main()
