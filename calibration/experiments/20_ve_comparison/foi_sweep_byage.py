"""FOI sweep outputting age-specific incidence for both arms (novax + vax).
Same simulation logic as foi_sweep_alluniq.py but stores by_age IRs in 4 bins:
  <6 m, 6-11 m, 12-23 m, 24-35 m  (widths: 6, 6, 12, 12 months)

Used for demographic standardization: applying UK case-age weights to the
Bangladesh-calibrated VE-by-age results, isolating the pure demographic
composition effect on population-level VE.

Run (covaguest, after ts32/ts33 finish):
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python foi_sweep_byage.py \\
    --model infnum --factors 0.55 0.60 0.65 0.70 0.80 0.90 1.00 1.10 1.25 1.40 1.60 \\
    --n-workers 118

Then plot with demographic_standardization.py.
"""
import sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc

HERE  = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
vt = sc.importbypath(CALIB / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')
from hm_calibrate import untransform  # noqa: E402

POST = {
    'age_binned': CALIB / 'experiments' / '25_age_binned_titer_fixedshape' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
    'infnum':     CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
}
FOI_BASE = 36000  # distinct seed base from foi_sweep_alluniq.py (35000)

# MALED_AGE_BINS in order, with clean column suffixes and bin widths (months)
BINS      = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
BIN_COLS  = ['0_6',  '6_12',   '12_24',   '24_36']
BIN_WIDTH = [6,       6,        12,         12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age_binned', 'infnum'])
    ap.add_argument('--response', type=float, default=0.75)
    ap.add_argument('--factors', type=float, nargs='+',
                    default=[0.65, 0.70, 0.80, 0.90, 1.00, 1.10])
    ap.add_argument('--n-agents', type=int, default=20000)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--tag',   default='', help='output filename suffix')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    import multiprocessing
    n_workers = a.n_workers or multiprocessing.cpu_count()

    if a.smoke:
        a.factors, a.n_agents, n_workers = [1.0, 0.70], 8000, 4

    post = pd.read_csv(POST[a.model]).drop_duplicates().reset_index(drop=True)
    if a.smoke:
        post = post.iloc[:4]

    print(f"FOI byage sweep {a.model}: {len(post)} draws x {len(a.factors)} factors, "
          f"n_agents={a.n_agents}, workers={n_workers}", flush=True)

    pair_tasks, meta = [], []
    for i, (_, row) in enumerate(post.iterrows()):
        p = untransform(row, a.model, 'titer', fix_titer_shape=True)
        for k, f in enumerate(a.factors):
            seed = FOI_BASE + i * 100 + k
            bb = p['base_beta'] * f
            pair_tasks.append((a.model, p, bb, a.response, seed, a.n_agents))
            meta.append((i, f))

    with get_context('spawn').Pool(processes=min(n_workers, len(pair_tasks)),
                                   maxtasksperchild=4) as pool:
        pair_outs = pool.map(vt._build_pair, pair_tasks)

    rows = []
    for (i, f), (no, vx) in zip(meta, pair_outs):
        if vx is None:
            continue   # novax extinct — skip
        ve = (1 - vx['overall'] / no['overall']) if no['overall'] > 0 else float('nan')
        row = dict(model=a.model, draw=int(i), factor=float(f),
                   age_of_inf=no['first_inf_median'],
                   novax_ir=no['overall'], vax_ir=vx['overall'], ve_overall=ve)
        # store age-specific IRs for both arms
        for b, col in zip(BINS, BIN_COLS):
            row[f'novax_ir_{col}'] = no['by_age'].get(b, 0.0)
            row[f'vax_ir_{col}']   = vx['by_age'].get(b, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    suf = ('_smoke' if a.smoke else '') + a.tag
    out = HERE / 'outputs' / f'{a.model}_foi_sweep_byage{suf}.csv'
    df.to_csv(out, index=False)

    # quick summary: case distribution at LMIC (factor~1.0) and HIC-ish (factor~0.70)
    for fac in [1.0, 0.70]:
        sub = df[(df.factor == fac) & (df.novax_ir > 0.1)]
        if len(sub) == 0:
            continue
        total = sum(sub[f'novax_ir_{c}'].median() * w for c, w in zip(BIN_COLS, BIN_WIDTH))
        if total == 0:
            continue
        print(f"\nfactor={fac:.2f} (n={len(sub)}) — novax case distribution:")
        for c, w, b in zip(BIN_COLS, BIN_WIDTH, BINS):
            frac = sub[f'novax_ir_{c}'].median() * w / total
            print(f"  {b:10s}: {frac:.1%}")

    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
