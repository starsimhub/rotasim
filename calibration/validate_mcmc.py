"""Validate the age emulator-MCMC posterior with REAL ABM sims: draw N points from
posterior_mcmc.csv, simulate each (cohort, real model -- not the emulator), and check the
posterior-predictive lands on the 5 MAL-ED targets. The emulator-MCMC is emulator-approximate
(age repeat/first-inf emulators weak), so this confirms the posterior is trustworthy before VE.
Streams to mcmc_validation.jsonl (imap_unordered -> stragglers don't stall the batch; resumable).

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python validate_mcmc.py --model age --n 400
"""
import os, sys, json, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm
from hm_calibrate import build_sim_config, untransform, CAL_WINDOW, IR_BINS
import process_incidence_maled as P

EXP_DIR = {'age': '18_age_posterior', 'infnum': '19_infnum_posterior'}
N_WORKERS = int(os.environ.get('HM_WORKERS', '118'))
VAL_SEED = 40000
GRID = np.arange(0, 37)


def _km_surv(times, obs, grid):
    times = np.asarray(times, float); obs = np.asarray(obs) == 1
    et = np.unique(times[obs]); s = 1.0; out = []; i = 0
    for g in grid:
        while i < len(et) and et[i] <= g:
            tt = et[i]; d = int(np.sum((times == tt) & obs)); risk = int(np.sum(times >= tt))
            if risk > 0: s *= (1 - d / risk)
            i += 1
        out.append(round(float(s), 5))
    return out


def _one(args):
    i, sim_config, sp, seed = args
    try:
        mo = cm._run_one_replicate((sim_config, sp, int(seed), CAL_WINDOW))
        km = np.asarray(mo.get('km_observed', []))
        return dict(i=int(i),
                    extinct=bool(km.size == 0 or km.mean() < 0.05),
                    repeat=mo.get('repeat_frac'),
                    first_med=float(mo['first_infection']['median']),
                    km_surv=_km_surv(mo.get('km_time', []), mo.get('km_observed', []), GRID),
                    **{f'ir_{b}': float(mo['ir_by_age'].loc[b, 'IR']) for b in IR_BINS})
    except Exception as e:
        return dict(i=int(i), error=repr(e)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='age', choices=['age', 'infnum'])
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--n-agents', type=int, default=40000)
    ap.add_argument('--fix-titer-shape', action='store_true',
                    help='posterior was fit with the titer shape fixed (8 params); '
                         'fill the 4 shape params from FIXED_TITER_SHAPE via untransform')
    ap.add_argument('--exp-dir', default=None,
                    help='experiment folder under experiments/ holding posterior_mcmc.csv '
                         '(e.g. 23_age_titer_fixedshape); defaults to the canonical posterior for --model')
    a = ap.parse_args()
    out_dir = THISDIR / 'experiments' / (a.exp_dir or EXP_DIR[a.model]) / 'outputs'
    post = pd.read_csv(out_dir / 'posterior_mcmc.csv').drop_duplicates().reset_index(drop=True)
    samp = post.sample(min(a.n, len(post)), random_state=0).reset_index(drop=True)
    sim_config = build_sim_config(a.model, a.n_agents)
    tasks = [(i, sim_config, untransform(row, a.model, fix_titer_shape=a.fix_titer_shape), VAL_SEED + i)
             for i, (_, row) in enumerate(samp.iterrows())]
    print(f"validating {a.model} MCMC posterior: {len(tasks)} real sims, {a.n_agents} agents, {N_WORKERS} workers", flush=True)

    jsonl = out_dir / 'mcmc_validation.jsonl'
    done = set()
    if jsonl.exists():
        done = {json.loads(l)['i'] for l in open(jsonl) if l.strip()}
    tasks = [t for t in tasks if t[0] not in done]
    t0 = sc.tic(); n = len(done)
    with get_context('spawn').Pool(processes=min(N_WORKERS, max(len(tasks), 1)), maxtasksperchild=4) as pool:
        for out in pool.imap_unordered(_one, tasks):
            with jsonl.open('a') as f: f.write(json.dumps(out) + '\n')
            n += 1
            if n % 100 == 0: print(f"  {n} done, {sc.toc(t0, output=True):.0f}s", flush=True)

    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    ok = [r for r in recs if 'error' not in r and not r.get('extinct')]
    ext = sum(1 for r in recs if r.get('extinct'))
    tgt = P.load_targets('bangladesh')['ir_by_age']
    print(f"\n{a.model} MCMC posterior validation: {len(recs)} sims, {ext} extinct ({100*ext/max(len(recs),1):.0f}%), {len(ok)} usable")
    print("posterior-predictive (real sims) median [5,95] vs target:")
    for b in IR_BINS:
        v = np.array([r['ir_' + b] for r in ok]); t = tgt.loc[b, 'IR']
        print(f"  IR {b:8s} {np.median(v):5.2f} [{np.percentile(v,5):.2f}, {np.percentile(v,95):.2f}]   target {t:.2f}")
    rv = np.array([r['repeat'] for r in ok if r['repeat'] is not None])
    mv = np.array([r['first_med'] for r in ok if np.isfinite(r['first_med'])])
    print(f"  repeat   {np.median(rv):.3f} [{np.percentile(rv,5):.3f}, {np.percentile(rv,95):.3f}]   target 0.403")
    print(f"  first-inf med {np.median(mv):.2f} [{np.percentile(mv,5):.2f}, {np.percentile(mv,95):.2f}]   target 12.12")


if __name__ == '__main__':
    main()
