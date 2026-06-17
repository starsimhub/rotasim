"""Exp 20 (Phase B) — VE distribution for one model, propagated from its trajectory-selection
posterior. For each posterior draw: run baseline (no vaccine) vs vaccinated at the SAME seed
(common random numbers -> a paired, variance-reduced VE estimate), measure symptomatic IR
(overall + by age <=36mo), VE = 1 - IR(vax)/IR(novax). Reuses the exp-15 sim machinery
(VaccinePrime, SympIRObserver, _build_run) unchanged; only the parameter source differs
(posterior draws instead of the Optuna point fit).

Run in the pinned env on a 120-core VM, in tmux:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ve_compare.py --model age   --n-ve 800 --responses 0.63 0.75 0.9
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ve_compare.py --model infnum --n-ve 800 --responses 0.63 0.75 0.9
Overlay the two with plot_ve.py (run after both finish).
"""
import sys, json, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
vt = sc.importbypath(CALIB / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')  # VaccinePrime, SympIRObserver, _build_run, BINS
from hm_calibrate import untransform   # noqa: E402

BINS = vt.BINS
# corrected-maternal clean pair (PR#38 fix): binned age (exp25) + infnum (exp27), overdispersed posteriors
POST = {'age_binned': CALIB / 'experiments' / '25_age_binned_titer_fixedshape'      / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
        'infnum':     CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'}
VE_BASE = 30000   # paired-seed base; novax & vax of draw i BOTH use VE_BASE + i


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(median=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    return dict(median=float(np.median(x)), lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)), n=int(x.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age_binned', 'infnum'])
    ap.add_argument('--n-ve', type=int, default=800)
    ap.add_argument('--responses', type=float, nargs='+', default=[0.63, 0.75, 0.9])  # seroconversion probs to compare (does the model gap change with efficacy?)
    ap.add_argument('--n-agents', type=int, default=40000)
    ap.add_argument('--n-workers', type=int, default=118)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.n_ve, a.n_agents, a.n_workers = 6, 8000, 12

    post = pd.read_csv(POST[a.model])
    uniq = post.drop_duplicates().reset_index(drop=True)      # resampling makes duplicates; run each unique point once
    if len(uniq) > a.n_ve:
        uniq = uniq.sample(a.n_ve, random_state=0).reset_index(drop=True)
    print(f"VE {a.model}: {len(post)} posterior rows -> {len(uniq)} unique draws; responses={a.responses}, agents={a.n_agents}", flush=True)

    # novax once per draw (shared across responses); vax once per (draw, response). All paired on seed = VE_BASE + i.
    tasks, meta = [], []
    for i, (_, row) in enumerate(uniq.iterrows()):
        p = untransform(row, a.model, 'titer', fix_titer_shape=True); seed = VE_BASE + i
        tasks.append((a.model, p, p['base_beta'], 0.0, False, seed, a.n_agents)); meta.append((i, 'novax', None))
        for resp in a.responses:
            tasks.append((a.model, p, p['base_beta'], resp, True, seed, a.n_agents)); meta.append((i, 'vax', resp))

    with get_context('spawn').Pool(processes=min(a.n_workers, len(tasks)), maxtasksperchild=4) as pool:
        outs = pool.map(vt._build_run, tasks)

    novax, vaxr = {}, {}
    for (i, kind, resp), o in zip(meta, outs):
        if kind == 'novax':
            novax[i] = o
        else:
            vaxr[(i, resp)] = o
    rows = []
    for i in novax:
        nov = novax[i]
        for resp in a.responses:
            vax = vaxr.get((i, resp))
            if vax is None:
                continue
            rec = dict(draw=int(i), model=a.model, response=resp,
                       novax_overall=nov['overall'], vax_overall=vax['overall'],
                       ve_overall=(1 - vax['overall'] / nov['overall']) if nov['overall'] > 0 else float('nan'))
            for b in BINS:
                nb, vb = nov['by_age'][b], vax['by_age'][b]
                rec[f've_{b}'] = (1 - vb / nb) if nb > 0 else float('nan')
            rows.append(rec)
    df = pd.DataFrame(rows)
    suf = '_smoke' if a.smoke else ''
    df.to_csv(HERE / 'outputs' / f'{a.model}_ve_draws{suf}.csv', index=False)

    summ = dict(model=a.model, responses=a.responses, by_response={})
    for resp in a.responses:
        sub = df[df.response == resp]
        cr = dict(n_draws=len(sub), ve_overall=_ci(sub['ve_overall']),
                  **{f've_{b}': _ci(sub[f've_{b}']) for b in BINS})
        summ['by_response'][str(resp)] = cr
        o = cr['ve_overall']
        print(f"\nVE_overall {a.model} @resp {resp}: median {o['median']:.3f}  "
              f"95% CrI [{o['lo']:.3f}, {o['hi']:.3f}]  (n={o['n']})")
        for b in BINS:
            c = cr[f've_{b}']
            print(f"  VE {b:8s}: median {c['median']:.3f}  [{c['lo']:.3f}, {c['hi']:.3f}]")
    json.dump(summ, (HERE / 'outputs' / f'{a.model}_ve_summary{suf}.json').open('w'), indent=2)
    print(f"\nwrote {a.model}_ve_draws{suf}.csv + summary", flush=True)


if __name__ == '__main__':
    main()
