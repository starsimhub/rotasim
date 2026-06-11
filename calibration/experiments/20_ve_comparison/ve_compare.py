"""Exp 20 (Phase B) — VE distribution for one model, propagated from its trajectory-selection
posterior. For each posterior draw: run baseline (no vaccine) vs vaccinated at the SAME seed
(common random numbers -> a paired, variance-reduced VE estimate), measure symptomatic IR
(overall + by age <=36mo), VE = 1 - IR(vax)/IR(novax). Reuses the exp-15 sim machinery
(VaccinePrime, SympIRObserver, _build_run) unchanged; only the parameter source differs
(posterior draws instead of the Optuna point fit).

Run in the pinned env on a 120-core VM, in tmux:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ve_compare.py --model age   --n-ve 800 --response 0.75
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ve_compare.py --model infnum --n-ve 800 --response 0.75
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
POST = {'age':    CALIB / 'experiments' / '18_age_posterior'    / 'outputs' / 'posterior.csv',
        'infnum': CALIB / 'experiments' / '19_infnum_posterior' / 'outputs' / 'posterior.csv'}
VE_BASE = 30000   # paired-seed base; novax & vax of draw i BOTH use VE_BASE + i


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(median=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    return dict(median=float(np.median(x)), lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)), n=int(x.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum'])
    ap.add_argument('--n-ve', type=int, default=800)
    ap.add_argument('--response', type=float, default=0.75)
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
    print(f"VE {a.model}: {len(post)} posterior rows -> {len(uniq)} unique draws; response={a.response}, agents={a.n_agents}", flush=True)

    tasks, meta = [], []
    for i, (_, row) in enumerate(uniq.iterrows()):
        p = untransform(row, a.model); seed = VE_BASE + i
        tasks.append((a.model, p, p['base_beta'], 0.0, False, seed, a.n_agents)); meta.append((i, 'novax'))
        tasks.append((a.model, p, p['base_beta'], a.response, True, seed, a.n_agents)); meta.append((i, 'vax'))

    with get_context('spawn').Pool(processes=min(a.n_workers, len(tasks)), maxtasksperchild=4) as pool:
        outs = pool.map(vt._build_run, tasks)

    by = {}
    for (i, kind), o in zip(meta, outs):
        by.setdefault(i, {})[kind] = o
    rows = []
    for i, d in by.items():
        if 'novax' not in d or 'vax' not in d:
            continue
        nov, vax = d['novax'], d['vax']
        rec = dict(draw=int(i), model=a.model, response=a.response,
                   novax_overall=nov['overall'], vax_overall=vax['overall'],
                   ve_overall=(1 - vax['overall'] / nov['overall']) if nov['overall'] > 0 else float('nan'))
        for b in BINS:
            nb, vb = nov['by_age'][b], vax['by_age'][b]
            rec[f've_{b}'] = (1 - vb / nb) if nb > 0 else float('nan')
        rows.append(rec)
    df = pd.DataFrame(rows)
    suf = '_smoke' if a.smoke else ''
    df.to_csv(HERE / 'outputs' / f'{a.model}_ve_draws{suf}.csv', index=False)

    summ = dict(model=a.model, response=a.response, n_draws=len(df), ve_overall=_ci(df['ve_overall']))
    print(f"\nVE_overall {a.model}: median {summ['ve_overall']['median']:.3f}  "
          f"95% CrI [{summ['ve_overall']['lo']:.3f}, {summ['ve_overall']['hi']:.3f}]  (n={summ['ve_overall']['n']})")
    for b in BINS:
        c = _ci(df[f've_{b}']); summ[f've_{b}'] = c
        print(f"  VE {b:8s}: median {c['median']:.3f}  [{c['lo']:.3f}, {c['hi']:.3f}]")
    json.dump(summ, (HERE / 'outputs' / f'{a.model}_ve_summary{suf}.json').open('w'), indent=2)
    print(f"wrote {a.model}_ve_draws{suf}.csv + summary", flush=True)


if __name__ == '__main__':
    main()
