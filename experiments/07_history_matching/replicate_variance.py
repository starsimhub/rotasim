"""
Exp 07 (prerequisite) — replicate-variance + winner's-curse check.

Runs the exp-06 best-fit parameter point at many seeds to:
  (a) confirm the good IR fit reproduces (not a 1-draw fluke), and
  (b) measure the model's stochastic SD per observable -> the MODEL-variance
      component of each HM target (added to the observational/sampling SD).

Usage: uv run python experiments/07_history_matching/replicate_variance.py --n-seeds 16 --n-workers 16
"""
import os, json, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, sciris as sc, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
exp06 = sc.importbypath(REPO / 'experiments' / '06_titer_maternal_peak' / 'run.py')

# exp-06 best-fit point (lowest IR log-SSE among persistent draws).
BEST = dict(base_beta=0.22462, young_reservoir=2.5018, infant_exposure=3.69496,
            adult_contacts=1.77815, sus_after_1=0.95401, sus_after_2=0.8694,
            sus_after_3plus=0.77104, maternal_efficacy=0.96788, titer_median=41.2415,
            titer_gsd=2.57797, titer_half_life_days=67.61863, hill_slope=6.20551,
            p_symp_1=0.55476, p_symp_2=0.39218, p_symp_3plus=0.07273)
TARGET = {'ir_symp_<6 m': 1.91, 'ir_symp_6-11 m': 5.37, 'ir_symp_12-23 m': 2.35,
          'repeat_detected_frac': 0.43, 'frac_ever_detected': 0.638}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=16)
    ap.add_argument('--n-agents', type=int, default=40_000)
    ap.add_argument('--n-workers', type=int, default=16)
    args = ap.parse_args()

    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    cens = fi.loc[fi['event_observed'] == 0, 'age_event_months'].dropna().values
    cens = cens[cens > 0]
    tasks = [(i, BEST, args.n_agents, 1000 + i, cens, 0.5) for i in range(args.n_seeds)]

    t0 = sc.tic()
    with get_context('spawn').Pool(processes=min(args.n_workers, args.n_seeds)) as pool:
        recs = [r for r in pool.imap_unordered(exp06._run_one, tasks)]
    ok = [r for r in recs if r['ok']]
    print(f'{len(ok)}/{len(recs)} ok in {sc.toc(t0, output=True):.0f}s\n')

    # Extinctions (ever-infected ~0) are treated as model FAILURES (NaN in HM),
    # so the relevant variance is conditional on persistence.
    ever = np.array([r['frac_ever_infected'] for r in ok])
    persist = ever > 0.05
    print(f'EXTINCTION: {(~persist).sum()}/{len(ok)} seeds extinct ({100*(~persist).mean():.0f}%); '
          f'{persist.sum()} persisted\n')
    cols = ['ir_symp_<6 m', 'ir_symp_6-11 m', 'ir_symp_12-23 m',
            'repeat_detected_frac', 'frac_ever_detected', 'true_first_median']
    print(f'{"observable":>20} | {"ALL mean":>8} {"CV":>5} | {"PERSIST mean":>12} {"SD":>7} {"CV":>5} | {"target":>7}')
    out = {}
    for c in cols:
        v = np.array([r[c] for r in ok], float)
        vp = v[persist]; vp = vp[~np.isnan(vp)]
        ma, cva = float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / max(np.nanmean(v), 1e-9))
        mp, sdp = float(vp.mean()), float(vp.std(ddof=1))
        cvp = sdp / mp if mp else float('nan')
        out[c] = dict(persist_mean=round(mp, 4), persist_sd=round(sdp, 4), persist_cv=round(cvp, 3))
        print(f'{c:>20} | {ma:>8.3f} {cva:>5.2f} | {mp:>12.3f} {sdp:>7.3f} {cvp:>5.2f} | {TARGET.get(c, float("nan")):>7.3f}')
    (HERE / 'outputs').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'outputs' / 'replicate_variance.json', 'w'), indent=2)
    print('\nReproducibility: the IR-by-age should sit near target with SD << the data signal.')
    print('Model SD per observable -> add (in quadrature) to observational SD for HM target std.')


if __name__ == '__main__':
    main()
