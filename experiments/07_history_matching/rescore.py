"""
Exp 07 stage-2 (re-score) — importance-resample the existing SIR sims with the
SOFTENED likelihood, no re-simulation. Drops the pathological 265-record survival
component (it dominated the logL spread and collapsed ESS to ~1.5) and keeps:
  - Poisson(IR by age),
  - Binomial(repeat fraction),
  - Binomial(frac_ever_detected)  [= the coarse first-infection summary, 1 - S(24mo)].
All re-scorable from the stored observables -> no re-run needed.

Usage: uv run python experiments/07_history_matching/rescore.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]  # bin, cases, person-months
REPEAT_OBS, REPEAT_N = 58, 136
EVER_OBS, EVER_N = 136, 213
TARGET = {'ir_symp_<6 m': 1.91, 'ir_symp_6-11 m': 5.37, 'ir_symp_12-23 m': 2.35,
          'repeat_detected_frac': 0.43, 'frac_ever_detected': 0.638}


def logL_pb(r):
    if (not r.get('ok')) or r.get('frac_ever_detected', 0) < 0.05:
        return -np.inf
    ll = 0.0
    for b, c, PT in IR:
        lam = r[f'ir_symp_{b}'] / 100.0 * PT
        if lam <= 0:
            return -np.inf
        ll += c * np.log(lam) - lam                       # Poisson
    p = min(max(r['repeat_detected_frac'], 1e-6), 1 - 1e-6)
    ll += REPEAT_OBS * np.log(p) + (REPEAT_N - REPEAT_OBS) * np.log(1 - p)   # Binomial repeat
    q = min(max(r['frac_ever_detected'], 1e-6), 1 - 1e-6)
    ll += EVER_OBS * np.log(q) + (EVER_N - EVER_OBS) * np.log(1 - q)         # Binomial ever
    return ll


def main():
    recs = [json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')]
    logL = np.array([logL_pb(r) for r in recs])
    fin = np.isfinite(logL)
    print(f'completed sims: {len(recs)}; usable (finite): {fin.sum()} ({100*fin.mean():.0f}%)')
    w = np.zeros(len(recs))
    w[fin] = np.exp(logL[fin] - logL[fin].max()); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    order = np.argsort(w)[::-1]
    print(f'P+B likelihood: ESS = {ess:.1f}  ({100*ess/fin.sum():.1f}% of usable)')
    print(f'  top weight {w[order[0]]:.3f}; top-5 {w[order[:5]].sum():.3f}; top-50 {w[order[:50]].sum():.3f}')

    rng = np.random.default_rng(0)
    post_idx = rng.choice(len(recs), size=10000, replace=True, p=w)
    pars = [k for k in recs[0] if k.startswith('par_')]
    posterior = pd.DataFrame([{k: recs[i][k] for k in pars} for i in post_idx])
    posterior.to_csv(HERE / 'outputs' / 'posterior_pb.csv', index=False)
    print('\nPosterior marginals (median [IQR]):')
    for k in pars:
        v = posterior[k]; print(f'  {k[4:]:>24}: {v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]')

    print('\nPosterior-predictive vs targets (resampled-trajectory observable: median [IQR] | target):')
    for o, t in TARGET.items():
        v = pd.Series([recs[i][o] for i in post_idx])
        print(f'  {o:>20}: {v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}] | {t}')


if __name__ == '__main__':
    main()
