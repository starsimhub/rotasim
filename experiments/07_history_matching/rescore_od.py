"""
Exp 07 stage-2 (re-score, overdispersion sweep) — importance-resample the existing
SIR sims under composite likelihoods with TUNABLE overdispersion, no re-simulation.

Components (all re-scored from stored observables):
  - IR by age bin: Poisson (phi=1) or Gamma-Poisson/NB (phi = variance-to-mean ratio >1)
  - repeat fraction & ever-detected: Binomial (rho=0) or Beta-Binomial (rho = ICC > 0)

Sweeps a small (phi, rho) grid and reports ESS + posterior-predictive sensitivity, so
we can see whether a *defensible* overdispersion lifts ESS without blowing the posterior
wide (gaming) — not tune dispersion to a target ESS.

Usage: uv run python experiments/07_history_matching/rescore_od.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import nbinom, betabinom

HERE = Path(__file__).resolve().parent
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]
REPEAT_OBS, REPEAT_N = 58, 136
EVER_OBS, EVER_N = 136, 213
TARGET = {'ir_symp_<6 m': 1.91, 'ir_symp_6-11 m': 5.37, 'ir_symp_12-23 m': 2.35,
          'repeat_detected_frac': 0.43, 'frac_ever_detected': 0.638}
GRID = [(1.0, 0.0), (2.0, 0.03), (2.0, 0.05), (3.0, 0.05), (3.0, 0.10)]  # (phi=var/mean IR, rho=ICC props)


def nb_logpmf(y, mu, phi):
    """Gamma-Poisson with variance = phi*mu (phi=1 -> Poisson)."""
    if phi <= 1.0 + 1e-9:
        return y * np.log(mu) - mu                    # Poisson (drop const)
    r = mu / (phi - 1.0)                              # NB size s.t. var = phi*mu
    return nbinom.logpmf(y, r, r / (r + mu))


def bb_logpmf(k, n, p, rho):
    """Beta-Binomial with intraclass corr rho (rho=0 -> Binomial)."""
    if rho <= 1e-9:
        return k * np.log(p) + (n - k) * np.log(1 - p)   # Binomial (drop const)
    M = (1 - rho) / rho                                  # concentration
    return betabinom.logpmf(k, n, p * M, (1 - p) * M)


def logL(r, phi, rho):
    if (not r.get('ok')) or (r.get('frac_ever_detected') or 0) < 0.05:
        return -np.inf
    ll = 0.0
    for b, c, PT in IR:
        mu = r[f'ir_symp_{b}'] / 100.0 * PT
        if mu <= 0:
            return -np.inf
        ll += nb_logpmf(c, mu, phi)
    p = min(max(r['repeat_detected_frac'], 1e-6), 1 - 1e-6)
    q = min(max(r['frac_ever_detected'], 1e-6), 1 - 1e-6)
    ll += bb_logpmf(REPEAT_OBS, REPEAT_N, p, rho) + bb_logpmf(EVER_OBS, EVER_N, q, rho)
    return ll


def main():
    recs = [json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')]
    nfin = sum(r['ok'] and (r.get('frac_ever_detected') or 0) >= 0.05 for r in recs)
    print(f'{len(recs)} sims; {nfin} usable\n')
    print(f'{"phi(IR var/mean)":>16} {"rho(prop ICC)":>13} | {"ESS":>7} {"top w":>6} | '
          f'posterior-predictive medians (ir<6/6-11/12-23, repeat, ever)')
    rng = np.random.default_rng(0)
    for phi, rho in GRID:
        L = np.array([logL(r, phi, rho) for r in recs])
        fin = np.isfinite(L)
        w = np.zeros(len(recs)); w[fin] = np.exp(L[fin] - L[fin].max()); w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        idx = rng.choice(len(recs), size=20000, replace=True, p=w)
        pp = {o: np.median([recs[i][o] for i in idx]) for o in TARGET}
        ppstr = f"{pp['ir_symp_<6 m']:.2f}/{pp['ir_symp_6-11 m']:.2f}/{pp['ir_symp_12-23 m']:.2f}, " \
                f"{pp['repeat_detected_frac']:.2f}, {pp['frac_ever_detected']:.2f}"
        tag = 'Poisson+Binom' if (phi == 1 and rho == 0) else ''
        print(f'{phi:>16.1f} {rho:>13.2f} | {ess:>7.1f} {w.max():>6.3f} | {ppstr}  {tag}')
    print(f'\ntargets: {TARGET["ir_symp_<6 m"]}/{TARGET["ir_symp_6-11 m"]}/{TARGET["ir_symp_12-23 m"]}, '
          f'{TARGET["repeat_detected_frac"]}, {TARGET["frac_ever_detected"]}')


if __name__ == '__main__':
    main()
