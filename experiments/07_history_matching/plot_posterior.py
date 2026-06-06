"""
Exp 07 stage-2 — SIR posterior figures (re-runnable; watch resolve as sims stream in).

Re-scores the current sir_results.jsonl with the overdispersion composite likelihood
(defensible middle: Gamma-Poisson phi=2 for IR, Beta-Binomial rho=0.05 for proportions),
importance-resamples, and draws:
  fig_fit.png    : posterior-predictive checks (3 IR bins, repeat, ever) + first-inf KM band vs data
  fig_params.png : prior (NROY) vs posterior marginals for every box parameter

Usage: uv run python experiments/07_history_matching/plot_posterior.py [--phi 2 --rho 0.05]
"""
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib;  matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import nbinom, betabinom

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]
REPEAT_OBS, REPEAT_N = 58, 136
EVER_OBS, EVER_N = 136, 213
TARGET = {'ir_symp_<6 m': 1.91, 'ir_symp_6-11 m': 5.37, 'ir_symp_12-23 m': 2.35,
          'repeat_detected_frac': 0.43, 'frac_ever_detected': 0.638}
GRID = np.arange(0, 37)


def km_survival(times, observed, grid):
    times = np.asarray(times, float); observed = np.asarray(observed, bool)
    et = np.unique(times[observed]); surv = 1.0; S = []; i = 0
    for g in grid:
        while i < len(et) and et[i] <= g:
            t = et[i]; d = int(np.sum((times == t) & observed)); risk = int(np.sum(times >= t))
            if risk > 0: surv *= (1 - d / risk)
            i += 1
        S.append(surv)
    return np.array(S)


def nb_logpmf(y, mu, phi):
    if phi <= 1 + 1e-9: return y * np.log(mu) - mu
    r = mu / (phi - 1.0); return nbinom.logpmf(y, r, r / (r + mu))


def bb_logpmf(k, n, p, rho):
    if rho <= 1e-9: return k * np.log(p) + (n - k) * np.log(1 - p)
    M = (1 - rho) / rho; return betabinom.logpmf(k, n, p * M, (1 - p) * M)


def logL(r, phi, rho):
    if (not r.get('ok')) or (r.get('frac_ever_detected') or 0) < 0.05: return -np.inf
    ll = 0.0
    for b, c, PT in IR:
        mu = r[f'ir_symp_{b}'] / 100.0 * PT
        if mu <= 0: return -np.inf
        ll += nb_logpmf(c, mu, phi)
    p = min(max(r['repeat_detected_frac'], 1e-6), 1 - 1e-6)
    q = min(max(r['frac_ever_detected'], 1e-6), 1 - 1e-6)
    return ll + bb_logpmf(REPEAT_OBS, REPEAT_N, p, rho) + bb_logpmf(EVER_OBS, EVER_N, q, rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi', type=float, default=2.0)
    ap.add_argument('--rho', type=float, default=0.05)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')]
    L = np.array([logL(r, args.phi, args.rho) for r in recs])
    fin = np.isfinite(L)
    w = np.zeros(len(recs)); w[fin] = np.exp(L[fin] - L[fin].max()); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(recs), size=40000, replace=True, p=w)
    usable = [r for r in recs if np.isfinite(logL(r, args.phi, args.rho))]  # prior cloud = persisting sims
    sub = f'{len(recs)} sims, {fin.sum()} usable | phi={args.phi}, rho={args.rho} | ESS={ess:.1f}'
    print(sub)

    # ---------- fig_fit: posterior-predictive + first-inf KM ----------
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    obs_panels = [('ir_symp_<6 m', 'IR symptomatic <6 m (/100 PY)'),
                  ('ir_symp_6-11 m', 'IR symptomatic 6-11 m'),
                  ('ir_symp_12-23 m', 'IR symptomatic 12-23 m'),
                  ('repeat_detected_frac', 'repeat fraction (among detected)'),
                  ('frac_ever_detected', 'ever-detected fraction')]
    for ax, (o, title) in zip(axs.flat, obs_panels):
        prior = np.array([r[o] for r in usable if r.get(o) is not None])
        post = np.array([recs[i][o] for i in idx])
        lo, hi = np.percentile(np.concatenate([prior, post]), [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(prior, bins=bins, density=True, color='0.8', label='prior (NROY)')
        ax.hist(post, bins=bins, density=True, color='steelblue', alpha=0.75, label='posterior')
        ax.axvline(TARGET[o], color='crimson', lw=2, label='target')
        ax.axvline(np.median(post), color='navy', ls='--', lw=1.2, label='post. median')
        ax.set_title(title, fontsize=10); ax.set_yticks([])
        ax.legend(fontsize=7)

    # first-infection cumulative-detected curve: data KM vs posterior band
    ax = axs.flat[5]
    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    dataS = km_survival(fi['age_event_months'].values, fi['event_observed'].values == 1, GRID)
    curves = np.array([recs[i]['km_surv'] for i in idx
                       if recs[i].get('km_surv') is not None and len(recs[i]['km_surv']) == len(GRID)])
    if len(curves):
        cum = 1 - curves                       # fraction ever-detected by age
        q = np.percentile(cum, [2.5, 50, 97.5], axis=0)
        ax.fill_between(GRID, q[0], q[2], color='steelblue', alpha=0.3, label='posterior 95%')
        ax.plot(GRID, q[1], color='navy', lw=1.5, label='posterior median')
    ax.plot(GRID, 1 - dataS, color='crimson', lw=2, label='MAL-ED (KM)')
    ax.set_xlabel('age (months)'); ax.set_ylabel('fraction first-detected')
    ax.set_title('age at first detection', fontsize=10); ax.set_xlim(0, 24); ax.legend(fontsize=7)

    fig.suptitle(f'SIR posterior-predictive — {sub}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / 'fig_fit.png', dpi=120); plt.close(fig)

    # ---------- fig_params: prior vs posterior marginals ----------
    pars = [k for k in recs[0] if k.startswith('par_')]
    n = len(pars); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.4 * nrow))
    for ax, k in zip(axs.flat, pars):
        prior = np.array([r[k] for r in usable if r.get(k) is not None])
        post = np.array([recs[i][k] for i in idx])
        bins = np.linspace(prior.min(), prior.max(), 30)
        ax.hist(prior, bins=bins, density=True, color='0.8', label='prior')
        ax.hist(post, bins=bins, density=True, color='steelblue', alpha=0.75, label='post')
        ax.axvline(np.median(post), color='navy', ls='--', lw=1.2)
        ax.set_title(k[4:], fontsize=9); ax.set_yticks([])
    for ax in axs.flat[n:]: ax.axis('off')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f'SIR posterior parameter marginals — {sub}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / 'fig_params.png', dpi=120); plt.close(fig)
    print(f'wrote {FIGDIR}/fig_fit.png and fig_params.png')


if __name__ == '__main__':
    main()
