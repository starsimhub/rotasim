"""
Exp 07 stage-2 — SIR diagnostic figures (what's going on inside the model).

From the stored 10k SIR sims (each = one NROY draw + its simulated observables +
the model's first-detection KM curve), re-scored with the overdispersion composite
likelihood (phi=2 IR, rho=0.05 props), produces:

  fig_trajectories.png : the LATENT trajectories we kept — per-sim first-detection
                         curves and IR-by-age profiles, colored by posterior weight, vs data
  fig_pairplot.png     : full box -> NROY (all draws) -> posterior, for the key params
  fig_observable_space.png : where the posterior lives in observable space (+ target CIs),
                         exposing the cross-target trade-offs
  fig_sensitivity.png  : Spearman(param, observable) over persisting sims — which knobs
                         move which targets (stiff vs sloppy directions)

Usage: uv run python experiments/07_history_matching/plot_diagnostics.py [--phi 2 --rho 0.05]
"""
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import nbinom, betabinom, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]
REPEAT_OBS, REPEAT_N = 58, 136
EVER_OBS, EVER_N = 136, 213
GRID = np.arange(0, 37)
AGE_MID = [3.0, 8.5, 18.0]                                  # bin midpoints for IR profile
TARGET = {'<6 m': 1.91, '6-11 m': 5.37, '12-23 m': 2.35}
TARGET_SE = {b: np.sqrt(c) / PT * 100 for b, c, PT in IR}
# Full HM box (natural units) -> NROY -> posterior
BOX = {'base_beta': (0.1, 0.6), 'young_reservoir': (1, 15), 'adult_contacts': (0.5, 2.5),
       'infant_exposure': (0.3, 4.0), 'sus_after_1': (0.1, 1.0), 'sus_after_2': (0, 1.0),
       'sus_after_3plus': (0, 1.0), 'titer_median': (4, 60), 'titer_gsd': (1.3, 3.5),
       'titer_half_life_days': (25, 70), 'hill_slope': (1.5, 8.0), 'maternal_efficacy': (0.7, 0.99),
       'p_symp_1': (0.4, 1.0), 'p_symp_2': (0, 1.0), 'p_symp_3plus': (0, 1.0)}
PAIR_KEYS = ['base_beta', 'sus_after_1', 'sus_after_2', 'sus_after_3plus', 'maternal_efficacy', 'p_symp_3plus']


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
    pidx = rng.choice(len(recs), size=40000, replace=True, p=w)
    sub = f'10k SIR | {fin.sum()} usable | phi={args.phi}, rho={args.rho} | ESS={ess:.0f}'
    print(sub)
    usable = np.where(fin)[0]
    wn = w[usable] / w[usable].max()                              # 0..1 for alpha/color

    # ---------- fig_trajectories: the latent trajectories we kept ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    # data KM
    et = np.unique(fi.loc[fi.event_observed == 1, 'age_event_months'].values)
    times = fi['age_event_months'].values; obs = fi['event_observed'].values == 1
    surv = 1.0; dataS = []; j = 0
    for g in GRID:
        while j < len(et) and et[j] <= g:
            t = et[j]; d = np.sum((times == t) & obs); risk = np.sum(times >= t)
            if risk > 0: surv *= (1 - d / risk)
            j += 1
        dataS.append(surv)
    dataS = np.array(dataS)
    # Most kept trajectories have tiny weight (ESS~70 of 1821); plotting them all swamps
    # the figure. Show the full kept set as a soft 5-95% envelope (context), then overplot
    # only the HIGH-weight trajectories (those carrying ~90% of posterior mass) in a hot
    # colormap with weight-scaled thickness, so the ones that matter stand out.
    sortw = usable[np.argsort(w[usable])[::-1]]
    cw = np.cumsum(w[sortw]); top = sortw[:max(np.searchsorted(cw, 0.90) + 1, 40)]
    cmap = plt.cm.autumn_r
    wtop = w[top]; tnorm = Normalize(wtop.min(), wtop.max())

    def km_curve(k):
        ks = recs[k].get('km_surv')
        return 1 - np.array(ks) if (ks and len(ks) == len(GRID)) else None

    allc = np.array([c for c in (km_curve(k) for k in usable) if c is not None])
    env = np.percentile(allc, [5, 95], axis=0)
    a1.fill_between(GRID, env[0], env[1], color='0.88', label='persisting, unweighted (5-95%)')
    for k in top[::-1]:                                           # highest weight drawn last/on top
        c = km_curve(k)
        if c is not None:
            a1.plot(GRID, c, color=cmap(tnorm(w[k])), alpha=0.55, lw=0.6 + 2.4 * tnorm(w[k]))
    a1.plot(GRID, 1 - dataS, color='navy', lw=3, label='MAL-ED (KM)', zorder=10)
    a1.set_xlim(0, 24); a1.set_xlabel('age (months)'); a1.set_ylabel('fraction first-detected')
    a1.set_title(f'kept first-detection trajectories (top {len(top)} by weight)'); a1.legend()
    # IR-by-age profiles
    allp = np.array([[recs[k][f'ir_symp_{b}'] for b, _, _ in IR] for k in usable])
    penv = np.percentile(allp, [5, 95], axis=0)
    a2.fill_between(AGE_MID, penv[0], penv[1], color='0.88', label='persisting, unweighted (5-95%)')
    for k in top[::-1]:
        prof = [recs[k][f'ir_symp_{b}'] for b, _, _ in IR]
        a2.plot(AGE_MID, prof, color=cmap(tnorm(w[k])), alpha=0.55, lw=0.6 + 2.4 * tnorm(w[k]))
    a2.errorbar(AGE_MID, [TARGET[b] for b, _, _ in IR], yerr=[1.96 * TARGET_SE[b] for b, _, _ in IR],
                fmt='o-', color='navy', lw=3, capsize=4, label='MAL-ED (95% CI)', zorder=10)
    a2.set_xticks(AGE_MID); a2.set_xticklabels(['<6 m', '6-11 m', '12-23 m'])
    a2.set_ylabel('symptomatic IR (/100 PY)'); a2.set_ylim(0, 8)
    a2.set_title(f'kept IR-by-age profiles (top {len(top)} by weight)')
    sm = ScalarMappable(cmap=cmap, norm=tnorm); sm.set_array([])
    fig.colorbar(sm, ax=a2, label='posterior weight (top set)')
    fig.suptitle(f'Latent trajectories kept — {sub}'); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / 'fig_trajectories.png', dpi=120); plt.close(fig)

    # ---------- fig_pairplot: full box -> NROY -> posterior ----------
    n = len(PAIR_KEYS)
    fig, axs = plt.subplots(n, n, figsize=(2.1 * n, 2.1 * n))
    nroy = {k: np.array([r[f'par_{k}'] for r in recs]) for k in PAIR_KEYS}
    post = {k: np.array([recs[i][f'par_{k}'] for i in pidx]) for k in PAIR_KEYS}
    for i, ki in enumerate(PAIR_KEYS):
        for jj, kj in enumerate(PAIR_KEYS):
            ax = axs[i, jj]
            if i == jj:
                ax.hist(nroy[ki], bins=25, range=BOX[ki], color='0.8', density=True)
                ax.hist(post[ki], bins=25, range=BOX[ki], color='steelblue', alpha=0.7, density=True)
                ax.set_yticks([])
            elif i > jj:
                ax.scatter(nroy[kj], nroy[ki], s=2, color='0.75', alpha=0.25, label='NROY', rasterized=True)
                ax.scatter(post[kj][::20], post[ki][::20], s=2, color='steelblue', alpha=0.25, label='posterior', rasterized=True)
                ax.set_xlim(*BOX[kj]); ax.set_ylim(*BOX[ki])
            else:
                ax.axis('off')
            if i == n - 1: ax.set_xlabel(kj, fontsize=8)
            if jj == 0 and i > 0: ax.set_ylabel(ki, fontsize=8)
            ax.tick_params(labelsize=6)
    axs[1, 0].scatter([], [], s=10, color='0.75', label='NROY (all draws)')
    axs[1, 0].scatter([], [], s=10, color='steelblue', label='posterior')
    axs[0, n - 1].axis('off'); axs[0, n - 1].legend(*axs[1, 0].get_legend_handles_labels(), loc='center', fontsize=9)
    fig.suptitle(f'Full box (axis range) -> NROY (gray) -> posterior (blue) — {sub}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / 'fig_pairplot.png', dpi=110); plt.close(fig)

    # ---------- fig_observable_space: trade-offs + where posterior lives ----------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    def sc(ax, xo, yo, xt, yt, xse, yse, xl, yl):
        # background CI bands + target box (under the samples), then gray context, then high-weight points
        ax.axvspan(xt - 1.96 * xse, xt + 1.96 * xse, color='navy', alpha=0.04, zorder=0)
        ax.axhspan(yt - 1.96 * yse, yt + 1.96 * yse, color='navy', alpha=0.04, zorder=0)
        ax.add_patch(Rectangle((xt - 1.96 * xse, yt - 1.96 * yse), 2 * 1.96 * xse, 2 * 1.96 * yse,
                               fill=False, edgecolor='navy', lw=1.8, zorder=1))   # under the samples
        ax.scatter([recs[k][xo] for k in usable], [recs[k][yo] for k in usable],
                   s=7, color='0.55', alpha=0.4, zorder=2, rasterized=True)
        ax.scatter([recs[k][xo] for k in top], [recs[k][yo] for k in top], c=wtop, cmap=cmap,
                   s=12 + 80 * tnorm(wtop), alpha=0.85, edgecolor='k', linewidth=0.2, zorder=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    sc(axs[0], 'ir_symp_<6 m', 'ir_symp_6-11 m', 1.91, 5.37, TARGET_SE['<6 m'], TARGET_SE['6-11 m'],
       'IR <6 m', 'IR 6-11 m'); axs[0].set_title('peak vs early IR')
    sc(axs[1], 'ir_symp_6-11 m', 'ir_symp_12-23 m', 5.37, 2.35, TARGET_SE['6-11 m'], TARGET_SE['12-23 m'],
       'IR 6-11 m', 'IR 12-23 m'); axs[1].set_title('peak vs late IR')
    # repeat vs ever (binomial SE)
    rse = np.sqrt((REPEAT_OBS / REPEAT_N) * (1 - REPEAT_OBS / REPEAT_N) / REPEAT_N)
    ese = np.sqrt((EVER_OBS / EVER_N) * (1 - EVER_OBS / EVER_N) / EVER_N)
    sc(axs[2], 'repeat_detected_frac', 'frac_ever_detected', 0.43, 0.638, rse, ese,
       'repeat fraction', 'ever-detected'); axs[2].set_title('repeat vs ever-detected')
    sm = ScalarMappable(cmap=cmap, norm=tnorm); sm.set_array([])
    fig.colorbar(sm, ax=axs, label='posterior weight (top set)', shrink=0.8)
    fig.suptitle(f'Observable space (gray=persisting, colored=top-weight; box=target 95% CI) — {sub}', fontsize=11)
    fig.savefig(FIGDIR / 'fig_observable_space.png', dpi=120, bbox_inches='tight'); plt.close(fig)

    # ---------- fig_sensitivity: Spearman(param, observable) over persisting sims ----------
    obs_keys = ['ir_symp_<6 m', 'ir_symp_6-11 m', 'ir_symp_12-23 m', 'repeat_detected_frac', 'frac_ever_detected']
    par_keys = list(BOX.keys())
    M = np.zeros((len(par_keys), len(obs_keys)))
    for pi, pk in enumerate(par_keys):
        pv = np.array([recs[k][f'par_{pk}'] for k in usable])
        for oi, ok in enumerate(obs_keys):
            ov = np.array([recs[k][ok] for k in usable])
            M[pi, oi] = spearmanr(pv, ov).correlation
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(len(obs_keys))); ax.set_xticklabels(['IR<6', 'IR6-11', 'IR12-23', 'repeat', 'ever'], rotation=30)
    ax.set_yticks(range(len(par_keys))); ax.set_yticklabels(par_keys, fontsize=8)
    for pi in range(len(par_keys)):
        for oi in range(len(obs_keys)):
            ax.text(oi, pi, f'{M[pi,oi]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(M[pi, oi]) > 0.35 else 'black')
    fig.colorbar(im, label='Spearman rho (over persisting sims)')
    ax.set_title(f'What moves what inside the model\n{sub}', fontsize=10)
    fig.tight_layout(); fig.savefig(FIGDIR / 'fig_sensitivity.png', dpi=120); plt.close(fig)
    print(f'wrote fig_trajectories, fig_pairplot, fig_observable_space, fig_sensitivity to {FIGDIR}')


if __name__ == '__main__':
    main()
