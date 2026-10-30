"""Fast posterior for the UK surveillance fit: reweight the samples HM ALREADY simulated.

HM stores, per wave, the input `samples` and their `simulation_results` (the per-bin case
proportions). Instead of re-running thousands of fresh sims (trajectory_select_uk), we score
those existing samples with the MULTINOMIAL likelihood on the observed UK case counts and
importance-resample to a posterior. This is prior-predictive importance sampling over the LHS
draws -- a fast first posterior + posterior-predictive shape check. (trajectory_select_uk
remains the rigorous fresh-sim version.)

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python score_hm_uk.py --model age_binned --fix-titer-shape
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python score_hm_uk.py --model infnum     --fix-titer-shape
"""
import os, sys, json, argparse, pathlib
import numpy as np
import pandas as pd
import historymatching as hm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import process_surveillance_uk as PU                # noqa: E402
import hm_calibrate_uk as UK                         # noqa: E402
from hm_calibrate_uk import uk_bounds                # noqa: E402

EPS = 1e-9
MIN_CASES = 50
EXP_DIR = {'age_binned': '28_hm_uk_age_binned', 'infnum': '28_hm_uk_infnum', 'age': '28_hm_uk_age', 'age_and_infection': '28_hm_uk_age_and_infection'}


def _load_engine(hm_dir, bounds, obs):
    ckpt = hm_dir / 'checkpoint.pkl'
    tmp = hm.HistoryMatching(function=lambda df: df, bounds=bounds, observations=obs,
                             emulator_type='bayes_linear', sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(hm_dir.parent), run_name=hm_dir.name, random_seed=20260618)
    return hm.HistoryMatching.load_checkpoint(ckpt, tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned', 'age_and_infection'])
    ap.add_argument('--maternal', default='titer', choices=['titer', 'erlang'])
    ap.add_argument('--fix-titer-shape', action='store_true')
    ap.add_argument('--cap-age-months', type=float, default=UK.DEFAULT_CAP_M)
    ap.add_argument('--beta-max', type=float, default=UK.DEFAULT_BETA_MAX)
    ap.add_argument('--deff', type=float, default=1.0,
                    help='overdispersion / design effect: divide the multinomial logL by deff '
                         '(quasi-multinomial, like Bangladesh phi). deff=1 = raw (ESS collapses at '
                         'N~3500); higher softens -> more posterior spread. Prints an ESS-vs-deff sweep.')
    ap.add_argument('--hm-dir', default=None)
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    cap = a.cap_age_months

    run_name = f'uk_{a.model}_{a.maternal}' + ('_fixedshape' if a.fix_titer_shape else '')
    hm_dir = pathlib.Path(a.hm_dir) if a.hm_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs' / 'hm' / run_name)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs')
    fig_dir = out_dir.parent / 'figures'; out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    bounds = uk_bounds(a.model, a.maternal, a.fix_titer_shape, a.beta_max)
    obs = UK.make_observations(cap)
    cols = UK.obs_cols(cap)
    tgt = PU.load_targets_uk(cap_age_m=cap)
    counts = tgt['counts']; labels = tgt['bin_labels']

    engine = _load_engine(hm_dir, bounds, obs)
    results = engine.get_all_results()
    # Pair input samples with their simulated proportions across all waves.
    samp = pd.concat([r.samples.reset_index(drop=True) for r in results], ignore_index=True)
    sim = pd.concat([r.simulation_results.reset_index(drop=True) for r in results], ignore_index=True)
    P = sim[cols].to_numpy(float)                              # (n, nbins) model proportions
    n = P.shape[0]
    print(f"{a.model}: {n} simulated samples across {len(results)} wave(s); bins={cols}")

    # Raw multinomial logL per sample (NaN/extinct -> -inf).
    valid = np.isfinite(P).all(axis=1) & (np.nansum(P, axis=1) > 0)
    logL_raw = np.full(n, -np.inf)
    logL_raw[valid] = (counts * np.log(np.clip(P[valid], EPS, None))).sum(axis=1)
    finite = np.isfinite(logL_raw)
    print(f"finite-logL samples: {finite.sum()}/{n}")

    def _ess(deff):
        w = np.zeros(n)
        le = logL_raw[finite] / deff          # quasi-multinomial: soften by the design effect
        w[finite] = np.exp(le - le.max())
        w /= w.sum()
        return w, float(1.0 / np.sum(w ** 2))

    print("ESS-vs-deff sweep:")
    for d in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        print(f"  deff={d:6.1f}  ESS={_ess(d)[1]:7.1f}  ({100*_ess(d)[1]/n:.1f}% of {n})")
    w, ess = _ess(a.deff)
    print(f"--> using deff={a.deff}: ESS = {ess:.1f}  ({100*ess/n:.1f}% of {n})")

    rng = np.random.default_rng(0)
    post_idx = rng.choice(n, size=n, replace=True, p=w)
    samp.iloc[post_idx].reset_index(drop=True).to_csv(out_dir / 'posterior_hmreweight.csv', index=False)

    # Posterior-predictive: weighted quantiles of each bin proportion vs the UK target.
    obs_prop = tgt['proportions']
    Ppost = P[post_idx]
    med = np.median(Ppost, axis=0); lo = np.quantile(Ppost, .025, axis=0); hi = np.quantile(Ppost, .975, axis=0)
    print("\nbin        UK_obs   model_med   model_95%CrI")
    for i, lab in enumerate(labels):
        print(f"  {lab:8s}  {obs_prop[i]:.3f}    {med[i]:.3f}      [{lo[i]:.3f}, {hi[i]:.3f}]")

    # Figure: UK observed (with multinomial SE) vs posterior-predictive median + 95% CrI.
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x - 0.10, obs_prop, yerr=tgt['se'], fmt='o', color='k', capsize=4, label='UK observed', zorder=3)
    ax.errorbar(x + 0.10, med, yerr=[med - lo, hi - med], fmt='s', color='#2c7fb8', capsize=4,
                label='model posterior-predictive (median, 95% CrI)', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('fraction of symptomatic cases'); ax.set_ylim(0, max(0.6, hi.max() * 1.15))
    cap_lab = 'all ages' if cap is None else f'<{int(cap)}mo'
    ax.set_title(f'UK surveillance posterior-predictive — {a.model} (cap {cap_lab}, ESS={ess:.0f})')
    ax.legend(frameon=False, fontsize=9); fig.tight_layout()
    figpath = fig_dir / f'uk_posterior_predictive_{a.model}.png'
    fig.savefig(figpath, dpi=140); plt.close(fig)
    json.dump(dict(model=a.model, cap=cap, deff=a.deff, n=int(n), finite=int(finite.sum()), ess=ess,
                   uk_obs=list(map(float, obs_prop)), model_med=list(map(float, med)),
                   model_lo=list(map(float, lo)), model_hi=list(map(float, hi))),
              open(out_dir / 'hmreweight_stats.json', 'w'), indent=2)
    print(f"\nwrote {figpath}")
    print(f"wrote {out_dir / 'posterior_hmreweight.csv'} + hmreweight_stats.json")


if __name__ == '__main__':
    main()
