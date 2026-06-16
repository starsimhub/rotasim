"""Emulator-based MCMC on the NROY -> posterior, for models where importance resampling
collapses (age+titer: loose NROY, ESS~1). Runs Metropolis against the TRAINED HM emulators
(fast surrogates in the checkpoint) -- NO new simulations -- so it samples the posterior
DIRECTLY (no weight collapse). Target = Gaussian likelihood on the 5 HM summary features with
the emulator predictive variance + observational variance + discrepancy floors folded in; the
prior is uniform on {NROY (max implausibility <= 3) ∩ box}. Multiple chains are vectorized
(one batched emulator.predict per step across chains).

Output: experiments/{18,19}_*/outputs/posterior_mcmc.csv (transformed params) + mcmc_stats.json.
VALIDATE the emulator posterior afterwards by running real ABM sims at posterior draws
(emulators are approximate -- esp. age's weak repeat/first-inf, R^2 0.41/0.14).

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python mcmc_nroy.py --model age --n-chains 8 --n-steps 40000
"""
import os, sys, json, argparse, pathlib
import numpy as np, pandas as pd
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
from hm_calibrate import BOUNDS, bounds_for, make_observations, OBS_COLS, untransform  # noqa: E402

HM_RUN = {'age':    THISDIR / 'experiments' / '16_hm_age_titer'    / 'outputs' / 'hm' / 'maled_age_titer',
          'infnum': THISDIR / 'experiments' / '17_hm_infnum_titer' / 'outputs' / 'hm' / 'maled_infnum_titer'}
EXP_DIR = {'age': '18_age_posterior', 'infnum': '19_infnum_posterior'}
IMPL_THRESH = 3.0


def load_emulators(ckpt_dir, run_name, bounds):
    """Latest trained emulator per feature, from the HM checkpoint."""
    ckpt = ckpt_dir / 'checkpoint.pkl'
    tmp = hm.HistoryMatching(function=lambda df: df, bounds=bounds, observations=make_observations(),
                             emulator_type='bayes_linear', sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(ckpt_dir.parent), run_name=run_name, random_seed=20260610)
    engine = hm.HistoryMatching.load_checkpoint(ckpt, tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    allem = engine.emulator_bank.get_all_emulators()         # {iter: {feat: emu}}
    latest = {}
    for it in sorted(allem):
        for feat, emu in allem[it].items():
            latest[feat] = emu                                # keep highest-iteration emulator per feature
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned'])
    ap.add_argument('--n-chains', type=int, default=8)
    ap.add_argument('--n-steps', type=int, default=40000)
    ap.add_argument('--burn', type=int, default=10000)
    ap.add_argument('--thin', type=int, default=20)
    ap.add_argument('--step-scale', type=float, default=0.04)   # proposal sd as fraction of box width
    ap.add_argument('--seed', type=int, default=20260611)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fix-titer-shape', action='store_true',
                    help='match a --fix-titer-shape HM run: drop the 4 titer-shape params, only maternal_efficacy free')
    ap.add_argument('--exp-dir', default=None,
                    help='output experiment folder under experiments/ (e.g. 23_age_titer_fixedshape); '
                         'defaults to the canonical titer-free posterior for --model')
    ap.add_argument('--ckpt-dir', default=None,
                    help='HM checkpoint dir holding checkpoint.pkl (e.g. '
                         'experiments/23_age_titer_fixedshape/outputs/hm/maled_age_titer_fixedshape); '
                         'defaults to the exp16/17 titer run')
    ap.add_argument('--run-name', default=None,
                    help='HM run_name used when the checkpoint was written (e.g. maled_age_titer_fixedshape)')
    a = ap.parse_args()
    if a.smoke:
        a.n_chains, a.n_steps, a.burn, a.thin = 4, 3000, 500, 5

    bounds = bounds_for(a.model, 'titer', fix_titer_shape=a.fix_titer_shape)
    keys = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in keys]); hi = np.array([bounds[k][1] for k in keys])
    obs = make_observations()
    feats = OBS_COLS
    tgt = np.array([obs[f][0] for f in feats]); tgt_sd = np.array([obs[f][1] for f in feats])  # sd already includes discrepancy floor
    ckpt_dir = pathlib.Path(a.ckpt_dir).resolve() if a.ckpt_dir else HM_RUN[a.model]
    run_name = a.run_name or f'maled_{a.model}_titer'
    emus = load_emulators(ckpt_dir, run_name, bounds)
    missing = [f for f in feats if f not in emus]
    if missing:
        raise SystemExit(f"no emulator for features {missing}; have {list(emus)}")
    print(f"MCMC model={a.model}  {len(keys)} params, {len(feats)} features, chains={a.n_chains}, steps={a.n_steps}", flush=True)

    def predict_all(X):                       # X: (m, d) -> per-feature (mean, var) arrays, shape (m,)
        df = pd.DataFrame(X, columns=keys)
        means, vars_ = [], []
        for f in feats:
            r = emus[f].predict(df)
            means.append(np.asarray(r.get_mean(), float)); vars_.append(np.asarray(r.get_std(), float) ** 2)
        return np.array(means).T, np.array(vars_).T   # (m, n_feat)

    def logpost(X):
        m, ev = predict_all(X)
        var = tgt_sd[None, :] ** 2 + ev                 # observational+discrepancy + emulator variance
        z2 = ((m - tgt[None, :]) ** 2) / var
        ll = -0.5 * np.sum(z2, axis=1)
        in_box = np.all((X >= lo[None, :]) & (X <= hi[None, :]), axis=1)
        in_nroy = np.max(np.sqrt(z2), axis=1) <= IMPL_THRESH   # NROY = max implausibility <= 3
        ll = np.where(in_box & in_nroy, ll, -np.inf)
        return ll

    rng = np.random.default_rng(a.seed)
    # init chains from NROY draws (uniform-in-box rejection until in NROY)
    X = np.empty((a.n_chains, len(keys))); filled = 0
    while filled < a.n_chains:
        cand = lo + rng.random((a.n_chains * 20, len(keys))) * (hi - lo)
        good = cand[np.isfinite(logpost(cand))]
        take = good[:a.n_chains - filled]; X[filled:filled + len(take)] = take; filled += len(take)
    lp = logpost(X)
    step = a.step_scale * (hi - lo)
    samples, acc = [], 0
    for t in range(a.n_steps):
        prop = X + rng.normal(size=X.shape) * step[None, :]
        lpp = logpost(prop)
        al = np.exp(np.clip(lpp - lp, -700, 0))
        u = rng.random(a.n_chains); take = (u < al) & np.isfinite(lpp)
        X[take] = prop[take]; lp[take] = lpp[take]; acc += take.sum()
        if t >= a.burn and (t - a.burn) % a.thin == 0:
            samples.append(X.copy())
    S = np.concatenate(samples, axis=0)              # (n_samples, d)  pooled across chains
    acc_rate = acc / (a.n_steps * a.n_chains)

    out_dir = THISDIR / 'experiments' / (a.exp_dir or EXP_DIR[a.model]) / 'outputs'; out_dir.mkdir(parents=True, exist_ok=True)
    suf = '_smoke' if a.smoke else ''
    post = pd.DataFrame(S, columns=keys)
    post.to_csv(out_dir / f'posterior_mcmc{suf}.csv', index=False)
    # per-chain means for a crude R-hat on logpost-space proxy (report acceptance + spread)
    chain_means = np.stack([np.mean(np.stack(samples)[:, c, :], axis=0) for c in range(a.n_chains)])
    stats = dict(model=a.model, n_samples=len(S), n_chains=a.n_chains, acceptance=float(acc_rate),
                 param_keys=keys,
                 post_median={k: float(np.median(S[:, i])) for i, k in enumerate(keys)},
                 post_q05={k: float(np.percentile(S[:, i], 5)) for i, k in enumerate(keys)},
                 post_q95={k: float(np.percentile(S[:, i], 95)) for i, k in enumerate(keys)},
                 between_chain_sd={k: float(np.std(chain_means[:, i])) for i, k in enumerate(keys)})
    json.dump(stats, (out_dir / f'mcmc_stats{suf}.json').open('w'), indent=2)
    print(f"acceptance={acc_rate:.2f}  pooled samples={len(S)}", flush=True)
    print("posterior (median [5,95]) on key params:")
    show = [k for k in ['log_base_beta', 'sus_after_1', 'maternal_efficacy', 'beta0', 'p_symp_1'] if k in keys]
    for k in show:
        i = keys.index(k); print(f"  {k:16s} {np.median(S[:,i]):.3f}  [{np.percentile(S[:,i],5):.3f}, {np.percentile(S[:,i],95):.3f}]")
    print(f"wrote posterior_mcmc{suf}.csv + mcmc_stats{suf}.json", flush=True)


if __name__ == '__main__':
    main()
