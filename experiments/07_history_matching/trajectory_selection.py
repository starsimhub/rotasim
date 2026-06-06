"""
Exp 07 (stage 2) — trajectory selection / SIR on the converged NROY.

Draws N unbiased NROY samples from the HM checkpoint, simulates each (1 seed),
scores each with the COMPOSITE pseudo-likelihood (no squared error):
  - IR by age bin: Poisson(obs_cases | model_rate * data_PT)
  - repeat fraction & ever-detected: Binomial(obs | n, model_p)
  - age-at-first-infection: censored-survival likelihood of the data's
    (age, event) records under the model's first-detection distribution
Extinct trajectories get log-weight -inf (drop out). Importance-resample by
weight -> posterior over parameters. Streams per-sim results (resumable).

Usage:
  uv run --python 3.13 python experiments/07_history_matching/trajectory_selection.py --smoke
  uv run --python 3.13 python experiments/07_history_matching/trajectory_selection.py --n 10000 --n-workers 118
"""
import os, json, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
rw = sc.importbypath(HERE / 'run_wave.py')      # BOUNDS, OBSERVATIONS, _untransform, simulate machinery
exp06 = sc.importbypath(REPO / 'experiments' / '06_titer_maternal_peak' / 'run.py')
N_AGENTS = 40_000
CROSS = 0.5
_fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
CENS = _fi.loc[_fi['event_observed'] == 0, 'age_event_months'].dropna().values
CENS = CENS[CENS > 0]
# Data first-infection records (age months, event_observed) for the survival LL
FIRSTINF = _fi[['age_event_months', 'event_observed']].dropna().values
# IR data: bin -> (cases, person-time months)
_s = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'ir_by_age_symp_bangladesh.csv').set_index('age_cat')
IR_DATA = {b: (int(_s.loc[b, 'cases']), float(_s.loc[b, 'PT'])) for b in ['<6 m', '6-11 m', '12-23 m']}
REPEAT_OBS, REPEAT_N = 58, 136          # repeat infections among ever-detected
EVER_OBS, EVER_N = 136, 213             # ever-detected among enrolled
GRID = np.arange(0, 37)                 # monthly edges for the survival LL
EPS = 1e-9


def _km_survival(times, observed, grid):
    """Kaplan-Meier survival S(t) on grid (months)."""
    times = np.asarray(times, float); observed = np.asarray(observed, bool)
    et = np.unique(times[observed]); surv = 1.0; S = []; i = 0
    for g in grid:
        while i < len(et) and et[i] <= g:
            t = et[i]; d = int(np.sum((times == t) & observed)); risk = int(np.sum(times >= t))
            if risk > 0: surv *= (1 - d / risk)
            i += 1
        S.append(surv)
    return np.array(S)


def _survival_logL(km_time, km_obs):
    """Censored-survival log-likelihood of the data's first-inf records under the model."""
    S = _km_survival(np.asarray(km_time), np.asarray(km_obs) == 1, GRID)
    ll = 0.0
    for a, ev in FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], EPS))     # density over [m, m+1)
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], EPS))                # survival past censor age
    return float(ll)


def _composite_logL(rec):
    """Composite log-likelihood; -inf for extinct/failed trajectories."""
    if (not rec.get('ok')) or rec.get('frac_ever_detected', 0) < 0.05:
        return -np.inf, {}
    comp = {}
    ll = 0.0
    # Poisson IR per bin
    for b, (cases, PT) in IR_DATA.items():
        lam = rec[f'ir_symp_{b}'] / 100.0 * PT
        if lam <= 0:
            return -np.inf, {}
        c = cases * np.log(lam) - lam
        comp[f'pois_{b}'] = c; ll += c
    # Binomial fractions
    p = min(max(rec['repeat_detected_frac'], 1e-6), 1 - 1e-6)
    comp['binom_repeat'] = REPEAT_OBS * np.log(p) + (REPEAT_N - REPEAT_OBS) * np.log(1 - p); ll += comp['binom_repeat']
    q = min(max(rec['frac_ever_detected'], 1e-6), 1 - 1e-6)
    comp['binom_ever'] = EVER_OBS * np.log(q) + (EVER_N - EVER_OBS) * np.log(1 - q); ll += comp['binom_ever']
    # Survival first-infection
    comp['surv_firstinf'] = _survival_logL(rec['km_time'], rec['km_observed']); ll += comp['surv_firstinf']
    return float(ll), comp


def _sir_one(args):
    idx, params, seed = args
    rec = exp06._run_one((idx, params, N_AGENTS, seed, CENS, CROSS))
    ll, comp = _composite_logL(rec)
    # Store the model's first-detection KM survival on a monthly grid (0..36) so a
    # FINER age-at-first-infection likelihood can be re-scored offline (not just the
    # single 24mo point in frac_ever_detected), without re-simulating.
    km_surv = None
    if rec.get('ok') and 'km_time' in rec:
        km_surv = [round(float(x), 5)
                   for x in _km_survival(np.asarray(rec['km_time']), np.asarray(rec['km_observed']) == 1, GRID)]
    out = dict(idx=idx, seed=seed, ok=bool(rec.get('ok')), logL=ll,
               frac_ever_detected=rec.get('frac_ever_detected'),
               repeat_detected_frac=rec.get('repeat_detected_frac'),
               km_surv=km_surv,   # model S(age) at months 0..36; 1-S(a) = fraction detected by age a
               **{f'ir_symp_{b}': rec.get(f'ir_symp_{b}') for b in ['<6 m', '6-11 m', '12-23 m']},
               **{f'par_{k}': round(v, 6) for k, v in params.items()})
    return out


def draw_nroy(n, seed=20260605, cache=None):
    # Cache the NROY draw so a resumed run uses the SAME fixed sample set (resume by idx).
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache)
    import historymatching as hm
    ckpt = HERE / 'outputs' / 'hm' / 'maled_bd' / 'checkpoint.pkl'
    # load_checkpoint is a classmethod needing the strategy objects; build a throwaway
    # engine with the same config to obtain them, then restore the checkpointed engine.
    tmp = hm.HistoryMatching(function=rw.simulate, bounds=rw.BOUNDS, observations=rw.OBSERVATIONS,
                             emulator_type='bayes_linear', sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(HERE / 'outputs' / 'hm'), run_name='maled_bd', random_seed=seed)
    engine = hm.HistoryMatching.load_checkpoint(ckpt, tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    # Record NROY draw statistics (historymatching#251: get_nroy_samples doesn't
    # return them) from the engine's tracked results: NROY fraction = box acceptance rate.
    try:
        res = engine.get_all_results()
        nf = float(res[-1].nroy_fraction) if res else None
        stats = dict(nroy_fraction=nf, n_drawn=int(n),
                     est_box_candidates=int(n / nf) if nf else None,
                     status_summary=engine.get_status_summary())
        json.dump(stats, open(HERE / 'outputs' / 'nroy_stats.json', 'w'), indent=2)
        print(f'NROY fraction {nf:.4f}; ~{int(n/nf):,} box candidates for {n} draws -> nroy_stats.json', flush=True)
    except Exception as e:
        print(f'NROY stats recording failed (non-fatal): {e!r}', flush=True)
    return engine.get_nroy_samples(n, method='lhs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--n-agents', type=int, default=40_000)
    ap.add_argument('--n-workers', type=int, default=80)  # trimmed for headroom vs the starsim leak
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=str(HERE / 'outputs' / 'sir_results.jsonl'))
    args = ap.parse_args()
    global N_AGENTS
    if args.smoke:
        args.n, N_AGENTS, args.out = 30, 8000, str(HERE / 'outputs' / 'sir_smoke.jsonl')
    else:
        N_AGENTS = args.n_agents

    cache = HERE / 'outputs' / ('nroy_draw_smoke.csv' if args.smoke else 'nroy_draw.csv')
    print(f'Drawing {args.n} NROY samples (lhs) from checkpoint (cache={cache.name})...', flush=True)
    nroy = draw_nroy(args.n, cache=cache)
    if not cache.exists():
        nroy.to_csv(cache, index=False)
    print(f'  {len(nroy)} NROY samples; columns: {list(nroy.columns)}', flush=True)

    outp = Path(args.out)
    done_idx = set()
    if outp.exists():
        done_idx = {json.loads(l)['idx'] for l in open(outp)}
        print(f'  resuming: {len(done_idx)} already simulated', flush=True)
    tasks = [(i, rw._untransform(row), 20260605 + i)
             for i, (_, row) in enumerate(nroy.iterrows()) if i not in done_idx]

    t0 = sc.tic(); done = len(done_idx)
    if tasks:
        # maxtasksperchild recycles each worker after a few sims, releasing memory
        # leaked across sequential starsim runs (starsim sequential-leak bug) -- without
        # it, 118 workers x ~85 sequential persisting sims accumulate until OOM.
        with get_context('spawn').Pool(processes=min(args.n_workers, len(tasks)),
                                       maxtasksperchild=4) as pool:
            for out in pool.imap_unordered(_sir_one, tasks):
                with outp.open('a') as f: f.write(json.dumps(out) + '\n')
                done += 1
                if done % 200 == 0 or done == args.n:
                    print(f'  {done}/{args.n} simulated, {sc.toc(t0, output=True):.0f}s', flush=True)
    print(f'\nSimulated total {done} -> {outp}', flush=True)

    # --- importance resample ---
    recs = [json.loads(l) for l in open(outp)]
    logL = np.array([r['logL'] for r in recs], float)
    finite = np.isfinite(logL)
    print(f'finite-weight (persisting & scored) trajectories: {finite.sum()}/{len(recs)}')
    w = np.zeros(len(recs))
    if finite.any():
        w[finite] = np.exp(logL[finite] - np.nanmax(logL[finite]))
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    print(f'ESS = {ess:.1f}  ({100*ess/len(recs):.1f}% of {len(recs)})')
    rng = np.random.default_rng(0)
    post_idx = rng.choice(len(recs), size=len(recs), replace=True, p=w)
    pars = [k for k in recs[0] if k.startswith('par_')]
    posterior = pd.DataFrame([{k: recs[i][k] for k in pars} for i in post_idx])
    posterior.to_csv(HERE / 'outputs' / 'posterior.csv', index=False)
    print(f'Saved posterior ({len(posterior)} resamples) -> outputs/posterior.csv')
    # quick posterior summary
    print('\nPosterior marginals (median [IQR]):')
    for k in pars:
        v = posterior[k]; print(f'  {k[4:]:>24}: {v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]')


if __name__ == '__main__':
    main()
