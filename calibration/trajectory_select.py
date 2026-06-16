"""Trajectory selection (stage 2) on a converged HM NROY -> posterior, for the MAL-ED
two-model VE work (exp 18 = age, exp 19 = infnum). Mirrors hm_calibrate.py: one script,
parameterized by --model. Draws N NROY samples from the model's HM checkpoint, simulates each
(cohort observation, ONE fixed seed per draw = BASE+idx, reproducible), scores with a COMPOSITE
log-likelihood, and importance-resamples to a posterior.

Composite log-likelihood (agreed 2026-06-11; Dan Klein's exp-07 form, minus ever-detected):
  - IR by age bin (<6, 6-11, 12-23):  Poisson( cases | lambda = IR/100 * PT )
  - repeat-detected fraction:          Binomial( obs | n, model_p )
  - age-at-first-infection:            censored-survival LL of every infant's (age, event)
                                       record under the model's monthly KM first-detection survival
  - NO ever-detected channel (a marginal of the survival data -> would double-count).
  - extinct / failed sims -> log-L = -inf (dropped).

Run in the pinned env (calibration/hm_env_pins.txt) on a 120-core VM, in tmux (spot):
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=118 python trajectory_select.py --model age   --n 5000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=118 python trajectory_select.py --model infnum --n 5000
Resumable: per-sim results stream to outputs/sir_results.jsonl (skip done idx); NROY draw cached.
"""
import os, sys, json, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm
import process_incidence_maled as P


def _read_jsonl(path):
    """Robustly read a JSONL stream, skipping null-padded/truncated lines left by a
    spot-eviction mid-write (so a resumed run self-heals instead of crashing)."""
    out = []
    if not pathlib.Path(path).exists():
        return out
    for line in open(path, errors='ignore'):
        line = line.replace('\x00', '').strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

SITE = 'bangladesh'
BASE_SEED = 20260611
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '118'))
IR_BINS = ['<6 m', '6-11 m', '12-23 m']
GRID = np.arange(0, 37)            # monthly edges for the survival LL
EPS = 1e-9

EXP_DIR = {'age': '18_age_posterior', 'infnum': '19_infnum_posterior'}
HM_RUN = {'age':    THISDIR / 'experiments' / '16_hm_age_titer'   / 'outputs' / 'hm' / 'maled_age_titer',
          'infnum': THISDIR / 'experiments' / '17_hm_infnum_titer' / 'outputs' / 'hm' / 'maled_infnum_titer'}

# ---- targets for the likelihood ----
_t = P.load_targets(SITE)
_ir = _t['ir_by_age']
IR_DATA = {b: (int(_ir.loc[b, 'cases']), float(_ir.loc[b, 'PT'])) for b in IR_BINS}
_rf = _t['repeat_frac']                               # {'frac', 'se', 'n'}
REPEAT_N = int(_rf['n']); REPEAT_OBS = int(round(_rf['frac'] * REPEAT_N))
_fi = pd.read_csv(THISDIR / 'maled_data' / f'first_infection_{SITE}.csv')
FIRSTINF = _fi[['age_event_months', 'event_observed']].dropna().values


def _km_survival(times, observed, grid):
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
    S = _km_survival(np.asarray(km_time), np.asarray(km_obs) == 1, GRID)
    ll = 0.0
    for a, ev in FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], EPS))    # density over [m, m+1)
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], EPS))                # survival past censor age
    return float(ll)


def _composite_logL(mo):
    """Composite log-L from a model_out dict; -inf for extinct/failed."""
    km_obs = np.asarray(mo.get('km_observed', []))
    if km_obs.size == 0 or km_obs.mean() < 0.05 or float(mo['ir_by_age']['IR'].sum()) <= 0:
        return -np.inf, {}
    comp = {}; ll = 0.0
    ir = mo['ir_by_age']
    for b, (cases, PT) in IR_DATA.items():
        lam = ir.loc[b, 'IR'] / 100.0 * PT
        if lam <= 0:
            return -np.inf, {}
        comp[f'pois_{b}'] = cases * np.log(lam) - lam; ll += comp[f'pois_{b}']
    p = min(max(float(mo['repeat_frac']), 1e-6), 1 - 1e-6)
    comp['binom_repeat'] = REPEAT_OBS * np.log(p) + (REPEAT_N - REPEAT_OBS) * np.log(1 - p); ll += comp['binom_repeat']
    comp['surv_firstinf'] = _survival_logL(mo['km_time'], mo['km_observed']); ll += comp['surv_firstinf']
    return float(ll), comp


def _score_one(args):
    idx, sim_config, sp, seed = args
    try:
        mo = cm._run_one_replicate((sim_config, sp, int(seed), CAL_WINDOW))
        ll, comp = _composite_logL(mo)
        km = _km_survival(np.asarray(mo['km_time']), np.asarray(mo['km_observed']) == 1, GRID)
        out = dict(idx=int(idx), seed=int(seed), logL=ll,
                   repeat_frac=mo.get('repeat_frac'),
                   km_surv=[round(float(x), 5) for x in km],
                   **{f'ir_{b}': float(mo['ir_by_age'].loc[b, 'IR']) for b in IR_BINS})
    except Exception as e:
        out = dict(idx=int(idx), seed=int(seed), logL=None, error=repr(e)[:200])
    return out


def draw_nroy(hm_dir, bounds, run_name, n, cache):
    if cache.exists():
        return pd.read_csv(cache)
    ckpt = hm_dir / 'checkpoint.pkl'
    tmp = hm.HistoryMatching(function=lambda df: df, bounds=bounds, observations=OBS,
                             emulator_type='bayes_linear', sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(hm_dir.parent), run_name=run_name, random_seed=20260610)
    engine = hm.HistoryMatching.load_checkpoint(ckpt, tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    nroy = engine.get_nroy_samples(n, method='lhs')
    nroy.to_csv(cache, index=False)
    return nroy


# import here so the names exist for draw_nroy
from hm_calibrate import bounds_for, make_observations, build_sim_config, untransform
OBS = make_observations()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned'])
    ap.add_argument('--maternal', choices=['titer', 'erlang'], default='titer')
    ap.add_argument('--fix-titer-shape', action='store_true')
    ap.add_argument('--hm-dir', default=None, help='HM run folder with checkpoint.pkl (default: free-titer exp 16/17)')
    ap.add_argument('--out-dir', default=None, help='where to write posterior/sir_results (default: exp 18/19)')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--early-stop', action='store_true',
                    help='abort extinct draws early (StopWhenExtinct); ~3.5x faster on the ~80%% that burn out')
    ap.add_argument('--early-stop-burn-in', type=float, default=2.0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    n_agents = N_AGENTS
    if a.smoke:
        a.n = 24; n_agents = 8000

    hm_dir = pathlib.Path(a.hm_dir) if a.hm_dir else HM_RUN[a.model]
    run_name = hm_dir.name
    bounds = bounds_for(a.model, a.maternal, a.fix_titer_shape)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ('nroy_draw_smoke.csv' if a.smoke else 'nroy_draw.csv')
    jsonl = out_dir / ('sir_smoke.jsonl' if a.smoke else 'sir_results.jsonl')

    print(f"trajectory selection model={a.model} maternal={a.maternal} fix_shape={a.fix_titer_shape}  N={a.n}  workers={N_WORKERS}  agents={n_agents}")
    print(f"  HM dir : {hm_dir}")
    print(f"  out dir: {out_dir}")
    print(f"likelihood: Poisson(IR {IR_BINS}) + Binomial(repeat {REPEAT_OBS}/{REPEAT_N}) + survival(first-inf, {len(FIRSTINF)} records); no ever-detected")
    nroy = draw_nroy(hm_dir, bounds, run_name, a.n, cache)
    print(f"NROY draw: {len(nroy)} samples (cache={cache.name})")

    sim_config = build_sim_config(a.model, n_agents, a.maternal)
    sim_config['early_stop_extinct'] = a.early_stop
    sim_config['early_stop_burn_in_years'] = a.early_stop_burn_in
    done = {r['idx'] for r in _read_jsonl(jsonl)}
    if done:
        print(f"resuming: {len(done)} already scored")
    tasks = [(i, sim_config, untransform(row, a.model, a.maternal, a.fix_titer_shape), BASE_SEED + i)
             for i, (_, row) in enumerate(nroy.iterrows()) if i not in done]

    t0 = sc.tic(); n_done = len(done)
    if tasks:
        with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
            for out in pool.imap_unordered(_score_one, tasks):
                with jsonl.open('a') as f: f.write(json.dumps(out) + '\n')
                n_done += 1
                if n_done % 200 == 0 or n_done == a.n:
                    print(f"  {n_done}/{a.n} scored, {sc.toc(t0, output=True):.0f}s", flush=True)

    # ---- importance resample ----
    recs = _read_jsonl(jsonl)
    logL = np.array([r['logL'] if r.get('logL') is not None else -np.inf for r in recs], float)
    finite = np.isfinite(logL)
    print(f"finite-logL trajectories: {finite.sum()}/{len(recs)}")
    w = np.zeros(len(recs))
    if finite.any():
        w[finite] = np.exp(logL[finite] - np.nanmax(logL[finite]))
    w = w / w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    print(f"ESS = {ess:.1f}  ({100*ess/len(recs):.1f}% of {len(recs)})")
    rng = np.random.default_rng(0)
    post_idx = rng.choice(len(recs), size=len(recs), replace=True, p=w)
    idx_to_row = {r['idx']: r for r in recs}
    # carry parameters (transformed) from the cached NROY draw, by idx
    nroy = nroy.reset_index(drop=True)
    posterior = nroy.iloc[[recs[i]['idx'] for i in post_idx]].reset_index(drop=True)
    posterior.to_csv(out_dir / ('posterior_smoke.csv' if a.smoke else 'posterior.csv'), index=False)
    json.dump(dict(model=a.model, n=a.n, finite=int(finite.sum()), ess=ess,
                   max_logL=float(np.nanmax(logL[finite])) if finite.any() else None),
              open(out_dir / ('ts_stats_smoke.json' if a.smoke else 'ts_stats.json'), 'w'), indent=2)
    print(f"saved posterior ({len(posterior)} resamples) + ts_stats.json")


if __name__ == '__main__':
    main()
