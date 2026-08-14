"""Trajectory selection (stage 2) on a converged HM NROY -> posterior, for the MAL-ED
two-model VE work (exp 18 = age, exp 19 = infnum). Mirrors hm_calibrate.py: one script,
parameterized by --model. Draws N NROY samples from the model's HM checkpoint, simulates each
with N_SURVIVAL_SEEDS independent seeds (default 5, BASE+idx*seeds_per_draw+j), scores with a
COMPOSITE log-likelihood, and importance-resamples to a posterior.

Composite log-likelihood (agreed 2026-06-11; Dan Klein's exp-07 form, minus ever-detected):
  - IR by age bin (<6, 6-11, 12-23):  Poisson( cases | lambda = IR/100 * PT )
  - repeat-detected fraction:          Binomial( obs | n, model_p )
  - age-at-first-infection:            censored-survival LL of every infant's (age, event)
                                       record under the model's monthly KM first-detection survival
  - NO ever-detected channel (a marginal of the survival data -> would double-count).
  - optional VE term (--ve-target): Gaussian logL on population-impact VE at 6-11m, estimated
    from a paired novax+vax surveillance sim (single seed; Rotavac 3-dose at 6/10/14 weeks, 90%
    coverage). Written to sir_results_ve<N>.jsonl to preserve existing cohort-only results.

Multi-seed survival vote (replaces the old "extinct -> logL=-inf, dropped" rule; see
experiments/50/51's finding that extinction near the viable boundary is genuinely seed-dependent,
not just noise on top of a deterministic outcome -- a single unlucky seed could zero out a
parameter point that's actually viable most of the time). For each draw, run N_SURVIVAL_SEEDS
(default 5) replicates. Among the k that survive, the composite logL is combined via
log-mean-exp (a Monte Carlo estimate of E[L(D_obs|theta,survives)], not a mean of logs) and then
multiplied by frac_survived = (k+1)/(n+2) (Laplace-smoothed): this is the hurdle-model
decomposition P(D_obs|theta) = P(survives|theta) x f(D_obs|theta,survives), justified because
the real Vellore/Bangladesh cohort implicitly conditions on the population having sustained
transmission. All n seeds extinct -> logL=-inf as before (dropped).

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

SITE = os.environ.get('MALED_SITE', 'bangladesh')
BASE_SEED = 20260611
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '118'))
IR_BINS = ['<6 m', '6-11 m', '12-23 m']
GRID = np.arange(0, 37)            # monthly edges for the survival LL
EPS = 1e-9

# VE scoring constants (Rotavac 3-dose Universal Immunization Programme schedule)
VE_BIN_EDGES  = (0.0, 6.0, 12.0)   # only need <6m and 6-11m bins
VE_DOSE_AGES_Y = [6/52, 10/52, 14/52]
VE_COVERAGE    = 0.90

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


def _build_ve_sim_configs(base_model, n_agents, maternal, ve_take):
    """Return (novax_surv_cfg, vax_surv_cfg) for paired VE scoring at 6-11m."""
    from hm_calibrate import build_sim_config as _bsc
    sc_novax = _bsc(base_model, n_agents, maternal)
    sc_novax['observation'] = 'surveillance'
    sc_novax['bin_edges_m'] = VE_BIN_EDGES
    sc_novax['cap_age_m']   = 12.0
    sc_vax = dict(sc_novax)
    sc_vax['vaccine'] = dict(response_prob=ve_take, coverage=VE_COVERAGE,
                             dose_ages_y=VE_DOSE_AGES_Y, moa='infection_blocking')
    return sc_novax, sc_vax


def _score_one(args):
    idx, sim_config, ve_novax_cfg, ve_vax_cfg, ve_target, ve_sigma, sp, seed0, n_seeds = args
    try:
        logLs, mo0 = [], None
        for j in range(n_seeds):
            mo = cm._run_one_replicate((sim_config, sp, int(seed0) + j, CAL_WINDOW))
            ll, _ = _composite_logL(mo)
            if np.isfinite(ll):
                logLs.append(ll)
                if mo0 is None:
                    mo0 = mo   # first survivor -- used for the reported km_surv/repeat_frac/ir_ diagnostics
        k = len(logLs)
        frac_survived = (k + 1.0) / (n_seeds + 2.0)   # Laplace-smoothed vote
        if k == 0:
            ll_final = -np.inf
        else:
            # E[L(D_obs|theta,survives)] via the k survivor replicates: log-mean-exp of their
            # composite logLs (a Monte Carlo average of likelihoods, NOT a mean of log-likelihoods),
            # then the hurdle-model P(survives)xP(D_obs|survives) decomposition in log space.
            ll_final = float(np.logaddexp.reduce(logLs) - np.log(k) + np.log(frac_survived))
        ve = float('nan')
        if ve_target is not None and ve_novax_cfg is not None and mo0 is not None and np.isfinite(ll_final):
            try:
                novax_mo = cm._run_one_replicate((ve_novax_cfg, sp, int(seed0), CAL_WINDOW))
                vax_mo   = cm._run_one_replicate((ve_vax_cfg,   sp, int(seed0), CAL_WINDOW))
                # ir_per_100cy: list[float] per bin; index 1 = 6-11m
                ir_novax = novax_mo.get('ir_per_100cy', [0.0, 0.0])
                ir_vax   = vax_mo.get('ir_per_100cy',   [0.0, 0.0])
                ir_n = float(ir_novax[1]) if len(ir_novax) > 1 else 0.0
                ir_v = float(ir_vax[1])   if len(ir_vax)   > 1 else 0.0
                if ir_n > 0:
                    ve = 1.0 - ir_v / ir_n
                    ll_final += -0.5 * ((ve - ve_target) / ve_sigma) ** 2
            except Exception:
                pass   # VE scoring failure is non-fatal; cohort logL still recorded
        km = (_km_survival(np.asarray(mo0['km_time']), np.asarray(mo0['km_observed']) == 1, GRID)
              if mo0 is not None else np.full(len(GRID), np.nan))
        out = dict(idx=int(idx), seed=int(seed0), logL=ll_final, ve=ve,
                   n_survived=k, n_seeds=n_seeds, frac_survived=frac_survived,
                   repeat_frac=(mo0.get('repeat_frac') if mo0 is not None else None),
                   km_surv=[round(float(x), 5) for x in km],
                   **({f'ir_{b}': float(mo0['ir_by_age'].loc[b, 'IR']) for b in IR_BINS} if mo0 is not None else {}))
    except Exception as e:
        out = dict(idx=int(idx), seed=int(seed0), logL=None, ve=float('nan'), error=repr(e)[:200])
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
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned', 'age_and_infection'])
    ap.add_argument('--maternal', choices=['titer', 'erlang'], default='titer')
    ap.add_argument('--fix-titer-shape', action='store_true')
    ap.add_argument('--hm-dir', default=None, help='HM run folder with checkpoint.pkl (default: free-titer exp 16/17)')
    ap.add_argument('--out-dir', default=None, help='where to write posterior/sir_results (default: exp 18/19)')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--early-stop', action='store_true',
                    help='abort extinct draws early (StopWhenExtinct); ~3.5x faster on the ~80%% that burn out')
    ap.add_argument('--early-stop-burn-in', type=float, default=2.0)
    ap.add_argument('--fix-psymp', action='store_true',
                    help='fix infnum p_symp params at Vellore biweekly values (exp32); must match the HM run flag')
    ap.add_argument('--fix-age-psymp', action='store_true',
                    help='fix age_binned p_symp per-bin at Vellore biweekly values; must match the HM run flag')
    ap.add_argument('--ve-target', type=float, default=None,
                    help='add Gaussian VE logL term centred at this population-impact value (e.g. 0.50). '
                         'Requires a paired novax+vax surveillance sim per draw (~3× compute). '
                         'Results written to sir_results_ve<N>.jsonl to preserve existing outputs.')
    ap.add_argument('--ve-sigma', type=float, default=0.07,
                    help='Gaussian sigma for the VE target (default 0.07 ≈ ±14 pp at 2σ)')
    ap.add_argument('--ve-take', type=float, default=0.74,
                    help='Rotavac seroconversion take rate for VE scoring (default 0.74 = middle-income)')
    ap.add_argument('--n-survival-seeds', type=int, default=5,
                    help='independent seeds per NROY draw for the survival vote + composite logL '
                         '(default 5; see module docstring). 1 recovers the old single-seed behavior, '
                         'except extinct draws now get logL=-inf via frac_survived=0 rather than being '
                         'silently NaN\'d -- same net effect.')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    n_agents = N_AGENTS
    if a.smoke:
        a.n = 24; n_agents = 8000

    hm_dir = pathlib.Path(a.hm_dir) if a.hm_dir else HM_RUN[a.model]
    run_name = hm_dir.name
    bounds = bounds_for(a.model, a.maternal, a.fix_titer_shape, getattr(a, 'fix_psymp', False), getattr(a, 'fix_age_psymp', False))
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)

    # VE suffix keeps VE-scored results separate from plain cohort results
    ve_suffix = f'_ve{int(a.ve_target * 100)}' if a.ve_target is not None else ''
    cache = out_dir / ('nroy_draw_smoke.csv' if a.smoke else 'nroy_draw.csv')
    jsonl = out_dir / (f'sir_smoke{ve_suffix}.jsonl' if a.smoke else f'sir_results{ve_suffix}.jsonl')

    print(f"trajectory selection model={a.model} maternal={a.maternal} fix_shape={a.fix_titer_shape}  N={a.n}  workers={N_WORKERS}  agents={n_agents}  seeds/draw={a.n_survival_seeds}")
    print(f"  HM dir : {hm_dir}")
    print(f"  out dir: {out_dir}")
    print(f"likelihood: Poisson(IR {IR_BINS}) + Binomial(repeat {REPEAT_OBS}/{REPEAT_N}) + survival(first-inf, {len(FIRSTINF)} records); no ever-detected")

    # Optional VE scoring
    ve_novax_cfg = ve_vax_cfg = None
    if a.ve_target is not None:
        ve_novax_cfg, ve_vax_cfg = _build_ve_sim_configs(a.model, n_agents, a.maternal, a.ve_take)
        print(f"VE target: {a.ve_target:.2f} ± {a.ve_sigma:.2f} (Gaussian sigma), take={a.ve_take}, coverage={VE_COVERAGE}")
        print(f"  Rotavac: 3 doses at {[round(d*52,1) for d in VE_DOSE_AGES_Y]} weeks | +2 surveillance sims per draw")
        print(f"  Output: {jsonl.name}")

    nroy = draw_nroy(hm_dir, bounds, run_name, a.n, cache)
    print(f"NROY draw: {len(nroy)} samples (cache={cache.name})")

    sim_config = build_sim_config(a.model, n_agents, a.maternal)
    sim_config['early_stop_extinct'] = a.early_stop
    sim_config['early_stop_burn_in_years'] = a.early_stop_burn_in
    done = {r['idx'] for r in _read_jsonl(jsonl)}
    if done:
        print(f"resuming: {len(done)} already scored")
    tasks = [(i, sim_config, ve_novax_cfg, ve_vax_cfg, a.ve_target, a.ve_sigma,
              untransform(row, a.model, a.maternal, a.fix_titer_shape,
                          getattr(a, 'fix_psymp', False), getattr(a, 'fix_age_psymp', False)),
              BASE_SEED + i * a.n_survival_seeds, a.n_survival_seeds)
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

    # VE diagnostics
    if a.ve_target is not None:
        ve_vals = np.array([r.get('ve', float('nan')) for r in recs], float)
        ve_fin  = np.isfinite(ve_vals)
        if ve_fin.any():
            ve_med = float(np.median(ve_vals[ve_fin]))
            ve_w   = w[ve_fin] / w[ve_fin].sum() if w[ve_fin].sum() > 0 else w[ve_fin]
            ve_wmean = float(np.sum(ve_w * ve_vals[ve_fin]))
            print(f"VE at 6-11m: median={ve_med:.3f}  weighted-mean={ve_wmean:.3f}  "
                  f"(target={a.ve_target:.2f}, n_finite={ve_fin.sum()})")

    rng = np.random.default_rng(0)
    post_idx = rng.choice(len(recs), size=len(recs), replace=True, p=w)
    idx_to_row = {r['idx']: r for r in recs}
    # carry parameters (transformed) from the cached NROY draw, by idx
    nroy = nroy.reset_index(drop=True)
    posterior = nroy.iloc[[recs[i]['idx'] for i in post_idx]].reset_index(drop=True)
    post_csv  = out_dir / (f'posterior_smoke{ve_suffix}.csv' if a.smoke else f'posterior{ve_suffix}.csv')
    stats_json = out_dir / (f'ts_stats_smoke{ve_suffix}.json' if a.smoke else f'ts_stats{ve_suffix}.json')
    posterior.to_csv(post_csv, index=False)
    json.dump(dict(model=a.model, n=a.n, finite=int(finite.sum()), ess=ess,
                   max_logL=float(np.nanmax(logL[finite])) if finite.any() else None,
                   ve_target=a.ve_target, ve_sigma=a.ve_sigma, ve_take=a.ve_take),
              open(stats_json, 'w'), indent=2)
    print(f"saved posterior ({len(posterior)} resamples) → {post_csv.name} + {stats_json.name}")


if __name__ == '__main__':
    main()
