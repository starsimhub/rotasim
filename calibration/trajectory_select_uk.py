"""Trajectory selection for the UK SURVEILLANCE anchor (companion to trajectory_select.py).

Draws NROY samples from a UK HM run (hm_calibrate_uk), runs the surveillance sim once per
draw (ONE fixed seed = BASE+idx, reproducible), scores each with a MULTINOMIAL log-likelihood
on the observed case age-distribution, and importance-resamples to a posterior.

Likelihood (shape-only): for kept age bins with observed counts n_b and model-predicted
proportions p_b (mo['case_proportions']),
    logL = sum_b  n_b * log(p_b)            (multinomial kernel; constant term dropped)
-inf for extinct / no-case draws. Cap (cap_age_m) excludes older cases to match the cohort
window -- the SAME cap used to build the targets, so counts and model bins align.

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python trajectory_select_uk.py --model age_binned --fix-titer-shape
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python trajectory_select_uk.py --model infnum     --fix-titer-shape
"""
import os, sys, json, argparse, pathlib
from multiprocessing import get_context
import numpy as np
import pandas as pd
import sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm                       # noqa: E402
import process_surveillance_uk as PU               # noqa: E402
from hm_calibrate import bounds_for, untransform   # noqa: E402
import hm_calibrate_uk as UK                        # noqa: E402

BASE_SEED = 20260618
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '118'))
EPS = 1e-9
MIN_CASES = 50          # too few cases -> shape too noisy to score; treat as failed draw
EXP_DIR = {'age_binned': '28_hm_uk_age_binned', 'infnum': '28_hm_uk_infnum', 'age': '28_hm_uk_age'}


def _read_jsonl(path):
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _multinom_logL(mo, counts):
    """Multinomial log-L (kernel) of observed per-bin counts under model proportions."""
    if mo.get('total_cases', 0) < MIN_CASES:
        return -np.inf
    p = np.asarray(mo['case_proportions'], float)
    if p.shape[0] != len(counts) or p.sum() <= 0:
        return -np.inf
    return float(np.sum(np.asarray(counts) * np.log(np.maximum(p, EPS))))


def _score_one(args):
    idx, sim_config, sp, seed, counts = args
    try:
        mo = cm._run_one_replicate((sim_config, sp, int(seed), CAL_WINDOW))
        ll = _multinom_logL(mo, counts)
        out = dict(idx=int(idx), seed=int(seed), logL=ll,
                   total_cases=int(mo.get('total_cases', 0)),
                   case_proportions=[round(float(x), 5) for x in mo.get('case_proportions', [])])
    except Exception as e:
        out = dict(idx=int(idx), seed=int(seed), logL=None, error=repr(e)[:200])
    return out


def draw_nroy(hm_dir, bounds, run_name, n, cache, obs):
    if cache.exists():
        return pd.read_csv(cache)
    ckpt = hm_dir / 'checkpoint.pkl'
    tmp = hm.HistoryMatching(function=lambda df: df, bounds=bounds, observations=obs,
                             emulator_type='bayes_linear', sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(hm_dir.parent), run_name=run_name, random_seed=20260618)
    engine = hm.HistoryMatching.load_checkpoint(ckpt, tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    nroy = engine.get_nroy_samples(n, method='lhs')
    nroy.to_csv(cache, index=False)
    return nroy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned'])
    ap.add_argument('--maternal', choices=['titer', 'erlang'], default='titer')
    ap.add_argument('--fix-titer-shape', action='store_true')
    ap.add_argument('--cap-age-months', type=float, default=UK.DEFAULT_CAP_M,
                    help='upper age cap (mo): 36 -> <3y (default), 24 -> <2y, 0 -> no cap')
    ap.add_argument('--hm-dir', default=None, help='UK HM run folder with checkpoint.pkl')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--early-stop', action='store_true')
    ap.add_argument('--early-stop-burn-in', type=float, default=2.0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    n_agents = N_AGENTS
    if a.smoke:
        a.n = 24; n_agents = 8000
    cap = None if a.cap_age_months == 0 else a.cap_age_months

    run_name = f'uk_{a.model}_{a.maternal}' + ('_fixedshape' if a.fix_titer_shape else '')
    hm_dir = pathlib.Path(a.hm_dir) if a.hm_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs' / 'hm' / run_name)
    bounds = bounds_for(a.model, a.maternal, a.fix_titer_shape)
    obs = UK.make_observations(cap)
    counts = PU.load_targets_uk(cap_age_m=cap)['counts']
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else (THISDIR / 'experiments' / EXP_DIR[a.model] / 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ('nroy_draw_smoke.csv' if a.smoke else 'nroy_draw.csv')
    jsonl = out_dir / ('sir_smoke.jsonl' if a.smoke else 'sir_results.jsonl')

    print(f"UK trajectory selection model={a.model} maternal={a.maternal} fix_shape={a.fix_titer_shape} cap={cap}mo  N={a.n}  workers={N_WORKERS}  agents={n_agents}")
    print(f"  HM dir : {hm_dir}")
    print(f"likelihood: Multinomial(case age-dist, counts={list(map(int, counts))}, MIN_CASES={MIN_CASES})")
    nroy = draw_nroy(hm_dir, bounds, run_name, a.n, cache, obs)
    print(f"NROY draw: {len(nroy)} samples (cache={cache.name})")

    sim_config = UK.build_sim_config(a.model, n_agents, a.maternal, cap_age_m=cap)
    sim_config['early_stop_extinct'] = a.early_stop
    sim_config['early_stop_burn_in_years'] = a.early_stop_burn_in
    done = {r['idx'] for r in _read_jsonl(jsonl)}
    if done:
        print(f"resuming: {len(done)} already scored")
    tasks = [(i, sim_config, untransform(row, a.model, a.maternal, a.fix_titer_shape), BASE_SEED + i, counts)
             for i, (_, row) in enumerate(nroy.iterrows()) if i not in done]

    t0 = sc.tic(); n_done = len(done)
    if tasks:
        with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
            for out in pool.imap_unordered(_score_one, tasks):
                with jsonl.open('a') as f:
                    f.write(json.dumps(out) + '\n')
                n_done += 1
                if n_done % 200 == 0 or n_done == a.n:
                    print(f"  {n_done}/{a.n} scored, {sc.toc(t0, output=True):.0f}s", flush=True)

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
    nroy = nroy.reset_index(drop=True)
    posterior = nroy.iloc[[recs[i]['idx'] for i in post_idx]].reset_index(drop=True)
    posterior.to_csv(out_dir / ('posterior_smoke.csv' if a.smoke else 'posterior.csv'), index=False)
    json.dump(dict(model=a.model, n=a.n, cap=cap, finite=int(finite.sum()), ess=ess,
                   max_logL=float(np.nanmax(logL[finite])) if finite.any() else None),
              open(out_dir / ('ts_stats_smoke.json' if a.smoke else 'ts_stats.json'), 'w'), indent=2)
    print(f"saved posterior ({len(posterior)} resamples) + ts_stats.json")


if __name__ == '__main__':
    main()
