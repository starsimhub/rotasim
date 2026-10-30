"""Trajectory selection for exp37 — UK HM (infnum) posterior.

Draws N NROY samples from the wave-3 checkpoint, re-simulates each (no-vaccine,
fixed seed), and scores with multinomial log-likelihood on case age-proportions vs
the UK surveillance counts. Writes streaming JSONL + posterior CSV.

Run on zebra:
  cd /home/akraay/rotasim/rotasim/calibration
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=150 \\
    python experiments/37_uk_seroprev_hm/ts_exp37.py --n 5000
"""
import os, sys, json, pathlib
from multiprocessing import get_context
import numpy as np
import pandas as pd
import sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent.parent.parent   # calibration/
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm
import process_surveillance_uk as PU
from hm_calibrate import untransform
from hm_calibrate_uk import build_sim_config, DEFAULT_CAP_M

_BD = dict(sus_after_1=0.751, sus_r2=0.615, sus_r3=0.659)
_CONSTRAINED_SUS = {k: (round(max(v * 0.50, 0.05), 4), round(min(v * 1.50, 0.99), 4))
                    for k, v in _BD.items()}

def exp37_bounds():
    return {
        'log_base_beta':     (float(np.log(0.05)), float(np.log(0.35))),
        **_CONSTRAINED_SUS,
        'maternal_efficacy': (0.50, 0.99),
        'p_symp_1':          (0.20, 1.0),
        'p_r2':              (0.0,  1.0),
        'p_r3':              (0.0,  1.0),
    }

HERE     = pathlib.Path(__file__).resolve().parent
HM_DIR   = HERE / 'outputs' / 'hm' / 'uk_infnum_ve_anchor'
OUT_DIR  = HERE / 'outputs'
CACHE    = OUT_DIR / 'nroy_draw.csv'
JSONL    = OUT_DIR / 'sir_results.jsonl'
POST_CSV = OUT_DIR / 'posterior.csv'

N_AGENTS  = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS  = int(os.environ.get('HM_WORKERS', '150'))
BASE_SEED  = 20260708
EPS        = 1e-9


def _read_jsonl(path):
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


def draw_nroy(n):
    if CACHE.exists():
        print(f'  loading cached NROY draw from {CACHE}')
        return pd.read_csv(CACHE)
    bounds = exp37_bounds()
    # Dummy HM engine just to load checkpoint
    tmp = hm.HistoryMatching(
        function=lambda df: df, bounds=bounds,
        observations={'ve_overall': (0.74, 0.05)},
        emulator_type='bayes_linear', sampling_strategy='lhs',
        feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
        output_dir=str(HM_DIR.parent), run_name='uk_infnum_ve_anchor', random_seed=20260707)
    engine = hm.HistoryMatching.load_checkpoint(
        HM_DIR / 'checkpoint.pkl',
        tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    nroy = engine.get_nroy_samples(n, method='lhs')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nroy.to_csv(CACHE, index=False)
    print(f'  NROY draw saved to {CACHE}  ({len(nroy)} rows)')
    return nroy


def _multinomial_logL(prop_model, counts):
    """Multinomial LL: sum(c_k * log(p_k)); -inf if any zero proportion in a non-zero count bin."""
    ll = 0.0
    for c, p in zip(counts, prop_model):
        if c > 0:
            if p <= 0:
                return -np.inf
            ll += c * np.log(max(p, EPS))
    return float(ll)


def _score_one(args):
    idx, sim_config, sp, seed, counts = args
    try:
        mo = cm._run_one_replicate((sim_config, sp, int(seed), CAL_WINDOW))
        tot = mo.get('total_cases', 0)
        if tot <= 0:
            return dict(idx=int(idx), seed=int(seed), logL=None, total_cases=0)
        prop = mo['case_proportions']
        ll = _multinomial_logL(prop, counts)
        return dict(idx=int(idx), seed=int(seed), logL=float(ll),
                    total_cases=int(tot),
                    case_proportions=prop)
    except Exception as e:
        return dict(idx=int(idx), seed=int(seed), logL=None, error=repr(e)[:200])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.n = 50

    t = PU.load_targets_uk(cap_age_m=DEFAULT_CAP_M)
    counts = t['counts']   # raw integer counts per bin for multinomial LL

    sim_config = build_sim_config('infnum', N_AGENTS, maternal='titer', cap_age_m=DEFAULT_CAP_M)
    sim_config['early_stop_extinct']       = True
    sim_config['early_stop_burn_in_years'] = 2.0
    sim_config['init_prevalence_override'] = 0.005
    sim_config['init_age_dist_override']   = [(0, 80, 1.0)]
    # No vaccine — shape-only for posterior scoring

    print(f'Exp37 trajectory selection: drawing {a.n} NROY samples')
    nroy = draw_nroy(a.n)
    nroy = nroy.head(a.n)

    # Resume: skip already-scored indices
    done = {r['idx'] for r in _read_jsonl(JSONL)}
    print(f'  {len(done)} already done, {len(nroy) - len(done)} remaining')

    tasks = []
    for idx, row in nroy.iterrows():
        if idx in done:
            continue
        sp   = untransform(row, 'infnum', 'titer', fix_titer_shape=True)
        seed = BASE_SEED + idx
        tasks.append((idx, sim_config, sp, int(seed), counts))

    t0 = sc.tic()
    ctx = get_context('spawn')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ctx.Pool(processes=min(N_WORKERS, max(1, len(tasks))), maxtasksperchild=4) as pool:
        for result in pool.imap_unordered(_score_one, tasks):
            with open(JSONL, 'a') as f:
                f.write(json.dumps(result) + '\n')
    print(f'  simulations done in {sc.toc(t0, output=True):.0f}s')

    # Build posterior via importance sampling from logL
    records = _read_jsonl(JSONL)
    df_all  = pd.DataFrame(records)
    df_ok   = df_all[df_all['logL'].notna() & np.isfinite(df_all['logL'].astype(float))].copy()
    df_ok['logL'] = df_ok['logL'].astype(float)
    print(f'  finite logL: {len(df_ok)}/{len(df_all)}')

    if len(df_ok) < 5:
        print('  WARNING: too few finite logL — posterior not saved')
        return

    # Merge NROY parameters back in
    nroy_idx = nroy.reset_index().rename(columns={'index': 'idx'})
    df_merged = df_ok.merge(nroy_idx, on='idx', how='left')

    # Importance weights: exp(logL - logL_max)
    lmax = df_merged['logL'].max()
    w = np.exp(df_merged['logL'].values - lmax)
    w /= w.sum()
    ess = float(1.0 / (w**2).sum())
    print(f'  ESS = {ess:.1f}  (out of {len(df_merged)} finite draws)')

    # Resample to equal-weight posterior
    rng = np.random.default_rng(42)
    idx_resample = rng.choice(len(df_merged), size=5000, replace=True, p=w)
    param_cols = [c for c in nroy.columns]
    posterior = df_merged.iloc[idx_resample][param_cols].reset_index(drop=True)
    posterior.to_csv(POST_CSV, index=False)
    print(f'  posterior saved to {POST_CSV}  (ESS={ess:.1f})')
    print(f'Exp37 TS done.')


if __name__ == '__main__':
    main()
