"""exp30 (VM step) — UK vaccine-impact VALIDATION by forward prediction (no refit to post-vaccine).

UK pre-vaccine-calibrated infnum model + vaccine (2+4mo, coverage 0.90, per-dose efficacy band) ->
predict post-vaccine case-age distribution. Validate on the age-distribution SHIFT (older), NOT
absolute counts (UK genotyped surveillance expanded post-rollout). Compare predicted shift vs
OBSERVED pre->post (UK_age_byEra: pre 2008-12, post 2015-19, vaccine-derived excluded).

Ensemble = TOP-K best-fitting UK infnum samples (by multinomial logL on the pre-vaccine target) --
the 'good fits' near the optimum, which have healthy incidence; uniform NROY draws hit the low-FOI
near-extinct tail. 40k agents to reduce stochastic extinction in the low-FOI UK regime.

NB: this is the NO-WANING baseline. Vaccine protection is permanent here, so it suppresses cases
proportionally across ages (little older shift). The observed older shift is a waning signature;
setting-specific waning (UK ~2yr, shorter in LMICs) is the follow-up.

Run on a VM:  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python uk_vaccine_predict_run.py
"""
import os, sys, json, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; CALIB = HERE.parents[1]; sys.path.insert(0, str(CALIB))
import historymatching as hm                        # noqa: E402
import calibrate_maled as cm                       # noqa: E402
from hm_calibrate import untransform               # noqa: E402
import hm_calibrate_uk as UK                        # noqa: E402
from hm_calibrate_uk import uk_bounds              # noqa: E402
import process_surveillance_uk as PU               # noqa: E402

N_WORKERS = int(os.environ.get('HM_WORKERS', '90'))
N_AGENTS, CAP, COVERAGE = 40000, 60.0, 0.90
EFF = [0.63, 0.9]                                   # per-dose seroconversion band
N_TOPK = 40                                          # top-K best-fitting UK infnum draws
MIN_NOVAX_CASES = 300                                # drop near-extinct draws (unreliable shape)


def draw_topk(k):
    """Top-k UK infnum samples by multinomial logL on the pre-vaccine target (good fits ~ healthy
    incidence), from the fixed-maternal HM run's stored samples."""
    run = 'uk_infnum_titer_fixedshape'
    d = CALIB / 'experiments' / '28_hm_uk_infnum' / 'outputs' / 'hm' / run
    b = uk_bounds('infnum', 'titer', True, UK.DEFAULT_BETA_MAX); obs = UK.make_observations(CAP)
    cols = UK.obs_cols(CAP); counts = PU.load_targets_uk_era('pre', CAP)['counts'].astype(float)
    tmp = hm.HistoryMatching(function=lambda x: x, bounds=b, observations=obs, emulator_type='bayes_linear',
                             sampling_strategy='lhs',
                             feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
                             output_dir=str(d.parent), run_name=run, random_seed=20260618)
    e = hm.HistoryMatching.load_checkpoint(d / 'checkpoint.pkl', tmp.sampling_strategy, tmp.feature_selection, tmp.emulator_factory)
    R = e.get_all_results()
    S = pd.concat([r.samples.reset_index(drop=True) for r in R], ignore_index=True)
    SR = pd.concat([r.simulation_results.reset_index(drop=True) for r in R], ignore_index=True)
    P = SR[cols].to_numpy(float); ok = np.isfinite(P).all(1) & (P.sum(1) > 0)
    ll = np.full(len(P), -np.inf); ll[ok] = (counts * np.log(np.clip(P[ok], 1e-9, None))).sum(1)
    return S.iloc[np.argsort(-ll)[:k]].reset_index(drop=True)


def _run(args):
    sp, vaccine, seed = args
    sc = UK.build_sim_config('infnum', N_AGENTS, 'titer', cap_age_m=CAP)
    if vaccine:
        sc['vaccine'] = vaccine
    mo = cm._run_one_replicate((sc, sp, int(seed), (5.0, 10.0)))
    return dict(prop=mo['case_proportions'], total=mo['total_cases'])


def _agg(P):
    if len(P) == 0:
        return dict(prop_med=[float('nan')] * 6, prop_lo=[float('nan')] * 6, prop_hi=[float('nan')] * 6, n=0)
    P = np.array(P)
    return dict(prop_med=np.median(P, 0).tolist(), prop_lo=np.quantile(P, .1, 0).tolist(),
                prop_hi=np.quantile(P, .9, 0).tolist(), n=len(P))


def main():
    draws = draw_topk(N_TOPK)
    tasks, meta = [], []
    for i, (_, row) in enumerate(draws.iterrows()):
        sp = untransform(row, 'infnum', 'titer', fix_titer_shape=True)
        seed = 50000 + i
        tasks.append((sp, None, seed)); meta.append((i, 'novax', None))
        for e in EFF:
            tasks.append((sp, dict(response_prob=e, coverage=COVERAGE, dose_ages_y=[2/12, 4/12],
                                   moa='infection_blocking'), seed)); meta.append((i, 'vax', e))
    with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
        res = pool.map(_run, tasks)
    by = {(i, k, e): r for r, (i, k, e) in zip(res, meta)}
    healthy = [i for i in range(len(draws)) if by.get((i, 'novax', None), {}).get('total', 0) >= MIN_NOVAX_CASES]
    out = {'n_topk': len(draws), 'n_healthy': len(healthy),
           'novax': _agg([by[(i, 'novax', None)]['prop'] for i in healthy])}
    for e in EFF:
        out[f'vax_{e}'] = _agg([by[(i, 'vax', e)]['prop'] for i in healthy])
        if healthy:
            ve = [1 - by[(i, 'vax', e)]['total'] / by[(i, 'novax', None)]['total'] for i in healthy]
            out[f'vax_{e}'].update(ve_med=float(np.median(ve)), ve_lo=float(np.quantile(ve, .1)), ve_hi=float(np.quantile(ve, .9)))
    out['observed_pre'] = PU.load_targets_uk_era('pre', CAP)['proportions'].tolist()
    out['observed_post'] = PU.load_targets_uk_era('post', CAP)['proportions'].tolist()
    out['bin_labels'] = PU.load_targets_uk_era('pre', CAP)['bin_labels']
    out['coverage'] = COVERAGE; out['eff'] = EFF
    (HERE / 'outputs').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'outputs' / 'uk_vaccine_predict.json', 'w'), indent=2)
    f12 = lambda p: round(sum(p[:2]), 3)
    print(f"top-{out['n_topk']} draws ({out['n_healthy']} healthy >={MIN_NOVAX_CASES} novax cases)")
    print(f"<12mo:  observed pre {f12(out['observed_pre'])} -> post {f12(out['observed_post'])};  model no-vax {f12(out['novax']['prop_med'])}")
    for e in EFF:
        v = out[f'vax_{e}']
        print(f"  vax eff={e}: <12mo {f12(v['prop_med'])} [{f12(v['prop_lo'])},{f12(v['prop_hi'])}]  VE {v.get('ve_med', float('nan')):.2f}")
    print("wrote outputs/uk_vaccine_predict.json")


if __name__ == '__main__':
    main()
