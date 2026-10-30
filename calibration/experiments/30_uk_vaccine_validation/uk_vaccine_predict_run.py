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
    r = dict(prop=mo['case_proportions'], total=mo['total_cases'])
    for k in ('cases_vax', 'cases_unvax', 'py_vax', 'py_unvax'):   # direct-VE split (vax sims only)
        if k in mo:
            r[k] = mo[k]
    return r


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
    def ve_group(e, lo, hi):   # per-draw achieved VE over age bins [lo:hi) = 1 - vax_cases/novax_cases
        out_ve = []
        for i in healthy:
            nv = by[(i, 'novax', None)]; vx = by[(i, 'vax', e)]
            nc = nv['total'] * sum(nv['prop'][lo:hi]); vc = vx['total'] * sum(vx['prop'][lo:hi])
            if nc > 0:
                out_ve.append(1 - vc / nc)
        return out_ve
    q3 = lambda a: (float(np.median(a)), float(np.quantile(a, .1)), float(np.quantile(a, .9))) if a else (float('nan'),)*3
    def direct_ve(e, lo, hi):  # POOLED test-negative-analog VE = 1 - IRR(vax/unvax), counts pooled over draws
        cv = cu = pv = pu = 0.0
        for i in healthy:
            vx = by[(i, 'vax', e)]
            if 'cases_vax' not in vx:
                continue
            cv += sum(vx['cases_vax'][lo:hi]); cu += sum(vx['cases_unvax'][lo:hi])
            pv += sum(vx['py_vax'][lo:hi]);    pu += sum(vx['py_unvax'][lo:hi])
        ve = (1 - (cv / pv) / (cu / pu)) if (pv > 0 and pu > 0 and cu > 0) else float('nan')
        return dict(ve=float(ve), cases_vax=cv, cases_unvax=cu, py_vax=pv, py_unvax=pu)
    for e in EFF:
        out[f'vax_{e}'] = _agg([by[(i, 'vax', e)]['prop'] for i in healthy])
        if healthy:
            for tag, (lo, hi) in {'ve': (0, 6), 've12': (0, 2), 've12_59': (2, 6)}.items():  # overall / <12mo / 12-59mo
                m, l, h = q3(ve_group(e, lo, hi))
                out[f'vax_{e}'].update(**{f'{tag}_med': m, f'{tag}_lo': l, f'{tag}_hi': h})
            for tag, (lo, hi) in {'direct': (0, 6), 'direct12': (0, 2), 'direct12_59': (2, 6)}.items():
                out[f'vax_{e}'][tag] = direct_ve(e, lo, hi)   # restricted to ages >= last dose (4mo)
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
        d12 = v.get('direct12', {}); d = v.get('direct', {})
        print(f"  vax eff={e}: <12mo case-frac {f12(v['prop_med'])}  |  TOTAL VE: overall {v.get('ve_med', float('nan')):.2f} <12mo {v.get('ve12_med', float('nan')):.2f}"
              f"  |  DIRECT VE (>=4mo): overall {d.get('ve', float('nan')):.2f} <12mo {d12.get('ve', float('nan')):.2f}"
              f"  (unvax cases<12mo={d12.get('cases_unvax', 0):.0f})")
    print("wrote outputs/uk_vaccine_predict.json")


if __name__ == '__main__':
    main()
