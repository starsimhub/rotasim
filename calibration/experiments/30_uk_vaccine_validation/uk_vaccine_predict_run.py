"""exp30 (VM step) — UK vaccine-impact VALIDATION by forward prediction (no refit to post-vaccine).

Take the UK pre-vaccine-calibrated infnum model, introduce the vaccine (2+4mo, coverage 0.90) at a
BAND of per-dose efficacies, and predict the post-vaccine case-age distribution. Validation metric
is the age-distribution SHIFT (older), NOT absolute counts (UK genotyped surveillance expanded
post-rollout, so counts aren't comparable). Compare predicted shift vs OBSERVED pre->post
(UK_age_byEra: pre 2008-12, post 2015-19, vaccine-derived excluded). The efficacy that reproduces
the observed older-shift is the implied per-dose efficacy; if none does, the FOI/age mechanism
under-explains the UK impact.

Run on a VM:  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python uk_vaccine_predict_run.py
"""
import os, sys, json, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; CALIB = HERE.parents[1]; sys.path.insert(0, str(CALIB))
import calibrate_maled as cm                       # noqa: E402
from hm_calibrate import untransform               # noqa: E402
import hm_calibrate_uk as UK                        # noqa: E402
import process_surveillance_uk as PU               # noqa: E402

N_WORKERS = int(os.environ.get('HM_WORKERS', '90'))
N_AGENTS, CAP, COVERAGE = 25000, 60.0, 0.90
EFF = [0.5, 0.63, 0.75, 0.9]                        # per-dose seroconversion band
N_SEEDS = 12
POST_CSV = CALIB / 'experiments' / '28_hm_uk_infnum' / 'outputs' / 'posterior_hmreweight.csv'


def _run(args):
    sp, vaccine, seed = args
    sc = UK.build_sim_config('infnum', N_AGENTS, 'titer', cap_age_m=CAP)
    if vaccine:
        sc['vaccine'] = vaccine
    mo = cm._run_one_replicate((sc, sp, int(seed), (5.0, 10.0)))
    return dict(prop=mo['case_proportions'], total=mo['total_cases'])


def main():
    row = pd.read_csv(POST_CSV).iloc[0]            # ESS=1 reweight posterior -> single best-fit row
    sp = untransform(row, 'infnum', 'titer', fix_titer_shape=True)
    tasks, meta = [], []
    for s in range(N_SEEDS):
        tasks.append((sp, None, 50000 + s)); meta.append(('novax', None))      # counterfactual
        for e in EFF:
            tasks.append((sp, dict(response_prob=e, coverage=COVERAGE, dose_ages_y=[2/12, 4/12],
                                   moa='infection_blocking'), 50000 + s))
            meta.append(('vax', e))
    with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
        res = pool.map(_run, tasks)

    def agg(keep):
        P = np.array([r['prop'] for r, (k, e) in zip(res, meta) if keep(k, e) and r['total'] > 0])
        T = [r['total'] for r, (k, e) in zip(res, meta) if keep(k, e) and r['total'] > 0]
        return dict(prop_med=np.median(P, 0).tolist(), prop_lo=np.quantile(P, .1, 0).tolist(),
                    prop_hi=np.quantile(P, .9, 0).tolist(), total_med=float(np.median(T)), n=len(P))
    out = {'novax': agg(lambda k, e: k == 'novax')}
    for e in EFF:
        out[f'vax_{e}'] = agg(lambda k, ee, e=e: k == 'vax' and ee == e)
    out['observed_pre'] = PU.load_targets_uk_era('pre', CAP)['proportions'].tolist()
    out['observed_post'] = PU.load_targets_uk_era('post', CAP)['proportions'].tolist()
    out['bin_labels'] = PU.load_targets_uk_era('pre', CAP)['bin_labels']
    out['coverage'] = COVERAGE; out['eff'] = EFF
    (HERE / 'outputs').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'outputs' / 'uk_vaccine_predict.json', 'w'), indent=2)
    f12 = lambda p: round(sum(p[:2]), 3)
    print(f"<12mo fraction:  observed pre {f12(out['observed_pre'])} -> post {f12(out['observed_post'])}")
    print(f"  model no-vax {f12(out['novax']['prop_med'])} (sanity ~ pre); novax total {out['novax']['total_med']:.0f}")
    for e in EFF:
        v = out[f'vax_{e}']
        print(f"  model vax eff={e}: <12mo {f12(v['prop_med'])}  (reduction vs novax {1 - v['total_med']/out['novax']['total_med']:.2f})")
    print("wrote outputs/uk_vaccine_predict.json")


if __name__ == '__main__':
    main()
