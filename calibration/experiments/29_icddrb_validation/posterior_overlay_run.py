"""Posterior-predictive overlay (VM step): push the MAL-ED posteriors (exp25 age_binned,
exp27 infnum) through the Surveillance observer with Bangladesh demographics + icddr,b bins
-- NO refit -- and save the predicted case-age distribution + incidence-by-age (median + 5-95%
band across posterior draws) to outputs/posterior_overlay.json. Plot locally afterward.

Run on a VM:  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python posterior_overlay_run.py
"""
import os, sys, json, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]; sys.path.insert(0, str(CALIB))
import calibrate_maled as cm                                    # noqa: E402
from hm_calibrate import untransform, SYMPTOM_MODEL             # noqa: E402

N_WORKERS = int(os.environ.get('HM_WORKERS', '90'))
EDGES = [0.0, 6.0, 12.0, 24.0, 60.0]                            # icddr,b bins <6/6-11/12-23/24-59
POST = {'age_binned': CALIB / 'experiments' / '25_age_binned_titer_fixedshape' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv',
        'infnum':     CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'}


def _sim_config(model):
    demo = cm.SITE_DEMOGRAPHICS['bangladesh']
    return dict(n_agents=25000, start='2003-01-01', stop='2013-01-01', n_contacts=7,
                birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
                constant_severity=cm.FIXED_CONSTANT_SEVERITY, reporting_rate=cm.FIXED_REPORTING_RATE,
                age_data_path=str(CALIB / 'bangladesh_age_data.csv'), symptom_model=SYMPTOM_MODEL[model],
                maternal_n_stages=6, maternal_model='titer', observation='surveillance',
                bin_edges_m=EDGES, cap_age_m=60.0)


def _run(args):
    model, row, seed = args
    try:
        sp = untransform(row, model, 'titer', fix_titer_shape=True)
        mo = cm._run_one_replicate((_sim_config(model), sp, int(seed), (5.0, 10.0)))
        return dict(prop=mo['case_proportions'], ir=mo['ir_per_100cy'], total=mo['total_cases'])
    except Exception as e:
        return dict(error=repr(e)[:150], total=0)


def main():
    out = {}
    for model in ['infnum', 'age_binned']:
        post = pd.read_csv(POST[model]).drop_duplicates().reset_index(drop=True)
        tasks = [(model, row, 40000 + i) for i, (_, row) in enumerate(post.iterrows())]
        with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
            res = pool.map(_run, tasks)
        res = [r for r in res if r.get('total', 0) > 0]
        props = np.array([r['prop'] for r in res]); irs = np.array([r['ir'] for r in res])
        q = lambda a, p: np.quantile(a, p, axis=0).tolist()
        out[model] = dict(n=len(res),
                          prop_med=np.median(props, 0).tolist(), prop_lo=q(props, .05), prop_hi=q(props, .95),
                          ir_med=np.median(irs, 0).tolist(), ir_lo=q(irs, .05), ir_hi=q(irs, .95))
        print(f"{model}: n={len(res)}  prop_med={[round(x,3) for x in out[model]['prop_med']]}")
    (HERE / 'outputs').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'outputs' / 'posterior_overlay.json', 'w'), indent=2)
    print("wrote outputs/posterior_overlay.json")


if __name__ == '__main__':
    main()
