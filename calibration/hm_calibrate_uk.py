"""History-matching driver for the UK pre-vaccine SURVEILLANCE anchor (low-FOI setting).

Companion to hm_calibrate.py (Bangladesh birth-cohort). Same ABM, same symptom models
(age_binned / infnum), same FIXED maternal-titer shape -- but a SURVEILLANCE observation:
the age-DISTRIBUTION of symptomatic cases (a shape; no cohort KM/first-detection/repeat).

Targets: the proportion of (genotyped) symptomatic cases in 5 age bins
[<6, 6-11, 12-23, 24-35, 36+ mo], pooled over UK 2008-2012 (process_surveillance_uk).
Each bin proportion is one HM feature with a multinomial SE + a small model-noise floor.

The cross-setting design: hold the symptom STRUCTURE and maternal shape fixed across
settings and re-fit FOI (transmission) + symptom params + maternal efficacy to the UK
case age-distribution. UK is older (peak 12-23 mo) than Bangladesh (~8 mo first infection)
-> a lower-FOI fit. Downstream this anchors the FOI-gradient / achieved-VE comparison.

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python hm_calibrate_uk.py --model age_binned --fix-titer-shape
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python hm_calibrate_uk.py --model infnum     --fix-titer-shape
"""
import os, sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np
import pandas as pd
import sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm                       # noqa: E402
import process_surveillance_uk as PU               # noqa: E402
# Reuse the shared transform / bounds / symptom-model machinery from the cohort driver.
from hm_calibrate import (untransform, bounds_for, SYMPTOM_MODEL,   # noqa: E402
                          CycleFeatureSelection)

SITE = 'uk'
# Shape target (case proportions) needs ~1k cases, not 40k agents; 8k sims run ~13-16s vs
# ~60-90s at 40k. Lower default keeps the local run fast and avoids pool stalls on the slow
# high-FOI tail. Override with --n-agents.
N_AGENTS = 8_000
# Cap base_beta below the cohort default (0.5): high-FOI all-age sims are both slow AND
# irrelevant to the low-FOI UK setting. Override with --beta-max.
DEFAULT_BETA_MAX = 0.35
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '100'))


def uk_bounds(model, maternal='titer', fix_titer_shape=False, beta_max=DEFAULT_BETA_MAX):
    """Cohort bounds with the base_beta upper tightened to the low-FOI UK regime."""
    import numpy as _np
    b = dict(bounds_for(model, maternal, fix_titer_shape))
    lo, _ = b['log_base_beta']
    b['log_base_beta'] = (lo, float(_np.log(beta_max)))
    return b
# Default: cap the observation at <5y (60 mo). Care-seeking is high/uniform through age 5, so
# this stays out of the confounded adult regime while including the 2-5y bins that diagnose
# whether infection-number symptoms overshoot older-child reinfections. (36 -> <3y, 24 -> <2y.)
DEFAULT_CAP_M = 60.0
MODEL_SD_PROP = 0.02   # small model-noise floor added in quadrature to the multinomial SE


def build_sim_config(model, n_agents, maternal='titer', cap_age_m=DEFAULT_CAP_M):
    demo = cm.SITE_DEMOGRAPHICS[SITE]
    t = PU.load_targets_uk(cap_age_m=cap_age_m)
    return dict(
        n_agents=n_agents, start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=cm.FIXED_CONSTANT_SEVERITY, reporting_rate=cm.FIXED_REPORTING_RATE,
        age_data_path=str(THISDIR / f'{SITE}_age_data.csv'),
        symptom_model=SYMPTOM_MODEL[model], maternal_n_stages=6, maternal_model=maternal,
        observation='surveillance', bin_edges_m=list(t['bin_edges_m']), cap_age_m=cap_age_m,
    )


def obs_cols(cap_age_m=DEFAULT_CAP_M):
    return list(PU.load_targets_uk(cap_age_m=cap_age_m)['feature_keys'])


def make_observations(cap_age_m=DEFAULT_CAP_M):
    """(mean, std) per kept bin proportion: multinomial SE sqrt(p(1-p)/N) + model-noise floor."""
    t = PU.load_targets_uk(cap_age_m=cap_age_m)
    obs = {}
    for key, p, se in zip(t['feature_keys'], t['proportions'], t['se']):
        obs[key] = (float(p), float(np.hypot(se, MODEL_SD_PROP)))
    return obs


def make_simulator(model, sim_config, maternal='titer', fix_titer_shape=False):
    OBS_COLS = obs_cols(sim_config.get('cap_age_m', DEFAULT_CAP_M))
    def simulate(params_df: pd.DataFrame) -> pd.DataFrame:
        args = []
        for _, row in params_df.iterrows():
            sp = untransform(row, model, maternal, fix_titer_shape)
            seed = abs(hash(tuple(np.round(row.values, 6)))) % (2**31)
            args.append((sim_config, sp, int(seed), CAL_WINDOW))
        with get_context('spawn').Pool(processes=min(N_WORKERS, len(args)), maxtasksperchild=4) as pool:
            outs = pool.map(cm._run_one_replicate, args)
        rows = []
        for mo in outs:
            try:
                if mo.get('total_cases', 0) <= 0:        # extinct / no cases -> shape undefined
                    rows.append({c: np.nan for c in OBS_COLS}); continue
                prop = mo['case_proportions']
                rows.append({k: float(prop[i]) for i, k in enumerate(OBS_COLS)})
            except Exception:
                rows.append({c: np.nan for c in OBS_COLS})
        return pd.DataFrame(rows, index=params_df.index)
    return simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned'])
    ap.add_argument('--maternal', default='titer', choices=['titer', 'erlang'])
    ap.add_argument('--n-samples', type=int, default=1500)
    ap.add_argument('--max-iter', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--features', default=None,
                    help='comma-separated feature names to force every wave via ManualFeatureSelection')
    ap.add_argument('--all-targets', action='store_true',
                    help='CYCLE 1/wave over all kept bin-proportion features (every bin constrains the NROY)')
    ap.add_argument('--n-agents', type=int, default=N_AGENTS, help='population size (default 8000; shape target')
    ap.add_argument('--beta-max', type=float, default=DEFAULT_BETA_MAX,
                    help='upper bound on base_beta (default 0.35; UK is low-FOI, high-beta sims are slow+irrelevant)')
    ap.add_argument('--cap-age-months', type=float, default=DEFAULT_CAP_M,
                    help='upper age cap (mo) for observed cases: 60 -> <5y/6 bins (default), '
                         '36 -> <3y/4 bins (cohort window), 24 -> <2y/3 bins (Bangladesh IR fit bins)')
    ap.add_argument('--fix-titer-shape', action='store_true',
                    help='hold titer SHAPE at the identified curve (FIXED_TITER_SHAPE); fit only maternal '
                         'efficacy + transmission + symptom. Maternal shape is setting-independent biology.')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--early-stop', action='store_true',
                    help='abort extinct draws early (StopWhenExtinct) for more samples/wave at fixed wall time')
    ap.add_argument('--early-stop-burn-in', type=float, default=2.0)
    a = ap.parse_args()
    n_agents = a.n_agents
    if a.smoke:
        a.n_samples = 16; a.max_iter = 1; n_agents = 8000
    cap = a.cap_age_months
    EXP_FOLDER = {'age_binned': '28_hm_uk_age_binned', 'infnum': '28_hm_uk_infnum', 'age': '28_hm_uk_age'}
    out_dir = a.out_dir or str(THISDIR / 'experiments' / EXP_FOLDER[a.model] / 'outputs' / 'hm')

    obs = make_observations(cap)
    OBS_COLS = obs_cols(cap)
    if a.all_targets:
        feature_selection = CycleFeatureSelection(OBS_COLS)
        fs_desc = f"CYCLE 1/wave over all {len(OBS_COLS)} bin-proportion features"
    elif a.features:
        feats = [s.strip() for s in a.features.split(',') if s.strip()]
        feature_selection = hm.ManualFeatureSelection(feats); fs_desc = f"MANUAL {feats}"
    else:
        feature_selection = hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2)
        fs_desc = "AUTO mean_sq_z (1/wave, cooldown 2)"
    print(f"HM-UK model={a.model}  maternal={a.maternal}  cap={cap}mo  n_samples={a.n_samples}  max_iter={a.max_iter}  workers={N_WORKERS}")
    print(f"Feature selection: {fs_desc}")
    print(f"n_agents={n_agents}  beta_max={a.beta_max}")
    print("UK surveillance targets (bin proportions):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in obs.items()})
    sim_config = build_sim_config(a.model, n_agents, a.maternal, cap_age_m=cap)
    sim_config['early_stop_extinct'] = a.early_stop
    sim_config['early_stop_burn_in_years'] = a.early_stop_burn_in
    run_name = f'uk_{a.model}_{a.maternal}' + ('_fixedshape' if a.fix_titer_shape else '')
    engine = hm.HistoryMatching(
        function=make_simulator(a.model, sim_config, a.maternal, a.fix_titer_shape),
        bounds=uk_bounds(a.model, a.maternal, a.fix_titer_shape, a.beta_max), observations=obs,
        emulator_type='bayes_linear', sampling_strategy='lhs',
        feature_selection=feature_selection,
        n_samples=a.n_samples, implausibility_threshold=3.0, max_iterations=a.max_iter,
        output_dir=out_dir, run_name=run_name, random_seed=20260618,
    )
    t0 = sc.tic()
    engine.run(resume=a.resume)
    print(f'\nHM-UK done in {sc.toc(t0, output=True):.0f}s')
    try:
        print(engine.get_status_summary())
    except Exception as e:
        print('status summary unavailable:', e)


if __name__ == '__main__':
    main()
