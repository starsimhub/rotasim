"""
History-matching driver for the MAL-ED two-model VE work (exp 16 = age+titer,
exp 17 = infnum+titer). Reuses calibrate_maled._run_one_replicate (cohort observation,
homogeneous mixing, titer maternal) as the HM simulator, and the historymatching v2.0.1
HistoryMatching engine (D. Klein's exp-07 pattern). One script, parameterized by --model.

Targets (cohort, Bangladesh): symptomatic IR in 3 reliable age bins (<6, 6-11, 12-23;
24-35mo dropped -- 1 case), repeat-detected fraction 0.403, age-at-first-DETECTION KM
median. SD = sqrt(observational^2 + small model^2): Poisson on IR counts, binomial on the
fraction, bootstrap on the KM median.

Run in the rota-hm env with the protobuf flag, in tmux (covaguest is spot):
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python conda run -n rota-hm \
      python hm_calibrate.py --model age   --max-iter 1 --n-samples 1500   # validate wave 1
  ... --model age --max-iter 6 --n-samples 1500 --resume
  ... --model infnum --max-iter 6 --n-samples 1500
"""
import os, sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm
import process_incidence_maled as P

SITE = 'bangladesh'
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '100'))
IR_BINS = ['<6 m', '6-11 m', '12-23 m']            # 24-35m dropped (1 case)
OBS_COLS = [f'ir_symp_{b}' for b in IR_BINS] + ['repeat_detected_frac', 'first_inf_median']

import numpy as _np
LOG = _np.log
TRANSMISSION_BOUNDS = {
    'log_base_beta':        (LOG(0.05), LOG(0.5)),
    'sus_after_1':          (0.1, 1.0),
    'sus_r2':               (0.0, 1.0),    # sus_2 = sus_1 * r2
    'sus_r3':               (0.0, 1.0),    # sus_3 = sus_2 * r3
}
MATERNAL_BOUNDS = {
    'titer':  {'log_titer_median': (LOG(4.0), LOG(60.0)), 'titer_gsd': (1.3, 3.5),
               'titer_half_life_days': (25.0, 70.0), 'hill_slope': (1.5, 8.0), 'maternal_efficacy': (0.5, 0.99)},
    'erlang': {'maternal_efficacy': (0.5, 0.99), 'maternal_mean_duration_days': (30.0, 300.0)},  # n_stages fixed at 6
}
SYMPTOM_BOUNDS = {
    'age':    {'beta0': (-5.0, 2.0), 'beta1': (-1.0, 1.0), 'beta2': (-0.5, 0.5)},
    'infnum': {'p_symp_1': (0.4, 1.0), 'p_r2': (0.0, 1.0), 'p_r3': (0.0, 1.0)},
}
# Fixed titer-shape values (infnum posterior medians) -- the identified maternal curve, used
# to remove titer's redundant shape flexibility (--fix-titer-shape): only maternal_efficacy stays free.
FIXED_TITER_SHAPE = dict(median=20.0, gsd=2.3, half_life_days=50.0, hill=4.7)

def bounds_for(model, maternal, fix_titer_shape=False):
    mat = dict(MATERNAL_BOUNDS[maternal])
    if fix_titer_shape and maternal == 'titer':
        for k in ('log_titer_median', 'titer_gsd', 'titer_half_life_days', 'hill_slope'):
            mat.pop(k, None)   # fix the shape; keep maternal_efficacy free
    return {**TRANSMISSION_BOUNDS, **mat, **SYMPTOM_BOUNDS[model]}
BOUNDS = {m: bounds_for(m, 'titer') for m in ('age', 'infnum')}   # back-compat default (exp 16/17 = titer)
SYMPTOM_MODEL = {'age': 'age_only', 'infnum': 'infection_number'}


class CycleFeatureSelection(hm.FeatureSelectionStrategy):
    """One feature per wave, cycling through a fixed list -> every target constrains the NROY
    over the waves (unlike auto, which kept picking IR and skipped repeat/first-inf), while
    keeping emulator training well-conditioned (1 feature/wave, vs all-at-once which can go
    singular when a feature is degenerate)."""
    def __init__(self, feats):
        self.feats = list(feats)
    def select_features(self, simulation_results, observations, iteration=1):
        f = self.feats[(int(iteration) - 1) % len(self.feats)]
        return self.validate_features([f], simulation_results, observations)
    def get_strategy_name(self):
        return f"Cycle 1/wave over {len(self.feats)} targets"


def untransform(row, model, maternal='titer', fix_titer_shape=False):
    s1 = float(row['sus_after_1']); s2 = s1 * float(row['sus_r2']); s3 = s2 * float(row['sus_r3'])
    p = dict(base_beta=float(np.exp(row['log_base_beta'])),
             sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
             maternal_immunity_efficacy=float(row['maternal_efficacy']))
    if maternal == 'titer':   # presence of maternal_titer_median triggers the titer branch in _run_one_replicate
        if fix_titer_shape:   # shape held at the identified curve; only efficacy fitted
            p.update(maternal_titer_median=FIXED_TITER_SHAPE['median'], maternal_titer_gsd=FIXED_TITER_SHAPE['gsd'],
                     maternal_titer_half_life_days=FIXED_TITER_SHAPE['half_life_days'], maternal_hill_slope=FIXED_TITER_SHAPE['hill'])
        else:
            p.update(maternal_titer_median=float(np.exp(row['log_titer_median'])),
                     maternal_titer_gsd=float(row['titer_gsd']),
                     maternal_titer_half_life_days=float(row['titer_half_life_days']),
                     maternal_hill_slope=float(row['hill_slope']))
    else:                     # erlang (n_stages fixed in sim_config)
        p.update(maternal_immunity_mean_duration_days=float(row['maternal_mean_duration_days']))
    if model == 'age':
        p.update(beta0=float(row['beta0']), beta1=float(row['beta1']), beta2=float(row['beta2']))
    else:
        p1 = float(row['p_symp_1']); p2 = p1 * float(row['p_r2']); p3 = p2 * float(row['p_r3'])
        p.update(p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)
    return p


def build_sim_config(model, n_agents, maternal='titer'):
    demo = cm.SITE_DEMOGRAPHICS[SITE]
    fi = pd.read_csv(THISDIR / 'maled_data' / f'first_infection_{SITE}.csv')
    cens = fi.loc[(fi['event_observed'] == 0) & (fi['age_event_months'] > 0), 'age_event_months'].astype(float).tolist()
    return dict(
        n_agents=n_agents, start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=cm.FIXED_CONSTANT_SEVERITY, reporting_rate=cm.FIXED_REPORTING_RATE,
        age_data_path=str(THISDIR / f'{SITE}_age_data.csv'),
        symptom_model=SYMPTOM_MODEL[model], maternal_n_stages=6, maternal_model=maternal,
        observation='cohort', censoring_ages=cens,
        symp_collection=0.80, eia_sensitivity=0.85, shed_days=13.0,
    )


def make_observations():
    """(mean, std) per target. Poisson SD on IR counts, binomial SE on the fraction,
    bootstrap SE on the KM first-detection median; small model SD added in quadrature."""
    t = P.load_targets(SITE)
    tir = t['ir_by_age']; rf = t['repeat_frac']; km = t['first_infection_km']
    MODEL_SD_IR, MODEL_SD_FRAC, MODEL_SD_T = 0.15, 0.02, 0.5   # small model noise floors
    obs = {}
    for b in IR_BINS:
        ir = float(tir.loc[b, 'IR']); cases = max(int(tir.loc[b, 'cases']), 1)
        obs[f'ir_symp_{b}'] = (ir, float(np.hypot(ir / np.sqrt(cases), MODEL_SD_IR)))
    obs['repeat_detected_frac'] = (rf['frac'], float(np.hypot(rf['se'], MODEL_SD_FRAC)))
    # bootstrap SE of the KM median
    df = pd.read_csv(THISDIR / 'maled_data' / f'first_infection_{SITE}.csv')
    df = df[df['age_event_months'] > 0]
    rng = np.random.default_rng(0); meds = []
    for _ in range(300):
        s = df.sample(len(df), replace=True)
        _, m, _ = P.km_quartiles(s['age_event_months'].values, s['event_observed'].values)
        if np.isfinite(m):
            meds.append(m)
    se = float(np.std(meds)) if meds else 1.0
    obs['first_inf_median'] = (km['median'], float(np.hypot(se, MODEL_SD_T)))
    return obs


def make_simulator(model, sim_config, maternal='titer', fix_titer_shape=False):
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
                ir = mo['ir_by_age']; fim = mo['first_infection']['median']
                if (not np.isfinite(fim)) or float(ir['IR'].sum()) <= 0:
                    rows.append({c: np.nan for c in OBS_COLS}); continue
                rows.append({f'ir_symp_{b}': float(ir.loc[b, 'IR']) for b in IR_BINS}
                            | {'repeat_detected_frac': mo.get('repeat_frac'), 'first_inf_median': float(fim)})
            except Exception:
                rows.append({c: np.nan for c in OBS_COLS})
        return pd.DataFrame(rows, index=params_df.index)
    return simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum'])
    ap.add_argument('--maternal', default='titer', choices=['titer', 'erlang'])
    ap.add_argument('--n-samples', type=int, default=1500)
    ap.add_argument('--max-iter', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--features', default=None,
                    help='comma-separated feature names to force every wave via '
                         'ManualFeatureSelection (overrides auto). Used to make targets the '
                         'auto-selector skipped (e.g. repeat_detected_frac,first_inf_median) bite '
                         'on a --resume continuation.')
    ap.add_argument('--all-targets', action='store_true',
                    help='force ManualFeatureSelection over ALL 5 targets every wave (IR bins + '
                         'repeat_detected_frac + first_inf_median), so the NROY is constrained by every '
                         'target -- not just incidence (the auto selector skipped repeat/first-inf in exp 16/17).')
    ap.add_argument('--fix-titer-shape', action='store_true',
                    help='hold the titer SHAPE params (median/gsd/half_life/hill) at the identified curve '
                         '(FIXED_TITER_SHAPE); fit only maternal_efficacy + transmission + symptom. Tests whether '
                         'removing titer\'s redundant flexibility makes age+titer identifiable.')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    n_agents = N_AGENTS
    if a.smoke:
        a.n_samples = 16; a.max_iter = 1; n_agents = 8000
    EXP_FOLDER = {('age', 'titer'): '16_hm_age_titer', ('infnum', 'titer'): '17_hm_infnum_titer',
                  ('age', 'erlang'): '21_hm_age_erlang', ('infnum', 'erlang'): '22_hm_infnum_erlang'}
    out_dir = a.out_dir or str(THISDIR / 'experiments' / EXP_FOLDER[(a.model, a.maternal)] / 'outputs' / 'hm')

    obs = make_observations()
    if a.all_targets:
        feature_selection = CycleFeatureSelection(OBS_COLS)
        fs_desc = f"CYCLE 1/wave over all {len(OBS_COLS)} targets: {OBS_COLS}"
    elif a.features:
        feats = [s.strip() for s in a.features.split(',') if s.strip()]
        feature_selection = hm.ManualFeatureSelection(feats)
        fs_desc = f"MANUAL {feats}"
    else:
        feature_selection = hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2)
        fs_desc = "AUTO mean_sq_z (1/wave, cooldown 2)"
    print(f"HM model={a.model}  maternal={a.maternal}  n_samples={a.n_samples}  max_iter={a.max_iter}  workers={N_WORKERS}")
    print(f"Feature selection: {fs_desc}")
    print("Targets:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in obs.items()})
    sim_config = build_sim_config(a.model, n_agents, a.maternal)
    run_name = f'maled_{a.model}_{a.maternal}' + ('_fixedshape' if a.fix_titer_shape else '')
    engine = hm.HistoryMatching(
        function=make_simulator(a.model, sim_config, a.maternal, a.fix_titer_shape),
        bounds=bounds_for(a.model, a.maternal, a.fix_titer_shape), observations=obs,
        emulator_type='bayes_linear', sampling_strategy='lhs',
        feature_selection=feature_selection,
        n_samples=a.n_samples, implausibility_threshold=3.0, max_iterations=a.max_iter,
        output_dir=out_dir, run_name=run_name, random_seed=20260610,
    )
    t0 = sc.tic()
    engine.run(resume=a.resume)
    print(f'\nHM done in {sc.toc(t0, output=True):.0f}s')
    try:
        print(engine.get_status_summary())
    except Exception as e:
        print('status summary unavailable:', e)


if __name__ == '__main__':
    main()
