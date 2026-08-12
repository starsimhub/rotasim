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

SITE = os.environ.get('MALED_SITE', 'bangladesh')   # 'bangladesh' (default) or 'india' (Vellore)
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '100'))
IR_BINS = ['<6 m', '6-11 m', '12-23 m']            # 24-35m dropped (1 case)
# India/Vellore: ALSO fit the ALL-infection IR (same MAL-ED population + monthly detection) to pin
# FOI -- the symptomatic-only fit was under-determined (low IR <-> few infections OR many mild ones)
# and mis-chose low FOI. Off for Bangladesh (its existing fit is unchanged).
USE_IR_ALL = (SITE == 'india')
# India + NeonatalPriming active (NEO_PRIME=1): exp44 fits order_effect (fraction of primed
# children whose next real infection is order-credited) -- only meaningful under order-sensitive
# symptom models (infnum, age_and_infection); age_binned ignores order entirely (see exp39-43).
NEO_PRIME_ACTIVE = (SITE == 'india' and os.environ.get('NEO_PRIME', '0') == '1')
# Extinction penalty: log(sum of symptomatic IRs) returned as finite log(1e-9) for extinct sims,
# so the emulator learns the extinction zone rather than treating it as unknown/plausible.
# Activate via EXT_PENALTY=1 env var. Target log(2)=0.69: viable sims have ir_sum~3 (z≈0.5),
# extinct sims have log(1e-9)≈-20.7 (z≈-28) — decisively ruled out.
USE_EXT_PENALTY = os.environ.get('EXT_PENALTY', '0') == '1'
EXT_PENALTY_OBS = {'log_symp_ir_sum': (float(np.log(2.0)), 0.75)}
OBS_COLS = [f'ir_symp_{b}' for b in IR_BINS] + ['repeat_detected_frac', 'first_inf_median']
if USE_IR_ALL:
    OBS_COLS += [f'ir_all_{b}' for b in IR_BINS]
if USE_EXT_PENALTY:
    OBS_COLS = ['log_symp_ir_sum'] + OBS_COLS  # first in cycle → selected wave 1, cuts extinction zone immediately
# Age-at-first-DETECTION feature: KM median for Bangladesh, but India/Vellore detects only ~27%
# of children (low incidence) so KM survival never reaches 0.5 -> median undefined. Use the KM
# Q25 (=15.1mo, defined) for India. The OBS_COLS name stays 'first_inf_median' (cosmetic scalar
# feature); both target and model output use FIRST_INF_QUANT, so the emulator is consistent.
FIRST_INF_QUANT = 'q25' if SITE == 'india' else 'median'
_QI = {'q25': 0, 'median': 1, 'q75': 2}[FIRST_INF_QUANT]

import numpy as _np
LOG = _np.log
_BETA_MAX = 1.5 if SITE == 'india' else 0.5   # India: all-infection IR needs a higher-FOI ceiling
TRANSMISSION_BOUNDS = {
    'log_base_beta':        (LOG(0.05), LOG(_BETA_MAX)),
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
    'age':        {'beta0': (-5.0, 2.0), 'beta1': (-1.0, 1.0), 'beta2': (-0.5, 0.5)},
    'infnum':     {'p_symp_1': (0.20, 1.0), 'p_r2': (0.0, 1.0), 'p_r3': (0.0, 1.0)},
    # Non-parametric age-symptom: free P(symptomatic) per age bin (<6, 6-11, >=12 mo).
    # Tests whether the quadratic ('age') over-constrains the age-symptom curve.
    'age_binned': {'p_symp_age_0_6': (0.0, 1.0), 'p_symp_age_6_11': (0.0, 1.0), 'p_symp_age_12plus': (0.0, 1.0)},
    # Combined: quadratic age logit + a linear infection-order slope beta3*min(n,5). Nests 'age'
    # (beta3=0). beta3<0 = symptoms decline with infection experience (the infnum-like effect).
    'age_and_infection': {'beta0': (-5.0, 2.0), 'beta1': (-1.0, 1.0), 'beta2': (-0.5, 0.5), 'beta3': (-2.0, 0.5)},
}
# Fixed infnum symptom parameters from Vellore biweekly cohort (near-complete detection).
# P(symp|infected) by age bin: <6m=0.381, 6-11m=0.407, 12-23m=0.189, 24-35m=0.122.
# Mapped to infnum order params: p_symp_1 = 6-11m fraction (mostly 1st infections);
# p_r2 = 12-23m / 6-11m; p_r3 = 24-35m / 12-23m.
FIXED_PSYMP = dict(p_symp_1=0.407, p_r2=round(0.189 / 0.407, 4), p_r3=round(0.122 / 0.189, 4))
# Fixed age-binned symptom parameters from Vellore biweekly cohort (Lewnard et al., near-complete detection).
# P(symp|infected) per age bin: <6m=0.381, 6-11m=0.407, 12-23m=0.189 (24-35m=0.122 folded into 12+).
# Used with --fix-age-psymp + --model age_binned; frees only FOI + immunity (5 params).
FIXED_AGE_PSYMP = dict(p_symp_age_0_6=0.381, p_symp_age_6_11=0.407, p_symp_age_12plus=0.189)
# Fixed titer-shape values (infnum posterior medians) -- the identified maternal curve, used
# to remove titer's redundant shape flexibility (--fix-titer-shape): only maternal_efficacy stays free.
FIXED_TITER_SHAPE = dict(median=20.0, gsd=2.3, half_life_days=50.0, hill=4.7)
# exp44: fraction of neonatally-primed children whose next real infection is order-credited.
# Only added to order-sensitive models (infnum, age_and_infection) when NEO_PRIME is active --
# a no-op parameter under age_binned, so not offered there (avoids wasting a search dimension).
NEONATAL_BOUNDS = {'neonatal_order_effect': (0.0, 1.0)}

def bounds_for(model, maternal, fix_titer_shape=False, fix_psymp=False, fix_age_psymp=False):
    mat = dict(MATERNAL_BOUNDS[maternal])
    if fix_titer_shape and maternal == 'titer':
        for k in ('log_titer_median', 'titer_gsd', 'titer_half_life_days', 'hill_slope'):
            mat.pop(k, None)   # fix the shape; keep maternal_efficacy free
    symp = dict(SYMPTOM_BOUNDS[model])
    if fix_psymp and model == 'infnum':
        for k in ('p_symp_1', 'p_r2', 'p_r3'):
            symp.pop(k, None)  # fix at FIXED_PSYMP; only FOI + immunity free
    if fix_age_psymp and model == 'age_binned':
        for k in ('p_symp_age_0_6', 'p_symp_age_6_11', 'p_symp_age_12plus'):
            symp.pop(k, None)  # fix at FIXED_AGE_PSYMP; only FOI + immunity free
    neo = dict(NEONATAL_BOUNDS) if (NEO_PRIME_ACTIVE and model in ('infnum', 'age_and_infection')) else {}
    return {**TRANSMISSION_BOUNDS, **mat, **symp, **neo}
BOUNDS = {m: bounds_for(m, 'titer') for m in ('age', 'infnum', 'age_binned', 'age_and_infection')}   # back-compat default (exp 16/17 = titer)
SYMPTOM_MODEL = {'age': 'age_only', 'infnum': 'infection_number', 'age_binned': 'age_binned',
                 'age_and_infection': 'age_and_infection'}


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


def untransform(row, model, maternal='titer', fix_titer_shape=False, fix_psymp=False, fix_age_psymp=False):
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
    if 'neonatal_order_effect' in row.index:   # exp44: only present when bounds_for added it
        p['neonatal_order_effect'] = float(row['neonatal_order_effect'])
    if model == 'age':
        p.update(beta0=float(row['beta0']), beta1=float(row['beta1']), beta2=float(row['beta2']))
    elif model == 'age_and_infection':
        p.update(beta0=float(row['beta0']), beta1=float(row['beta1']),
                 beta2=float(row['beta2']), beta3=float(row['beta3']))
    elif model == 'age_binned':
        if fix_age_psymp:
            p.update(**FIXED_AGE_PSYMP)
        else:
            p.update(p_symp_age_0_6=float(row['p_symp_age_0_6']), p_symp_age_6_11=float(row['p_symp_age_6_11']),
                     p_symp_age_12plus=float(row['p_symp_age_12plus']))
    else:
        p1 = FIXED_PSYMP['p_symp_1'] if fix_psymp else float(row['p_symp_1'])
        r2 = FIXED_PSYMP['p_r2']    if fix_psymp else float(row['p_r2'])
        r3 = FIXED_PSYMP['p_r3']    if fix_psymp else float(row['p_r3'])
        p.update(p_symp_1=p1, p_symp_2=p1 * r2, p_symp_3plus=p1 * r2 * r3)
    return p


def build_sim_config(model, n_agents, maternal='titer'):
    demo = cm.SITE_DEMOGRAPHICS[SITE]
    fi = pd.read_csv(THISDIR / 'maled_data' / f'first_infection_{SITE}.csv')
    cens = fi.loc[(fi['event_observed'] == 0) & (fi['age_event_months'] > 0), 'age_event_months'].astype(float).tolist()
    cfg = dict(
        n_agents=n_agents, start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=cm.FIXED_CONSTANT_SEVERITY, reporting_rate=cm.FIXED_REPORTING_RATE,
        age_data_path=str(THISDIR / f'{SITE}_age_data.csv'),
        symptom_model=SYMPTOM_MODEL[model], maternal_n_stages=6, maternal_model=maternal,
        observation='cohort', censoring_ages=cens,
        symp_collection=0.80, eia_sensitivity=0.85, shed_days=13.0,
    )
    # India/Vellore: persistent COMMUNITY neonatal strain (G10P[11], asymptomatic, ~50% of neonates,
    # immunizing). Bangladesh's documented neonatal infections were NOSOCOMIAL (not the community
    # cohort) -> p_neo~0 there; UK p_neo=0. p_neo FIXED from literature (not fitted).
    # Neonatal priming OFF by default for the clean FOI test (all-infection IR alone). Enable via
    # NEO_PRIME=1 to add the literature-set <6m correction once FOI is pinned.
    if SITE == 'india' and os.environ.get('NEO_PRIME', '0') == '1':
        cfg['neonatal_priming'] = dict(p_neo=0.5, age_weeks=2.0, sus_effect=0.0)  # decoupled: symptom-order only
    return cfg


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
        m = P.km_quartiles(s['age_event_months'].values, s['event_observed'].values)[_QI]
        if np.isfinite(m):
            meds.append(m)
    se = float(np.std(meds)) if meds else 1.0
    obs['first_inf_median'] = (km[FIRST_INF_QUANT], float(np.hypot(se, MODEL_SD_T)))
    if USE_IR_ALL:
        tall = P.load_ir_all_targets(SITE)
        for b in IR_BINS:
            ir = float(tall.loc[b, 'IR']); cases = max(int(tall.loc[b, 'cases']), 1)
            obs[f'ir_all_{b}'] = (ir, float(np.hypot(ir / np.sqrt(cases), MODEL_SD_IR)))
    if USE_EXT_PENALTY:
        obs.update(EXT_PENALTY_OBS)
    return obs


def make_simulator(model, sim_config, maternal='titer', fix_titer_shape=False, fix_psymp=False, fix_age_psymp=False):
    def simulate(params_df: pd.DataFrame) -> pd.DataFrame:
        args = []
        for _, row in params_df.iterrows():
            sp = untransform(row, model, maternal, fix_titer_shape, fix_psymp, fix_age_psymp)
            seed = abs(hash(tuple(np.round(row.values, 6)))) % (2**31)
            args.append((sim_config, sp, int(seed), CAL_WINDOW))
        with get_context('spawn').Pool(processes=min(N_WORKERS, len(args)), maxtasksperchild=4) as pool:
            outs = pool.map(cm._run_one_replicate, args)
        _EXT_LOG = float(np.log(1e-9))  # sentinel for extinct: ≈-20.7, far below target log(2)=0.69
        rows = []
        for mo in outs:
            try:
                ir = mo['ir_by_age']
                ir_sum = float(ir['IR'].sum())
                log_ir_sum = float(np.log(max(ir_sum, 1e-9)))
                fim = mo['first_infection'][FIRST_INF_QUANT]
                if (not np.isfinite(fim)) or ir_sum <= 0:
                    row_out = {c: np.nan for c in OBS_COLS}
                    if USE_EXT_PENALTY:
                        row_out['log_symp_ir_sum'] = _EXT_LOG  # finite, not NaN — extinction signal
                    rows.append(row_out); continue
                row = ({f'ir_symp_{b}': float(ir.loc[b, 'IR']) for b in IR_BINS}
                       | {'repeat_detected_frac': mo.get('repeat_frac'), 'first_inf_median': float(fim)})
                if USE_IR_ALL:
                    ira = mo['ir_all_by_age']
                    row |= {f'ir_all_{b}': float(ira.loc[b, 'IR']) for b in IR_BINS}
                if USE_EXT_PENALTY:
                    row['log_symp_ir_sum'] = log_ir_sum
                rows.append(row)
            except Exception:
                row_out = {c: np.nan for c in OBS_COLS}
                if USE_EXT_PENALTY:
                    row_out['log_symp_ir_sum'] = _EXT_LOG
                rows.append(row_out)
        return pd.DataFrame(rows, index=params_df.index)
    return simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum', 'age_binned', 'age_and_infection'])
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
    ap.add_argument('--fix-psymp', action='store_true',
                    help='fix infnum symptom params (p_symp_1, p_r2, p_r3) at Vellore biweekly values '
                         '(0.407 / 0.465 / 0.645); only valid with --model infnum')
    ap.add_argument('--fix-age-psymp', action='store_true',
                    help='fix age_binned p_symp per-bin at Vellore biweekly values '
                         '(<6m=0.381, 6-11m=0.407, 12+m=0.189); only valid with --model age_binned')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--early-stop', action='store_true',
                    help='abort extinct draws early (StopWhenExtinct): ~3.5x faster on the ~80%% that burn '
                         'out, so more samples/wave for the same wall time; surviving draws are unchanged')
    ap.add_argument('--early-stop-burn-in', type=float, default=2.0,
                    help='years to let the epidemic establish before arming early-stop (default 2.0)')
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
    sim_config['early_stop_extinct'] = a.early_stop
    sim_config['early_stop_burn_in_years'] = a.early_stop_burn_in
    run_name = (f'maled_{a.model}_{a.maternal}'
                + ('_fixedshape' if a.fix_titer_shape else '')
                + ('_fixedpsymp' if a.fix_psymp else '')
                + ('_fixedagepsymp' if a.fix_age_psymp else ''))
    engine = hm.HistoryMatching(
        function=make_simulator(a.model, sim_config, a.maternal, a.fix_titer_shape, a.fix_psymp, a.fix_age_psymp),
        bounds=bounds_for(a.model, a.maternal, a.fix_titer_shape, a.fix_psymp, a.fix_age_psymp), observations=obs,
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
