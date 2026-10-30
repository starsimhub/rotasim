"""Exp 64, single-seed worker -- Bangladesh age_binned ODE direct-fit.
Identical method to exp57/58 (India): fits the ODE+cohort reduction
directly against the real composite likelihood via differential_evolution,
one seed per OS process (avoids exp58's open-file-descriptor bug from
repeated in-process Pool creation). Only the site changes: SITE='bangladesh',
Bangladesh demographics, Bangladesh's MAL-ED targets. Titer maternal shape
is freed (not fixed at Bangladesh's old ABM-pipeline values) per AK:
cross-site comparability matters more here than backwards ABM compatibility.
"""
import sys, pathlib, time, signal, json
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))
from ode_model import ODEParams, sigma_from_params
from ode_model_age import simulate_age, N_CONTACTS, IDX_S, IDX_IS, IDX_IA, N_STATE, N_AGE_BINS
from cohort_model import CohortParams, simulate_cohort, detected_count_distribution
import process_incidence_maled as P

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

SITE = 'bangladesh'
BIRTH_RATE, DEATH_RATE = 19, 6   # SITE_DEMOGRAPHICS['bangladesh'] in calibrate_maled.py
IR_BINS = ['<6 m', '6-11 m', '12-23 m']
GRID = np.arange(0, 37)
EPS = 1e-9

_t = P.load_targets(SITE)
_ir = _t['ir_by_age']
IR_DATA = {b: (int(_ir.loc[b, 'cases']), float(_ir.loc[b, 'PT'])) for b in IR_BINS}
_rf = _t['repeat_frac']
REPEAT_N = int(_rf['n']); REPEAT_OBS = int(round(_rf['frac'] * REPEAT_N))
_fi = pd.read_csv(CALIB / 'maled_data' / f'first_infection_{SITE}.csv')
FIRSTINF = _fi[['age_event_months', 'event_observed']].dropna().values
CENS_AGES = _fi.loc[(_fi['event_observed'] == 0) & (_fi['age_event_months'] > 0),
                     'age_event_months'].astype(float).tolist()
print(f"[{SITE}] Targets: IR={IR_DATA}, repeat={REPEAT_OBS}/{REPEAT_N}, n_firstinf={len(FIRSTINF)}, "
      f"n_censoring_ages={len(CENS_AGES)}")

PARAM_NAMES = ['log_base_beta', 'sus_after_1', 'sus_r2', 'sus_r3',
               'log_titer_median', 'titer_gsd', 'titer_half_life_days', 'hill_slope',
               'maternal_efficacy', 'p_symp_age_0_6', 'p_symp_age_6_11', 'p_symp_age_12plus']
BOUNDS = [(np.log(0.05), np.log(1.5)), (0.1, 1.0), (0.0, 1.0), (0.0, 1.0),
          (np.log(4.0), np.log(60.0)), (1.3, 3.5), (25.0, 70.0), (1.5, 8.0),
          (0.5, 0.99), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def params_from_vec(x):
    d = dict(zip(PARAM_NAMES, x))
    return ODEParams(
        base_beta=float(np.exp(d['log_base_beta'])), sus_after_1=d['sus_after_1'],
        sus_r2=d['sus_r2'], sus_r3=d['sus_r3'],
        maternal_titer_median=float(np.exp(d['log_titer_median'])), maternal_titer_gsd=d['titer_gsd'],
        maternal_titer_half_life_days=d['titer_half_life_days'], maternal_hill_slope=d['hill_slope'],
        maternal_immunity_efficacy=d['maternal_efficacy'],
        birth_rate_per_1000=BIRTH_RATE, death_rate_per_1000=DEATH_RATE,
    ), {'<6m': d['p_symp_age_0_6'], '6-11m': d['p_symp_age_6_11'], '12plus': d['p_symp_age_12plus']}


DET_SYMP = 0.80 * 0.85
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85


def model_predict(x):
    p, p_symp_age = params_from_vec(x)
    sol_age = simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    Y = sol_age.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum(); IA_total = Y[:, IDX_IA].sum()
    foi_eq = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total

    age_bin_labels = ['<6m', '6-11m', '12-23m', '24-35m']
    ir_symp = {}
    for a, label in enumerate(age_bin_labels):
        y = Y[a]; n_a = y.sum()
        s_weighted = float((p.sigma * y[IDX_S]).sum())
        incidence_per_day = foi_eq * s_weighted / n_a
        ps = p_symp_age['<6m'] if label == '<6m' else (p_symp_age['6-11m'] if label == '6-11m' else p_symp_age['12plus'])
        ir_symp[label] = incidence_per_day * (365.25 / 12.0) * 100 * ps * DET_SYMP

    cp = CohortParams(foi_eq=foi_eq, sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_age=p_symp_age)
    sol_c = simulate_cohort(cp, max_age_months=40, n_eval=600)
    age_m, p0, p1, p2 = detected_count_distribution(sol_c)

    S = np.interp(GRID, age_m, p0)
    S = np.clip(S, 0.0, 1.0)

    d1_at_cens = np.interp(CENS_AGES, age_m, p1)
    d2_at_cens = np.interp(CENS_AGES, age_m, p2)
    sum_d1plus = float((d1_at_cens + d2_at_cens).sum())
    sum_d2plus = float(d2_at_cens.sum())
    repeat_frac = sum_d2plus / sum_d1plus if sum_d1plus > 0 else 0.0

    return dict(ir_symp=ir_symp, repeat_frac=repeat_frac, S=S, foi_eq=foi_eq)


EVAL_TIMEOUT_S = 20.0


class _EvalTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EvalTimeout()


def composite_logL(x):
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EVAL_TIMEOUT_S)
    try:
        pred = model_predict(x)
    except Exception:
        return -1e6
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
    ll = 0.0
    for b in IR_BINS:
        label = {'<6 m': '<6m', '6-11 m': '6-11m', '12-23 m': '12-23m'}[b]
        cases, PT = IR_DATA[b]
        lam = pred['ir_symp'][label] / 100.0 * PT
        if lam <= 0:
            return -1e6
        ll += cases * np.log(lam) - lam
    p_rep = min(max(pred['repeat_frac'], 1e-6), 1 - 1e-6)
    ll += REPEAT_OBS * np.log(p_rep) + (REPEAT_N - REPEAT_OBS) * np.log(1 - p_rep)
    S = pred['S']
    for a, ev in FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], EPS))
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], EPS))
    return float(ll)


def neg_logL(x):
    return -composite_logL(x)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    RUNS_FILE = OUT_DIR / 'seed_runs.jsonl'
    POOL_FILE = OUT_DIR / 'pooled_candidates.jsonl'
    TOP_K_PER_RUN = 8

    print(f"\n=== seed {seed} ===")
    t0 = time.time()
    result = differential_evolution(neg_logL, BOUNDS, maxiter=60, popsize=15, tol=1e-6,
                                     seed=seed, workers=-1, updating='deferred', polish=True)
    dt = time.time() - t0
    best_logL = -result.fun
    best_params = dict(zip(PARAM_NAMES, result.x))
    print(f"seed {seed}: best logL={best_logL:.2f}  nit={result.nit}  nfev={result.nfev}  [{dt:.0f}s]")

    with RUNS_FILE.open('a') as f:
        f.write(json.dumps(dict(seed=seed, best_logL=best_logL, best_params=best_params,
                                 nit=int(result.nit), nfev=int(result.nfev), wall_s=dt)) + '\n')

    energies = np.asarray(result.population_energies)
    order = np.argsort(energies)[:TOP_K_PER_RUN]
    with POOL_FILE.open('a') as f:
        for rank, i in enumerate(order):
            params = dict(zip(PARAM_NAMES, result.population[i]))
            f.write(json.dumps(dict(seed=seed, rank=rank, logL=float(-energies[i]),
                                     params=params)) + '\n')
    print(f"  appended to {RUNS_FILE.name} and {POOL_FILE.name}")
