"""Exp 59 -- India Vellore: infnum variant of exp57/58's ODE direct-fit. Same
pipeline (age-structured equilibrium -> birth-cohort detection layer -> exact
trajectory_select.py composite likelihood -> differential_evolution), same
BOUNDS budget (12 params), same BDF solver, same per-eval timeout guard --
the ONLY change is p_symp keyed by infection ORDER (this cohort_model.py)
instead of by age bin (exp57's cohort_model.py). See README.md for why this
is a small, localized recode: ode_model_age.py (reused unchanged from exp55)
already tracks per-order state, so infnum only needed cohort_model.py's
detection lookup and this file's IR-by-age calculation re-keyed by order.
"""
import sys, pathlib, time, signal, json
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

SITE = 'india'
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
print(f"Targets: IR={IR_DATA}, repeat={REPEAT_OBS}/{REPEAT_N}, n_firstinf={len(FIRSTINF)}, "
      f"n_censoring_ages={len(CENS_AGES)}")

# ---- parameter bounds: identical budget to exp57 (12 params), only the last 3
# renamed order1/order2/order3plus instead of age_0_6/age_6_11/age_12plus ----
PARAM_NAMES = ['log_base_beta', 'sus_after_1', 'sus_r2', 'sus_r3',
               'log_titer_median', 'titer_gsd', 'titer_half_life_days', 'hill_slope',
               'maternal_efficacy', 'p_symp_order1', 'p_symp_order2', 'p_symp_order3plus']
BOUNDS = [(np.log(0.05), np.log(1.5)), (0.1, 1.0), (0.0, 1.0), (0.0, 1.0),
          (np.log(4.0), np.log(60.0)), (1.3, 3.5), (25.0, 70.0), (1.5, 8.0),
          (0.5, 0.99), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def params_from_vec(x):
    d = dict(zip(PARAM_NAMES, x))
    p_symp_order = [d['p_symp_order1'], d['p_symp_order2'], d['p_symp_order3plus']]
    return ODEParams(
        base_beta=float(np.exp(d['log_base_beta'])), sus_after_1=d['sus_after_1'],
        sus_r2=d['sus_r2'], sus_r3=d['sus_r3'],
        maternal_titer_median=float(np.exp(d['log_titer_median'])), maternal_titer_gsd=d['titer_gsd'],
        maternal_titer_half_life_days=d['titer_half_life_days'], maternal_hill_slope=d['hill_slope'],
        maternal_immunity_efficacy=d['maternal_efficacy'],
        birth_rate_per_1000=16.0, death_rate_per_1000=7.0,
    ), p_symp_order


DET_SYMP = 0.80 * 0.85
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85


def model_predict(x):
    """Returns dict with ir_symp_by_age (3 bins), repeat_frac, and the KM-comparable
    survival curve S(age_months) (P(no detected infection by that age)). p_symp
    is looked up by infection ORDER here, not age -- the equilibrium age model's
    per-age-bin state already breaks susceptibles out by order (IDX_S is an
    array over order), so the age-binned IR just needs summing each order's
    incidence contribution BEFORE multiplying by that order's p_symp, instead
    of summing incidence across orders and multiplying by one age-bin p_symp."""
    p, p_symp_order = params_from_vec(x)
    sol_age = simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    Y = sol_age.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum(); IA_total = Y[:, IDX_IA].sum()
    foi_eq = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total

    age_bin_labels = ['<6m', '6-11m', '12-23m', '24-35m']
    ir_symp = {}
    for a, label in enumerate(age_bin_labels):
        y = Y[a]; n_a = y.sum()
        s_by_order = y[IDX_S]  # array len 4, order 0..3("3+")
        ir = 0.0
        for j in range(4):
            incidence_j_per_day = foi_eq * p.sigma[j] * s_by_order[j] / n_a
            ps_j = p_symp_order[min(j, 2)]
            ir += incidence_j_per_day * (365.25 / 12.0) * 100 * ps_j * DET_SYMP
        ir_symp[label] = ir

    cp = CohortParams(foi_eq=foi_eq, sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_order=p_symp_order)
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
    raise _EvalTimeout(f"model_predict exceeded {EVAL_TIMEOUT_S}s")


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
    t0 = time.time()
    x0 = [np.log(0.15), 0.5, 0.3, 0.15, np.log(10.86), 3.26, 38.7, 3.81, 0.816, 0.6, 0.4, 0.2]
    ll0 = composite_logL(x0)
    print(f"logL at a placeholder infnum starting guess: {ll0:.2f}  (single eval took {time.time()-t0:.2f}s)")

    print("\nRunning differential_evolution over 12 parameters (infnum)...")
    t0 = time.time()
    result = differential_evolution(neg_logL, BOUNDS, maxiter=60, popsize=15, tol=1e-6,
                                     seed=20260817, workers=-1, updating='deferred', polish=True)
    print(f"Done in {time.time()-t0:.0f}s. Best logL={-result.fun:.2f}")
    best = dict(zip(PARAM_NAMES, result.x))
    best['base_beta'] = float(np.exp(best['log_base_beta']))
    best['maternal_titer_median'] = float(np.exp(best['log_titer_median']))
    print("Best-fit parameters:")
    for k, v in best.items():
        print(f"  {k:25s} {v:.4f}")

    pred_best = model_predict(result.x)
    print("\nModel prediction at best fit vs targets:")
    for b in IR_BINS:
        label = {'<6 m': '<6m', '6-11 m': '6-11m', '12-23 m': '12-23m'}[b]
        cases, PT = IR_DATA[b]
        print(f"  ir_symp_{b}: model={pred_best['ir_symp'][label]:.3f}  target={cases/PT*100:.3f}")
    print(f"  repeat_frac: model={pred_best['repeat_frac']:.3f}  target={_rf['frac']:.3f}")

    out = dict(best_params=best, logL=-result.fun, logL_at_placeholder=ll0,
               n_iterations=result.nit, n_evaluations=result.nfev,
               pred_ir_symp=pred_best['ir_symp'], pred_repeat_frac=pred_best['repeat_frac'])
    with open(OUT_DIR / 'fit_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    pd.DataFrame({'age_months': GRID, 'survival': pred_best['S']}).to_csv(OUT_DIR / 'survival_curve_best.csv', index=False)
    print(f"\nSaved outputs/fit_result.json, outputs/survival_curve_best.csv")
