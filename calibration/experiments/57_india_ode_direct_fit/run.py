"""Exp 57 -- India Vellore: fit the ODE reduction directly against the REAL composite
likelihood used in trajectory_select.py (Poisson symptomatic-IR-by-age + Binomial
repeat-detected fraction + survival log-L on age-at-first-DETECTED-infection).

Rationale (AK, 2026-08-17): extinction/persistence risk is not a concern for this
fit -- in the real world this disease does not go extinct, so the ODE's inability to
model stochastic extinction (exp53/54's finding) is not a limitation for this
purpose. What the ODE needs to add is a detection layer (cohort_model.py), which is
a straightforward multiplicative/probabilistic addition -- and once added, the ODE
can be fit directly via cheap deterministic optimization instead of waiting on
expensive, noisy ABM waves.

Pipeline: for a given parameter draw, (1) run the age-structured equilibrium model
(exp55/56's ode_model_age.py) to get the population's steady-state FOI, (2) run the
birth-cohort model (cohort_model.py) forward from age 0 using that FOI to get the
age-at-first-detection survival curve and the detected-count distribution at the
real cohort's empirical exit ages, (3) compute the exact composite log-likelihood
used in trajectory_select.py, (4) optimize over all 12 free parameters (age_binned +
titer maternal, matching hm_calibrate.py's bounds_for('age_binned','titer')) via
differential evolution (global, gradient-free -- appropriate given HM's own
experience of a possibly multimodal/degenerate likelihood surface).
"""
import sys, pathlib, time
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
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

# ---- real targets (exact same source as trajectory_select.py) ----
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

# ---- parameter bounds (mirrors hm_calibrate.bounds_for('age_binned','titer'), India beta ceiling) ----
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
        birth_rate_per_1000=16.0, death_rate_per_1000=7.0,
    ), {'<6m': d['p_symp_age_0_6'], '6-11m': d['p_symp_age_6_11'], '12plus': d['p_symp_age_12plus']}


DET_SYMP = 0.80 * 0.85
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85


def det_asymp(age_m):
    return (SHED_DAYS / (MONTHLY_D if age_m < 12 else QUARTERLY_D)) * EIA


def model_predict(x):
    """Returns dict with ir_symp_by_age (3 bins), repeat_frac, and the KM-comparable
    survival curve S(age_months) (P(no detected infection by that age))."""
    p, p_symp_age = params_from_vec(x)
    sol_age = simulate_age(p, n_agents=40_000, years=60, n_eval=400)
    Y = sol_age.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum(); IA_total = Y[:, IDX_IA].sum()
    foi_eq = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total

    # symptomatic-detected IR by age bin, straight from the equilibrium age model
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

    # survival curve on the monthly GRID (interpolate)
    S = np.interp(GRID, age_m, p0)
    S = np.clip(S, 0.0, 1.0)

    # repeat_frac: mixture over the real cohort's empirical censoring ages
    d1_at_cens = np.interp(CENS_AGES, age_m, p1)
    d2_at_cens = np.interp(CENS_AGES, age_m, p2)
    sum_d1plus = float((d1_at_cens + d2_at_cens).sum())
    sum_d2plus = float(d2_at_cens.sum())
    repeat_frac = sum_d2plus / sum_d1plus if sum_d1plus > 0 else 0.0

    return dict(ir_symp=ir_symp, repeat_frac=repeat_frac, S=S, foi_eq=foi_eq)


def composite_logL(x):
    try:
        pred = model_predict(x)
    except Exception:
        return -1e6
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
    x0 = [np.log(0.079), 0.457, 0.912, 0.832, np.log(10.86), 3.26, 38.7, 3.81, 0.816, 0.322, 0.548, 0.315]
    ll0 = composite_logL(x0)
    print(f"logL at exp47's MLE: {ll0:.2f}  (single eval took {time.time()-t0:.2f}s)")

    print("\nRunning differential_evolution over 12 parameters...")
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

    out = dict(best_params=best, logL=-result.fun, logL_at_exp47_mle=ll0,
               n_iterations=result.nit, n_evaluations=result.nfev,
               pred_ir_symp=pred_best['ir_symp'], pred_repeat_frac=pred_best['repeat_frac'])
    import json
    with open(OUT_DIR / 'fit_result.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    pd.DataFrame({'age_months': GRID, 'survival': pred_best['S']}).to_csv(OUT_DIR / 'survival_curve_best.csv', index=False)
    print(f"\nSaved outputs/fit_result.json, outputs/survival_curve_best.csv")
