"""Exp 67, infnum worker. Same joint natural-history + full-window direct-VE
fit as run_age_binned.py, but against exp65's infnum model instead of exp64's
age_binned model. See run_age_binned.py and README.md for the shared design
rationale (VE anchor, fixed take, monkeypatch technique).
"""
import sys, pathlib, time, signal, json, importlib.util
import numpy as np
from scipy.optimize import differential_evolution

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP65 = HERE.parents[0] / '65_bangladesh_infnum_ode'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("exp65_run_one_seed", str(EXP65 / "run_one_seed.py"))
exp65 = importlib.util.module_from_spec(_spec)
sys.modules["exp65_run_one_seed"] = exp65
_spec.loader.exec_module(exp65)

_spec2 = importlib.util.spec_from_file_location("exp67_infnum_cohort_model", str(HERE / "infnum_cohort_model.py"))
cohort_model = importlib.util.module_from_spec(_spec2)
sys.modules["exp67_infnum_cohort_model"] = cohort_model
_spec2.loader.exec_module(cohort_model)
exp65.CohortParams = cohort_model.CohortParams
exp65.simulate_cohort = cohort_model.simulate_cohort
exp65.detected_count_distribution = cohort_model.detected_count_distribution

CohortParams = cohort_model.CohortParams
simulate_cohort = cohort_model.simulate_cohort
cum_symp_curve = cohort_model.cum_symp_curve

DOSE_AGES_DAYS = [42.0, 70.0, 98.0]
TAKE = 0.74
VE_TARGET = 0.631
VE_SIGMA = 0.119
WINDOW_LO_M = 18 * 7 / 30.4375
WINDOW_HI_M = 24.0
MAX_AGE_MONTHS = 26.0


def rate_window_take(cp, dose):
    sol = simulate_cohort(cp, max_age_months=MAX_AGE_MONTHS, n_eval=600,
                           dose_ages_days=DOSE_AGES_DAYS if dose else None,
                           coverage=1.0 if dose else 0.0, take=TAKE if dose else 0.0)
    age_m, cum_symp = cum_symp_curve(sol)
    c_lo = np.interp(WINDOW_LO_M, age_m, cum_symp)
    c_hi = np.interp(WINDOW_HI_M, age_m, cum_symp)
    return c_hi - c_lo


def _ve_and_nh_logL(x):
    pred = exp65.model_predict(x)
    ll = 0.0
    for b in exp65.IR_BINS:
        label = {'<6 m': '<6m', '6-11 m': '6-11m', '12-23 m': '12-23m'}[b]
        cases, PT = exp65.IR_DATA[b]
        lam = pred['ir_symp'][label] / 100.0 * PT
        if lam <= 0:
            return -1e6, -1e6, float('nan')
        ll += cases * np.log(lam) - lam
    p_rep = min(max(pred['repeat_frac'], 1e-6), 1 - 1e-6)
    ll += exp65.REPEAT_OBS * np.log(p_rep) + (exp65.REPEAT_N - exp65.REPEAT_OBS) * np.log(1 - p_rep)
    S = pred['S']
    for a, ev in exp65.FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], exp65.EPS))
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], exp65.EPS))
    nh_ll = float(ll)

    p, p_symp_order = exp65.params_from_vec(x)
    cp = CohortParams(foi_eq=pred['foi_eq'], sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_order=p_symp_order)
    unvax = rate_window_take(cp, dose=False)
    vax = rate_window_take(cp, dose=True)
    ve_model = 1.0 - vax / unvax if unvax > 0 else float('nan')
    ve_ll = -0.5 * ((ve_model - VE_TARGET) / VE_SIGMA) ** 2 if np.isfinite(ve_model) else -1e6
    return nh_ll + ve_ll, nh_ll, ve_model


EVAL_TIMEOUT_S = 30.0


class _EvalTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EvalTimeout()


def composite_logL_with_ve(x):
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EVAL_TIMEOUT_S)
    try:
        joint_ll, nh_ll, ve_model = _ve_and_nh_logL(x)
    except Exception:
        joint_ll = -1e6
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
    return joint_ll


def neg_logL(x):
    return -composite_logL_with_ve(x)


OUT_DIR = HERE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    RUNS_FILE = OUT_DIR / 'seed_runs_infnum.jsonl'

    print(f"\n=== infnum seed {seed} ===")
    t0 = time.time()
    result = differential_evolution(neg_logL, exp65.BOUNDS, maxiter=60, popsize=15, tol=1e-6,
                                     seed=seed, workers=-1, updating='deferred', polish=True)
    dt = time.time() - t0
    joint_ll, nh_ll, ve_model = _ve_and_nh_logL(result.x)
    best_params = dict(zip(exp65.PARAM_NAMES, result.x))
    print(f"seed {seed}: joint logL={joint_ll:.2f}  nh_component={nh_ll:.2f}  ve_model={ve_model:.4f}  "
          f"[{dt:.0f}s]")

    with RUNS_FILE.open('a') as f:
        f.write(json.dumps(dict(seed=seed, joint_logL=joint_ll, nh_logL=nh_ll, ve_model=ve_model,
                                 best_params=best_params, nit=int(result.nit), nfev=int(result.nfev),
                                 wall_s=dt)) + '\n')
    print(f"  appended to {RUNS_FILE.name}")
