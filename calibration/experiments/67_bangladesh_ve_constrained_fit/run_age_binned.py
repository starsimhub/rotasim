"""Exp 67, age_binned worker. Joint natural-history + full-window (18wk-2y)
direct-VE fit for Bangladesh, using PROVIDE's traditional per-protocol
severe-RVD estimate as the VE anchor. Mirrors exp61's monkeypatch technique
(loading exp64's run_one_seed.py by file path + rebinding CohortParams/
simulate_cohort/detected_count_distribution to this dir's vaccine-extended
age_binned_cohort_model.py), generalizing exp61's 6-11m-specific VE window
to PROVIDE's much wider 18wk-2y postvaccination window. See README.md.
"""
import sys, pathlib, time, signal, json, importlib.util
import numpy as np
from scipy.optimize import differential_evolution

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP64 = HERE.parents[0] / '64_bangladesh_age_binned_ode'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))  # this dir's cohort_model-named import must win

_spec = importlib.util.spec_from_file_location("exp64_run_one_seed", str(EXP64 / "run_one_seed.py"))
exp64 = importlib.util.module_from_spec(_spec)
sys.modules["exp64_run_one_seed"] = exp64
_spec.loader.exec_module(exp64)

# exp64's run_one_seed.py unconditionally inserts its own directory into
# sys.path right before `from cohort_model import ...`, so that import always
# resolves to exp64's own plain cohort_model.py regardless of prior sys.path
# ordering (same gotcha as exp57/exp61). Load this dir's vaccine-extended
# cohort model under a distinct module name and monkeypatch it into exp64's
# already-loaded namespace; exp64.model_predict looks up these names as
# globals at CALL time, so rebinding after the fact works.
_spec2 = importlib.util.spec_from_file_location("exp67_age_binned_cohort_model", str(HERE / "age_binned_cohort_model.py"))
cohort_model = importlib.util.module_from_spec(_spec2)
sys.modules["exp67_age_binned_cohort_model"] = cohort_model
_spec2.loader.exec_module(cohort_model)
exp64.CohortParams = cohort_model.CohortParams
exp64.simulate_cohort = cohort_model.simulate_cohort
exp64.detected_count_distribution = cohort_model.detected_count_distribution

CohortParams = cohort_model.CohortParams
simulate_cohort = cohort_model.simulate_cohort
cum_symp_curve = cohort_model.cum_symp_curve

DOSE_AGES_DAYS = [42.0, 70.0, 98.0]   # assumed future Rotavac-like schedule: 6, 10, 14 weeks (kept as-is per AK)
TAKE = 0.74                           # fixed -- reused from India's exp61, no independent Bangladesh estimate
VE_TARGET = 0.631                     # PROVIDE traditional (uncorrected) per-protocol severe RVD VE, postvax window
VE_SIGMA = 0.119                      # from PROVIDE's 95% CI (33.0-79.7%): (0.797-0.330)/(2*1.96)
WINDOW_LO_M = 18 * 7 / 30.4375         # 18 weeks in months (~4.14)
WINDOW_HI_M = 24.0                    # 2 years
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
    """One model_predict call, reused for both the natural-history logL
    (exact copy of exp64.composite_logL's body, minus its own timeout wrapper
    -- this function is wrapped once at the top level instead) and the VE
    term (reuses foi_eq from the same call, no duplicate simulate_age)."""
    pred = exp64.model_predict(x)
    ll = 0.0
    for b in exp64.IR_BINS:
        label = {'<6 m': '<6m', '6-11 m': '6-11m', '12-23 m': '12-23m'}[b]
        cases, PT = exp64.IR_DATA[b]
        lam = pred['ir_symp'][label] / 100.0 * PT
        if lam <= 0:
            return -1e6, -1e6, float('nan')
        ll += cases * np.log(lam) - lam
    p_rep = min(max(pred['repeat_frac'], 1e-6), 1 - 1e-6)
    ll += exp64.REPEAT_OBS * np.log(p_rep) + (exp64.REPEAT_N - exp64.REPEAT_OBS) * np.log(1 - p_rep)
    S = pred['S']
    for a, ev in exp64.FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], exp64.EPS))
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], exp64.EPS))
    nh_ll = float(ll)

    p, p_symp_age = exp64.params_from_vec(x)
    cp = CohortParams(foi_eq=pred['foi_eq'], sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_age=p_symp_age)
    unvax = rate_window_take(cp, dose=False)
    vax = rate_window_take(cp, dose=True)
    ve_model = 1.0 - vax / unvax if unvax > 0 else float('nan')
    ve_ll = -0.5 * ((ve_model - VE_TARGET) / VE_SIGMA) ** 2 if np.isfinite(ve_model) else -1e6
    return nh_ll + ve_ll, nh_ll, ve_model


EVAL_TIMEOUT_S = 30.0  # extra headroom over exp64's 20s: 2 extra cohort sims per eval now


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
    RUNS_FILE = OUT_DIR / 'seed_runs_age_binned.jsonl'

    print(f"\n=== age_binned seed {seed} ===")
    t0 = time.time()
    result = differential_evolution(neg_logL, exp64.BOUNDS, maxiter=60, popsize=15, tol=1e-6,
                                     seed=seed, workers=-1, updating='deferred', polish=True)
    dt = time.time() - t0
    joint_ll, nh_ll, ve_model = _ve_and_nh_logL(result.x)
    best_params = dict(zip(exp64.PARAM_NAMES, result.x))
    print(f"seed {seed}: joint logL={joint_ll:.2f}  nh_component={nh_ll:.2f}  ve_model={ve_model:.4f}  "
          f"[{dt:.0f}s]")

    with RUNS_FILE.open('a') as f:
        f.write(json.dumps(dict(seed=seed, joint_logL=joint_ll, nh_logL=nh_ll, ve_model=ve_model,
                                 best_params=best_params, nit=int(result.nit), nfev=int(result.nfev),
                                 wall_s=dt)) + '\n')
    print(f"  appended to {RUNS_FILE.name}")
