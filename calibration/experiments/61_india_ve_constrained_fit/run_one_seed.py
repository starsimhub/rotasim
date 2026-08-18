"""Exp 61, single-seed worker. Joint natural-history + direct-VE fit: adds one
Gaussian VE term to exp57/58's exact 12-parameter composite likelihood,
computed from the SAME model_predict call (reuses its foi_eq -- no duplicate
simulate_age), via this dir's own cohort_model.py (a copy of exp60's, with
the cum_symp accumulator + dose-based vaccination). See README.md for the
target/sigma/take rationale.

Loads exp57's run.py by explicit file path + sys.modules registration (same
pattern as exp58/59/60) so its `from cohort_model import ...` resolves to
THIS directory's vaccine-extended cohort_model.py, not exp57's plain one --
letting exp57.model_predict/composite_logL machinery work completely
unchanged, just with a different cohort_model.py transparently swapped in.
"""
import sys, pathlib, time, signal, json, importlib.util
import numpy as np
from scipy.optimize import differential_evolution

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP57 = HERE.parents[0] / '57_india_ode_direct_fit'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(HERE))  # exp61's own cohort_model.py must win the name resolution

_spec = importlib.util.spec_from_file_location("exp57_run", str(EXP57 / "run.py"))
exp57 = importlib.util.module_from_spec(_spec)
sys.modules["exp57_run"] = exp57
_spec.loader.exec_module(exp57)

# exp57's run.py unconditionally inserts ITS OWN directory into sys.path right
# before its `from cohort_model import ...` line, so no amount of sys.path
# priority juggling from here can make that import resolve to this dir's
# cohort_model.py instead -- exp57's own insert always wins. Load this dir's
# cohort_model.py under a distinct module name (avoiding the 'cohort_model'
# cache collision) and monkeypatch it into exp57's already-loaded namespace;
# exp57.model_predict looks up CohortParams/simulate_cohort as globals at
# CALL time, so rebinding them here after the fact works cleanly.
_spec2 = importlib.util.spec_from_file_location("exp61_cohort_model", str(HERE / "cohort_model.py"))
cohort_model = importlib.util.module_from_spec(_spec2)
sys.modules["exp61_cohort_model"] = cohort_model
_spec2.loader.exec_module(cohort_model)
exp57.CohortParams = cohort_model.CohortParams
exp57.simulate_cohort = cohort_model.simulate_cohort
exp57.detected_count_distribution = cohort_model.detected_count_distribution

CohortParams = cohort_model.CohortParams
simulate_cohort = cohort_model.simulate_cohort
cum_symp_curve = cohort_model.cum_symp_curve

DOSE_AGES_DAYS = [42.0, 70.0, 98.0]  # Rotavac: 6, 10, 14 weeks
TAKE = 0.74                          # fixed, not a free parameter -- see README.md
VE_TARGET = 0.59                     # Nair et al. 6-11m test-negative VE point estimate
VE_SIGMA = 0.054                     # from Nair's 95% CI (47-68%): (0.68-0.47)/(2*1.96)


def symp_rate_6_11m(cp: CohortParams, dose: bool):
    sol = simulate_cohort(cp, max_age_months=14, n_eval=400,
                           dose_ages_days=DOSE_AGES_DAYS if dose else None,
                           coverage=1.0 if dose else 0.0, take=TAKE if dose else 0.0)
    age_m, cum_symp = cum_symp_curve(sol)
    c6 = np.interp(6.0, age_m, cum_symp)
    c12 = np.interp(12.0, age_m, cum_symp)
    return c12 - c6


def _ve_and_nh_logL(x):
    """One model_predict call, reused for both the natural-history logL
    (exact copy of exp57.composite_logL's body, minus its own timeout wrapper
    -- this function is wrapped once at the top level instead) and the VE
    term (reuses foi_eq from the same call, no duplicate simulate_age)."""
    pred = exp57.model_predict(x)
    ll = 0.0
    for b in exp57.IR_BINS:
        label = {'<6 m': '<6m', '6-11 m': '6-11m', '12-23 m': '12-23m'}[b]
        cases, PT = exp57.IR_DATA[b]
        lam = pred['ir_symp'][label] / 100.0 * PT
        if lam <= 0:
            return -1e6, -1e6, float('nan')
        ll += cases * np.log(lam) - lam
    p_rep = min(max(pred['repeat_frac'], 1e-6), 1 - 1e-6)
    ll += exp57.REPEAT_OBS * np.log(p_rep) + (exp57.REPEAT_N - exp57.REPEAT_OBS) * np.log(1 - p_rep)
    S = pred['S']
    for a, ev in exp57.FIRSTINF:
        m = int(min(np.floor(a), 35))
        if ev == 1:
            ll += np.log(max(S[m] - S[m + 1], exp57.EPS))
        else:
            c = int(min(np.ceil(a), 36))
            ll += np.log(max(S[c], exp57.EPS))
    nh_ll = float(ll)

    p, p_symp_age = exp57.params_from_vec(x)
    cp = CohortParams(foi_eq=pred['foi_eq'], sigma=p.sigma, mat_mean_duration_days=p.mat_mean_duration,
                       p_symp_age=p_symp_age)
    unvax = symp_rate_6_11m(cp, dose=False)
    vax = symp_rate_6_11m(cp, dose=True)
    ve_model = 1.0 - vax / unvax if unvax > 0 else float('nan')
    ve_ll = -0.5 * ((ve_model - VE_TARGET) / VE_SIGMA) ** 2 if np.isfinite(ve_model) else -1e6
    return nh_ll + ve_ll, nh_ll, ve_model


EVAL_TIMEOUT_S = 30.0  # a bit more headroom than exp57's 20s: 2 extra cohort sims per eval now


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
    RUNS_FILE = OUT_DIR / 'seed_runs.jsonl'

    print(f"\n=== seed {seed} ===")
    t0 = time.time()
    result = differential_evolution(neg_logL, exp57.BOUNDS, maxiter=60, popsize=15, tol=1e-6,
                                     seed=seed, workers=-1, updating='deferred', polish=True)
    dt = time.time() - t0
    joint_ll, nh_ll, ve_model = _ve_and_nh_logL(result.x)
    best_params = dict(zip(exp57.PARAM_NAMES, result.x))
    print(f"seed {seed}: joint logL={joint_ll:.2f}  nh_component={nh_ll:.2f}  ve_model={ve_model:.4f}  "
          f"[{dt:.0f}s]")

    with RUNS_FILE.open('a') as f:
        f.write(json.dumps(dict(seed=seed, joint_logL=joint_ll, nh_logL=nh_ll, ve_model=ve_model,
                                 best_params=best_params, nit=int(result.nit), nfev=int(result.nfev),
                                 wall_s=dt)) + '\n')
    print(f"  appended to {RUNS_FILE.name}")
