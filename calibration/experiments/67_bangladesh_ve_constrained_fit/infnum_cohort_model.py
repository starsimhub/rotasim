"""infnum + vaccine variant, mirroring exp60/61's age_binned cohort_model.py
exactly (same 55-state layout, same cum_symp accumulator, same dose-based
order-jump vaccination) but with pdet keyed by infection ORDER (exp59's
p_symp_of_order) instead of age (exp60's p_symp_of_age_months). The vaccine
mechanism itself (order j -> j+1 transfer at dose ages) doesn't reference
the symptom model at all, so it's identical either way -- only the pdet
lookup inside rhs_cohort/the cum_symp accumulator differs.
"""
import numpy as np
from scipy.integrate import solve_ivp

N_ORDERS, N_PHASES, N_DET = 4, 4, 3
N_MAT_STAGES = 6
N_COHORT_STATE_BASE = N_MAT_STAGES + N_ORDERS * N_PHASES * N_DET
IDX_CUM_SYMP = N_COHORT_STATE_BASE
N_COHORT_STATE = N_COHORT_STATE_BASE + 1

IDX_MAT = slice(0, N_MAT_STAGES)


def _block(order, phase, det):
    return N_MAT_STAGES + order * (N_PHASES * N_DET) + phase * N_DET + det


DUR_SYMP_DAYS = 5.0
DUR_ASYMP_DAYS = 8.0
WANING_MEAN_DAYS = 91.0
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85
DET_SYMP = 0.80 * 0.85


def p_symp_of_order(j, p_symp_order):
    return p_symp_order[min(j, 2)]


def det_asymp_of_age_months(age_m):
    interval = MONTHLY_D if age_m < 12 else QUARTERLY_D
    return (SHED_DAYS / interval) * EIA


class CohortParams:
    def __init__(self, foi_eq, sigma, mat_mean_duration_days, p_symp_order):
        self.foi = foi_eq
        self.sigma = sigma
        self.rate_M = N_MAT_STAGES / mat_mean_duration_days
        self.gamma_s = 1.0 / DUR_SYMP_DAYS
        self.gamma_a = 1.0 / DUR_ASYMP_DAYS
        self.omega_R = 1.0 / WANING_MEAN_DAYS
        self.p_symp_order = p_symp_order


def rhs_cohort(t, y, p: CohortParams):
    age_m = t / 30.4375
    dydt = np.zeros(N_COHORT_STATE)

    M = y[IDX_MAT]
    dM = np.zeros(N_MAT_STAGES)
    dM[0] = -p.rate_M * M[0]
    for i in range(1, N_MAT_STAGES):
        dM[i] = p.rate_M * M[i - 1] - p.rate_M * M[i]
    dydt[IDX_MAT] = dM

    cum_symp_rate = 0.0
    for j in range(N_ORDERS):
        ps = p_symp_of_order(j, p.p_symp_order)
        pdet = ps * DET_SYMP + (1 - ps) * det_asymp_of_age_months(age_m)
        for d in range(N_DET):
            S = y[_block(j, 0, d)]
            IS = y[_block(j, 1, d)]
            IA = y[_block(j, 2, d)]
            R = y[_block(j, 3, d)]

            new_inf = p.foi * p.sigma[j] * S
            cum_symp_rate += new_inf * ps * DET_SYMP
            new_inf_undet = new_inf * (1 - pdet)
            new_inf_det = new_inf * pdet
            d_next_undet = d
            d_next_det = min(d + 1, N_DET - 1)

            dydt[_block(j, 0, d)] -= new_inf
            dydt[_block(j, 1, d_next_undet)] += new_inf_undet
            dydt[_block(j, 1, d_next_det)] += new_inf_det
            dydt[_block(j, 1, d)] -= p.gamma_s * IS
            dydt[_block(j, 2, d)] += p.gamma_s * IS - p.gamma_a * IA
            dydt[_block(j, 3, d)] += p.gamma_a * IA - p.omega_R * R

    for d in range(N_DET):
        R0 = y[_block(0, 3, d)]; R1 = y[_block(1, 3, d)]
        R2 = y[_block(2, 3, d)]; R3 = y[_block(3, 3, d)]
        dydt[_block(1, 0, d)] += p.omega_R * R0
        dydt[_block(2, 0, d)] += p.omega_R * R1
        dydt[_block(3, 0, d)] += p.omega_R * (R2 + R3)

    dydt[_block(0, 0, 0)] += p.rate_M * M[-1]
    dydt[IDX_CUM_SYMP] = cum_symp_rate

    return dydt


def _y0():
    y0 = np.zeros(N_COHORT_STATE)
    y0[0] = 1.0
    return y0


def _apply_vaccine_dose(y, coverage, take):
    frac = coverage * take
    y = y.copy()
    for j in (2, 1, 0):
        for phase in range(N_PHASES):
            for d in range(N_DET):
                idx_from = _block(j, phase, d)
                idx_to = _block(j + 1, phase, d)
                moved = frac * y[idx_from]
                y[idx_from] -= moved
                y[idx_to] += moved
    return y


def simulate_cohort(p: CohortParams, max_age_months=40.0, n_eval=800,
                     dose_ages_days=None, coverage=0.0, take=0.0):
    t_final = max_age_months * 30.4375
    if not dose_ages_days:
        y0 = _y0()
        t_eval = np.linspace(0, t_final, n_eval)
        sol = solve_ivp(rhs_cohort, (0, t_final), y0, args=(p,), method='BDF',
                         t_eval=t_eval, rtol=1e-9, atol=1e-10, max_step=1.0)
        return sol

    boundaries = sorted(dose_ages_days) + [t_final]
    y = _y0()
    t_prev = 0.0
    all_t, all_y = [], []
    for i, t_bound in enumerate(boundaries):
        n_this = max(int(n_eval * (t_bound - t_prev) / t_final), 5)
        t_eval = np.linspace(t_prev, t_bound, n_this)
        sol = solve_ivp(rhs_cohort, (t_prev, t_bound), y, args=(p,), method='BDF',
                         t_eval=t_eval, rtol=1e-9, atol=1e-10, max_step=1.0)
        start = 1 if all_t else 0
        all_t.append(sol.t[start:])
        all_y.append(sol.y[:, start:])
        y = sol.y[:, -1]
        if i < len(dose_ages_days):
            y = _apply_vaccine_dose(y, coverage, take)
        t_prev = t_bound

    class _Sol:
        pass
    out = _Sol()
    out.t = np.concatenate(all_t)
    out.y = np.concatenate(all_y, axis=1)
    return out


def cum_symp_curve(sol):
    return sol.t / 30.4375, sol.y[IDX_CUM_SYMP]


def detected_count_distribution(sol):
    """Returns (age_months array, P(d=0), P(d=1), P(d>=2)) at every eval point.
    Added for exp67 -- exp66 never needed this (it only reused exp65's
    already-fitted seeds), but exp67's joint natural-history refit does, via
    exp65.model_predict's call to it. Block-layout logic only, model-agnostic
    -- identical to age_binned_cohort_model.py's version."""
    age_m = sol.t / 30.4375
    Y = sol.y
    p0 = Y[IDX_MAT].sum(axis=0)
    for j in range(N_ORDERS):
        for ph in range(N_PHASES):
            p0 += Y[_block(j, ph, 0)]
    p1 = np.zeros_like(p0)
    for j in range(N_ORDERS):
        for ph in range(N_PHASES):
            p1 += Y[_block(j, ph, 1)]
    p2 = np.zeros_like(p0)
    for j in range(N_ORDERS):
        for ph in range(N_PHASES):
            p2 += Y[_block(j, ph, 2)]
    return age_m, p0, p1, p2
