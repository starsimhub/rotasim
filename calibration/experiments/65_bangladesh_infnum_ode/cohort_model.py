"""infnum variant of exp57's cohort_model.py -- IDENTICAL state layout and
transmission/recovery mechanics (order x phase x detected-count, 54 states).
The only change: the detection probability at a new infection is looked up
by infection ORDER (j, the number of PRIOR infections, so this new infection
is order j+1) instead of by age. The asymptomatic-detection rate itself
(det_asymp_of_age_months, driven by MAL-ED's monthly-then-quarterly stool
collection cadence) stays age-dependent regardless of symptom model -- that's
a surveillance-logistics fact, not a symptom-model choice -- so p_det is now
a function of BOTH order (for p_symp) and age (for det_asymp).

p_symp_order: list/array of length 3 = [order1 (first infection), order2,
order3plus] -- order index j=2 and j=3 ("3+" in the state's own order
categorization) share the order3plus value, matching age_binned's 3-parameter
budget for a fair comparison (see exp59/README.md).
"""
import numpy as np
from scipy.integrate import solve_ivp

N_ORDERS, N_PHASES, N_DET = 4, 4, 3
N_MAT_STAGES = 6
N_COHORT_STATE = N_MAT_STAGES + N_ORDERS * N_PHASES * N_DET   # 6 + 48 = 54

IDX_MAT = slice(0, N_MAT_STAGES)


def _block(order, phase, det):
    return N_MAT_STAGES + order * (N_PHASES * N_DET) + phase * N_DET + det


DUR_SYMP_DAYS = 5.0
DUR_ASYMP_DAYS = 8.0
WANING_MEAN_DAYS = 91.0
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85
DET_SYMP = 0.80 * 0.85


def p_symp_of_order(j, p_symp_order):
    """j = number of PRIOR infections (0..3, "3"="3+"); this new infection is
    order j+1. p_symp_order has 3 entries [order1, order2, order3plus]; j=2
    and j=3 both map to order3plus."""
    return p_symp_order[min(j, 2)]


def det_asymp_of_age_months(age_m):
    interval = MONTHLY_D if age_m < 12 else QUARTERLY_D
    return (SHED_DAYS / interval) * EIA


def p_det_of_order_age(j, age_m, p_symp_order):
    ps = p_symp_of_order(j, p_symp_order)
    return ps * DET_SYMP + (1 - ps) * det_asymp_of_age_months(age_m)


class CohortParams:
    def __init__(self, foi_eq, sigma, mat_mean_duration_days, p_symp_order):
        self.foi = foi_eq
        self.sigma = sigma  # array len 4: [1, sus_after_1, sus_after_2, sus_after_3plus]
        self.rate_M = N_MAT_STAGES / mat_mean_duration_days
        self.gamma_s = 1.0 / DUR_SYMP_DAYS
        self.gamma_a = 1.0 / DUR_ASYMP_DAYS
        self.omega_R = 1.0 / WANING_MEAN_DAYS
        self.p_symp_order = p_symp_order  # [order1, order2, order3plus]


def rhs_cohort(t, y, p: CohortParams):
    age_m = t / 30.4375
    dydt = np.zeros(N_COHORT_STATE)

    M = y[IDX_MAT]
    dM = np.zeros(N_MAT_STAGES)
    dM[0] = -p.rate_M * M[0]
    for i in range(1, N_MAT_STAGES):
        dM[i] = p.rate_M * M[i - 1] - p.rate_M * M[i]
    dydt[IDX_MAT] = dM

    # S=0, IS=1, IA=2, R=3
    for j in range(N_ORDERS):
        pdet = p_det_of_order_age(j, age_m, p.p_symp_order)  # order-keyed, not age-keyed
        for d in range(N_DET):
            S = y[_block(j, 0, d)]
            IS = y[_block(j, 1, d)]
            IA = y[_block(j, 2, d)]
            R = y[_block(j, 3, d)]

            new_inf = p.foi * p.sigma[j] * S
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

    return dydt


def simulate_cohort(p: CohortParams, max_age_months=40.0, n_eval=800):
    y0 = np.zeros(N_COHORT_STATE)
    y0[0] = 1.0
    t_span = (0, max_age_months * 30.4375)
    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(rhs_cohort, t_span, y0, args=(p,), method='BDF',
                     t_eval=t_eval, rtol=1e-9, atol=1e-10, max_step=1.0)
    return sol


def detected_count_distribution(sol):
    """Returns (age_months array, P(d=0), P(d=1), P(d>=2)) at every eval point."""
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
