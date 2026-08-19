"""Birth-cohort extension of the age-structured ODE (exp55/56), adding a detection
layer so the ODE can be fit directly against the REAL composite likelihood used in
trajectory_select.py (Poisson IR-by-age [symptomatic] + Binomial repeat-detected
fraction + survival log-L on age-at-first-DETECTED-infection).

Key simplification (exact, not approximate): the general population is vastly
larger than any one birth cohort, so a single cohort's own infection status has
negligible feedback on the population-wide force of infection. At the population's
stationary equilibrium (exp55 confirmed the age-structured model converges to a
fixed-proportion equilibrium), the FOI experienced by ANY susceptible individual is
a single constant, foi_eq (well-mixed, non-age-assortative -- confirmed via the
ABM's plain ss.RandomNet). So: (1) compute foi_eq ONCE from exp55/56's
age-structured equilibrium model, (2) use it as a fixed external forcing on a
SEPARATE cohort-following ODE that tracks one newborn cohort's joint
(true infection order, transmission phase, DETECTED-infection count) distribution
from age 0 forward -- decoupled from, not co-simulated with, the population model.

Detection layer (mirrors rotasim/analyzers.py's MALEDCohort exactly, confirmed
read-only / does not feed back into transmission): at the moment of EVERY new
infection (order j -> j+1), independently of the transmission IS/IA phase
mechanic, the infection is a. symptomatic w.p. p_symp(age) [detected w.p.
symp_collection*eia_sensitivity, age-independent] or b. asymptomatic w.p.
1-p_symp(age) [detected w.p. p_surv(age)*eia_sensitivity, p_surv = shed_days /
(30.4375d if age<12mo else 91.3125d) -- MALEDCohort._p_surv]. So each new
infection is detected w.p. p_det(age) = p_symp(age)*0.68 + (1-p_symp(age))*p_asymp_det(age),
independent of order.

State layout: order j in {0,1,2,3} ("3"="3+"), phase in {S,IS,IA,R}, detected
count d in {0,1,2} ("2"="2+", matching repeat_frac's ">=2" definition) -- 4x4x3=48
states, plus the maternal chain M_0..M_5 (order=0, detected=0, phase=undefined) -- 6
states. Total 54.
"""
import numpy as np
from scipy.integrate import solve_ivp

N_ORDERS, N_PHASES, N_DET = 4, 4, 3
N_MAT_STAGES = 6
N_COHORT_STATE = N_MAT_STAGES + N_ORDERS * N_PHASES * N_DET   # 6 + 48 = 54

# phase index within each (order, det) block: 0=S, 1=IS, 2=IA, 3=R
IDX_MAT = slice(0, N_MAT_STAGES)


def _block(order, phase, det):
    return N_MAT_STAGES + order * (N_PHASES * N_DET) + phase * N_DET + det


DUR_SYMP_DAYS = 5.0
DUR_ASYMP_DAYS = 8.0
WANING_MEAN_DAYS = 91.0
MONTHLY_D, QUARTERLY_D, SHED_DAYS, EIA = 30.4375, 91.3125, 13.0, 0.85
DET_SYMP = 0.80 * 0.85


def p_symp_of_age_months(age_m, p_symp_age):
    """p_symp_age: dict with keys '<6m','6-11m','12plus' (age_binned model)."""
    if age_m < 6:
        return p_symp_age['<6m']
    elif age_m < 12:
        return p_symp_age['6-11m']
    else:
        return p_symp_age['12plus']


def det_asymp_of_age_months(age_m):
    interval = MONTHLY_D if age_m < 12 else QUARTERLY_D
    return (SHED_DAYS / interval) * EIA


def p_det_of_age_months(age_m, p_symp_age):
    ps = p_symp_of_age_months(age_m, p_symp_age)
    return ps * DET_SYMP + (1 - ps) * det_asymp_of_age_months(age_m)


class CohortParams:
    def __init__(self, foi_eq, sigma, mat_mean_duration_days, p_symp_age):
        self.foi = foi_eq
        self.sigma = sigma  # array len 4: [1, sus_after_1, sus_after_2, sus_after_3plus]
        self.rate_M = N_MAT_STAGES / mat_mean_duration_days
        self.gamma_s = 1.0 / DUR_SYMP_DAYS
        self.gamma_a = 1.0 / DUR_ASYMP_DAYS
        self.omega_R = 1.0 / WANING_MEAN_DAYS
        self.p_symp_age = p_symp_age


def rhs_cohort(t, y, p: CohortParams):
    age_m = t / 30.4375
    pdet = p_det_of_age_months(age_m, p.p_symp_age)
    dydt = np.zeros(N_COHORT_STATE)

    M = y[IDX_MAT]
    dM = np.zeros(N_MAT_STAGES)
    dM[0] = -p.rate_M * M[0]
    for i in range(1, N_MAT_STAGES):
        dM[i] = p.rate_M * M[i - 1] - p.rate_M * M[i]
    dydt[IDX_MAT] = dM

    # S=0, IS=1, IA=2, R=3
    for j in range(N_ORDERS):
        for d in range(N_DET):
            S = y[_block(j, 0, d)]
            IS = y[_block(j, 1, d)]
            IA = y[_block(j, 2, d)]
            R = y[_block(j, 3, d)]

            new_inf = p.foi * p.sigma[j] * S
            new_inf_undet = new_inf * (1 - pdet)
            new_inf_det = new_inf * pdet
            d_next_undet = d          # stays at same detected-count
            d_next_det = min(d + 1, N_DET - 1)

            dydt[_block(j, 0, d)] -= new_inf           # S outflow (inflow handled below)
            dydt[_block(j, 1, d_next_undet)] += new_inf_undet
            dydt[_block(j, 1, d_next_det)] += new_inf_det
            dydt[_block(j, 1, d)] -= p.gamma_s * IS
            dydt[_block(j, 2, d)] += p.gamma_s * IS - p.gamma_a * IA
            dydt[_block(j, 3, d)] += p.gamma_a * IA - p.omega_R * R

    # recovery -> next order's S (same detected-count carried over), or recycle at "3+"
    for d in range(N_DET):
        R0 = y[_block(0, 3, d)]; R1 = y[_block(1, 3, d)]
        R2 = y[_block(2, 3, d)]; R3 = y[_block(3, 3, d)]
        dydt[_block(1, 0, d)] += p.omega_R * R0
        dydt[_block(2, 0, d)] += p.omega_R * R1
        dydt[_block(3, 0, d)] += p.omega_R * (R2 + R3)

    # maternal chain exit -> S_{order=0, det=0}
    dydt[_block(0, 0, 0)] += p.rate_M * M[-1]

    return dydt


def simulate_cohort(p: CohortParams, max_age_months=40.0, n_eval=800):
    y0 = np.zeros(N_COHORT_STATE)
    y0[0] = 1.0  # entire (unit-mass) cohort starts in M_0 at birth
    t_span = (0, max_age_months * 30.4375)
    t_eval = np.linspace(*t_span, n_eval)
    # BDF, not LSODA -- see ode_model_age.py's simulate_age for why (LSODA's
    # Fortran callback bridge hangs/can't be safely interrupted on some
    # parameter corners; BDF matches it to ~13-14 sig figs and has no such issue).
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
