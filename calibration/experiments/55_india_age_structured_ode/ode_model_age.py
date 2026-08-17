"""Age-structured extension of exp54's order-structured ODE.

Motivation: transmission itself is NOT age-structured in the ABM (a
well-mixed random contact network, confirmed via source -- ss.RandomNet has
no age-assortativity), and susceptibility-by-order (sus_after_1/2/3) is also
age-independent. But because people are born, age, and die, the CROSS-
SECTIONAL age distribution of infection-order/immune-status is age-
structured anyway -- a 30-month-old has simply had more time to accumulate
infections than a 3-month-old, purely mechanically. That age-structured
immune-status distribution is exactly what you'd want to initialize an ABM
population at (instead of starting everyone fully naive), and its per-age-
bin infection incidence is the same sufficient statistic the real symptom-
detection layer (p_symp by age) multiplies to produce the cohort's IR-by-age
targets -- see SUMMARY.md.

Reuses exp54's ode_model.py building blocks (ODEParams, sigma_from_params,
the 22-compartment per-age-bin layout: M x6, S/IS/IA/R x4 each) and adds a
second, age dimension. Each age bin is treated as a well-mixed compartment
with a fixed EXIT rate = 1/bin_width_days (the standard "flux out the top
of a uniformly-populated bin" approximation used throughout age-structured
compartmental models -- adequate at MAL-ED's bin widths; NOT adequate if
within-bin age structure narrower than the bin itself mattered, which it
doesn't for this purpose).
"""
import sys, pathlib
import numpy as np
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
from ode_model import (ODEParams, N_MAT_STAGES, N_ORDERS, N_STATE, N_CONTACTS,
                        IDX_M, IDX_S, IDX_IS, IDX_IA, IDX_R)

# MAL-ED age bins, in days. Last bin is absorbing (no exit -- "36m+"/adult pool).
AGE_BIN_LABELS = ['<6m', '6-11m', '12-23m', '24-35m', '36m+']
AGE_BIN_WIDTHS_DAYS = [182.625, 182.625, 365.25, 365.25, None]
N_AGE_BINS = len(AGE_BIN_LABELS)
AGE_EXIT_RATE = np.array([1.0 / w if w is not None else 0.0 for w in AGE_BIN_WIDTHS_DAYS])

N_STATE_FLAT = N_AGE_BINS * N_STATE


def rhs_age(t, y_flat, p: ODEParams):
    Y = y_flat.reshape(N_AGE_BINS, N_STATE)
    N_total = Y.sum()
    IS_total = Y[:, IDX_IS].sum()
    IA_total = Y[:, IDX_IA].sum()
    # Force of infection is a SINGLE population-wide number -- transmission is
    # well-mixed across ages (see module docstring) -- applied identically to
    # every age bin's susceptible classes below.
    foi = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / max(N_total, 1e-9)

    dY = np.zeros_like(Y)
    births = p.birth_rate * N_total
    for a in range(N_AGE_BINS):
        M = Y[a, IDX_M]; S = Y[a, IDX_S]; IS = Y[a, IDX_IS]; IA = Y[a, IDX_IA]; R = Y[a, IDX_R]

        dM = np.zeros(N_MAT_STAGES)
        dM[0] = (births if a == 0 else 0.0) - p.rate_M * M[0]
        for i in range(1, N_MAT_STAGES):
            dM[i] = p.rate_M * M[i - 1] - p.rate_M * M[i]

        dS = np.zeros(N_ORDERS)
        dS[0] = p.rate_M * M[-1] - foi * p.sigma[0] * S[0]
        dS[1] = p.omega_R * R[0] - foi * p.sigma[1] * S[1]
        dS[2] = p.omega_R * R[1] - foi * p.sigma[2] * S[2]
        dS[3] = p.omega_R * (R[2] + R[3]) - foi * p.sigma[3] * S[3]

        dIS = foi * p.sigma * S - p.gamma_s * IS
        dIA = p.gamma_s * IS - p.gamma_a * IA
        dR = p.gamma_a * IA - p.omega_R * R

        dY[a, IDX_M] = dM; dY[a, IDX_S] = dS; dY[a, IDX_IS] = dIS
        dY[a, IDX_IA] = dIA; dY[a, IDX_R] = dR

    # Aging: shift each age bin's ENTIRE compartment vector into the next bin
    # (order/immune status is preserved across an aging event -- aging and
    # infection are independent processes here).
    for a in range(N_AGE_BINS - 1):
        flow = AGE_EXIT_RATE[a] * Y[a]
        dY[a] -= flow
        dY[a + 1] += flow

    # Deaths: uniform per-capita, every compartment, every age bin (matches the
    # ABM's own demography -- no age-specific mortality implemented there either).
    dY -= p.death_rate * Y

    return dY.flatten()


def initial_state_age(n_agents, init_prevalence=0.04):
    """Rough starting point -- doesn't need to be realistic, just non-degenerate;
    run long enough (decades) for both the order dynamics and the age structure
    to reach their joint equilibrium, same logic as exp54."""
    Y = np.zeros((N_AGE_BINS, N_STATE))
    # crude stable-age-distribution guess: exponential-in-age weights using the
    # bin widths and a nominal death rate, just to avoid a wildly-wrong start
    weights = np.array([w if w is not None else 3 * 365.25 for w in AGE_BIN_WIDTHS_DAYS])
    weights = weights / weights.sum()
    for a in range(N_AGE_BINS):
        Y[a, IDX_S][0] = n_agents * weights[a]
    n_seed = n_agents * init_prevalence
    Y[N_AGE_BINS // 2, IDX_S][0] -= n_seed
    Y[N_AGE_BINS // 2, IDX_IS][0] += n_seed
    return Y.flatten()


def simulate_age(p: ODEParams, n_agents=40_000, years=40, n_eval=1000):
    y0 = initial_state_age(n_agents)
    t_span = (0, years * 365.25)
    t_eval = np.linspace(*t_span, n_eval)
    # BDF (not LSODA): matches LSODA to ~13-14 sig figs on typical draws, same
    # speed, but LSODA's Fortran/ODEPACK callback bridge can drive step size to
    # underflow and hang indefinitely on some parameter corners (found during
    # exp57's differential_evolution search); BDF is a pure-Python solve_ivp
    # method with no such failure mode and resolved the same corner in ~1s.
    sol = solve_ivp(rhs_age, t_span, y0, args=(p,), method='BDF',
                     t_eval=t_eval, rtol=1e-8, atol=1e-6, max_step=10.0)
    return sol


def summarize_by_age(sol, p: ODEParams):
    """Returns a dict per age bin of equilibrium (final-timepoint) summary
    stats: population share, % maternally protected, % by susceptible order,
    % currently infected, % recently-immune, mean prior infections, and
    per-capita ALL-INFECTION incidence (IR per 100 person-MONTHS -- matches
    process_incidence_maled's convention, confirmed via its own docstring and
    the PT magnitudes, NOT person-years) -- the quantity the real symptom-
    detection layer (p_symp by age) would multiply to get the model-implied
    SYMPTOMATIC IR-by-age. Directly comparable to process_incidence_maled's
    ir_all_by_age targets as a fidelity check (see SUMMARY.md)."""
    Y_final = sol.y[:, -1].reshape(N_AGE_BINS, N_STATE)
    N_total = Y_final.sum()
    IS_total = Y_final[:, IDX_IS].sum(); IA_total = Y_final[:, IDX_IA].sum()
    foi = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total
    rows = []
    for a, label in enumerate(AGE_BIN_LABELS):
        y = Y_final[a]
        n_a = y.sum()
        s_weighted = float((p.sigma * y[IDX_S]).sum())
        incidence_per_day_per_capita = foi * s_weighted / n_a
        ir_per_100_person_months = incidence_per_day_per_capita * (365.25 / 12.0) * 100
        m_frac = y[IDX_M].sum() / n_a
        s = y[IDX_S]
        is_ = y[IDX_IS]; ia = y[IDX_IA]; r = y[IDX_R]
        infected_frac = (is_.sum() + ia.sum()) / n_a
        recent_immune_frac = r.sum() / n_a
        s_frac_by_order = s / n_a
        mean_order = float(np.dot(np.arange(4), s + is_ + ia + r) / n_a)
        rows.append(dict(age_bin=label, n_agents=n_a, pct_of_pop=100 * n_a / Y_final.sum(),
                          pct_maternal=100 * m_frac,
                          pct_susceptible_order0=100 * s_frac_by_order[0],
                          pct_susceptible_order1=100 * s_frac_by_order[1],
                          pct_susceptible_order2=100 * s_frac_by_order[2],
                          pct_susceptible_order3plus=100 * s_frac_by_order[3],
                          ir_all_per_100pm=ir_per_100_person_months,
                          pct_currently_infected=100 * infected_frac,
                          pct_recently_immune=100 * recent_immune_frac,
                          mean_prior_infections=mean_order))
    return rows
