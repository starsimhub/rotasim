"""Deterministic ODE reduction of the rotasim ABM's transmission mechanics.

Scope, deliberately: this reproduces the TRANSMISSION side of the model
faithfully enough to study deterministic steady-state / threshold behaviour
(the same question exp50/51/53 approached stochastically and analytically).
It does NOT reproduce the symptom-model / detection layer (age_binned,
infnum p_symp) -- those are purely an observation layer in the real model
(they decide which infections get COUNTED as symptomatic cases; they do not
feed back into transmission at all, confirmed by reading rotasim/rotavirus.py
directly -- rel_trans depends only on symptomatic/asymptomatic PHASE, never
on the age/order-based p_symp probabilities). So this ODE cannot speak to
the IR-by-age cohort-fit question, only to persistence/extinction dynamics.

Compartments (all counts, not fractions; total N(t) emerges from births/deaths):
  M_0..M_5    : maternally protected (6-stage Erlang chain, matches the fitted
                titer/Hill curve's mean protected duration -- see
                `maternal_mean_duration_days`). Treated as fully protected
                (rel_sus=0) while in this chain -- a documented simplification
                of the smooth log-normal-titer/Hill decay (see README.md).
  S_j (j=0..3): susceptible with j prior infections (j=3 means "3+", capped --
                matches the ABM's sus_after_3plus applying to all order>=3).
                Relative susceptibility sigma_j.
  IS_j        : symptomatic-transmission phase (rel_trans=1.0) of the
                (j+1)-th infection. Mean duration dur_symp days.
  IA_j        : asymptomatic-transmission phase (rel_trans=0.1) of the same
                infection, following IS_j (confirmed via
                rotasim/rotavirus.py:step_state -- infected/symptomatic phase
                comes FIRST, asymptomatic SECOND). Mean duration dur_asymp days.
  R_j         : just recovered from the (j+1)-th infection; temporarily
                strongly protected (fixed model default: mean duration 91
                days, rotasim/rotavirus.py's `waning_rate_dist`) before
                settling to the PERMANENT floor susceptibility sigma_{j+1}
                (or sigma_3 if j>=2) as S_{j+1}. This stage is a genuine ABM
                mechanism most of this project's prior back-of-envelope R0/Re
                calculations omitted (they used sigma_1 as if it applied
                immediately post-recovery, which understates the true
                post-wave protection for the first ~91 days).

Force of infection (frequency-dependent, matching the ABM's daily-resampled
n_contacts=7 random network):
  lambda(t) = n_contacts * base_beta * (IS_total + 0.1*IA_total) / N(t)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad

N_ORDERS = 4          # j = 0,1,2,3 ("3" = "3+")
N_MAT_STAGES = 6       # matches the ABM's fitted maternal n_stages
WANING_MEAN_DAYS = 91.0  # fixed ABM default (rotasim/rotavirus.py: waning_rate_dist)
DUR_ASYMP_DAYS = 8.0     # fixed ABM default (rotasim/rotavirus.py: dur_asymptomatic)
DUR_INF_MEAN_DAYS = 13.0  # fixed ABM default (rotasim/rotavirus.py: dur_inf)
DUR_SYMP_DAYS = DUR_INF_MEAN_DAYS - DUR_ASYMP_DAYS  # mean symptomatic-phase duration
N_CONTACTS = 7.0


def maternal_mean_duration_days(mat_eff, titer_median, titer_gsd, half_life_days, hill_slope):
    """Mean protected duration (days) = integral of the population-median
    protection curve over age. Ignores the log-normal spread across infants
    (uses the median trajectory) -- a point simplification; see README.md."""
    def protection(age_days):
        titer = titer_median * np.exp(-np.log(2) * age_days / half_life_days)
        th = np.maximum(titer, 0.0) ** hill_slope
        return mat_eff * (th / (th + 1.0))
    val, _ = quad(protection, 0, 5 * half_life_days * max(hill_slope, 1.0))
    return max(val, 1.0)  # guard against degenerate (near-zero) durations


def sigma_from_params(sus_after_1, sus_r2, sus_r3):
    s1 = sus_after_1
    s2 = s1 * sus_r2
    s3 = s2 * sus_r3
    return np.array([1.0, s1, s2, s3])


class ODEParams:
    """Bundles one parameter draw (same schema as hm_calibrate.untransform's
    output) into the rates this ODE needs."""
    def __init__(self, base_beta, sus_after_1, sus_r2, sus_r3,
                 maternal_titer_median, maternal_titer_gsd,
                 maternal_titer_half_life_days, maternal_hill_slope,
                 maternal_immunity_efficacy,
                 birth_rate_per_1000=16.0, death_rate_per_1000=7.0):
        self.base_beta = base_beta
        self.sigma = sigma_from_params(sus_after_1, sus_r2, sus_r3)
        self.gamma_s = 1.0 / DUR_SYMP_DAYS
        self.gamma_a = 1.0 / DUR_ASYMP_DAYS
        self.omega_R = 1.0 / WANING_MEAN_DAYS
        self.mat_mean_duration = maternal_mean_duration_days(
            maternal_immunity_efficacy, maternal_titer_median, maternal_titer_gsd,
            maternal_titer_half_life_days, maternal_hill_slope)
        self.rate_M = N_MAT_STAGES / self.mat_mean_duration
        self.birth_rate = birth_rate_per_1000 / 1000.0 / 365.25   # per day
        self.death_rate = death_rate_per_1000 / 1000.0 / 365.25   # per day


# State vector layout: [M(6), S(4), IS(4), IA(4), R(4)] = 22 compartments
IDX_M = slice(0, 6)
IDX_S = slice(6, 10)
IDX_IS = slice(10, 14)
IDX_IA = slice(14, 18)
IDX_R = slice(18, 22)
N_STATE = 22


def rhs(t, y, p: ODEParams):
    M = y[IDX_M]; S = y[IDX_S]; IS = y[IDX_IS]; IA = y[IDX_IA]; R = y[IDX_R]
    N = y.sum()
    IS_total = IS.sum(); IA_total = IA.sum()
    foi = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / max(N, 1e-9)

    dydt = np.zeros(N_STATE)
    births = p.birth_rate * N

    # Maternal Erlang chain
    dM = np.zeros(N_MAT_STAGES)
    dM[0] = births - p.rate_M * M[0] - p.death_rate * M[0]
    for i in range(1, N_MAT_STAGES):
        dM[i] = p.rate_M * M[i - 1] - p.rate_M * M[i] - p.death_rate * M[i]
    dydt[IDX_M] = dM

    # Susceptible classes: j=0 fed by maternal-chain exit; j=1,2 fed by R_{j-1};
    # j=3 ("3+") fed by BOTH R_2 (entering 3+ for the first time) and R_3 (recycling)
    dS = np.zeros(N_ORDERS)
    dS[0] = p.rate_M * M[-1] - foi * p.sigma[0] * S[0] - p.death_rate * S[0]
    dS[1] = p.omega_R * R[0] - foi * p.sigma[1] * S[1] - p.death_rate * S[1]
    dS[2] = p.omega_R * R[1] - foi * p.sigma[2] * S[2] - p.death_rate * S[2]
    dS[3] = p.omega_R * (R[2] + R[3]) - foi * p.sigma[3] * S[3] - p.death_rate * S[3]
    dydt[IDX_S] = dS

    # Symptomatic-phase infections
    dIS = foi * p.sigma * S - p.gamma_s * IS - p.death_rate * IS
    dydt[IDX_IS] = dIS

    # Asymptomatic-phase infections
    dIA = p.gamma_s * IS - p.gamma_a * IA - p.death_rate * IA
    dydt[IDX_IA] = dIA

    # Temporarily strongly-immune (post-recovery, pre-floor)
    dR = p.gamma_a * IA - p.omega_R * R - p.death_rate * R
    dydt[IDX_R] = dR

    return dydt


def initial_state(n_agents, init_prevalence=0.04):
    """Mimic the ABM's init_post seeding: a small fraction start infected
    (symptomatic phase), the rest fully susceptible (order 0). No attempt to
    pre-seed the maternal chain or higher orders at t=0 -- the ODE runs long
    enough that initial-condition details wash out before the horizon of
    interest (a handful of years), same as the ABM's 5-year burn-in."""
    y0 = np.zeros(N_STATE)
    n_seed = n_agents * init_prevalence
    y0[IDX_S][0] = n_agents - n_seed
    y0[IDX_IS][0] = n_seed
    return y0


def simulate(p: ODEParams, n_agents=40_000, years=15, n_eval=2000):
    y0 = initial_state(n_agents)
    t_span = (0, years * 365.25)
    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(rhs, t_span, y0, args=(p,), method='LSODA',
                     t_eval=t_eval, rtol=1e-8, atol=1e-6, max_step=5.0)
    return sol


def prevalence(sol):
    """Total currently-infected (IS+IA) fraction of the population, per eval point."""
    Y = sol.y
    N = Y.sum(axis=0)
    infected = Y[IDX_IS].sum(axis=0) + Y[IDX_IA].sum(axis=0)
    return infected / N


def r0_ngm(p: ODEParams):
    """Next-generation-matrix R0 in a fully naive population (S_0=N, all
    other compartments 0) -- should match the closed-form formula used in
    exp51/53: N_CONTACTS * base_beta * [(1-e^-b)... ] approximated here via
    the linearised NGM instead of the discrete-hazard formula, as a
    consistency check between the two derivations."""
    # New infections per unit time per current infected, in a naive population:
    # an IS individual generates N_CONTACTS*base_beta*sigma_0 new infections/day
    # (sigma_0=1) for 1/gamma_s days on average while symptomatic, then becomes
    # IA (still infectious, at 10% rate) for 1/gamma_a days.
    new_infections_per_IS = N_CONTACTS * p.base_beta * 1.0 / p.gamma_s
    new_infections_per_IA = N_CONTACTS * p.base_beta * 0.1 * 1.0 / p.gamma_a
    # An infection passes through IS then IA in sequence, contributing both:
    return new_infections_per_IS + new_infections_per_IA
