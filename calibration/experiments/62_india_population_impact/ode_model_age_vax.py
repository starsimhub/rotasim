"""exp62: age-structured population model extended with a resolved 3-dose
vaccination schedule in the <6m window, per AK's feedback (2026-08-19):
doses must be applied at their real ages (not lumped at 6mo, since <6m
incidence is itself one of this arc's key targets), and per-dose coverage
cascades (independence assumption: C1=sqrt(C2), C2=sqrt(C3)) rather than
one flat coverage number applied identically to all 3 doses.

State layout: the <6m window (0-182.625 days) is split into 4 fine
sub-bins at the real dose ages (6, 10, 14 weeks: [0,6wk), [6,10wk),
[10,14wk), [14wk,26wk)). Each fine sub-bin carries a "doses-received-so-far"
axis (0..3), uniform depth 4 for all 4 fine bins for simplicity (dose-levels
not yet reachable in early bins just carry zero mass -- e.g. bin0 only ever
has mass at dose-level 0). The dose-level axis is collapsed (summed) when
population ages out of the last fine bin into the existing 6-11m bin, since
only the cumulative immunity-order effect matters after that point, not
dose history itself.

Because this models a POPULATION in steady flow through age bins (not a
single birth cohort), the "dose event" at each fine-bin boundary is a
continuous FLOW SPLIT (a fraction of the outflow diverts to a different
dose-level / order state), not a discrete jump requiring segmented
integration -- the whole thing is one smooth RHS integrated over 60 years,
unlike exp60/61's birth-cohort approach.

Reuses ode_model.py's ODEParams/sigma_from_params (transmission mechanics,
per-order susceptibility) unchanged.
"""
import sys, pathlib
import numpy as np
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
from ode_model import (ODEParams, N_MAT_STAGES, N_ORDERS, N_STATE, N_CONTACTS,
                        IDX_M, IDX_S, IDX_IS, IDX_IA, IDX_R)

N_DOSE_LEVELS = 4  # 0, 1, 2, 3 doses received so far

# Fine <6m sub-bins at the real Rotavac dose ages (days): birth, dose1(6wk),
# dose2(10wk), dose3(14wk), end of <6m window (26wk =~ 182.625d).
FINE_BIN_EDGES_DAYS = [0.0, 42.0, 70.0, 98.0, 182.625]
N_FINE_BINS = 4
FINE_BIN_WIDTHS = np.diff(FINE_BIN_EDGES_DAYS)
FINE_EXIT_RATE = 1.0 / FINE_BIN_WIDTHS

COARSE_BIN_LABELS = ['6-11m', '12-23m', '24-35m', '36m+']
COARSE_BIN_WIDTHS_DAYS = [182.625, 365.25, 365.25, None]
N_COARSE_BINS = len(COARSE_BIN_LABELS)
COARSE_EXIT_RATE = np.array([1.0 / w if w is not None else 0.0 for w in COARSE_BIN_WIDTHS_DAYS])

AGE_BIN_LABELS = ['0-6wk', '6-10wk', '10-14wk', '14-26wk'] + COARSE_BIN_LABELS
N_AGE_BINS_TOTAL = N_FINE_BINS + N_COARSE_BINS

N_STATE_FINE = N_DOSE_LEVELS * N_STATE
N_STATE_FLAT = N_FINE_BINS * N_STATE_FINE + N_COARSE_BINS * N_STATE

FINE_OFFSETS = [i * N_STATE_FINE for i in range(N_FINE_BINS)]
COARSE_START = N_FINE_BINS * N_STATE_FINE
COARSE_OFFSETS = [COARSE_START + i * N_STATE for i in range(N_COARSE_BINS)]


class VaxParams:
    def __init__(self, ode_params: ODEParams, take: float, coverage3: float):
        self.p = ode_params
        self.take = take
        c3 = coverage3
        c2 = np.sqrt(c3)
        c1 = np.sqrt(c2)
        self.C = [c1, c2, c3]  # cumulative "on-schedule" coverage for dose 1,2,3
        # conditional coverage at each dose boundary -- guard div-by-zero (c3=0 -> c1=c2=0)
        self.cond = [c1, (c2 / c1) if c1 > 0 else 0.0, (c3 / c2) if c2 > 0 else 0.0]


def _fine_block(y, bin_idx, dose_level):
    i = FINE_OFFSETS[bin_idx] + dose_level * N_STATE
    return y[i:i + N_STATE]


def _fine_idx(bin_idx, dose_level):
    return FINE_OFFSETS[bin_idx] + dose_level * N_STATE


def _coarse_idx(bin_idx):
    return COARSE_OFFSETS[bin_idx]


def _shift_order_by_1(block):
    """Advance S/IS/IA/R one susceptibility order (order3 is the '3+'
    ceiling -- order2 and order3 both collapse into it). M (maternal)
    untouched -- same non-maternal-compartment caveat as exp60/61: the
    order-jump only applies to S/IS/IA/R."""
    out = block.copy()
    for idx_group in (IDX_S, IDX_IS, IDX_IA, IDX_R):
        vec = block[idx_group]
        shifted = np.zeros(N_ORDERS)
        shifted[1] += vec[0]
        shifted[2] += vec[1]
        shifted[3] += vec[2] + vec[3]
        out[idx_group] = shifted
    return out


def _local_dynamics(block, p: ODEParams, foi, birth_inflow=0.0):
    """Standard within-bin M/S/IS/IA/R transmission dynamics (no aging/deaths
    -- those are handled separately), identical mechanics to
    ode_model_age.rhs_age's per-bin block."""
    M = block[IDX_M]; S = block[IDX_S]; IS = block[IDX_IS]; IA = block[IDX_IA]; R = block[IDX_R]
    dM = np.zeros(N_MAT_STAGES)
    dM[0] = birth_inflow - p.rate_M * M[0]
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
    out = np.zeros(N_STATE)
    out[IDX_M] = dM; out[IDX_S] = dS; out[IDX_IS] = dIS; out[IDX_IA] = dIA; out[IDX_R] = dR
    return out


def rhs_age_vax(t, y, vp: VaxParams):
    p = vp.p
    dydt = np.zeros(N_STATE_FLAT)

    # ---- population-wide FOI (well-mixed across ages AND dose-levels) ----
    IS_total = 0.0; IA_total = 0.0; N_total = 0.0
    for b in range(N_FINE_BINS):
        for d in range(N_DOSE_LEVELS):
            blk = _fine_block(y, b, d)
            IS_total += blk[IDX_IS].sum(); IA_total += blk[IDX_IA].sum(); N_total += blk.sum()
    for b in range(N_COARSE_BINS):
        i = _coarse_idx(b); blk = y[i:i + N_STATE]
        IS_total += blk[IDX_IS].sum(); IA_total += blk[IDX_IA].sum(); N_total += blk.sum()
    foi = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / max(N_total, 1e-9)
    births = p.birth_rate * N_total

    # ---- local transmission dynamics, every (fine bin, dose level) and coarse bin ----
    for b in range(N_FINE_BINS):
        for d in range(N_DOSE_LEVELS):
            i = _fine_idx(b, d)
            birth_inflow = births if (b == 0 and d == 0) else 0.0
            dydt[i:i + N_STATE] += _local_dynamics(y[i:i + N_STATE], p, foi, birth_inflow)
    for b in range(N_COARSE_BINS):
        i = _coarse_idx(b)
        dydt[i:i + N_STATE] += _local_dynamics(y[i:i + N_STATE], p, foi, 0.0)

    # ---- aging flow between fine bins, with the dose-schedule logic at each boundary ----
    for b in range(N_FINE_BINS):
        rate = FINE_EXIT_RATE[b]
        is_dose_boundary = b < 3  # boundaries 0->1, 1->2, 2->3 are dose 1,2,3; boundary 3->coarse0 is not
        for d in range(N_DOSE_LEVELS):
            i = _fine_idx(b, d)
            flow = rate * y[i:i + N_STATE]
            dydt[i:i + N_STATE] -= flow
            if is_dose_boundary and d == b:
                # on-schedule sub-population: eligible for dose b+1 at this boundary
                cond = vp.cond[b]
                not_dosed = (1 - cond) * flow
                dosed_no_take = cond * (1 - vp.take) * flow
                dosed_take = cond * vp.take * flow
                j_next_same = _fine_idx(b + 1, d)          # didn't receive this dose
                j_next_dosed = _fine_idx(b + 1, d + 1)      # received dose (dose-level +1)
                dydt[j_next_same:j_next_same + N_STATE] += not_dosed
                dydt[j_next_dosed:j_next_dosed + N_STATE] += dosed_no_take + _shift_order_by_1(dosed_take)
            elif b < N_FINE_BINS - 1:
                # off-schedule (already missed an earlier dose) or not the eligible dose-level:
                # ages on unchanged at the same dose-level
                j_next = _fine_idx(b + 1, d)
                dydt[j_next:j_next + N_STATE] += flow
            else:
                # b == N_FINE_BINS-1 (last fine bin): flows into coarse[0], dose-level axis
                # collapsed (summed) -- no more dosing, only cumulative order matters now
                j_next = _coarse_idx(0)
                dydt[j_next:j_next + N_STATE] += flow

    # ---- aging flow between coarse bins (unchanged from ode_model_age.py) ----
    for b in range(N_COARSE_BINS - 1):
        i = _coarse_idx(b)
        flow = COARSE_EXIT_RATE[b] * y[i:i + N_STATE]
        dydt[i:i + N_STATE] -= flow
        j = _coarse_idx(b + 1)
        dydt[j:j + N_STATE] += flow

    # ---- deaths: uniform per-capita, every compartment ----
    dydt -= p.death_rate * y

    return dydt


def initial_state_age_vax(n_agents, init_prevalence=0.04):
    y0 = np.zeros(N_STATE_FLAT)
    all_widths = list(FINE_BIN_WIDTHS) + [w if w is not None else 3 * 365.25 for w in COARSE_BIN_WIDTHS_DAYS]
    weights = np.array(all_widths) / sum(all_widths)
    s0_offset = IDX_S.start       # order-0 susceptible sits at the first element of the S block
    is0_offset = IDX_IS.start
    for b in range(N_FINE_BINS):
        i = _fine_idx(b, 0)  # all initial mass at dose-level 0 -- fine, equilibrates regardless
        y0[i + s0_offset] = n_agents * weights[b]
    for b in range(N_COARSE_BINS):
        i = _coarse_idx(b)
        y0[i + s0_offset] = n_agents * weights[N_FINE_BINS + b]
    # seed a little infection prevalence in a middle coarse bin, same trick as ode_model_age.py
    seed_i = _coarse_idx(1)
    n_seed = n_agents * init_prevalence
    y0[seed_i + s0_offset] -= n_seed
    y0[seed_i + is0_offset] += n_seed
    return y0


def simulate_age_vax(p: ODEParams, take: float, coverage3: float, n_agents=40_000, years=60, n_eval=400):
    vp = VaxParams(p, take, coverage3)
    y0 = initial_state_age_vax(n_agents)
    t_span = (0, years * 365.25)
    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(rhs_age_vax, t_span, y0, args=(vp,), method='BDF',
                     t_eval=t_eval, rtol=1e-8, atol=1e-6, max_step=10.0)
    return sol


def summarize_by_age_vax(sol, p: ODEParams):
    """Equilibrium (final-timepoint) summary per age bin, dose-level-collapsed
    for the fine bins -- mirrors ode_model_age.summarize_by_age's per-age-bin
    output shape (pct_of_pop, pct_maternal, susceptible-by-order fractions,
    all-infection incidence) so downstream comparison code can treat this
    exactly like the unvaccinated model's output."""
    Y_final = sol.y[:, -1]
    IS_total = 0.0; IA_total = 0.0; N_total = 0.0
    blocks = []
    for b in range(N_FINE_BINS):
        agg = np.zeros(N_STATE)
        for d in range(N_DOSE_LEVELS):
            agg += Y_final[_fine_idx(b, d):_fine_idx(b, d) + N_STATE]
        blocks.append(agg)
    for b in range(N_COARSE_BINS):
        i = _coarse_idx(b)
        blocks.append(Y_final[i:i + N_STATE])
    for blk in blocks:
        IS_total += blk[IDX_IS].sum(); IA_total += blk[IDX_IA].sum(); N_total += blk.sum()
    foi = N_CONTACTS * p.base_beta * (IS_total + 0.1 * IA_total) / N_total

    rows = []
    for label, y in zip(AGE_BIN_LABELS, blocks):
        n_a = y.sum()
        s_weighted = float((p.sigma * y[IDX_S]).sum())
        incidence_per_day = foi * s_weighted / n_a
        ir_per_100pm = incidence_per_day * (365.25 / 12.0) * 100
        m_frac = y[IDX_M].sum() / n_a
        s = y[IDX_S]; is_ = y[IDX_IS]; ia = y[IDX_IA]; r = y[IDX_R]
        s_frac_by_order = s / n_a
        rows.append(dict(age_bin=label, n_agents=n_a, pct_of_pop=100 * n_a / Y_final.sum(),
                          pct_maternal=100 * m_frac,
                          pct_susceptible_order0=100 * s_frac_by_order[0],
                          pct_susceptible_order1=100 * s_frac_by_order[1],
                          pct_susceptible_order2=100 * s_frac_by_order[2],
                          pct_susceptible_order3plus=100 * s_frac_by_order[3],
                          ir_all_per_100pm=ir_per_100pm,
                          pct_currently_infected=100 * (is_.sum() + ia.sum()) / n_a,
                          pct_recently_immune=100 * r.sum() / n_a))
    return rows
