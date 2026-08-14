"""Exp 54 -- India Vellore: deterministic ODE reduction of the ABM, run
while exp52 (multi-seed survival vote) continues in the background.

Question: exp53 showed all 3 exp50/51 test points have Re_after_wave > 1 in
a hand-computed mean-field sense, yet stochastic extinction risk swings
100%->50%->0% across them. A full deterministic ODE lets us (a) confirm that
finding numerically (all three SHOULD reach a stable positive endemic
equilibrium deterministically, since a deterministic model has no
persistence-by-chance concept), and (b) sweep base_beta cheaply (milliseconds
per point vs minutes for an ABM run) to find the exact deterministic
Re=1 threshold -- the absolute floor below which NO population size or luck
could ever sustain transmission, complementing (not replacing) the
stochastic risk-zone already mapped empirically.

See ode_model.py for the compartment structure and README.md for the
simplifications relative to the full ABM.
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CALIB))
from ode_model import ODEParams, simulate, prevalence, r0_ngm, DUR_SYMP_DAYS, DUR_ASYMP_DAYS

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'

nroy = pd.read_csv(CALIB / 'experiments/39_india_age_binned_fixed/outputs/ts/nroy_draw.csv')
nroy['base_beta'] = np.exp(nroy['log_base_beta'])

POINTS = {2320: '2320 (R0~2.0, 100% extinct)', 1720: '1720 (R0~3.0, 50% extinct)',
          1072: '1072 (R0~4.1, 0% extinct)'}

# India's current best fit (exp47, age_binned + freed p_symp): the trajectory-selection
# posterior collapsed to essentially one point (2977/3000 resamples identical, ESS=1.02)
# -- the modal row IS the MLE in every meaningful sense. Same schema as nroy_draw.csv's
# transmission/maternal columns (p_symp columns are irrelevant to this ODE -- see README.md).
_mle47 = pd.read_csv(CALIB / 'experiments/47_india_age_psymp_interp/outputs/ts/posterior.csv')
MLE_ROW = _mle47.mode().iloc[0]  # modal row across all columns == the dominant (2977/3000) point
MLE_ROW['base_beta'] = np.exp(MLE_ROW['log_base_beta'])


def params_from_row(row, override_beta=None):
    return ODEParams(
        base_beta=override_beta if override_beta is not None else float(row['base_beta']),
        sus_after_1=float(row['sus_after_1']), sus_r2=float(row['sus_r2']), sus_r3=float(row['sus_r3']),
        maternal_titer_median=float(np.exp(row['log_titer_median'])),
        maternal_titer_gsd=float(row['titer_gsd']),
        maternal_titer_half_life_days=float(row['titer_half_life_days']),
        maternal_hill_slope=float(row['hill_slope']),
        maternal_immunity_efficacy=float(row['maternal_efficacy']),
        birth_rate_per_1000=16.0, death_rate_per_1000=7.0,  # India (calibrate_maled.SITE_DEMOGRAPHICS)
    )


def naive_r0_formula(base_beta):
    """The discrete-daily-hazard formula used in exp51/53, for comparison."""
    return 7.0 * ((1 - np.exp(-base_beta)) * DUR_SYMP_DAYS + (1 - np.exp(-0.1 * base_beta)) * DUR_ASYMP_DAYS)


# ---- Part 1: run the 3 tested points + India's current MLE, check convergence to
# endemic equilibrium ----
print("=== R0 consistency check (NGM ODE formula vs exp51/53 discrete-hazard formula) ===")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
results = []
all_points = list(POINTS.items()) + [('MLE', "India MLE (exp47, base_beta=%.3f)" % MLE_ROW['base_beta'])]
for idx, label in all_points:
    row = nroy.loc[idx] if idx != 'MLE' else MLE_ROW
    p = params_from_row(row)
    r0_ode = r0_ngm(p)
    r0_disc = naive_r0_formula(p.base_beta)
    print(f"  idx={idx}: base_beta={p.base_beta:.4f}  R0(NGM-ODE)={r0_ode:.2f}  "
          f"R0(discrete)={r0_disc:.2f}  maternal_mean_dur={p.mat_mean_duration:.1f}d")
    sol = simulate(p, n_agents=40_000, years=15)
    prev = prevalence(sol)
    t_years = sol.t / 365.25
    axes[0].plot(t_years, prev * 100, label=f"idx={idx} (R0={r0_ode:.1f})")
    eq_prev = prev[-1]
    n_infected = prev * 40_000
    # post-peak trough: minimum infected count after the initial wave's peak, before
    # the trajectory either recovers toward equilibrium or (deterministically) decays further
    peak_i = int(np.argmax(n_infected))
    trough_i = peak_i + int(np.argmin(n_infected[peak_i:])) if peak_i < len(n_infected) - 1 else peak_i
    trough_n = float(n_infected[trough_i])
    # effective R AT the trough: NGM-style, using the actual (depleted) susceptible
    # composition at that moment rather than the naive fully-susceptible R0
    Y_trough = sol.y[:, trough_i]
    S_trough = Y_trough[6:10]  # IDX_S
    N_trough = Y_trough.sum()
    s_weighted = float((p.sigma * S_trough).sum()) / max(N_trough, 1e-9)
    re_trough = r0_ode * s_weighted
    # crude branching-process extinction-probability approximation (linear birth-death,
    # n0 = trough count as the number of independent residual lineages): P_ext ~ (1/Re)^n0
    p_ext_approx = (1.0 / re_trough) ** trough_n if re_trough > 1 else 1.0
    print(f"    -> post-peak trough: {trough_n:.2f} infected (of 40,000), "
          f"Re_trough={re_trough:.2f}, naive branching P(ext)~{p_ext_approx:.3g}")
    results.append(dict(orig_idx=idx, base_beta=p.base_beta, r0_ode=r0_ode, r0_discrete=r0_disc,
                         mat_mean_duration_days=p.mat_mean_duration, equilibrium_prevalence_pct=eq_prev * 100,
                         post_peak_trough_n_infected=trough_n, re_trough=re_trough,
                         branching_p_ext_approx=p_ext_approx))

axes[0].set_xlabel('years'); axes[0].set_ylabel('prevalence (%)')
axes[0].set_title('ODE prevalence over time (deterministic)')
axes[0].legend(fontsize=8)

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / 'ode_vs_r0_check.csv', index=False)
print()
print(res_df.to_string(index=False))

# ---- Part 2: sweep base_beta (holding 1720's other params fixed) to find the
# deterministic Re=1 threshold ----
print("\n=== base_beta sweep: deterministic equilibrium prevalence ===")
row_1720 = nroy.loc[1720]
betas = np.linspace(0.01, 0.15, 40)
eq_prevs = []
for b in betas:
    p = params_from_row(row_1720, override_beta=b)
    sol = simulate(p, n_agents=40_000, years=20, n_eval=500)
    eq_prevs.append(prevalence(sol)[-1] * 100)
eq_prevs = np.array(eq_prevs)

axes[1].plot(betas, eq_prevs, 'o-', color='#55A868')
# mark the 3 tested points' betas + India's current MLE
for idx, label in POINTS.items():
    b = float(nroy.loc[idx, 'base_beta'])
    axes[1].axvline(b, color='gray', ls=':', lw=1)
    axes[1].annotate(f"{idx}", (b, axes[1].get_ylim()[1] * 0.9), fontsize=8, ha='center')
b_mle = float(MLE_ROW['base_beta'])
axes[1].axvline(b_mle, color='crimson', ls='--', lw=1.3)
axes[1].annotate("MLE", (b_mle, axes[1].get_ylim()[1] * 0.75), fontsize=8, ha='center', color='crimson')
axes[1].set_xlabel('base_beta (other params = orig_idx 1720)')
axes[1].set_ylabel('deterministic equilibrium prevalence (%)')
axes[1].set_title('Deterministic threshold: where does eq. prevalence hit 0?')

plt.tight_layout()
plt.savefig(FIG_DIR / 'ode_threshold.png', dpi=150)
print(f"\nSaved figures/ode_threshold.png")

sweep_df = pd.DataFrame(dict(base_beta=betas, equilibrium_prevalence_pct=eq_prevs))
sweep_df.to_csv(OUT_DIR / 'beta_sweep.csv', index=False)

# Find the approximate critical beta (first beta where eq. prevalence > 0.01%)
viable = sweep_df[sweep_df.equilibrium_prevalence_pct > 0.01]
if len(viable):
    print(f"Approx deterministic critical base_beta: {viable.base_beta.min():.4f} "
          f"(vs tested points 2320={nroy.loc[2320,'base_beta']:.4f}, "
          f"1720={nroy.loc[1720,'base_beta']:.4f}, 1072={nroy.loc[1072,'base_beta']:.4f}, "
          f"MLE={float(MLE_ROW['base_beta']):.4f})")
