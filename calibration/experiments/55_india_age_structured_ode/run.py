"""Exp 55 -- India Vellore: age-structured extension of exp54's ODE.

Question: exp54's ODE tracks infection-order/immune-status but has no age
dimension (matching the ABM's own age-blind, well-mixed transmission). But
because people are born, age, and die, the CROSS-SECTIONAL age distribution
of immune status is age-structured anyway -- purely mechanically, older
children have had more time to accumulate infections. This experiment adds
that age dimension (MAL-ED bins: <6m/6-11m/12-23m/24-35m/36m+) to compute
the equilibrium age-structured immune-status distribution -- useful for (a)
initializing an ABM population directly at its endemic-equilibrium immune
structure instead of starting fully naive + a synchronized seed wave (which
exp50 showed drives the whole extinction problem), and (b) as a fast
analytical proxy for the model-implied all-infection IR-by-age, directly
comparable to the real calibration targets.

Runs at India's current MLE (exp47's collapsed posterior) -- see
ode_model_age.py for the age-structured mechanics.
"""
import sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
from ode_model import ODEParams
from ode_model_age import simulate_age, summarize_by_age, AGE_BIN_LABELS
import process_incidence_maled as P

OUT_DIR = HERE / 'outputs'
FIG_DIR = HERE / 'figures'

nroy = pd.read_csv(CALIB / 'experiments/39_india_age_binned_fixed/outputs/ts/nroy_draw.csv')
nroy['base_beta'] = np.exp(nroy['log_base_beta'])
_mle47 = pd.read_csv(CALIB / 'experiments/47_india_age_psymp_interp/outputs/ts/posterior.csv')
MLE_ROW = _mle47.mode().iloc[0]
MLE_ROW['base_beta'] = np.exp(MLE_ROW['log_base_beta'])


def params_from_row(row):
    return ODEParams(
        base_beta=float(row['base_beta']),
        sus_after_1=float(row['sus_after_1']), sus_r2=float(row['sus_r2']), sus_r3=float(row['sus_r3']),
        maternal_titer_median=float(np.exp(row['log_titer_median'])),
        maternal_titer_gsd=float(row['titer_gsd']),
        maternal_titer_half_life_days=float(row['titer_half_life_days']),
        maternal_hill_slope=float(row['hill_slope']),
        maternal_immunity_efficacy=float(row['maternal_efficacy']),
        birth_rate_per_1000=16.0, death_rate_per_1000=7.0,
    )


print("Running age-structured ODE at India's MLE (exp47)...")
p = params_from_row(MLE_ROW)
YEARS = 60
sol = simulate_age(p, n_agents=40_000, years=YEARS, n_eval=1500)

# Convergence check: compare PROPORTIONS (not absolute counts -- birth_rate >
# death_rate here, so the population grows ~0.9%/year forever and never
# stops in absolute terms; the compartment SHARES are what should equilibrate)
# at the second-to-last decade vs the final timepoint.
n_eval = sol.y.shape[1]
prop = sol.y / sol.y.sum(axis=0)
y_prev = prop[:, int(n_eval * (YEARS - 10) / YEARS) - 1]
y_end = prop[:, -1]
rel_change = np.abs(y_end - y_prev).sum()
print(f"Convergence check: change in compartment PROPORTIONS, year {YEARS-10}->{YEARS}: {rel_change:.2e}")

rows = summarize_by_age(sol, p)
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'equilibrium_by_age.csv', index=False)
pd.set_option('display.width', 160)
print(df.to_string(index=False))

# ---- Compare model-implied all-infection IR-by-age against the real targets ----
# This ODE has no detection layer -- it counts every TRUE infection. The real ir_all
# target is explicitly a DETECTED-infection rate (process_incidence_maled.py's own
# docstring; rotasim/analyzers.py's MALEDCohort). Detection probabilities, age-
# dependent for the asymptomatic pathway (monthly surveillance <12mo, quarterly
# >=12mo -- a 3x drop, MALEDCohort._p_surv):
SYMP_COLLECTION, EIA_SENS, SHED_DAYS = 0.80, 0.85, 13.0
MONTHLY_D, QUARTERLY_D = 30.4375, 91.3125
DET_SYMP = SYMP_COLLECTION * EIA_SENS                              # 0.68, age-independent
DET_ASYMP = {'<6m': (SHED_DAYS / MONTHLY_D) * EIA_SENS,            # ~0.363
             '6-11m': (SHED_DAYS / MONTHLY_D) * EIA_SENS,
             '12-23m': (SHED_DAYS / QUARTERLY_D) * EIA_SENS,       # ~0.121
             '24-35m': (SHED_DAYS / QUARTERLY_D) * EIA_SENS}
# Use the MLE's own fitted p_symp (age_binned: <6m / 6-11m / 12m+ -- the same value
# applies to both 12-23m and 24-35m, since the fitted symptom model doesn't split them).
P_SYMP_AGE = {'<6m': float(MLE_ROW['p_symp_age_0_6']), '6-11m': float(MLE_ROW['p_symp_age_6_11']),
              '12-23m': float(MLE_ROW['p_symp_age_12plus']), '24-35m': float(MLE_ROW['p_symp_age_12plus'])}

def poisson_exact_ci(cases, pt, scale=100.0, alpha=0.05):
    """Exact ("Garwood") Poisson CI on a rate cases/pt * scale, from the count alone --
    handles cases=0 correctly (lower=0, finite upper), unlike a sqrt(cases) Wald SE."""
    lo = 0.5 * chi2.ppf(alpha / 2, 2 * cases) if cases > 0 else 0.0
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (cases + 1))
    return lo / pt * scale, hi / pt * scale


target = P.load_ir_all_targets('india')
print("\n=== Model TRUE incidence vs. detection-adjusted vs. real (detected) target, ir per 100pm ===")
detected_vals = {}
target_ci = {}
for _, r in df.iterrows():
    b = r['age_bin']
    if b == '36m+':
        continue
    tgt_label = {'<6m': '<6 m', '6-11m': '6-11 m', '12-23m': '12-23 m', '24-35m': '24-35 m'}[b]
    tgt = target.loc[tgt_label, 'IR']
    cases = int(target.loc[tgt_label, 'cases']); pt = float(target.loc[tgt_label, 'PT'])
    ci_lo, ci_hi = poisson_exact_ci(cases, pt)
    target_ci[b] = (ci_lo, ci_hi)
    psymp = P_SYMP_AGE[b]
    det_prob = psymp * DET_SYMP + (1 - psymp) * DET_ASYMP[b]
    detected = r['ir_all_per_100pm'] * det_prob
    detected_vals[b] = detected
    in_ci = ci_lo <= detected <= ci_hi
    print(f"  {b:8s}  true={r['ir_all_per_100pm']:.3f}  det_prob={det_prob:.3f}  "
          f"detected={detected:.3f}  target={tgt:.3f} (95% exact CI [{ci_lo:.3f}, {ci_hi:.3f}], "
          f"n={cases} cases/{pt:.0f}pm)  model_in_CI={in_ci}")

# ---- Figure 1: stacked compartment shares by age bin ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labels = df['age_bin'].tolist()
comp_cols = ['pct_maternal', 'pct_susceptible_order0', 'pct_susceptible_order1',
             'pct_susceptible_order2', 'pct_susceptible_order3plus',
             'pct_currently_infected', 'pct_recently_immune']
comp_labels = ['maternally protected', 'susceptible (naive)', 'susceptible (1 prior inf)',
               'susceptible (2 prior inf)', 'susceptible (3+ prior inf)',
               'currently infected', 'recently immune (post-recovery)']
bottom = np.zeros(len(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(comp_cols)))
for col, clabel, color in zip(comp_cols, comp_labels, colors):
    axes[0].bar(labels, df[col], bottom=bottom, label=clabel, color=color)
    bottom += df[col].values
axes[0].set_ylabel('% of age-bin population')
axes[0].set_title("Equilibrium immune-status composition by age (India MLE)")
axes[0].legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.02, 1.0))
axes[0].set_ylim(0, 100)

# ---- Figure 2: detection-adjusted ODE vs. real (detected) target, with Poisson error
# bars on the target. Raw ODE true-incidence bars dropped (AK: distracting -- the
# comparison that actually matters is detection-adjusted vs. target); still printed
# above and in outputs/equilibrium_by_age.csv for anyone who wants them.
age_order = ['<6m', '6-11m', '12-23m', '24-35m']
tgt_labels_ordered = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
tgt_vals = [target.loc[t, 'IR'] for t in tgt_labels_ordered]
det_vals = [detected_vals[b] for b in age_order]
x = np.arange(4)
tgt_err_lo = [tgt_vals[i] - target_ci[b][0] for i, b in enumerate(age_order)]
tgt_err_hi = [target_ci[b][1] - tgt_vals[i] for i, b in enumerate(age_order)]
axes[1].bar(x - 0.175, det_vals, width=0.35, label='ODE, detection-adjusted', color='#55A868')
axes[1].bar(x + 0.175, tgt_vals, width=0.35, label='real MAL-ED target (detected)', color='#DD8452',
            yerr=[tgt_err_lo, tgt_err_hi], capsize=4, ecolor='black', error_kw=dict(elinewidth=1.2))
axes[1].set_xticks(x); axes[1].set_xticklabels(tgt_labels_ordered)
axes[1].set_ylabel('all-infection IR (per 100 person-months)')
axes[1].set_title('Detection-adjusted ODE vs. real all-infection IR-by-age\n(error bars: exact 95% Poisson CI on the target)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'age_structured_equilibrium.png', dpi=150)
print("\nSaved figures/age_structured_equilibrium.png")

# ---- Figure 3: mean prior infections by age (a direct "how would you initialize
# an ABM population" summary) ----
fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
ax2.plot(labels, df['mean_prior_infections'], 'o-', color='#55A868')
ax2.set_ylabel('mean # prior infections')
ax2.set_title('Equilibrium mean infection count by age (India MLE)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'mean_prior_infections_by_age.png', dpi=150)
print("Saved figures/mean_prior_infections_by_age.png")
