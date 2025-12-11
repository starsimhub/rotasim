"""
Fresh 20-trial calibration of infection_number model with fitted immunity
Uses FIXED age distribution extraction code
Generates figures for MLE and optionally top 5 models
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna
import json
from pathlib import Path

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibrate_infection_number_fitted_immunity import UKAgeCalibrationFittedImmunity
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("FRESH 20-TRIAL CALIBRATION: Fitted Immunity Model (FIXED Age Extraction)")
print("=" * 80)
print("\nThis calibration uses the FIXED age distribution extraction code.")
print("Both incidence AND age distribution will be properly optimized.\n")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print(f"Target incidence: {target_incidence:.2f} per 100,000")
print("Target age distribution:")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Define calibration parameters
calib_pars = dict(
    reporting_rate=[0.01, 0.001, 0.05],  # [best_guess, lower_bound, upper_bound]
    base_beta=[2.5, 1.0, 8.0],
    sus_after_1=[0.85, 0.5, 1.0],       # After 1st infection
    sus_after_2=[0.7, 0.3, 0.95],        # After 2nd infection
    sus_after_3plus=[0.5, 0.1, 0.9],     # After 3+ infections
)

print("\n" + "-" * 80)
print("Calibration Parameter Ranges:")
print("-" * 80)
for param, (guess, lower, upper) in calib_pars.items():
    print(f"  {param:<20}: [{lower:.3f}, {upper:.3f}] (guess: {guess:.3f})")

# Create analyzer with infection-based severity
analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

# Create immunity connector (will be configured with fitted values during calibration)
immunity_connector = rs.RotaImmunityConnector(
    use_fixed_susceptibility=True,  # Use fixed model with fitted values
    sus_after_1=0.80,  # Initial guess (will be updated during calibration)
    sus_after_2=0.65,  # Initial guess (will be updated during calibration)
    sus_after_3plus=0.50,  # Initial guess (will be updated during calibration)
)

# Create base simulation (will be configured with immunity connector in run_sim)
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
base_sim = rs.Sim(
    n_agents=100000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    people=people,
    analyzers=[analyzer],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[],
    connectors=[immunity_connector],
)

# Create calibration object
n_trials = 20
print(f"\n{'=' * 80}")
print(f"Starting calibration with {n_trials} trials...")
print(f"{'=' * 80}\n")

calib = UKAgeCalibrationFittedImmunity(
    sim=base_sim,
    data=(target_incidence, target_age_distribution),
    calib_pars=calib_pars,
    total_trials=n_trials,
    debug=False,
)

# Run calibration
study = calib.calibrate()

# Extract results
best_pars = study.best_params
best_gof = study.best_value

print("\n" + "=" * 80)
print("CALIBRATION COMPLETE")
print("=" * 80)
print(f"\nBest GOF: {best_gof:.6f}")
print("\nBest parameters:")
for k, v in best_pars.items():
    if k.startswith('sus_'):
        protection = (1 - v) * 100
        print(f"  {k:<20}: {v:.6f} ({protection:.2f}% protection)")
    else:
        print(f"  {k:<20}: {v:.6f}")

# Save calibration results
output_file = thisdir / 'uk_calibration_results_infection_number_fitted_immunity_20trials.json'
results = {
    'model': 'infection_number_fitted_immunity',
    'description': 'Fresh 20-trial calibration with FIXED age distribution extraction',
    'n_trials': n_trials,
    'best_parameters': best_pars,
    'best_gof': best_gof,
    'target_incidence': float(target_incidence),
    'calibration_pars': {k: list(v) for k, v in calib_pars.items()},
}

sc.savejson(output_file, results, indent=2)
print(f"\n✓ Calibration results saved to: {output_file}")

# ============================================================================
# GENERATE FIGURE FOR MLE (BEST MODEL)
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATING BEST MODEL AND GENERATING FIGURE")
print("=" * 80)

print("\nRunning simulation with best parameters...")
# Run simulation with best parameters
sim = calib.run_sim(calib_pars=None, sim_pars=best_pars, trial=None)
overall_incidence, age_distribution = calib.sim_to_df(sim)

print("✓ Simulation complete")

# Display results
print("\nMLE Results:")
print(f"  Incidence: {overall_incidence:.2f} per 100k (target: {target_incidence:.2f})")
print("  Age distribution:")
for i, label in enumerate(age_labels):
    print(f"    {label:<15}: {age_distribution.proportion.iloc[i]*100:>6.2f}% (target: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%)")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Age Distribution
ax1 = axes[0]
x = np.arange(len(age_labels))
width = 0.35

target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(4)]
fitted_props = [age_distribution.proportion.iloc[i] * 100 for i in range(4)]

bars1 = ax1.bar(x - width/2, target_props, width, label='Target (UK Data)',
                color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, fitted_props, width, label='MLE Fit (20 trials)',
                color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_title('A. Age Distribution of Reported Cases', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(['0-11\nmonths', '12-23\nmonths', '24-59\nmonths', '5+\nyears'], fontsize=10)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(max(target_props), max(fitted_props)) * 1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel B: Incidence Comparison
ax2 = axes[1]
incidence_vals = [target_incidence, overall_incidence]
colors = ['#3498DB', '#E74C3C']
bars = ax2.bar(['Target\n(UK Data)', 'MLE Fit\n(20 trials)'],
               incidence_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
ax2.set_title('B. Overall Incidence Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, max(incidence_vals) * 1.3)
ax2.axhline(y=target_incidence, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, incidence_vals):
    height = bar.get_height()
    pct_of_target = (val / target_incidence) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{val:.1f}\n({pct_of_target:.0f}% of target)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add GOF annotation
incidence_gof = abs(overall_incidence - target_incidence) / target_incidence
age_gof = ((age_distribution.proportion - target_age_distribution.proportion).abs() / target_age_distribution.proportion).sum()
error = overall_incidence - target_incidence
error_pct = (error / target_incidence) * 100
ax2.text(0.5, 0.15,
        f'GOF: {best_gof:.4f}\nIncidence error: {error:+.1f} ({error_pct:+.1f}%)\nAge GOF: {age_gof:.4f}',
        transform=ax2.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# Overall title
fig.suptitle('20-Trial Calibration Results: MLE Fit (FIXED Age Extraction)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
figure_file = thisdir / 'calibration_20trials_mle_fit.png'
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved to: {figure_file}")

figure_pdf = thisdir / 'calibration_20trials_mle_fit.pdf'
plt.savefig(figure_pdf, bbox_inches='tight')
print(f"✓ Figure saved to: {figure_pdf}")

# Save detailed results with fit metrics
results['mle_evaluation'] = {
    'incidence': float(overall_incidence),
    'age_distribution': age_distribution.proportion.tolist(),
    'incidence_gof': float(incidence_gof),
    'age_gof': float(age_gof),
}

sc.savejson(output_file, results, indent=2)

print("\n" + "=" * 80)
print("CALIBRATION AND EVALUATION COMPLETE")
print("=" * 80)
print("\nResults:")
print(f"  Best GOF: {best_gof:.6f}")
print(f"  Incidence error: {error:+.1f} per 100k ({error_pct:+.1f}%)")
print(f"  Age GOF: {age_gof:.4f}")
print("\nNext step: Run multi-seed evaluation with best parameters")
print("=" * 80)
