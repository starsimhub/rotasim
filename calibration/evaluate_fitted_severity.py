"""
Quick evaluation of fitted severity model to extract age distribution
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import matplotlib.pyplot as plt
import json

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibrate_age_simple_with_severity import UKAgeCalibrationWithSeverity
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Load best parameters
with open(thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json', 'r') as f:
    results = json.load(f)
    best_pars = results['best_parameters']

print("="*60)
print("Evaluating Fitted Severity Model")
print("="*60)
print("\nBest parameters:")
for k, v in best_pars.items():
    print(f"  {k}: {v:.6f}")

# Create analyzer
analyzer = rs.InfectedStrainStats(
    use_infection_based_severity=False, 
    constant_severity=best_pars['constant_severity']
)

# Create sim
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
sim = rs.Sim(
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
    rand_seed=12345,
)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Create calibration object
calib = UKAgeCalibrationWithSeverity(
    sim=sim,
    data=(target_incidence, target_age_distribution),
    calib_pars={},
    total_trials=1,
    debug=False,
    symptom_model='age_and_infection_simple',
)

print("\nRunning simulation with best parameters...")
fitted_sim = calib.run_sim(calib_pars=best_pars, sim_pars=best_pars, trial=None)

print("Extracting results...")
overall_incidence, age_distribution = calib.sim_to_df(fitted_sim)

print(f"\n{'='*60}")
print("Results:")
print(f"{'='*60}")
print(f"\nOverall Incidence:")
print(f"  Target:  {target_incidence:.2f} per 100k")
print(f"  Fitted:  {overall_incidence:.2f} per 100k")
print(f"  Error:   {overall_incidence - target_incidence:+.2f} ({(overall_incidence - target_incidence)/target_incidence*100:+.1f}%)")

print(f"\nAge Distribution:")
print(f"{'Age Group':<15} {'Target':>10} {'Fitted':>10} {'Difference':>12}")
print("-"*50)
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution.proportion.iloc[i]
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Bar chart comparison
ax1 = axes[0]
x = np.arange(len(age_labels))
width = 0.35

target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(4)]
fitted_props = [age_distribution.proportion.iloc[i] * 100 for i in range(4)]

bars1 = ax1.bar(x - width/2, target_props, width, label='Target (Data)', 
                color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, fitted_props, width, label='Fitted Model (Severity=14%)',
                color='#27AE60', alpha=0.7, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_title('A. Age Distribution of Reported Cases', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(['0-11\nmonths', '12-23\nmonths', '24-59\nmonths', '5+\nyears'], fontsize=10)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Panel 2: Incidence comparison
ax2 = axes[1]
incidence_vals = [target_incidence, overall_incidence]
colors = ['#3498DB', '#27AE60']
bars = ax2.bar(['Target\n(Data)', 'Fitted Model\n(Severity=14%)'], 
               incidence_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
ax2.set_title('B. Overall Incidence Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 35)
ax2.axhline(y=target_incidence, color='blue', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, incidence_vals):
    height = bar.get_height()
    pct_of_target = (val / target_incidence) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.1f}\n({pct_of_target:.0f}% of target)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

error = overall_incidence - target_incidence
error_pct = (error / target_incidence) * 100
if abs(error) > 1:
    ax2.text(0.5, target_incidence + 2,
            f'Error: {error:+.1f} per 100k\n({error_pct:+.1f}%)',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = thisdir / 'fitted_severity_age_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved to: {output_file}")

output_pdf = thisdir / 'fitted_severity_age_distribution.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ Figure saved to: {output_pdf}")

# Save results with age distribution
results_updated = results.copy()
results_updated['incidence']['after'] = overall_incidence
results_updated['age_distribution'] = {
    'target': target_props,
    'fitted': fitted_props,
    'labels': age_labels
}

with open(thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json', 'w') as f:
    json.dump(results_updated, f, indent=2)

print(f"✓ Results updated with age distribution data")
print("="*60)
