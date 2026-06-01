"""
Plot best fit results from hybrid calibration (50 trials)

Shows comparison between model output and target data for:
1. Overall incidence (per 100k)
2. Age distribution of cases
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import rotasim as rs
import matplotlib.pyplot as plt
import json

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("PLOTTING BEST FIT FROM HYBRID CALIBRATION")
print("=" * 80)

# Load best parameters from JSON
with open(thisdir / 'calibration_50trials_results.json', 'r') as f:
    results = json.load(f)

best_params = results['best_params']
print(f"\nBest Trial: #{results['best_trial_number']}")
print(f"Best GOF: {results['best_gof']:.4f}")
print("\nBest Parameters:")
for key, value in best_params.items():
    if key.startswith('sus_'):
        protection = (1 - value) * 100
        print(f"  {key:<20}: {value:.6f} ({protection:.2f}% protection)")
    else:
        print(f"  {key:<20}: {value:.6f}")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print(f"\nTarget Data:")
print(f"  Incidence: {target_incidence:.2f} per 100,000")
print("  Age distribution:")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    print(f"    {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Create and run simulation with best parameters
print("\n" + "=" * 80)
print("Running simulation with best parameters...")
print("=" * 80)

analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)
immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)

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
    connectors=[immunity_connector],
)

# Update base_beta BEFORE initialization
sim.pars.base_beta = best_params['base_beta']
for disease in sim.pars.diseases:
    if isinstance(disease, rs.Rotavirus):
        disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

# Store parameters for data extraction
sim._reporting_rate = best_params['reporting_rate']
sim._beta0 = best_params['beta0']
sim._beta1 = best_params['beta1']
sim._beta2 = best_params['beta2']

# Initialize
sim.init()

# Update immunity connector AFTER initialization
immunity_connector = sim.connectors.rotaimmunityconnector
immunity_connector.pars['use_fixed_susceptibility'] = True
immunity_connector.pars['sus_after_1'] = best_params['sus_after_1']
immunity_connector.pars['sus_after_2'] = best_params['sus_after_2']
immunity_connector.pars['sus_after_3plus'] = best_params['sus_after_3plus']

# Initialize adult immunity
immunity_connector.initialize_immunity(
    min_age=18, max_age=125, min_exposures=5, max_exposures=15
)

# Run simulation
print("Running simulation...")
sim.run()
print("Simulation complete!")

# Extract results
df = sim.analyzers['infectedstrainstats'].to_df()
df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

# Get age counts
ages_years = sim.people.age.values
age_counts = {
    '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
    '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
    '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
    '>=5 y': int((ages_years >= 5).sum()),
}

# Process results
model_incidence, model_age_distribution = process_incidence_uk_age.process_model(
    df,
    age_counts=age_counts,
    symptom_model='age_and_infection_simple',
    beta0=best_params['beta0'],
    beta1=best_params['beta1'],
    beta2=best_params['beta2'],
    beta3=0,
    reporting_rate=best_params['reporting_rate']
)

print(f"\nModel Output:")
print(f"  Incidence: {model_incidence:.2f} per 100,000")
print("  Age distribution:")
for i, label in enumerate(age_labels):
    print(f"    {label:<15}: {model_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Calculate GOF components
eps = 1e-6
log_target = np.log(target_incidence + eps)
log_fitted = np.log(model_incidence + eps)
incidence_gof = (log_target - log_fitted) ** 2

age_gof = 0.0
for i in range(len(target_age_distribution)):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = model_age_distribution.proportion.iloc[i]
    age_gof += (target_prop - fitted_prop) ** 2

total_gof = 10 * age_gof + incidence_gof

print(f"\nGOF Components:")
print(f"  Incidence GOF: {incidence_gof:.4f}")
print(f"  Age GOF: {age_gof:.4f}")
print(f"  Total GOF: {total_gof:.4f}")
print(f"  (Expected: {results['best_gof']:.4f})")

# Create plots
print("\n" + "=" * 80)
print("Creating plots...")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Incidence comparison
ax = axes[0]
x_pos = [0, 1]
incidence_values = [target_incidence, model_incidence]
colors = ['#2E86AB', '#A23B72']

bars = ax.bar(x_pos, incidence_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Incidence per 100,000', fontsize=12, fontweight='bold')
ax.set_title('Overall Incidence Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Target\n(UK Data)', 'Model\n(Best Fit)'], fontsize=11)
ax.set_ylim(0, max(incidence_values) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, incidence_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(incidence_values)*0.02,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add GOF annotation
ax.text(0.98, 0.98, f'Incidence GOF: {incidence_gof:.4f}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Age distribution comparison
ax = axes[1]
x = np.arange(len(age_labels))
width = 0.35

target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(len(age_labels))]
model_props = [model_age_distribution.proportion.iloc[i] * 100 for i in range(len(age_labels))]

bars1 = ax.bar(x - width/2, target_props, width, label='Target (UK Data)',
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, model_props, width, label='Model (Best Fit)',
               color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax.set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_labels, fontsize=10, rotation=15, ha='right')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(target_props), max(model_props)) * 1.2)

# Add value labels on bars
def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

autolabel(bars1, ax)
autolabel(bars2, ax)

# Add GOF annotation
ax.text(0.98, 0.98, f'Age GOF: {age_gof:.4f}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add overall GOF to figure
fig.text(0.5, 0.02, f'Total GOF = 10 × Age GOF + Incidence GOF = {total_gof:.4f}',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Save figure
output_file = thisdir / 'best_fit_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also save a summary table
summary_data = {
    'Metric': ['Incidence (per 100k)', 'Age 0-11m (%)', 'Age 12-23m (%)', 'Age 24-59m (%)', 'Age 5+ years (%)'],
    'Target': [target_incidence] + [target_age_distribution.proportion.iloc[i]*100 for i in range(4)],
    'Model': [model_incidence] + [model_age_distribution.proportion.iloc[i]*100 for i in range(4)],
}
summary_df = pd.DataFrame(summary_data)
summary_df['Difference'] = summary_df['Model'] - summary_df['Target']
summary_df['Relative Error (%)'] = (summary_df['Difference'] / summary_df['Target'] * 100).round(2)

summary_file = thisdir / 'best_fit_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"Summary table saved to: {summary_file}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
