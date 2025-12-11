"""
Run multiple simulations at MLE parameters and create summary figures
This shows the variability in model predictions at the best-fit parameters
"""

# Fix for PyCharm: Remove parent directory from sys.path
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import matplotlib.pyplot as plt
import pandas as pd

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

# MLE parameters from calibration
MLE_PARAMS = {
    'reporting_rate': 0.022754,
    'base_beta': 0.604223,
}

N_REPLICATES = 10
N_AGENTS = 50000

print("="*60)
print("Evaluating Model Fit at MLE Parameters")
print("="*60)
print(f"\nMLE Parameters:")
print(f"  reporting_rate: {MLE_PARAMS['reporting_rate']:.6f}")
print(f"  base_beta: {MLE_PARAMS['base_beta']:.6f}")
print(f"\nRunning {N_REPLICATES} replicate simulations...")
print("="*60)


def calculate_reported_cases(df, reporting_rate):
    """Calculate reported cases using severity-based reporting"""
    df['reported'] = np.random.random(len(df)) < (reporting_rate * df['severity'])
    reported_df = df[df['reported']].copy()
    return reported_df


def extract_age_specific_population_counts(sim):
    """Extract actual age-specific population counts from simulation"""
    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }
    return age_counts


def run_single_simulation(reporting_rate, base_beta, replicate_num):
    """Run a single simulation with given parameters"""
    print(f"\n  Replicate {replicate_num + 1}/{N_REPLICATES}...")

    # Create simulation
    people = ss.People(n_agents=N_AGENTS, age_data=thisdir / 'uk_age_data.csv')
    sim = rs.Sim(
        n_agents=N_AGENTS,
        start='2003-01-01',
        stop='2013-01-01',
        verbose=False,
        scenario='single',
        people=people,
        analyzers=[rs.InfectedStrainStats()],
        networks=ss.RandomNet(n_contacts=7),
        demographics=[
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
        interventions=[],
    )

    # Update parameters
    sim.init()
    for disease in sim.diseases.values():
        disease.pars.beta = ss.perday(base_beta)
    sim._reporting_rate = reporting_rate

    # Initialize immunity
    sim.connectors.rotaimmunityconnector.initialize_immunity(
        min_age=18, max_age=125, min_exposures=5, max_exposures=15
    )

    # Run simulation
    sim.run()

    # Extract results
    infected_analyzer = None
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            infected_analyzer = analyzer
            break

    df = infected_analyzer.to_df()

    # Apply severity-based reporting
    if 'severity' in df.columns:
        df = calculate_reported_cases(df, reporting_rate)

    # Extract age-specific population counts
    age_counts = extract_age_specific_population_counts(sim)

    # Process results
    overall_incidence, age_distribution = process_incidence_uk.process_model(df, age_counts=age_counts)

    return overall_incidence, age_distribution


# Run replicates
results = []
for i in range(N_REPLICATES):
    overall_inc, age_dist = run_single_simulation(
        MLE_PARAMS['reporting_rate'],
        MLE_PARAMS['base_beta'],
        i
    )
    results.append({
        'overall_incidence': overall_inc,
        'age_distribution': age_dist
    })

print("\n" + "="*60)
print("All replicates complete!")
print("="*60)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk.process_data()

# Extract results
incidences = [r['overall_incidence'] for r in results]
age_dists = [r['age_distribution'] for r in results]

# Calculate summary statistics
mean_incidence = np.mean(incidences)
std_incidence = np.std(incidences)
min_incidence = np.min(incidences)
max_incidence = np.max(incidences)

# Age distribution statistics
age_categories = target_age_distribution['ages'].values
n_ages = len(age_categories)
age_props_matrix = np.array([ad['proportion'].values for ad in age_dists])
mean_age_props = np.mean(age_props_matrix, axis=0)
std_age_props = np.std(age_props_matrix, axis=0)
min_age_props = np.min(age_props_matrix, axis=0)
max_age_props = np.max(age_props_matrix, axis=0)

# Print summary
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"\nOverall Incidence (per 100k):")
print(f"  Target:     {target_incidence:.2f}")
print(f"  Mean:       {mean_incidence:.2f} ± {std_incidence:.2f}")
print(f"  Range:      [{min_incidence:.2f}, {max_incidence:.2f}]")
print(f"  Error:      {mean_incidence - target_incidence:+.2f} ({(mean_incidence - target_incidence)/target_incidence*100:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age':<10} {'Target':<12} {'Mean ± SD':<20} {'Range':<20}")
print("-"*65)
age_labels_map = {0: '0-11mo', 1: '12-23mo', 2: '24-59mo', 5: '5+yr'}
for i, age in enumerate(age_categories):
    target_p = target_age_distribution['proportion'].iloc[i] * 100
    mean_p = mean_age_props[i] * 100
    std_p = std_age_props[i] * 100
    min_p = min_age_props[i] * 100
    max_p = max_age_props[i] * 100
    print(f"{age_labels_map[age]:<10} {target_p:<12.1f}% {mean_p:.1f} ± {std_p:.1f}%      [{min_p:.1f}, {max_p:.1f}]%")

# Create figure
print("\n" + "="*60)
print("Creating summary figure...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Age distribution
ax1 = axes[0]
x_positions = np.arange(n_ages)
width = 0.35

target_props = target_age_distribution['proportion'].values * 100

# Plot target
ax1.bar(x_positions - width/2, target_props, width, label='Target (NHS Wales)',
        color='black', alpha=0.7)

# Plot model mean with error bars
ax1.bar(x_positions + width/2, mean_age_props * 100, width,
        yerr=std_age_props * 100, label='Model (mean ± SD)',
        color='steelblue', alpha=0.7, capsize=5)

ax1.set_xlabel('Age Group', fontsize=12)
ax1.set_ylabel('Proportion (%)', fontsize=12)
ax1.set_title('Age Distribution of Cases', fontsize=14, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels([age_labels_map[age] for age in age_categories])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add GOF text
age_gof = np.mean(np.abs(mean_age_props - target_age_distribution['proportion'].values))
ax1.text(0.02, 0.98, f'GOF: {age_gof:.3f}\nN={N_REPLICATES} replicates',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right panel: Overall incidence
ax2 = axes[1]

# Plot target
ax2.bar([0], [target_incidence], color='black', alpha=0.7, label='Target')

# Plot model with error bar (asymmetric: [lower, upper])
lower_err = abs(mean_incidence - min_incidence)
upper_err = abs(max_incidence - mean_incidence)
ax2.bar([1], [mean_incidence], yerr=[[lower_err], [upper_err]],
        color='steelblue', alpha=0.7, label='Model', capsize=10)

ax2.set_ylabel('Incidence (per 100k per year)', fontsize=12)
ax2.set_title('Overall Incidence', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Target\n(NHS Wales)', f'Model\n(N={N_REPLICATES})'])
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Add values on bars
ax2.text(0, target_incidence, f'{target_incidence:.1f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.text(1, mean_incidence, f'{mean_incidence:.1f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add error text
error_pct = (mean_incidence - target_incidence) / target_incidence * 100
ax2.text(0.02, 0.98, f'Error: {error_pct:+.1f}%\nRange: [{min_incidence:.1f}, {max_incidence:.1f}]',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Main title
plt.suptitle(f'Model Fit at MLE Parameters (β={MLE_PARAMS["base_beta"]:.3f}, r={MLE_PARAMS["reporting_rate"]:.6f})',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figure
fig_path = thisdir / 'mle_fit_uncertainty.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved to: {fig_path}")

# Also save numerical results
results_dict = {
    'mle_parameters': MLE_PARAMS,
    'n_replicates': N_REPLICATES,
    'incidence': {
        'target': float(target_incidence),
        'mean': float(mean_incidence),
        'std': float(std_incidence),
        'min': float(min_incidence),
        'max': float(max_incidence),
    },
    'age_distribution': {
        'target': target_age_distribution.to_dict(),
        'mean': mean_age_props.tolist(),
        'std': std_age_props.tolist(),
    }
}

results_file = thisdir / 'mle_fit_results.json'
sc.savejson(results_file, results_dict, indent=2)
print(f"✓ Results saved to: {results_file}")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)

plt.show()
