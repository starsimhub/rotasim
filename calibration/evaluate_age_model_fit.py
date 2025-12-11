"""
Evaluate age-based symptom model at MLE parameters with uncertainty quantification

This script runs multiple replicates with different random seeds to assess stochastic variability.
Supports all three symptom models: infection_number, age_only, age_and_infection
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
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# ============================================================
# CONFIGURATION
# ============================================================
# Options:
#   'infection_number' - Infection history affects immunity AND severity
#   'age_and_infection' - Infection history affects immunity AND severity, age affects symptoms
#   'age_and_infection_simple' - Infection history affects immunity only, age affects symptoms
SYMPTOM_MODEL = 'age_and_infection_simple'

# MLE parameters for each model (update these with your fitted values)
MLE_PARAMS = {
    'infection_number': {
        'reporting_rate': 0.001636, #was 0.022754
        'base_beta': 0.792471, #was 0.604223
    },
    'age_and_infection': {
        'beta0': -0.249168,
        'beta1': -0.008747,
        'beta2': -0.491601,
        'reporting_rate': 0.002663,
        'base_beta': 0.844395,
    },
    'age_and_infection_simple': {
        'beta0': -2.0,       # Placeholder - update with fitted values
        'beta1': -0.3,       # Placeholder - update with fitted values
        'beta2': -0.01,      # Placeholder - update with fitted values
        'reporting_rate': 0.01,  # Placeholder - update with fitted values
        'base_beta': 0.6,    # Placeholder - update with fitted values
    },
}

N_REPLICATES = 3  # Reduced for testing
N_AGENTS = 10000  # Reduced for testing

print("="*60)
print(f"Evaluating {SYMPTOM_MODEL.upper()} Model Fit at MLE Parameters")
print("="*60)
print(f"\nMLE Parameters for {SYMPTOM_MODEL}:")
for key, val in MLE_PARAMS[SYMPTOM_MODEL].items():
    print(f"  {key}: {val:.6f}")
print(f"\nRunning {N_REPLICATES} replicate simulations with different random seeds...")
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


def run_single_simulation(params, symptom_model, replicate_num, base_seed=12345):
    """
    Run a single simulation with given parameters and unique random seed

    Args:
        params: Dictionary of model parameters
        symptom_model: 'infection_number', 'age_only', or 'age_and_infection'
        replicate_num: Replicate number (0-indexed)
        base_seed: Base random seed (each replicate gets base_seed + replicate_num)
    """
    print(f"\n  Replicate {replicate_num + 1}/{N_REPLICATES}...")

    # Create unique seed for this replicate
    sim_seed = base_seed + replicate_num
    print(f"    Using random seed: {sim_seed}")

    # Create analyzer with appropriate severity settings based on model
    if symptom_model in ['infection_number', 'age_and_infection']:
        # Use infection-based severity
        analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
    elif symptom_model == 'age_and_infection_simple':
        # Use constant severity (age affects symptoms, not severity)
        analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.05)
    else:
        raise ValueError(f"Unknown symptom model: {symptom_model}")

    # Create simulation with unique seed
    people = ss.People(n_agents=N_AGENTS, age_data=thisdir / 'uk_age_data.csv')

    sim = rs.Sim(
        n_agents=N_AGENTS,
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
        rand_seed=sim_seed,  # Set unique random seed
    )

    # Update parameters BEFORE init (following calibration.py pattern)
    sim.init()

    # Set base_beta AFTER init (following calibration.py lines 162-175)
    if 'base_beta' in params:
        print(f"DEBUG: Setting base_beta={params['base_beta']}")
        sim.pars.base_beta = params['base_beta']
        # Update disease beta parameters
        for disease in sim.diseases.values():
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)
            print(f"DEBUG: Disease {disease.name} beta set to {disease.pars.beta}")

    # Store parameters for use in processing
    sim._reporting_rate = params.get('reporting_rate', 0.01)
    sim._symptom_model = symptom_model
    sim._beta0 = params.get('beta0', 0)
    sim._beta1 = params.get('beta1', 0)
    sim._beta2 = params.get('beta2', 0)
    sim._beta3 = params.get('beta3', 0)

    # Initialize immunity for adults
    sim.connectors.rotaimmunityconnector.initialize_immunity(
        min_age=18, max_age=125, min_exposures=5, max_exposures=15
    )

    # Run simulation
    print(f"DEBUG: Starting simulation from {sim.t.start} to {sim.t.stop}")
    sim.run()
    print(f"DEBUG: Simulation completed. Final time: {sim.t.now}")

    # Extract results
    infected_analyzer = None
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            infected_analyzer = analyzer
            break

    df = infected_analyzer.to_df()

    # DEBUG: Check if infections are being recorded
    print(f"DEBUG: Total infections in dataframe: {len(df)}")
    if len(df) > 0:
        print(f"DEBUG: Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f}")
        print(f"DEBUG: Sample rows:\n{df.head()}")
    else:
        print("DEBUG: NO INFECTIONS RECORDED!")

    # For age-based models, pass reporting_rate to process_model to apply AFTER age filtering
    # For infection_number model, apply reporting filter here BEFORE process_model
    if symptom_model == 'infection_number':
        # Apply severity-based reporting filter for infection_number model
        if 'severity' in df.columns:
            df = calculate_reported_cases(df, sim._reporting_rate)
        reporting_rate_to_pass = None  # Already filtered, don't apply again
    else:
        # For age-based models, pass reporting_rate to process_model
        reporting_rate_to_pass = sim._reporting_rate

    # Extract age-specific population counts
    age_counts = extract_age_specific_population_counts(sim)

    # Process results using age-based model
    overall_incidence, age_distribution = process_incidence_uk_age.process_model(
        df,
        age_counts=age_counts,
        symptom_model=symptom_model,
        beta0=sim._beta0,
        beta1=sim._beta1,
        beta2=sim._beta2,
        beta3=sim._beta3,
        reporting_rate=reporting_rate_to_pass,
        verbose=True  # Enable verbose output for debugging
    )

    # Calculate infections per child per year in children <3 years
    # Get all infections during calibration period (years 5-10)
    df_calib = df[(df['CollectionTime'] < 10) & (df['CollectionTime'] >= 5)].copy()

    # Calculate age in years at infection
    # Use age mapping: <1y, 1-2y, 2-5y, >=5y
    ages_years = sim.people.age.values / 365.25  # Convert days to years

    # Count infections in children <3 years during calibration period
    # Age bins from InfectedStrainStats: '0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+'
    age_bins_under3 = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36']
    infections_under3 = df_calib[df_calib['Age'].isin(age_bins_under3)]

    # Count unique children under 3 at midpoint of calibration (year 7.5 of simulation)
    midpoint_time = 7.5 * 365.25  # Convert to days
    ages_at_midpoint = sim.people.age.values  # Ages are in days in Starsim
    n_children_under3 = ((ages_at_midpoint >= 0) & (ages_at_midpoint < 3 * 365.25)).sum()

    # Calculate infections per child per year
    n_years = 5  # Calibration period is 5 years (2008-2012)
    child_years_under3 = n_children_under3 * n_years

    if child_years_under3 > 0:
        infections_per_child_year_under3 = len(infections_under3) / child_years_under3
    else:
        infections_per_child_year_under3 = 0.0

    return overall_incidence, age_distribution, infections_per_child_year_under3


# Run replicates with different seeds
results = []
params = MLE_PARAMS[SYMPTOM_MODEL]

for i in range(N_REPLICATES):
    overall_inc, age_dist, infections_per_child_year = run_single_simulation(
        params=params,
        symptom_model=SYMPTOM_MODEL,
        replicate_num=i,
        base_seed=12345  # Base seed, each replicate gets +i
    )
    results.append({
        'overall_incidence': overall_inc,
        'age_distribution': age_dist,
        'infections_per_child_year_under3': infections_per_child_year,
        'replicate': i,
    })

print("\n" + "="*60)
print("All replicates complete!")
print("="*60)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Extract results
incidences = [r['overall_incidence'] for r in results]
age_dists = [r['age_distribution'] for r in results]
infections_per_child_year_values = [r['infections_per_child_year_under3'] for r in results]

# Calculate summary statistics for incidence
mean_incidence = np.mean(incidences)
std_incidence = np.std(incidences)
min_incidence = np.min(incidences)
max_incidence = np.max(incidences)

# Calculate summary statistics for infections per child year
mean_infections_per_child_year = np.mean(infections_per_child_year_values)
std_infections_per_child_year = np.std(infections_per_child_year_values)
min_infections_per_child_year = np.min(infections_per_child_year_values)
max_infections_per_child_year = np.max(infections_per_child_year_values)

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
print(f"\nModel: {SYMPTOM_MODEL}")
print(f"\nOverall Incidence (per 100k):")
print(f"  Target:     {target_incidence:.2f}")
print(f"  Mean:       {mean_incidence:.2f} ± {std_incidence:.2f}")
print(f"  Range:      [{min_incidence:.2f}, {max_incidence:.2f}]")
print(f"  Error:      {mean_incidence - target_incidence:+.2f} ({(mean_incidence - target_incidence)/target_incidence*100:+.1f}%)")
print(f"  CV:         {std_incidence/mean_incidence*100:.2f}% (coefficient of variation)")

print(f"\nInfections per child per year (children <3 years):")
print(f"  Mean:       {mean_infections_per_child_year:.3f} ± {std_infections_per_child_year:.3f}")
print(f"  Range:      [{min_infections_per_child_year:.3f}, {max_infections_per_child_year:.3f}]")
print(f"  CV:         {std_infections_per_child_year/mean_infections_per_child_year*100:.2f}%")

print(f"\nAge Distribution (proportions):")
print(f"{'Age':<10} {'Target':<12} {'Mean ± SD':<25} {'Range':<20}")
print("-"*70)
age_labels_map = {0: '0-11mo', 1: '12-23mo', 2: '24-59mo', 5: '5+yr'}
for i, age in enumerate(age_categories):
    target_p = target_age_distribution['proportion'].iloc[i] * 100
    mean_p = mean_age_props[i] * 100
    std_p = std_age_props[i] * 100
    min_p = min_age_props[i] * 100
    max_p = max_age_props[i] * 100
    cv = (std_p / mean_p * 100) if mean_p > 0 else 0
    print(f"{age_labels_map[age]:<10} {target_p:>6.1f}%    {mean_p:>6.1f} ± {std_p:>5.1f}% (CV:{cv:>5.1f}%)   [{min_p:>5.1f}, {max_p:>5.1f}]%")

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

# Plot model with error bar
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
ax2.text(0.02, 0.98, f'Error: {error_pct:+.1f}%\nRange: [{min_incidence:.1f}, {max_incidence:.1f}]\nCV: {std_incidence/mean_incidence*100:.1f}%',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Main title
param_str = ', '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                       for k, v in params.items()])
plt.suptitle(f'Model Fit: {SYMPTOM_MODEL.replace("_", " ").title()} ({param_str[:60]}...)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figure
fig_path = thisdir / f'fit_uncertainty_{SYMPTOM_MODEL}.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved to: {fig_path}")

# Save numerical results
results_dict = {
    'symptom_model': SYMPTOM_MODEL,
    'mle_parameters': params,
    'n_replicates': N_REPLICATES,
    'incidence': {
        'target': float(target_incidence),
        'mean': float(mean_incidence),
        'std': float(std_incidence),
        'min': float(min_incidence),
        'max': float(max_incidence),
        'cv_percent': float(std_incidence/mean_incidence*100),
    },
    'age_distribution': {
        'target': target_age_distribution.to_dict(),
        'mean': mean_age_props.tolist(),
        'std': std_age_props.tolist(),
        'cv_percent': (std_age_props / mean_age_props * 100).tolist(),
    },
    'individual_replicates': {
        'incidence': incidences,
        'age_distributions': [ad.to_dict() for ad in age_dists],
    }
}

results_file = thisdir / f'fit_results_{SYMPTOM_MODEL}.json'
sc.savejson(results_file, results_dict, indent=2)
print(f"✓ Results saved to: {results_file}")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)
print(f"\nKey findings:")
print(f"  - Incidence variability (CV): {std_incidence/mean_incidence*100:.2f}%")
print(f"  - Age distribution variability (mean CV): {np.mean(std_age_props/mean_age_props*100):.2f}%")
if std_incidence == 0:
    print(f"\n⚠ WARNING: Zero variance detected! Random seeds may not be working properly.")
else:
    print(f"\n✓ Stochastic variation detected across replicates")

plt.show()
