"""
Evaluate age-based symptom model at MLE parameters with uncertainty quantification

This script runs multiple replicates with different random seeds using the calibration framework
to ensure proper parameter handling.
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
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Import the calibration class
import sys
sys.path.insert(0, str(thisdir))
from calibrate_uk_age_model import UKAgeCalibration

# ============================================================
# CONFIGURATION
# ============================================================
SYMPTOM_MODEL = 'age_and_infection_simple'

# MLE parameters for each model
MLE_PARAMS = {
    'infection_number': {
        'reporting_rate': 0.001636,
        'base_beta': 0.792471,
    },
    'age_and_infection': {
        'beta0': -0.249168,
        'beta1': -0.008747,
        'beta2': -0.491601,
        'reporting_rate': 0.002663,
        'base_beta': 0.844395,
    },
    'age_and_infection_simple': {
        'beta0': -2.0,
        'beta1': -0.3,
        'beta2': -0.01,
        'reporting_rate': 0.01,
        'base_beta': 0.6,
    },
}

N_REPLICATES = 3
N_AGENTS = 10000

print("="*60)
print(f"Evaluating {SYMPTOM_MODEL.upper()} Model Fit at MLE Parameters")
print("="*60)
print(f"\nMLE Parameters for {SYMPTOM_MODEL}:")
for key, val in MLE_PARAMS[SYMPTOM_MODEL].items():
    print(f"  {key}: {val:.6f}")
print(f"\nRunning {N_REPLICATES} replicate simulations with different random seeds...")
print("="*60)


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


def run_single_simulation(calib_obj, params, replicate_num, base_seed=12345):
    """
    Run a single simulation using the calibration framework

    Args:
        calib_obj: UKAgeCalibration instance
        params: Dictionary of parameters
        replicate_num: Replicate number (0-indexed)
        base_seed: Base random seed
    """
    print(f"\n  Replicate {replicate_num + 1}/{N_REPLICATES}...")

    # Create unique seed for this replicate
    sim_seed = base_seed + replicate_num
    print(f"    Using random seed: {sim_seed}")

    # Use calibration's run_sim method with parameters
    # This properly handles translate_pars and parameter setting
    sim = calib_obj.run_sim(calib_pars=params, sim_pars=params, trial=None)

    # The sim has already been run by run_sim
    # Extract results using calibration's sim_to_df method
    overall_incidence, age_distribution = calib_obj.sim_to_df(sim)

    # Calculate infections per child per year in children <3 years
    # Access the analyzer directly from the sim
    infected_analyzer = None
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            infected_analyzer = analyzer
            break

    if infected_analyzer:
        df = pd.DataFrame(infected_analyzer.infection_events)

        # Get all infections during calibration period (years 5-10)
        df_calib = df[(df['CollectionTime'] < 10) & (df['CollectionTime'] >= 5)].copy()

        # Count infections in children <3 years during calibration period
        age_bins_under3 = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36']
        infections_under3 = df_calib[df_calib['Age'].isin(age_bins_under3)]

        # Count unique children under 3 at midpoint of calibration
        ages_at_midpoint = sim.people.age.values
        n_children_under3 = ((ages_at_midpoint >= 0) & (ages_at_midpoint < 3 * 365.25)).sum()

        # Calculate infections per child per year
        n_years = 5
        child_years_under3 = n_children_under3 * n_years

        if child_years_under3 > 0:
            infections_per_child_year_under3 = len(infections_under3) / child_years_under3
        else:
            infections_per_child_year_under3 = 0.0
    else:
        infections_per_child_year_under3 = 0.0

    return overall_incidence, age_distribution, infections_per_child_year_under3


# ============================================================
# Create base simulation and calibration object
# ============================================================
# Create analyzer with appropriate severity settings based on model
if SYMPTOM_MODEL in ['infection_number', 'age_and_infection']:
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
elif SYMPTOM_MODEL == 'age_and_infection_simple':
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.05)
else:
    raise ValueError(f"Unknown symptom model: {SYMPTOM_MODEL}")

# Create base sim
people = ss.People(n_agents=N_AGENTS, age_data=thisdir / 'uk_age_data.csv')
base_sim = rs.Sim(
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
)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Create calibration object
calib = UKAgeCalibration(
    sim=base_sim,
    data=(target_incidence, target_age_distribution),
    calib_pars={},  # Not used for evaluation
    total_trials=1,
    debug=False,
    symptom_model=SYMPTOM_MODEL,
)

# ============================================================
# Run replicates
# ============================================================
params = MLE_PARAMS[SYMPTOM_MODEL]
results = []

for i in range(N_REPLICATES):
    overall_inc, age_dist, infections_per_child_year = run_single_simulation(
        calib, params, replicate_num=i, base_seed=12345
    )
    results.append({
        'overall_incidence': overall_inc,
        'age_distribution': age_dist,
        'infections_per_child_year_under3': infections_per_child_year,
        'replicate': i,
    })

# ============================================================
# Calculate summary statistics
# ============================================================
print("\n" + "="*60)
print("All replicates complete!")
print("="*60)

# Extract results
incidence_values = [r['overall_incidence'] for r in results]
infections_per_child_year_values = [r['infections_per_child_year_under3'] for r in results]

# Calculate statistics for overall incidence
mean_incidence = np.mean(incidence_values)
std_incidence = np.std(incidence_values)
min_incidence = np.min(incidence_values)
max_incidence = np.max(incidence_values)

# Calculate statistics for age distribution
age_distributions = [r['age_distribution']['proportion'].values for r in results]
mean_age_dist = np.mean(age_distributions, axis=0)
std_age_dist = np.std(age_distributions, axis=0)

# Calculate statistics for infections per child year
mean_infections_per_child_year = np.mean(infections_per_child_year_values)
std_infections_per_child_year = np.std(infections_per_child_year_values)
min_infections_per_child_year = np.min(infections_per_child_year_values)
max_infections_per_child_year = np.max(infections_per_child_year_values)

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"\nModel: {SYMPTOM_MODEL}")

print(f"\nOverall Incidence (per 100k):")
print(f"  Target:     {target_incidence:.2f}")
print(f"  Mean:       {mean_incidence:.2f} ± {std_incidence:.2f}")
print(f"  Range:      [{min_incidence:.2f}, {max_incidence:.2f}]")
print(f"  Error:      {mean_incidence - target_incidence:+.2f} ({(mean_incidence - target_incidence)/target_incidence*100:+.1f}%)")
if mean_incidence > 0:
    print(f"  CV:         {std_incidence/mean_incidence*100:.2f}% (coefficient of variation)")
else:
    print(f"  CV:         nan% (coefficient of variation)")

print(f"\nInfections per child per year (children <3 years):")
print(f"  Mean:       {mean_infections_per_child_year:.3f} ± {std_infections_per_child_year:.3f}")
print(f"  Range:      [{min_infections_per_child_year:.3f}, {max_infections_per_child_year:.3f}]")
if mean_infections_per_child_year > 0:
    print(f"  CV:         {std_infections_per_child_year/mean_infections_per_child_year*100:.2f}%")
else:
    print(f"  CV:         nan%")

print(f"\nAge Distribution (proportions):")
print(f"{'Age':<12} {'Target':>10} {'Mean ± SD':>25} {'Range':>20}")
print("-"*70)
age_labels = ['0-11mo', '12-23mo', '24-59mo', '5+yr']
target_props = target_age_distribution['proportion'].values
for i, label in enumerate(age_labels):
    target = target_props[i] * 100
    mean_val = mean_age_dist[i] * 100
    std_val = std_age_dist[i] * 100
    min_val = min([ad[i] for ad in age_distributions]) * 100
    max_val = max([ad[i] for ad in age_distributions]) * 100
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
    print(f"{label:<12} {target:>7.1f}% {mean_val:>7.1f} ± {std_val:>5.1f}% (CV: {cv:>5.1f}%)   [{min_val:>5.1f}, {max_val:>5.1f}]%")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)

# Check for zero variance warning
if std_incidence == 0 or np.all(std_age_dist == 0):
    print("\n⚠ WARNING: Zero variance detected! Random seeds may not be working properly.")
