"""
Run 50 simulations with Trial #49 parameters (best fit from original calibration)

This script runs 50 independent simulations using the best-fit parameters
from trial #49 to assess uncertainty and variability in model outputs.

Run on VM with: python run_trial49_50sims.py
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
import json
from datetime import datetime

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("RUNNING 50 SIMULATIONS WITH TRIAL #49 PARAMETERS")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Trial #49 parameters (best fit from original calibration with new GOF)
TRIAL_49_PARAMS = {
    'reporting_rate': 0.0033931311169984376,
    'base_beta': 4.536296913268577,
    'beta0': -0.9843161818324684,
    'beta1': 0.25815046011475173,
    'beta2': -0.008516687036056366,
    'sus_after_1': 0.7997334705973642,
    'sus_after_2': 0.7137361698991934,
    'sus_after_3plus': 0.6315704872942117,
}

print("Trial #49 Parameters:")
print("  Reporting rate: {:.6f}".format(TRIAL_49_PARAMS['reporting_rate']))
print("  Base beta: {:.4f}".format(TRIAL_49_PARAMS['base_beta']))
print("  Age model: beta0={:.4f}, beta1={:.4f}, beta2={:.4f}".format(
    TRIAL_49_PARAMS['beta0'], TRIAL_49_PARAMS['beta1'], TRIAL_49_PARAMS['beta2']))
print("  Susceptibility after 1/2/3+ infections: {:.3f}/{:.3f}/{:.3f}".format(
    TRIAL_49_PARAMS['sus_after_1'], TRIAL_49_PARAMS['sus_after_2'], TRIAL_49_PARAMS['sus_after_3plus']))
print()

# Get target data for comparison
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("Target Data:")
print(f"  Incidence: {target_incidence:.2f} per 100,000")
print("  Age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    print(f"    {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")
print()


def run_simulation(seed, params):
    """Run a single simulation with given parameters and seed"""
    print(f"Running simulation {seed+1}/50 (seed={seed})...", end=' ')

    # Create analyzer
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)

    # Create immunity connector
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)

    # Create simulation with specific seed
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
        rand_seed=seed,  # Set random seed for reproducibility
    )

    # Update base_beta BEFORE initialization
    sim.pars.base_beta = params['base_beta']
    for disease in sim.pars.diseases:
        if isinstance(disease, rs.Rotavirus):
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

    # Store parameters for data extraction
    sim._reporting_rate = params['reporting_rate']
    sim._beta0 = params['beta0']
    sim._beta1 = params['beta1']
    sim._beta2 = params['beta2']

    # Initialize
    sim.init()

    # Update immunity connector AFTER initialization
    immunity_connector = sim.connectors.rotaimmunityconnector
    immunity_connector.pars['use_fixed_susceptibility'] = True
    immunity_connector.pars['sus_after_1'] = params['sus_after_1']
    immunity_connector.pars['sus_after_2'] = params['sus_after_2']
    immunity_connector.pars['sus_after_3plus'] = params['sus_after_3plus']

    # Initialize adult immunity
    immunity_connector.initialize_immunity(
        min_age=18, max_age=125, min_exposures=5, max_exposures=15
    )

    # Run simulation
    sim.run()

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
    overall_incidence, age_distribution = process_incidence_uk_age.process_model(
        df,
        age_counts=age_counts,
        symptom_model='age_and_infection_simple',
        beta0=params['beta0'],
        beta1=params['beta1'],
        beta2=params['beta2'],
        beta3=0,
        reporting_rate=params['reporting_rate']
    )

    print(f"Incidence: {overall_incidence:.2f} per 100k")

    return {
        'seed': seed,
        'incidence': overall_incidence,
        'age_0_1': age_distribution.proportion.iloc[0],
        'age_1_2': age_distribution.proportion.iloc[1],
        'age_2_5': age_distribution.proportion.iloc[2],
        'age_5plus': age_distribution.proportion.iloc[3],
    }


# Run 50 simulations
print("=" * 80)
print("Running 50 simulations...")
print("=" * 80)
print()

results = []
for seed in range(50):
    result = run_simulation(seed, TRIAL_49_PARAMS)
    results.append(result)

# Create results dataframe
df_results = pd.DataFrame(results)

# Compute summary statistics
print()
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

print("Incidence (per 100,000):")
print(f"  Target: {target_incidence:.2f}")
print(f"  Mean:   {df_results['incidence'].mean():.2f} (SD: {df_results['incidence'].std():.2f})")
print(f"  Median: {df_results['incidence'].median():.2f}")
print(f"  Min:    {df_results['incidence'].min():.2f}")
print(f"  Max:    {df_results['incidence'].max():.2f}")
print(f"  25th percentile: {df_results['incidence'].quantile(0.25):.2f}")
print(f"  75th percentile: {df_results['incidence'].quantile(0.75):.2f}")
print()

print("Age Distribution (mean proportions):")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
age_cols = ['age_0_1', 'age_1_2', 'age_2_5', 'age_5plus']
for i, (label, col) in enumerate(zip(age_labels, age_cols)):
    target_prop = target_age_distribution.proportion.iloc[i]
    mean_prop = df_results[col].mean()
    sd_prop = df_results[col].std()
    print(f"  {label:<15}: {mean_prop*100:>6.2f}% (SD: {sd_prop*100:.2f}%) [Target: {target_prop*100:.2f}%]")
print()

# Compute GOF for each simulation
print("GOF Distribution:")
gof_values = []
for _, row in df_results.iterrows():
    # Incidence GOF (log-scale squared error)
    eps = 1e-6
    log_target = np.log(target_incidence + eps)
    log_fitted = np.log(row['incidence'] + eps)
    incidence_gof = (log_target - log_fitted) ** 2

    # Age distribution GOF
    age_gof = 0.0
    for i, col in enumerate(age_cols):
        target_prop = target_age_distribution.proportion.iloc[i]
        fitted_prop = row[col]
        age_gof += (target_prop - fitted_prop) ** 2

    # Total GOF
    total_gof = 10 * age_gof + incidence_gof
    gof_values.append(total_gof)

df_results['gof'] = gof_values

print(f"  Mean:   {np.mean(gof_values):.4f} (SD: {np.std(gof_values):.4f})")
print(f"  Median: {np.median(gof_values):.4f}")
print(f"  Min:    {np.min(gof_values):.4f}")
print(f"  Max:    {np.max(gof_values):.4f}")
print()

# Save results
output_csv = thisdir / 'trial49_50sims_results.csv'
df_results.to_csv(output_csv, index=False)
print(f"Detailed results saved to: {output_csv}")

# Save summary statistics
summary = {
    'trial_number': 49,
    'parameters': TRIAL_49_PARAMS,
    'n_simulations': 50,
    'target_incidence': target_incidence,
    'target_age_distribution': target_age_distribution.proportion.tolist(),
    'incidence_stats': {
        'mean': float(df_results['incidence'].mean()),
        'std': float(df_results['incidence'].std()),
        'median': float(df_results['incidence'].median()),
        'min': float(df_results['incidence'].min()),
        'max': float(df_results['incidence'].max()),
        'q25': float(df_results['incidence'].quantile(0.25)),
        'q75': float(df_results['incidence'].quantile(0.75)),
    },
    'age_distribution_stats': {
        label: {
            'mean': float(df_results[col].mean()),
            'std': float(df_results[col].std()),
        }
        for label, col in zip(age_labels, age_cols)
    },
    'gof_stats': {
        'mean': float(np.mean(gof_values)),
        'std': float(np.std(gof_values)),
        'median': float(np.median(gof_values)),
        'min': float(np.min(gof_values)),
        'max': float(np.max(gof_values)),
    },
    'completed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

output_json = thisdir / 'trial49_50sims_summary.json'
with open(output_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary statistics saved to: {output_json}")

print()
print("=" * 80)
print("DONE")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
