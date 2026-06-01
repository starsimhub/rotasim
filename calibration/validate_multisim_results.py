"""
Validate best fit from MultiSim calibration by running multiple replicates

This script:
1. Loads the best parameters from calibration_multisim_results.json
2. Runs 50 simulations with those parameters
3. Computes GOF distribution and summary statistics
4. Creates visualization comparing target vs model output

Run with: python validate_multisim_results.py
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
from datetime import datetime

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("VALIDATING MULTISIM CALIBRATION RESULTS")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load best parameters from JSON
with open(thisdir / 'calibration_multisim_results.json', 'r') as f:
    results = json.load(f)

best_params = results['best_params']
print(f"Best Trial: #{results['best_trial_number']}")
print(f"Best GOF (median from calibration): {results['best_gof']:.4f}")
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
print("\n" + "=" * 80)
print("Running 50 validation simulations...")
print("=" * 80)
print()

validation_results = []
for seed in range(50):
    result = run_simulation(seed, best_params)
    validation_results.append(result)

# Create results dataframe
df_results = pd.DataFrame(validation_results)

# Compute summary statistics
print()
print("=" * 80)
print("VALIDATION RESULTS SUMMARY")
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
age_cols = ['age_0_1', 'age_1_2', 'age_2_5', 'age_5plus']
for i, (label, col) in enumerate(zip(age_labels, age_cols)):
    target_prop = target_age_distribution.proportion.iloc[i]
    mean_prop = df_results[col].mean()
    sd_prop = df_results[col].std()
    median_prop = df_results[col].median()
    print(f"  {label:<15}: Mean={mean_prop*100:>6.2f}% (SD={sd_prop*100:.2f}%), Median={median_prop*100:>6.2f}% [Target: {target_prop*100:.2f}%]")
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
print(f"  Median: {np.median(gof_values):.4f} (Expected from calibration: {results['best_gof']:.4f})")
print(f"  Min:    {np.min(gof_values):.4f}")
print(f"  Max:    {np.max(gof_values):.4f}")
print(f"  25th percentile: {np.percentile(gof_values, 25):.4f}")
print(f"  75th percentile: {np.percentile(gof_values, 75):.4f}")
print()

# Save results
output_csv = thisdir / 'multisim_validation_results.csv'
df_results.to_csv(output_csv, index=False)
print(f"Detailed results saved to: {output_csv}")

# Create plots
print("\n" + "=" * 80)
print("Creating validation plots...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Incidence comparison with distribution
ax = axes[0, 0]
x_pos = [0, 1]
incidence_values = [target_incidence, df_results['incidence'].median()]
colors = ['#2E86AB', '#A23B72']

bars = ax.bar(x_pos, incidence_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.errorbar([1], [df_results['incidence'].median()],
            yerr=[[df_results['incidence'].median() - df_results['incidence'].quantile(0.25)],
                  [df_results['incidence'].quantile(0.75) - df_results['incidence'].median()]],
            fmt='none', color='black', capsize=10, linewidth=2)
ax.set_ylabel('Incidence per 100,000', fontsize=12, fontweight='bold')
ax.set_title('Overall Incidence Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Target\n(UK Data)', 'Model\n(Median ± IQR)'], fontsize=11)
ax.set_ylim(0, max(incidence_values) * 1.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, incidence_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(incidence_values)*0.02,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Age distribution comparison with error bars
ax = axes[0, 1]
x = np.arange(len(age_labels))
width = 0.35

target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(len(age_labels))]
model_props_median = [df_results[col].median() * 100 for col in age_cols]
model_props_q25 = [df_results[col].quantile(0.25) * 100 for col in age_cols]
model_props_q75 = [df_results[col].quantile(0.75) * 100 for col in age_cols]

bars1 = ax.bar(x - width/2, target_props, width, label='Target (UK Data)',
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, model_props_median, width, label='Model (Median)',
               color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars for IQR
ax.errorbar(x + width/2, model_props_median,
            yerr=[np.array(model_props_median) - np.array(model_props_q25),
                  np.array(model_props_q75) - np.array(model_props_median)],
            fmt='none', color='black', capsize=5, linewidth=1.5)

ax.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax.set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_labels, fontsize=10, rotation=15, ha='right')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(target_props), max(model_props_q75)) * 1.2)

# Plot 3: GOF distribution histogram
ax = axes[1, 0]
ax.hist(gof_values, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
ax.axvline(np.median(gof_values), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(gof_values):.4f}')
ax.axvline(results['best_gof'], color='blue', linestyle='--', linewidth=2, label=f'Calibration: {results["best_gof"]:.4f}')
ax.set_xlabel('GOF', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('GOF Distribution Across 50 Replicates', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 4: Incidence distribution histogram
ax = axes[1, 1]
ax.hist(df_results['incidence'], bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
ax.axvline(df_results['incidence'].median(), color='red', linestyle='--', linewidth=2,
           label=f'Median: {df_results["incidence"].median():.1f}')
ax.axvline(target_incidence, color='blue', linestyle='--', linewidth=2, label=f'Target: {target_incidence:.1f}')
ax.set_xlabel('Incidence per 100,000', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Incidence Distribution Across 50 Replicates', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
output_file = thisdir / 'multisim_validation_plots.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlots saved to: {output_file}")

print("\n" + "=" * 80)
print("DONE")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
