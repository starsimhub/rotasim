"""Evaluate age_and_infection_simple model with fixed 5% severity"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import json

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("="*60)
print("EVALUATING: age_and_infection_simple (Fixed 5% Severity)")
print("="*60)

# Best-fit parameters from calibration
best_pars = {
    'reporting_rate': 0.0284,
    'base_beta': 2.082,
    'beta0': -1.421,
    'beta1': 0.171,
    'beta2': -0.004516
}

print("\nBest parameters:")
for k, v in best_pars.items():
    print(f"  {k}: {v:.6f}")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Create analyzer with FIXED 5% severity
analyzer = rs.InfectedStrainStats(
    use_infection_based_severity=False,
    constant_severity=0.05  # FIXED at 5%
)

# Create simulation
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

# Set parameters
sim.pars.diseases[0].pars.beta = best_pars['base_beta'] * 0.16 / 3.0
sim.pars.diseases[0].pars.beta0 = best_pars['beta0']
sim.pars.diseases[0].pars.beta1 = best_pars['beta1']
sim.pars.diseases[0].pars.beta2 = best_pars['beta2']

# Set reporting rate in analyzer
analyzer.reporting_rate = best_pars['reporting_rate']

# Initialize adult immunity
print("\nInitializing immunity...")
sim.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=18, max_age=125, min_exposures=5, max_exposures=15
)

# Run simulation
print("\n" + "="*60)
print("Running simulation...")
print("="*60)
sim.run()

# Extract results from analyzer
print("\nProcessing results...")

# Get reported infections by age from analyzer
results_df = analyzer.results
reported_by_age = results_df.groupby('age_group')['n_reported'].sum()

# Calculate overall incidence per 100k
total_population = sim.pars.n_agents
years_simulated = 5  # 2008-2012
total_reported = reported_by_age.sum()
overall_incidence = (total_reported / total_population / years_simulated) * 100000

# Calculate age distribution proportions
age_distribution_props = reported_by_age / reported_by_age.sum()

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nOverall Incidence (per 100,000):")
print(f"  Target:  {target_incidence:.2f}")
print(f"  Fitted:  {overall_incidence:.2f}")
error = overall_incidence - target_incidence
error_pct = (error / target_incidence) * 100
print(f"  Error:   {error:+.2f} ({error_pct:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age Group':<15} {'Target':>10} {'Fitted':>10} {'Difference':>12}")
print("-"*50)
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
age_groups = [0, 1, 2, 5]
for i, (label, age_group) in enumerate(zip(age_labels, age_groups)):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution_props[age_group] if age_group in age_distribution_props.index else 0.0
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

# Calculate GOF metrics
incidence_gof = abs(overall_incidence - target_incidence) / target_incidence

age_gof = 0.0
for i, age_group in enumerate(age_groups):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution_props[age_group] if age_group in age_distribution_props.index else 0.0
    age_gof += abs(fitted_prop - target_prop) / target_prop

total_gof = incidence_gof + age_gof

print(f"\nGoodness of Fit:")
print(f"  Incidence GOF:       {incidence_gof:.4f}")
print(f"  Age distribution GOF: {age_gof:.4f}")
print(f"  Total GOF:           {total_gof:.4f}")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)
