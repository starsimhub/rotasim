"""
Direct evaluation of fitted severity model - no calibration framework
"""
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
print("DIRECT EVALUATION - Fitted Severity Model")
print("="*60)

# Load best parameters
with open(thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json', 'r') as f:
    results = json.load(f)
    best_pars = results['best_parameters']

print("\nBest parameters from calibration:")
for k, v in best_pars.items():
    print(f"  {k}: {v:.6f}")

# Create analyzer with fitted severity
analyzer = rs.InfectedStrainStats(
    use_infection_based_severity=False,
    constant_severity=best_pars['constant_severity']
)

# Create and initialize sim
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
sim = rs.Sim(
    n_agents=100000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=True,
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

# Set base_beta
sim.pars['diseases']['rotavirus']['beta'] = best_pars['base_beta']

# Initialize sim
sim.init()

# Initialize adult immunity
sim.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=18, max_age=125, min_exposures=5, max_exposures=15
)

# Run simulation
print("\n" + "="*60)
print("Running simulation...")
print("="*60)
sim.run()

# Extract infection data
infected_analyzer = sim.analyzers['infectedstrainstats']
df = infected_analyzer.to_df()

print(f"\nTotal infections recorded: {len(df)}")

# Extract age-specific population counts
ages_years = sim.people.age.values
age_counts = {
    '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
    '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
    '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
    '>=5 y': int((ages_years >= 5).sum()),
}

print(f"\nAge-specific population counts:")
for age_group, count in age_counts.items():
    print(f"  {age_group}: {count:,}")

# Process results
overall_incidence, age_distribution = process_incidence_uk_age.process_model(
    df,
    age_counts=age_counts,
    symptom_model='age_and_infection_simple',
    beta0=best_pars['beta0'],
    beta1=best_pars['beta1'],
    beta2=best_pars['beta2'],
    beta3=0,
    reporting_rate=best_pars['reporting_rate']
)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

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
for i, label in enumerate(age_labels):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution.proportion.iloc[i]
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

# Save updated results
results['incidence']['after'] = overall_incidence
results['age_distribution'] = {
    'target': target_age_distribution.proportion.tolist(),
    'fitted': age_distribution.proportion.tolist(),
    'labels': age_labels
}

with open(thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved!")
print("="*60)
