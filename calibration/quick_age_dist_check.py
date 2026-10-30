"""
Quick check: Run ONE simulation with best-fit parameters and show age distribution
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
from calibrate_uk import initialize_uk_ages, initialize_adult_immunity, seed_infections_by_age, calculate_reported_cases
import process_incidence_uk

print("="*80)
print("SINGLE SIMULATION - AGE DISTRIBUTION CHECK")
print("="*80)

# Best-fit parameters
reporting_rate = 0.397
homotypic_immunity_efficacy = 0.489
partial_heterotypic_immunity_efficacy = 0.235
complete_heterotypic_immunity_efficacy = 0.284
base_beta = 0.291
adult_baseline_immunity = 0.916

print("\nBest-fit parameters:")
print(f"  reporting_rate: {reporting_rate:.3f}")
print(f"  base_beta: {base_beta:.3f}")
print(f"  adult_baseline_immunity: {adult_baseline_immunity:.3f}")
print(f"  homotypic_immunity_efficacy: {homotypic_immunity_efficacy:.3f}")

# Create simulation
print("\nCreating simulation...")
sim = rs.Sim(
    n_agents=50000,
    start='2008-01-01',
    stop='2013-01-01',  # 5 years
    verbose=False,
    scenario='single',
    base_beta=base_beta,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            homotypic_immunity_efficacy=homotypic_immunity_efficacy,
            partial_heterotypic_immunity_efficacy=partial_heterotypic_immunity_efficacy,
            complete_heterotypic_immunity_efficacy=complete_heterotypic_immunity_efficacy,
            maternal_immunity_efficacy=0.0,
            adult_baseline_immunity=adult_baseline_immunity,
        )
    ],
    interventions=[
        rs.InitializeChildImmunity(
            max_age_years=3.0,
            min_infections=1,
            max_infections=1,
            verbose=False
        )
    ],
)

# Initialize
print("Initializing...")
sim.init()
initialize_uk_ages(sim)
initialize_adult_immunity(sim, adult_baseline_immunity=adult_baseline_immunity)
seed_infections_by_age(sim, overall_prevalence=0.002)

# Run
print("Running simulation...")
sim.run()
print("✓ Simulation complete")

# Extract infection data
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()
print(f"\nTotal infections: {len(df)}")

# Apply severity-based reporting
print(f"\nApplying severity-based reporting (rate={reporting_rate:.3f})...")
reported_df = calculate_reported_cases(df, reporting_rate)
print(f"Reported cases: {len(reported_df)}")

# Calculate age distribution
overall_incidence, age_distribution = process_incidence_uk.process_model(reported_df)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nOverall incidence: {overall_incidence:.2f} per 100k")
print(f"Target incidence:  1.40 per 100k")

print("\n" + "="*80)
print("AGE DISTRIBUTION COMPARISON")
print("="*80)

# Target data
target = {
    0: 0.137698,
    1: 0.276685,
    2: 0.468881,
    5: 0.116737
}

age_labels = {
    0: '0-11 months',
    1: '12-23 months',
    2: '24-59 months',
    5: '5+ years'
}

print(f"\n{'Age Group':<16} {'Target %':<12} {'Model %':<12} {'Difference':<12}")
print("-" * 56)

for age_code in [0, 1, 2, 5]:
    target_prop = target[age_code] * 100

    # Find model proportion
    model_row = age_distribution[age_distribution['ages'] == age_code]
    if len(model_row) > 0:
        model_prop = model_row['proportion'].values[0] * 100
    else:
        model_prop = 0.0

    diff = model_prop - target_prop

    age_label = age_labels[age_code]
    print(f"{age_label:<16} {target_prop:>10.2f}%  {model_prop:>10.2f}%  {diff:>+10.2f}%")

print("\n" + "="*80)
