"""
Diagnose why the calibration is producing incorrect results
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
print("DIAGNOSING CALIBRATION ISSUE")
print("="*80)

# Use best-fit parameters from the calibration
best_pars = {
    'reporting_rate': 0.9298843758582356,
    'homotypic_immunity_efficacy': 0.1423948249459815,
    'partial_heterotypic_immunity_efficacy': 0.06528308706866637,
    'complete_heterotypic_immunity_efficacy': 0.19610783473184937,
    'base_beta': 0.263304768412464,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9328929709256558
}

print("\nCreating simulation with best-fit parameters...")
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',  # 5-year burn-in
    stop='2013-01-01',   # 10 years total
    verbose=False,
    scenario='single',
    base_beta=best_pars['base_beta'],
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            homotypic_immunity_efficacy=best_pars['homotypic_immunity_efficacy'],
            partial_heterotypic_immunity_efficacy=best_pars['partial_heterotypic_immunity_efficacy'],
            complete_heterotypic_immunity_efficacy=best_pars['complete_heterotypic_immunity_efficacy'],
            maternal_immunity_efficacy=0.0,
            adult_baseline_immunity=best_pars['adult_baseline_immunity'],
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

print("Initializing simulation...")
sim.init()
initialize_uk_ages(sim)
initialize_adult_immunity(sim, adult_baseline_immunity=best_pars['adult_baseline_immunity'])
seed_infections_by_age(sim, overall_prevalence=0.002)

print("Running simulation...")
sim.run()
print("✓ Simulation complete")

# Get infection data
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()
print(f"\nTotal infections recorded: {len(df)}")

# Check the time range of infections
print(f"\nTime range of infections:")
print(f"  Start date: {sim.pars.start}")
print(f"  Stop date: {sim.pars.stop}")
print(f"  First infection: day {df['t'].min()}")
print(f"  Last infection: day {df['t'].max()}")

# Check if there's a year column
if 'year' in df.columns:
    print(f"\nInfections by year:")
    print(df['year'].value_counts().sort_index())

# Check age distribution
print(f"\nAge column values:")
print(df['Age'].value_counts().sort_index())

# Check severity
print(f"\nSeverity distribution:")
print(f"  Min: {df['severity'].min():.4f}")
print(f"  Max: {df['severity'].max():.4f}")
print(f"  Mean: {df['severity'].mean():.4f}")

# Apply severity-based reporting
reporting_rate = best_pars['reporting_rate']
print(f"\nApplying severity-based reporting (rate={reporting_rate:.3f})...")
reported_df = calculate_reported_cases(df, reporting_rate)
print(f"Reported cases: {len(reported_df)} ({len(reported_df)/len(df)*100:.1f}% of all infections)")

# Process through process_incidence_uk
print("\nProcessing through process_incidence_uk.process_model()...")
overall_incidence, age_distribution = process_incidence_uk.process_model(reported_df)

print(f"\nResults:")
print(f"  Overall incidence: {overall_incidence:.1f} per 100k")
print(f"  Target incidence:  1.4 per 100k")
print(f"\n  Age distribution:")
print(age_distribution)

# Check what ages are in the reported cases
print(f"\nReported cases age breakdown:")
for age_code in [0, 1, 2, 5]:
    count = (age_distribution['ages'] == age_code).sum()
    if count > 0:
        prop = age_distribution[age_distribution['ages'] == age_code]['proportion'].values[0]
        print(f"  Age {age_code}: {prop*100:.1f}%")

# Let's also check the raw Age values
print(f"\nRaw Age values in reported cases:")
print(reported_df['Age'].value_counts())

print("\n" + "="*80)
print("ISSUE IDENTIFICATION")
print("="*80)

# Check if the issue is with time filtering
calibration_start_year = 2008
calibration_end_year = 2012
if 'year' in reported_df.columns:
    calib_period = reported_df[(reported_df['year'] >= calibration_start_year) &
                                (reported_df['year'] <= calibration_end_year)]
    print(f"\nInfections in calibration period (2008-2012): {len(calib_period)}")
    print(f"Infections outside calibration period: {len(reported_df) - len(calib_period)}")
else:
    print("\nNo 'year' column found - may be counting all infections!")

print("="*80)
