"""
Count infections diagnostic - Run ONE simulation and examine infection counts
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
from calibrate_uk import initialize_uk_ages, initialize_adult_immunity, seed_infections_by_age

print("=" * 80)
print("INFECTION COUNT DIAGNOSTIC")
print("=" * 80)

# Use calibrated parameters from the last run
best_pars = {
    'reporting_rate': 0.9298843758582356,
    'homotypic_immunity_efficacy': 0.1423948249459815,
    'partial_heterotypic_immunity_efficacy': 0.06528308706866637,
    'complete_heterotypic_immunity_efficacy': 0.19610783473184937,
    'base_beta': 0.263304768412464,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9328929709256558
}

print("\nSimulation parameters:")
print(f"  Population: 5,000 agents")
print(f"  Duration: 2003-01-01 to 2013-01-01 (10 years)")
print(f"  Calibration period: Years 5-9 (2008-2012)")
print(f"  Base beta: {best_pars['base_beta']:.3f}")
print(f"  Reporting rate: {best_pars['reporting_rate']:.3f}")

# Create simulation
print("\nCreating simulation...")
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
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

print("Initializing...")
sim.init()
initialize_uk_ages(sim)
initialize_adult_immunity(sim, adult_baseline_immunity=best_pars['adult_baseline_immunity'])
seed_infections_by_age(sim, overall_prevalence=0.002)

print("Running simulation...")
sim.run()
print("✓ Complete")

# Get infection data
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()

print("\n" + "=" * 80)
print("INFECTION COUNTS")
print("=" * 80)

print(f"\nTotal infection events recorded: {len(df)}")
print(f"Unique agents infected: {df['uid'].nunique()}")
print(f"Population size: {sim.pars.n_agents}")

# Check time distribution
if 'CollectionTime' in df.columns:
    df['Year'] = np.floor(df['CollectionTime']).astype(int)
    print(f"\nInfections by year:")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  Year {year}: {count} infections")

    # Calibration period (years 5-9)
    calib_df = df[(df['Year'] >= 5) & (df['Year'] <= 9)]
    print(f"\nInfections in calibration period (years 5-9): {len(calib_df)}")
    print(f"  Per year (average): {len(calib_df) / 5.0:.1f}")

# Check multiple infections per person
infections_per_person = df['uid'].value_counts()
print(f"\nInfections per person:")
print(f"  Mean: {infections_per_person.mean():.2f}")
print(f"  Median: {infections_per_person.median():.0f}")
print(f"  Max: {infections_per_person.max()}")
print(f"\nDistribution of infections per person:")
for n_inf in sorted(infections_per_person.unique())[:10]:  # Show first 10
    count = (infections_per_person == n_inf).sum()
    pct = count / len(infections_per_person) * 100
    print(f"  {n_inf} infection(s): {count} people ({pct:.1f}%)")

# Expected vs actual
print("\n" + "=" * 80)
print("EXPECTED VS ACTUAL")
print("=" * 80)

# Expected: 1.4 per 100k per year for 5 years
expected_per_year = (5000 * 1.4 / 100000)
expected_total = expected_per_year * 5
print(f"\nExpected (based on target 1.4 per 100k):")
print(f"  Per year: {expected_per_year:.1f} infections")
print(f"  Over 5 years: {expected_total:.1f} infections")

if 'Year' in df.columns:
    actual_total = len(calib_df)
    actual_per_year = actual_total / 5.0
    print(f"\nActual (calibration period):")
    print(f"  Per year: {actual_per_year:.1f} infections")
    print(f"  Over 5 years: {actual_total} infections")
    print(f"\nRatio (actual/expected): {actual_total / expected_total:.1f}x")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if 'Year' in df.columns and len(calib_df) > 0:
    ratio = len(calib_df) / expected_total
    if ratio > 1000:
        print(f"\n⚠ CRITICAL: Model is producing {ratio:.0f}x too many infections!")
        print("\nLikely causes:")
        print("  1. base_beta is WAY too high")
        print("  2. Adult immunity is not working properly")
        print("  3. Immunity waning is too fast")
        print("  4. Force of infection calculation is wrong")
    elif ratio > 100:
        print(f"\n⚠ Model is producing {ratio:.0f}x too many infections")
        print("\nThis cannot be fixed by reporting_rate alone.")
        print("The base transmission rate needs to be reduced.")
    elif ratio > 10:
        print(f"\n⚠ Model is producing {ratio:.0f}x too many infections")
        print("Reporting rate can partially compensate, but transmission is still too high.")
    else:
        print(f"\n✓ Infection count is reasonable (ratio: {ratio:.1f}x)")

print("\n" + "=" * 80)
