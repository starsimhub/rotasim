"""
Trace through the incidence calculation to find where it goes wrong
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import rotasim as rs

print("=" * 80)
print("TRACING INCIDENCE CALCULATION")
print("=" * 80)

# Best parameters from last calibration
best_pars = {
    'reporting_rate': 0.9298843758582356,
    'homotypic_immunity_efficacy': 0.1423948249459815,
    'partial_heterotypic_immunity_efficacy': 0.06528308706866637,
    'complete_heterotypic_immunity_efficacy': 0.19610783473184937,
    'base_beta': 0.263304768412464,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9328929709256558
}

print("\nCreating simulation...")
print(f"  Population: 5,000 agents")
print(f"  Duration: 10 years (2003-2013)")
print(f"  Base beta: {best_pars['base_beta']:.3f}")
print(f"  Adult baseline immunity: {best_pars['adult_baseline_immunity']:.3f}")

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
)

print("\nRunning simulation...")
sim.run()
print("✓ Complete\n")

# Get infection data
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()

print("=" * 80)
print("STEP 1: RAW INFECTION DATA")
print("=" * 80)
print(f"\nTotal infection events: {len(df)}")
print(f"Unique agents: {df['id'].nunique()}")
print(f"Population size: {sim.pars.n_agents}")
print(f"\nInfections per person:")
infections_per_person = df['id'].value_counts()
print(f"  Mean: {infections_per_person.mean():.2f}")
print(f"  Median: {infections_per_person.median():.1f}")
print(f"  Max: {infections_per_person.max()}")

# Add Year column
df['Year'] = np.floor(df['CollectionTime']).astype(int)

print("\n" + "=" * 80)
print("STEP 2: FILTER TO CALIBRATION PERIOD (Years 5-9)")
print("=" * 80)
calib_df = df[(df['Year'] >= 5) & (df['Year'] <= 9)].copy()
print(f"\nInfections in calibration period: {len(calib_df)}")
print(f"  Year 5: {(calib_df['Year'] == 5).sum()}")
print(f"  Year 6: {(calib_df['Year'] == 6).sum()}")
print(f"  Year 7: {(calib_df['Year'] == 7).sum()}")
print(f"  Year 8: {(calib_df['Year'] == 8).sum()}")
print(f"  Year 9: {(calib_df['Year'] == 9).sum()}")

print("\n" + "=" * 80)
print("STEP 3: SEVERITY-BASED REPORTING")
print("=" * 80)
reporting_rate = best_pars['reporting_rate']
print(f"Reporting rate: {reporting_rate:.3f}")
print(f"\nSeverity distribution:")
print(f"  Min: {calib_df['severity'].min():.4f}")
print(f"  Max: {calib_df['severity'].max():.4f}")
print(f"  Mean: {calib_df['severity'].mean():.4f}")

# Apply reporting
calib_df['reported'] = np.random.random(len(calib_df)) < (reporting_rate * calib_df['severity'])
reported_df = calib_df[calib_df['reported']].copy()
print(f"\nReported infections: {len(reported_df)} ({len(reported_df)/len(calib_df)*100:.1f}% of calibration period)")

print("\n" + "=" * 80)
print("STEP 4: CALCULATE INCIDENCE")
print("=" * 80)
print("\nTarget calculation:")
print("  Expected infections over 5 years: (5000 * 1.4/100000) * 5 = 0.35 infections")
print("  Actual reported infections: ", len(reported_df))
print(f"  Ratio: {len(reported_df) / 0.35:.1f}x")

# Now trace through process_incidence_uk logic
print("\n" + "=" * 80)
print("STEP 5: TRACE THROUGH process_incidence_uk.process_model()")
print("=" * 80)

# Map age categories
age_mapping_reverse = {
    '0-2': '<1 y',
    '0-12': '<1 y',
    '12-24': '1-2 y',
    '24-60': '2-5 y',
    '60+': '>=5 y'
}

reported_df['AgeCat'] = reported_df['Age'].map(age_mapping_reverse)
print(f"\nAge distribution of reported cases:")
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    count = (reported_df['AgeCat'] == age_cat).sum()
    pct = count / len(reported_df) * 100 if len(reported_df) > 0 else 0
    print(f"  {age_cat}: {count} cases ({pct:.1f}%)")

# Calculate incidence per age group
# This is where the issue likely is
print("\n" + "=" * 80)
print("KEY INSIGHT: DENOMINATOR CALCULATION")
print("=" * 80)

# The process_incidence_uk function calculates incidence as:
# IR_100k = (Cases / Pop) * 100,000

# But what is "Pop" here? Is it:
# A) Total population (5000)?
# B) Population in each age group?
# C) Something else?

print("\nIf using total population as denominator:")
if len(reported_df) > 0:
    incidence_total_pop = (len(reported_df) / 5000) * 100000 / 5  # per year
    print(f"  Incidence = ({len(reported_df)} / 5000) * 100000 / 5 years")
    print(f"  Incidence = {incidence_total_pop:.1f} per 100k per year")
else:
    print("  No reported cases")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("\nThe simulation is producing {:.0f} infections over 5 years".format(len(calib_df)))
print("Expected: ~0.35 infections")
print(f"Ratio: {len(calib_df) / 0.35:.0f}x TOO MANY")
print("\nThis suggests:")
print("  1. base_beta = 0.263 is TOO HIGH (needs to be ~100x lower)")
print("  2. Adult immunity (93%) is not providing enough protection")
print("  3. The model is allowing too many reinfections")

print("\n" + "=" * 80)
