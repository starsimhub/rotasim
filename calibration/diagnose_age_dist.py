"""
Quick diagnostic to understand the age distribution problem
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')
from calibrate_uk import initialize_uk_ages, initialize_adult_immunity, seed_infections_by_age

# Best parameters from calibration
best_params = {
    'reporting_rate': 0.0001539909745373953,
    'homotypic_immunity_efficacy': 0.16127362929023997,
    'partial_heterotypic_immunity_efficacy': 0.006089726965997355,
    'complete_heterotypic_immunity_efficacy': 0.12178083594800583,
    'base_beta': 0.060285715739135576,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9660509093091926
}

print("="*70)
print("DIAGNOSTIC: Age Distribution Investigation")
print("="*70)

# Create sim with best parameters
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=True,
    scenario='single',
    base_beta=best_params['base_beta'],
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Initialize sim
sim.initialize()

# Apply UK-specific initialization
print("\n" + "="*70)
print("STEP 1: Initializing UK ages")
print("="*70)
initialize_uk_ages(sim)

ages = sim.people.age.values
print(f"\nAge distribution of population:")
print(f"  <1y:   {((ages < 1).sum() / len(ages)) * 100:.2f}%  ({(ages < 1).sum()} agents)")
print(f"  1-2y:  {(((ages >= 1) & (ages < 2)).sum() / len(ages)) * 100:.2f}%  ({((ages >= 1) & (ages < 2)).sum()} agents)")
print(f"  2-5y:  {(((ages >= 2) & (ages < 5)).sum() / len(ages)) * 100:.2f}%  ({((ages >= 2) & (ages < 5)).sum()} agents)")
print(f"  ≥5y:   {((ages >= 5).sum() / len(ages)) * 100:.2f}%  ({(ages >= 5).sum()} agents)")

# Initialize adult immunity
print("\n" + "="*70)
print("STEP 2: Initializing adult immunity")
print("="*70)
initialize_adult_immunity(sim, adult_baseline_immunity=best_params['adult_baseline_immunity'])

# Seed infections
print("\n" + "="*70)
print("STEP 3: Seeding initial infections")
print("="*70)
seed_infections_by_age(sim, overall_prevalence=0.002)

# Check initial infection distribution
print("\n" + "="*70)
print("STEP 4: Running simulation")
print("="*70)
sim.run()

# Get analyzer results
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        break

# Analyze infection ages over time
print("\n" + "="*70)
print("ANALYZING INFECTION AGE DISTRIBUTION")
print("="*70)

# Focus on calibration period (years 5-9)
calibration_df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

print(f"\nTotal infections in calibration period: {len(calibration_df)}")
print(f"Unique agents infected: {calibration_df['id'].nunique()}")

# Look at age distribution of infections
print("\nInfection ages (raw Age field from analyzer):")
print(calibration_df['Age'].value_counts().sort_index())

# Convert to age categories
calibration_df['AgeCat'] = 'Unknown'
calibration_df.loc[calibration_df['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
calibration_df.loc[calibration_df['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
calibration_df.loc[calibration_df['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
calibration_df.loc[calibration_df['Age'] == '60+', 'AgeCat'] = '>=5 y'

print("\nInfection age categories:")
age_cat_counts = calibration_df['AgeCat'].value_counts()
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    count = age_cat_counts.get(age_cat, 0)
    pct = (count / len(calibration_df)) * 100
    print(f"  {age_cat:6s}: {count:5d} infections ({pct:5.1f}%)")

# Look at infection numbers per person
calibration_df = calibration_df.sort_values(['id', 'CollectionTime'])
calibration_df['infection_number'] = calibration_df.groupby('id').cumcount() + 1

print("\nInfection numbers (how many times each agent was infected):")
print(calibration_df['infection_number'].value_counts().sort_index().head(10))

# Count symptomatic infections (first 3 per agent)
symptomatic = calibration_df[calibration_df['infection_number'] <= 3]
print(f"\nSymptomatic infections (first 3 per agent): {len(symptomatic)}")
print(f"Symptomatic age distribution:")
symp_age_counts = symptomatic['AgeCat'].value_counts()
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    count = symp_age_counts.get(age_cat, 0)
    pct = (count / len(symptomatic)) * 100 if len(symptomatic) > 0 else 0
    print(f"  {age_cat:6s}: {count:5d} infections ({pct:5.1f}%)")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
