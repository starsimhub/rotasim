"""
Diagnose why all infections are in <1 year olds
Check if it's related to reinfection rates
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')
from calibrate_uk import make_sim
import pandas as pd
import numpy as np

# Run simulation with best parameters
print("Running simulation with best parameters...")
print("="*60)

# Best parameters from calibration
best_pars = {
    'reporting_rate': 0.00010644302524290888,
    'homotypic_immunity_efficacy': 0.8876968951690282,
    'partial_heterotypic_immunity_efficacy': 0.02436103705417797,
    'complete_heterotypic_immunity_efficacy': 0.1610717505617276,
    'base_beta': 0.19884273385365497,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9146826460364551
}

sim = make_sim(best_pars)
sim.run()

# Get infection data from analyzer
infected_analyzer = None
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        infected_analyzer = analyzer
        break

if infected_analyzer is None:
    print("ERROR: InfectedStrainStats analyzer not found")
    sys.exit(1)

df = infected_analyzer.to_df()

# Focus on calibration period (years 5-10)
dat = df[(df['CollectionTime'] < 10) & (df['CollectionTime'] > 4)].copy()

print(f"\nTotal infection events in calibration period: {len(dat)}")
print(f"Unique individuals infected: {dat['id'].nunique()}")

# Add infection number per person
dat = dat.sort_values(['id', 'CollectionTime'])
dat['infection_number'] = dat.groupby('id').cumcount() + 1

# Analyze infection numbers
print("\n" + "="*60)
print("INFECTION NUMBER DISTRIBUTION")
print("="*60)
infection_counts = dat['infection_number'].value_counts().sort_index()
print(infection_counts)
print(f"\nInfections beyond 4th: {len(dat[dat['infection_number'] > 4])}")
print(f"Percentage beyond 4th: {100 * len(dat[dat['infection_number'] > 4]) / len(dat):.1f}%")

# Look at symptomatic threshold (<=4)
symptomatic = dat[dat['infection_number'] <= 4].copy()
print(f"\nSymptomatic infections (<=4): {len(symptomatic)}")

# Categorize ages
symptomatic['AgeCat'] = np.nan
symptomatic.loc[symptomatic['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
symptomatic.loc[symptomatic['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
symptomatic.loc[symptomatic['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
symptomatic.loc[symptomatic['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Count infections by age category
print("\n" + "="*60)
print("SYMPTOMATIC INFECTIONS BY AGE")
print("="*60)
age_counts = symptomatic['AgeCat'].value_counts().sort_index()
print(age_counts)
print("\nProportions:")
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in age_counts.index:
        count = age_counts[age_cat]
        prop = count / len(symptomatic) * 100
        print(f"  {age_cat}: {count:5d} ({prop:5.1f}%)")
    else:
        print(f"  {age_cat}: {0:5d} ({0:5.1f}%)")

# Now analyze infection number by age
print("\n" + "="*60)
print("INFECTION NUMBER BY AGE CATEGORY")
print("="*60)
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    age_data = symptomatic[symptomatic['AgeCat'] == age_cat]
    if len(age_data) > 0:
        print(f"\n{age_cat}:")
        inf_nums = age_data['infection_number'].value_counts().sort_index()
        for inf_num, count in inf_nums.items():
            prop = count / len(age_data) * 100
            print(f"  Infection #{inf_num}: {count:5d} ({prop:5.1f}%)")

# Check reinfection rates
print("\n" + "="*60)
print("REINFECTION ANALYSIS")
print("="*60)
people_with_1_inf = len(dat[dat['infection_number'] == 1]['id'].unique())
people_with_2plus_inf = len(dat[dat['infection_number'] >= 2]['id'].unique())
total_infected = dat['id'].nunique()

print(f"People with only 1 infection: {people_with_1_inf} ({100*people_with_1_inf/total_infected:.1f}%)")
print(f"People with 2+ infections: {people_with_2plus_inf} ({100*people_with_2plus_inf/total_infected:.1f}%)")

# Calculate theoretical reinfection rate
homotypic_protection = best_pars['homotypic_immunity_efficacy']
expected_susceptibility = 1 - homotypic_protection
print(f"\nWith homotypic immunity = {homotypic_protection:.3f}:")
print(f"  Expected susceptibility to reinfection: {expected_susceptibility:.3f} ({100*expected_susceptibility:.1f}%)")
print(f"  Observed reinfection rate: {100*people_with_2plus_inf/total_infected:.1f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if people_with_2plus_inf / total_infected < 0.3:
    print("Low reinfection rate explains why infections concentrate in youngest age group.")
    print("With ONE strain and HIGH homotypic immunity:")
    print("  - Most people get infected once (as infants)")
    print("  - Few get reinfected → few infections in older age groups")
    print("\nPossible solutions:")
    print("  1. Use multiple strains (heterotypic reinfections)")
    print("  2. Lower homotypic immunity (allow more same-strain reinfections)")
    print("  3. Age-graduated immunity/susceptibility")
else:
    print("Reinfection rate seems adequate - issue may be elsewhere")
