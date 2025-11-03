"""
Test if infections spread across age groups
Seeds all initial infections in <1 year olds and tracks transmission
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')
from calibrate_uk import make_sim
import pandas as pd
import numpy as np

print("="*80)
print("TRANSMISSION SPREAD TEST")
print("="*80)
print("\nSeeding infections ONLY in children <1 year old")
print("Tracking whether transmission spreads to older age groups over time")
print("="*80)

# Use best parameters from calibration
best_pars = {
    'reporting_rate': 0.00010644302524290888,
    'homotypic_immunity_efficacy': 0.8876968951690282,
    'partial_heterotypic_immunity_efficacy': 0.02436103705417797,
    'complete_heterotypic_immunity_efficacy': 0.1610717505617276,
    'base_beta': 0.19884273385365497,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9146826460364551
}

print("\nRunning simulation...")
sim = make_sim(best_pars)
sim.run()

# Get infection data
infected_analyzer = None
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        infected_analyzer = analyzer
        break

if infected_analyzer is None:
    print("ERROR: InfectedStrainStats analyzer not found")
    sys.exit(1)

df = infected_analyzer.to_df()

# Analyze age distribution over time
print("\n" + "="*80)
print("INFECTION AGE DISTRIBUTION OVER TIME")
print("="*80)

# Create age categories
df['AgeCat'] = np.nan
df.loc[df['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
df.loc[df['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
df.loc[df['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
df.loc[df['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Create time bins (yearly)
df['Year'] = np.floor(df['CollectionTime']).astype(int)

# Analyze each year
for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year]

    # Count infections by age
    age_counts = year_data['AgeCat'].value_counts()
    total = len(year_data)

    print(f"\nYear {year} ({total} infections):")
    for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
        if age_cat in age_counts.index:
            count = age_counts[age_cat]
            prop = count / total * 100
            print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
        else:
            print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Focus on calibration period (years 5-10)
print("\n" + "="*80)
print("CALIBRATION PERIOD (Years 5-10)")
print("="*80)

calib_data = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]
age_counts = calib_data['AgeCat'].value_counts()
total = len(calib_data)

print(f"\nTotal infections: {total}")
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in age_counts.index:
        count = age_counts[age_cat]
        prop = count / total * 100
        print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
    else:
        print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Count unique individuals infected per age group
print("\n" + "="*80)
print("UNIQUE INDIVIDUALS INFECTED (Calibration Period)")
print("="*80)

# Count first infection per person in each age category
calib_data_sorted = calib_data.sort_values(['id', 'CollectionTime'])
first_infections = calib_data_sorted.groupby('id').first().reset_index()

age_unique = first_infections['AgeCat'].value_counts()
total_unique = len(first_infections)

print(f"\nTotal unique individuals: {total_unique}")
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in age_unique.index:
        count = age_unique[age_cat]
        prop = count / total_unique * 100
        print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%) - Age at FIRST infection")
    else:
        print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Check if infections are spreading or just concentrated in initial age group
print("\n" + "="*80)
print("TRANSMISSION SPREAD ANALYSIS")
print("="*80)

# Compare early period (year 0-2) vs later period (year 8-10)
early_data = df[(df['CollectionTime'] >= 0) & (df['CollectionTime'] < 2)]
late_data = df[(df['CollectionTime'] >= 8) & (df['CollectionTime'] < 10)]

print("\nEarly period (Years 0-2):")
early_counts = early_data['AgeCat'].value_counts()
early_total = len(early_data)
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in early_counts.index:
        count = early_counts[age_cat]
        prop = count / early_total * 100
        print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
    else:
        print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

print("\nLate period (Years 8-10):")
late_counts = late_data['AgeCat'].value_counts()
late_total = len(late_data)
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in late_counts.index:
        count = late_counts[age_cat]
        prop = count / late_total * 100
        print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
    else:
        print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Calculate change
print("\nChange from early to late period:")
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    early_prop = (early_counts[age_cat] / early_total * 100) if age_cat in early_counts.index else 0
    late_prop = (late_counts[age_cat] / late_total * 100) if age_cat in late_counts.index else 0
    change = late_prop - early_prop
    if change > 5:
        print(f"  {age_cat:8s}: +{change:5.1f} pp ← INCREASING")
    elif change < -5:
        print(f"  {age_cat:8s}: {change:5.1f} pp ← DECREASING")
    else:
        print(f"  {age_cat:8s}: {change:5.1f} pp (stable)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Check if transmission is spreading
infant_early = (early_counts['<1 y'] / early_total * 100) if '<1 y' in early_counts.index else 0
infant_late = (late_counts['<1 y'] / late_total * 100) if '<1 y' in late_counts.index else 0

if infant_late > 90 and infant_early > 90:
    print("❌ TRANSMISSION NOT SPREADING")
    print("   Infections remain concentrated in <1 year olds throughout simulation")
    print("   This indicates a structural problem with transmission dynamics")
elif infant_late < infant_early - 10:
    print("✓ TRANSMISSION IS SPREADING")
    print("  Infections are spreading to older age groups over time")
else:
    print("⚠ UNCLEAR - Need more analysis")
    print(f"  <1y proportion: Early={infant_early:.1f}%, Late={infant_late:.1f}%")
