"""
Simple diagnostic to identify why incidence is 70,000x too high
WITHOUT running a full simulation
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import pandas as pd
import numpy as np

print("=" * 80)
print("SIMPLE DIAGNOSTIC - TRACING INCIDENCE CALCULATION")
print("=" * 80)

# Create mock infection data that should produce ~1.4 per 100k incidence
# Target: 1.4 per 100k over 5 years
# Population: 50,000 agents
# Expected total infections over 5 years: 50000 * (1.4/100000) * 5 = 3.5 infections

# Let's create 4 infections (close to 3.5) spread across the 5-year period
mock_data = {
    't': [1825, 2190, 2555, 2920],  # Days 5.0, 6.0, 7.0, 8.0 years (calibration years 5-9)
    'CollectionTime': [5.0, 6.0, 7.0, 8.0],  # Years
    'uid': [100, 200, 300, 400],  # 4 different agents
    'id': [100, 200, 300, 400],  # Same as uid (unique infection events)
    'Age': ['0-2', '12-24', '24-60', '60+'],  # Mix of ages
    'PopulationSize': [50000, 50000, 50000, 50000],  # Population size
    'severity': [0.05, 0.06, 0.04, 0.05],
    'Strain': ['G1P8A1B1', 'G1P8A1B1', 'G1P8A1B1', 'G1P8A1B1'],
}

df = pd.DataFrame(mock_data)

print("\nMock data created:")
print(f"  Total infections: {len(df)}")
print(f"  Population: {df['PopulationSize'].iloc[0]}")
print(f"  Time range: years {df['CollectionTime'].min()} to {df['CollectionTime'].max()}")

# Now trace through process_model logic
print("\n" + "-" * 80)
print("TRACING THROUGH process_model() LOGIC")
print("-" * 80)

# Step 1: Create Year column
df['Year'] = np.floor(df['CollectionTime']).astype(int)
print(f"\nStep 1 - Year calculation:")
print(f"  Years: {df['Year'].values}")

# Step 2: Filter to years 5-9
dat_calib = df[(df['Year'] >= 5) & (df['Year'] <= 9)]
print(f"\nStep 2 - Filter to calibration period (years 5-9):")
print(f"  Infections in calibration period: {len(dat_calib)}")
print(f"  Years: {dat_calib['Year'].values}")

# Step 3: Count symptomatic cases (first 3 infections per person)
# For now, assume all are first infections (symptomatic)
print(f"\nStep 3 - Count symptomatic cases:")
print(f"  Assuming all {len(dat_calib)} are first infections (symptomatic)")

# Step 4: Group by age category
age_mapping_reverse = {
    '0-2': '<1 y',
    '0-12': '<1 y',
    '12-24': '1-2 y',
    '24-60': '2-5 y',
    '60+': '>=5 y'
}
dat_calib['AgeCat'] = dat_calib['Age'].map(age_mapping_reverse)
print(f"\nStep 4 - Age categorization:")
for age_cat in dat_calib['AgeCat'].unique():
    count = (dat_calib['AgeCat'] == age_cat).sum()
    print(f"  {age_cat}: {count} infections")

# Step 5: Calculate incidence
# Formula: (Cases / Pop) * 100,000
# Expected: (4 infections / 50,000 pop) / 5 years * 100,000 = 1.6 per 100k per year

total_cases_per_year = len(dat_calib) / 5.0  # 5 years
total_pop = 50000
expected_incidence = (total_cases_per_year / total_pop) * 100000

print(f"\nStep 5 - Calculate incidence:")
print(f"  Total cases over 5 years: {len(dat_calib)}")
print(f"  Cases per year (average): {total_cases_per_year:.1f}")
print(f"  Population: {total_pop}")
print(f"  Expected incidence: (cases/pop) * 100,000 = ({total_cases_per_year}/{total_pop}) * 100,000")
print(f"  Expected incidence: {expected_incidence:.2f} per 100k")
print(f"  Target incidence: 1.40 per 100k")

print("\n" + "=" * 80)
print("KEY QUESTION: What would cause 100,000 per 100k incidence?")
print("=" * 80)

# If we're getting 100,000 per 100k, that means:
# (Cases / Pop) * 100,000 = 100,000
# Cases / Pop = 1.0
# Cases = Pop

print("\nFor incidence to be 100,000 per 100k:")
print("  (Cases / Pop) * 100,000 = 100,000")
print("  → Cases = Pop")
print(f"  → We would need {total_pop} cases (everyone infected!)")

print("\nPossible causes:")
print("  1. Model producing way too many infections (~12,500x too many)")
print("  2. Population denominator wrong (too small by ~70,000x)")
print("  3. Not filtering to calibration period correctly (counting all years)")
print("  4. Counting multiple infections per person without deduplication")

print("\n" + "=" * 80)
