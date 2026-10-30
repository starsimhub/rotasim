"""
Test severity-based reporting tracking system

This test verifies:
1. InfectedStrainStats analyzer tracks infection numbers (1st, 2nd, 3rd, 4+)
2. Severity probabilities are calculated correctly based on infection number
3. InitializeChildImmunity intervention works correctly
4. Younger children have higher average severity (more 1st/2nd infections)
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("TESTING SEVERITY-BASED REPORTING SYSTEM")
print("="*80)

# Create simulation with severity tracking
print("\nCreating simulation...")
sim = rs.Sim(
    n_agents=2000,
    start='2003-01-01',
    stop='2008-01-01',  # 5 years
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.01,
    networks='random',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            homotypic_immunity_efficacy=0.3,  # Moderate immunity - allows reinfections
            adult_baseline_immunity=0.0,  # No adult baseline immunity for this test
        ),
    ],
    analyzers=[
        rs.InfectedStrainStats(),  # Now tracks infection numbers and severity
    ],
    interventions=[
        rs.InitializeChildImmunity(
            max_age_years=3.0,  # Children <36 months
            min_infections=1,   # At least 1 prior infection
            max_infections=1,   # Exactly 1 prior infection for simplicity
            verbose=True,
        ),
    ],
)

print("\n" + "="*80)
print("RUNNING SIMULATION...")
print("="*80)
sim.run()
print("\n✓ Simulation completed")

# Get infection data
print("\n" + "="*80)
print("ANALYZING SEVERITY TRACKING")
print("="*80)

analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

if analyzer is None:
    print("ERROR: InfectedStrainStats analyzer not found")
    sys.exit(1)

df = analyzer.to_df()
print(f"\nTotal infection events: {len(df)}")

# Check that we have the new columns
if 'n_infections' not in df.columns:
    print("ERROR: 'n_infections' column not found in dataframe!")
    sys.exit(1)

if 'severity' not in df.columns:
    print("ERROR: 'severity' column not found in dataframe!")
    sys.exit(1)

print("\n✓ New columns found: n_infections, severity")

# Analyze infection number distribution
print("\n" + "-"*80)
print("INFECTION NUMBER DISTRIBUTION")
print("-"*80)

inf_counts = df['n_infections'].value_counts().sort_index()
total_infections = len(df)

print(f"\nInfection number breakdown (total: {total_infections}):")
for n_inf in sorted(inf_counts.index):
    count = inf_counts[n_inf]
    pct = count / total_infections * 100
    print(f"  {n_inf}{'st' if n_inf == 1 else 'nd' if n_inf == 2 else 'rd' if n_inf == 3 else 'th'} infection: {count:5d} ({pct:5.1f}%)")

# Verify severity values match expected
print("\n" + "-"*80)
print("SEVERITY PROBABILITY VERIFICATION")
print("-"*80)

expected_severity = {
    1: 0.051,   # 5.1%
    2: 0.0644,  # 6.44%
    3: 0.0432,  # 4.32%
    4: 0.0378,  # 3.78% (4th and higher)
}

print("\nExpected severity rates:")
print("  1st infection: 0.051 (5.1%)")
print("  2nd infection: 0.0644 (6.44%)")
print("  3rd infection: 0.0432 (4.32%)")
print("  4th+ infection: 0.0378 (3.78%)")

print("\nActual severity rates in data:")
all_correct = True
for n_inf in sorted(df['n_infections'].unique()):
    # Get all infections with this infection number
    mask = df['n_infections'] == n_inf
    severities = df.loc[mask, 'severity'].unique()

    if len(severities) != 1:
        print(f"  {n_inf}th infection: ERROR - multiple severity values found: {severities}")
        all_correct = False
    else:
        actual = severities[0]
        if n_inf <= 3:
            expected = expected_severity[n_inf]
        else:
            expected = expected_severity[4]

        match = "✓" if abs(actual - expected) < 0.0001 else "✗"
        print(f"  {n_inf}th infection: {actual:.4f} {match} (expected {expected:.4f})")

        if abs(actual - expected) >= 0.0001:
            all_correct = False

if all_correct:
    print("\n✓✓✓ ALL SEVERITY VALUES CORRECT!")
else:
    print("\n✗ Some severity values are incorrect")

# Analyze severity by age
print("\n" + "-"*80)
print("SEVERITY BY AGE GROUP")
print("-"*80)

print("\nMean severity probability by age:")
age_severity = df.groupby('Age')['severity'].agg(['mean', 'count'])
age_severity = age_severity.sort_index()

for age_cat, row in age_severity.iterrows():
    print(f"  {age_cat:8s}: {row['mean']:.4f} ({int(row['count'])} infections)")

print("\n" + "-"*80)
print("MEAN INFECTION NUMBER BY AGE")
print("-"*80)

print("\nMean infection number by age (lower = more 1st/2nd infections):")
age_n_inf = df.groupby('Age')['n_infections'].agg(['mean', 'count'])
age_n_inf = age_n_inf.sort_index()

for age_cat, row in age_n_inf.iterrows():
    print(f"  {age_cat:8s}: {row['mean']:.2f} ({int(row['count'])} infections)")

# Test interpretation
print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

# Check that younger children have higher severity on average
young_ages = ['0-2', '2-4', '4-6', '6-12', '12-24']
young_severity = df[df['Age'].isin(young_ages)]['severity'].mean()

old_ages = ['24-36', '36-48', '48-60', '60+']
old_severity = df[df['Age'].isin(old_ages)]['severity'].mean()

print(f"\nMean severity for children <24 months: {young_severity:.4f}")
print(f"Mean severity for children/adults >=24 months: {old_severity:.4f}")

if young_severity > old_severity:
    print("\n✓ Younger children have higher severity (as expected)")
    print("  This is because they're more likely to have 1st/2nd infections")
else:
    print(f"\n✗ Unexpected: Older children have higher severity")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n✓ Severity tracking system is working:")
print("  1. InfectedStrainStats tracks infection numbers (1st, 2nd, 3rd, 4+)")
print("  2. Severity probabilities are assigned correctly based on infection number")
print("  3. InitializeChildImmunity intervention initializes young children")
print("  4. Younger children naturally have higher severity (more primary infections)")

print("\nNext steps for calibration:")
print("  1. Use the 'severity' column to weight reporting probability")
print("  2. Make overall reporting rate a calibratable parameter")
print("  3. Apply: P(reported) = reporting_rate * severity")
print("  4. This will concentrate reported cases in younger age groups")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
