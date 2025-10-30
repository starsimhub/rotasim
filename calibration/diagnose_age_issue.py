"""
Diagnostic script to understand why infections are concentrated in adults instead of children
"""

import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np
import pandas as pd

print("="*80)
print("DIAGNOSTIC TEST: Age Distribution of Infections")
print("="*80)

# Create test simulation
sim = rs.Sim(
    n_agents=10000,  # Larger population for better statistics
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning simulation...")
sim.run()
print("✓ Simulation complete")

# Extract infection data
analyzer = list(sim.analyzers.values())[0]
df = analyzer.to_df()

print(f"\n{'='*80}")
print("POPULATION AGE STRUCTURE")
print(f"{'='*80}")

# Check population age structure at end of simulation
people = sim.people
ages_days = people.age.values
ages_years = ages_days / 365.25

age_bins = [0, 1, 2, 5, 200]
age_labels = ['<1 y', '1-2 y', '2-5 y', '>=5 y']
age_counts = pd.cut(ages_years, bins=age_bins, labels=age_labels, right=False).value_counts().sort_index()
total_pop = len(people)

print("\nPopulation distribution at end of simulation:")
for age_cat, count in age_counts.items():
    pct = count / total_pop * 100
    print(f"  {age_cat:10s}: {count:6d} agents ({pct:5.2f}%)")

print(f"\n{'='*80}")
print("INFECTION AGE STRUCTURE")
print(f"{'='*80}")

# Filter to calibration period (years 5-10)
df_cal = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)].copy()

print(f"\nTotal infections in calibration period (years 5-10): {len(df_cal):,}")

# Check age distribution of infections
if len(df_cal) > 0:
    # Parse age categories from the data
    age_cat_counts = df_cal['Age'].value_counts()

    # Create age category mapping
    def map_age_to_category(age_str):
        """Map age string like '0-2' to age category"""
        if age_str == '60+':
            return '>=5 y'

        # Parse the age range (e.g., '0-2', '12-24')
        parts = age_str.split('-')
        if len(parts) == 2:
            age_months = int(parts[0])
            age_years = age_months / 12

            if age_years < 1:
                return '<1 y'
            elif age_years < 2:
                return '1-2 y'
            elif age_years < 5:
                return '2-5 y'
            else:
                return '>=5 y'
        return 'Unknown'

    df_cal['AgeCat'] = df_cal['Age'].apply(map_age_to_category)
    age_cat_infection_counts = df_cal['AgeCat'].value_counts().sort_index()

    print("\nInfections by age category (years 5-10):")
    total_infections = len(df_cal)
    for age_cat in age_labels:
        if age_cat in age_cat_infection_counts.index:
            count = age_cat_infection_counts[age_cat]
            pct = count / total_infections * 100
            print(f"  {age_cat:10s}: {count:8,} infections ({pct:5.2f}%)")
        else:
            print(f"  {age_cat:10s}:        0 infections (  0.00%)")

    print(f"\n{'='*80}")
    print("INFECTION RATES BY AGE (per capita)")
    print(f"{'='*80}")

    # Calculate infection rate per capita
    print("\nInfection rate = infections / population:")
    for age_cat in age_labels:
        infections = age_cat_infection_counts.get(age_cat, 0)
        population = age_counts.get(age_cat, 1)  # Avoid division by zero
        rate = infections / population if population > 0 else 0
        print(f"  {age_cat:10s}: {infections:8,} / {population:6d} = {rate:.2f} infections/person")

    print(f"\n{'='*80}")
    print("IMMUNITY ANALYSIS")
    print(f"{'='*80}")

    # Check number of infections per person
    infections_per_person = df_cal.groupby('id').size()
    print(f"\nInfections per person during calibration period (years 5-10):")
    print(f"  Mean: {infections_per_person.mean():.2f}")
    print(f"  Median: {infections_per_person.median():.1f}")
    print(f"  Max: {infections_per_person.max()}")

    # Check distribution
    print(f"\nDistribution of infections per person:")
    for i in range(1, min(11, infections_per_person.max() + 1)):
        count = (infections_per_person == i).sum()
        pct = count / len(infections_per_person) * 100
        print(f"  {i:2d} infection(s): {count:6,} people ({pct:5.2f}%)")
    if infections_per_person.max() > 10:
        count = (infections_per_person > 10).sum()
        pct = count / len(infections_per_person) * 100
        print(f"  >10 infections: {count:6,} people ({pct:5.2f}%)")

    # Check immunity waning
    print(f"\n{'='*80}")
    print("SUSCEPTIBILITY ANALYSIS")
    print(f"{'='*80}")

    # Get current susceptibility levels by age
    disease = sim.diseases[0]
    rel_sus = disease.rel_sus.values

    # Group by age category
    sus_by_age = {}
    for age_cat, label in zip(age_bins[:-1], age_labels):
        mask = (ages_years >= age_cat) & (ages_years < age_bins[age_labels.index(label) + 1])
        sus_values = rel_sus[mask]
        if len(sus_values) > 0:
            sus_by_age[label] = {
                'mean': np.mean(sus_values),
                'median': np.median(sus_values),
                'n_susceptible': np.sum(sus_values == 1.0),
                'n_total': len(sus_values),
            }

    print("\nSusceptibility (rel_sus) by age at end of simulation:")
    for age_cat in age_labels:
        if age_cat in sus_by_age:
            info = sus_by_age[age_cat]
            pct_sus = info['n_susceptible'] / info['n_total'] * 100
            print(f"  {age_cat:10s}: mean={info['mean']:.3f}, median={info['median']:.3f}, " +
                  f"fully susceptible: {info['n_susceptible']}/{info['n_total']} ({pct_sus:.1f}%)")

else:
    print("\n⚠ No infections recorded in calibration period!")

print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

# Compare population vs infection distribution
if len(df_cal) > 0:
    print("\nPopulation vs Infection comparison:")
    print(f"{'Age Category':12s} {'Population %':>12s} {'Infections %':>12s} {'Difference':>12s}")
    print("-" * 52)
    for age_cat in age_labels:
        pop_pct = (age_counts.get(age_cat, 0) / total_pop * 100) if total_pop > 0 else 0
        inf_pct = (age_cat_infection_counts.get(age_cat, 0) / total_infections * 100) if total_infections > 0 else 0
        diff = inf_pct - pop_pct
        print(f"{age_cat:12s} {pop_pct:11.2f}% {inf_pct:11.2f}% {diff:+11.2f}pp")

    print("\nInterpretation:")
    print("  - If 'Difference' is positive: More infections than expected based on population")
    print("  - If 'Difference' is negative: Fewer infections than expected based on population")
    print("  - For rotavirus, children should have POSITIVE difference (more infections)")

print(f"\n{'='*80}")
print("✓ Diagnostic complete")
print(f"{'='*80}")
