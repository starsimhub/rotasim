"""
Test if infections spread across age groups with random mixing

This is a diagnostic test to check if there's a bug in transmission dynamics.
Setup:
- Random mixing (NO age assortment)
- ALL initial infections in <1 year olds
- No adult baseline immunity (all adults susceptible)
- Low homotypic immunity (allow reinfections)

Expected result: Should see cases in adults. If not, there's a transmission bug.
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss
import pandas as pd

print("="*80)
print("RANDOM MIXING TRANSMISSION TEST")
print("="*80)
print("\nTest parameters:")
print("  - Random mixing (NO age assortment)")
print("  - ALL infections seeded in <1 year olds")
print("  - Adult baseline immunity: 0 (all susceptible)")
print("  - Homotypic immunity: 0.2 (low - allows reinfections)")
print("  - Running for 10 years")
print("="*80)

# Create simulation with RANDOM mixing
print("\nCreating simulation...")

# Define helper functions first
def setup_equal_susceptibility(sim):
    """Set low immunity parameters and zero baseline immunity"""
    # Get immunity connector
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is not None:
        # Set low immunity parameters
        immunity_connector.pars['homotypic_immunity_efficacy'] = 0.2  # LOW - allows reinfections
        immunity_connector.pars['partial_heterotypic_immunity_efficacy'] = 0.1
        immunity_connector.pars['complete_heterotypic_immunity_efficacy'] = 0.05
        immunity_connector.pars['maternal_immunity_efficacy'] = 0.0

        # Set all baseline immunity to 0 (everyone is equally susceptible)
        immunity_connector.baseline_immunity[:] = 0.0

        if sim.pars.verbose:
            print("\n✓ Set immunity parameters:")
            print(f"  Homotypic immunity efficacy: 0.2 (low - allows reinfections)")
            print(f"  Baseline immunity: 0 (all ages equally susceptible)")

def seed_infant_infections(sim, pct=0.01):
    """Seed infections only in <1 year olds"""
    ages_years = sim.people.age.values
    infant_mask = ages_years < 1
    infant_uids = np.where(infant_mask)[0]

    if len(infant_uids) == 0:
        print("ERROR: No infants found")
        return

    # Infect specified percentage of infants
    n_to_infect = max(int(len(infant_uids) * pct), 20)
    infected_uids = np.random.choice(infant_uids, size=min(n_to_infect, len(infant_uids)), replace=False)

    # Clear any existing infections and set new ones
    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            disease.infected[:] = False
            disease.susceptible[:] = True
            disease.ti_infected[:] = np.nan

            disease.infected[infected_uids] = True
            disease.susceptible[infected_uids] = False
            disease.ti_infected[infected_uids] = sim.ti

    if sim.pars.verbose:
        print(f"\n✓ Seeded {len(infected_uids)} infections in <1 year olds only")
        print(f"  Total infants: {len(infant_uids)}")
        print(f"  Infection prevalence in infants: {len(infected_uids)/len(infant_uids)*100:.1f}%")

# Create a custom initialization callback class
class CustomInit:
    """Intervention to modify sim state after initialization"""
    __name__ = 'CustomInit'  # Required by starsim

    def __init__(self):
        self.initialized = False

    def __call__(self, sim):
        if not self.initialized and sim.ti == 0:
            # Called at the very first timestep
            setup_equal_susceptibility(sim)
            seed_infant_infections(sim, pct=0.01)
            self.initialized = True
        return

# Create simulation with the custom intervention
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,  # Small initial prevalence
    analyzers=[rs.InfectedStrainStats()],
    networks='random',  # RANDOM MIXING - no age assortment
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[CustomInit()],  # Add custom intervention to seed infections and set immunity
)

print("  Using random mixing (no age assortment)")

# Run simulation
print("\n" + "="*80)
print("Running simulation...")
print("="*80)
sim.run()
print("\n✓ Simulation completed")

# Analyze results
print("\n" + "="*80)
print("ANALYZING INFECTION AGE DISTRIBUTION")
print("="*80)

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
print(f"\nTotal infection events: {len(df)}")

# Create age categories
df['AgeCat'] = np.nan
df.loc[df['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
df.loc[df['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
df.loc[df['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
df.loc[df['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Overall age distribution
print("\n" + "-"*80)
print("OVERALL AGE DISTRIBUTION (All years)")
print("-"*80)
age_counts = df['AgeCat'].value_counts().sort_index()
total = len(df)

for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    if age_cat in age_counts.index:
        count = age_counts[age_cat]
        prop = count / total * 100
        print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
    else:
        print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Time-stratified analysis
print("\n" + "-"*80)
print("AGE DISTRIBUTION BY TIME PERIOD")
print("-"*80)

periods = [
    (0, 2, "Early (Years 0-2)"),
    (2, 5, "Mid (Years 2-5)"),
    (5, 10, "Late (Years 5-10)"),
]

for start, end, label in periods:
    period_data = df[(df['CollectionTime'] >= start) & (df['CollectionTime'] < end)]
    if len(period_data) == 0:
        print(f"\n{label}: NO INFECTIONS")
        continue

    age_counts = period_data['AgeCat'].value_counts().sort_index()
    total = len(period_data)

    print(f"\n{label} ({total} infections):")
    for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
        if age_cat in age_counts.index:
            count = age_counts[age_cat]
            prop = count / total * 100
            print(f"  {age_cat:8s}: {count:6d} ({prop:5.1f}%)")
        else:
            print(f"  {age_cat:8s}: {0:6d} ({0:5.1f}%)")

# Check for adult infections
print("\n" + "="*80)
print("DIAGNOSTIC RESULTS")
print("="*80)

adult_infections = df[df['AgeCat'] == '>=5 y']
n_adult_inf = len(adult_infections)
pct_adult = n_adult_inf / len(df) * 100 if len(df) > 0 else 0

print(f"\nAdult infections (>=5 years): {n_adult_inf} ({pct_adult:.1f}%)")
print(f"Total infections: {len(df)}")

if n_adult_inf > 0:
    print("\n" + "✓"*40)
    print("✓ SUCCESS: Infections ARE spreading to adults")
    print("  Transmission dynamics appear to be working correctly")
    print("  The age distribution problem is likely due to parameter/immunity settings")
    print("✓"*40)
else:
    print("\n" + "✗"*40)
    print("✗ POSSIBLE BUG: NO adult infections detected")
    print("  With random mixing and no adult immunity, we should see some adult cases")
    print("  This suggests a problem with transmission dynamics")
    print("✗"*40)

# Additional diagnostics
print("\n" + "-"*80)
print("ADDITIONAL DIAGNOSTICS")
print("-"*80)

# Check if any reinfections occurred
df_sorted = df.sort_values(['id', 'CollectionTime'])
df_sorted['infection_number'] = df_sorted.groupby('id').cumcount() + 1
reinfections = df_sorted[df_sorted['infection_number'] > 1]
n_reinfections = len(reinfections)
n_unique_infected = df['id'].nunique()

print(f"\nUnique individuals infected: {n_unique_infected}")
print(f"Total infection events: {len(df)}")
print(f"Reinfection events: {n_reinfections} ({n_reinfections/len(df)*100:.1f}%)")
print(f"Average infections per person: {len(df)/n_unique_infected:.2f}")

if n_reinfections > 0:
    print("\n✓ Reinfections ARE occurring (immunity is not 100%)")
else:
    print("\n⚠ NO reinfections detected (might be too short simulation or too strong immunity)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
