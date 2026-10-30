"""
Test with high adult immunity - infections should shift to children

Setup:
- Very high adult baseline immunity (0.99)
- Initialize infections in ADULTS (not children)
- Low homotypic immunity to allow reinfections
- Random mixing

Expected result:
- Initially see infections in adults (where we seeded them)
- Over time, transmission should shift to children (<5 years)
- Adults should have few infections due to high baseline immunity
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("HIGH ADULT IMMUNITY TRANSMISSION TEST")
print("="*80)
print("\nTest parameters:")
print("  - Random mixing (NO age assortment)")
print("  - ALL infections seeded in ADULTS (>=5 years)")
print("  - Adult baseline immunity: 0.99 (very high)")
print("  - Homotypic immunity: 0.2 (low - allows reinfections)")
print("  - Running for 10 years")
print("="*80)

# Define helper functions
def setup_high_adult_immunity(sim):
    """Set high adult baseline immunity"""
    # Get immunity connector
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is not None:
        # Set low immunity parameters for all
        immunity_connector.pars['homotypic_immunity_efficacy'] = 0.2  # LOW - allows reinfections
        immunity_connector.pars['partial_heterotypic_immunity_efficacy'] = 0.1
        immunity_connector.pars['complete_heterotypic_immunity_efficacy'] = 0.05
        immunity_connector.pars['maternal_immunity_efficacy'] = 0.0

        # Set very high baseline immunity for adults (>=5 years)
        ages_years = sim.people.age.values
        adult_mask = ages_years >= 5

        immunity_connector.baseline_immunity[:] = 0.0  # Start with 0 for all
        immunity_connector.baseline_immunity[adult_mask] = 0.99  # Very high for adults

        if sim.pars.verbose:
            n_adults = np.sum(adult_mask)
            n_total = len(ages_years)
            print(f"\n✓ Set immunity parameters:")
            print(f"  Homotypic immunity efficacy: 0.2 (low - allows reinfections)")
            print(f"  Adult baseline immunity: 0.99 (very high)")
            print(f"  Adults protected: {n_adults}/{n_total} ({n_adults/n_total*100:.1f}%)")
            print(f"  Children baseline immunity: 0.0 (susceptible)")

def seed_adult_infections(sim, pct=0.05):
    """Seed infections only in adults (>=5 years)"""
    ages_years = sim.people.age.values
    adult_mask = ages_years >= 5
    adult_uids = np.where(adult_mask)[0]

    if len(adult_uids) == 0:
        print("ERROR: No adults found")
        return

    # Infect specified percentage of adults
    n_to_infect = max(int(len(adult_uids) * pct), 20)
    infected_uids = np.random.choice(adult_uids, size=min(n_to_infect, len(adult_uids)), replace=False)

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
        print(f"\n✓ Seeded {len(infected_uids)} infections in ADULTS (>=5 years) only")
        print(f"  Total adults: {len(adult_uids)}")
        print(f"  Infection prevalence in adults: {len(infected_uids)/len(adult_uids)*100:.1f}%")

# Create a custom initialization callback class
class CustomInit:
    """Intervention to modify sim state after initialization"""
    __name__ = 'CustomInit'  # Required by starsim

    def __init__(self):
        self.initialized = False

    def __call__(self, sim):
        if not self.initialized and sim.ti == 0:
            # Called at the very first timestep
            setup_high_adult_immunity(sim)
            seed_adult_infections(sim, pct=0.05)
            self.initialized = True
        return

# Create simulation with the custom intervention
print("\nCreating simulation...")
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
    interventions=[CustomInit()],  # Add custom intervention
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
    (0, 1, "Year 0 (Initial seeding in adults)"),
    (1, 3, "Years 1-3 (Early transition)"),
    (3, 6, "Years 3-6 (Mid transition)"),
    (6, 10, "Years 6-10 (Late - should be mostly children)"),
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

# Check the shift from adults to children
print("\n" + "="*80)
print("DIAGNOSTIC RESULTS")
print("="*80)

early_data = df[(df['CollectionTime'] >= 0) & (df['CollectionTime'] < 1)]
late_data = df[(df['CollectionTime'] >= 6) & (df['CollectionTime'] < 10)]

if len(early_data) > 0:
    early_adult_pct = (early_data['AgeCat'] == '>=5 y').sum() / len(early_data) * 100
else:
    early_adult_pct = 0

if len(late_data) > 0:
    late_child_pct = (late_data['AgeCat'].isin(['<1 y', '1-2 y', '2-5 y'])).sum() / len(late_data) * 100
else:
    late_child_pct = 0

print(f"\nYear 0 (seeded in adults): {early_adult_pct:.1f}% adult infections")
print(f"Years 6-10 (after transition): {late_child_pct:.1f}% child infections (<5y)")

print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

if early_adult_pct > 70:
    print("✓ PASS: Initial infections mostly in adults (where we seeded them)")
else:
    print(f"✗ FAIL: Expected >70% adult infections initially, got {early_adult_pct:.1f}%")

if late_child_pct > 70:
    print("✓ PASS: Late infections shifted to children (adults protected by immunity)")
    print("\n  This confirms:")
    print("  - The analyzer fix is working correctly")
    print("  - Age-dependent immunity is functioning properly")
    print("  - Transmission dynamics respond appropriately to immunity")
else:
    print(f"✗ FAIL: Expected >70% child infections late, got {late_child_pct:.1f}%")
    print("\n  This suggests:")
    print("  - Adult baseline immunity may not be working")
    print("  - Or transmission is still not spreading properly")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
