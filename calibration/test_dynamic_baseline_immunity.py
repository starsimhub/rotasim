"""
Test with DYNAMIC baseline immunity updating
Children automatically get 0.99 immunity when they reach age 5

This tests whether the immunity erosion is really the problem, or if there's
something else preventing the expected shift to children.
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("DYNAMIC BASELINE IMMUNITY TEST")
print("="*80)
print("\nTest parameters:")
print("  - Random mixing (NO age assortment)")
print("  - Infections seeded in ADULTS (>=5 years)")
print("  - DYNAMIC immunity: Children get 0.99 immunity when they turn 5")
print("  - Homotypic immunity: 0.2 (low - allows reinfections)")
print("  - Running for 10 years")
print("="*80)

# Define dynamic immunity updater
class DynamicImmunityUpdater:
    """Intervention that updates baseline immunity based on age EVERY timestep"""
    __name__ = 'DynamicImmunityUpdater'

    def __init__(self):
        self.initialized = False
        self.initial_uids = None

    def __call__(self, sim):
        # Get immunity connector
        immunity_connector = None
        for connector in sim.connectors.values():
            if type(connector).__name__ == 'RotaImmunityConnector':
                immunity_connector = connector
                break

        if immunity_connector is None:
            return

        if not self.initialized and sim.ti == 0:
            # Initial setup
            immunity_connector.pars['homotypic_immunity_efficacy'] = 0.2
            immunity_connector.pars['partial_heterotypic_immunity_efficacy'] = 0.1
            immunity_connector.pars['complete_heterotypic_immunity_efficacy'] = 0.05
            immunity_connector.pars['maternal_immunity_efficacy'] = 0.0

            # Store initial UIDs for tracking
            self.initial_uids = set(range(len(sim.people)))

            # Seed infections in adults
            self._seed_adult_infections(sim)

            if sim.pars.verbose:
                print("\n✓ Initial setup complete")
                print("  - Immunity parameters configured")
                print("  - Infections seeded in adults")

            self.initialized = True

        # Update baseline immunity EVERY timestep based on current age
        ages_years = sim.people.age.values
        adult_mask = ages_years >= 5

        # Set immunity based on age
        immunity_connector.baseline_immunity[adult_mask] = 0.99
        immunity_connector.baseline_immunity[~adult_mask] = 0.0

        # Verbose reporting every year
        if sim.pars.verbose and sim.ti % 365 == 0 and sim.ti > 0:
            year = sim.ti / 365
            n_adults = np.sum(adult_mask)
            n_total = len(ages_years)
            adult_imm = immunity_connector.baseline_immunity[adult_mask]
            print(f"\nYear {year:.0f}: {n_adults}/{n_total} adults with mean immunity {adult_imm.mean():.3f}")

    def _seed_adult_infections(self, sim, pct=0.05):
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

# Create simulation
print("\nCreating simulation...")
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',  # 10 years
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks='random',  # RANDOM MIXING - no age assortment
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[DynamicImmunityUpdater()],
)

print("  Using random mixing (no age assortment)")
print("  Dynamic immunity: Adults always have 0.99 baseline immunity")

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
    print("✓ PASS: Late infections shifted to children with dynamic immunity!")
    print("\n  This confirms:")
    print("  - The immunity erosion WAS the problem")
    print("  - Dynamic updating of baseline immunity works correctly")
    print("  - The immunity connector applies baseline immunity properly")
else:
    print(f"✗ FAIL: Expected >70% child infections late, got {late_child_pct:.1f}%")
    print("\n  This suggests:")
    print("  - The immunity erosion is NOT the only problem")
    print("  - There may be additional issues with:")
    print("    * How baseline immunity is applied")
    print("    * How susceptibility is calculated")
    print("    * Transmission dynamics")
    print("  - Further investigation needed")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
