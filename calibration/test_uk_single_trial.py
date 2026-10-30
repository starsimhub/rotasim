"""
Test single trial of UK calibration to debug failures
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import sys

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*60)
print("UK Calibration - Single Trial Test")
print("="*60)

# Helper function to initialize UK age distribution
def initialize_uk_ages(sim):
    """Initialize population with UK age distribution"""
    import numpy as np
    n = len(sim.people)

    # UK age distribution (from calibration targets)
    age_bins = [
        (0, 1, 0.0126),    # <1 year: 1.26%
        (1, 2, 0.0127),    # 1-2 years: 1.27%
        (2, 5, 0.0366),    # 2-5 years: 3.66%
        (5, 80, 0.9381),   # 5+ years: 93.81%
    ]

    # Sample ages from distribution
    ages_years = []
    for low, high, prop in age_bins:
        n_in_bin = int(n * prop)
        # Use realistic distribution for adults
        if low >= 5:
            bin_ages = np.random.beta(2, 2, n_in_bin) * (high - low) + low
        else:
            bin_ages = np.random.uniform(low, high, n_in_bin)
        ages_years.extend(bin_ages)

    # Handle rounding
    while len(ages_years) < n:
        ages_years.append(np.random.beta(2, 2) * 75 + 5)

    ages_years = np.array(ages_years[:n])
    ages_days = ages_years * 365.25

    # Set ages
    sim.people.age[:] = ages_days

    if sim.pars.verbose:
        print(f"Initialized UK age distribution:")
        print(f"  Age range: {ages_years.min():.1f} - {ages_years.max():.1f} years")
        print(f"  Mean age: {ages_years.mean():.1f} years")

def initialize_adult_immunity(sim, adult_baseline_immunity=0.95):
    """Initialize adults with immunity reflecting prior childhood infections"""
    import numpy as np

    # Get ages in years
    ages_years = sim.people.age.values / 365.25

    # Identify adults (≥5 years)
    adult_mask = ages_years >= 5
    n_adults = adult_mask.sum()

    if n_adults == 0:
        return  # No adults to initialize

    # Get the immunity connector
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is None:
        if sim.pars.verbose:
            print("  Warning: No RotaImmunityConnector found, cannot initialize adult immunity")
        return

    adult_uids = np.where(adult_mask)[0]

    # Record infection history for tracking purposes
    immunity_connector.num_recovered_infections[adult_uids] = np.random.choice([2, 3], size=n_adults)
    immunity_connector.has_immunity[adult_uids] = True

    # Set bitmasks indicating prior exposure to circulating strains
    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            G = disease.G
            P = disease.P

            immunity_connector.exposed_G_bitmask[adult_uids] |= (1 << G)
            immunity_connector.exposed_P_bitmask[adult_uids] |= (1 << P)

    # Set permanent baseline immunity for adults (doesn't decay over time)
    # This represents cumulative immunity from repeated childhood exposures
    immunity_connector.baseline_immunity[adult_uids] = adult_baseline_immunity

    if sim.pars.verbose:
        # Check actual rel_sus values after initialization
        first_disease = list(sim.diseases.values())[0]
        adult_rel_sus = first_disease.rel_sus[adult_uids]
        child_rel_sus = first_disease.rel_sus[~adult_mask]

        print(f"\n✓ Initialized {n_adults} adults with baseline immunity:")
        print(f"  Prior infections: 2-3 (typical childhood exposure)")
        print(f"  Cumulative protection: {adult_baseline_immunity*100:.1f}% (from repeated childhood exposures)")
        print(f"  Adult rel_sus: {adult_rel_sus.mean():.3f} (children: {child_rel_sus.mean():.1f})")
        print(f"  Note: This baseline doesn't wane - new infections add temporary immunity on top")

def seed_infections_by_age(sim, overall_prevalence=0.002):
    """Seed initial infections according to UK case age distribution"""
    import numpy as np

    # Target age distribution for infections (from UK_agedistribution data)
    infection_age_dist = [
        (0, 1, 0.138),    # <1 year: 13.8% of infections
        (1, 2, 0.277),    # 1-2 years: 27.7% of infections
        (2, 5, 0.469),    # 2-5 years: 46.9% of infections
        (5, 200, 0.116),  # ≥5 years: 11.6% of infections
    ]

    # Total number of initial infections
    n_infections = int(len(sim.people) * overall_prevalence)

    if n_infections == 0:
        return  # No infections to seed

    # Get ages in years
    ages_years = sim.people.age.values / 365.25

    # Find agents in each age category
    age_groups = []
    for low, high, target_prop in infection_age_dist:
        mask = (ages_years >= low) & (ages_years < high)
        agents_in_group = np.where(mask)[0]
        age_groups.append((low, high, target_prop, agents_in_group))

    # Allocate infections according to target proportions
    infected_agents = []
    for low, high, target_prop, agents_in_group in age_groups:
        n_to_infect = int(n_infections * target_prop)

        if len(agents_in_group) == 0:
            continue

        # Sample from this age group (without replacement)
        n_available = len(agents_in_group)
        if n_to_infect > n_available:
            sampled = agents_in_group
        else:
            sampled = np.random.choice(agents_in_group, size=n_to_infect, replace=False)

        infected_agents.extend(sampled)

    # Ensure we have the right total (may differ due to rounding)
    infected_agents = np.array(infected_agents)
    if len(infected_agents) < n_infections:
        remaining = n_infections - len(infected_agents)
        available = np.setdiff1d(np.arange(len(sim.people)), infected_agents)
        additional = np.random.choice(available, size=remaining, replace=False)
        infected_agents = np.concatenate([infected_agents, additional])
    elif len(infected_agents) > n_infections:
        infected_agents = np.random.choice(infected_agents, size=n_infections, replace=False)

    # Set infections for all rotavirus diseases
    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):  # Is a Rotavirus disease
            # Clear any existing infections (from default init_prev)
            disease.infected[:] = False
            disease.susceptible[:] = True
            disease.ti_infected[:] = np.nan

            # Set new infections
            disease.infected[infected_agents] = True
            disease.susceptible[infected_agents] = False
            disease.ti_infected[infected_agents] = sim.ti

    print(f"  Seeded {len(infected_agents)} initial infections")

# Create sim with UK demographics
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        rs.Aging(),
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")

# Initialize sim
sim.init()

# Initialize UK age distribution
print("\nInitializing UK age distribution...")
initialize_uk_ages(sim)

# Initialize adults with baseline immunity
print("\nInitializing adult baseline immunity (0.95)...")
initialize_adult_immunity(sim, adult_baseline_immunity=0.95)

# Seed infections by age
print("\nSeeding infections by age...")
seed_infections_by_age(sim, overall_prevalence=0.002)

# Run the simulation
print("\nRunning simulation...")
try:
    sim.run()
    print("\n✓ Simulation completed successfully!")

    # Check for any issues with results
    analyzer = sim.analyzers[0]
    print(f"\nResults check:")
    df = analyzer.to_df()
    print(f"  Total infection events collected: {len(df)}")
    if len(df) > 0:
        print(f"  Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")
        print(f"  Strains: {df['Strain'].unique()}")
        print(f"  Age distribution: {df['Age'].value_counts().to_dict()}")

except Exception as e:
    print(f"\n✗ Simulation failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
