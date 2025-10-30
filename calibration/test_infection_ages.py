"""Test infection age recording with age-targeted seeding"""
import sciris as sc
import starsim as ss
import rotasim as rs
import pandas as pd
import numpy as np

thisdir = sc.thispath(__file__)

def initialize_uk_ages(sim):
    """Initialize population with UK age distribution"""
    n = len(sim.people)

    age_bins = [
        (0, 1, 0.0126),    # <1 year: 1.26%
        (1, 2, 0.0127),    # 1-2 years: 1.27%
        (2, 5, 0.0366),    # 2-5 years: 3.66%
        (5, 80, 0.9381),   # 5+ years: 93.81%
    ]

    ages_years = []
    for low, high, prop in age_bins:
        n_in_bin = int(n * prop)
        if low >= 5:
            bin_ages = np.random.beta(2, 2, n_in_bin) * (high - low) + low
        else:
            bin_ages = np.random.uniform(low, high, n_in_bin)
        ages_years.extend(bin_ages)

    while len(ages_years) < n:
        ages_years.append(np.random.beta(2, 2) * 75 + 5)

    ages_years = np.array(ages_years[:n])
    ages_days = ages_years * 365.25

    sim.people.age[:] = ages_days

def initialize_adult_immunity(sim, homotypic_protection=0.5):
    """Initialize adults with immunity reflecting prior childhood infections

    NOTE: We set rel_sus directly rather than trying to simulate full immunity history,
    because the SIRS model's ~91 day waning would erase all protection from childhood.
    This baseline protection represents cumulative immunity from repeated childhood exposures.
    """
    ages_years = sim.people.age.values / 365.25
    adult_mask = ages_years >= 5
    n_adults = adult_mask.sum()

    if n_adults == 0:
        return

    # Get immunity connector
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is None:
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

            # Set rel_sus directly to reflect baseline immunity from childhood exposures
            # This is NOT temporary (doesn't wane in 91 days) - it's cumulative protection
            disease.rel_sus[adult_uids] = 1.0 - homotypic_protection

    # Check actual rel_sus after initialization
    first_disease = list(sim.diseases.values())[0]
    adult_rel_sus = first_disease.rel_sus[adult_uids]
    child_rel_sus = first_disease.rel_sus[~adult_mask]

    print(f"\n✓ Initialized {n_adults} adults with prior immunity")
    print(f"  Prior infections: 2-3 (childhood exposure)")
    print(f"  Baseline protection: {homotypic_protection*100:.0f}% (homotypic)")
    print(f"  Adult rel_sus: {adult_rel_sus.mean():.3f} (children: {child_rel_sus.mean():.1f})")
    print(f"  Note: This baseline doesn't wane - new infections add temporary immunity on top")

def seed_infections_by_age(sim, overall_prevalence=0.002):
    """Seed initial infections according to UK case age distribution"""
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
        return

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
        n_available = len(agents_in_group)
        if n_to_infect > n_available:
            sampled = agents_in_group
        else:
            sampled = np.random.choice(agents_in_group, size=n_to_infect, replace=False)
        infected_agents.extend(sampled)

    # Ensure we have the right total
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
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            disease.infected[:] = False
            disease.susceptible[:] = True
            disease.ti_infected[:] = np.nan
            disease.infected[infected_agents] = True
            disease.susceptible[infected_agents] = False
            disease.ti_infected[infected_agents] = sim.ti

    print(f"\n✓ Seeded {len(infected_agents)} infections by age:")
    for low, high, target_prop, _ in age_groups:
        mask = (ages_years[infected_agents] >= low) & (ages_years[infected_agents] < high)
        actual_prop = mask.sum() / len(infected_agents)
        print(f"  {low}-{high}y: {actual_prop*100:.1f}% (target: {target_prop*100:.1f}%)")

print("="*80)
print("Testing Infection Age Recording")
print("="*80)

# Run a simulation with UK ages
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2008-01-01',  # Just 5 years
    verbose=False,
    scenario='single',
    base_beta=0.22,  # Best-fit value from calibration
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),  # Age-assortative mixing
    demographics=[
        rs.Aging(),
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Initialize and set UK ages
print("\nInitializing with UK age distribution...")
sim.init()
initialize_uk_ages(sim)

ages_years = sim.people.age.values / 365.25
print(f"\n✓ Population initialized:")
print(f"  <1 year: {(ages_years < 1).sum() / len(ages_years) * 100:.1f}%")
print(f"  1-2 years: {((ages_years >= 1) & (ages_years < 2)).sum() / len(ages_years) * 100:.1f}%")
print(f"  2-5 years: {((ages_years >= 2) & (ages_years < 5)).sum() / len(ages_years) * 100:.1f}%")
print(f"  >=5 years: {(ages_years >= 5).sum() / len(ages_years) * 100:.1f}%")

# Initialize adult immunity (NEW: adults start with prior immunity)
print("\nInitializing adult immunity (reflecting childhood infections)...")
initialize_adult_immunity(sim, homotypic_protection=0.5)

# Seed infections by age (NEW: age-targeted seeding)
print("\nSeeding infections by age (not uniform)...")
seed_infections_by_age(sim, overall_prevalence=0.002)

# Run simulation
print("\nRunning simulation...")
sim.run()

# Check infection age distribution
print("\n" + "="*80)
print("Analyzing recorded infections...")
print("="*80)

infected_analyzer = sim.analyzers['infectedstrainstats']
df = infected_analyzer.to_df()

print(f"\n✓ Total infections recorded: {len(df)}")
print(f"\nAge categories in infection events:")
age_counts = df['Age'].value_counts().sort_index()
print(age_counts)

# Calculate proportions
total = len(df)
print(f"\nProportions:")
for age_cat in age_counts.index:
    prop = age_counts[age_cat] / total * 100
    print(f"  {age_cat}: {prop:.1f}%")

print(f"\nFirst 20 infection events:")
print(df[['id', 'CollectionTime', 'Age', 'Strain']].head(20))

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if infections are primarily in children
child_infections = df[df['Age'].isin(['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60'])]
child_pct = len(child_infections) / len(df) * 100

adult_infections = df[df['Age'] == '60+']
adult_pct = len(adult_infections) / len(df) * 100

print(f"\nInfection distribution:")
print(f"  Children <5 years: {child_pct:.1f}%")
print(f"  Adults ≥5 years: {adult_pct:.1f}%")

if child_pct > 50:
    print(f"\n✓ GOOD: Majority of infections in children")
    print(f"   This matches rotavirus epidemiology (should be ~86% in children)")
    if child_pct > 70:
        print(f"   ✓ EXCELLENT: {child_pct:.1f}% in children is very realistic")
else:
    print(f"\n⚠ WARNING: Only {child_pct:.1f}% of infections in children")
    print(f"   Target should be ~86% in children <5 years")
