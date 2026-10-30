"""Test different assortativity levels to find what produces realistic age distribution"""
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
    """Initialize adults with immunity reflecting prior childhood infections"""
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
            disease.rel_sus[adult_uids] = 1.0 - homotypic_protection

def seed_infections_by_age(sim, overall_prevalence=0.002):
    """Seed initial infections according to UK case age distribution"""
    infection_age_dist = [
        (0, 1, 0.138),    # <1 year: 13.8% of infections
        (1, 2, 0.277),    # 1-2 years: 27.7% of infections
        (2, 5, 0.469),    # 2-5 years: 46.9% of infections
        (5, 200, 0.116),  # ≥5 years: 11.6% of infections
    ]

    n_infections = int(len(sim.people) * overall_prevalence)
    if n_infections == 0:
        return

    ages_years = sim.people.age.values / 365.25

    age_groups = []
    for low, high, target_prop in infection_age_dist:
        mask = (ages_years >= low) & (ages_years < high)
        agents_in_group = np.where(mask)[0]
        age_groups.append((low, high, target_prop, agents_in_group))

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

    infected_agents = np.array(infected_agents)
    if len(infected_agents) < n_infections:
        remaining = n_infections - len(infected_agents)
        available = np.setdiff1d(np.arange(len(sim.people)), infected_agents)
        additional = np.random.choice(available, size=remaining, replace=False)
        infected_agents = np.concatenate([infected_agents, additional])
    elif len(infected_agents) > n_infections:
        infected_agents = np.random.choice(infected_agents, size=n_infections, replace=False)

    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            disease.infected[:] = False
            disease.susceptible[:] = True
            disease.ti_infected[:] = np.nan
            disease.infected[infected_agents] = True
            disease.susceptible[infected_agents] = False
            disease.ti_infected[infected_agents] = sim.ti

def run_test(assortativity, n_agents=5000, n_years=5):
    """Run test with specified assortativity level"""
    print(f"\n{'='*80}")
    print(f"Testing Assortativity = {assortativity:.1f}")
    print(f"{'='*80}")

    sim = rs.Sim(
        n_agents=n_agents,
        start='2003-01-01',
        stop=f'{2003+n_years}-01-01',
        verbose=False,
        scenario='single',
        base_beta=0.22,
        override_prevalence=0.002,
        analyzers=[rs.InfectedStrainStats()],
        networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=assortativity),
        demographics=[
            rs.Aging(),
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
    )

    sim.init()
    initialize_uk_ages(sim)
    initialize_adult_immunity(sim, homotypic_protection=0.5)
    seed_infections_by_age(sim, overall_prevalence=0.002)

    print("\nRunning simulation...")
    sim.run()

    # Analyze results
    infected_analyzer = sim.analyzers['infectedstrainstats']
    df = infected_analyzer.to_df()

    # Calculate age distribution
    child_infections = df[df['Age'].isin(['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60'])]
    child_pct = len(child_infections) / len(df) * 100 if len(df) > 0 else 0

    adult_infections = df[df['Age'] == '60+']
    adult_pct = len(adult_infections) / len(df) * 100 if len(df) > 0 else 0

    print(f"\n✓ Results:")
    print(f"  Total infections: {len(df)}")
    print(f"  Children <5 years: {child_pct:.1f}%")
    print(f"  Adults ≥5 years: {adult_pct:.1f}%")
    print(f"  Target: 86% in children, 14% in adults")
    print(f"  Error: {abs(child_pct - 86):.1f} percentage points off target")

    return assortativity, child_pct, adult_pct, len(df)

# Test different assortativity levels
print("="*80)
print("Testing Impact of Assortativity on Age Distribution")
print("="*80)
print("\nAll tests use:")
print("  - UK age distribution (6.2% children, 93.8% adults)")
print("  - Adult baseline immunity: 50% protection (rel_sus=0.5)")
print("  - Equal contacts for children and adults (n=7)")
print("  - Age-targeted initial seeding")

results = []

# Test assortativity levels from 0.5 to 0.95
for assort in [0.5, 0.7, 0.8, 0.9, 0.95]:
    try:
        assort_val, child_pct, adult_pct, total = run_test(assort)
        results.append({
            'assortativity': assort_val,
            'child_pct': child_pct,
            'adult_pct': adult_pct,
            'total_infections': total,
            'error': abs(child_pct - 86)
        })
    except Exception as e:
        print(f"\n⚠ Error with assortativity={assort}: {e}")
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)
print(f"\n{'Assortativity':<15} {'Child %':<12} {'Adult %':<12} {'Error':<12} {'Total Infections'}")
print("-" * 65)
for r in results:
    print(f"{r['assortativity']:<15.2f} {r['child_pct']:<12.1f} {r['adult_pct']:<12.1f} {r['error']:<12.1f} {r['total_infections']}")

if results:
    best = min(results, key=lambda x: x['error'])
    print(f"\n✓ Best assortativity: {best['assortativity']:.2f}")
    print(f"  Achieved: {best['child_pct']:.1f}% in children (target: 86%)")
    print(f"  Error: {best['error']:.1f} percentage points")
