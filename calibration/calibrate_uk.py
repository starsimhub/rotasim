"""
Calibration script for UK data (2008-2012)

UK Demographics:
- Population: [0.63%, 0.63%, 1.27%, 3.66%, 93.81%] for 6mo bins
- Aggregated to 4 bins: [1.26%, 1.27%, 3.66%, 93.81%]
- Birth rate: ~13/1000
- Death rate: ~6/1000 (accounting for net immigration of +4/1000)
- Follow-up period: 5 years (2008-2012)
"""

import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*60)
print("UK Calibration (2008-2012)")
print("="*60)
print("\nUK Demographics:")
print("  Birth rate: 13/1000")
print("  Death rate: 6/1000")
print("  Net migration: +4/1000 adults (approximated in death rate)")
print("  Target age distribution: [1.26%, 1.27%, 3.66%, 93.81%]")
print("  Follow-up: 5 years (2008-2012)")
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

def initialize_adult_immunity(sim, homotypic_protection=0.5):
    """
    Initialize adults with immunity reflecting prior childhood infections

    In reality, adults in UK have experienced rotavirus infections in childhood
    and have built up immunity. Initialize them with reduced susceptibility
    reflecting homotypic protection against circulating strains.

    NOTE: We set rel_sus directly rather than trying to simulate full immunity history,
    because the SIRS model's ~91 day waning would erase all protection from childhood.
    This baseline protection represents cumulative immunity from repeated childhood exposures.

    Args:
        sim: Initialized simulation with people and immunity connector
        homotypic_protection: Protection level (e.g., 0.5 = 50% protection → rel_sus=0.5)
    """
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

            # Set rel_sus directly to reflect baseline immunity from childhood exposures
            # This is NOT temporary (doesn't wane in 91 days) - it's cumulative protection
            disease.rel_sus[adult_uids] = 1.0 - homotypic_protection

    if sim.pars.verbose:
        # Check actual rel_sus values after initialization
        first_disease = list(sim.diseases.values())[0]
        adult_rel_sus = first_disease.rel_sus[adult_uids]
        child_rel_sus = first_disease.rel_sus[~adult_mask]

        print(f"\n✓ Initialized {n_adults} adults with prior immunity:")
        print(f"  Prior infections: 2-3 (typical childhood exposure)")
        print(f"  Baseline protection: {homotypic_protection*100:.0f}% (homotypic)")
        print(f"  Adult rel_sus: {adult_rel_sus.mean():.3f} (children: {child_rel_sus.mean():.1f})")
        print(f"  Note: This baseline doesn't wane - new infections add temporary immunity on top")

def seed_infections_by_age(sim, overall_prevalence=0.002):
    """
    Seed initial infections according to UK case age distribution

    Instead of uniform random seeding across all ages, seed infections
    according to the epidemiologically realistic age distribution:
    - 13.8% in <1 year
    - 27.7% in 1-2 years
    - 46.9% in 2-5 years
    - 11.6% in ≥5 years

    Args:
        sim: Initialized simulation with people and diseases
        overall_prevalence: Total fraction of population to infect (default 0.002 = 0.2%)
    """
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
            # No agents in this age group - skip
            if sim.pars.verbose:
                print(f"  Warning: No agents in age group {low}-{high} years")
            continue

        # Sample from this age group (without replacement)
        n_available = len(agents_in_group)
        if n_to_infect > n_available:
            # More infections needed than agents available - infect all
            sampled = agents_in_group
            if sim.pars.verbose:
                print(f"  Warning: Need {n_to_infect} infections in {low}-{high}y but only {n_available} agents available")
        else:
            # Randomly sample from this age group
            sampled = np.random.choice(agents_in_group, size=n_to_infect, replace=False)

        infected_agents.extend(sampled)

    # Ensure we have the right total (may differ due to rounding)
    infected_agents = np.array(infected_agents)
    if len(infected_agents) < n_infections:
        # Need more infections - randomly add from any age
        remaining = n_infections - len(infected_agents)
        available = np.setdiff1d(np.arange(len(sim.people)), infected_agents)
        additional = np.random.choice(available, size=remaining, replace=False)
        infected_agents = np.concatenate([infected_agents, additional])
    elif len(infected_agents) > n_infections:
        # Too many infections - randomly remove some
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

            if sim.pars.verbose:
                print(f"  Seeded {len(infected_agents)} infections for {disease.name}")

    if sim.pars.verbose:
        print(f"\nAge distribution of {len(infected_agents)} seeded infections:")
        for low, high, target_prop, _ in age_groups:
            mask = (ages_years[infected_agents] >= low) & (ages_years[infected_agents] < high)
            actual_prop = mask.sum() / len(infected_agents)
            print(f"  {low}-{high}y: {actual_prop*100:.1f}% (target: {target_prop*100:.1f}%)")

# Create sim with UK demographics
# Start 5 years before calibration period to allow for burn-in (2003-2007)
# Calibration period: 2008-2012 (years 5-9 in simulation time)
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',  # 5-year burn-in before 2008
    stop='2013-01-01',   # 10 years total (5 burn-in + 5 calibration)
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),  # 50% contacts within same age group
    demographics=[
        rs.Aging(),  # CRITICAL: Add aging module (starsim doesn't age automatically)
        ss.Births(birth_rate=ss.peryear(13)),  # UK birth rate
        ss.Deaths(death_rate=ss.peryear(6)),   # UK death rate (adjusted for immigration)
    ],
)

# Get target data first (before initializing sim)
overall_incidence, age_distribution = process_incidence_uk.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# Calibration parameters - use same cross-protection approach as Bangladesh
calib_pars = sc.objdict(
    reporting_rate=[0.0002, 0.0001, 0.001],
    homotypic_immunity_efficacy=[0.5, 0.1, 0.9],
    partial_heterotypic_immunity_efficacy=[0.2, 0.0, 0.5],
    complete_heterotypic_immunity_efficacy=[0.1, 0.0, 0.3],
    base_beta=[0.16, 0.05, 0.30],  # Base transmission rate (replaces rel_beta)
    maternal_immunity_efficacy=[0.0, 0.0, 0.0],  # Keep at 0
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (20 trials)...")
print("="*60)

# Create custom calibration class that initializes UK ages for each trial
class UKCalibration(Calibration):
    """Custom calibration that initializes UK age distribution for each trial"""

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """Override to initialize UK ages, adult immunity, and seed infections by age"""
        # First, translate parameters (creates a copy of sim with new parameters)
        if calib_pars is not None:
            sim_pars = self.trial_to_sim_pars(calib_pars=calib_pars, trial=trial)

        # Update sim with new parameters (this already calls sim.init())
        sim = self.translate_pars(sim_pars=sim_pars)

        # Initialize UK age distribution (people object already exists from init)
        initialize_uk_ages(sim)

        # Get homotypic immunity efficacy from the immunity connector for adult initialization
        homotypic_efficacy = 0.5  # Default
        for connector in sim.connectors.values():
            if type(connector).__name__ == 'RotaImmunityConnector':
                if hasattr(connector.pars, 'homotypic_immunity_efficacy'):
                    homotypic_efficacy = connector.pars.homotypic_immunity_efficacy
                break

        # Initialize adults with immunity reflecting prior childhood infections
        # Adults get reduced susceptibility equal to homotypic protection
        initialize_adult_immunity(sim, homotypic_protection=homotypic_efficacy)

        # Seed infections according to epidemiologically realistic age distribution
        # (Instead of uniform random seeding which gives 94% to adults)
        seed_infections_by_age(sim, overall_prevalence=0.002)

        # Now run the full simulation
        sim.run()

        return sim

calib = UKCalibration(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=20,
    debug=False,
)

calib.calibrate()

print("\n" + "="*60)
print("Checking fit...")
print("="*60)
calib.check_fit()

print("\n" + "="*60)
print("Results:")
print("="*60)

print("\nBest parameters:")
for par, val in calib.best_pars.items():
    print(f"  {par}: {val:.6f}")

print("\n" + "="*60)
print("Comparing Overall Incidence:")
print("="*60)
print(f"Target:  {overall_incidence:.1f} per 100k")
print(f"Before:  {calib.before_overall_incidence:.1f} per 100k")
print(f"After:   {calib.after_overall_incidence:.1f} per 100k")
err_before_inci = (calib.before_overall_incidence - overall_incidence) / overall_incidence * 100
err_after_inci = (calib.after_overall_incidence - overall_incidence) / overall_incidence * 100
print(f"\nError before: {err_before_inci:+.1f}%")
print(f"Error after:  {err_after_inci:+.1f}%")

print("\n" + "="*60)
print("Comparing Age Distribution (proportions):")
print("="*60)
print(f"\n{'Age':<10} {'Target':<15} {'Before':<15} {'After':<15} {'Error Before':<20} {'Error After':<20}")
print("-"*100)

for i in range(len(age_distribution)):
    if i < len(calib.after_age_distribution):
        age = age_distribution.ages.iloc[i]
        target_prop = age_distribution.proportion.iloc[i] * 100
        before_prop = calib.before_age_distribution.proportion.iloc[i] * 100
        after_prop = calib.after_age_distribution.proportion.iloc[i] * 100

        err_before = before_prop - target_prop
        err_after = after_prop - target_prop

        print(f"{age:<10} {target_prop:<15.1f}% {before_prop:<15.1f}% {after_prop:<15.1f}% {err_before:<20.1f}pp {err_after:<20.1f}pp")

print("\n" + "="*60)
print("Summary:")
print("="*60)

print(f"\nOverall Incidence:")
print(f"  Target:  {overall_incidence:.1f} per 100k")
print(f"  Before:  {calib.before_overall_incidence:.1f} per 100k ({err_before_inci:+.1f}%)")
print(f"  After:   {calib.after_overall_incidence:.1f} per 100k ({err_after_inci:+.1f}%)")

improvement_inci = abs(err_before_inci) - abs(err_after_inci)
print(f"  Improvement: {improvement_inci:.1f} percentage points")

print(f"\nAge Distribution GOF:")
print(f"  Before: {calib.before_age_gof:.4f}")
print(f"  After:  {calib.after_age_gof:.4f}")
improvement_age = calib.before_age_gof - calib.after_age_gof
print(f"  Improvement: {improvement_age:.4f}")

if abs(err_after_inci) < 20 and calib.after_age_gof < 0.5:
    print("\n✓ Excellent fit: Both incidence and age distribution match well!")
elif abs(err_after_inci) < 50 and calib.after_age_gof < 1.0:
    print("\n✓ Good fit: Both metrics improved")
else:
    print("\n⚠ Model fit could be improved further")

print("\n✓ UK calibration complete!")
