"""
Calibration script for UK data (2008-2012)

UK Demographics:
- Population: [0.63%, 0.63%, 1.27%, 3.66%, 93.81%] for 6mo bins
- Aggregated to 4 bins: [1.26%, 1.27%, 3.66%, 93.81%]
- Birth rate: ~13/1000
- Death rate: ~6/1000 (accounting for net immigration of +4/1000)
- Follow-up period: 5 years (2008-2012)
"""

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

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

def calculate_reported_cases(df, reporting_rate):
    """
    Calculate reported cases using severity-based reporting

    P(reported | infected) = reporting_rate * severity
    """
    # Add a column for whether each infection is reported
    df['reported'] = np.random.random(len(df)) < (reporting_rate * df['severity'])

    # Filter to only reported cases
    reported_df = df[df['reported']].copy()

    return reported_df



INFECTION_AGE_DIST = [
    (0, 1, 0.138),  # <1 year: 13.8% of infections
    (1, 2, 0.277),  # 1-2 years: 27.7% of infections
    (2, 5, 0.469),  # 2-5 years: 46.9% of infections
    (5, 200, 0.116),  # ≥5 years: 11.6% of infections
]

OVERALL_PREVALENCE = 0.04

def init_prevalence_by_age(self, sim, uids):
    p = np.zeros(len(uids))
    total_pop = len(uids)
    n_infections_expected = total_pop * OVERALL_PREVALENCE
    for age_min, age_max, percent in INFECTION_AGE_DIST:
        age_group_members = ((sim.people.age >= age_min) & (sim.people.age < age_max)).uids
        age_group_pop = len(age_group_members)
        n_infections_in_age_group = percent * n_infections_expected
        p[age_group_members] = n_infections_in_age_group / age_group_pop
        print(f" ({age_min}-{age_max}): percent of infections: {percent} / total infections: {n_infections_in_age_group} / age group size: {age_group_pop} / prob per agent in age group: {n_infections_in_age_group/age_group_pop}")
    return p

def init_prevalence_by_age_G1P8(self, sim, uids):
    p = np.zeros(len(uids))
    if self.name != "G1P8":
        return p

    total_pop = len(uids)
    n_infections_expected = total_pop * OVERALL_PREVALENCE
    for age_min, age_max, percent in INFECTION_AGE_DIST:
        age_group_members = ((sim.people.age >= age_min) & (sim.people.age < age_max)).uids
        age_group_pop = len(age_group_members)
        n_infections_in_age_group = percent * n_infections_expected
        p[age_group_members] = n_infections_in_age_group / age_group_pop
        print(f" ({age_min}-{age_max}): percent of infections: {percent} / total infections: {n_infections_in_age_group} / age group size: {age_group_pop} / prob per agent in age group: {n_infections_in_age_group/age_group_pop}")
    return p


# def seed_infections_by_age(sim, overall_prevalence=0.002):
#     """
#     Seed initial infections according to UK case age distribution
#
#     Instead of uniform random seeding across all ages, seed infections
#     according to the epidemiologically realistic age distribution:
#     - 13.8% in <1 year
#     - 27.7% in 1-2 years
#     - 46.9% in 2-5 years
#     - 11.6% in ≥5 years
#
#     Args:
#         sim: Initialized simulation with people and diseases
#         overall_prevalence: Total fraction of population to infect (default 0.002 = 0.2%)
#     """
#     import numpy as np
#
#     # Target age distribution for infections (from UK_agedistribution data)
#     infection_age_dist = [
#         (0, 1, 0.138),    # <1 year: 13.8% of infections
#         (1, 2, 0.277),    # 1-2 years: 27.7% of infections
#         (2, 5, 0.469),    # 2-5 years: 46.9% of infections
#         (5, 200, 0.116),  # ≥5 years: 11.6% of infections
#     ]
#
#     # Total number of initial infections
#     n_infections = int(len(sim.people) * overall_prevalence)
#
#     if n_infections == 0:
#         return  # No infections to seed
#
#     # Get ages in years
#     ages_years = sim.people.age.values  # Already in years
#
#     # Find agents in each age category
#     age_groups = []
#     for low, high, target_prop in infection_age_dist:
#         mask = (ages_years >= low) & (ages_years < high)
#         agents_in_group = np.where(mask)[0]
#         age_groups.append((low, high, target_prop, agents_in_group))
#
#     # Allocate infections according to target proportions
#     infected_agents = []
#     for low, high, target_prop, agents_in_group in age_groups:
#         n_to_infect = int(n_infections * target_prop)
#
#         if len(agents_in_group) == 0:
#             # No agents in this age group - skip
#             if sim.pars.verbose:
#                 print(f"  Warning: No agents in age group {low}-{high} years")
#             continue
#
#         # Sample from this age group (without replacement)
#         n_available = len(agents_in_group)
#         if n_to_infect > n_available:
#             # More infections needed than agents available - infect all
#             sampled = agents_in_group
#             if sim.pars.verbose:
#                 print(f"  Warning: Need {n_to_infect} infections in {low}-{high}y but only {n_available} agents available")
#         else:
#             # Randomly sample from this age group
#             sampled = np.random.choice(agents_in_group, size=n_to_infect, replace=False)
#
#         infected_agents.extend(sampled)
#
#     # Ensure we have the right total (may differ due to rounding)
#     infected_agents = np.array(infected_agents)
#     if len(infected_agents) < n_infections:
#         # Need more infections - randomly add from any age
#         remaining = n_infections - len(infected_agents)
#         available = np.setdiff1d(np.arange(len(sim.people)), infected_agents)
#         additional = np.random.choice(available, size=remaining, replace=False)
#         infected_agents = np.concatenate([infected_agents, additional])
#     elif len(infected_agents) > n_infections:
#         # Too many infections - randomly remove some
#         infected_agents = np.random.choice(infected_agents, size=n_infections, replace=False)
#
#     # Set infections for all rotavirus diseases
#     for disease in sim.diseases.values():
#         if hasattr(disease, 'G') and hasattr(disease, 'P'):  # Is a Rotavirus disease
#             # Clear any existing infections (from default init_prev)
#             disease.infected[:] = False
#             disease.susceptible[:] = True
#             disease.ti_infected[:] = np.nan
#
#             # Set new infections
#             disease.infected[infected_agents] = True
#             disease.susceptible[infected_agents] = False
#             disease.ti_infected[infected_agents] = sim.ti
#
#             if sim.pars.verbose:
#                 print(f"  Seeded {len(infected_agents)} infections for {disease.name}")
#
#     if sim.pars.verbose:
#         print(f"\nAge distribution of {len(infected_agents)} seeded infections:")
#         for low, high, target_prop, _ in age_groups:
#             mask = (ages_years[infected_agents] >= low) & (ages_years[infected_agents] < high)
#             actual_prop = mask.sum() / len(infected_agents)
#             print(f"  {low}-{high}y: {actual_prop*100:.1f}% (target: {target_prop*100:.1f}%)")


def extract_age_specific_population_counts(sim):
    """
    Extract actual age-specific population counts from simulation

    Returns dict with age category keys and population count values:
        {'<1 y': count, '1-2 y': count, '2-5 y': count, '>=5 y': count}
    """
    import numpy as np

    # Get ages in years from sim.people.age (use .values for alive agents)
    ages_years = sim.people.age.values

    # Count agents in each age category matching process_incidence_uk categories
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }

    return age_counts

# Create sim with UK demographics
# Start 5 years before calibration period to allow for burn-in (2003-2007)
# Calibration period: 2008-2012 (years 5-9 in simulation time)
# Population increased to 50,000 to reduce stochastic noise for low incidence target (1.4 per 100k)
people = ss.People(n_agents=50000, age_data='./uk_age_data.csv')
sim = rs.Sim(
    n_agents=50000,
    start='2003-01-01',  # 5-year burn-in before 2008
    stop='2013-01-01',   # 10 years total (5 burn-in + 5 calibration)
    verbose=False,
    scenario='single',
    # base_beta=0.16,
    override_prevalence=init_prevalence_by_age_G1P8,
    people=people,
    analyzers=[rs.InfectedStrainStats(), rs.UidTracker([0, 1, 72, 644], track_fields=['rel_sus', 'infected'])],
    # networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),  # 50% contacts within same age group
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),  # UK birth rate
        ss.Deaths(death_rate=ss.peryear(6)),   # UK death rate (adjusted for immigration)
    ],
    interventions=[],
)

# Get target data first (before initializing sim)
overall_incidence, age_distribution = process_incidence_uk.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# Calibration parameters - use same cross-protection approach as Bangladesh
# UPDATED: Tighter parameter ranges to prevent corner solutions
calib_pars = sc.objdict(
    reporting_rate=[0.01, 0.005, 0.1],  # Reporting rate (1% best, 0.5-10% range) - increased from previous unrealistically low range
    # homotypic_immunity_efficacy=[1.0, 1.0, 1.0],
    # partial_heterotypic_immunity_efficacy=[0.5, 0.3, 0.7],
    # complete_heterotypic_immunity_efficacy=[0.1, 0.1, 0.1],
    base_beta=[0.4, 0.35, 0.55],  # Base transmission rate - MIN raised from 0.05 to 0.08 to prevent unrealistically low transmission
    # maternal_immunity_efficacy=[0.0, 0.0, 0.0],  # Keep at 0
    # adult_baseline_immunity=[0.45, 0.4, 0.5],  # Cumulative immunity from childhood infections - MAX lowered from 0.99 to 0.98 to allow some adult susceptibility
    # baseline_immunity_exponential_rate = [0.1, 0.05, 0.15] # adjust the rate term in the increasing form exponential decay function for baseline immunity based on number of cumulative infections.
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
        print(f"Running trial with pars: {sim_pars}")

        # Extract adult_baseline_immunity from sim_pars (actual trial values)
        # This parameter is handled manually during initialization, not a sim parameter
        # adult_baseline_immunity = 0.95  # Default, check this
        # if sim_pars is not None and 'adult_baseline_immunity' in sim_pars:
        #     adult_baseline_immunity = float(sim_pars.pop('adult_baseline_immunity'))

        # Update sim with new parameters (this already calls sim.init())
        sim = self.translate_pars(sim_pars=sim_pars)

        # Initialize baseline immunity and exposure history
        # Baseline immunity represents cumulative immunity from repeated prior infections (~95%)
        # which is DISTINCT from homotypic_immunity_efficacy (single infection ~50%)
        # sim.connectors.rotaimmunityconnector.pars.adult_baseline_immunity=adult_baseline_immunity
        sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
        # sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=3, max_age=18, min_exposures=1, max_exposures=3)
        # sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=0, max_age=3, min_exposures=0, max_exposures=2)

        # Seed infections according to epidemiologically realistic age distribution
        # (Instead of uniform random seeding which gives 94% to adults)
        # seed_infections_by_age(sim, overall_prevalence=0.002)

        # Now run the full simulation
        sim.run()

        # NOTE: Plotting commented out to avoid matplotlib fork issues in parallel workers
        # These plots can be generated after calibration completes
        # fix, ax = sim.analyzers.uidtracker.plot_field_heatmap(strain_name="G1P8")
        # # fig, ax = sim.analyzers.uidtracker.plot_field_heatmap(strain_name="G2P4")
        # fix, ax = sim.analyzers.uidtracker.plot_field_heatmap(field_name='infected', strain_name="G1P8")
        # # fix, ax = sim.analyzers.uidtracker.plot_field_heatmap(field_name='infected', strain_name="G2P4")
        # plt.show()

        return sim

    @staticmethod
    def sim_to_df(sim):
        """
        Convert the sim output to data format using severity-based reporting

        This overrides the parent method to use severity-weighted reporting:
        P(reported | infected) = reporting_rate * severity

        Returns:
            overall_incidence: float - overall incidence per 100k
            age_distribution: dataframe - proportions by age
        """
        # Extract infection data from InfectedStrainStats analyzer
        infected_analyzer = None
        for analyzer in sim.analyzers.values():
            if type(analyzer).__name__ == 'InfectedStrainStats':
                infected_analyzer = analyzer
                break

        if infected_analyzer is None:
            raise ValueError("InfectedStrainStats analyzer not found in simulation. Please add it to the sim.analyzers list.")

        # Get the infection events dataframe
        df = infected_analyzer.to_df()

        # Check if severity column exists (required for severity-based reporting)
        if 'severity' not in df.columns:
            raise ValueError("'severity' column not found in infection data. Make sure you're using the updated InfectedStrainStats analyzer.")

        # Apply severity-based reporting if reporting_rate is specified
        if hasattr(sim, '_reporting_rate') and sim._reporting_rate is not None:
            reporting_rate = sim._reporting_rate
            # Use the calculate_reported_cases function to filter based on severity
            df = calculate_reported_cases(df, reporting_rate)

        # Extract actual age-specific population counts from sim
        age_counts = extract_age_specific_population_counts(sim)

        # Process the (filtered) data using the process_incidence module
        # Returns (overall_incidence, age_distribution)
        overall_incidence, age_distribution = process_incidence_uk.process_model(df, age_counts=age_counts)

        return overall_incidence, age_distribution

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

# Create figure summarizing goodness of fit
print("\n" + "="*60)
print("Creating goodness of fit figure...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Age distribution comparison
ax1 = axes[0]
age_labels_map = {0: '0-11mo', 1: '12-23mo', 2: '24-59mo', 5: '5+yr'}
x_positions = np.arange(len(age_distribution))

target_props = age_distribution.proportion.values * 100
before_props = calib.before_age_distribution.proportion.values * 100
after_props = calib.after_age_distribution.proportion.values * 100

width = 0.25
ax1.bar(x_positions - width, target_props, width, label='Target', color='black', alpha=0.7)
ax1.bar(x_positions, before_props, width, label='Before', color='lightcoral', alpha=0.7)
ax1.bar(x_positions + width, after_props, width, label='After', color='steelblue', alpha=0.7)

ax1.set_xlabel('Age Group', fontsize=12)
ax1.set_ylabel('Proportion (%)', fontsize=12)
ax1.set_title('Age Distribution of Cases', fontsize=14, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels([age_labels_map[age] for age in age_distribution.ages.values])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add GOF text
ax1.text(0.02, 0.98, f'GOF Before: {calib.before_age_gof:.3f}\nGOF After: {calib.after_age_gof:.3f}',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right panel: Overall incidence comparison
ax2 = axes[1]
categories = ['Target', 'Before\nCalibration', 'After\nCalibration']
incidences = [overall_incidence, calib.before_overall_incidence, calib.after_overall_incidence]
colors = ['black', 'lightcoral', 'steelblue']

bars = ax2.bar(categories, incidences, color=colors, alpha=0.7)
ax2.set_ylabel('Incidence (per 100k)', fontsize=12)
ax2.set_title('Overall Incidence', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, incidences):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10)

# Add error percentage text
ax2.text(0.02, 0.98, f'Error Before: {err_before_inci:+.1f}%\nError After: {err_after_inci:+.1f}%',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('UK Calibration Goodness of Fit Summary (2008-2012)',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

# Save figure
fig_path = thisdir / 'uk_calibration_fit.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Figure saved to: {fig_path}")
plt.close()

print("="*60)
