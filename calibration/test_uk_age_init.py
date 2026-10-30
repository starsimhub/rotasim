"""Test UK age initialization in calibration setup"""

import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*80)
print("Testing UK Age Initialization")
print("="*80)

# Helper function
def initialize_uk_ages(sim):
    """Initialize population with UK age distribution"""
    n = len(sim.people)

    age_bins = [
        (0, 1, 0.0126),
        (1, 2, 0.0127),
        (2, 5, 0.0366),
        (5, 80, 0.9381),
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
    sim.people.age[:] = ages_years * 365.25

# Create sim
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        rs.Aging(),  # Add aging module
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Initialize
print("\nInitializing sim...")
sim.run(until=1)
initialize_uk_ages(sim)

# Check ages
ages_years = sim.people.age.values / 365.25
print(f"\n✓ Ages initialized:")
print(f"  Range: {ages_years.min():.1f} - {ages_years.max():.1f} years")
print(f"  Mean: {ages_years.mean():.1f} years")

age_dist = {
    '<1 y': ((ages_years < 1).sum() / len(ages_years) * 100, 1.26),
    '1-2 y': (((ages_years >= 1) & (ages_years < 2)).sum() / len(ages_years) * 100, 1.27),
    '2-5 y': (((ages_years >= 2) & (ages_years < 5)).sum() / len(ages_years) * 100, 3.66),
    '>=5 y': ((ages_years >= 5).sum() / len(ages_years) * 100, 93.81),
}

print(f"\n  {'Age':8s} {'Actual':>8s} {'Target':>8s} {'Error':>8s}")
print("  " + "-"*35)
for age, (actual, target) in age_dist.items():
    error = actual - target
    print(f"  {age:8s} {actual:7.2f}% {target:7.2f}% {error:+7.2f}pp")

# Now run full simulation and check if ages stay reasonable
print(f"\n{'='*80}")
print("Running full 10-year simulation...")
print(f"{'='*80}")

# Reset and run full sim
sim2 = rs.Sim(
    n_agents=1000,  # Smaller for speed
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        rs.Aging(),  # Add aging module
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

sim2.run(until=1)
initialize_uk_ages(sim2)

# Store initial ages
initial_ages = sim2.people.age.values.copy() / 365.25

# Run to completion
sim2.run()

# Check final ages
final_ages = sim2.people.age.values / 365.25

print(f"\n✓ Simulation complete")
print(f"\nInitial ages: {initial_ages.min():.1f} - {initial_ages.max():.1f} years (mean: {initial_ages.mean():.1f})")
print(f"Final ages:   {final_ages.min():.1f} - {final_ages.max():.1f} years (mean: {final_ages.mean():.1f})")

age_change = final_ages.mean() - initial_ages.mean()
print(f"\nMean age change: {age_change:+.1f} years")

if age_change > 8:
    print("✓ Ages increased by ~10 years as expected (aging is working!)")
elif age_change > 0:
    print(f"⚠ Ages increased by only {age_change:.1f} years (expected ~10)")
else:
    print("❌ Ages did not increase (aging is NOT working - population stayed static)")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if age_change > 8:
    print("\n✓ Both initial age distribution AND aging are working correctly!")
else:
    print("\n⚠ Initial age distribution works, but population aging needs to be implemented")
    print("  This means births/deaths work, but existing agents don't age during simulation")

print(f"{'='*80}")
