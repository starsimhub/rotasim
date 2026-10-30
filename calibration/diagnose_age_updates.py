"""
Diagnose if agent ages are actually being updated during simulation
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("AGE UPDATE DIAGNOSTIC")
print("="*80)
print("\nTesting if agent ages update during simulation...")

# Create a minimal simulation
sim = rs.Sim(
    n_agents=100,
    start='2003-01-01',
    stop='2005-01-01',  # 2 years
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.01,
    networks='random',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Initialize
sim.initialize()

# Get initial ages
initial_ages = sim.people.age.copy()
initial_ages_years = initial_ages / 365.25

print(f"\n{'='*80}")
print("INITIAL STATE (t=0)")
print(f"{'='*80}")
print(f"Number of agents: {len(sim.people)}")
print(f"Age statistics (in years):")
print(f"  Min age: {initial_ages_years.min():.2f} years")
print(f"  Max age: {initial_ages_years.max():.2f} years")
print(f"  Mean age: {initial_ages_years.mean():.2f} years")
print(f"  Agents <1 year: {np.sum(initial_ages_years < 1)}")
print(f"  Agents 1-5 years: {np.sum((initial_ages_years >= 1) & (initial_ages_years < 5))}")
print(f"  Agents >=5 years: {np.sum(initial_ages_years >= 5)}")

# Run for 1 year
print(f"\n{'='*80}")
print("RUNNING SIMULATION FOR 1 YEAR...")
print(f"{'='*80}")

# Run for 365 days
for i in range(365):
    sim.step()
    if i % 100 == 0:
        print(f"  Day {i+1}/365...")

# Get ages after 1 year
ages_after_1yr = sim.people.age.copy()
ages_after_1yr_years = ages_after_1yr / 365.25

print(f"\n{'='*80}")
print("AFTER 1 YEAR (t=365 days)")
print(f"{'='*80}")
print(f"Number of agents: {len(sim.people)} (births/deaths occurred)")

# Check agents that existed at start and still exist
still_alive = np.where(~np.isnan(ages_after_1yr[:len(initial_ages)]))[0]
if len(still_alive) > 0:
    age_increases = ages_after_1yr[still_alive] - initial_ages[still_alive]
    age_increases_years = age_increases / 365.25

    print(f"\nAgents still alive from initial population: {len(still_alive)}")
    print(f"Age increase statistics:")
    print(f"  Min increase: {age_increases_years.min():.2f} years")
    print(f"  Max increase: {age_increases_years.max():.2f} years")
    print(f"  Mean increase: {age_increases_years.mean():.2f} years")
    print(f"  Expected: ~1.0 years")

    if abs(age_increases_years.mean() - 1.0) < 0.01:
        print("\n  ✓ Ages ARE being updated correctly!")
    else:
        print(f"\n  ✗ Ages NOT updating correctly! Mean increase: {age_increases_years.mean():.2f} years")

print(f"\nCurrent age distribution (after 1 year):")
print(f"  Min age: {ages_after_1yr_years.min():.2f} years")
print(f"  Max age: {ages_after_1yr_years.max():.2f} years")
print(f"  Mean age: {ages_after_1yr_years.mean():.2f} years")
print(f"  Agents <1 year: {np.sum(ages_after_1yr_years < 1)}")
print(f"  Agents 1-5 years: {np.sum((ages_after_1yr_years >= 1) & (ages_after_1yr_years < 5))}")
print(f"  Agents >=5 years: {np.sum(ages_after_1yr_years >= 5)}")

# Now test the InfectedStrainStats analyzer age categorization
print(f"\n{'='*80}")
print("TESTING ANALYZER AGE CATEGORIZATION")
print(f"{'='*80}")

# Get the analyzer's age categorization function
analyzer = rs.InfectedStrainStats()
analyzer.sim = sim  # Mock assignment
analyzer.init_results()

# Test age categorization for various ages
test_ages_years = [0.1, 0.5, 1.5, 3.0, 6.0, 10.0, 20.0]
print("\nTesting _get_age_category() function:")
for age in test_ages_years:
    category = analyzer._get_age_category(age)
    print(f"  Age {age:5.1f} years → Category: {category}")

print(f"\n{'='*80}")
print("CHECKING ACTUAL INFECTION RECORDING")
print(f"{'='*80}")

# Manually check what ages would be recorded for current agents
print("\nSample of 10 random agents and their age categories:")
sample_indices = np.random.choice(len(sim.people), size=min(10, len(sim.people)), replace=False)
for idx in sample_indices:
    age_days = sim.people.age[idx]
    age_years = age_days / 365.25
    category = analyzer._get_age_category(age_years)
    print(f"  Agent {idx}: age={age_years:.2f} years → category={category}")

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
