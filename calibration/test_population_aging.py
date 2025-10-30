"""Test to understand why population isn't aging correctly"""

import starsim as ss
import rotasim as rs
import numpy as np

print("="*80)
print("Testing Population Aging with Star sim Demographics")
print("="*80)

# Create a simple simulation
sim = rs.Sim(
    n_agents=1000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=True,
    scenario='single',
    base_beta=0.05,  # Low beta to minimize infections for cleaner test
    override_prevalence=0.0,  # Start with no infections
    analyzers=[],  # No analyzers needed
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Run simulation
print("\nRunning 10-year simulation...")
sim.run()

print("\n" + "="*80)
print("Final population:")
print(f"  Total agents: {len(sim.people)}")
print(f"  Age range (days): {sim.people.age.min():.0f} - {sim.people.age.max():.0f}")
print(f"  Age range (years): {sim.people.age.min()/365.25:.2f} - {sim.people.age.max()/365.25:.2f}")

# Check age distribution
ages_years = sim.people.age.values / 365.25
age_bins = [0, 1, 2, 5, 10, 20, 100]
age_labels = ['<1 y', '1-2 y', '2-5 y', '5-10 y', '10-20 y', '>=20 y']
age_counts = np.histogram(ages_years, bins=age_bins)[0]

print("\nAge distribution:")
for label, count in zip(age_labels, age_counts):
    pct = count / len(sim.people) * 100
    print(f"  {label:10s}: {count:6d} agents ({pct:5.2f}%)")

# Check if ages actually changed during simulation
print("\n" + "="*80)
print("Conclusion:")
if sim.people.age.max() /365.25 >= 9:
    print("✓ Population IS aging correctly (max age increased)")
else:
    print("✗ Population NOT aging correctly (max age did not increase enough)")
print("="*80)
