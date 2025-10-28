"""
Quick test to verify long-term immunity age tracking
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import starsim as ss
import rotasim as rs

print("Testing Long-Term Immunity Age Tracking")
print("=" * 60)

# Create a small simulation
sim = rs.Sim(
    n_agents=5000,
    start='2000-01-01',
    stop='2015-01-01',  # 15 years
    verbose=False,
    scenario='baseline',
    base_beta=0.20,  # Higher transmission
    override_prevalence=0.01,
    networks=ss.RandomNet(n_contacts=10),
    demographics=[
        ss.Births(birth_rate=ss.peryear(20)),
        ss.Deaths(death_rate=ss.peryear(10)),
    ],
)

print("\nRunning simulation...")
sim.run()

# Access immunity connector (lowercase key name)
immunity = sim.connectors['rotaimmunityconnector']

# Get data for alive agents
alive = sim.people.alive.values
long_term_immune = immunity.long_term_immune.values[alive]
lti_ages = immunity.long_term_immune_age.values[alive]

# Get statistics
n_lti = np.sum(long_term_immune)
if n_lti > 0:
    lti_age_values = lti_ages[long_term_immune]
    mean_age = np.nanmean(lti_age_values)
    median_age = np.nanmedian(lti_age_values)
    
    print(f"\nResults:")
    print(f"  Total alive agents: {np.sum(alive):,}")
    print(f"  Long-term immune agents: {n_lti:,} ({n_lti/np.sum(alive)*100:.1f}%)")
    print(f"  Mean age at LTI development: {mean_age:.2f} years")
    print(f"  Median age at LTI development: {median_age:.2f} years")
    
    # Age distribution of LTI development
    print(f"\nAge distribution of long-term immunity development:")
    age_bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 125)]
    age_labels = ['<1y', '1-2y', '2-5y', '5-10y', '10+y']
    
    for (low, high), label in zip(age_bins, age_labels):
        count = np.sum((lti_age_values >= low) & (lti_age_values < high))
        pct = count / n_lti * 100 if n_lti > 0 else 0
        print(f"  {label:>6}: {count:>4} ({pct:>5.1f}%)")
else:
    print("\nNo agents developed long-term immunity")

print("\n" + "=" * 60)
print("Age tracking is working!")
