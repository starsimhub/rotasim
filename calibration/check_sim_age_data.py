"""
Check what age data is available in sim.people
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import rotasim as rs
import starsim as ss

print("Creating a quick simulation to check available data...")
sim = rs.Sim(
    n_agents=1000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

sim.run()

print("\nChecking sim.people attributes:")
print(f"  Available attributes: {[a for a in dir(sim.people) if not a.startswith('_')][:20]}")

if hasattr(sim.people, 'age'):
    print(f"\n  sim.people.age exists!")
    print(f"    Type: {type(sim.people.age)}")
    print(f"    Shape: {sim.people.age.shape if hasattr(sim.people.age, 'shape') else 'N/A'}")
    print(f"    Sample values: {sim.people.age.values[:10]}")

    # Check age distribution
    ages = sim.people.age
    print(f"\n  Age distribution:")
    print(f"    Min: {ages.min():.1f}")
    print(f"    Max: {ages.max():.1f}")
    print(f"    Mean: {ages.mean():.1f}")

    # Count by age categories matching process_incidence_uk
    under_1 = (ages < 1).sum()
    age_1_2 = ((ages >= 1) & (ages < 2)).sum()
    age_2_5 = ((ages >= 2) & (ages < 5)).sum()
    age_5plus = (ages >= 5).sum()

    print(f"\n  Age categories (in years):")
    n_alive = len(ages.values)
    print(f"    <1 y:    {under_1} ({under_1/n_alive*100:.2f}%)")
    print(f"    1-2 y:   {age_1_2} ({age_1_2/n_alive*100:.2f}%)")
    print(f"    2-5 y:   {age_2_5} ({age_2_5/n_alive*100:.2f}%)")
    print(f"    >=5 y:   {age_5plus} ({age_5plus/n_alive*100:.2f}%)")
    print(f"    Total:   {n_alive}")
