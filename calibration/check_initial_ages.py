"""Check the initial age distribution when model starts"""

import starsim as ss
import rotasim as rs
import numpy as np
import pandas as pd

print("="*80)
print("Checking Initial Age Distribution")
print("="*80)

# Create the same sim as UK calibration
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
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning simulation for 1 day to initialize...")
sim.run(until=1)  # Just run 1 timestep

# Check initial ages
ages_days = sim.people.age.values
ages_years = ages_days / 365.25

print("\n" + "="*80)
print("INITIAL AGE DISTRIBUTION (Day 1)")
print("="*80)

print(f"\nBasic statistics:")
print(f"  Total agents: {len(ages_years)}")
print(f"  Age range (days): {ages_days.min():.0f} - {ages_days.max():.0f}")
print(f"  Age range (years): {ages_years.min():.2f} - {ages_years.max():.2f}")
print(f"  Mean age: {ages_years.mean():.2f} years")
print(f"  Median age: {np.median(ages_years):.2f} years")

# Detailed age distribution
age_bins = [0, 1, 2, 5, 10, 20, 50, 100]
age_labels = ['<1 y', '1-2 y', '2-5 y', '5-10 y', '10-20 y', '20-50 y', '>=50 y']
age_counts = np.histogram(ages_years, bins=age_bins)[0]

print("\nAge distribution:")
print(f"{'Age Group':12s} {'Count':>8s} {'Percentage':>12s}")
print("-" * 35)
for label, count in zip(age_labels, age_counts):
    pct = count / len(ages_years) * 100
    print(f"{label:12s} {count:8d} {pct:11.2f}%")

print("\n" + "="*80)
print("TARGET UK AGE DISTRIBUTION (for comparison)")
print("="*80)

uk_target = {
    '<1 y': 1.26,
    '1-2 y': 1.27,
    '2-5 y': 3.66,
    '>=5 y': 93.81,
}

print(f"{'Age Group':12s} {'Target %':>12s}")
print("-" * 25)
for age, pct in uk_target.items():
    print(f"{age:12s} {pct:11.2f}%")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Map our bins to UK target bins
model_uk_bins = {
    '<1 y': age_counts[0] / len(ages_years) * 100,  # <1 y
    '1-2 y': age_counts[1] / len(ages_years) * 100,  # 1-2 y
    '2-5 y': age_counts[2] / len(ages_years) * 100,  # 2-5 y
    '>=5 y': sum(age_counts[3:]) / len(ages_years) * 100,  # 5+ y
}

print(f"{'Age Group':12s} {'Model %':>12s} {'Target %':>12s} {'Difference':>12s}")
print("-" * 52)
for age in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
    model_pct = model_uk_bins[age]
    target_pct = uk_target[age]
    diff = model_pct - target_pct
    print(f"{age:12s} {model_pct:11.2f}% {target_pct:11.2f}% {diff:+11.2f}pp")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check how starsim initializes ages
print("\nHow does starsim initialize ages?")
print("Looking at People initialization...")

# Check if there's an age initialization parameter
if hasattr(sim.pars, 'age_dist'):
    print(f"  Found age_dist parameter: {sim.pars.age_dist}")
else:
    print("  No age_dist parameter found in sim.pars")

# Check if ages are all zero
if ages_days.max() == 0:
    print("\n⚠ WARNING: All agents initialized with age = 0!")
    print("  This explains why population doesn't have proper age structure.")
elif ages_years.max() < 1:
    print(f"\n⚠ WARNING: Max age is only {ages_years.max():.2f} years")
    print("  Population is heavily skewed toward young ages.")
else:
    print(f"\n✓ Age range appears reasonable (0 - {ages_years.max():.1f} years)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if abs(model_uk_bins['>=5 y'] - uk_target['>=5 y']) > 10:
    print("\n❌ Initial age distribution DOES NOT match UK demographics")
    print("   Model has {:.1f}% adults vs target {:.1f}%".format(
        model_uk_bins['>=5 y'], uk_target['>=5 y']))
    print("\n   RECOMMENDATION: Initialize population with correct age distribution")
else:
    print("\n✓ Initial age distribution matches UK demographics")

print("="*80)
